"""Build learned pages from a recorded prefill cache, then ingest into custom Qdrant.

CPU only: numpy, safetensors, requests, qdrant-client; zstandard on Python <3.14.
Generated files belong in an external work directory, never in the source checkout.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import time

import numpy as np
import requests
from safetensors.numpy import load

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kv_search.qdrant_pages import PageAttentionClient

EXTENSIONS = ("meta", "pages", "side", "graph", "summary", "inverse")


def sha256(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def fnv(data):
    value = 0xCBF29CE484222325
    for byte in data:
        value = ((value ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{value:016x}"


def layer_arrays(path):
    try:
        from compression import zstd
        blob = zstd.decompress(path.read_bytes())
    except ImportError:
        import zstandard
        with path.open("rb") as f, zstandard.ZstdDecompressor().stream_reader(f) as reader:
            blob = reader.read()
    tensors = load(blob)
    if "keys" not in tensors or "values" not in tensors:
        return None  # Qwen linear-attention state, not a K/V layer.
    arrays = []
    for name in ("keys", "values"):
        u = tensors[name]
        if u.dtype != np.uint8 or u.ndim != 5 or u.shape[:2] != (2, 1):
            raise ValueError(f"{path}: expected byte-shuffled BF16 {name} [2,1,heads,tokens,dim]")
        arrays.append(u.reshape(2, -1).T.copy().view("<u2").reshape(u.shape[1:])[0])
    if arrays[0].shape != arrays[1].shape:
        raise ValueError(f"{path}: K/V shapes differ")
    return arrays


def query_array(path):
    # Map BF16 as raw bits: NumPy has no native bfloat16 dtype.
    with path.open("rb") as f:
        size = struct.unpack("<Q", f.read(8))[0]
        if size > 16 * 1024 * 1024:
            raise ValueError(f"{path}: invalid safetensors header")
        header = json.loads(f.read(size))
    q = header["queries"]
    shape, (start, end) = q["shape"], q["data_offsets"]
    if (q["dtype"] != "BF16" or len(shape) != 4 or shape[0] != 1
            or min(shape) <= 0 or start < 0 or end - start != int(np.prod(shape)) * 2
            or 8 + size + end > path.stat().st_size):
        raise ValueError(f"{path}: expected BF16 queries [1,tokens,q_heads,dim]")
    return np.memmap(path, dtype="<u2", mode="r", offset=8 + size + start, shape=tuple(shape))[0]


def f32(bits):
    return (bits.astype(np.uint32) << 16).view(np.float32)


def stems(manifest):
    return [f"l{l:04}h{h:04}" for l in range(manifest["spec"]["num_layers"])
            for h in range(manifest["spec"]["num_kv_heads"])]


def prepare(args):
    work = args.work_dir.resolve()
    cache = args.cache.resolve()
    if work.is_relative_to(Path(__file__).resolve().parents[1]):
        raise ValueError("Choose an external --work-dir, outside the source checkout")
    files = sorted(cache.glob("layer_*.safetensors.zst"))
    if not files:
        raise ValueError("No layer_*.safetensors.zst files in cache")
    context = json.loads((cache / "meta.json").read_text())["context_len"]
    layers, q_heads, geometry = [], None, None
    sources = {"meta.json": sha256(cache / "meta.json")}
    print("Validating cache and recording source checksums...", flush=True)
    for path in files:
        arrays = layer_arrays(path)
        if arrays is None:
            continue
        k, v = arrays
        physical = int(path.name.split("_")[1].split(".")[0])
        q_path = cache / f"queries_{physical:02d}.safetensors"
        if not q_path.is_file():
            raise ValueError(f"Missing {q_path}: learned pages require recorded prefill Q as well as K/V")
        q = query_array(q_path)
        heads, tokens, dim = k.shape
        if (tokens != context or dim not in (128, 256) or tokens <= dim
                or q.shape[0] != tokens or q.shape[2] != dim or q.shape[1] % heads):
            raise ValueError(f"Incompatible K/V/Q geometry in layer {physical}")
        if geometry is not None and (geometry != k.shape or q_heads != q.shape[1]):
            raise ValueError("All dense layers must have the same K/V/Q geometry")
        # Qdrant retains original rows as float16 for rescoring.
        for values in (k, v, q[::args.stride]):
            x = f32(values)
            if not np.isfinite(x).all() or np.max(np.abs(x)) > 65504:
                raise ValueError(f"Layer {physical}: non-finite or out-of-float16-range values")
        geometry, q_heads = k.shape, q.shape[1]
        layers.append(physical)
        sources[path.name], sources[q_path.name] = sha256(path), sha256(q_path)
        del arrays, k, v, q, x
    if not layers:
        raise ValueError("No dense K/V layers found")
    if args.builder:
        builder = str(args.builder.resolve())
        build_id = sha256(Path(builder))
    else:
        build_id = subprocess.check_output(
            ["docker", "image", "inspect", "--format", "{{.Id}}", args.builder_image], text=True).strip()
    heads, tokens, dim = geometry
    identity = {"format": 1, "sources": sources, "cache_layers": layers, "query_stride": args.stride,
                "builder": build_id, "num_q_heads": q_heads,
                "spec": {"num_layers": len(layers), "num_kv_heads": heads, "head_dim": dim},
                "context_len": context}
    # A rebuild or Docker's image-ID conversion must not change the rotation seed.
    # Builder identity still binds resumable work to the exact executable/image.
    seed_identity = {key: value for key, value in identity.items() if key != "builder"}
    identity["session_id"] = args.session_id or hashlib.sha256(
        json.dumps(seed_identity, sort_keys=True).encode()).hexdigest()[:32]
    work.mkdir(parents=True, exist_ok=True)
    manifest_path = work / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != identity:
            raise ValueError("Work directory belongs to different inputs/options/builder; choose a new directory")
    elif any(work.iterdir()):
        raise ValueError("Work directory must be empty for its first preparation")
    else:
        write_json(manifest_path, identity)
    if (work / "pages").exists():
        validate_generation(work)
        print(f"Generation already complete: {work / 'pages'}")
        return
    (work / "raw").mkdir(exist_ok=True)
    (work / "heads").mkdir(exist_ok=True)
    for l, physical in enumerate(layers):
        k, v = layer_arrays(cache / f"layer_{physical:02d}.safetensors.zst")
        q = query_array(cache / f"queries_{physical:02d}.safetensors")
        positions = np.arange(0, tokens, args.stride, dtype="<u4")
        group = q_heads // heads

        def build_head(h):
            stem = f"l{l:04}h{h:04}"
            raw, output = work / "raw" / stem, work / "heads" / stem
            marker = output / "complete.json"
            if marker.exists():
                saved = json.loads(marker.read_text())
                expected_names = {f"{stem}.{ext}" for ext in EXTENSIONS}
                if (set(saved["pages"]) == expected_names
                        and all(sha256(output / name) == digest for name, digest in saved["pages"].items())
                        and all(sha256(raw / name) == digest for name, digest in saved["originals"].items())):
                    print(f"{stem}: already complete", flush=True)
                    return
                raise ValueError(f"{stem}: completed files changed; choose a new work directory")
            if output.exists():
                # Preserve interrupted output for inspection, build into a fresh directory.
                output.rename(output.with_name(stem + f".interrupted-{time.time_ns()}"))
            raw.mkdir(exist_ok=True)
            k[h].tofile(raw / "keys.bf16")
            v[h].tofile(raw / "values.bf16")
            # Match the original builder: query-head order, then increasing token position.
            q[::args.stride, h * group:(h + 1) * group].transpose(1, 0, 2).copy().tofile(raw / "queries.bf16")
            np.tile(positions, group).tofile(raw / "positions.u32")
            if args.builder:
                command = [builder, str(raw), str(output)]
            else:
                command = ["docker", "run", "--rm", "--user", f"{os.getuid()}:{os.getgid()}",
                           "--mount", f"type=bind,source={work},target=/work", "--entrypoint",
                           "/qdrant/page-attention-prepare", args.builder_image,
                           f"/work/raw/{stem}", f"/work/heads/{stem}"]
            subprocess.run(command + [identity["session_id"], str(l), str(h), str(dim), str(args.threads)], check=True)
            write_json(marker, {
                "pages": {f"{stem}.{ext}": sha256(output / f"{stem}.{ext}") for ext in EXTENSIONS},
                "originals": {name: sha256(raw / name) for name in ("keys.bf16", "values.bf16")},
            })

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(build_head, range(heads)))
        del k, v, q
    staging = work / "pages.building"
    staging.mkdir(exist_ok=True)
    for stem in stems(identity):
        for ext in EXTENSIONS:
            dest = staging / f"{stem}.{ext}"
            if not dest.exists():
                os.link(work / "heads" / stem / dest.name, dest)
    (staging / "source.fnv").write_text(fnv(manifest_path.read_bytes()) + "\n")
    staging.rename(work / "pages")
    print(f"Prepared {len(layers) * heads} heads: {work / 'pages'}", flush=True)


def validate_generation(work):
    data = (work / "manifest.json").read_bytes()
    manifest = json.loads(data)
    if (work / "pages/source.fnv").read_text().strip() != fnv(data):
        raise ValueError("Generation manifest checksum mismatch")
    n, d = manifest["context_len"], manifest["spec"]["head_dim"]
    for stem in stems(manifest):
        expected = json.loads((work / "heads" / stem / "complete.json").read_text())
        for ext in EXTENSIONS:
            name = f"{stem}.{ext}"
            if sha256(work / "pages" / name) != expected["pages"][name]:
                raise ValueError(f"Generation checksum mismatch: {name}")
        for name in ("keys.bf16", "values.bf16"):
            if (work / "raw" / stem / name).stat().st_size != n * d * 2:
                raise ValueError(f"Original row length mismatch: {stem}/{name}")
            if sha256(work / "raw" / stem / name) != expected["originals"][name]:
                raise ValueError(f"Original row checksum mismatch: {stem}/{name}")
    return manifest


def ingest(args):
    from qdrant_client import QdrantClient, models
    work = args.work_dir.resolve()
    m = validate_generation(work)
    n, d = m["context_len"], m["spec"]["head_dim"]
    collection = f"{args.url.rstrip('/')}/collections/{args.collection}"
    headers = {"api-key": args.api_key} if args.api_key else {}
    r = requests.get(collection, headers=headers, timeout=30)
    if r.status_code != 404:
        r.raise_for_status()
        raise ValueError("Collection already exists; use a new name (no automatic overwrite)")
    generation = args.server_work_dir.rstrip("/") + "/pages" if args.server_work_dir else str(work / "pages")
    vectors = {}
    for l in range(m["spec"]["num_layers"]):
        for h in range(m["spec"]["num_kv_heads"]):
            vectors[f"l{l:04}h{h:04}"] = {"size": 2 * d, "distance": "Dot", "datatype": "float16", "on_disk": True,
                "page_attention": {"generation": generation, "session_id": m["session_id"],
                                   "layer": l, "head": h, "head_dim": d, "rescore": args.rescore}}
    r = requests.put(collection, headers=headers, json={"vectors": vectors, "shard_number": 1,
        "optimizers_config": {"indexing_threshold": 0, "default_segment_number": 1, "max_optimization_threads": 0}}, timeout=60)
    r.raise_for_status()
    configured = requests.get(collection, headers=headers, timeout=30)
    configured.raise_for_status()
    actual = configured.json()["result"]["config"]["params"]["vectors"]
    if any(actual[name].get("page_attention") != spec["page_attention"] for name, spec in vectors.items()):
        raise ValueError("Server did not retain page_attention configuration; use our custom Qdrant image")
    originals = {stem: [np.memmap(work / "raw" / stem / name, mode="r", dtype="<u2", shape=(n, d))
                        for name in ("keys.bf16", "values.bf16")] for stem in vectors}
    client = QdrantClient(url=args.url, grpc_port=args.grpc_port, prefer_grpc=True, api_key=args.api_key,
                          timeout=120, check_compatibility=False)
    try:
        for start in range(0, n, args.batch):
            end = min(start + args.batch, n)
            batch = {stem: f32(np.concatenate([k[start:end], v[start:end]], axis=1)).tolist()
                     for stem, (k, v) in originals.items()}
            client.upsert(args.collection, models.Batch(ids=list(range(start, end)), vectors=batch), wait=True)
            if start // args.batch % 100 == 0 or end == n:
                print(f"Uploaded {end:,}/{n:,} tokens", flush=True)
    finally:
        client.close()
    r = requests.patch(collection, headers=headers, json={"optimizers_config": {
        "indexing_threshold": 1, "max_optimization_threads": 1}}, timeout=60)
    r.raise_for_status()
    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:
        r = requests.get(collection, headers=headers, timeout=30)
        r.raise_for_status()
        state = r.json()["result"]
        if state["optimizer_status"] != "ok":
            raise RuntimeError(f"Optimizer failed: {state['optimizer_status']}")
        if state["status"] == "green" and state["indexed_vectors_count"] == n * len(vectors):
            with_client = PageAttentionClient(args.url, args.grpc_port, args.collection, args.api_key)
            try:
                with_client.validate(0, m["spec"]["num_kv_heads"], d, n, rescore=bool(args.rescore))
            finally:
                with_client.close()
            print(f"Ready: {args.collection}, {n:,} points, {n * len(vectors):,} indexed vectors")
            return
        print(f"Indexing: {state['indexed_vectors_count']:,}/{n * len(vectors):,}", flush=True)
        time.sleep(5)
    raise TimeoutError("Indexing deadline exceeded; inspect Qdrant logs. Collection has been preserved.")


def verify(args):
    """Compare returned attention with stable softmax(QK^T/sqrt(d))V on original rows."""
    work = args.work_dir.resolve()
    m = validate_generation(work)
    n, d = m["context_len"], m["spec"]["head_dim"]
    kv_heads, q_heads = m["spec"]["num_kv_heads"], m["num_q_heads"]
    positions = [int(p) for p in args.positions.split(",")]
    if not positions or len(set(positions)) != len(positions) or min(positions) < 0 or max(positions) >= n:
        raise ValueError("--positions must be distinct token positions inside this context")
    if any(p % m["query_stride"] == 0 for p in positions):
        raise ValueError("Use positions outside the training stride for this held-out check")
    results = {(ef, rescore): [] for ef in (16, 64) for rescore in (False, True)}
    client = PageAttentionClient(args.url, args.grpc_port, args.collection, args.api_key)
    try:
        for l, physical in enumerate(m["cache_layers"]):
            path = args.cache / f"queries_{physical:02d}.safetensors"
            if sha256(path) != m["sources"][path.name]:
                raise ValueError(f"Query cache differs from prepared source: {path}")
            queries = f32(query_array(path)[positions]).transpose(1, 0, 2).copy()
            exact = np.empty_like(queries, dtype=np.float64)
            for h in range(kv_heads):
                stem = f"l{l:04}h{h:04}"
                k, v = [f32(np.memmap(work / "raw" / stem / name, mode="r", dtype="<u2", shape=(n, d)))
                        for name in ("keys.bf16", "values.bf16")]
                group = slice(h * q_heads // kv_heads, (h + 1) * q_heads // kv_heads)
                q = queries[group].reshape(-1, d)
                logits = (q @ k.T).astype(np.float64) / np.sqrt(d)
                weights = np.exp(logits - logits.max(axis=1, keepdims=True))
                weights /= weights.sum(axis=1, keepdims=True)
                exact[group] = (weights @ v).reshape(exact[group].shape)
                del k, v, logits, weights
            for (ef, rescore), errors in results.items():
                client.validate(l, kv_heads, d, n, rescore=rescore)
                out, _ = client.query(queries, l, kv_heads, ef=ef, rescore=rescore)
                error = np.linalg.norm(out - exact, axis=-1) / np.maximum(np.linalg.norm(exact, axis=-1), 1e-12)
                errors.extend(error.reshape(-1).tolist())
            print(f"Checked layer {physical}: {q_heads * len(positions)} held-out queries", flush=True)
    finally:
        client.close()
    report = {"collection": args.collection, "context_len": n, "positions": positions,
              "reference": "Full-context stable softmax(QK^T/sqrt(d))V over original BF16 rows; no causal mask",
              "metric": "relative L2 of attention output: ||approx - exact||2 / ||exact||2, per query head",
              "checks": [{"ef": ef, "rescore": rescore, "heads_checked": len(errors),
                          "mean_rel_l2": float(np.mean(errors)), "max_rel_l2": float(np.max(errors))}
                         for (ef, rescore), errors in results.items()]}
    write_json(work / "validation.json", report)
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="build from K/V/Q cache; reruns resume completed heads")
    p.add_argument("cache", type=Path)
    p.add_argument("--work-dir", type=Path, required=True)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--builder", type=Path)
    group.add_argument("--builder-image", help="image containing /qdrant/page-attention-prepare (Linux/WSL Docker)")
    p.add_argument("--stride", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--threads", type=int, default=3, help="threads per builder process")
    p.add_argument("--session-id", help="optional rotation seed identity for reproducing an older generation")
    p.set_defaults(run=prepare)
    p = sub.add_parser("ingest", help="create, upload original K/V and import learned page generation")
    p.add_argument("--work-dir", type=Path, required=True)
    p.add_argument("--server-work-dir", help="work directory as seen by Qdrant, e.g. /generations/100k")
    p.add_argument("--url", default="http://localhost:6433")
    p.add_argument("--grpc-port", type=int, default=6434)
    p.add_argument("--api-key")
    p.add_argument("--collection", required=True)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--rescore", type=int, default=256)
    p.add_argument("--timeout", type=int, default=1800)
    p.set_defaults(run=ingest)
    p = sub.add_parser("verify", help="check held-out cached queries against exact full-context attention")
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--work-dir", type=Path, required=True)
    p.add_argument("--positions", default="32,12832", help="comma-separated positions outside the training stride")
    p.add_argument("--url", default="http://localhost:6433")
    p.add_argument("--grpc-port", type=int, default=6434)
    p.add_argument("--api-key")
    p.add_argument("--collection", required=True)
    p.set_defaults(run=verify)
    args = parser.parse_args()
    for name in ("stride", "workers", "threads", "batch", "timeout"):
        if hasattr(args, name) and getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive")
    if getattr(args, "rescore", 0) < 0:
        parser.error("--rescore must be nonnegative")
    try:
        args.run(args)
    except (ValueError, OSError, RuntimeError, requests.RequestException, subprocess.CalledProcessError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
