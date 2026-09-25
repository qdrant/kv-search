#!/usr/bin/env python3
"""Build qdrant codebase summaries at increasing Qwen-token sizes: a constant condensed-prose
preamble + deterministic signature extraction (bodies stripped, signatures/types/doc-comments
kept) per file in a fixed priority order, so every size is a superset of the smaller ones.
Overshoot each target except 1M, a hard ceiling. `--dry-run` calibrates without writing."""

from __future__ import annotations

import argparse
import json
import os
import sys

from kv_search.data.rust_strip import strip_file

QDRANT = os.path.expanduser("~/Projects/qdrant/dev")
KV = os.path.expanduser("~/Projects/kv-search")
CACHE = os.path.join(KV, "cache")
OLD_SUMMARY = os.path.join(CACHE, "CODEBASE_SUMMARY.md")
TOKCACHE = os.path.join(CACHE, ".summary_tokcache.json")

# crate priority (used as the round-robin order; sub-crates inherit the longest
# matching prefix's rank). project-relevant crates (segment/edge/collection) first.
PRIORITY = [
    "src", "lib/api", "lib/segment", "lib/edge", "lib/collection", "lib/storage",
    "lib/shard", "lib/common", "lib/wal", "lib/quantization", "lib/sparse",
    "lib/blobstore", "lib/gpu", "lib/bm25", "lib/posting_list", "lib/gridstore",
    "lib/uio-grpc-client", "lib/macros", "lib/trififo",
]

TARGETS = [100_000, 200_000, 400_000, 600_000, 800_000, 1_000_000]
CEILING = {1_000_000: 980_000}  # hard upper bound; stay under

SKIP_DIRS = {"tests", "benches", "examples", "target", ".git", ".direnv"}

# machine-generated / bulk-data files: low information density, excluded.
def is_excluded(rel: str) -> bool:
    return (
        rel.endswith("lib/api/src/grpc/qdrant.rs")   # generated protobuf (~100k tok)
        or "/stop_words/" in rel                      # language stop-word tables
        or rel.endswith(".tonic.rs")                  # generated tonic services
    )


def label(t: int) -> str:
    return f"{t // 1000}k" if t < 1_000_000 else "1M"


def discover_crates():
    """Return [(label, src_root)] for every crate, handling nested workspaces
    like lib/common/<subcrate>/src. Deterministic order."""
    crates = [("src", os.path.join(QDRANT, "src"))]
    libdir = os.path.join(QDRANT, "lib")
    for name in sorted(os.listdir(libdir)):
        d = os.path.join(libdir, name)
        if not os.path.isdir(d):
            continue
        direct = os.path.join(d, "src")
        if os.path.isdir(direct):
            crates.append((f"lib/{name}", direct))
        else:  # nested workspace
            for sub in sorted(os.listdir(d)):
                sd = os.path.join(d, sub, "src")
                if os.path.isdir(sd):
                    crates.append((f"lib/{name}/{sub}", sd))
    return crates


def crate_rank(label: str) -> int:
    """Rank by the longest PRIORITY prefix the label starts with."""
    best = len(PRIORITY)
    for i, p in enumerate(PRIORITY):
        if label == p or label.startswith(p + "/"):
            best = min(best, i)
    return best


def list_crate_files(root: str):
    files = []
    for dp, dn, fn in os.walk(root):
        dn[:] = [d for d in dn if d not in SKIP_DIRS]
        if set(dp.split(os.sep)) & SKIP_DIRS:
            continue
        for f in fn:
            if not f.endswith(".rs"):
                continue
            p = os.path.join(dp, f)
            if is_excluded(os.path.relpath(p, QDRANT)):
                continue
            files.append(p)

    def key(p):
        rel = os.path.relpath(p, root)
        depth = rel.count(os.sep)
        entry = 0 if os.path.basename(p) in ("lib.rs", "main.rs", "mod.rs") else 1
        return (depth, os.path.dirname(rel), entry, rel)

    files.sort(key=key)
    return files


def ordered_files():
    """Breadth-balanced round-robin over crates (entry file lib.rs/mod.rs first), so even small
    budgets touch every crate and the fixed order makes every size a prefix (monotonic nesting)."""
    crates = discover_crates()
    crates.sort(key=lambda c: (crate_rank(c[0]), c[0]))
    per_crate = [(label, list_crate_files(root)) for label, root in crates]
    out = []
    maxlen = max((len(fs) for _, fs in per_crate), default=0)
    for r in range(maxlen):
        for label, fs in per_crate:
            if r < len(fs):
                out.append((label, fs[r]))
    return out


def render_section(crate: str, path: str) -> str:
    rel = os.path.relpath(path, QDRANT)
    with open(path, encoding="utf-8", errors="ignore") as f:
        stripped = strip_file(f.read())
    return f"### `{rel}`\n\n{stripped}\n"


def build_preamble() -> str:
    """Condensed overview: title + TOC + every '## ' section intro (drop the
    detailed '### ' subsections). Constant across all sizes."""
    lines = open(OLD_SUMMARY, encoding="utf-8").read().split("\n")
    out, skipping = [], False
    for ln in lines:
        if ln.startswith("## "):
            skipping = False
            out.append(ln)
        elif ln.startswith("### "):
            skipping = True
        elif not skipping:
            out.append(ln)
    body = "\n".join(out).rstrip() + "\n"
    note = (
        "# Qdrant Codebase Summary (signature-level)\n\n"
        "> Generated by `scripts/build_summaries.py`. Function and method bodies are\n"
        "> stripped (shown as `{ ... }`); all item signatures, type definitions, trait\n"
        "> and impl headers, constants, and doc comments are kept. Each section is headed\n"
        "> by the source file path. Unit-test modules (`#[cfg(test)]`) are omitted.\n"
        "> The overview below is condensed from the curated summary; the per-file\n"
        "> signature listing follows in dependency/priority order.\n\n---\n\n"
    )
    return note + body + "\n\n---\n\n## Per-file signatures\n\n"


def get_tokenizer():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    tok = get_tokenizer()

    def ntok(s: str) -> int:
        return len(tok(s, add_special_tokens=False).input_ids)

    cache = {}
    if os.path.exists(TOKCACHE):
        cache = json.load(open(TOKCACHE))

    preamble = build_preamble()
    pre_tok = ntok(preamble)

    files = ordered_files()
    rows = []  # (rel, crate, section_text, ntokens, cumulative)
    cum = pre_tok
    print(f"preamble: {pre_tok} tokens; scanning {len(files)} files ...",
          file=sys.stderr)
    for idx, (crate, path) in enumerate(files):
        st = os.stat(path)
        ck = f"{path}:{int(st.st_mtime)}:{st.st_size}"
        sec = render_section(crate, path)
        if ck in cache:
            nt = cache[ck]
        else:
            nt = ntok(sec)
            cache[ck] = nt
        cum += nt
        rows.append((os.path.relpath(path, QDRANT), crate, sec, nt, cum))
        if idx % 400 == 0:
            print(f"  {idx}/{len(files)}  cum={cum}", file=sys.stderr)

    json.dump(cache, open(TOKCACHE, "w"))
    total = cum
    print(f"\nTOTAL if all included: {total:,} tokens "
          f"({len(files)} files + {pre_tok:,} preamble)\n")

    # choose prefix length per target
    plan = {}
    for t in TARGETS:
        if t in CEILING:
            c = CEILING[t]
            k = 0
            for i, r in enumerate(rows):
                if r[4] <= c:
                    k = i + 1
                else:
                    break
            plan[t] = k
        else:
            k = len(rows)
            for i, r in enumerate(rows):
                if r[4] >= t:
                    k = i + 1
                    break
            plan[t] = k

    print(f"{'target':>8} {'files':>6} {'tokens':>10} {'note'}")
    for t in TARGETS:
        k = plan[t]
        actual = rows[k - 1][4] if k else pre_tok
        note = ""
        if t in CEILING:
            note = f"<= {CEILING[t]:,} ceiling"
        else:
            note = f"overshoot +{actual - t:,}"
        lastcrate = rows[k - 1][1] if k else "-"
        print(f"{label(t):>8} {k:>6} {actual:>10,}  {note}; last crate: {lastcrate}")

    if args.dry_run:
        print("\n(dry run: no files written)")
        return

    os.makedirs(CACHE, exist_ok=True)
    manifest = {"preamble_tokens": pre_tok, "total_tokens": total, "sizes": {}}
    for t in TARGETS:
        k = plan[t]
        body = preamble + "\n".join(r[2] for r in rows[:k])
        outpath = os.path.join(CACHE, f"CODEBASE_SUMMARY_{label(t)}.md")
        with open(outpath, "w", encoding="utf-8") as f:
            f.write(body)
        real = ntok(body)
        manifest["sizes"][label(t)] = {
            "path": outpath, "files": k, "tokens_est": rows[k - 1][4] if k else pre_tok,
            "tokens_real": real,
        }
        print(f"wrote {outpath}  files={k}  real_tokens={real:,}")
    json.dump(manifest, open(os.path.join(CACHE, "summary_manifest.json"), "w"), indent=2)
    print("\nmanifest -> cache/summary_manifest.json")


if __name__ == "__main__":
    main()
