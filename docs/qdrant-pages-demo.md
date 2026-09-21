# Qdrant page-attention demo

Select `-r qdrant-pages` to use the custom server. Existing retrievers are unchanged.
The image is CPU-only Linux x86-64; full-model chat still requires a suitable GPU
and the matching Qwen3.5-9B prefill cache.

## External data

Reports, fixtures, snapshots, image archives, and the Qdrant source patch live
outside this repository, in the sibling `../kv-search-data` directory
(`E:\github\kv-search-data` here). Override it with `KV_SEARCH_DATA` when needed.

- `artifacts/qdrant-pages/`: Docker image `.tar`, collection `.snapshot`, checksums.
- `tests/fixtures/`: recorded queries and reference attention outputs.
- `docker/qdrant-pages/`: source patch and pinned base/checksum in `source.json`.
- `docs/`: benchmark reports and packaging checks.

The source bundle is required only for rebuilding the image; colleagues using
the published image need the snapshot and, for CPU regression checks, the fixture.

## Build and publish

Build in Linux/WSL, never with Cargo on Windows. Tested with Ubuntu 24.04,
Rust 1.98.1, C/C++ tools, clang, cmake, pkg-config, protobuf-compiler,
libssl-dev, git, curl, Python 3, and Docker. The runtime requires a build host
with glibc no newer than 2.39.

```bash
bash docker/qdrant-pages/build.sh qdrant-pages:demo
docker tag qdrant-pages:demo YOUR_ACCOUNT/qdrant-pages:demo-20260921
docker push YOUR_ACCOUNT/qdrant-pages:demo-20260921
```

The build reads the external pinned patch, tests/builds Qdrant, and writes image
archives and provenance to `../kv-search-data/artifacts/qdrant-pages`.
Set `EXPORT_IMAGE=0` to skip exporting a tar. `QDRANT_BUILD_ROOT`,
`CARGO_TARGET_DIR`, `BUILD_JOBS`, and `QDRANT_BASE_REPO` configure Linux build
storage, parallelism, and an optional local clone of the pinned Qdrant base.

The existing `qdrant-pages:demo` image is already loaded in Docker Desktop.
On another Windows machine, load the exported archive with:

```powershell
docker load -i E:\github\kv-search-data\artifacts\qdrant-pages\qdrant-pages-demo.tar
```

## Start and test

Set `QDRANT_PAGES_IMAGE` to the published tag when using a registry.
Compose exposes localhost REST/gRPC ports 6433/6434 and persists data in volumes.

```bash
docker compose -f docker/qdrant-pages/compose.yaml up -d --wait
python scripts/qdrant_pages_snapshot.py restore \
  ../kv-search-data/artifacts/qdrant-pages/pages_100k.snapshot --collection pages_100k
python scripts/qdrant_pages_smoke.py --collection pages_100k
```

These scripts need Python 3.11+, numpy, requests, qdrant-client>=1.18, and curl;
no model, PyTorch, or GPU. Restore refuses to overwrite an existing collection.
The CPU smoke test checks ef=16/64, rescore off/on against external reference data.

With the project dependencies and matching 100k prefill cache installed:

```bash
uv run kv-search chat -r qdrant-pages -d qdrant -s 100k -g 128 \
  --retriever.collection pages_100k --retriever.ef 16
```

Use `--retriever.ef 64` for higher accuracy and `--retriever.rescore` to enable
rescoring (budget 256 in this snapshot). URL and gRPC port are configurable with
`--retriever.url` and `--retriever.grpc-port`.

This demo uses an immutable single-shard collection, one populated segment,
Qwen3.5-9B, batch size one, and attention scaling `1/sqrt(head_dim)`. Use the
matching prefill cache: shape checks do not verify its contents. New learned
page generations still require the separate prototype builder. Ordinary top-k,
filters, exact mode, fusion, and updates do not implement this attention protocol.
GPU model E2E remains untested locally; CPU retrieval and snapshot restart passed.
