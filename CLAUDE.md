# kv-search-clean

## tailM (tail correction for retrieval attention)

- **Building the per-head files:** [`docs/tailm-build.md`](docs/tailm-build.md) covers how to run
  `scripts/build_tailm.py`, its modes, gate, output and file format. Read it before building or
  re-gating tailM files.
- **Using them:** `kv-search chat --tailm`, see `README.md` ("Tail correction (tailM)").

## Projection edges (query-aware key HNSW graph)

- **Building the shards:** [`docs/proj-build.md`](docs/proj-build.md) covers `prefill --proj`, the
  Qdrant fork it needs, the recipes, `proj.json` and the shard checks. Read it before building.
- **Using them:** `kv-search chat --proj --no-retriever.exact`, see `README.md` ("Projection edges").
