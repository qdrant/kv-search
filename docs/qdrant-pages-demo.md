# Qdrant page-attention demo

Select `-r qdrant-pages` to use the custom server. Existing retrievers are unchanged.
The image is CPU-only Linux x86-64; full-model chat still requires a suitable GPU
and the matching Qwen3.5-9B prefill cache.

## External data

Reports, fixtures, snapshots, image archives, and generated indexes live
outside this repository, in the sibling `../kv-search-data` directory
(`E:\github\kv-search-data` here). Override it with `KV_SEARCH_DATA` when needed.

- `artifacts/qdrant-pages/`: Docker image `.tar`, collection `.snapshot`, checksums.
- `tests/fixtures/`: recorded queries and reference attention outputs.
- `docs/`: benchmark reports and packaging checks.

No external source bundle or prepared snapshot is required. Build from the
Qdrant `page-attention-demo` branch, or use its published image. A snapshot is an
optional shortcut; the raw-cache workflow below builds a fresh collection.

## Build and publish

Build in Linux/WSL, never with Cargo on Windows. Tested with Ubuntu 24.04,
Rust 1.98.1, C/C++ tools, clang, cmake, pkg-config, protobuf-compiler,
libssl-dev, git, curl, Python 3, and Docker. The runtime requires a build host
with glibc no newer than 2.39.

```bash
QDRANT_SOURCE=../qdrant-ood bash docker/qdrant-pages/build.sh qdrant-pages:demo
docker tag qdrant-pages:demo YOUR_ACCOUNT/qdrant-pages:demo-20260921
docker push YOUR_ACCOUNT/qdrant-pages:demo-20260921
```

The build snapshots the Qdrant checkout, tests/builds the server and offline
`page-attention-prepare` utility, and writes image archives and provenance to
`../kv-search-data/artifacts/qdrant-pages`. Provenance includes the Git revision,
source-tree content hash (including local code changes), and both binary hashes.
Set `EXPORT_IMAGE=0` to skip exporting a tar. `QDRANT_BUILD_ROOT`,
`CARGO_TARGET_DIR`, and `BUILD_JOBS` configure Linux build storage and parallelism.

The existing `qdrant-pages:demo` image is already loaded in Docker Desktop.
On another Windows machine, load the exported archive with:

```powershell
docker load -i E:\github\kv-search-data\artifacts\qdrant-pages\qdrant-pages-demo.tar
```

## Build a collection from the prefill cache

Run in Linux/WSL. These steps need CPU and disk space, not a GPU, model weights,
the old prototype, a reference fixture, or a prepared snapshot. Use a separate
Python environment to avoid installing the chat application's CUDA dependencies:

```bash
python3 -m venv /tmp/kv-pages-env
source /tmp/kv-pages-env/bin/activate
pip install numpy safetensors requests 'qdrant-client>=1.18' zstandard
```

The cache must contain `meta.json`, `layer_NN.safetensors.zst` and recorded
`queries_NN.safetensors` for every dense attention layer. Linear-attention state
files are detected and skipped. Dense layers are assigned ordinal names in file
order, matching the retriever. Missing Q is an error: obtain the recorded queries
or repeat prefill with query recording enabled. K/V alone cannot reproduce the
learned layout. The original caches came from
`gs://retrieval-attention/qdrant-qwen3.5` (100k) and
`gs://retrieval-attention/qdrant1M-qwen3.5` (1M); bucket access is separate from
these repositories. This script reads an already downloaded cache.

```bash
export QDRANT_PAGES_IMAGE=qdrant-pages:demo  # image built above, includes the builder
export QDRANT_GENERATIONS="$HOME/kv-search-data/generations"
CACHE="$HOME/kv-search/cache/qdrant/100k/qwen3_5"
WORK="$QDRANT_GENERATIONS/100k"

python scripts/qdrant_pages_ingest.py prepare "$CACHE" \
  --work-dir "$WORK" --builder-image "$QDRANT_PAGES_IMAGE"

docker compose -f docker/qdrant-pages/compose.yaml \
  -f docker/qdrant-pages/compose.ingest.yaml up -d --wait

python scripts/qdrant_pages_ingest.py ingest --work-dir "$WORK" \
  --server-work-dir /generations/100k --collection pages_100k

python scripts/qdrant_pages_ingest.py verify --cache "$CACHE" \
  --work-dir "$WORK" --collection pages_100k
```

Alternatively, pass `--builder /path/to/page-attention-prepare` to `prepare`.
Build that CPU binary alone in the Qdrant checkout with
`cargo build --locked -p page-attention --features builder --bin page-attention-prepare --profile perf`.
For a server on another host, copy the **complete work directory** there and set
`--server-work-dir` to its path inside the server. Qdrant needs read access to
`manifest.json` and `pages/` while importing; it copies the generation into storage.

Preparation checks input hashes and shapes, builds the original token graph with
miss-driven edge repair, fits the learned page layout, and publishes the completed
generation. It samples prefill Q at stride 64 for edge repair and up to 256 of those
queries per KV head for page packing. `--stride` changes the training sample.
`--workers 4 --threads 3` bounds concurrent heads and threads per process. Inserts
within a head are sequential for reproducibility; the older local benchmark
used 12 concurrent insertions, so freshly built graphs need not match its bytes.
`--session-id` can reuse an old rotation identity; normally it is derived from
source hashes and build options.

Rerun the same `prepare` command to resume completed heads; input or builder changes
require a new work directory. Each completed head and its original rows have SHA-256
checksums. Generated files remain external. Work storage retains original BF16 K/V
for upload plus page files; Qdrant stores another copy. Allow roughly 10 GB extra
for 100k or 100 GB for 1M, in addition to the source cache, and several GB of RAM.

Ingest creates an immutable single-shard collection, uploads all named K+V vectors,
then enables indexing and waits for every vector to be indexed. It refuses to
overwrite an existing collection. On upload/indexing failure the partial collection
is preserved for inspection; retry with a new collection name. Prepared work can
be reused. Use our custom server: release Qdrant does not implement this protocol.

`verify` requires no frozen fixture. It compares held-out Q at positions 32 and 12832
with exact full-context `softmax(QK^T/sqrt(d))V`, for ef=16/64 and rescore off/on.
Use `--positions 32,96` for shorter contexts. Results go to `WORK/validation.json`;
the reported error is relative L2 of the output per query head. This is a retrieval
quality check, not a model E2E test or a latency benchmark. Position selection is
explicitly checked against the training stride. Rescoring budget defaults to 256
at ingest; the four-mode verification requires a nonzero budget.

Locally validated on 102,516 tokens: all 32 KV heads built from raw cache,
3,280,512 vectors indexed, and 1,024 query-head responses unchanged after server
restart. The builder/serving Rust suite (49 tests) and Python suite (10 tests)
passed. CPU preparation and ingestion also passed in a separate environment
without PyTorch, using the builder inside the Docker image.

## Typed attention API

The custom server exposes `POST /collections/{name}/attention` and
`POST /collections/{name}/attention/batch`. The Python retriever now uses this
REST API with persistent HTTP connections. Update the server image together
with the client; older images do not expose these endpoints.

```json
{
  "query": 42,
  "using": "l0000h0000",
  "ef": 64,
  "rescore": false,
  "return_top_k": 16
}
```

`query` accepts either a finite `head_dim` Q vector (without zero padding) or
a point ID. An ID uses the stored **K of the selected KV head as Q**; it does
not recover that token's original prefill Q. The query point is not excluded.
This prototype uses numeric token-position IDs. `ef` defaults to 16 and
`rescore` to false. `return_top_k` defaults to **0**: no IDs are returned or
ranked for delivery. Positive values return up to that many candidate IDs,
best score first, without changing attention or LSE. `ef` is in 1..8192;
`return_top_k` is in 0..8192.

The usual Qdrant envelope contains `result.attention` (a `head_dim` array),
`result.lse`, and, only when requested, `result.token_ids`. Batch input is
`{"queries": [request, ...]}` (1..64 items); its `result` is an array in request
order. Heads and query input types can be mixed in one batch.

Candidate IDs are the top scoring tokens among pages visited by the HNSW
search, including sink pages. Scores use the page approximation with original
scores substituted for rescored rows when enabled. They are not an exact
global top-k or a complete list of attention contributors: the output also
includes an aggregate approximation for unvisited tokens. Fetching only these
candidates and computing attention locally will generally give a different
result. The API does not expose a residual with top-k removed.

Fetch candidate K/V separately with ordinary Qdrant retrieval:

```http
POST /collections/pages_100k/points
Content-Type: application/json

{"ids": [42, 100], "with_vector": ["l0000h0000"], "with_payload": false}
```

Each returned named vector is `[K, V]`, of length `2 * head_dim`, stored as
float16 originals. Split it at `head_dim`.

```python
from kv_search.qdrant_pages import PageAttentionClient

client = PageAttentionClient("http://localhost:6433", collection="pages_100k")
answer = client.attention(42, "l0000h0000", ef=64)  # no token_ids field
answer = client.attention(q_vector, "l0000h0000", ef=64, return_top_k=128)
client.close()
```

Only one active local replica/shard and one populated immutable page-attention
segment are supported. Unindexed heads, proxy segments, filters, exact mode,
and distributed read-consistency options are rejected. Existing collection
authorization, strict-mode query/batch/ef limits, read rate limiting and request
timeouts apply. Cancellation is checked between head computations, not inside
an individual page-attention kernel. Kernel hardware counters are not yet
instrumented. The old score-channel search remains available for regression
checks; the typed endpoint does not use it internally.

Run the read-only API integration checks against a built collection:

```bash
QDRANT_ATTENTION_URL=http://localhost:6433 \
QDRANT_ATTENTION_COLLECTION=pages_100k \
python -m unittest discover -s tests -p test_qdrant_attention_api.py
```

Validated on the freshly prepared 102,516-token collection: 1,024 recorded-Q
head responses (8 layers, 16 query heads, 2 positions, ef=16/64, rescore off/on)
are bit-identical in float32 to the legacy computation. Requesting top-32 IDs
leaves every attention coordinate and LSE unchanged. Point retrieval, ID/K
query equivalence, mixed batches, authorization, strict-mode limits and restart
were checked too. The updated `qdrant-pages:demo` image and tar archive include
the API. These are correctness checks, not new latency benchmarks; earlier
timings used the old protobuf transport.

## Optional snapshot shortcut and regression fixture

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
matching prefill cache: retriever shape checks do not verify its contents. Ordinary top-k,
filters, exact mode, fusion, and updates do not implement this attention protocol.
GPU model E2E remains untested locally; CPU retrieval and snapshot restart passed.
