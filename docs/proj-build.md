# Projection edges — build and use

`kv-search prefill --proj` builds the edge shards with **query-aware projection edges** in the key
HNSW graph. `kv-search chat --proj --no-retriever.exact` searches them. This page covers what you
need to build the shards, check them and use them.

## What the projection pass does

Plain HNSW links a key to the keys close to it. Attention queries don't look like keys, though: the
keys one query retrieves together can sit far apart in key space. The graph then has no short path
between them, and a search with a small beam (`ef`) misses part of the top-n.

The projection pass rewires level 0 of the graph from the queries' point of view:

1. Take a set of **training queries** and find each one's exact top-`topn` keys.
2. Count how often two keys are retrieved by the same query.
3. Give every key up to `m` extra edges to the keys it is most often retrieved together with.

Search itself is unchanged; only the level-0 links differ.

Decode queries don't exist at build time. The prompt's own queries stand in for them:

- `prefill` records them post-RoPE into `queries_LL.safetensors`.
- The pass takes the last `--proj-window` prompt positions of the KV head's 4 query heads.
- It RoPE-shifts each one to a decode position `context_len + U[0, --proj-shift-span)`.

The code is in `kv_search/proj.py`.

## Requirements

- **Build server: the Qdrant fork.** Branch `ood-edge-patching` of qdrant/qdrant, commit
  `2750e316d`. Build and start its `qdrant` binary as usual and point `--url` at it (default
  `localhost`). The endpoint that receives the training queries
  (`PUT /collections/{name}/hnsw_training_vectors`) exists only in the fork. Against plain Qdrant
  the build therefore stops with an error at the first shard's training upload.
- **Chat needs no fork.** The shards load with the upstream `qdrant-edge` crate: it ignores the
  extra settings block, and the links are ordinary graph links. Don't let an upstream Qdrant
  re-optimize such a shard, though: it would rebuild a plain graph. kv-search never does.
- **A prefill cache with its `queries_LL.safetensors`,** written by the same prefill as the keys.
  The build checks this before any upload.

## Quick start

```bash
# shards from an existing prefill cache: no model, no GPU
.venv/bin/kv-search prefill -d niah --skip-prefill --proj

# or together with a fresh prefill
.venv/bin/kv-search prefill -d niah --upsert --proj

# chat on HNSW over the projection graph
.venv/bin/kv-search chat -d niah -r native --proj --no-retriever.exact --retriever.hnsw-ef 128
```

Use `.venv/bin/…`, not `uv run`: a sync can rebuild flash-attn and the Rust extension.

Build time on niah (69,517 keys, 32,768 training queries per shard): all 32 shards in 12 min 21 s
through the fork, about 23 s per shard.

## Build options (`prefill`)

| option | default | meaning |
|---|---|---|
| `--proj` | off | build with projection edges. Needs `--upsert` or `--skip-prefill` |
| `--proj-recipe plain\|barred` | `plain` | see "Recipes" below |
| `--skip-prefill` | off | build the shards from the saved cache: loads only the model config (YaRN above the native window) and the cache on the CPU. Implies the upload |
| `--edge-only L,…` | all | rebuild only these shards, e.g. `layer03_head3`. With `--proj`, the folder's `proj.json` must match the flags |
| `--proj-m` | 16 | projected edges per key |
| `--proj-topn` | 100 | exact top-n per training query |
| `--proj-maxq` | 64 | at most this many training queries sampled per key |
| `--proj-cands` | 200 | co-retrieval candidates per key |
| `--proj-seed` | 0 | seeds the server pass and the shift targets (per layer and head) |
| `--proj-window` | 8192 | last prompt positions used as training queries (× 4 q-heads) |
| `--proj-shift-span` | 256 | decode positions the queries are shifted to: `context_len + U[0, span)` |
| `--proj-max-training-vectors` | unset | server-side bound on the training vectors. Unset = every uploaded vector. A bound below the upload size is refused, because the server would sample the set down silently |
| `--proj-exclude` | `0,1,2` | barred only: positions that never receive a projected edge |
| `--proj-repair-cap` | 32 | barred only: per-point cap of the repair pass |

The key graph itself is built as before: `m` 16, the server's `ef_construct`, indexing threshold
20,000 KB.

## Recipes

| recipe | folder | sent to the fork | chat |
|---|---|---|---|
| `plain` | `edge_proj/` | `excluded_points: []`, `repair: false` | yes |
| `barred` | `edge_proj_barred/` | `excluded_points: --proj-exclude`, `repair: true` | **refused** |

`barred` keeps the attention sinks (positions 0–2) out of the projected edges and repairs
reachability afterwards. That only pays off when the decode side scores the excluded points
directly. kv-search doesn't do that yet, so `chat --proj` refuses barred shards, and `prefill`
warns when you build them.

Every field is sent explicitly, because the fork's own defaults are the barred recipe.

## Output

```
<cache>/edge_proj/
  proj.json            the recipe (ProjConfig): chat reads it
  layer03_head0/ …     one qdrant-edge shard per (full-attention layer, KV head)
```

- `edge/` (no `--proj`) keeps the plain HNSW shards.
- Keys of `proj.json` this version does not know are ignored; a training source other than
  `prefill` is refused.

## Checks

Every shard is checked right after the build (`main._wait_indexed`, `main._verify_shard`), also
without `--proj`:

- **Wait.** A green status alone doesn't mean the graph is built: right after HNSW is switched on,
  the collection is still green with nothing indexed. The build waits until all of these hold:
  - every point has arrived and the update queue is empty;
  - the optimizer is idle;
  - something is indexed, and the indexed count and segment count stay the same over three polls.
- **Verify.** Every segment whose `segment.json` claims an HNSW key index must have
  `vector_index-key/hnsw_config.json` and non-empty link files, and at least one segment must have a
  graph. With `--proj`, every key graph's `hnsw_config.json` must carry the `projection` block.
- **Retry.** A shard that fails is snapshotted again, up to 3 attempts, and then the build stops.

To look at a shard by hand:

```bash
cat <cache>/edge_proj/layer15_head3/segments/*/vector_index-key/hnsw_config.json
# {"m":16, …, "indexed_vector_count":69517, "projection":{"m":16,"topn":100, …}}
```

## Chat

```bash
.venv/bin/kv-search chat -d niah -r native --proj --no-retriever.exact --retriever.hnsw-ef 128
```

- `--proj [--proj-recipe plain]` loads `edge_proj/` instead of `edge/` and prints its recipe. It
  works with `-r native` and `-r edge`.
- **Search is exact by default**, which is today's behaviour. Exact top-n is what tailM's gate was
  measured on, so on exact search the graph changes nothing, and chat warns about it.
  `--no-retriever.exact` switches to HNSW.
- `--retriever.hnsw-ef` sets the beam width (default 128). qdrant-edge raises it to at least `-n`.
- A retriever switched to inside the REPL (`/native`, `/edge`) keeps the startup search mode and
  folder.

## Measured

niah, Qwen3.5-9B, 69,517 keys, 60 questions answered by greedy decoding (160 tokens), ef 128.
Projection shards built by `prefill --skip-prefill --proj` with the defaults (plain recipe).

| | plain HNSW | projection HNSW |
|---|---|---|
| recall@128 vs exact top-128 (16 decode queries per shard, mean over 32) | 0.837 | **0.926** (32 / 32 heads better) |
| attention error vs full attention, mean, without → with tailM | 0.185 → 0.067 | **0.164 → 0.044** |
| attention error, p99 row, with tailM | 0.472 | **0.185** |
| questions answered of the 52 full attention answers, without / with tailM | 49 / 51 | 50 / 51 |
| decode speed, without tailM (median, prompts 2–60) | 15.6 tok/s | 15.9 tok/s |

The errors are relative L2 of each head's attention output against full attention, over every
generated token (about 915,000 head readings per run). Each run follows its own generated tokens.
Exact search costs 91 ms per decode token for the 8 full-attention layers, HNSW 15.7 ms on either
graph (6 cores / 12 threads, `max_search_threads(8)`, after a warm-up pass).

## Limits

- **HNSW changes answers compared with exact search.** It stays opt-in.
- **tailM on HNSW:** tailM truncates its tail at the weakest *retrieved* logit, which on HNSW is not
  the exact top-n boundary its gate was measured on.
- **Segments:** small or appendable segments stay plain and are scanned in full. Check the segment
  types after the first build of a new tier.
- **Upstream compatibility is structural, not guaranteed.** A new graph file in the fork, or
  upstream rejecting unknown settings, would break loading. The shard check and a test load cover
  this.
- **Only the prefill queries are supported as the training source.**
