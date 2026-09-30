# §06 Precision: halving the bytes per vector

§03–§05 shrink the fetch by touching **fewer** vectors. Precision is the orthogonal lever: shrink the **bytes of each vector**. Both components of the download are vectors — the keys scored during traversal (K) and the values of the returned set (V) — so a smaller per-vector encoding multiplies straight through whatever count reduction we already have.

Today's edge shards store every key and value as **f32: 1024 bytes** per 256-d vector. But the model runs in **bf16** end to end — the f32 score is more precise than the attention the model itself computes. That precision is paid for on every fetched byte and buys nothing.

## What Qdrant can actually store

The constraint is what qdrant-edge supports per named vector: storage datatype `Float32 / Float16 / Uint8 / Turbo4`, and quantization (scalar int8 / product / binary). There is **no bf16**. So the first question is whether f16 — the only 16-bit float on offer — is safe for bf16-origin data.

## f16 is lossless for our data

f16 carries **10 mantissa bits to bf16's 7**, so every bf16 value inside f16's range is representable *exactly*; the only risks are components above f16's ceiling (65504, clip) or below its subnormal floor (underflow). A scan of the prefill keys and values across all 32 (layer, kv-head) cells settles it:

| size | max \|key\| | max \|value\| | f16 round-trip err (rms) | clipped | underflow |
|---|---|---|---|---|---|
| 100k | 18.25 | 109.5 | 2.8e-11 | 0 | 1.6e-7 |
| 200k | 18.25 | 109.5 | 2.9e-11 | 0 | 1.4e-7 |
| 1M | 18.25 | 110.0 | 2.8e-11 | 0 | 6.0e-8 |

Peak magnitude is ~110 against a ceiling of 65504 — roughly 600× of headroom, so **nothing clips**. The round-trip error sits at ~1e-11: not rounding but the underflow of about one component in ten million (near-zero values that fall below f16's subnormal floor). f16 storage reproduces the bf16 values the model uses, bit for bit.

## Retrieval is unchanged

Storage fidelity is not the whole story — the HNSW graph is built from f16-scored distances and the search rescoring runs on f16. But qdrant's f16 dot product upcasts each element to f32 and accumulates in f32, and the query is rounded to f16 too (lossless, same argument). So the scores are identical to the f32 shard's, and recall should be as well. Rebuilding L15H0 and L31H0 at 1M as f16 shards and replaying the recorded decode sessions confirms it:

| cell | recall@128 f32 | f16 | Δ | weight recall f32 | f16 | Δ |
|---|---|---|---|---|---|---|
| L15H0 | 0.7910 | 0.7916 | +0.0006 | 0.8490 | 0.8497 | +0.0006 |
| L31H0 | 0.6769 | 0.6788 | +0.0019 | 0.7370 | 0.7357 | −0.0013 |

Every difference is within HNSW build nondeterminism (±0.002; the f16 graph is a fresh build, not the same topology as the f32 shard). f16 retrieval is indistinguishable from f32.

## int8 is nearly free too

Below f16, qdrant's scalar quantization maps each key component to an int8 code (256 B/key, 4× smaller than f32) and keeps the f32 original for optional rescoring. On the two tuned cells it costs no measurable recall:

| cell | recall@128 f32 | int8 (quantized-only) | int8 (rescored) | weight recall f32 → int8 |
|---|---|---|---|---|
| L15H0 | 0.7910 | 0.7890 | 0.7890 | 0.8490 → 0.8521 |
| L31H0 | 0.6769 | 0.6820 | 0.6820 | 0.7370 → 0.7331 |

Every delta is within noise. (Quantized-only and rescored read identical here for a mechanical reason spelled out under PQ below — at oversampling 1, rescoring cannot change the returned set.) Retrieval quality is flat from 1024 B down to 256 B/key.

![Edge storage precision vs retrieval quality at 1M, top-128: recall and weight recall against stored bytes per key, per cell. Quality is flat from f32 (1024 B) through int8 (256 B), then falls sharply under product quantization (64 B, 32 B).](precision_pareto.png)

**Down to int8, precision is a free 4× on the edge download** — no measurable retrieval cost, deployable today (qdrant `datatype: Float16`, or scalar `quantization_config`), and it composes directly with the count reductions of §03–§05.

> Baseline note: our f32 recall (L15H0 0.791, L31H0 0.677) runs below the colleague's graph-visits baseline (0.846 / 0.758) because these shards build at `ef_construct=100` (his 400) and replay a different recorded session set. The precision deltas above are measured on our own shards and queries, so they are unaffected.

## Product quantization: where it bends

Below int8 the code stops preserving the ranking. Product quantization compresses the key far harder — X16 to 64 B, X32 to 32 B — and recall falls off a cliff (quantized-only, scores read from the codes):

| cell | f32 (1024 B) | pq16 (64 B) | pq32 (32 B) |
|---|---|---|---|
| L15H0 | 0.7910 | 0.6330 | 0.4623 |
| L31H0 | 0.6769 | 0.5458 | 0.3879 |

PQ ×16 costs ~0.16 recall, PQ ×32 roughly halves it. So on the retrieval axis the practical floor is **int8 — a clean 4× at no cost — with PQ trading steep recall for the further 4–8×.**

## Rescore recovers it — at a cost

PQ keeps the f32 original, so rescoring the top candidates on the originals recovers the lost recall. With `oversampling=4` (gather ~4×128 candidates on the codes, re-rank on the f32 originals, return the best 128):

| cell | f32 (ef 128) | int8 only → rescore | pq16 only → rescore | pq32 only → rescore |
|---|---|---|---|---|
| L15H0 | 0.7910 | 0.789 → 0.943 | 0.633 → 0.927 | 0.462 → 0.790 |
| L31H0 | 0.6769 | 0.682 → 0.882 | 0.546 → 0.835 | 0.388 → 0.708 |

![PQ/int8 recall recovery via rescore at 1M, top-128: quantized-only vs rescore (oversampling 4) per config, against the f32 (ef 128) reference. Rescore lifts every code back to or above f32.](precision_rescore.png)

pq32 rescored matches f32; pq16 and int8 rescored *exceed* it. Two caveats keep this honest:

- **Part of the lift is wider search, not precision.** `oversampling=4` makes the walk gather ~512 candidates — effectively `ef≈512` against the f32 baseline's `ef=128` — so rescore beats f32 partly because it searches wider, and an f32 shard at the same `ef` would rise too.
- **Rescore fetches the f32 originals.** Its download is `code×(scored) + f32×(oversampling·128 rescored)`, not the code size — so a 32 B code that rescores 512 originals per query is not a 32 B fetch. Rescore is the "small codes + rescore fetch" fetch model, whose net byte value needs the same K-style accounting as §05, not the storage-byte axis of the Pareto above.

So the clean, unambiguous win stays **int8 quantized-only** — 256 B, f32-equal recall, no rescore fetch. PQ+rescore is a recall-recovery lever whose payoff depends on that fetch accounting (and is cheaper if the rescored originals are themselves f16).

## Next

The rescore byte accounting (oversampling vs recovered recall vs originals fetched, under the §05 cache model), and the end-to-end **accuracy** pass — output error, not just recall, where **value** precision enters (values only feed the softmax-weighted average, so they should compress far harder than keys) and where tail correction reconstructs the mass a coarse code drops.
