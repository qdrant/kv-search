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

![Edge storage precision vs retrieval quality at 1M, top-128: recall and weight recall against stored bytes per vector, per cell. f16 halves the bytes at unchanged quality.](precision_pareto.png)

**f16 is a free 2× on the edge download** — half the bytes on both K and V, no measurable quality cost, and deployable today as qdrant `datatype: Float16` per named vector. It composes directly with the count reductions of §03–§05.

> Baseline note: our f32 recall (L15H0 0.791, L31H0 0.677) runs below the colleague's graph-visits baseline (0.846 / 0.758) because these shards build at `ef_construct=100` (his 400) and replay a different recorded session set. The precision deltas above are measured on our own shards and queries, so they are unaffected.

## Next

The real Pareto is below f16. int8 (scalar quantization, learned min/max + rescore), Turbo4 (4-bit), and product quantization each trade recall for bytes — and, unlike a storage datatype, keep a full-precision original and rescore the top candidates, which reshapes the fetch model rather than flatly shrinking it. Those points fill in the figure next, keys and values as independent knobs.
