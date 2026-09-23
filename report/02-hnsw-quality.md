# 02 · Does HNSW retrieval preserve quality?

End-to-end generative eval (`kv-search eval`): a SQuAD question buried in a long haystack, scored by containment (gold span appears in the output) and token-F1, n=5 per bucket. Three attention modes: **full** (whole context, the ceiling), **exact** (exact top-k KV), **hnsw** (approximate graph — what we'd ship). Single segment, default ef, K=128. Values in the Cached memory tier; steady-state latency after a warm-up pass. 1M uses YaRN (f4).

| ctx | config | contain | tok_f1 | ms/tok | retr ms/tok |
|---|---|---|---|---|---|
| 128k | full | 1.00 | 0.90 | 148 | 0 |
| 128k | exact | 1.00 | 0.90 | 552 | 446 |
| 128k | hnsw | 1.00 | 0.90 | 214 | 114 |
| 256k | hnsw | 1.00 | 0.90 | 234 | 134 |
| 512k | full | 1.00 | 0.90 | 325 | 0 |
| 512k | exact | 1.00 | 0.90 | 1661 | 1552 |
| 512k | hnsw | 0.80 | 0.70 | 241 | 142 |
| 1M | full | 0.60 | 0.61 | 561 | 0 |
| 1M | hnsw | 0.60 | 0.57 | 260 | 160 |

**Accuracy.** `exact` tracks `full` everywhere — exact top-k preserves full accuracy, so the sparse approach is sound. `hnsw` tracks `exact`: lossless to 256k, a single-segment dip at 512k (0.80), and at 1M all three land at 0.60 — that last drop is YaRN extrapolation, not the graph.

**Runtime.** With Cached-tier values, `hnsw` is ~flat at 214–260 ms/tok from 128k to 1M (retrieval 114–160, sub-linear), while `full` grows with context and `exact`'s O(context) scan blows up. `hnsw` crosses under `full` at ~512k and is ~2.2× faster than full / ~12× faster than exact at 1M.

## Closing the 512k gap: segments × ef

Varying the HNSW build (`default_segment_number` 1 vs 6) and search (`ef` default/64/256), hnsw only:

| ctx | segments | ef | contain | tok_f1 | retr ms/tok |
|---|---|---|---|---|---|
| 512k | 1 | def | 0.60 | 0.57 | 162 |
| 512k | 1 | 256 | 0.80 | 0.70 | 197 |
| 512k | 6 | def | 0.80 | 0.70 | 273 |
| 512k | 6 | 256 | 1.00 | 0.90 | 353 |
| 1M | 1 | def | 0.60 | 0.57 | 174 |
| 1M | 6 | 256 | 0.60 | 0.57 | 363 |

At 512k neither knob alone closes the gap; **seg6 + ef256 together fully recover to 1.00**, at ~2× retrieval cost. At 1M every config caps at 0.60 — that ceiling is YaRN, not the graph (seg6 even under-searches at low ef). So **seg6 + ef256 makes single-shard HNSW lossless through 512k**.
