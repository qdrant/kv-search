# 04 · Cutting the value fetch

Reductions on V, scored in attention-output MSE vs the full top-k. All simulatable from recorded top-k + prefill values (no HNSW).

## Weight-threshold policy

Only fetch a newly-seen position if its softmax weight clears τ; reuse what's cached otherwise. Trades download for a small output-MSE hit.

![download vs weight-retained and output MSE](tradeoff.png)

| size | download @ τ=3e-3 | output MSE | weight kept |
|---|---|---|---|
| 100k | 54% | 1.5e-02 | 95% |
| 200k | 53% | 1.9e-02 | 95% |
| 1M | 58% | 4.1e-02 | 94% |

Halving the fetch keeps ~95% of attention weight at low MSE; the same τ costs 2–3× more MSE at 1M, so tune it per context size.

## Is the low-weight tail just an average?

![tail mean-field MSE vs head size](tail_meanfield.png)

Keeping only the top-h positions exact: **within a step the tail is essentially its mean** — replacing the ~120 low-weight positions with their in-step average recovers the output ~30× better than dropping them (at head=8, 100k: drop 0.13 vs step-mean 0.006). The tail's *identity* is redundant. **But** a cached/historical tail-mean (windowed or global) drifts and lands near dropping — you cannot keep one running mean and skip fetching. The redundancy is only exploitable by reusing the cached *set*, not a mean.

## Static centroids — ruled out

Representing regions by k-means centroids (keys included). Two failures, both worse at scale: coarse centroid scoring over-fetches **12–49× (100k) to 239–956× (1M)** the top-128 to recover 90% of the attention weight; and centroid values as a tail stand-in are only ~2–3× better than dropping at 100k and **≈ dropping at 1M**. The tail's effective value is query-specific, not regional.

## Reuse the cached set — the win

Fetch the top-h each step (they join the shard's local cache), reuse cached positions exactly, and substitute the uncached tail with its nearest *already-cached* value. The cache is an adaptive codebook of exact fine vectors.

| size | head | download vs floor | drop | in-step mean | **cached-sub** |
|---|---|---|---|---|---|
| 100k | 8 | 14.6% | 0.039 | 0.0024 | **0.0125** |
| 1M | 8 | 11.0% | 0.065 | 0.0022 | **0.0201** |

Cached-substitution beats dropping ~3× and is **Pareto-better than the threshold policy** (fetch top-8 → ~11–15% of the fetch-every-distinct floor at MSE ~0.012–0.020, vs the threshold's ~54% download at similar MSE). It holds at 1M where static centroids collapsed. This is the deployable V-side lever: prefer already-fetched positions, substitute the low-weight tail from the local cache.
