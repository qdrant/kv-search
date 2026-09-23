# 01 · Retrieval fundamentals

Offline analysis of the recorded top-128 picks (and their scores) for every full-attention layer, on the 200k context.

## Attention needs only a small top-k

Keeping the k highest-scoring tokens per query and dropping the rest, the attention-output MSE falls to near zero within a few hundred tokens; keeping k *random* tokens stays high until k is nearly the whole context. Attention mass sits on a small set, so the top ~128 reconstructs the output well — the premise the whole approach rests on.

![output MSE vs k, top-k against a random baseline](mse.png)

## Retrieval is concentrated and stable

The per-position retrieval heatmap is vertical streaks, not scattered dots: a limited set of positions is retrieved again and again across steps, plus a bright recency band at the far right. Retrieval does not roam the whole context.

![which context positions get retrieved, over generation](indices_heatmap.png)

Per-layer, aggregated over generation:

- **Hit ceiling** (fraction of retrievals that repeat something already seen — the best case for an unlimited cache) is high everywhere, **~0.83–0.89**: most retrievals repeat, so caching pays off.
- **Retention** (of a step's 128 picks, how many were also picked last step) climbs with depth, **~0.26 → 0.55**: deeper layers settle on a steady working set.

## Only a small slice is new each step

The flip side of retention: after an initial spike, the fraction of the 128 that were never retrieved before falls to a low, steady trickle. Most of what each step retrieves it already fetched earlier.

![new positions retrieved per step](indices_unique.png)

## Live context vs retrieved context

The retrieval cache is the frozen prefill; the *dynamic* cache is the live context (prompt + generated so far). In shallow layers the best live token out-scores the best retrieved token by a few logits; by the deeper layers the advantage shrinks and flips. The gap is roughly a constant offset, not something that grows over generation.

![retrieved (fixed) vs live (dynamic) scores](scores.png)

## Prefetch across layers: positions yes, queries no

If a later layer wants positions an earlier layer already fetched, we could fetch once and reuse down the stack. Coverage of a layer's needed positions by the union of earlier layers climbs to **~0.7** by the deep layers, well above a popularity baseline (~0.35) — real, reusable overlap.

The other prefetch route — reuse layer L's *query* to start L+1's search early — does **not** work: recall of L+1's true top-k from L's query sits at the random baseline even with 32× over-fetch. Queries are too layer-specific. Prefetch has to reuse *positions*, not queries.
