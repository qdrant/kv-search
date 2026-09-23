# 01 · Retrieval fundamentals

Offline analysis of the recorded top-128 picks (and their scores) for every full-attention layer. Each figure aggregates over layers/heads/prompts and compares context sizes.

## Attention needs only a small top-k

Keeping the k highest-scoring tokens per query and dropping the rest, the attention-output MSE falls to near zero within a few hundred tokens; keeping k *random* tokens (dashed) stays high until k is nearly the whole context. Attention mass sits on a small set, so the top ~128 reconstructs the output well — the premise the whole approach rests on, and it holds across context sizes.

![output MSE vs k, top-k (solid) vs random (dashed), per size](topk_mse.png)

## Retrieval reuses a stable set

Retrieval is concentrated and repetitive, not roaming. **Hit ceiling** — the fraction of retrievals that repeat a position already seen, i.e. the best case for an unlimited cache — is high everywhere (~0.83–0.89), so caching pays off. **Retention** — of a step's 128 picks, how many were also picked last step — climbs with depth, so deeper layers settle on a steady working set and barely change it step to step.

![hit ceiling and step-to-step retention vs layer, per size](layer_reuse.png)

## Live context vs retrieved context

The retrieval cache is the frozen prefill; the *dynamic* cache is the live context (prompt + generated so far). In shallow layers the best live token out-scores the best retrieved token by a few logits; the gap shrinks with depth and flips. It's roughly a constant offset, not something that grows over generation.

![top live logit minus top retrieved logit, vs layer, per size](live_vs_retrieved.png)

## Prefetch across layers: positions yes, queries no

If a later layer wants positions an earlier layer already fetched, we could fetch once and reuse down the stack. Coverage of a layer's needed positions by the union of earlier layers climbs to ~0.7 by the deep layers, well above a popularity baseline (dashed) — real, reusable overlap.

![cross-layer coverage (solid) vs popularity baseline (dashed), per size](cross_layer_coverage.png)

The other prefetch route — reuse layer L's *query* to start L+1's search early — does **not** work: recall of L+1's true top-k from L's query sits at the random baseline even with 32× over-fetch. Queries are too layer-specific, so prefetch has to reuse *positions*, not queries.
