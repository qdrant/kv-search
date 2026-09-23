# 03 · Edge download: the value fetch (V)

How much of the context an edge instance fetches to serve a generation, measured from the recorded top-k retrievals (5 diverse prompts per size, 256-token greedy). This section is the **V** component — distinct *returned* top-128 positions, summed over the 32 (layer, kv-head) shards.

## Download accumulates with generation length

![cumulative download over generation, per context size](download_over_generation.png)

The footprint accumulates with *generation length*, not prompt breadth — the 5 prompts (a pinpoint config lookup through a whole-architecture synthesis) land within ~1 point of each other. At 100k it reaches ~13% of context and is still climbing at 256 tokens.

## The fetched fraction shrinks with context

![download fraction and absolute, vs context length](download_scaling.png)

| size | context | download | distinct positions | weight on new |
|---|---|---|---|---|
| 100k | 100,049 | 13.2% | 422,391 | 3.4% |
| 200k | 200,388 | 8.7% | 556,575 | 4.9% |
| 1M | 979,289 | 3.0% | 954,316 | 10.6% |

At a fixed generation length the distinct positions retrieved is bounded by `steps × k`, roughly independent of context — so the **fraction falls as ~1/context while the absolute download grows sub-linearly** (~context^0.4). A bigger remote context costs edge a *smaller* share to serve.

## Freshly-fetched positions carry little weight

![attention weight on newly-fetched positions over generation](weight_on_new.png)

After the first few steps, only a small slice of each query's attention weight lands on positions the shard is fetching for the first time — **~3% at 100k, rising to ~11% at 1M**. Fresh fetches matter mostly "on average," which is what makes the download reducible (§04) — though *less* so at long context, where retrieval roams more.
