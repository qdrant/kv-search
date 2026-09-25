# 05 · The traversal fetch (K)

> **Provisional.** K is measured by replaying recorded decode queries through a patched edge that reports the nodes HNSW scores (`retrieve_visited`). That patch is not yet upstreamed, so this section's runner and numbers stay uncommitted until it is. 100k is measured; 200k / 1M pending.

The value fetch (§03) is a *lower bound* on the real download — it ignores every key HNSW reads while walking the graph to find the top-128. That traversal set is **K**, and it dominates.

## K vs V at 100k

Replaying a 256-token generation (5 prompts) through the HNSW-indexed shards at edge's default `ef` (HNSW recall 94–97%):

| | per shard, over a 256-tok generation |
|---|---|
| V — returned top-k values (§03 floor) | 13.2% of context (422k vectors) |
| K — nodes scored during traversal (keys) | 55.9% of context (1.79M vectors) |
| K / V amplification | 4.2× |

So the real download is **K + V ≈ 69% of context** at 100k with default `ef` — not the 13% the value floor suggested. And it is **still climbing** at 256 tokens: the 34-token prompt alone visits 28.7% of context, rising to 63.4% by the end of generation.

## Implications

- **Traversal dominates** (~4× the values), so the value-side reductions of §04 address only the smaller half. The real prize is cutting what *search visits*: the immediate knob is `ef` (traded against recall, §02), and the structural levers are shipping HNSW's upper layers with edge and biasing traversal toward already-cached nodes.
- **Pivotal open question:** does K% *drop* with context, like V? Visited-per-query is ~`ef`-bounded (roughly context-independent), so cumulative distinct over a fixed generation should be a smaller fraction of a larger context. If so, edge stays viable at scale. The 200k / 1M measurement (shards now rebuilt) will answer it.
