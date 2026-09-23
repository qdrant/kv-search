# KV-search: retrieval attention on long context

We serve long-context generation by replacing full attention with **retrieval attention**: at each decode step every full-attention head attends only to its top-128 context tokens, fetched from a vector index instead of a resident KV cache. The target deployment is **qdrant-edge** — an on-device index that pulls parts of the data from a remote Qdrant on demand. Model: Qwen3.5-9B; context: the Qdrant codebase at 100k / 200k / 1M tokens.

This report collects what we've learned, in two arcs:

- **Does retrieval attention work at all?** — attention needs only a small top-k, HNSW retrieval preserves generation quality, and retrieval concentrates on a stable, cacheable set of positions (§01, §02).
- **What does it cost to serve, and how do we shrink it?** — how much of the context an edge instance must fetch, how that scales, and which reductions hold up (§03, §04, §05).

## The fetch has two components

A serverless edge instance fetches two distinct things per decode step, per (layer, kv-head) shard. Keeping them separate is the key to reading the cost:

- **V — value fetch.** The value vectors of the *returned* top-128, needed to form the attention output.
- **K — traversal fetch.** Every graph node HNSW *scores while walking the graph* to find that top-128 — i.e. the keys read during search. K ⊇ the returned set and is `ef`-dependent.

The total remote download is **K + V**. §03–§04 measure and reduce V (the smaller, well-understood half); §05 measures K (the larger, traversal half).
