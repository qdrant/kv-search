# %% [markdown]
# # KV-cache retrieval: what the indices tell us
#
# When we generate with retrieval instead of full attention, each step picks the
# top-128 context tokens per head and attends only to those. We recorded those
# picks (and their scores) for every full-attention layer of Qwen3.5-9B on the
# Qdrant-codebase prompt. This notebook walks through what those recordings say.
#
# Run top to bottom. Every cell prints a table or draws a labeled plot; the short
# text above each cell says what to look for. Run it from the repo root (the cache
# path below is relative). Export with
# `jupytext --to markdown notebooks/index_investigation.py`, or open directly in
# VSCode as a notebook.

# %%
from pathlib import Path

from IPython.display import Markdown, display

from kv_search.analysis import CachedData


def md_table(headers, rows) -> Markdown:
    """Render a (headers, rows) table as markdown for clean inline display."""
    esc = lambda cells: "| " + " | ".join(str(c) for c in cells) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    return Markdown("\n".join([esc(headers), sep, *(esc(r) for r in rows)]))


data = CachedData(Path("../cache/qdrant/qwen3_5"), model_name="Qwen/Qwen3.5-9B")
print(f"context length: {data.context_len} tokens")

# %% [markdown]
# ## 1. How many tokens does attention actually need?
#
# Before trusting retrieval at all: if we keep only the top-k tokens per query and
# drop the rest, how wrong is the attention output? The blue line keeps the k
# highest-scoring tokens; the orange line keeps k random tokens, as a baseline.
#
# **Takeaway:** blue drops to near-zero by a few hundred tokens, while orange stays
# high until k is almost the whole context. Attention mass sits on a small set of
# tokens, so keeping the top ~128 reconstructs the output well. This is the whole
# reason retrieval works.

# %%
_ = data.plot_mse()

# %% [markdown]
# ## 2. Which context positions get retrieved?
#
# Each row is a generation step, each column a context position (binned).
# Brightness = how many heads retrieved that position at that step.
#
# **Takeaway:** the bright parts are vertical streaks, not scattered dots. A
# limited set of positions gets retrieved again and again across all steps (plus a
# bright band at the far right — the most recent context). Retrieval is
# concentrated and stable, not roaming the whole context.

# %%
_ = data.plot_indices_heatmap()

# %% [markdown]
# ## 3. Per-layer summary
#
# One row per full-attention layer, aggregated over all generation steps:
#
# - **# touched** — how many distinct positions a head ever retrieves.
# - **Hit Ceiling** — fraction of retrievals that are repeats of something already
#   seen. This is the best-case hit rate of an unlimited cache.
# - **Retention** — of a step's 128 picks, how many were also picked last step.
# - **Redundancy** — how much the 16 heads overlap within a step (1 = no overlap).
#
# **Takeaway:** hit ceiling is high everywhere (0.83–0.89): most retrievals repeat,
# so caching pays off. Retention climbs with depth (0.26 → 0.55): deeper layers
# settle on a steady working set and barely change it step to step.

# %%
display(md_table(*data.index_stats()))

# %% [markdown]
# ## 4. How much is genuinely new each step?
#
# The flip side of retention: at each step, what fraction of the 128 retrieved
# positions were never retrieved before? Blue counts against everything seen since
# the prompt; orange resets the "seen" set when generation begins.
#
# **Takeaway:** novelty spikes at the very start and then falls to a low, steady
# trickle. After the first stretch of generation, most of what each step retrieves
# is something it already fetched earlier — only a small slice is new.

# %%
_ = data.plot_indices_unique_per_generation()

# %% [markdown]
# ## 5. Retrieved (fixed) context vs the live (dynamic) context
#
# The retrieval cache is the frozen prefill context. The *dynamic* cache is the
# live context the model builds up as it runs: the user's prompt plus everything
# generated so far (so the first steps on the x-axis are the prompt, not
# generation). Does that live context score higher than the retrieved one? Three
# columns per layer:
#
# - **left** — top scores of each cache over the query steps.
# - **middle** — their difference (dynamic − fixed). Above zero = dynamic wins.
# - **right** — score of only the newly-retrieved positions (from section 4 above).
#
# **Takeaway:** in the shallow layers the live tokens (prompt + generated) clearly
# out-score the retrieved ones — the best live token beats the best retrieved token
# by a few logits; by the deeper layers this advantage shrinks and flips. The gap
# is roughly a constant offset, not something that keeps growing over generation.

# %%
_ = data.plot_scores()

# %% [markdown]
# ## 6. Do neighbouring layers retrieve the same positions? (prefetch)
#
# If layer L+1 tends to want the same positions layer L already fetched, we could
# fetch once and reuse down the stack. The line is how much of a layer's needed
# positions are already covered by the union of all earlier layers; the dashed line
# is a baseline that only reflects generically popular positions.
#
# **Takeaway:** coverage climbs to ~0.7 by the deep layers, well above the
# popularity baseline (~0.35). So there is real, reusable overlap: much of what a
# deep layer needs was already pulled by the layers above it.

# %%
headers, rows, _ = data.cross_layer_overlap()
display(md_table(headers, rows))

# %% [markdown]
# ## 7. Can an earlier layer's query trigger the next layer's search early?
#
# The other way to prefetch: reuse layer L's *query* to start layer L+1's search
# before L+1 has even run. We score L's query against L+1's keys and check how much
# of L+1's true top-k it recovers, even if we over-fetch by up to 32x.
#
# **Takeaway:** recall stays down at the random baseline. An earlier layer's query
# is essentially useless for finding the next layer's tokens — queries are too
# layer-specific. So prefetching has to reuse *positions* (section 6), not queries.

# %%
headers, rows, _ = data.query_proxy_recall()
display(md_table(headers, rows))

# %% [markdown]
# ## Summary
#
# - Attention needs only a small top-k, so retrieval is sound (§1).
# - Retrieval concentrates on a stable, repeated set of positions (§2–4); an
#   unlimited cache would hit ~85% of the time, and deeper layers are steadier.
# - The live context (prompt + generated) out-scores retrieved tokens in shallow
#   layers by a roughly constant margin, fading with depth (§5).
# - For prefetching across layers, reusing *positions* works (~0.7 coverage) but
#   reusing *queries* does not (§6–7).
