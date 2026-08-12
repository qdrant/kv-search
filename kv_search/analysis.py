import math
from collections.abc import Generator
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rich
import torch
from matplotlib.axes import Axes
from rich.progress import track
from rich.table import Table
from safetensors.torch import safe_open
from transformers import AutoConfig, PreTrainedConfig
from transformers.cache_utils import CacheLayerMixin, LinearAttentionCacheLayerMixin

from kv_search.cache import _repeat_kv, load_cache


class CachedData:
    def __init__(self, cache_dir: Path, model_name: str, device: str = "cuda"):
        self.cache_dir = cache_dir
        self.device = device

        self.config: PreTrainedConfig = AutoConfig.from_pretrained(model_name)

        text_config = getattr(self.config, "text_config", self.config)
        self.head_dim = getattr(
            text_config,
            "head_dim",
            text_config.hidden_size // text_config.num_attention_heads,
        )
        self.num_key_value_groups = (
            text_config.num_attention_heads // text_config.num_key_value_heads
        )
        self.scaling = self.head_dim**-0.5

        self.prefill, self.context_len = load_cache(cache_dir, self.config, device)

    @property
    def full_layers(self) -> Generator[tuple[int, CacheLayerMixin]]:
        for idx, layer in enumerate(self.prefill.layers):
            if isinstance(layer, CacheLayerMixin) and not isinstance(
                layer, LinearAttentionCacheLayerMixin
            ):
                yield idx, layer

    def queries(
        self, layer_idx: int, positions: slice = slice(-1, None)
    ) -> torch.Tensor:
        path = self.cache_dir / f"queries_{layer_idx:02d}.safetensors"
        with safe_open(path, framework="pt", device=self.device) as f:
            q = f.get_slice("queries")[:, positions, :, :]
        return q.transpose(1, 2).contiguous()

    def indices(self, layer_idx: int) -> torch.Tensor:
        path = self.cache_dir / f"indices_{layer_idx:02d}.safetensors"
        with safe_open(path, framework="pt", device=self.device) as f:
            return f.get_slice("indices")[:]

    def plot_mse(self):
        layers = list(self.full_layers)
        fig = plt.figure(figsize=(12, 4 * len(layers)), layout="constrained")
        fig.suptitle(self.config.model_type, fontsize=24)

        axs: list[Axes] = fig.subplots(nrows=len(layers))

        for ax, (layer_idx, layer) in zip(axs, layers):
            assert layer.keys is not None and layer.values is not None

            with torch.no_grad():
                query = self.queries(layer_idx).to(torch.float32)
                keys = _repeat_kv(
                    layer.keys[:, :, : self.context_len, :], self.num_key_value_groups
                ).to(torch.float32)
                values = _repeat_kv(
                    layer.values[:, :, : self.context_len, :], self.num_key_value_groups
                ).to(torch.float32)

                logits = torch.matmul(query, keys.transpose(2, 3)) * self.scaling
                o_full = torch.matmul(torch.softmax(logits, dim=-1), values)

                ordered_idx = torch.argsort(logits, dim=-1, descending=True)
                rand_idx = torch.argsort(torch.rand_like(logits), dim=-1)

                ns = torch.unique(
                    torch.logspace(
                        0, math.log2(self.context_len - 1), 100, 2, dtype=torch.int
                    )
                ).tolist()

                mse = []
                mse_random = []
                for n in track(ns):
                    s_partial = logits.scatter(-1, ordered_idx[..., n:], float("-inf"))
                    w = torch.softmax(s_partial, dim=-1)
                    o = torch.matmul(w, values)
                    mse.append(torch.nn.functional.mse_loss(o_full, o))

                    s_partial = logits.scatter(-1, rand_idx[..., n:], float("-inf"))
                    w = torch.softmax(s_partial, dim=-1)
                    o = torch.matmul(w, values)
                    mse_random.append(torch.nn.functional.mse_loss(o_full, o))

                mse = torch.stack(mse).cpu().numpy()
                mse_random = torch.stack(mse_random).cpu().numpy()

                ax.semilogx(ns, mse, base=2, label="top-k")
                ax.semilogx(ns, mse_random, base=2, label="random")
                ax.set_xlabel("k in top-k")
                ax.set_ylabel("MSE $\\left|o - \\tilde{o}\\right|^2$")
                ax.set_title(f"Layer {layer_idx}")
                ax.legend()
        fig.savefig(self.cache_dir / "mse.png")

    def plot_indices(self) -> None:
        layers = list(self.full_layers)
        fig = plt.figure(figsize=(12, 2.5 * len(layers)), layout="constrained")
        fig.suptitle(self.config.model_type, fontsize=24)

        axs: list[Axes] = fig.subplots(nrows=len(layers))

        n_bins = 512

        for ax, (layer_idx, layer) in zip(axs, layers):
            indices = self.indices(layer_idx).cpu().numpy()  # [16, q_len, n]

            heat = np.zeros((indices.shape[1], n_bins))  # [q_len, c_len (binned)]

            bins = indices * n_bins // self.context_len

            for head_idx in range(indices.shape[0]):
                present = np.zeros(
                    (indices.shape[1], n_bins), dtype=bool
                )  # [q_len, c_len (binned)]
                np.put_along_axis(present, bins[head_idx], True, axis=1)
                heat += present

            im = ax.imshow(
                heat,
                cmap="magma",
                aspect="auto",
                vmin=0,
                vmax=indices.shape[0],
                interpolation="nearest",
            )
            ax.set_title(f"Layer {layer_idx}")
            ax.set_xlabel(f"context position (binned into {n_bins})")
            ax.set_ylabel("query step")
            fig.colorbar(im, ax=ax, label="# heads retrieving position")

        fig.savefig(self.cache_dir / "index_heatmap.png")

    def index_stats(self) -> tuple[list[str], list[tuple[str, ...]]]:
        layers = list(self.full_layers)
        headers = ["Layer", "# touched", "Hit Ceiling", "Retention", "Redundancy"]
        rows: list[tuple[str, ...]] = []
        table = Table(*headers)
        for layer_idx, layer in layers:
            indices: np.ndarray = (
                self.indices(layer_idx).cpu().numpy()
            )  # [16, q_len, n]
            num_touched_per_head = np.mean(
                [np.unique(indices[h]).size for h in range(indices.shape[0])],
                dtype=float,
            )
            hit_ceiling = 1 - num_touched_per_head / indices.shape[1] / indices.shape[2]
            retention = np.mean(
                [
                    len(set(indices[h, t]) & set(indices[h, t - 1])) / indices.shape[2]
                    for h in range(indices.shape[0])
                    for t in range(1, indices.shape[1])
                ],
                dtype=float,
            )
            redundancy = np.mean(
                [
                    indices.shape[0]
                    * indices.shape[2]
                    / np.unique(indices[:, t, :]).size
                    for t in range(indices.shape[1])
                ]
            )

            # for capacity in (1, 2, 4):
            #     capacity = capacity * indices.shape[2]
            #     for head_idx in range(indices.shape[0]):
            #         for token_idx in range(indices.shape[1]):

            row = (
                str(layer_idx),
                f"{num_touched_per_head:.2f}",
                f"{hit_ceiling:.2f}",
                f"{retention:.2f}",
                f"{redundancy:.2f}",
            )
            rows.append(row)
            table.add_row(*row)

        rich.print(table)
        return headers, rows

    def cross_layer_overlap(self) -> tuple[list[str], list[tuple[str, ...]]]:
        """Exp 1: do successive full-attn layers retrieve the same context
        positions (same token)? Measures whether a later layer's retrieval can
        be prefetched from what earlier layers already retrieved.

        Coverage = fraction of a layer's needed positions already held by the
        prefetch pool. Consecutive: pool = previous full-attn layer. Cumulative:
        pool = union of all prior full-attn layers (fetch working set once, reuse
        it down the stack). The shuffled baseline pairs each pool with a *random
        other token's* target set, isolating overlap explained by generic
        position popularity from genuine per-token cross-layer structure.
        """
        layers = list(self.full_layers)

        # per-layer, per-token union-over-heads sets of retrieved positions
        # (heads are not aligned across layers, so the union is the right "what
        # does this layer need" granularity; positions mean the same token in
        # every layer, which is what makes cross-layer intersection meaningful)
        sets: list[list[set[int]]] = []
        for layer_idx, _ in layers:
            indices = self.indices(layer_idx).cpu().numpy()  # [H, q_len, K]
            sets.append(
                [set(indices[:, t, :].ravel().tolist()) for t in range(indices.shape[1])]
            )

        q_len = len(sets[0])
        # popularity baseline: same pool, but a random other token's target set
        perm = np.random.default_rng(0).permutation(q_len)

        def coverage(pool: list[set[int]], target: list[set[int]], order) -> float:
            return float(
                np.mean(
                    [
                        len(pool[t] & target[order[t]]) / len(target[order[t]])
                        for t in range(q_len)
                        if target[order[t]]
                    ]
                )
            )

        identity = np.arange(q_len)

        headers = [
            "From→To",
            "target |U|",
            "consec (act)",
            "consec (base)",
            "cumul (act)",
            "cumul (base)",
            "pool |P|",
        ]
        rows: list[tuple[str, ...]] = []
        table = Table(*headers)

        cumul_act: list[float] = []
        cumul_base: list[float] = []
        cumul_layer_ids: list[int] = []

        prefix = [set(s) for s in sets[0]]  # running union of prior layers
        for i in range(1, len(layers)):
            prev_id = layers[i - 1][0]
            cur_id = layers[i][0]
            target = sets[i]
            target_size = float(np.mean([len(s) for s in target]))
            pool_size = float(np.mean([len(s) for s in prefix]))

            consec_act = coverage(sets[i - 1], target, identity)
            consec_base = coverage(sets[i - 1], target, perm)
            cum_act = coverage(prefix, target, identity)
            cum_base = coverage(prefix, target, perm)

            cumul_act.append(cum_act)
            cumul_base.append(cum_base)
            cumul_layer_ids.append(cur_id)

            row = (
                f"{prev_id}→{cur_id}",
                f"{target_size:.0f}",
                f"{consec_act:.2f}",
                f"{consec_base:.2f}",
                f"{cum_act:.2f}",
                f"{cum_base:.2f}",
                f"{pool_size:.0f}",
            )
            rows.append(row)
            table.add_row(*row)

            for t in range(q_len):
                prefix[t] |= sets[i][t]

        rich.print(table)

        fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
        ax.plot(cumul_layer_ids, cumul_act, "o-", label="cumulative (actual)")
        ax.plot(cumul_layer_ids, cumul_base, "o--", label="cumulative (popularity baseline)")
        ax.set_ylim(0, 1)
        ax.set_xlabel("full-attention layer")
        ax.set_ylabel("coverage of layer's retrieved positions")
        ax.set_title(f"{self.config.model_type}: cross-layer prefetch coverage")
        ax.legend()
        fig.savefig(self.cache_dir / "cross_layer_overlap.png")
        return headers, rows

    def query_proxy_recall(
        self, n_positions: int = 256, chunk: int = 32
    ) -> tuple[list[str], list[tuple[str, ...]]]:
        """Exp 2: can an EARLIER layer's query act as a proxy to launch a later
        layer's search early? For each adjacent full-attn pair, score layer L's
        query against layer L+1's keys and measure recall of layer L+1's true
        top-k (computed from its own query). High recall at modest over-fetch k'
        => a stale query is a usable prefetch trigger (mechanism (b): search
        early), independent of whether the positions were already retrieved
        (Exp 1 set-overlap = mechanism (a)).

        Uses the last `n_positions` PREFILL queries (post-RoPE, same space as
        keys) as the query set and computes ground-truth top-k on the fly, so it
        needs no recorded indices. Caveat: prefill-distribution queries, not the
        decode queries that produced the Exp 1 indices — a mechanistic proxy.
        """
        layers = list(self.full_layers)
        K = self.indices(layers[0][0]).shape[-1]
        P = min(n_positions, self.context_len)
        kprimes = [k for k in (K, 2 * K, 4 * K, 8 * K) if k <= self.context_len]
        # k' sweep for the curve (log-spaced from K up to 32K / context)
        ks = np.unique(
            np.logspace(
                math.log2(K), math.log2(min(32 * K, self.context_len)), 30, base=2, dtype=int
            )
        )

        headers = ["From→To", *[f"recall@{k // K}K" for k in kprimes], "rand@1K"]
        rows: list[tuple[str, ...]] = []
        table = Table(*headers)
        curves: list[tuple[str, np.ndarray]] = []

        with torch.no_grad():
            for i in range(1, len(layers)):
                prev_idx, _ = layers[i - 1]
                cur_idx, cur_layer = layers[i]
                assert cur_layer.keys is not None

                keys = _repeat_kv(
                    cur_layer.keys[:, :, : self.context_len, :],
                    self.num_key_value_groups,
                ).to(torch.float32)  # [1, 16, ctx, d]

                q_prev = self.queries(prev_idx, slice(-P, None)).to(torch.float32)
                q_cur = self.queries(cur_idx, slice(-P, None)).to(torch.float32)

                # rank of layer L+1's true top-k positions under the proxy (L)
                # logits, accumulated over position-chunks to bound memory
                true_ranks: list[torch.Tensor] = []
                for s in range(0, P, chunk):
                    kc = keys.transpose(2, 3)  # [1, 16, d, ctx]
                    gt = torch.matmul(q_cur[:, :, s : s + chunk], kc) * self.scaling
                    true = gt.topk(K, dim=-1).indices  # [1, 16, c, K]
                    proxy = torch.matmul(q_prev[:, :, s : s + chunk], kc) * self.scaling
                    ranks = proxy.argsort(-1, descending=True).argsort(-1)
                    true_ranks.append(ranks.gather(-1, true).reshape(-1))

                tr = torch.cat(true_ranks)  # [16 * P * K] ranks in [0, ctx)
                recall = {k: float((tr < k).float().mean()) for k in kprimes}
                curve = np.array([float((tr < int(k)).float().mean()) for k in ks])
                curves.append((f"{prev_idx}→{cur_idx}", curve))

                row = (
                    f"{prev_idx}→{cur_idx}",
                    *[f"{recall[k]:.4f}" for k in kprimes],
                    f"{K / self.context_len:.4f}",
                )
                rows.append(row)
                table.add_row(*row)

        rich.print(table)

        fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
        for label, ys in curves:
            ax.semilogx(ks / K, ys, "-", base=2, label=label)
        ax.set_ylim(0, 1)
        ax.set_xlabel("over-fetch k' / K")
        ax.set_ylabel("recall of next layer's true top-k")
        ax.set_title(f"{self.config.model_type}: early-query proxy recall")
        ax.legend(fontsize=8)
        fig.savefig(self.cache_dir / "query_proxy_recall.png")
        return headers, rows

    @staticmethod
    def _md_table(headers: list[str], rows: list[tuple[str, ...]]) -> str:
        def esc(cells) -> str:
            # escape literal pipes (e.g. |U|, |P|) so they don't split columns
            return "| " + " | ".join(str(c).replace("|", "\\|") for c in cells) + " |"

        sep = "| " + " | ".join("---" for _ in headers) + " |"
        return "\n".join([esc(headers), sep, *(esc(r) for r in rows)])

    def report(self) -> None:
        """Run all index analyses and write a single markdown report
        (`index_report.md`) with explained tables + links to the plots."""
        self.plot_indices()
        stats = self.index_stats()
        overlap = self.cross_layer_overlap()
        recall = self.query_proxy_recall()

        layer0 = self.indices(next(self.full_layers)[0])
        H, T, K = layer0.shape

        md = f"""# Index findings — {self.config.model_type}

Analysis of the recorded per-full-attention-layer top-k retrieval indices
(`indices_*.safetensors`), shape `[H={H} query heads, T={T} query steps, K={K}]`,
over a context of length `{self.context_len}`. Retrieved indices are context
token *positions*; a position means the same token in every layer, which is what
makes cross-layer comparison meaningful.

## Per-layer index statistics

For each full-attention layer, aggregated over the `T={T}` recorded query steps.
`S_t` = the set of positions a head retrieves at step `t`; `W_head` = positions a
head ever touches; `H·K` picks per step collapse to `|∪_h S_t|` unique positions.

- **# touched** — `W_head`, mean over heads of unique positions ever retrieved.
- **Hit Ceiling** — `1 − W_head / (T·K)`, the infinite-cache hit rate (fraction of
  retrievals that are repeats of something seen before).
- **Retention** — mean `|S_t ∩ S_(t-1)| / K` over consecutive steps (= hit@1K, how
  much a step reuses the previous step's set).
- **Redundancy** — `H·K / |∪_h S_t|`, cross-head sharing within a step (1 = none).

{self._md_table(*stats)}

## Cross-layer prefetch coverage — Exp 1

Do successive full-attention layers retrieve the *same positions* (same query
step)? `U_L` = union over heads of a layer's retrieved positions. Coverage =
fraction of a layer's needed positions already held by a prefetch pool. The
**popularity baseline** pairs each pool with a *random other step's* target set,
isolating overlap explained by generic position popularity from genuine per-step
cross-layer structure.

- **target |U|** — mean size of the target layer's per-step position set.
- **consec (act / base)** — coverage of the target by the *previous* layer's set;
  actual vs popularity baseline.
- **cumul (act / base)** — coverage by the *union of all prior* layers (fetch the
  working set once, reuse down the stack); actual vs baseline.
- **pool |P|** — mean size of the cumulative pool.

{self._md_table(*overlap)}

![cross-layer prefetch coverage](cross_layer_overlap.png)

## Early-query proxy recall — Exp 2

Can an *earlier* layer's query act as a proxy to launch a later layer's search
early? Score layer L's query against layer L+1's keys and measure recall of
L+1's true top-k (from its own query), at over-fetch `k'`. Uses the last
`{min(256, self.context_len)}` prefill queries; caveat: prefill-distribution
queries, not the decode queries that produced the indices above.

- **recall@nK** — fraction of the next layer's true top-k found within the top
  `n·K` of the proxy ranking.
- **rand@1K** — random baseline `K / context_len`.

{self._md_table(*recall)}

![early-query proxy recall](query_proxy_recall.png)

## Index heatmap

Which context positions get retrieved, per layer, per query step (brightness =
number of heads retrieving that binned position).

![index heatmap](index_heatmap.png)
"""
        (self.cache_dir / "index_report.md").write_text(md)
        rich.print(f"[green]wrote[/green] {self.cache_dir / 'index_report.md'}")
