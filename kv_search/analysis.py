import json
import math
import warnings
from collections.abc import Generator
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rich
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from rich.progress import track
from rich.table import Table
from safetensors.torch import safe_open
from transformers import AutoConfig, PreTrainedConfig
from transformers.cache_utils import CacheLayerMixin, LinearAttentionCacheLayerMixin

from kv_search.cache import _repeat_kv, load_cache


class CachedData:
    def __init__(self, cache_dir: Path, model_name: str, device: str | None = None):
        self.cache_dir = cache_dir
        # skip mps: several ops here are flaky on it; pass device= to override
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

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

    def indices(
        self, layer_idx: int, prompt_idx: int = 0
    ) -> tuple[int | None, torch.Tensor]:
        tmp = self.cache_dir / f"indices{prompt_idx:02d}/"
        path = tmp / f"indices_{layer_idx:02d}.safetensors"
        with safe_open(path, framework="pt", device=self.device) as f:
            indices = f.get_slice("indices")[:]

        meta: dict[str, int] = json.loads((tmp / "meta.json").read_text())
        return meta.get("prompt_len"), indices

    def scores(self, layer_idx: int, prompt_idx: int = 0) -> torch.Tensor:
        tmp = self.cache_dir / f"indices{prompt_idx:02d}/"
        path = tmp / f"indices_{layer_idx:02d}.safetensors"
        with safe_open(path, framework="pt", device=self.device) as f:
            scores = f.get_slice("scores")[:]

        return scores

    def dynamic_scores(self, layer_idx: int, prompt_idx: int = 0) -> torch.Tensor:
        tmp = self.cache_dir / f"indices{prompt_idx:02d}/"
        path = tmp / f"indices_{layer_idx:02d}.safetensors"
        with safe_open(path, framework="pt", device=self.device) as f:
            scores = f.get_slice("dynamic_scores")[:]

        return scores

    def plot_mse(self) -> Figure:
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
        return fig

    def plot_indices_heatmap(self) -> Figure:
        layers = list(self.full_layers)
        fig = plt.figure(figsize=(12, 2.5 * len(layers)), layout="constrained")
        fig.suptitle(self.config.model_type, fontsize=24)

        axs: list[Axes] = fig.subplots(nrows=len(layers))

        n_bins = 512

        for ax, (layer_idx, layer) in zip(axs, layers):
            _, indices = self.indices(layer_idx)
            indices = indices.cpu().numpy()  # [16, q_len, n]

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

        return fig

    def plot_scores(self) -> Figure:
        layers = list(self.full_layers)
        fig = plt.figure(figsize=(20, 3 * len(layers)), layout="constrained")
        fig.suptitle(self.config.model_type, fontsize=24)

        # columns: fixed vs dynamic, their difference, fixed-cache novelty
        axs = fig.subplots(nrows=len(layers), ncols=3, squeeze=False)

        # below this = a sentinel, not a real logit (nan padding or causal-mask fill)
        sentinel = np.finfo(np.float32).min / 2

        for (ax_cmp, ax_rel, ax_new), (layer_idx, layer) in zip(axs, layers):
            # fixed: top-K logits over the static prefill context
            fixed = self.scores(layer_idx).cpu().numpy()[0]  # [q_len, K]
            # dynamic: all live keys (prompt + generation), ragged/padded
            dyn = self.dynamic_scores(layer_idx).cpu().numpy()[0]  # [q_len, max_len]
            dyn = np.where(np.isnan(dyn) | (dyn < sentinel), np.nan, dyn)

            k = fixed.shape[-1]
            steps = np.arange(fixed.shape[0])

            # take dynamic's top-K too, so its mean compares like-for-like with fixed
            dyn_desc = -np.sort(-dyn, axis=1)
            with warnings.catch_warnings():  # all-NaN steps -> NaN, no plotted point
                warnings.simplefilter("ignore", RuntimeWarning)
                fixed_max = np.nanmax(fixed, axis=1)
                dyn_max = np.nanmax(dyn, axis=1)
                fixed_mean = fixed.mean(axis=1)
                dyn_topk_mean = np.nanmean(dyn_desc[:, :k], axis=1)

            ax_cmp.plot(steps, fixed_max, label="fixed max")
            ax_cmp.plot(steps, dyn_max, label="dynamic max")
            ax_cmp.plot(steps, fixed_mean, "--", label=f"fixed top-{k} mean")
            ax_cmp.plot(steps, dyn_topk_mean, "--", label=f"dynamic top-{k} mean")

            ax_cmp.set_title(f"Layer {layer_idx} — fixed vs dynamic")
            ax_cmp.set_xlabel("Query step (prompt + generation)")
            ax_cmp.set_ylabel("Retrieved Token Scores (logits)")
            ax_cmp.legend()

            # difference (not ratio): logits add, so > 0 means dynamic wins the step
            ax_rel.axhline(0.0, color="0.6", lw=0.8)
            ax_rel.plot(steps, dyn_max - fixed_max, label="Δ max")
            ax_rel.plot(steps, dyn_topk_mean - fixed_mean, "--", label=f"Δ top-{k} mean")

            ax_rel.set_title(f"Layer {layer_idx} — dynamic − fixed")
            ax_rel.set_xlabel("Query step (prompt + generation)")
            ax_rel.set_ylabel("Score Difference (logits)")
            ax_rel.legend()

            # novelty: score of positions retrieved for the first time this step
            _, indices = self.indices(layer_idx)
            idx = indices.cpu().numpy()[0]  # [q_len, K]
            seen: set[int] = set()
            new_mask = np.zeros_like(idx, dtype=bool)
            for t in range(idx.shape[0]):
                new_mask[t] = np.fromiter((i not in seen for i in idx[t]), bool, k)
                seen.update(idx[t].tolist())

            fixed_new = np.where(new_mask, fixed, np.nan)
            with warnings.catch_warnings():  # steps with no new positions -> NaN
                warnings.simplefilter("ignore", RuntimeWarning)
                new_mean = np.nanmean(fixed_new, axis=1)
                new_max = np.nanmax(fixed_new, axis=1)

            ax_new.plot(steps, new_mean, ":", label="fixed new mean")
            ax_new.plot(steps, new_max, ":", label="fixed new max")

            ax_new.set_title(f"Layer {layer_idx} — fixed novelty")
            ax_new.set_xlabel("Query step (prompt + generation)")
            ax_new.set_ylabel("New Retrieved Token Scores (logits)")
            ax_new.legend()

        return fig

    def plot_indices_unique_per_generation(self) -> Figure:
        layers = list(self.full_layers)
        fig = plt.figure(figsize=(12, 2.5 * len(layers)), layout="constrained")
        fig.suptitle(self.config.model_type, fontsize=24)

        axs: list[Axes] = fig.subplots(nrows=len(layers))

        for ax, (layer_idx, layer) in zip(axs, layers):
            prompt_len, indices_t = self.indices(layer_idx)
            assert prompt_len is not None
            indices = indices_t.cpu().numpy()  # [16, q_len, n]

            # per step, how many retrieved positions are new, averaged over
            # head-groups (4 query heads per kv head, counted jointly)
            all_new = []
            all_new_gen = []
            for h in range(0, indices.shape[0], 4):
                seen: set[int] = set()
                new = []
                for t in range(indices.shape[1]):
                    tmp: set[int] = set()
                    new_ = 0
                    for h_ in range(4):
                        tokens = set(indices[h + h_, t])
                        new_ += len(tokens - seen)
                        tmp |= tokens
                    seen |= tmp
                    new.append(new_)
                all_new.append(new)

                seen = set()
                new = []
                for t in range(prompt_len, indices.shape[1]):
                    tmp = set()
                    new_ = 0
                    for h_ in range(4):
                        tokens = set(indices[h + h_, t])
                        new_ += len(tokens - seen)
                        tmp |= tokens
                    seen |= tmp
                    new.append(new_)
                all_new_gen.append(new)

            steps_pct = np.arange(indices.shape[1]) / indices.shape[1] * 100

            new = np.array(all_new).mean(axis=0)
            ax.plot(steps_pct, new / 128 / 4 * 100, label="new since prompt start")

            new = np.array(all_new_gen).mean(axis=0)
            ax.plot(
                steps_pct[prompt_len:],
                new / 128 / 4 * 100,
                label="new since generation start",
            )

            ax.set_title(f"Layer {layer_idx}")
            ax.set_xlabel("Generation progress [%]")
            ax.set_ylabel("New positions retrieved [%]")
            ax.legend()

        return fig

    def index_stats(self) -> tuple[list[str], list[tuple[str, ...]]]:
        layers = list(self.full_layers)
        headers = ["Layer", "# touched", "Hit Ceiling", "Retention", "Redundancy"]
        rows: list[tuple[str, ...]] = []
        table = Table(*headers)
        for layer_idx, layer in layers:
            _, indices_t = self.indices(layer_idx)
            indices: np.ndarray = indices_t.cpu().numpy()  # [16, q_len, n]
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

    def cross_layer_overlap(
        self,
    ) -> tuple[list[str], list[tuple[str, ...]], Figure]:
        """Do successive full-attn layers retrieve the same positions? If so, a
        later layer's retrieval can be served from what earlier layers fetched.

        Coverage = fraction of a layer's positions already in the prefetch pool
        (previous layer, or the union of all prior layers). The baseline pairs
        each pool with a random other step, to separate genuine cross-layer
        structure from positions that are just popular everywhere.
        """
        layers = list(self.full_layers)

        # per layer, per step: union over heads of retrieved positions
        sets: list[list[set[int]]] = []
        for layer_idx, _ in layers:
            _, indices_t = self.indices(layer_idx)
            indices = indices_t.cpu().numpy()  # [H, q_len, K]
            sets.append(
                [
                    set(indices[:, t, :].ravel().tolist())
                    for t in range(indices.shape[1])
                ]
            )

        q_len = len(sets[0])
        # baseline: same pool, but a random other step's target set
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

        prefix = [set(s) for s in sets[0]]  # running union over prior layers
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
        ax.plot(
            cumul_layer_ids, cumul_base, "o--", label="cumulative (popularity baseline)"
        )
        ax.set_ylim(0, 1)
        ax.set_xlabel("full-attention layer")
        ax.set_ylabel("coverage of layer's retrieved positions")
        ax.set_title(f"{self.config.model_type}: cross-layer prefetch coverage")
        ax.legend()
        return headers, rows, fig

    def query_proxy_recall(
        self, n_positions: int = 256, chunk: int = 32
    ) -> tuple[list[str], list[tuple[str, ...]], Figure]:
        """Can an earlier layer's query stand in for a later layer's, to start
        its search early? Score layer L's query against L+1's keys and measure
        how much of L+1's true top-k it recovers within an over-fetch of k'.
        High recall means a stale query is a usable prefetch trigger.

        Uses the last `n_positions` prefill queries and computes ground-truth
        top-k on the fly, so it needs no recorded indices. Caveat: these are
        prefill queries, not the decode queries used elsewhere.
        """
        layers = list(self.full_layers)
        _, idx0 = self.indices(layers[0][0])
        K = idx0.shape[-1]
        P = min(n_positions, self.context_len)
        kprimes = [k for k in (K, 2 * K, 4 * K, 8 * K) if k <= self.context_len]
        # k' sweep for the curve, log-spaced from K up to 32K / context
        ks = np.unique(
            np.logspace(
                math.log2(K),
                math.log2(min(32 * K, self.context_len)),
                30,
                base=2,
                dtype=int,
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

                # rank of L+1's true top-k under the proxy logits, chunked
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
        return headers, rows, fig

    def analyze(self) -> None:
        """Print every table to the console and save every plot into the cache dir."""
        figs: dict[str, Figure] = {}
        self.index_stats()
        *_, figs["cross_layer_overlap"] = self.cross_layer_overlap()
        *_, figs["query_proxy_recall"] = self.query_proxy_recall()
        figs["index_heatmap"] = self.plot_indices_heatmap()
        figs["scores"] = self.plot_scores()
        figs["indices_unique"] = self.plot_indices_unique_per_generation()
        figs["mse"] = self.plot_mse()

        for name, fig in figs.items():
            path = self.cache_dir / f"{name}.png"
            fig.savefig(path)
            rich.print(f"[green]saved[/green] {path}")
