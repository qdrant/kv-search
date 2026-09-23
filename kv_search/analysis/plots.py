import math
import warnings

import numpy as np
import rich
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from rich.progress import track
from rich.table import Table
import matplotlib.pyplot as plt

from kv_search.analysis.data import CachedData
from kv_search.cache import _repeat_kv


def plot_mse(d: CachedData) -> Figure:
    layers = list(d.full_layers)
    fig = plt.figure(figsize=(12, 4 * len(layers)), layout="constrained")
    fig.suptitle(d.config.model_type, fontsize=24)
    axs: list[Axes] = fig.subplots(nrows=len(layers))
    for ax, (layer_idx, layer) in zip(axs, layers):
        assert layer.keys is not None and layer.values is not None
        with torch.no_grad():
            query = d.queries(layer_idx).to(torch.float32)
            keys = _repeat_kv(
                layer.keys[:, :, : d.context_len, :], d.num_key_value_groups
            ).to(torch.float32)
            values = _repeat_kv(
                layer.values[:, :, : d.context_len, :], d.num_key_value_groups
            ).to(torch.float32)
            logits = torch.matmul(query, keys.transpose(2, 3)) * d.scaling
            o_full = torch.matmul(torch.softmax(logits, dim=-1), values)
            ordered_idx = torch.argsort(logits, dim=-1, descending=True)
            rand_idx = torch.argsort(torch.rand_like(logits), dim=-1)
            ns = torch.unique(
                torch.logspace(0, math.log2(d.context_len - 1), 100, 2, dtype=torch.int)
            ).tolist()
            mse = []
            mse_random = []
            for n in track(ns):
                s_partial = logits.scatter(-1, ordered_idx[..., n:], float("-inf"))
                o = torch.matmul(torch.softmax(s_partial, dim=-1), values)
                mse.append(torch.nn.functional.mse_loss(o_full, o))
                s_partial = logits.scatter(-1, rand_idx[..., n:], float("-inf"))
                o = torch.matmul(torch.softmax(s_partial, dim=-1), values)
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


def plot_indices_heatmap(d: CachedData) -> Figure:
    layers = list(d.full_layers)
    fig = plt.figure(figsize=(12, 2.5 * len(layers)), layout="constrained")
    fig.suptitle(d.config.model_type, fontsize=24)
    axs: list[Axes] = fig.subplots(nrows=len(layers))
    n_bins = 512
    for ax, (layer_idx, layer) in zip(axs, layers):
        _, indices = d.indices(layer_idx)
        indices = indices.cpu().numpy()  # [16, q_len, n]
        heat = np.zeros((indices.shape[1], n_bins))
        bins = indices * n_bins // d.context_len
        for head_idx in range(indices.shape[0]):
            present = np.zeros((indices.shape[1], n_bins), dtype=bool)
            np.put_along_axis(present, bins[head_idx], True, axis=1)
            heat += present
        im = ax.imshow(
            heat, cmap="magma", aspect="auto", vmin=0, vmax=indices.shape[0],
            interpolation="nearest",
        )
        ax.set_title(f"Layer {layer_idx}")
        ax.set_xlabel(f"context position (binned into {n_bins})")
        ax.set_ylabel("query step")
        fig.colorbar(im, ax=ax, label="# heads retrieving position")
    return fig


def plot_scores(d: CachedData) -> Figure:
    layers = list(d.full_layers)
    fig = plt.figure(figsize=(20, 3 * len(layers)), layout="constrained")
    fig.suptitle(d.config.model_type, fontsize=24)
    axs = fig.subplots(nrows=len(layers), ncols=3, squeeze=False)
    sentinel = np.finfo(np.float32).min / 2  # below this = nan pad / mask fill, not a logit
    for (ax_cmp, ax_rel, ax_new), (layer_idx, layer) in zip(axs, layers):
        fixed = d.scores(layer_idx).cpu().numpy()[0]  # [q_len, K] top-k over prefill
        dyn = d.dynamic_scores(layer_idx).cpu().numpy()[0]  # [q_len, max_len] live keys
        dyn = np.where(np.isnan(dyn) | (dyn < sentinel), np.nan, dyn)
        k = fixed.shape[-1]
        steps = np.arange(fixed.shape[0])
        dyn_desc = -np.sort(-dyn, axis=1)
        with warnings.catch_warnings():
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
        ax_rel.axhline(0.0, color="0.6", lw=0.8)
        ax_rel.plot(steps, dyn_max - fixed_max, label="Δ max")
        ax_rel.plot(steps, dyn_topk_mean - fixed_mean, "--", label=f"Δ top-{k} mean")
        ax_rel.set_title(f"Layer {layer_idx} — dynamic − fixed")
        ax_rel.set_xlabel("Query step (prompt + generation)")
        ax_rel.set_ylabel("Score Difference (logits)")
        ax_rel.legend()
        _, indices = d.indices(layer_idx)
        idx = indices.cpu().numpy()[0]  # [q_len, K]
        seen: set[int] = set()
        new_mask = np.zeros_like(idx, dtype=bool)
        for t in range(idx.shape[0]):
            new_mask[t] = np.fromiter((i not in seen for i in idx[t]), bool, k)
            seen.update(idx[t].tolist())
        fixed_new = np.where(new_mask, fixed, np.nan)
        with warnings.catch_warnings():
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


def plot_indices_unique_per_generation(d: CachedData) -> Figure:
    layers = list(d.full_layers)
    fig = plt.figure(figsize=(12, 2.5 * len(layers)), layout="constrained")
    fig.suptitle(d.config.model_type, fontsize=24)
    axs: list[Axes] = fig.subplots(nrows=len(layers))
    for ax, (layer_idx, layer) in zip(axs, layers):
        prompt_len, indices_t = d.indices(layer_idx)
        assert prompt_len is not None
        indices = indices_t.cpu().numpy()  # [16, q_len, n]
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
        ax.plot(steps_pct[prompt_len:], new / 128 / 4 * 100, label="new since generation start")
        ax.set_title(f"Layer {layer_idx}")
        ax.set_xlabel("Generation progress [%]")
        ax.set_ylabel("New positions retrieved [%]")
        ax.legend()
    return fig


def index_stats(d: CachedData) -> tuple[list[str], list[tuple[str, ...]]]:
    layers = list(d.full_layers)
    headers = ["Layer", "# touched", "Hit Ceiling", "Retention", "Redundancy"]
    rows: list[tuple[str, ...]] = []
    table = Table(*headers)
    for layer_idx, layer in layers:
        _, indices_t = d.indices(layer_idx)
        indices: np.ndarray = indices_t.cpu().numpy()  # [16, q_len, n]
        num_touched_per_head = np.mean(
            [np.unique(indices[h]).size for h in range(indices.shape[0])], dtype=float
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
                indices.shape[0] * indices.shape[2] / np.unique(indices[:, t, :]).size
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


def cross_layer_overlap(d: CachedData) -> tuple[list[str], list[tuple[str, ...]], Figure]:
    """Coverage of a layer's retrieved positions by the previous layer / union of all
    prior layers, vs a random-step popularity baseline."""
    layers = list(d.full_layers)
    sets: list[list[set[int]]] = []
    for layer_idx, _ in layers:
        _, indices_t = d.indices(layer_idx)
        indices = indices_t.cpu().numpy()  # [H, q_len, K]
        sets.append(
            [set(indices[:, t, :].ravel().tolist()) for t in range(indices.shape[1])]
        )
    q_len = len(sets[0])
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
        "From→To", "target |U|", "consec (act)", "consec (base)",
        "cumul (act)", "cumul (base)", "pool |P|",
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
            f"{prev_id}→{cur_id}", f"{target_size:.0f}", f"{consec_act:.2f}",
            f"{consec_base:.2f}", f"{cum_act:.2f}", f"{cum_base:.2f}", f"{pool_size:.0f}",
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
    ax.set_title(f"{d.config.model_type}: cross-layer prefetch coverage")
    ax.legend()
    return headers, rows, fig


def query_proxy_recall(
    d: CachedData, n_positions: int = 256, chunk: int = 32
) -> tuple[list[str], list[tuple[str, ...]], Figure]:
    """Can layer L's query stand in for L+1's? Score L's query against L+1's keys and
    measure recall of L+1's true top-k within over-fetch k'. Uses prefill queries."""
    layers = list(d.full_layers)
    _, idx0 = d.indices(layers[0][0])
    K = idx0.shape[-1]
    P = min(n_positions, d.context_len)
    kprimes = [k for k in (K, 2 * K, 4 * K, 8 * K) if k <= d.context_len]
    ks = np.unique(
        np.logspace(math.log2(K), math.log2(min(32 * K, d.context_len)), 30, base=2, dtype=int)
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
                cur_layer.keys[:, :, : d.context_len, :], d.num_key_value_groups
            ).to(torch.float32)
            q_prev = d.queries(prev_idx, slice(-P, None)).to(torch.float32)
            q_cur = d.queries(cur_idx, slice(-P, None)).to(torch.float32)
            true_ranks: list[torch.Tensor] = []
            for s in range(0, P, chunk):
                kc = keys.transpose(2, 3)
                gt = torch.matmul(q_cur[:, :, s : s + chunk], kc) * d.scaling
                true = gt.topk(K, dim=-1).indices
                proxy = torch.matmul(q_prev[:, :, s : s + chunk], kc) * d.scaling
                ranks = proxy.argsort(-1, descending=True).argsort(-1)
                true_ranks.append(ranks.gather(-1, true).reshape(-1))
            tr = torch.cat(true_ranks)
            recall = {k: float((tr < k).float().mean()) for k in kprimes}
            curve = np.array([float((tr < int(k)).float().mean()) for k in ks])
            curves.append((f"{prev_idx}→{cur_idx}", curve))
            row = (
                f"{prev_idx}→{cur_idx}",
                *[f"{recall[k]:.4f}" for k in kprimes],
                f"{K / d.context_len:.4f}",
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
    ax.set_title(f"{d.config.model_type}: early-query proxy recall")
    ax.legend(fontsize=8)
    return headers, rows, fig


def analyze(d: CachedData) -> None:
    """Print every table and save every figure into the cache dir."""
    figs: dict[str, Figure] = {}
    index_stats(d)
    *_, figs["cross_layer_overlap"] = cross_layer_overlap(d)
    *_, figs["query_proxy_recall"] = query_proxy_recall(d)
    figs["index_heatmap"] = plot_indices_heatmap(d)
    figs["scores"] = plot_scores(d)
    figs["indices_unique"] = plot_indices_unique_per_generation(d)
    figs["mse"] = plot_mse(d)
    for name, fig in figs.items():
        path = d.cache_dir / f"{name}.png"
        fig.savefig(path)
        rich.print(f"[green]saved[/green] {path}")
