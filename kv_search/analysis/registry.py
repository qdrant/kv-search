"""Registered analyses: each produces one per-size result dict. The `analyze` CLI loads
CachedData per size (prefill only if needed) and writes the result to an envelope."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from kv_search.analysis import fetch
from kv_search.analysis.data import CachedData


def _downsample(a, n: int = 200) -> list[float]:
    a = np.asarray(a, dtype=float)
    if len(a) <= n:
        return a.tolist()
    xs = np.linspace(0, len(a) - 1, n).round().astype(int)
    return a[xs].tolist()


def _sweep(d: CachedData, n_prompts: int) -> dict:
    ctx = d.context_len
    per = []
    for i in range(n_prompts):
        steps, cum, n_shards = fetch.download_curve(d, i)
        pct = cum / (ctx * n_shards) * 100
        wm = fetch.weight_mass_new(d, i)
        taus, dl, wr = fetch.weight_threshold_tradeoff(d, i)
        per.append(
            {
                "n_steps": int(len(steps)),
                "download_pct_curve": _downsample(pct),
                "final_download_pct": float(pct[-1]),
                "n_shards": int(n_shards),
                "weight_mass_new_mean": float(np.mean([wm[L].mean() for L in wm])),
                "weight_mass_new_curve": _downsample(np.mean([wm[L] for L in wm], axis=0)),
                "tradeoff_taus": taus.tolist(),
                "tradeoff_download_frac": dl.tolist(),
                "tradeoff_weight_retained": wr.tolist(),
            }
        )
    return {"context_len": ctx, "prompts": per}


def _mse(d: CachedData, n_prompts: int) -> dict:
    per = []
    for i in range(n_prompts):
        taus, dlf, wr = fetch.weight_threshold_tradeoff(d, i)
        _, mse = fetch.download_mse_tradeoff(d, i, taus=taus.tolist())
        per.append(
            {
                "taus": taus.tolist(),
                "download_frac": dlf.tolist(),
                "weight_retained": wr.tolist(),
                "mse": mse.tolist(),
            }
        )
    return {"context_len": d.context_len, "prompts": per}


def _meanfield(d: CachedData, n_prompts: int) -> dict:
    return {
        "context_len": d.context_len,
        "prompts": [fetch.tail_meanfield_mse(d, i) for i in range(n_prompts)],
    }


def _reuse(d: CachedData, n_prompts: int) -> dict:
    return {
        "context_len": d.context_len,
        "prompts": [fetch.cached_reuse_substitution(d, i) for i in range(n_prompts)],
    }


def _centroid(d: CachedData, n_prompts: int) -> dict:
    # self-contained on prefill queries; not per-prompt. ctx_chunk bounds memory at 1M.
    res = fetch.centroid_substitution(
        d, head_sizes=[4, 8, 16, 32, 64], ctx_chunk=16384
    )
    return {"context_len": d.context_len} | res


@dataclass
class Analysis:
    run: Callable[[CachedData, int], dict]
    needs_prefill: bool
    per_prompt: bool


ANALYSES: dict[str, Analysis] = {
    "sweep": Analysis(_sweep, needs_prefill=False, per_prompt=True),
    "mse": Analysis(_mse, needs_prefill=True, per_prompt=True),
    "meanfield": Analysis(_meanfield, needs_prefill=True, per_prompt=True),
    "reuse": Analysis(_reuse, needs_prefill=True, per_prompt=True),
    "centroid": Analysis(_centroid, needs_prefill=True, per_prompt=False),
    # retrieval fundamentals (aggregate over layer/head/prompt)
    "layer_reuse": Analysis(fetch.layer_reuse, needs_prefill=False, per_prompt=True),
    "live_vs_retrieved": Analysis(fetch.live_vs_retrieved, needs_prefill=False, per_prompt=True),
    "cross_layer_coverage": Analysis(fetch.cross_layer_coverage, needs_prefill=False, per_prompt=True),
    "topk_mse": Analysis(fetch.topk_mse, needs_prefill=True, per_prompt=False),
}
