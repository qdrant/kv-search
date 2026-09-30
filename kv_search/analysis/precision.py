"""Storage-precision fidelity of the prefill keys/values.

Qdrant-edge stores a vector as Float32, Float16, Uint8 or Turbo4 - there is no bf16. Our
keys/values originate as bf16, and f16 has more mantissa bits than bf16, so every bf16
value inside f16's range is representable exactly: f16 storage is lossless except for
components that exceed F16_MAX (clip) or fall below the smallest f16 subnormal (underflow).
This scan reports, per (layer, kv-head), the max magnitude and the f16 clip/underflow
fractions and round-trip error, to confirm f16 is a free 2x over today's f32 shards.

Streams one head at a time via LayerReader (mmap), so it needs no prefill load and runs on
CPU; set needs_prefill=False in the registry."""

from __future__ import annotations

import numpy as np
import torch

from kv_search.analysis.data import CachedData
from kv_search.tailm_build import LayerReader

F16_MAX = 65504.0  # largest finite float16


def _full_attention_layers(cache_dir) -> list[tuple[int, LayerReader]]:
    """(layer_idx, reader) for every full-attention layer file; others raise and are skipped."""
    out: dict[int, LayerReader] = {}
    for p in sorted(cache_dir.glob("layer_*.safetensors*")):
        idx = int(p.name.split(".")[0].split("_")[1])
        if idx in out:
            continue  # raw sorts before .zst; keep the first
        try:
            out[idx] = LayerReader(p)
        except (ValueError, FileNotFoundError):
            pass  # linear-attention layer file, no byte-shuffled keys
    return sorted(out.items())


def _cell_stats(x_bf16: torch.Tensor, chunk: int = 65536) -> dict:
    """f16 storage fidelity for one [n, dim] bf16 tensor."""
    n = x_bf16.shape[0]
    sq_err = sq_ref = 0.0
    max_abs = max_abs_err = 0.0
    clipped = underflowed = total = 0
    for s in range(0, n, chunk):
        c = x_bf16[s : s + chunk].to(torch.float32)
        rt = c.to(torch.float16).to(torch.float32)
        err = c - rt
        sq_err += float((err * err).sum())
        sq_ref += float((c * c).sum())
        a = c.abs()
        max_abs = max(max_abs, float(a.max()))
        max_abs_err = max(max_abs_err, float(err.abs().max()))
        clipped += int((a > F16_MAX).sum())
        underflowed += int(((c != 0) & (rt == 0)).sum())
        total += c.numel()
    return {
        "max_abs": max_abs,
        "max_abs_err": max_abs_err,
        "rms_rel_err": float(np.sqrt(sq_err / sq_ref)) if sq_ref > 0 else 0.0,
        "clip_frac": clipped / total,
        "underflow_frac": underflowed / total,
    }


def f16_fidelity(d: CachedData, n_prompts: int = 0) -> dict:
    per_cell = []
    for layer_idx, reader in _full_attention_layers(d.cache_dir):
        for h in range(reader.kv_heads):
            per_cell.append(
                {
                    "layer": layer_idx,
                    "head": h,
                    "key": _cell_stats(reader.head("keys", h)),
                    "value": _cell_stats(reader.head("values", h)),
                }
            )

    def agg(role: str) -> dict:
        cells = [c[role] for c in per_cell]
        return {
            "max_abs": max(c["max_abs"] for c in cells),
            "max_abs_err": max(c["max_abs_err"] for c in cells),
            "rms_rel_err_mean": float(np.mean([c["rms_rel_err"] for c in cells])),
            "rms_rel_err_max": max(c["rms_rel_err"] for c in cells),
            "clip_frac": max(c["clip_frac"] for c in cells),
            "underflow_frac": max(c["underflow_frac"] for c in cells),
        }

    return {
        "context_len": d.context_len,
        "f16_max": F16_MAX,
        "keys": agg("key"),
        "values": agg("value"),
        "per_cell": per_cell,
    }
