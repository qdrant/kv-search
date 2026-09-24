"""tailM: tail correction for retrieval attention, shared by the offline build and the runtime.

Retrieval attention keeps a head's top `n_retrieved` keys and drops the rest. tailM puts the dropped keys
back as ONE pseudo-key: a stand-in value vector û (a linear map of the query) weighted by an
estimate Ẑ of the dropped keys' softmax mass (keys ~ N(μ, Σ), truncated at the weakest kept score):

    out = (N_top + Ẑ·û) / (D_top + Ẑ)

Research log: research branch `FINDINGS_TAIL_CORRECTION.md` §6, §13. Build: `scripts/build_tailm.py`.
Design: `docs/superpowers/specs/2026-09-23-tailm-build-design.md`.

No model imports here (transformers only lazily in `rope_inv_freq`, for non-default RoPE).
"""

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from safetensors import safe_open
from safetensors.torch import save_file

VERSION = 2  # 2: runtime keeps `n_retrieved` keys, map fitted at `fit_cut`, α applied (spec revision 2)
TENSORS = ("weight", "bias", "mean", "cov")
F64 = torch.float64


# ------------------------------------------------------------------ file format


def head_file(out: Path, layer: int, head: int) -> Path:
    """`<out>/layerLL_headH.safetensors`, one file per (layer, KV head)."""
    return out / f"layer{layer:02d}_head{head}.safetensors"


@dataclass(frozen=True)
class TailmHead:
    """One head's tailM file: the map, the key moments and the build's decision (`meta`)."""

    layer: int
    head: int
    weight: torch.Tensor  # [d, d] f32, row-major: û = normalise(weight·q̂ + bias)·scale
    bias: torch.Tensor  # [d] f32
    mean: torch.Tensor  # [d] f32, key mean μ
    cov: torch.Tensor  # [d, d] f32, population key covariance Σ
    meta: dict[str, Any]

    @property
    def n_retrieved(self) -> int:
        """Exact keys runtime keeps; the file applies only at exactly this retrieval size (spec §7)."""
        return int(self.meta["n_retrieved"])

    @property
    def fit_cut(self) -> int:
        """Depth the map was fitted at (its tail starts below this rank). Build-only: runtime keeps
        `n_retrieved` keys and covers everything below them with the Gaussian mass (spec §6.5).
        """
        return int(self.meta["fit_cut"])

    @property
    def alpha(self) -> float:
        """Mass multiplier on Ẑ, applied at runtime (spec §6.6)."""
        return float(self.meta["alpha"])

    @property
    def scale(self) -> float:
        return float(self.meta["scale"])

    @property
    def gate_pass(self) -> bool:
        """False: runtime ignores this file and uses the global `n_retrieved` (spec §7)."""
        return bool(self.meta["gate_pass"])


def save_head(
    path: Path, tensors: dict[str, torch.Tensor], meta: dict[str, Any]
) -> None:
    """Write one head file atomically: a `.tmp` sibling, then `os.replace` onto `path`."""
    missing = sorted(set(TENSORS) - set(tensors))
    if missing:
        raise ValueError(f"save_head: missing tensors {missing}")
    data = {
        k: tensors[k].detach().to("cpu", torch.float32).contiguous() for k in TENSORS
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    save_file(data, str(tmp), metadata={"tailm": json.dumps(meta)})
    os.replace(tmp, path)


def load_head(path: Path) -> TailmHead:
    with safe_open(str(path), framework="pt") as f:
        md = f.metadata() or {}
        if "tailm" not in md:
            raise ValueError(f"{path}: not a tailM file (no 'tailm' metadata)")
        meta = json.loads(md["tailm"])
        if meta.get("version") != VERSION:
            raise ValueError(
                f"{path}: tailM file version {meta.get('version')!r}, this code reads version {VERSION}"
            )
        t = {k: f.get_tensor(k) for k in TENSORS}
    return TailmHead(layer=int(meta["layer"]), head=int(meta["head"]), meta=meta, **t)


def load_dir(out: Path) -> dict[tuple[int, int], TailmHead]:
    """Every head file in `out`, keyed by (layer, head)."""
    heads: dict[tuple[int, int], TailmHead] = {}
    for p in sorted(out.glob("layer*_head*.safetensors")):
        h = load_head(p)
        heads[(h.layer, h.head)] = h
    return heads


def export_replay(head: TailmHead, root: Path) -> None:
    """kv-replay's `--tail` / `--moments` layout (research `make_tail.py --model linear` /
    `make_moments.py`), byte-identical to the safetensors tensors (spec §7.1)."""
    name = f"layer{head.layer:02d}_head{head.head}"
    m = head.meta
    tail, mom = root / "tailvecs_linear" / name, root / "moments" / name
    tail.mkdir(parents=True, exist_ok=True)
    mom.mkdir(parents=True, exist_ok=True)

    def raw(t: torch.Tensor) -> bytes:
        return np.ascontiguousarray(t.numpy(), "<f4").tobytes()

    (tail / "weight.f32").write_bytes(raw(head.weight))
    (tail / "bias.f32").write_bytes(raw(head.bias))
    (tail / "meta.json").write_text(
        json.dumps(
            {
                "kind": "linear",
                "scale": m["scale"],
                "ridge": m["ridge"],
                "dim": m["dim"],
                "excluded_top_m": m["fit_cut"],
                "frame": "f32",
                "source": m["source_cache"],
                "prefill_samples": m["samples"] - m["heldout"],
                "seed": m["seed"],
                "n_retrieved": m["n_retrieved"],
                "alpha": m["alpha"],
            }
        )
    )
    (mom / "mean.f32").write_bytes(raw(head.mean))
    (mom / "cov.f32").write_bytes(raw(head.cov))
    (mom / "meta.json").write_text(json.dumps({"dim": m["dim"], "n_keys": m["n_keys"]}))


# ------------------------------------------------------------------ formulas


def unit(x: torch.Tensor) -> torch.Tensor:
    """Normalise the last axis; norms floored at 1e-30 (the research code's `unit`)."""
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-30)


def rel_err(out: torch.Tensor, exact: torch.Tensor) -> torch.Tensor:
    """Relative L2 error per row, ‖out − exact‖ / ‖exact‖ (spec §6.3)."""
    return (out - exact).norm(dim=-1) / exact.norm(dim=-1)


def standin(
    q: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, scale: float
) -> torch.Tensor:
    """The tail stand-in û = normalise(weight·q̂ + bias)·scale for queries `q` [..., d], in f64.
    Products (weight·q̂) here, the rest in `standin_finish`."""
    w = weight.to(q.device, F64)
    return standin_finish(unit(q.to(F64)) @ w.T, bias, scale)


def standin_finish(
    wq_hat: torch.Tensor,
    bias: torch.Tensor,
    scale: float | torch.Tensor,
    dtype: torch.dtype = F64,
) -> torch.Tensor:
    """û = unit(wq_hat + bias)·scale from the map's product `wq_hat` = weight·q̂ [..., d], in `dtype`.
    `bias` broadcasts against it ([d] for one head, [H, 1, d] for H heads); a tensor `scale` is cast
    to `dtype` too, so the result is always `dtype`."""
    if isinstance(scale, torch.Tensor):
        scale = scale.to(wq_hat.device, dtype)
    return unit(wq_hat.to(dtype) + bias.to(wq_hat.device, dtype)) * scale


def gaussian_tail_mass(
    q: torch.Tensor,
    mean: torch.Tensor,
    cov: torch.Tensor,
    scaling: float,
    n_keys: int,
    boundary: torch.Tensor | float,
) -> torch.Tensor:
    """Absolute log softmax mass of the keys scoring below `boundary` (scaled scores), keys ~ N(mean, cov):

        ln Ẑ = ln n_keys + μ_s + σ_s²/2 + ln Φ((boundary − μ_s − σ_s²) / σ_s)
        μ_s = scaling·q·μ,   σ_s² = scaling²·qᵀΣq

    `q` [..., d]; `boundary` broadcasts against q's leading shape (e.g. [C, P] for q [P, d]).
    Products (μ_s, σ_s²) here, the rest and the edge cases in `gaussian_tail_mass_finish`. The
    products are computed before the n_keys == 0 (empty tail) return, so mismatched q / mean / cov
    shapes raise even then.
    f64. Subtract a query's max score `m` for the mass relative to exp(m).
    """
    q = q.to(F64)
    mu = mean.to(q.device, F64)
    sig = cov.to(q.device, F64)
    mu_s = scaling * (q @ mu)
    var = scaling**2 * ((q @ sig) * q).sum(-1)
    return gaussian_tail_mass_finish(mu_s, var, n_keys, boundary)


def gaussian_tail_mass_finish(
    mu_s: torch.Tensor,
    var: torch.Tensor,
    n_keys: int,
    boundary: torch.Tensor | float,
    dtype: torch.dtype = F64,
) -> torch.Tensor:
    """ln Ẑ of `gaussian_tail_mass` from its products μ_s = scaling·q·μ and σ_s² = scaling²·qᵀΣq
    (same shape; `boundary` broadcasts against them), in `dtype`. Edge cases exactly as kv-replay
    `attn::gaussian_tail_partition`: σ_s = sqrt(max(σ_s², 0)); σ_s = 0 → no truncation factor;
    n_keys == 0 or a non-finite result → −inf (empty tail)."""
    mu_s, var = mu_s.to(dtype), var.to(dtype)
    boundary = torch.as_tensor(boundary, dtype=dtype, device=mu_s.device)
    if n_keys == 0:
        shape = torch.broadcast_shapes(mu_s.shape, boundary.shape)
        return torch.full(shape, -math.inf, dtype=dtype, device=mu_s.device)
    sigma = var.clamp_min(0.0).sqrt()
    base = math.log(n_keys) + mu_s + 0.5 * var
    trunc = torch.special.log_ndtr((boundary - mu_s - var) / sigma)
    lse = torch.where(sigma > 0, base + trunc, base)
    return torch.where(torch.isfinite(lse), lse, torch.full_like(lse, -math.inf))


def combine(
    n_top: torch.Tensor, d_top: torch.Tensor, u_hat: torch.Tensor, z: torch.Tensor
) -> torch.Tensor:
    """tailM output (N_top + Z·û) / (D_top + Z); all masses relative to the same exp(m)."""
    return (n_top + z[..., None] * u_hat) / (d_top + z)[..., None]


def merge_lse(
    out_a: torch.Tensor, lse_a: torch.Tensor, out_b: torch.Tensor, lse_b: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge two softmax partitions by absolute log-sum-exp (as `cache._merge_partitions`)."""
    lse = torch.logaddexp(lse_a, lse_b)
    return (
        torch.exp(lse_a - lse)[..., None] * out_a
        + torch.exp(lse_b - lse)[..., None] * out_b,
        lse,
    )


# ------------------------------------------------------------------ prefill queries + RoPE
# Ported from the research branch `kv_search/proj.py`.


def _query_file_layout(path: Path) -> tuple[int, list[int]]:
    """`(data_offset, shape)` of a `queries_LL.safetensors` written by `RecordingCache`
    (shape `[1, seq, q_heads, dim]`, bf16)."""
    with path.open("rb") as f:
        n = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(n))
    entry = header["queries"]
    if entry["dtype"] != "BF16":
        raise ValueError(f"{path}: expected BF16 queries, got {entry['dtype']}")
    return 8 + n + entry["data_offsets"][0], entry["shape"]


def load_prompt_queries_at(
    cache_dir: Path, layer: int, kv_head: int, positions: npt.ArrayLike, group: int
) -> npt.NDArray[np.float32]:
    """Recorded post-RoPE prompt queries of one KV head's `group` q-heads at sorted, unique
    `positions`: f32 `[group, len(positions), dim]`, read from the bf16 memmap."""
    path = cache_dir / f"queries_{layer:02d}.safetensors"
    offset, shape = _query_file_layout(path)
    _, seq, heads, dim = shape
    positions = np.asarray(positions, dtype=np.int64)
    if positions.size and not (0 <= positions.min() and positions.max() < seq):
        raise ValueError(f"{path}: positions outside seq {seq}")
    if not (0 <= kv_head * group + group <= heads):
        raise ValueError(f"{path}: kv_head {kv_head} outside {heads} q-heads")
    mm = np.memmap(
        path, dtype=np.uint16, mode="r", offset=offset, shape=(seq, heads, dim)
    )
    raw = np.ascontiguousarray(
        mm[positions, kv_head * group : (kv_head + 1) * group, :]
    )
    del mm
    f32 = (raw.astype(np.uint32) << 16).view(np.float32)  # bf16 = top 16 bits of an f32
    return np.ascontiguousarray(f32.transpose(1, 0, 2))


def rope_inv_freq(config: Any) -> npt.NDArray[np.float64]:
    """Inverse RoPE frequencies, one per rotated (i, i + rot/2) pair.

    Qwen3.5 uses interleaved M-RoPE, but for text all three position axes are equal, which
    collapses to plain 1-D RoPE on the first `head_dim * partial_rotary_factor` dims with
    rotate-half pairing. For any `rope_type` other than "default" (the 1M tier runs YaRN, see
    `main._apply_yarn`) the frequencies come from transformers' own init function, so a shift
    here equals the model's own rotation. YaRN's attention scaling (the cos/sin multiplier) is
    deliberately NOT applied: the recorded queries already carry it and a shift is a pure rotation.
    """
    text = getattr(config, "text_config", config)
    rp = getattr(text, "rope_parameters", None) or {}
    rope_type = rp.get("rope_type", rp.get("type", "default"))
    if rope_type != "default":
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

        inv_freq, _attention_scaling = ROPE_INIT_FUNCTIONS[rope_type](
            text, device="cpu"
        )
        return inv_freq.detach().cpu().numpy().astype(np.float64)
    theta = rp.get("rope_theta", getattr(text, "rope_theta", None))
    if theta is None:
        raise ValueError("model config has no rope_theta")
    factor = rp.get(
        "partial_rotary_factor", getattr(text, "partial_rotary_factor", 1.0)
    )
    rot = int(text.head_dim * factor)
    return 1.0 / (float(theta) ** (np.arange(0, rot, 2, dtype=np.float64) / rot))


def rope_shift(
    q: npt.ArrayLike, delta: npt.ArrayLike, inv_freq: npt.NDArray[np.float64]
) -> npt.NDArray[np.float32]:
    """Rotate post-RoPE queries `q` [n, d] forward by `delta` [n] positions (f64 internally).

    RoPE at position p multiplies pair (i, i + rot/2) by the rotation of angle p·inv_freq[i]; a
    query recorded at p and wanted at p' needs the extra rotation by (p' − p)·inv_freq. Dims
    beyond the rotary part are left alone.
    """
    npairs = inv_freq.shape[0]
    rot = 2 * npairs
    y = np.array(q, dtype=np.float64, copy=True)
    ang = np.asarray(delta, dtype=np.float64)[:, None] * inv_freq[None, :]
    c, s = np.cos(ang), np.sin(ang)
    a, b = y[:, :npairs].copy(), y[:, npairs:rot].copy()
    y[:, :npairs] = a * c - b * s
    y[:, npairs:rot] = a * s + b * c
    return y.astype(np.float32)
