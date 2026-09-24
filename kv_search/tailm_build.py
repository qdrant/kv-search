"""tailM build: loading, one scan over all keys, the per-cut ladder, ridge fit, cut choice, gate, α.

Spec §5–§6 (`docs/superpowers/specs/2026-09-23-tailm-build-design.md`). Everything is torch on one
device. Mirrors the research method (research branch `build/tail-cut/cut_sweep3.py`,
`tail_guard.py`) so the defaults reproduce its numbers.
"""

import compression.zstd
import json
import math
import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from kv_search.tailm import (
    VERSION,
    combine,
    gaussian_tail_mass,
    load_prompt_queries_at,
    rel_err,
    rope_shift,
    unit,
)

F64 = torch.float64
SHIFT_SPAN = 256  # prefill queries are re-positioned to context_len + U[0, SHIFT_SPAN)
_ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"


# ------------------------------------------------------------------ parsing


@dataclass(frozen=True)
class CutShift:
    """`--cut-shift`: `P%` adds round(cut·P/100), an integer `k` adds k (spec §4)."""

    text: str
    percent: bool
    value: float

    def apply(self, cut: int, n_keys: int) -> int:
        delta = cut * self.value / 100 if self.percent else self.value
        delta = math.copysign(
            math.floor(abs(delta) + 0.5), delta
        )  # round half away from zero
        return int(min(max(cut + delta, 1), n_keys - 1))


_SHIFT = re.compile(r"^([+-]?\d+(?:\.\d+)?)(%?)$")
_CELL = re.compile(r"^L(\d+)H(\d+)$", re.IGNORECASE)


def parse_cut_shift(text: str) -> CutShift:
    m = _SHIFT.match(text.strip())
    if not m or (not m.group(2) and "." in m.group(1)):
        raise ValueError(
            f"--cut-shift {text!r}: expected an integer (e.g. 2000, -16) or a percentage (e.g. 10%, -5%)"
        )
    return CutShift(
        text=text.strip(), percent=bool(m.group(2)), value=float(m.group(1))
    )


def make_ladder(min_cut: int, max_cut: int, cuts: list[int] | None = None) -> list[int]:
    """The candidate cuts: `cuts` if given (sorted, unique), else the powers of two in [min_cut, max_cut]."""
    if cuts:
        out = sorted(set(cuts))
    else:
        out = [
            1 << k
            for k in range(max(max_cut, 1).bit_length() + 1)
            if min_cut <= (1 << k) <= max_cut
        ]
    if not out or out[0] < 1:
        raise ValueError(
            f"empty or non-positive cut ladder {out} (min {min_cut}, max {max_cut}, cuts {cuts})"
        )
    return out


def eval_cuts(
    ladder: list[int], shift: CutShift, n_keys: int, n_retrieved: int = 0
) -> list[int]:
    """Every cut the build evaluates: the ladder, each ladder cut's shifted final cut and the runtime's
    `n_retrieved` kept keys (every fit cut max(final, n_retrieved) is among them). The scan's top-k
    must reach `max(eval_cuts(...))`."""
    return sorted(
        set(ladder)
        | {shift.apply(c, n_keys) for c in ladder}
        | ({n_retrieved} if n_retrieved else set())
    )


def parse_cells(text: str) -> list[tuple[int, int]]:
    cells = []
    for part in text.split(","):
        m = _CELL.match(part.strip())
        if not m:
            raise ValueError(f"--cells: {part.strip()!r} is not a cell like L15H3")
        cells.append((int(m.group(1)), int(m.group(2))))
    return cells


def plan_batches(
    cells: list[tuple[int, int]], batch_size: int
) -> list[list[tuple[int, int]]]:
    """Layer-major batches of at most `batch_size` heads. A layer's heads stay in one batch unless
    the layer alone has more heads than `batch_size` (then it is split into full batches).
    """
    if batch_size < 1:
        raise ValueError(f"--batch-size must be >= 1, got {batch_size}")
    by_layer: dict[int, list[tuple[int, int]]] = {}
    for cell in sorted(set(cells)):
        by_layer.setdefault(cell[0], []).append(cell)
    batches: list[list[tuple[int, int]]] = []
    cur: list[tuple[int, int]] = []
    for group in by_layer.values():
        if len(group) > batch_size:
            if cur:
                batches.append(cur)
                cur = []
            batches += [
                group[i : i + batch_size] for i in range(0, len(group), batch_size)
            ]
            continue
        if len(cur) + len(group) > batch_size:
            batches.append(cur)
            cur = []
        cur += group
    if cur:
        batches.append(cur)
    return batches


def tier_label(cache_dir: Path) -> str | None:
    """The qdrant tier of `cache/qdrant/<size>/<model_type>` (e.g. "1M"), else None."""
    parts = Path(os.path.abspath(cache_dir)).parts
    return parts[-2] if len(parts) >= 3 and parts[-3] == "qdrant" else None


def tier_context_tokens(cache_dir: Path) -> int:
    """The context tokens `prefill` hands the YaRN patch for this cache: the qdrant tier label's
    tokens (`_size_to_tokens`), else 0 (other datasets run without YaRN). Spec §3."""
    label = tier_label(cache_dir)
    if label is None:
        return 0
    from kv_search.main import _size_to_tokens

    return _size_to_tokens(label)


def resolve_context_tokens(cache_dir: Path, context_len: int, given: int | None) -> int:
    """The context tokens for the YaRN patch (spec §3): `given` if set, else the qdrant tier label's
    tokens, else 0 -- but only when the cache fits the native window. A longer cache at a path without
    a tier label is refused: prefill ran it with a YaRN factor we cannot see, and building without YaRN
    rotates the shifted queries wrongly while making the prefill numbers look better (measured on
    L15H3: prefill error 0.092 instead of 0.106, decode error 0.203 instead of 0.164; the validation
    sanity check does not use shifted queries and cannot catch it)."""
    if given is not None:
        return given
    if tier_label(cache_dir) is not None:
        return tier_context_tokens(cache_dir)
    from kv_search.main import NATIVE_MAX_POSITIONS

    if context_len > NATIVE_MAX_POSITIONS:
        raise ValueError(
            f"{cache_dir}: {context_len:,} tokens exceed the native {NATIVE_MAX_POSITIONS:,} and the path has no "
            f"cache/qdrant/<tier>/ label telling which YaRN factor prefill used; pass --context-tokens "
            f"(the tier's tokens, e.g. 1000000, or 0 if prefill ran without YaRN)"
        )
    return 0


# ------------------------------------------------------------------ layer files


def layer_file(cache_dir: Path, layer: int) -> Path:
    """The layer file, raw preferred over zstd (same preference as `cache._layer_files`)."""
    for name in (
        f"layer_{layer:02d}.safetensors",
        f"layer_{layer:02d}.safetensors.zst",
    ):
        if (cache_dir / name).exists():
            return cache_dir / name
    raise FileNotFoundError(f"{cache_dir}: no layer_{layer:02d}.safetensors[.zst]")


class LayerReader:
    """Keys/values of one full-attention layer file, one KV head at a time.

    The file holds `keys` / `values` as the byte-shuffled bf16 planes of `cache._shuffle_bf16`
    (`U8 [2, 1, kv_heads, n_keys, dim]`: plane 0 = low bytes, plane 1 = high bytes). A raw file is
    memory-mapped (only the requested head is read); a zstd file is decompressed once, here.
    """

    def __init__(self, path: Path):
        self.path = path
        with path.open("rb") as f:
            magic = f.read(4)
        if magic == _ZSTD_MAGIC:
            self._buf = np.frombuffer(
                compression.zstd.decompress(path.read_bytes()), dtype=np.uint8
            )
        else:
            self._buf = np.memmap(path, dtype=np.uint8, mode="r")
        n = int.from_bytes(self._buf[:8].tobytes(), "little")
        self._header = json.loads(self._buf[8 : 8 + n].tobytes())
        self._data = 8 + n
        e = self._header.get("keys")
        if (
            e is None
            or e["dtype"] != "U8"
            or len(e["shape"]) != 5
            or e["shape"][0] != 2
        ):
            raise ValueError(
                f"{path}: no byte-shuffled bf16 'keys' as cache.save_cache writes them "
                f"(got {None if e is None else (e['dtype'], e['shape'])}); is this a full-attention layer?"
            )
        _, _, self.kv_heads, self.n_keys, self.dim = e["shape"]

    def head(self, tensor: str, head: int) -> torch.Tensor:
        """`[n_keys, dim]` bf16 on the host for KV head `head` of `tensor` ("keys" or "values")."""
        if not 0 <= head < self.kv_heads:
            raise ValueError(
                f"{self.path}: head {head} outside {self.kv_heads} KV heads"
            )
        start = self._data + self._header[tensor]["data_offsets"][0]
        plane = self.kv_heads * self.n_keys * self.dim
        span = self.n_keys * self.dim
        lo = self._buf[start + head * span : start + (head + 1) * span]
        hi = self._buf[start + plane + head * span : start + plane + (head + 1) * span]
        u16 = lo.astype(np.uint16) | (hi.astype(np.uint16) << 8)
        return (
            torch.from_numpy(u16.view(np.int16))
            .view(torch.bfloat16)
            .reshape(self.n_keys, self.dim)
        )


# ------------------------------------------------------------------ queries


def sample_queries(
    cache_dir: Path,
    layer: int,
    kv_head: int,
    context_len: int,
    samples: int,
    seed: int,
    inv_freq: np.ndarray,
    group: int,
) -> np.ndarray:
    """The build's prefill queries for one head (spec §6.1; research `cut_sweep3.py queries()`):
    `samples` distinct prompt positions, their `group` q-heads RoPE-shifted to
    `context_len + U[0, SHIFT_SPAN)`, one random q-head kept per position. f32 `[samples, d]`.
    """
    rng = np.random.default_rng(seed)
    pos = np.sort(rng.choice(context_len, samples, replace=False))
    qp = load_prompt_queries_at(cache_dir, layer, kv_head, pos, group)
    delta = (context_len + rng.integers(0, SHIFT_SPAN, size=len(pos)) - pos).astype(
        np.float64
    )
    qs = np.stack([rope_shift(qp[g], delta, inv_freq) for g in range(group)])
    pick = rng.integers(0, group, size=len(pos))
    return np.ascontiguousarray(qs[pick, np.arange(len(pos))]).astype(np.float32)


def split_indices(samples: int, heldout: int) -> tuple[np.ndarray, np.ndarray]:
    """(held-out, train) row indices; the research split (`default_rng(0).permutation`)."""
    perm = np.random.default_rng(0).permutation(samples)
    return perm[:heldout], perm[heldout:]


# ------------------------------------------------------------------ scan


@dataclass
class HeadScan:
    """One head's slice of a `ScanResult`."""

    m: torch.Tensor  # [P] f32 max scaled score
    D: torch.Tensor  # [P] f64 Σ exp(s − m)
    N: torch.Tensor  # [P, d] f64 Σ exp(s − m)·v
    top_sc: torch.Tensor  # [P, k] f32, descending
    top_ix: torch.Tensor  # [P, k] int32 key ids


@dataclass
class ScanResult:
    m: torch.Tensor  # [H, P] f32
    D: torch.Tensor  # [H, P] f64
    N: torch.Tensor  # [H, P, d] f64
    top_sc: torch.Tensor  # [H, P, kmax] f32
    top_ix: torch.Tensor  # [H, P, kmax] int32
    s1: torch.Tensor | None  # [H, d] f64 Σ k
    s2: torch.Tensor | None  # [H, d, d] f64 Σ k kᵀ
    n_keys: int

    def head(self, i: int) -> HeadScan:
        return HeadScan(self.m[i], self.D[i], self.N[i], self.top_sc[i], self.top_ix[i])


def scan(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    scaling: float,
    kmax: int,
    chunk: int,
    *,
    moments: bool = True,
    on_chunk: Callable[[int, int], None] | None = None,
) -> ScanResult:
    """One pass over all keys for H heads at once (spec §6.2).

    Q [H, P, d] f32, K/V [H, n_keys, d] bf16, all on one device. Online softmax (f32 scores, exp
    and w @ v; f64 running totals, as research cut_sweep3.py), the top-`kmax` scores and ids per
    query (two-stage: the chunk's own top-kmax, then a merge with the running top-kmax), and with
    `moments` the f64 key sums Σk, ΣkkT. `on_chunk(keys_done, n_keys)` after every chunk.
    """
    H, P, d = Q.shape
    nk = K.shape[1]
    if not 1 <= kmax <= nk:
        raise ValueError(f"kmax {kmax} outside [1, {nk}]")
    dev = Q.device
    m = torch.full((H, P), -math.inf, device=dev)
    D = torch.zeros(H, P, dtype=F64, device=dev)
    N = torch.zeros(H, P, d, dtype=F64, device=dev)
    top_sc = torch.empty(H, P, 0, device=dev)
    top_ix = torch.empty(H, P, 0, dtype=torch.int32, device=dev)
    s1 = torch.zeros(H, d, dtype=F64, device=dev) if moments else None
    s2 = torch.zeros(H, d, d, dtype=F64, device=dev) if moments else None
    for s in range(0, nk, chunk):
        e = min(s + chunk, nk)
        kc = K[:, s:e].float()
        if moments:
            kd = kc.double()
            s1 += kd.sum(1)
            s2 += kd.transpose(1, 2) @ kd
            del kd
        sc = torch.bmm(Q, kc.transpose(1, 2)) * scaling  # [H, P, c]
        del kc
        m_new = torch.maximum(m, sc.amax(2))
        resc = torch.exp((m - m_new).double())
        w = (sc - m_new[..., None]).exp_()
        D = D * resc + w.sum(2).double()
        N = N * resc[..., None] + torch.bmm(w, V[:, s:e].float()).double()
        del w
        m = m_new
        c_sc, c_ix = sc.topk(min(kmax, e - s), dim=2)
        del sc
        cat_sc = torch.cat([top_sc, c_sc], 2)
        cat_ix = torch.cat([top_ix, (c_ix + s).to(torch.int32)], 2)
        del c_sc, c_ix
        top_sc, sel = cat_sc.topk(min(kmax, cat_sc.shape[2]), dim=2)
        top_ix = torch.gather(cat_ix, 2, sel)
        del cat_sc, cat_ix, sel
        if on_chunk is not None:
            on_chunk(e, nk)
    return ScanResult(m, D, N, top_sc, top_ix, s1, s2, nk)


def moments(res: ScanResult, i: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Key mean μ and population covariance Σ = ΣkkT/N − μμᵀ of head `i` (f64)."""
    if res.s1 is None or res.s2 is None:
        raise ValueError("scan ran with moments=False")
    mu = res.s1[i] / res.n_keys
    return mu, res.s2[i] / res.n_keys - torch.outer(mu, mu)


# ------------------------------------------------------------------ ladder sums + fit


def top_sums(
    top_sc: torch.Tensor,
    top_ix: torch.Tensor,
    m: torch.Tensor,
    V: torch.Tensor,
    cuts: list[int],
    max_bytes: int = 1 << 30,
) -> tuple[torch.Tensor, torch.Tensor]:
    """D_top and N_top at every cut for one head (spec §6.3), f64.

    top_sc [P, k] f32 descending, top_ix [P, k], m [P] (the scan's max), V [n_keys, d] bf16,
    `cuts` strictly ascending with cuts[-1] ≤ k. As research cut_sweep3.py: D by a cumulative sum
    over ranks; N by per-segment sums between consecutive cut edges and a cumulative sum over the
    segments. Values are gathered in query slices so one [slice, cuts[-1], d] f64 block stays
    under `max_bytes`. Returns (Dt [C, P], Nt [C, P, d]).
    """
    P = top_sc.shape[0]
    d = V.shape[1]
    kk = cuts[-1]
    if kk > top_sc.shape[1]:
        raise ValueError(f"cut {kk} beyond the scan's top-{top_sc.shape[1]}")
    wt = torch.exp((top_sc[:, :kk] - m[:, None]).to(F64))  # [P, kk]
    Dt = torch.cumsum(wt, 1)[:, [c - 1 for c in cuts]].T.contiguous()
    edges = [0, *cuts]
    seg = torch.zeros(len(cuts), P, d, dtype=F64, device=V.device)
    step = max(1, max_bytes // (kk * d * 8))
    for a in range(0, P, step):
        b = min(a + step, P)
        vt = V[top_ix[a:b, :kk].long()].to(F64)  # [slice, kk, d]
        for i in range(len(cuts)):
            lo, hi = edges[i], edges[i + 1]
            seg[i, a:b] = torch.einsum("pn,pnd->pd", wt[a:b, lo:hi], vt[:, lo:hi])
        del vt
    return Dt, torch.cumsum(seg, 0)


def ridge_fit(X: torch.Tensor, Y: torch.Tensor, ridge: float) -> torch.Tensor:
    """Ridge least squares for every cut at once (spec §6.3).

    X [n, d+1] f64 = [unit query, 1] (train rows), Y [C, n, d] f64 = unit tail direction per cut.
    A = XᵀX + λI with λ = ridge·mean(diag XᵀX) is the same for every cut, so one Cholesky solves
    all of them. Returns B [C, d+1, d]: pred = X @ B[c]; weight = B[c][:d].T, bias = B[c][d].
    """
    A = X.T @ X
    A = A + ridge * torch.diagonal(A).mean() * torch.eye(
        A.shape[0], dtype=A.dtype, device=A.device
    )
    L = torch.linalg.cholesky(A)
    C, _, d = Y.shape
    k = X.shape[1]
    rhs = torch.einsum("nk,cnd->kcd", X, Y).reshape(k, C * d)
    return torch.cholesky_solve(rhs, L).reshape(k, C, d).permute(1, 0, 2).contiguous()


# ------------------------------------------------------------------ evaluation + decision

ALPHAS = [
    round(i * 0.05, 2) for i in range(31)
]  # mass-factor grid 0 .. 1.5 (research tail_guard.py); rounded so the stored α reads 0.6, not 0.6000000000000001
_STAT_KEY = {"mean": "err_mean", "p90": "err_p90"}


@dataclass
class CutEval:
    """Held-out numbers and per-cut tensors for one head (spec §6.3)."""

    cuts: list[int]
    stats: dict[int, dict[str, float]]
    B: torch.Tensor  # [C, d+1, d] f64 ridge solution per cut
    scale: torch.Tensor  # [C] f64 median ‖u‖ over train
    Dt: torch.Tensor  # [C, P] f64
    Nt: torch.Tensor  # [C, P, d] f64
    uh: torch.Tensor  # [C, P, d] f64 stand-in û
    Z: torch.Tensor  # [C, P] f64 Gaussian mass relative to exp(m)
    exact: torch.Tensor  # [P, d] f64 exact attention output


def evaluate_cuts(
    q: torch.Tensor,
    hs: HeadScan,
    V: torch.Tensor,
    mu: torch.Tensor,
    Sigma: torch.Tensor,
    n_keys: int,
    scaling: float,
    cuts: list[int],
    tr: torch.Tensor,
    te: torch.Tensor,
    ridge: float,
) -> CutEval:
    """Every cut at once: true tail, ridge map (train rows), stand-in, truncation-aware Gaussian
    mass, and the held-out numbers (spec §6.3). `q` [P, d] f32 are the build queries; `tr`, `te`
    index rows of `q`."""
    P = q.shape[0]
    qn = unit(q.to(F64))
    X = torch.cat([qn, torch.ones(P, 1, dtype=F64, device=q.device)], 1)
    exact = hs.N / hs.D[:, None]
    Dt, Nt = top_sums(hs.top_sc, hs.top_ix, hs.m, V, cuts)
    Dtail = hs.D[None] - Dt  # [C, P]
    u = (hs.N[None] - Nt) / Dtail[..., None]  # true tail vector
    un = unit(u)
    B = ridge_fit(X[tr], un[:, tr], ridge)
    scale = torch.quantile(u[:, tr].norm(dim=2), 0.5, dim=1)  # = np.median (even count)
    Pd = torch.einsum("pk,ckd->cpd", X, B)
    uh = unit(Pd) * scale[:, None, None]
    bnd = hs.top_sc[:, [c - 1 for c in cuts]].T.to(
        F64
    )  # [C, P] weakest kept scaled score
    Z = torch.exp(gaussian_tail_mass(q, mu, Sigma, scaling, n_keys, bnd) - hs.m.to(F64))
    e_t = rel_err(
        combine(Nt[:, te], Dt[:, te], uh[:, te], Z[:, te]), exact[te]
    )  # [C, nte]
    e_n = rel_err(Nt[:, te] / Dt[:, te, None], exact[te])
    mu_dir = un[:, tr].mean(1)
    const = (mu_dir * mu_dir).sum(1)
    r2 = 1 - ((un[:, te] - Pd[:, te]) ** 2).sum((1, 2)) / (
        (un[:, te] - mu_dir[:, None]) ** 2
    ).sum((1, 2))
    phi = (Dtail[:, te] / hs.D[te]).mean(1)
    leftover = (1 - const) * (1 - r2)
    cols = {
        "err_mean": e_t.mean(1),
        "err_p90": torch.quantile(e_t, 0.9, dim=1),
        "none_mean": e_n.mean(1),
        "worse": (e_t > e_n).to(F64).mean(1),
        "phi": phi,
        "const": const,
        "r2": r2,
        "leftover": leftover,
        "pred": phi * leftover.sqrt(),
        "mass_ratio": torch.quantile(Z[:, te] / Dtail[:, te], 0.5, dim=1),
        "cos_median": torch.quantile(
            torch.nn.functional.cosine_similarity(uh[:, te], u[:, te], dim=2),
            0.5,
            dim=1,
        ),
    }
    cols["ratio"] = cols["err_mean"] / cols["none_mean"]
    host = {k: v.tolist() for k, v in cols.items()}
    stats = {c: {k: host[k][i] for k in host} for i, c in enumerate(cuts)}
    return CutEval(list(cuts), stats, B, scale, Dt, Nt, uh, Z, exact)


def choose_cut(
    stats: dict[int, dict[str, float]], ladder: list[int], eps: float, eps_stat: str
) -> tuple[int, bool]:
    """The smallest ladder cut whose `eps_stat` error is ≤ eps; else (deepest, False). Spec §6.4."""
    key = _STAT_KEY[eps_stat]
    for c in ladder:
        if stats[c][key] <= eps:
            return c, True
    return ladder[-1], False


def _runtime_errs(
    ev: CutEval, fit_cut: int, keep: int, idx: torch.Tensor, alphas: torch.Tensor
) -> torch.Tensor:
    """tailM error [A, n] at the runtime configuration (spec §6.5): exact part and Gaussian boundary at
    the `keep` kept keys, stand-in from the map fitted at `fit_cut`, mass scaled by each α.
    """
    k, f = ev.cuts.index(keep), ev.cuts.index(fit_cut)
    z = alphas[:, None] * ev.Z[k, idx][None]
    out = combine(ev.Nt[k, idx][None], ev.Dt[k, idx][None], ev.uh[f, idx][None], z)
    return rel_err(out, ev.exact[idx][None])


def fit_alpha(
    ev: CutEval, fit_cut: int, keep: int, tr: torch.Tensor, te: torch.Tensor
) -> tuple[float, float]:
    """α on the train split at the runtime configuration (grid ALPHAS, first minimum), and the held-out
    mean error with it. Mass multiplier out = (N_top + α·Ẑ·û)/(D_top + α·Ẑ), applied at runtime (spec §6.6).
    """
    grid = torch.tensor(ALPHAS, dtype=F64, device=ev.Z.device)
    k = int(torch.argmin(_runtime_errs(ev, fit_cut, keep, tr, grid).mean(1)))
    return ALPHAS[k], float(
        _runtime_errs(ev, fit_cut, keep, te, grid[k : k + 1]).mean()
    )


def ratio_stats(err: torch.Tensor, base: torch.Tensor) -> dict[str, float]:
    """`worse` = share of rows with err > base; p99 / max (and median) of err/base over those rows,
    1.0 when no row is worse (spec §6.7)."""
    w = err > base
    r = (err / base)[w]
    return {
        "worse": float(w.to(F64).mean()),
        "ratio_med": float(torch.quantile(r, 0.5)) if w.any() else 1.0,
        "ratio_p99": float(torch.quantile(r, 0.99)) if w.any() else 1.0,
        "ratio_max": float(r.max()) if w.any() else 1.0,
    }


def runtime_stats(
    ev: CutEval, fit_cut: int, keep: int, alpha: float, te: torch.Tensor
) -> dict[str, float]:
    """Held-out prefill numbers at the runtime configuration (spec §6.5); `none` = kept keys only."""
    grid = torch.tensor([alpha, 1.0], dtype=F64, device=ev.Z.device)
    e = _runtime_errs(ev, fit_cut, keep, te, grid)
    k = ev.cuts.index(keep)
    none = rel_err(ev.Nt[k, te] / ev.Dt[k, te, None], ev.exact[te])
    rs = ratio_stats(e[0], none)
    return {
        "err_mean": float(e[0].mean()),
        "err_p90": float(torch.quantile(e[0], 0.9)),
        "err_mean_a1": float(e[1].mean()),
        "none_mean": float(none.mean()),
        "worse": rs["worse"],
        "ratio_p99": rs["ratio_p99"],
        "ratio_max": rs["ratio_max"],
    }


@dataclass(frozen=True)
class BuildConfig:
    samples: int = 4096
    heldout: int = 1024
    seed: int = 3
    ridge: float = 1e-2
    ladder: tuple[int, ...] = (8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)
    eps: float = 0.10
    eps_stat: str = "mean"
    cut_shift: CutShift = field(default_factory=lambda: CutShift("0", False, 0.0))
    n_retrieved: int = 128


@dataclass
class HeadBuild:
    layer: int
    head: int
    tensors: dict[str, torch.Tensor]
    meta: dict[str, Any]
    warning: str | None


def build_head(
    layer: int,
    head: int,
    q: torch.Tensor,
    hs: HeadScan,
    V: torch.Tensor,
    mu: torch.Tensor,
    Sigma: torch.Tensor,
    n_keys: int,
    scaling: float,
    tr: torch.Tensor,
    te: torch.Tensor,
    cfg: BuildConfig,
    identity: dict[str, Any],
) -> HeadBuild:
    """Evaluate the ladder (+ shifted finals, + n_retrieved), choose and shift the cut, fit the map at
    fit_cut = max(final, n_retrieved), fit α and the prefill numbers at the runtime configuration, and
    assemble the file's tensors and metadata (spec §6.4–§7). The head is written gated off until the
    decode gate runs (spec §6.7). `identity`: context_len, model, source_cache, built_at.
    """
    ladder = list(cfg.ladder)
    keep = cfg.n_retrieved
    cuts = eval_cuts(ladder, cfg.cut_shift, n_keys, keep)
    if cuts[-1] > hs.top_sc.shape[1]:
        raise ValueError(
            f"cut {cuts[-1]} beyond the scan's top-{hs.top_sc.shape[1]}; scan with kmax >= max(eval_cuts)"
        )
    ev = evaluate_cuts(q, hs, V, mu, Sigma, n_keys, scaling, cuts, tr, te, cfg.ridge)
    detected, eps_met = choose_cut(ev.stats, ladder, cfg.eps, cfg.eps_stat)
    final = cfg.cut_shift.apply(detected, n_keys)
    fit_cut = max(final, keep)
    alpha, _ = fit_alpha(ev, fit_cut, keep, tr, te)
    warning = None
    if not eps_met:
        best = ev.stats[detected][_STAT_KEY[cfg.eps_stat]]
        warning = f"L{layer:02d}H{head}: eps {cfg.eps:g} not met at any cut (best {best:.3f} @ {detected}), using {detected}"
    i = cuts.index(fit_cut)
    d = q.shape[1]
    meta = {
        "version": VERSION,
        "layer": layer,
        "head": head,
        "dim": d,
        "scaling": scaling,
        "n_keys": n_keys,
        **identity,
        "samples": cfg.samples,
        "heldout": cfg.heldout,
        "seed": cfg.seed,
        "ridge": cfg.ridge,
        "ladder": ladder,
        "eps": cfg.eps,
        "eps_stat": cfg.eps_stat,
        "cut_shift": cfg.cut_shift.text,
        "n_retrieved": keep,
        "alpha": alpha,
        "gate_pass": False,
        "fit_cut": fit_cut,
        "scale": float(ev.scale[i]),
        "cut_detected": detected,
        "cut": final,
        "eps_met": eps_met,
        "prefill": runtime_stats(ev, fit_cut, keep, alpha, te),
        "cut_stats": dict(ev.stats[final]),
        "ladder_stats": [{"cut": c, **ev.stats[c]} for c in ladder],
        "gate": {"validated": False, "reason": "not validated"},
        "decode": None,
    }
    B = ev.B[i]
    tensors = {"weight": B[:d].T, "bias": B[d], "mean": mu, "cov": Sigma}
    return HeadBuild(layer, head, tensors, meta, warning)
