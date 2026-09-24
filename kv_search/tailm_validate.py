"""Optional check of built tailM heads against recorded decode sessions (spec §8).

A session file (research `replay-data/*.safetensors`) holds, per full-attention layer, the decode
`queries` [T, q_heads, d] (post-RoPE), the decode-generated `keys` / `values` [T, kv_heads, d] and
the model's `attn_out` [T, q_heads, d] (before the output gate and o_proj). Row t attends to the
whole prefill plus decode rows 0..t. The check runs the runtime configuration (spec §6.5: exact
top-`n_retrieved`, Gaussian below the weakest kept score, α) and compares it with the kept keys alone;
`gate` turns that into the per-head decision (spec §6.7). It uses the exact top-`n_retrieved`, so it
measures how the map and the mass hold up on real decode queries, not what an HNSW search misses.
"""

import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open

from kv_search.tailm import (
    TailmHead,
    combine,
    gaussian_tail_mass,
    merge_lse,
    rel_err,
    standin,
)
from kv_search.tailm_build import F64, ratio_stats, scan, top_sums

# Sanity = recomputed exact attention vs the recorded attn_out. It is gated on the median and the p99,
# not the max: the bf16 floor has rare outlier rows (all 32 heads of the 1M cache, 14,448 rows each:
# medians 1.26-1.74e-3, p99 2.07-3.72e-3, max 2.55-5.07e-3; L27H1's max comes from 1 row). A RoPE /
# scaling / loading bug shifts every row (median); a misaligned session shifts >= ~7 % of them (p99).
SANITY_MEDIAN_TOL = 2.5e-3  # 1.44x the worst measured median
SANITY_P99_TOL = 1e-2  # 2.7x the worst measured p99


@dataclass(frozen=True)
class Session:
    name: str  # trailing number of the file stem, e.g. "01"
    path: Path
    meta: dict[str, str]


def open_sessions(root: Path, only: list[str] | None = None) -> list[Session]:
    out = []
    for p in sorted(root.glob("*.safetensors")):
        m = re.search(r"(\d+)$", p.stem)
        if not m or (only and m.group(1) not in only):
            continue
        with safe_open(str(p), framework="pt") as f:
            out.append(Session(m.group(1), p, dict(f.metadata() or {})))
    if only:
        missing = sorted(set(only) - {s.name for s in out})
        if missing:
            raise ValueError(f"sessions {missing} not found in {root}")
    if not out:
        raise ValueError(f"no session files in {root}")
    return out


def check_sessions(
    sessions: list[Session],
    *,
    context_len: int,
    model: str,
    scaling: float,
    tier_size: str | None,
    layers: list[int],
) -> None:
    """Refuse sessions recorded on a different cache/model (spec §8)."""
    for s in sessions:
        md, problems = s.meta, []
        if md.get("context_len") != str(context_len):
            problems.append(
                f"context_len {md.get('context_len')} != cache {context_len}"
            )
        if md.get("model") != model:
            problems.append(f"model {md.get('model')!r} != {model!r}")
        sc = md.get("scaling")
        if sc is None or abs(float(sc) - scaling) > 1e-9:
            problems.append(f"scaling {sc} != {scaling:g}")
        if (
            tier_size is not None
            and md.get("dataset") == "qdrant"
            and md.get("qdrant_size") != tier_size
        ):
            problems.append(
                f"qdrant_size {md.get('qdrant_size')} != cache tier {tier_size}"
            )
        have = {int(x) for x in md.get("layers", "").split(",") if x.strip()}
        if not set(layers) <= have:
            problems.append(f"layers {sorted(set(layers) - have)} not recorded")
        if problems:
            raise ValueError(f"{s.path.name}: " + "; ".join(problems))


@dataclass
class DecodeRows:
    """One KV head's decode rows over all sessions, ordered (session, step, q-head)."""

    q: torch.Tensor  # [R, d] f32
    live_out: (
        torch.Tensor
    )  # [R, d] f64 exact attention over the session's decode keys ≤ step
    live_lse: torch.Tensor  # [R] f64
    recorded: torch.Tensor  # [R, d] f64 recorded attn_out


def decode_rows(
    sessions: list[Session],
    layer: int,
    kv_head: int,
    group: int,
    scaling: float,
    device: torch.device,
) -> DecodeRows:
    qs, outs, lses, recs = [], [], [], []
    for s in sessions:
        with safe_open(str(s.path), framework="pt") as f:
            q = f.get_tensor(f"layer{layer:02d}/queries")[
                :, kv_head * group : (kv_head + 1) * group
            ]
            k = f.get_tensor(f"layer{layer:02d}/keys")[:, kv_head]
            v = f.get_tensor(f"layer{layer:02d}/values")[:, kv_head]
            rec = f.get_tensor(f"layer{layer:02d}/attn_out")[
                :, kv_head * group : (kv_head + 1) * group
            ]
        T = q.shape[0]
        qd, kd, vd = (x.to(device, F64) for x in (q, k, v))
        sc = torch.einsum("tgd,sd->tgs", qd, kd) * scaling  # [T, g, T]
        causal = torch.ones(
            T, T, dtype=torch.bool, device=device
        ).tril()  # key s visible to step t iff s <= t
        sc = sc.masked_fill(~causal[:, None, :], -math.inf)
        qs.append(q.reshape(T * group, -1))
        lses.append(torch.logsumexp(sc, 2).reshape(-1))
        outs.append(
            torch.einsum("tgs,sd->tgd", torch.softmax(sc, 2), vd).reshape(T * group, -1)
        )
        recs.append(rec.reshape(T * group, -1).to(device, F64))
    return DecodeRows(
        torch.cat(qs).to(device, torch.float32),
        torch.cat(outs),
        torch.cat(lses),
        torch.cat(recs),
    )


def validate_batch(
    cells: list[tuple[int, int]],
    heads: dict[tuple[int, int], TailmHead],
    K: torch.Tensor,
    V: torch.Tensor,
    sessions: list[Session],
    *,
    group: int,
    scaling: float,
    chunk: int,
    block: int,
    on_chunk: Callable[[int, int], None] | None = None,
) -> dict[tuple[int, int], dict]:
    """Second pass over the resident K/V [H, n_keys, d] of `cells`, with the decode queries in
    blocks of `block` rows, at each head's runtime configuration (spec §6.5, §8)."""
    dev = K.device
    rows = [decode_rows(sessions, L, h, group, scaling, dev) for L, h in cells]
    R = rows[0].q.shape[0]
    Q = torch.stack([r.q for r in rows])  # [H, R, d]
    kmax = max(heads[c].n_retrieved for c in cells)
    per = [
        {k: [] for k in ("tail", "tail1", "none", "whole_t", "whole_n", "sanity")}
        for _ in cells
    ]
    n_blocks = -(-R // block)
    for bi, a in enumerate(range(0, R, block)):
        b = min(a + block, R)
        # progress over all blocks, so a caller sees one 0..100 % pass, not one per block
        cb = (
            None
            if on_chunk is None
            else (lambda d, t, bi=bi: on_chunk(bi * t + d, n_blocks * t))
        )
        res = scan(Q[:, a:b], K, V, scaling, kmax, chunk, moments=False, on_chunk=cb)
        for i, cell in enumerate(cells):
            hd, r, hs = heads[cell], rows[i], res.head(i)
            n = hd.n_retrieved
            Dt, Nt = top_sums(hs.top_sc, hs.top_ix, hs.m, V[i], [n])
            Dt, Nt = Dt[0], Nt[0]
            q = Q[i, a:b]
            m = hs.m.to(F64)
            exact = hs.N / hs.D[:, None]
            uh = standin(q, hd.weight, hd.bias, hd.scale)
            Z = torch.exp(
                gaussian_tail_mass(
                    q,
                    hd.mean,
                    hd.cov,
                    scaling,
                    int(hd.meta["n_keys"]),
                    hs.top_sc[:, n - 1],
                )
                - m
            )
            tail = combine(Nt, Dt, uh, hd.alpha * Z)
            none = Nt / Dt[:, None]
            per[i]["tail"].append(rel_err(tail, exact))
            per[i]["tail1"].append(rel_err(combine(Nt, Dt, uh, Z), exact))
            per[i]["none"].append(rel_err(none, exact))
            lo, ll = r.live_out[a:b], r.live_lse[a:b]
            ex_w, _ = merge_lse(exact, m + hs.D.log(), lo, ll)
            t_w, _ = merge_lse(tail, m + (Dt + hd.alpha * Z).log(), lo, ll)
            n_w, _ = merge_lse(none, m + Dt.log(), lo, ll)
            per[i]["whole_t"].append(rel_err(t_w, ex_w))
            per[i]["whole_n"].append(rel_err(n_w, ex_w))
            per[i]["sanity"].append(rel_err(ex_w, r.recorded[a:b]))
        del res
    out = {}
    for i, cell in enumerate(cells):
        c = {k: torch.cat(v) for k, v in per[i].items()}
        smed, sp99 = (float(torch.quantile(c["sanity"], x)) for x in (0.5, 0.99))
        w = c["tail"] > c["none"]
        out[cell] = {
            "rows": R,
            "sessions": [s.name for s in sessions],
            "dec_err_mean": float(c["tail"].mean()),
            "dec_err_p90": float(torch.quantile(c["tail"], 0.9)),
            "dec_err_mean_a1": float(c["tail1"].mean()),
            "dec_none_mean": float(c["none"].mean()),
            **ratio_stats(c["tail"], c["none"]),
            "excess_max": float((c["tail"] - c["none"])[w].max()) if w.any() else 0.0,
            "whole_err_mean": float(c["whole_t"].mean()),
            "whole_none_mean": float(c["whole_n"].mean()),
            "sanity_median": smed,
            "sanity_p99": sp99,
            "sanity_max": float(c["sanity"].max()),
            "sanity_pass": smed <= SANITY_MEDIAN_TOL and sp99 <= SANITY_P99_TOL,
        }
    return out


def gate(v: dict, *, worse_tol: float, p99_tol: float) -> dict:
    """The per-head decision from `validate_batch` numbers (spec §6.7): on only if the sanity check
    passes, at most `worse_tol` of the decode rows are worse than the kept keys alone, and the p99 of
    err(tailM)/err(kept only) over those rows is at most `p99_tol`. A gated-off head runs today's path.
    """
    reasons = []
    if not v["sanity_pass"]:
        reasons.append(
            "sanity failed (RoPE / scaling / loading bug or misaligned session)"
        )
    if v["worse"] > worse_tol:
        reasons.append(f"worse {100 * v['worse']:.1f}% > {100 * worse_tol:g}%")
    if v["ratio_p99"] > p99_tol:
        reasons.append(f"p99 {v['ratio_p99']:.2f}x > {p99_tol:g}x")
    return {
        "validated": True,
        "pass": not reasons,
        "reason": "; ".join(reasons),
        "worse_tol": worse_tol,
        "p99_tol": p99_tol,
        "sessions": v["sessions"],
        "rows": v["rows"],
        "worse": v["worse"],
        "ratio_p99": v["ratio_p99"],
        "sanity_pass": v["sanity_pass"],
    }
