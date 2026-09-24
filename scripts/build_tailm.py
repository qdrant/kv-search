#!/usr/bin/env python3
"""Build tailM files from a prefill cache: per (layer, KV head) the linear tail map (fitted at a depth
chosen from an error target ε), the key moments and α, for a runtime that keeps `n_retrieved` keys;
each head is gated on recorded decode sessions against those kept keys alone (spec revision 2).

  .venv/bin/python scripts/build_tailm.py --cache cache/qdrant/1M/qwen3_5 --validate ../kv-search/replay-data
  .venv/bin/python scripts/build_tailm.py --cache cache/qdrant/1M/qwen3_5 --cells L15H3 --eps 0.08 --cut-shift 10%
  .venv/bin/python scripts/build_tailm.py --cache cache/qdrant/1M/qwen3_5 --validate-only \\
      --validate ../kv-search/replay-data --cells L15H3
  .venv/bin/python scripts/build_tailm.py --cache cache/qdrant/1M/qwen3_5 --summary-only

Without --validate every head is written gated off; --validate-only gates the stored files later.
Negative percentage shifts need an equals sign: --cut-shift=-10%.
Design: docs/superpowers/specs/2026-09-23-tailm-build-design.md.
"""

import argparse
import datetime
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import NoReturn

import torch
from rich import box
from rich.console import Console
from rich.markup import escape
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from kv_search.tailm import (
    TailmHead,
    export_replay,
    head_file,
    load_dir,
    load_head,
    rope_inv_freq,
    save_head,
)
from kv_search.tailm_build import (
    BuildConfig,
    LayerReader,
    build_head,
    eval_cuts,
    layer_file,
    make_ladder,
    moments,
    parse_cells,
    parse_cut_shift,
    plan_batches,
    sample_queries,
    scan,
    resolve_context_tokens,
    split_indices,
    tier_label,
)
from kv_search.tailm_validate import (
    SANITY_MEDIAN_TOL,
    SANITY_P99_TOL,
    check_sessions,
    gate,
    open_sessions,
    validate_batch,
)

Cell = tuple[int, int]


def cell_name(cell: Cell) -> str:
    return f"L{cell[0]:02d}H{cell[1]}"


def die(msg: str) -> NoReturn:
    raise SystemExit(f"error: {msg}")


def ints(text: str, what: str) -> list[int]:
    try:
        return [int(x) for x in text.split(",") if x.strip()]
    except ValueError:
        die(f"{what}: expected comma-separated integers, got {text!r}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--cache", required=True, type=Path, help="prefill cache directory")
    p.add_argument("--out", type=Path, help="output folder (default <cache>/tailm)")
    p.add_argument("--model", default="Qwen/Qwen3.5-9B", help="HF config source")
    p.add_argument(
        "--context-tokens",
        type=int,
        help="YaRN factor source (default: the qdrant tier label, else 0, as prefill)",
    )
    p.add_argument(
        "--layers", help="layers, e.g. 15,31 (default: all full-attention layers)"
    )
    p.add_argument("--heads", help="KV heads, e.g. 0,3 (default: all)")
    p.add_argument(
        "--cells", help="explicit cells, e.g. L15H3,L31H0 (overrides --layers/--heads)"
    )
    p.add_argument(
        "--samples", type=int, default=4096, help="prefill query positions per head"
    )
    p.add_argument(
        "--heldout",
        type=int,
        default=1024,
        help="held out for all evaluation; the map is fitted on the rest",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=3,
        help="query sampling seed (3 = the research recipe)",
    )
    p.add_argument(
        "--ridge",
        type=float,
        default=1e-2,
        help="ridge as a fraction of mean(diag(XᵀX))",
    )
    p.add_argument(
        "--cuts",
        help="candidate cut ladder, e.g. 8,16,32 (overrides --min-cut/--max-cut)",
    )
    p.add_argument("--min-cut", type=int, default=8)
    p.add_argument("--max-cut", type=int, default=4096)
    p.add_argument(
        "--eps", type=float, default=0.10, help="max allowed tailM output error"
    )
    p.add_argument("--eps-stat", choices=["mean", "p90"], default="mean")
    p.add_argument(
        "--cut-shift",
        default="0",
        help="P%% or integer k added to the detected cut (negative: --cut-shift=-10%%)",
    )
    p.add_argument(
        "--n-retrieved",
        type=int,
        default=128,
        help="keys runtime keeps per head (alpha, numbers and gate use it)",
    )
    p.add_argument(
        "--gate-worse",
        type=float,
        default=0.03,
        help="gate: share of decode rows worse than the kept keys alone <= this",
    )
    p.add_argument(
        "--gate-p99",
        type=float,
        default=3.0,
        help="gate: p99 of err(tailM)/err(kept only) over those rows <= this",
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="max heads resident on the device at once",
    )
    p.add_argument(
        "--chunk",
        type=int,
        default=16384,
        help="keys per scan step (bounds working memory)",
    )
    p.add_argument(
        "--validate",
        type=Path,
        help="recorded decode sessions: validation and the gate (without: all heads off)",
    )
    p.add_argument("--validate-sessions", help="session subset, e.g. 01,02")
    p.add_argument(
        "--validate-only",
        action="store_true",
        help="skip building; validate and re-gate the files in --out",
    )
    p.add_argument(
        "--summary-only",
        action="store_true",
        help="print the summary of the files in --out; no compute",
    )
    p.add_argument("--json", type=Path, help="also write the summary as JSON")
    p.add_argument(
        "--export-replay", type=Path, help="also write kv-replay's layout here"
    )
    return p.parse_args(argv)


class Reporter:
    """Progress: rich bars in a terminal; time-stamped plain lines (at most one per `every` s) when piped."""

    def __init__(self, con: Console, tty: bool, total: int, every: float = 5.0):
        self.con, self.tty, self.total, self.every = con, tty, total, every
        self.t0 = time.monotonic()
        self.done = 0
        self.label = ""
        self._last = -math.inf
        self.bar = None
        if tty:
            self.bar = Progress(
                TextColumn("{task.description}"),
                BarColumn(),
                TextColumn("{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=con,
            )
            self._heads = self.bar.add_task("heads", total=total)
            self._keys = self.bar.add_task("", total=1)
            self.bar.start()

    def _stamp(self) -> str:
        s = int(time.monotonic() - self.t0)
        return f"[{s // 60:02d}:{s % 60:02d}]"

    def log(self, msg: str) -> None:
        self.con.print(escape(f"{self._stamp()} {msg}"))

    def batch(self, i: int, n: int, cells: str) -> None:
        """Start batch i of n. The ETA comes from the finished batches: a batch's heads all finish
        together after its scan, so a per-head rate would be meaningless."""
        self.label = f"batch {i}/{n} {cells}"
        self._last = -math.inf
        eta = ""
        if i > 1:
            left = int((time.monotonic() - self.t0) / (i - 1) * (n - i + 1))
            eta = f" (eta {left // 60:02d}:{left % 60:02d})"
        if self.bar is not None:
            self.bar.update(self._keys, description=self.label, completed=0, total=1)
        else:
            self.log(self.label + eta)

    def scan(self, done: int, total: int, what: str = "scan") -> None:
        if self.bar is not None:
            self.bar.update(
                self._keys,
                description=f"{self.label} {what}",
                completed=done,
                total=total,
            )
            return
        now = time.monotonic()
        if now - self._last >= self.every or done == total:
            self._last = now
            self.log(f"{self.label} {what} {100 * done / total:.0f}%")

    def head_done(self, msg: str) -> None:
        self.done += 1
        if self.bar is not None:
            self.bar.advance(self._heads)
        self.log(f"head {self.done}/{self.total} {msg}")

    def warn(self, msg: str) -> None:
        self.con.print(f"[yellow]{escape(self._stamp())} warning: {escape(msg)}[/]")

    def close(self) -> None:
        if self.bar is not None:
            self.bar.stop()


def _flags(m: dict) -> tuple[list[str], str | None]:
    flags, style = [], None
    if not m["eps_met"]:
        flags.append("! eps not met")
        style = "yellow"
    g, v = m["gate"], m.get("decode")
    if not g["validated"]:
        flags.append("? not validated")
        style = style or "yellow"
    elif v is not None and not v["sanity_pass"]:
        flags.append("✗ sanity")
        style = "red"
    elif not m["gate_pass"]:
        flags.append(f"✗ gated off ({g['reason']})")
        style = "red"
    return flags, style


def print_summary(
    con: Console,
    heads: dict[Cell, TailmHead],
    peaks: list[tuple[str, float]],
    *,
    cache: Path,
    out: Path,
    validate_dir: Path | None,
) -> None:
    """The end-of-run summary (spec §9): one row per head, per-layer totals, flags, next steps."""
    t = Table(
        box=box.SIMPLE, title=escape(f"tailM heads in {out}"), title_justify="left"
    )
    cols = [
        "cell",
        "ε cut",
        "fit",
        "α",
        "prefill err/none",
        "decode err mean/p90",
        "decode none",
        "worse",
        "p99 ratio",
        "sanity med/p99/max",
        "gate",
        "flags",
    ]
    for c in cols:
        t.add_column(c, justify="left" if c in ("cell", "flags") else "right")
    for cell in sorted(heads):
        m = heads[cell].meta
        v, pre = m.get("decode"), m["prefill"]
        flags, style = _flags(m)
        det = f"{m['cut_detected']}{'' if m['eps_met'] else '!'}"
        eps_cut = det if m["cut"] == m["cut_detected"] else f"{det} → {m['cut']}"
        dec = (
            [
                f"{v['dec_err_mean']:.3f}/{v['dec_err_p90']:.3f}",
                f"{v['dec_none_mean']:.3f}",
                f"{100 * v['worse']:.1f}%",
                f"{v['ratio_p99']:.2f}x",
                f"{v['sanity_median']:.1e}/{v['sanity_p99']:.1e}/{v['sanity_max']:.1e}",
            ]
            if v
            else ["-"] * 5
        )
        t.add_row(
            cell_name(cell),
            eps_cut,
            str(m["fit_cut"]),
            f"{m['alpha']:.2f}",
            f"{pre['err_mean']:.3f}/{pre['none_mean']:.3f}",
            *dec,
            "✓" if m["gate_pass"] else "✗",
            ", ".join(flags),
            style=style,
        )
    con.print(t)

    lt = Table(box=box.SIMPLE, title="per layer", title_justify="left")
    for c in ("layer", "heads", "on", "decode err (on)", "decode none (on)"):
        lt.add_column(c, justify="right")
    for layer in sorted({c[0] for c in heads}):
        ms = [heads[c].meta for c in sorted(heads) if c[0] == layer]
        on = [m for m in ms if m["gate_pass"]]
        err = (
            f"{sum(m['decode']['dec_err_mean'] for m in on) / len(on):.3f}"
            if on
            else "-"
        )
        none = (
            f"{sum(m['decode']['dec_none_mean'] for m in on) / len(on):.3f}"
            if on
            else "-"
        )
        lt.add_row(f"L{layer:02d}", str(len(ms)), str(len(on)), err, none)
    con.print(lt)
    keep = sorted({heads[c].n_retrieved for c in heads})
    off = [cell_name(c) for c in sorted(heads) if not heads[c].gate_pass]
    con.print(
        escape(
            f"runtime keeps n_retrieved {', '.join(map(str, keep))} keys per head; heads on: {len(heads) - len(off)}/{len(heads)}"
            + (f"; off: {', '.join(off)}" if off else "")
        )
    )
    for label, gib in peaks:
        con.print(escape(f"peak device memory, batch {label}: {gib:.1f} GiB"))

    bad = [
        cell_name(c)
        for c in sorted(heads)
        if heads[c].meta.get("decode") and not heads[c].meta["decode"]["sanity_pass"]
    ]
    if bad:
        con.print(
            f"[red]validation sanity failed on {', '.join(bad)}: the recomputed exact attention differs from the "
            f"recorded attn_out with a median above {SANITY_MEDIAN_TOL:g} or a p99 above {SANITY_P99_TOL:g} "
            f"(a RoPE / scaling / loading bug or a misaligned session); these heads are off and their decode "
            f"numbers are not trustworthy.[/]"
        )
    todo = [c for c in sorted(heads) if not heads[c].meta["gate"]["validated"]]
    if todo:
        sess = str(validate_dir) if validate_dir else "<sessions-dir>"
        con.print(
            "[yellow]note:[/] not validated heads stay off: only recorded decode queries can gate them (the map "
            "fits decode queries worse than prefill ones, FINDINGS §13.7). Activate them with:"
        )
        con.print(
            escape(
                f"  .venv/bin/python scripts/build_tailm.py --cache {cache} --out {out} --validate-only "
                f"--validate {sess} --cells {','.join(map(cell_name, todo))}"
            ),
            soft_wrap=True,
        )


def export_heads(con: Console, heads: dict[Cell, TailmHead], root: Path) -> None:
    """kv-replay's layout for every gated-on head (spec §7.1), in every mode."""
    on = [c for c in sorted(heads) if heads[c].gate_pass]
    for c in on:
        export_replay(heads[c], root)
    con.print(
        escape(
            f"kv-replay layout for {len(on)} gated-on heads in {root}; run kv-replay with --n = n_retrieved "
            f"({', '.join(sorted({str(heads[c].n_retrieved) for c in on})) or '-'}); per head fit cut (excluded_top_m) "
            f"and alpha: "
            + ", ".join(
                f"{cell_name(c)}={heads[c].fit_cut}/{heads[c].alpha:g}" for c in on
            )
        )
    )


def write_json(path: Path, heads: dict, peaks: list, cache: Path, out: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "cache": str(cache),
                "out": str(out),
                "heads": {cell_name(c): heads[c].meta for c in sorted(heads)},
                "peak_gib": dict(peaks),
            },
            indent=1,
        )
    )


def with_gate(meta: dict, v: dict, worse_tol: float, p99_tol: float) -> dict:
    """`meta` with the decode numbers and the gate decision of spec §6.7."""
    g = gate(v, worse_tol=worse_tol, p99_tol=p99_tol)
    return {**meta, "decode": v, "gate": g, "gate_pass": g["pass"]}


def load_kv(
    cache: Path, batch: list[Cell], dev: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """K and V [H, n_keys, d] bf16 on `dev` for the batch's heads; each layer file is opened once."""
    readers: dict[int, LayerReader] = {}
    K = V = None
    for i, (layer, head) in enumerate(batch):
        r = readers.get(layer)
        if r is None:
            r = readers[layer] = LayerReader(layer_file(cache, layer))
        if K is None:
            K = torch.empty(
                len(batch), r.n_keys, r.dim, dtype=torch.bfloat16, device=dev
            )
            V = torch.empty_like(K)
        elif (r.n_keys, r.dim) != tuple(K.shape[1:]):
            die(
                f"{r.path}: {r.n_keys} keys x {r.dim} differs from the batch's {tuple(K.shape[1:])}"
            )
        K[i].copy_(r.head("keys", head))
        V[i].copy_(r.head("values", head))
    return K, V


def main(argv: list[str] | None = None) -> int:
    a = parse_args(argv)
    tty = sys.stdout.isatty()
    con = Console(highlight=False, width=None if tty else 200)
    cache: Path = a.cache
    out: Path = a.out or cache / "tailm"

    if a.summary_only:
        heads = load_dir(out) if out.is_dir() else {}
        if not heads:
            die(f"no tailM files in {out}")
        print_summary(con, heads, [], cache=cache, out=out, validate_dir=a.validate)
        if a.export_replay:
            export_heads(con, heads, a.export_replay)
        if a.json:
            write_json(a.json, heads, [], cache, out)
        return 0
    if a.validate_only and not a.validate:
        die("--validate-only needs --validate DIR (the recorded decode sessions)")
    if not (cache / "meta.json").exists():
        die(f"{cache}: no meta.json, not a prefill cache")
    context_len = int(json.loads((cache / "meta.json").read_text())["context_len"])
    if a.batch_size < 1 or a.chunk < 1:
        die("--batch-size and --chunk must be >= 1")
    # parameter checks first: they need no model config, so bad input fails before anything is loaded
    cfg = None
    kmax = 0
    if not a.validate_only:
        if not 1 <= a.heldout < a.samples:
            die(
                f"--heldout must be at least 1 and below --samples (got --heldout {a.heldout}, --samples {a.samples})"
            )
        if a.samples > context_len:
            die(f"--samples {a.samples} exceeds the {context_len:,} prompt positions")
        if a.eps <= 0:
            die(f"--eps must be > 0, got {a.eps}")
        try:
            shift = parse_cut_shift(a.cut_shift)
            ladder = make_ladder(
                a.min_cut, a.max_cut, ints(a.cuts, "--cuts") if a.cuts else None
            )
        except ValueError as e:
            die(str(e))
        if ladder[-1] >= context_len:
            die(f"ladder cut {ladder[-1]} must be < the {context_len:,} keys")
        if not 1 <= a.n_retrieved < context_len:
            die(
                f"--n-retrieved must be at least 1 and below the {context_len:,} keys, got {a.n_retrieved}"
            )
        cfg = BuildConfig(
            samples=a.samples,
            heldout=a.heldout,
            seed=a.seed,
            ridge=a.ridge,
            ladder=tuple(ladder),
            eps=a.eps,
            eps_stat=a.eps_stat,
            cut_shift=shift,
            n_retrieved=a.n_retrieved,
        )
        kmax = max(eval_cuts(ladder, shift, context_len, a.n_retrieved))
    try:
        ctx_tokens = resolve_context_tokens(cache, context_len, a.context_tokens)
    except ValueError as e:
        die(str(e))

    from kv_search.main import (
        load_model_config,
    )  # transformers + qdrant imports: only needed from here on

    config = load_model_config(a.model, ctx_tokens)
    text = getattr(config, "text_config", config)
    rp = getattr(text, "rope_parameters", None) or {}
    yarn = rp.get("factor", 1.0) if rp.get("rope_type") == "yarn" else 1.0
    full = [i for i, t in enumerate(text.layer_types) if t == "full_attention"]
    kv_heads = text.num_key_value_heads
    group = text.num_attention_heads // kv_heads
    scaling = text.head_dim**-0.5
    con.print(
        escape(
            f"cache {cache}: context_len {context_len:,}, context_tokens {ctx_tokens:,} (YaRN factor {yarn:g}), "
            f"{len(full)} full-attention layers x {kv_heads} KV heads, scaling {scaling:g}"
        )
    )

    valid = [(L, h) for L in full for h in range(kv_heads)]
    try:
        if a.cells:
            cells = parse_cells(a.cells)
        else:
            layers = ints(a.layers, "--layers") if a.layers else full
            hs = ints(a.heads, "--heads") if a.heads else list(range(kv_heads))
            cells = [(L, h) for L in layers for h in hs]
    except ValueError as e:
        die(str(e))
    bad = [c for c in cells if c not in valid]
    if bad:
        die(
            f"not full-attention (layer, KV head) cells: {', '.join(map(cell_name, bad))}; "
            f"valid: {', '.join(map(cell_name, valid))}"
        )

    stored: dict[Cell, TailmHead] = {}
    if a.validate_only:
        stored = load_dir(out) if out.is_dir() else {}
        if not (a.cells or a.layers or a.heads):
            cells = sorted(stored)
        missing = [c for c in cells if c not in stored]
        if missing or not cells:
            die(
                f"no tailM files in {out} for {', '.join(map(cell_name, missing)) or 'any cell'}"
            )
    else:
        have = sorted(
            int(p.name[len("queries_") : -len(".safetensors")])
            for p in cache.glob("queries_*.safetensors")
        )
        no_q = sorted({L for L, _ in cells} - set(have))
        if no_q:
            die(
                f"no recorded prefill queries for layers {no_q} in {cache} (have {have}); "
                f"re-run `prefill`, which writes queries_LL.safetensors"
            )
    for layer in sorted({L for L, _ in cells}):
        try:
            layer_file(cache, layer)
        except FileNotFoundError as e:
            die(str(e))
    if a.json and not a.json.parent.is_dir():
        die(f"--json {a.json}: directory {a.json.parent} does not exist")

    dev = torch.device(a.device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        die("--device cuda but no CUDA device is available")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    sessions = None
    if a.validate:
        try:
            sessions = open_sessions(
                a.validate,
                a.validate_sessions.split(",") if a.validate_sessions else None,
            )
            check_sessions(
                sessions,
                context_len=context_len,
                model=a.model,
                scaling=scaling,
                tier_size=tier_label(cache),
                layers=sorted({L for L, _ in cells}),
            )
        except ValueError as e:
            die(str(e))

    inv = rope_inv_freq(config)
    identity = {
        "context_len": context_len,
        "model": a.model,
        "source_cache": os.path.abspath(cache),
        "built_at": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
    }
    batches = plan_batches(cells, a.batch_size)
    prog = Reporter(con, tty, len(cells))
    heads: dict[Cell, TailmHead] = {}
    peaks: list[tuple[str, float]] = []
    if not sessions:
        prog.warn(
            "no --validate sessions: every head is written gated off until --validate-only gates it (spec §6.7)"
        )
    try:
        for bi, batch in enumerate(batches, 1):
            label = f"{bi}/{len(batches)} " + ",".join(map(cell_name, batch))
            prog.batch(bi, len(batches), ",".join(map(cell_name, batch)))
            if dev.type == "cuda":
                torch.cuda.reset_peak_memory_stats(dev)
            K = V = None
            try:
                K, V = load_kv(cache, batch, dev)
                n_keys = K.shape[1]
                if n_keys != context_len:
                    die(
                        f"layer files hold {n_keys:,} keys but meta.json says context_len {context_len:,}"
                    )
                if not a.validate_only:
                    Q = torch.stack(
                        [
                            torch.from_numpy(
                                sample_queries(
                                    cache,
                                    L,
                                    h,
                                    context_len,
                                    a.samples,
                                    a.seed,
                                    inv,
                                    group,
                                )
                            )
                            for L, h in batch
                        ]
                    ).to(dev)
                    res = scan(Q, K, V, scaling, kmax, a.chunk, on_chunk=prog.scan)
                    te, tr = (
                        torch.from_numpy(x).to(dev)
                        for x in split_indices(a.samples, a.heldout)
                    )
                    for i, cell in enumerate(batch):
                        mu, sigma = moments(res, i)
                        hb = build_head(
                            *cell,
                            Q[i],
                            res.head(i),
                            V[i],
                            mu,
                            sigma,
                            n_keys,
                            scaling,
                            tr,
                            te,
                            cfg,
                            identity,
                        )
                        # written gated off first: an interrupted validation never leaves a head on
                        save_head(head_file(out, *cell), hb.tensors, hb.meta)
                        heads[cell] = load_head(head_file(out, *cell))
                        if hb.warning:
                            prog.warn(hb.warning)
                        m, pre = hb.meta, hb.meta["prefill"]
                        prog.head_done(
                            f"{cell_name(cell)} eps cut {m['cut_detected']}->{m['cut']} fit {m['fit_cut']} alpha "
                            f"{m['alpha']:g} prefill err {pre['err_mean']:.3f} (kept only {pre['none_mean']:.3f})"
                        )
                    del Q, res
                else:
                    heads.update({c: stored[c] for c in batch})
                if sessions:
                    heads_b = {c: heads[c] for c in batch}
                    for c, hd in heads_b.items():
                        if int(hd.meta["n_keys"]) != n_keys:
                            die(
                                f"{cell_name(c)}: file built from a {hd.meta['n_keys']:,}-key cache, this one has {n_keys:,}"
                            )
                    res_v = validate_batch(
                        batch,
                        heads_b,
                        K,
                        V,
                        sessions,
                        group=group,
                        scaling=scaling,
                        chunk=a.chunk,
                        block=a.samples,
                        on_chunk=lambda d, t: prog.scan(d, t, "validate"),
                    )
                    for c in batch:
                        hd = heads_b[c]
                        meta = with_gate(hd.meta, res_v[c], a.gate_worse, a.gate_p99)
                        save_head(
                            head_file(out, *c),
                            {
                                k: getattr(hd, k)
                                for k in ("weight", "bias", "mean", "cov")
                            },
                            meta,
                        )
                        heads[c] = load_head(head_file(out, *c))
                        v, g = res_v[c], meta["gate"]
                        msg = (
                            f"{cell_name(c)} decode err {v['dec_err_mean']:.3f} (kept only {v['dec_none_mean']:.3f}) "
                            f"worse {100 * v['worse']:.1f}% p99 {v['ratio_p99']:.2f}x sanity median "
                            f"{v['sanity_median']:.1e} gate {'on' if g['pass'] else 'OFF (' + g['reason'] + ')'}"
                        )
                        if a.validate_only:
                            prog.head_done(msg)
                        else:
                            prog.log(msg)
            except torch.OutOfMemoryError:
                peak = (
                    torch.cuda.max_memory_allocated(dev) / 2**30
                    if dev.type == "cuda"
                    else float("nan")
                )
                die(
                    f"out of device memory in batch {label} (peak {peak:.1f} GiB); try a smaller "
                    f"--batch-size (now {a.batch_size}) or --chunk (now {a.chunk})"
                )
            finally:
                del K, V
                if dev.type == "cuda":
                    peaks.append((label, torch.cuda.max_memory_allocated(dev) / 2**30))
                    torch.cuda.empty_cache()
    finally:
        prog.close()

    print_summary(con, heads, peaks, cache=cache, out=out, validate_dir=a.validate)
    if a.export_replay:
        export_heads(con, heads, a.export_replay)
    if a.json:
        write_json(a.json, heads, peaks, cache, out)
    return (
        0
        if all(
            h.meta["decode"] is None or h.meta["decode"]["sanity_pass"]
            for h in heads.values()
        )
        else 1
    )


if __name__ == "__main__":
    sys.exit(main())
