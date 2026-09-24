"""tailM at decode: the per-head files of `scripts/build_tailm.py` applied to retrieval attention.

Each active head keeps its `n_retrieved` retrieved keys as today and gets ONE extra pseudo-key for
every prefill key below them: the stand-in û (linear map of the query) with log mass
ln α + ln Ẑ (truncated Gaussian, boundary = the weakest kept score). Rule: build spec
`docs/superpowers/specs/2026-09-23-tailm-build-design.md` §7; integration:
`docs/superpowers/specs/2026-09-23-tailm-runtime-design.md`. Formulas: `kv_search.tailm` only.

Never imports `kv_search.cache` (cache imports this module).
"""

import math
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch

from kv_search.tailm import (
    F64,
    TailmHead,
    gaussian_tail_mass_finish,
    head_file,
    load_dir,
    merge_lse,
    rel_err,
    standin_finish,
)
from kv_search.tailm_build import scan

SCALING_TOL = 1e-9
EXACT_CHUNK = 16384  # prefill keys per step of the readout's exact pass (the build's --chunk default)
EXACT_ROWS = 512  # query rows per KV head per exact-pass call: bounds its [kv, rows, chunk] score blocks
GRAPH_WARMUP = 3  # eager runs on a side stream before a CUDA graph capture (PyTorch CUDA-graphs docs)


class TailmError(ValueError):
    """A tailM folder that cannot be used with this cache, model or retrieval size."""


@dataclass(frozen=True)
class HeadStatus:
    active: bool
    reason: str | None = None  # why the head is off; None when active
    alpha: float | None = None  # None when there is no file


@dataclass(frozen=True)
class _Layer:
    """One layer's active heads, packed on the device once at load in the runtime's dtype: one tail
    call per layer, one matmul in it."""

    heads: list[int]  # active KV heads, ascending
    index: torch.Tensor  # [h] int64, the same heads on the device
    # [h, d, 2d+1] = [Wᵀ | scaling²·Σ | scaling·μ] side by side (columns): for raw q rows x [h, P, d],
    # x @ packed = [W·q | scaling²·Σq | scaling·q·μ]. scaling is folded in: the files' scaling equals
    # the runtime's (identity check) and apply refuses another one.
    packed: torch.Tensor
    bias: torch.Tensor  # [h, 1, d], against the [h, P, d] product
    scale: torch.Tensor  # [h, 1, 1]
    log_alpha: torch.Tensor  # [h, 1], −inf for α = 0 (no tail)
    n_keys: int  # the same for every head of a cache (identity check: n_keys == context_len)


def _stack_layer(
    files: list[TailmHead], scaling: float, dtype: torch.dtype, device
) -> _Layer:
    n_keys = {int(f.meta["n_keys"]) for f in files}
    if len(n_keys) != 1:
        raise TailmError(
            f"layer {files[0].layer}: heads disagree on n_keys {sorted(n_keys)}"
        )

    def stack(
        name: str,
    ) -> torch.Tensor:  # f64 until the packed matrix is built, then cast once
        return torch.stack([getattr(f, name) for f in files]).to(device, F64)

    def column(values: list[float]) -> torch.Tensor:
        return torch.tensor(values, dtype=F64, device=device)[:, None].to(dtype)

    weight, cov, mean = stack("weight"), stack("cov"), stack("mean")
    packed = torch.cat([weight.mT, scaling**2 * cov, scaling * mean[..., None]], dim=-1)
    return _Layer(
        heads=[f.head for f in files],
        index=torch.tensor([f.head for f in files], dtype=torch.int64, device=device),
        packed=packed.to(dtype).contiguous(),
        bias=stack("bias")[:, None, :].to(dtype),
        scale=column([f.scale for f in files])[:, :, None],
        log_alpha=column(
            [math.log(f.alpha) if f.alpha > 0 else -math.inf for f in files]
        ),
        n_keys=n_keys.pop(),
    )


@dataclass(frozen=True)
class _Graph:
    """One layer's decode tail step captured as a CUDA graph: static inputs, static outputs."""

    graph: "torch.cuda.CUDAGraph"
    inputs: tuple[
        torch.Tensor, ...
    ]  # out, lse, boundary, q: copied into before every replay
    out: torch.Tensor
    lse: torch.Tensor


def _same_scaling(a: float, b: float) -> bool:
    """The one scaling comparison: file vs model at load, call vs runtime in `apply` and `-r native`."""
    return abs(a - b) <= SCALING_TOL


def _identity_problems(
    meta: dict, *, model: str, context_len: int, head_dim: int, scaling: float
) -> list[str]:
    """Why a file does not belong to this cache / model (runtime spec §7.2); empty when it does."""
    bad = []
    if meta.get("model") != model:
        bad.append(f"model {meta.get('model')!r} != {model!r}")
    for key in ("n_keys", "context_len"):
        if int(meta[key]) != context_len:
            bad.append(f"{key} {meta[key]} != prefill context_len {context_len}")
    if int(meta["dim"]) != head_dim:
        bad.append(f"dim {meta['dim']} != head_dim {head_dim}")
    if not _same_scaling(float(meta["scaling"]), scaling):
        bad.append(f"scaling {meta['scaling']} != {scaling}")
    return bad


def _build_command(folder: Path) -> str:
    """The build command for `folder`: its cache is `folder.parent` in the default layout
    (`<cache_dir>/tailm`), otherwise unknown, so a placeholder plus `--out`."""
    if folder.name == "tailm":
        return f".venv/bin/python scripts/build_tailm.py --cache {folder.parent} --validate <sessions-dir>"
    return f".venv/bin/python scripts/build_tailm.py --cache <cache_dir> --out {folder} --validate <sessions-dir>"


def _no_active_message(
    folder: Path,
    status: dict[tuple[int, int], HeadStatus],
    files: dict[tuple[int, int], TailmHead],
    n_retrieved: int,
) -> str:
    counts = Counter(s.reason for s in status.values())
    msg = (
        f"no head in {folder} is active ("
        + ", ".join(f"{n}× {r}" for r, n in counts.most_common())
        + ")"
    )
    if counts.get("not validated"):
        src = next(iter(files.values())).meta.get("source_cache", "<cache>")
        msg += (
            "\n  gate them on recorded sessions: .venv/bin/python scripts/build_tailm.py "
            f"--cache {src} --out {folder} --validate-only --validate <sessions-dir>"
        )
    if any(r is not None and r.startswith("n_retrieved") for r in counts):
        built = sorted({f.n_retrieved for f in files.values()})
        msg += f"\n  the files were built for n_retrieved {built}; this run keeps -n {n_retrieved}"
    return msg


class TailmRuntime:
    """The active heads' maps and moments on the device, plus why every other head is off."""

    def __init__(
        self,
        folder: Path,
        layers: dict[int, _Layer],
        status: dict[tuple[int, int], HeadStatus],
        group: int,
        n_retrieved: int,
        scaling: float,
        dtype: torch.dtype,
        graphs: bool,
    ):
        self.folder, self.group, self.n_retrieved = folder, group, n_retrieved
        self.status = status
        self.scaling, self.dtype, self.graphs = scaling, dtype, graphs
        self._layers = layers
        self._graphs: dict[tuple, _Graph] = (
            {}
        )  # (layer, input shapes / dtypes / device) -> decode graph
        # one warm-up stream and one memory pool for every capture: each new stream costs a cuBLAS
        # workspace, each private pool its own segments. The graphs replay one after another on one
        # stream. Because the pool is shared, a tensor a replay returns is valid only until the next
        # replay of ANY tailM graph (any layer), not just the same layer's.
        self._side: dict[torch.device, torch.cuda.Stream] = {}
        self._pool = None

    @classmethod
    def load(
        cls,
        folder: Path,
        *,
        model: str,
        context_len: int,
        n_retrieved: int,
        head_dim: int,
        scaling: float,
        cells: list[tuple[int, int]],
        group: int,
        device,
        dtype: torch.dtype = torch.float32,
        graphs: bool = True,
    ) -> "TailmRuntime":
        """Read `folder`, refuse files of another cache / model, and pick the active heads by the
        file contract: `gate_pass` and the file's `n_retrieved` equal to the runtime's.

        `dtype`: the tail math (packed matrix, bias, scale, ln α and every op of `apply`); outputs
        keep the partition's dtype. `graphs`: on CUDA, run decode steps (T = 1) as one CUDA graph
        per layer, captured on first use."""
        files = load_dir(folder) if folder.is_dir() else {}
        if not files:
            raise TailmError(
                f"{folder}: no tailM head files; build them with\n  {_build_command(folder)}"
            )
        cellset = set(cells)
        problems = []
        for cell, f in sorted(files.items()):
            bad = _identity_problems(
                f.meta,
                model=model,
                context_len=context_len,
                head_dim=head_dim,
                scaling=scaling,
            )
            if cell not in cellset:
                bad.append("not a full-attention (layer, KV head) of the model")
            if bad:
                problems.append(f"{head_file(folder, *cell).name}: " + "; ".join(bad))
        if problems:
            raise TailmError(
                "files do not belong to this cache / model:\n  " + "\n  ".join(problems)
            )

        status: dict[tuple[int, int], HeadStatus] = {}
        active: dict[int, list[TailmHead]] = {}
        for cell in sorted(cellset):
            f = files.get(cell)
            if f is None:
                status[cell] = HeadStatus(False, "no file")
            elif not f.gate_pass:
                gate = f.meta.get("gate") or {}
                why = (
                    "not validated"
                    if not gate.get("validated")
                    else f"gated off: {gate.get('reason') or 'no reason recorded'}"
                )
                status[cell] = HeadStatus(False, why, f.alpha)
            elif f.n_retrieved != n_retrieved:
                status[cell] = HeadStatus(
                    False, f"n_retrieved {f.n_retrieved} ≠ {n_retrieved}", f.alpha
                )
            else:
                status[cell] = HeadStatus(True, None, f.alpha)
                active.setdefault(cell[0], []).append(
                    f
                )  # cells sorted: heads ascending per layer
        if not active:
            raise TailmError(_no_active_message(folder, status, files, n_retrieved))
        layers = {
            layer: _stack_layer(fs, scaling, dtype, device)
            for layer, fs in active.items()
        }
        if (
            dtype == torch.float32
            and torch.device(device).type == "cuda"
            and torch.backends.cuda.matmul.allow_tf32
        ):
            warnings.warn(
                "tailM: torch.backends.cuda.matmul.allow_tf32 is on; the f32 tail matmul runs in TF32 and "
                "its accuracy drops to TF32 (10-bit mantissa)",
                stacklevel=2,
            )
        return cls(folder, layers, status, group, n_retrieved, scaling, dtype, graphs)

    def active_heads(self, layer: int) -> list[int]:
        """KV heads of `layer` that get the tail, ascending."""
        lay = self._layers.get(layer)
        return lay.heads if lay is not None else []

    def covers(self, layer: int, n_retrieved: int | None) -> bool:
        """Whether the tail applies to `layer` for a retriever keeping `n_retrieved` keys: the layer
        has active heads and the files were built for exactly that retrieval size."""
        return bool(self.active_heads(layer)) and n_retrieved == self.n_retrieved

    def same_scaling(self, scaling: float) -> bool:
        """`scaling` is the one folded into the tail at load (within SCALING_TOL)."""
        return _same_scaling(scaling, self.scaling)

    def native_state(self, layer: int) -> dict | None:
        """This layer's tail for `NativeEdgeRetriever.set_tailm` (the Rust decode tail of `-r native`),
        None without active heads: the packed matrix exactly as `apply` packs it (built in f64 with
        this runtime's `scaling` folded in, cast to f32), rounded to bf16 (RNE) as uint16 bits
        [h, d, 2d+1]; bias [h, d], scale [h], log_alpha [h] (−inf for α = 0) f32; n_keys; the heads.
        numpy on the CPU, C-contiguous."""
        lay = self._layers.get(layer)
        if lay is None:
            return None

        def f32(t: torch.Tensor):
            return t.detach().to("cpu", torch.float32).contiguous().numpy()

        bf16 = lay.packed.to(torch.float32).to(torch.bfloat16)
        return {
            "heads": list(lay.heads),
            "packed": bf16.cpu().contiguous().view(torch.uint16).numpy(),
            "bias": f32(lay.bias[:, 0]),
            "scale": f32(lay.scale[:, 0, 0]),
            "log_alpha": f32(lay.log_alpha[:, 0]),
            "n_keys": lay.n_keys,
        }

    def apply(
        self,
        out: torch.Tensor,
        lse: torch.Tensor,
        boundary: torch.Tensor,
        q: torch.Tensor,
        layer: int,
        scaling: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Merge the tail pseudo-key into the kept-keys partition on this layer's active heads
        (runtime spec §4). `out` [1, q_heads, T, d], `lse` / `boundary` [1, q_heads, T], `q`
        [1, q_heads, T, d]. Rows of inactive heads come back unchanged; a layer without active
        heads returns the inputs themselves.

        With `graphs` on CUDA at T = 1 this replays the layer's CUDA graph and returns the graph's
        own output tensors (no clone: that would add two launches per layer at decode). All graphs
        share one memory pool, so they are valid only until the next replay of ANY tailM graph (the
        next T = 1 call on any layer); use them before that (`attend` consumes them within the
        layer's forward)."""
        lay = self._layers.get(layer)
        if lay is None:
            return out, lse
        if not self.same_scaling(scaling):
            raise TailmError(
                f"apply: scaling {scaling} != {self.scaling}, the one folded into the tail at load"
            )
        if (
            self.graphs
            and q.is_cuda
            and q.shape[2] == 1
            and not torch.cuda.is_current_stream_capturing()
        ):
            return self._replay(layer, lay, out, lse, boundary, q)
        return self._tail(lay, out, lse, boundary, q)

    def _tail(
        self,
        lay: _Layer,
        out: torch.Tensor,
        lse: torch.Tensor,
        boundary: torch.Tensor,
        q: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The eager tail step: gather the active rows, one packed matmul, the `tailm` finish
        helpers, `merge_lse`, scatter back into copies of `out` / `lse`."""
        dt = self.dtype
        out = out.clone(memory_format=torch.contiguous_format)
        lse = lse.clone(memory_format=torch.contiguous_format)
        # KV head h owns q-heads g·h … g·h+g−1: view the rows as [kv_heads, g·T(, d)], pick the active heads
        _, n_q, T, d = q.shape
        kv, gT, idx = n_q // self.group, self.group * T, lay.index
        x = q[0].reshape(kv, gT, d)[idx].to(dt)  # [h, g·T, d], raw q
        # [h, g·T, 2d+1] = [W·q | scaling²·Σq | scaling·q·μ]. In f32 this assumes TF32 matmul is off
        # (the PyTorch default; chat never enables it); `load` warns when it is on.
        r = x @ lay.packed
        wq_hat = r[..., :d] / x.norm(dim=-1, keepdim=True).clamp_min(
            1e-30
        )  # W·q̂ = (W·q) / ‖q‖
        u = standin_finish(wq_hat, lay.bias, lay.scale, dt)
        var = (r[..., d : 2 * d] * x).sum(-1)  # σ_s² = scaling²·qᵀΣq
        mu_s = r[..., 2 * d]  # μ_s = scaling·q·μ
        lz = lay.log_alpha + gaussian_tail_mass_finish(
            mu_s, var, lay.n_keys, boundary[0].reshape(kv, gT)[idx], dt
        )
        out_rows, lse_rows = out[0].view(kv, gT, d), lse[0].view(
            kv, gT
        )  # views: writes land in out / lse
        o, l = merge_lse(out_rows[idx].to(dt), lse_rows[idx].to(dt), u, lz)
        out_rows[idx] = o.to(out.dtype)
        lse_rows[idx] = l.to(lse.dtype)
        return out, lse

    def _replay(
        self,
        layer: int,
        lay: _Layer,
        out: torch.Tensor,
        lse: torch.Tensor,
        boundary: torch.Tensor,
        q: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`_tail` as this layer's CUDA graph: capture on first use, then copy the inputs into its
        static buffers and replay."""
        args = (out, lse, boundary, q)
        key = (layer, q.device, *((a.shape, a.dtype) for a in args))
        g = self._graphs.get(key)
        if g is None:
            g = self._graphs[key] = self._capture(lay, args)
        else:
            for dst, src in zip(g.inputs, args):
                dst.copy_(src)
        g.graph.replay()
        return g.out, g.lse

    def _capture(self, lay: _Layer, args: tuple[torch.Tensor, ...]) -> _Graph:
        """Capture `_tail` on static copies of `args` (which hold this call's inputs), after a
        warm-up on a side stream as the PyTorch CUDA-graphs docs require."""
        # assumes the layer tensors' device (= the inputs' device) is the current CUDA device
        static = tuple(a.clone(memory_format=torch.contiguous_format) for a in args)
        dev = static[0].device
        side = self._side.get(dev)
        if side is None:
            side = self._side[dev] = torch.cuda.Stream(device=dev)
        if self._pool is None:
            self._pool = torch.cuda.graph_pool_handle()
        side.wait_stream(torch.cuda.current_stream(side.device))
        with torch.cuda.stream(side):
            for _ in range(GRAPH_WARMUP):
                self._tail(lay, *static)
        torch.cuda.current_stream(side.device).wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        # torch.cuda.graph.__enter__ runs synchronize + empty_cache before capture: that cost is part
        # of the one-time first decode step (~21.5 ms for 8 captures, measured on niah)
        with torch.cuda.graph(graph, pool=self._pool):
            out, lse = self._tail(lay, *static)
        return _Graph(graph, static, out, lse)

    def describe(self) -> list[str]:
        """The startup message (runtime spec §7.3): a summary line, then one line per layer."""
        on = sum(s.active for s in self.status.values())
        lines = [
            f"tailM on {on}/{len(self.status)} heads (n_retrieved {self.n_retrieved}) from {self.folder}"
        ]
        for layer in sorted({layer for layer, _ in self.status}):
            parts = [
                f"H{h} on α {s.alpha:.2f}" if s.active else f"H{h} off ({s.reason})"
                for (l, h), s in sorted(self.status.items())
                if l == layer
            ]
            lines.append(f"  L{layer:02d}  " + " · ".join(parts))
        return lines


def _ratio(a: float, b: float) -> str:
    return f"{a / b:.2f}×" if b else "n/a"


class TailmCheck:
    """In-chat readout (runtime spec §8): per (layer, active KV head), the whole-output error of
    today's retrieval and of tailM against exact attention over all prefill keys (+ the live keys),
    measured on the same queries."""

    def __init__(
        self, runtime: TailmRuntime, chunk: int = EXACT_CHUNK, rows: int = EXACT_ROWS
    ):
        self.runtime, self.chunk, self.rows = runtime, chunk, rows
        # (layer, head) -> [rows, Σ err today, Σ err tailM, worse rows]; sums stay on the device
        self._acc: dict[tuple[int, int], list] = {}

    def observe(
        self,
        layer: int,
        q: torch.Tensor,
        top_out: torch.Tensor,
        top_lse: torch.Tensor,
        tail_out: torch.Tensor,
        tail_lse: torch.Tensor,
        live_out: torch.Tensor,
        live_lse: torch.Tensor,
        prefill_keys: torch.Tensor,
        prefill_values: torch.Tensor,
        scaling: float,
    ) -> None:
        """One forward of one layer. Partitions [1, q_heads, T(, d)] as `attend` holds them before
        the live merge; `prefill_keys` / `prefill_values` [1, kv_heads, N, d] bf16."""
        heads = self.runtime.active_heads(layer)
        if not heads:
            return
        g = self.runtime.group
        kv, T = prefill_keys.shape[1], q.shape[2]
        Q = (
            q[0].reshape(kv, g * T, -1).float()
        )  # [kv, g·T, d]: q-heads g·h … g·h+g−1 of KV head h
        m, D, N = self._exact(Q, prefill_keys[0], prefill_values[0], scaling)
        ex_out = (N / D[..., None]).reshape(kv * g, T, -1)  # f64
        ex_lse = (m.to(F64) + D.log()).reshape(kv * g, T)
        lo, ll = live_out[0].to(F64), live_lse[0].to(F64)
        exact, _ = merge_lse(ex_out, ex_lse, lo, ll)
        today, _ = merge_lse(top_out[0].to(F64), top_lse[0].to(F64), lo, ll)
        tail, _ = merge_lse(tail_out[0].to(F64), tail_lse[0].to(F64), lo, ll)
        e_today, e_tail = rel_err(today, exact), rel_err(tail, exact)  # [q_heads, T]
        for h in heads:
            rows = slice(h * g, (h + 1) * g)
            a = self._acc.setdefault((layer, h), [0, 0.0, 0.0, 0])
            a[0] += g * T
            a[1] = a[1] + e_today[rows].sum()
            a[2] = a[2] + e_tail[rows].sum()
            a[3] = a[3] + (e_tail[rows] > e_today[rows]).sum()

    def _exact(
        self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, scaling: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Softmax totals (m, D, N) of `Q` [kv, P, d] over all prefill keys, `self.rows` query rows
        at a time (every row's arithmetic is independent of the others)."""
        parts = [
            scan(Q[:, i : i + self.rows], K, V, scaling, 1, self.chunk, moments=False)
            for i in range(0, Q.shape[1], self.rows)
        ]
        if len(parts) == 1:
            return parts[0].m, parts[0].D, parts[0].N
        return tuple(
            torch.cat([getattr(r, k) for r in parts], 1) for k in ("m", "D", "N")
        )

    def totals(self) -> dict[tuple[int, int], tuple[int, float, float, int]]:
        """(rows, Σ err today, Σ err tailM, rows where tailM is worse) per (layer, KV head)."""
        return {
            c: (a[0], float(a[1]), float(a[2]), int(a[3]))
            for c, a in sorted(self._acc.items())
        }

    def reset(self) -> None:
        self._acc.clear()

    def report(self) -> list[str]:
        """The readout table (runtime spec §8) for everything observed since the last reset()."""
        acc = self.totals()
        if not acc:
            return ["tailM check: no rows (the tail step did not run)"]

        def cols(cells: list[tuple[int, int]]) -> str:
            rows = sum(acc[c][0] for c in cells)
            today = sum(acc[c][1] for c in cells) / rows
            tail = sum(acc[c][2] for c in cells) / rows
            worse = sum(acc[c][3] for c in cells) / rows
            return f"{today:.3f} → {tail:.3f} ({_ratio(tail, today)})  {worse:6.1%}"

        lines = [
            f"tailM check ({max(a[0] for a in acc.values())} rows/head)   "
            "today → tailM          worse   worst head"
        ]
        for layer in sorted({layer for layer, _ in acc}):
            cells = [c for c in acc if c[0] == layer]
            n_kv = sum(1 for c in self.runtime.status if c[0] == layer)
            worst = max(
                cells, key=lambda c: acc[c][2] / acc[c][1] if acc[c][1] else math.inf
            )
            lines.append(
                f"  L{layer:02d}  "
                + f"on {len(cells)}/{n_kv}".ljust(22)
                + f"{cols(cells)}   "
                f"L{layer:02d}H{worst[1]} {_ratio(acc[worst][2], acc[worst][1])}"
            )
        lines.append(
            "  all  "
            + f"on {len(acc)}/{len(self.runtime.status)}".ljust(22)
            + cols(list(acc))
        )
        return lines
