# tailM build tool — usage

`scripts/build_tailm.py` builds the **tailM tail correction** for retrieval attention, one file per
(layer, KV head), from a prefill cache. This page covers what you need to run it and read its
output. The full design and the evidence behind every default are in
[`superpowers/specs/2026-09-23-tailm-build-design.md`](superpowers/specs/2026-09-23-tailm-build-design.md).

## What tailM does

Retrieval attention keeps a head's top `n_retrieved` keys (default 128) and drops the rest. tailM
puts the dropped keys back as **one pseudo-key**:

    out = (N_top + α·Ẑ·û) / (D_top + α·Ẑ)

- `N_top`, `D_top`: exact attention over the kept keys (as today).
- `Ẑ`: the softmax mass of every key scoring below the weakest kept key. It is estimated from a
  Gaussian over the head's keys (mean μ, covariance Σ), truncated at that score.
- `û`: the tail's average value vector, predicted from the query by a linear map fitted offline.
- `α`: a mass multiplier fitted offline (usually 0.4–1.05).

A file is switched on for a head only if it passes a **gate** on recorded decode sessions: tailM must
rarely be worse than the kept keys alone, and never by much. A head that fails, or has no file,
runs exactly today's path. So a correctly built set of files can only match or improve today's
output, head by head.

## Quick start

```bash
cd kv-search-clean

# build + gate every head (recommended: the gate needs recorded decode sessions)
.venv/bin/python scripts/build_tailm.py --cache cache/qdrant/1M/qwen3_5 \
  --validate ../kv-search/replay-data

# show what is on disk
.venv/bin/python scripts/build_tailm.py --cache cache/qdrant/1M/qwen3_5 --summary-only
```

On an RTX 3090 (24 GiB) the default 32-head build with `--validate` takes about 2.5 min and peaks at
12.8 GiB per batch. Use `.venv/bin/python`, not `uv run`: a sync may rebuild the Rust extension.

## Modes

| mode | command | what it does |
|---|---|---|
| build + gate | `--validate DIR` | builds the selected heads, validates them on the sessions in `DIR` and writes files with the gate decided |
| build only | (no `--validate`) | builds and writes every head **gated off** (`? not validated`). Nothing gets switched on. Prints the `--validate-only` command that activates the heads |
| re-gate | `--validate-only --validate DIR` | loads the existing files in `--out`, re-runs the decode check and rewrites only the gate fields. Map, moments and α stay byte-identical. Use it to activate a build-only run or to change the gate thresholds. Needs the same GPU memory as a build (about 13 GiB at batch size 8) |
| summary | `--summary-only` | prints the summary of the files in `--out`; no compute |

`--json PATH` and `--export-replay DIR` work in every mode.

## All options

### Input, output, selection

| option | default | meaning |
|---|---|---|
| `--cache DIR` | required | prefill cache: `meta.json`, `layer_NN.safetensors[.zst]`, `queries_NN.safetensors` (written by `prefill`) |
| `--out DIR` | `<cache>/tailm` | output folder. `kv-search-clean/cache` links to `../kv-search/cache`, so the default lands in the research checkout |
| `--model NAME` | `Qwen/Qwen3.5-9B` | Hugging Face config source (full-attention layers, head dim, KV heads, RoPE) |
| `--context-tokens N` | from the path | decides the YaRN factor and must match what `prefill` used. Default: the tier label of `cache/qdrant/<tier>/…` (`1M` → 1,000,000), otherwise 0 (no YaRN, e.g. niah). A cache longer than 262,144 tokens at a path without a tier label is **refused**; pass it explicitly |
| `--layers L,L,…` | all full-attention layers | e.g. `15,31` (Qwen3.5-9B: 3, 7, …, 31) |
| `--heads H,H,…` | all KV heads | e.g. `0,3` (Qwen3.5-9B: 0–3) |
| `--cells L15H3,L31H0` | — | explicit cells; overrides `--layers` / `--heads` |

### How the map is built

| option | default | meaning |
|---|---|---|
| `--samples N` | 4096 | prefill query positions sampled per head, re-positioned to the end of the context |
| `--heldout N` | 1024 | of those, held out for all prefill numbers; the map and α are fitted on the rest |
| `--seed N` | 3 | query sampling seed (3 = the research recipe, reproduces its numbers) |
| `--ridge F` | 0.01 | ridge strength as a fraction of `mean(diag(XᵀX))` |
| `--cuts a,b,…` | — | the candidate depth ladder; overrides the two below |
| `--min-cut N` / `--max-cut N` | 8 / 4096 | powers-of-two ladder bounds |
| `--eps F` | 0.10 | error target that picks the **fit depth**: the smallest ladder depth where keeping that many keys plus tailM has held-out error ≤ ε |
| `--eps-stat mean\|p90` | `mean` | which statistic ε applies to |
| `--cut-shift X` | 0 | move the picked depth: `10%` adds 10 %, `2000` adds 2000, negative allowed (`--cut-shift=-10%`, note the `=`) |
| `--n-retrieved N` | 128 | **keys runtime keeps per head.** α, the prefill numbers and the gate are computed for exactly this, and runtime ignores a file whose value differs from its own |

The map is fitted at `fit_cut = max(ε depth, n_retrieved)`, i.e. it learns the direction of the tail
below that rank. At runtime nothing is skipped between `n_retrieved` and `fit_cut`: the Gaussian mass
covers every key below the kept ones.

### Gate

| option | default | meaning |
|---|---|---|
| `--validate DIR` | off | recorded decode sessions (`*.safetensors` from the research `record` command). Needed for the gate |
| `--validate-sessions 00,01` | all in `DIR` | session subset, by the trailing number of the file name |
| `--gate-worse F` | 0.03 | max share of decode rows where tailM is worse than the kept keys alone |
| `--gate-p99 F` | 3.0 | max p99, over those worse rows, of `err(tailM) / err(kept keys only)` |

A head is on when **sanity passes and worse ≤ `--gate-worse` and p99 ratio ≤ `--gate-p99`**.
Sessions must come from the same cache: their `context_len`, `model`, `scaling`, tier and layers are
checked, and a mismatch aborts.

### Compute

| option | default | meaning |
|---|---|---|
| `--device cuda\|cpu` | cuda if available | CPU works too (2 heads with a short ladder: 58 s on CPU vs 11 s on the GPU, measured) |
| `--batch-size N` | 8 | heads resident on the device at once (8 = two whole layers, about 1 GB of K/V per head at 1M) |
| `--chunk N` | 16384 | keys per scan step. Bounds working memory; lower it (or `--batch-size`) after an out-of-memory error |

### Extra output

| option | meaning |
|---|---|
| `--json PATH` | the summary as JSON: every head's full metadata plus the peak memory per batch |
| `--export-replay DIR` | the research kv-replay layout (`tailvecs_linear/`, `moments/`) for the gated-on heads, for KL runs. Run kv-replay with `--n` = `n_retrieved`. `excluded_top_m` in the export is the fit depth |

## Choosing a gate

Measured on the qdrant 1M cache, 32 heads, 128 kept keys. The thresholds were set on sessions 00–04
and are reported on the held-out sessions 05–09. Errors are the relative L2 error of the attention
output against full attention.

| gate (`--gate-worse` / `--gate-p99`) | heads on | mean error, all 32 heads | worst on-head: rows worse / worst row |
|---|---|---|---|
| no tailM (kept keys only, today) | 0 | 0.428 | — |
| **0.03 / 3.0 (default)** | 22 | 0.206 | 2.6 % / 3.56× |
| 0.02 / 2.5 | 19 | 0.231 | 1.7 % / 2.87× |
| 0.01 / 2.0 (strict) | 16 | 0.250 | 0.2 % / 1.20× |

- **Default (0.03 / 3.0):** most of the gain. On rows where tailM is worse, it is typically about
  1.16× the kept-keys error. Off on qdrant 1M: L11H0–H3, L15H1–H3, L19H0, L19H1, L31H2.
- **Strict (0.01 / 2.0):** use it when no measurable regression is acceptable. It keeps about 80 % of
  the default gate's error reduction (0.428 → 0.250 against 0.428 → 0.206).
- Most L11, L15 and L19 heads fail even the default gate: their tail direction is hard to predict from
  the query. At the strict gate every head of those three layers is off.

To change the gate later, re-gate the existing files. This runs only the decode pass; nothing is rebuilt:

```bash
.venv/bin/python scripts/build_tailm.py --cache cache/qdrant/1M/qwen3_5 --validate-only \
  --validate ../kv-search/replay-data --gate-worse 0.01 --gate-p99 2.0
```

The sanity limits (median ≤ 2.5e-3, p99 ≤ 1e-2) are code constants in
`kv_search/tailm_validate.py`, not flags. They catch loading or RoPE bugs, not quality.

## Reading the output

### Progress

In a terminal you get progress bars. When piped you get time-stamped lines, for example
`[00:42] batch 2/4 L11H0,… scan 50%`, one line per finished head, and warnings (for example
`L23H0: eps 0.1 not met at any cut (best 0.106 @ 128), using 128`).

### Summary table, one row per head

| column | meaning |
|---|---|
| `cell` | `L15H3` = layer 15, KV head 3 (serves q-heads 12–15) |
| `ε cut` | the depth ε picked (`!` = ε not met at any ladder depth; `128 → 2128` = after `--cut-shift`) |
| `fit` | `fit_cut`, the depth the map was fitted at |
| `α` | the mass multiplier runtime applies |
| `prefill err/none` | held-out prefill error with tailM / with the kept keys only |
| `decode err mean/p90` | the same on the recorded decode queries, with tailM |
| `decode none` | decode error with the kept keys only (today's behaviour) |
| `worse` | share of decode rows where tailM is worse than `decode none` (gate input) |
| `p99 ratio` | p99 of `err(tailM)/err(none)` over those worse rows (gate input) |
| `sanity med/p99/max` | recomputed exact attention vs the recorded `attn_out`; about 1.3–1.7e-3 is normal (bf16 floor) |
| `gate` | ✓ on, ✗ off |
| `flags` | `! eps not met` · `? not validated` · `✗ gated off (<reason>)` · `✗ sanity` |

After the table: a per-layer table (heads on, mean decode errors), the line
`runtime keeps n_retrieved 128 keys per head; heads on: 22/32; off: …`, the peak device memory per
batch, and for unvalidated heads the ready-to-run `--validate-only` command.

Exit code: 0 on success, 1 if any sanity check failed. On bad input (a bad cell, `--heldout`,
`--n-retrieved`, a missing layer file, a missing `--json` folder and so on) the tool stops before any
compute with `error: …`.

### Files

`<out>/layerLL_headH.safetensors`, one per head, written atomically. Rebuilding a subset of heads
leaves the other files untouched.

Tensors (f32): `weight [d,d]`, `bias [d]` (the map: `û = normalise(weight·q̂ + bias)·scale`),
`mean [d]`, `cov [d,d]` (key moments).

The `tailm` metadata (JSON, format **version 2**; version 1 files are refused) holds:

- **runtime reads:** `gate_pass`, `n_retrieved`, `alpha`, `scale`, `n_keys`, `scaling`;
- **how the depth was chosen:** `cut_detected`, `cut`, `eps_met`, `fit_cut`, `ladder_stats` (the full ε
  curve), `cut_stats`;
- **numbers:** `prefill` (held-out, at the runtime configuration) and `decode` (all §8 validation
  numbers; `null` when not validated);
- **gate decision:** `gate`, with `validated`, `worse_tol`, `p99_tol`, `sessions`, `rows`, `worse`,
  `ratio_p99`, `sanity_pass`, and `reason` (why a head is off);
- **identity and build parameters:** `layer`, `head`, `dim`, `context_len`, `model`, `source_cache`,
  `built_at`, `samples`, `heldout`, `seed`, `ridge`, `ladder`, `eps`, `eps_stat`, `cut_shift`.

Reading a file in Python:

```python
from pathlib import Path
from kv_search.tailm import load_dir
heads = load_dir(Path("cache/qdrant/1M/qwen3_5/tailm"))   # {(layer, head): TailmHead}
h = heads[(15, 3)]
h.gate_pass, h.n_retrieved, h.alpha, h.fit_cut, h.meta["gate"]["reason"]
```

## How runtime uses a file

The runtime side is `kv-search chat --tailm` (README "Tail correction (tailM)"; design:
`docs/superpowers/specs/2026-09-23-tailm-runtime-design.md`). For each head:

1. No file, `gate_pass = false`, or a file `n_retrieved` ≠ the runtime's → today's path, unchanged.
2. Retrieve `n_retrieved` keys as today and compute exact attention over them.
3. Add the pseudo-key: `û = normalise(weight·q̂ + bias)·scale` with log-mass
   `ln α + gaussian_tail_mass(q, mean, cov, scaling, n_keys, b)`, where `b` is the weakest kept score
   (scaled, in f32) and `n_keys` counts prefill keys. Merge it with the exact part by log-sum-exp,
   then merge the decode keys exactly as today.

## Common recipes

```bash
# one head, strict gate, JSON for inspection
.venv/bin/python scripts/build_tailm.py --cache cache/qdrant/1M/qwen3_5 --cells L15H3 \
  --validate ../kv-search/replay-data --gate-worse 0.01 --gate-p99 2.0 --json /tmp/l15h3.json

# build now, gate later (no sessions yet); every head is off until the second command
.venv/bin/python scripts/build_tailm.py --cache cache/niah/qwen3_5
.venv/bin/python scripts/build_tailm.py --cache cache/niah/qwen3_5 --validate-only --validate <niah-sessions>

# a different retrieval size: rebuild (α and the gate depend on it)
.venv/bin/python scripts/build_tailm.py --cache cache/qdrant/1M/qwen3_5 --n-retrieved 64 \
  --validate ../kv-search/replay-data --out /tmp/tailm64

# a cache copied to a path without a tier label (1M tier = YaRN factor 4)
.venv/bin/python scripts/build_tailm.py --cache /data/copy/qwen3_5 --context-tokens 1000000 --validate …

# export the gated-on heads for a kv-replay KL run
.venv/bin/python scripts/build_tailm.py --cache cache/qdrant/1M/qwen3_5 --summary-only \
  --export-replay /tmp/replay-export
```

## Limits

- All gate numbers use the **exact** top `n_retrieved` keys. The runtime retrievers search exact
  too; keys an HNSW search would miss are not measured.
- The error is measured in the attention output, not in the model's output (KL), which is not
  checked yet.
- The numbers in this document come from one dataset (qdrant 1M). niah heads are gated on recorded
  niah decode sessions (runtime spec §10).
