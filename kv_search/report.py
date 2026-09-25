"""Report figures + HTML composition: figures from the analysis envelopes (cache/analysis/*.json),
prose from authored markdown under report/, composed into one HTML artifact."""

import base64
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from kv_search.analysis.io import read_envelope

FIG_DIR = Path("cache/report/figs")
REPORT_DIR = Path("report")
OUT_HTML = Path("cache/report/report.html")
ORDER = ["100k", "200k", "1M"]
COLORS = {"100k": "#1f77b4", "200k": "#ff7f0e", "1M": "#2ca02c"}


def _sizes(name: str) -> dict:
    try:
        return read_envelope(name)["sizes"]
    except FileNotFoundError:
        return {}


def _order(d: dict) -> list[str]:
    return [s for s in ORDER if s in d]


def make_figures(fig_dir: Path = FIG_DIR) -> list[Path]:
    """Regenerate every report figure from the current envelopes. Returns written paths."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    sweep, mse, mf = _sizes("sweep"), _sizes("mse"), _sizes("meanfield")
    written: list[Path] = []

    def save(fig, name: str) -> None:
        p = fig_dir / name
        fig.savefig(p, dpi=120)
        plt.close(fig)
        written.append(p)

    if sweep:
        order = _order(sweep)

        def prompts(s):
            return sweep[s]["prompts"]

        def abs_positions(s):
            ps = prompts(s)
            return (
                np.array([p["final_download_pct"] for p in ps])
                / 100 * sweep[s]["context_len"] * ps[0]["n_shards"]
            )

        # download % over generation
        fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
        for s in order:
            c = np.array([p["download_pct_curve"] for p in prompts(s)])
            x = np.linspace(0, 100, c.shape[1])
            ax.plot(x, c.mean(0), color=COLORS[s], label=f"{s} (ctx {sweep[s]['context_len']:,})")
            ax.fill_between(x, c.min(0), c.max(0), color=COLORS[s], alpha=0.15)
        ax.set_xlabel("generation progress [%]")
        ax.set_ylabel("cumulative download [% of context]")
        ax.set_title("Edge download over generation (mean +/- range, 5 prompts)")
        ax.legend()
        save(fig, "download_over_generation.png")

        # scaling with context
        ctxs = np.array([sweep[s]["context_len"] for s in order])
        frac = np.array([np.mean([p["final_download_pct"] for p in prompts(s)]) for s in order])
        absol = np.array([abs_positions(s).mean() for s in order])
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
        a1.plot(ctxs, frac, "o-", color="#1f77b4")
        a1.set(xscale="log", xlabel="context length [tokens]", ylabel="download [% of context]",
               title="Downloaded fraction shrinks with context")
        a2.plot(ctxs, absol, "o-", color="#2ca02c")
        a2.set(xscale="log", xlabel="context length [tokens]",
               ylabel="distinct positions (sum over 32 shards)",
               title="Absolute download grows sub-linearly")
        for a, ys in ((a1, frac), (a2, absol)):
            for x, s, y in zip(ctxs, order, ys):
                a.annotate(s, (x, y))
        save(fig, "download_scaling.png")

        # weight on newly-fetched positions
        fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
        for s in order:
            c = np.array([p["weight_mass_new_curve"] for p in prompts(s)]) * 100
            x = np.linspace(0, 100, c.shape[1])
            ax.plot(x, c.mean(0), color=COLORS[s], label=s)
            ax.fill_between(x, c.min(0), c.max(0), color=COLORS[s], alpha=0.15)
        ax.set_xlabel("generation progress [%]")
        ax.set_ylabel("attention weight on newly-fetched positions [%]")
        ax.set_title("Freshly-downloaded positions carry little weight")
        ax.legend()
        save(fig, "weight_on_new.png")

        # download-vs-quality tradeoff
        have_mse = bool(mse)
        fig, axes = plt.subplots(
            1, 2 if have_mse else 1, figsize=(12 if have_mse else 7, 5),
            layout="constrained", squeeze=False,
        )
        axw = axes[0][0]
        for s in order:
            dl = np.mean([p["tradeoff_download_frac"] for p in prompts(s)], 0) * 100
            wr = np.mean([p["tradeoff_weight_retained"] for p in prompts(s)], 0) * 100
            axw.plot(dl, wr, "o-", color=COLORS[s], label=s)
        axw.set(xlabel="download [% of naive]", ylabel="attention weight retained [%]",
                title="Weight-threshold policy: weight retained")
        axw.legend()
        if have_mse:
            axm = axes[0][1]
            for s in _order(mse):
                dl = np.mean([p["download_frac"] for p in mse[s]["prompts"]], 0) * 100
                ms = np.mean([p["mse"] for p in mse[s]["prompts"]], 0)
                m = ms > 0  # tau=0 has MSE 0; drop for log axis
                axm.plot(dl[m], ms[m], "o-", color=COLORS[s], label=s)
            axm.set(yscale="log", xlabel="download [% of naive]",
                    ylabel="output MSE vs full top-k (log)",
                    title="Weight-threshold policy: output MSE")
            axm.legend()
        save(fig, "tradeoff.png")

    if mf:
        sizes_mf = _order(mf)
        fig, axes = plt.subplots(
            1, len(sizes_mf), figsize=(5 * len(sizes_mf), 4.5),
            layout="constrained", squeeze=False,
        )
        for ax, s in zip(axes[0], sizes_mf):
            ps = mf[s]["prompts"]
            hs = ps[0]["head_sizes"]
            mean = lambda n: np.mean([p[n] for p in ps], 0)  # noqa: E731
            ax.semilogy(hs, mean("head_renorm"), "o-", label="drop tail")
            ax.semilogy(hs, mean("meanfield_inst"), "o-", label="head + step tail-mean")
            ax.semilogy(hs, mean("meanfield_window"), "o-", label="head + windowed mean")
            ax.set(xlabel="head size (exact positions kept)",
                   ylabel="output MSE vs full top-k (log)", title=s)
            ax.legend()
        save(fig, "tail_meanfield.png")

    # --- retrieval fundamentals (one axis, sizes as lines) ---
    tm = _sizes("topk_mse")
    if tm:
        fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
        for s in _order(tm):
            ax.loglog(tm[s]["ks"], tm[s]["mse"], "-", base=2, color=COLORS[s], label=s)
            ax.loglog(tm[s]["ks"], tm[s]["mse_random"], "--", base=2, color=COLORS[s], alpha=.5)
        ax.set_xlabel("k (top-k kept)")
        ax.set_ylabel("attention-output MSE vs full (log)")
        ax.set_title("Top-k suffices (solid); random-k baseline (dashed)")
        ax.legend()
        save(fig, "topk_mse.png")

    lr = _sizes("layer_reuse")
    if lr:
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
        for s in _order(lr):
            a1.plot(lr[s]["layers"], lr[s]["hit_ceiling"], "o-", color=COLORS[s], label=s)
            a2.plot(lr[s]["layers"], lr[s]["retention"], "o-", color=COLORS[s], label=s)
        a1.set(xlabel="full-attention layer", ylabel="hit ceiling", title="Reuse: hit ceiling")
        a2.set(xlabel="full-attention layer", ylabel="retention", title="Reuse: step-to-step retention")
        a1.legend()
        save(fig, "layer_reuse.png")

    lv = _sizes("live_vs_retrieved")
    if lv:
        fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
        ax.axhline(0, color="0.6", lw=0.8)
        for s in _order(lv):
            ax.plot(lv[s]["layers"], lv[s]["gap"], "o-", color=COLORS[s], label=s)
        ax.set_xlabel("full-attention layer")
        ax.set_ylabel("top live logit - top retrieved logit")
        ax.set_title("Live context out-scores retrieved in shallow layers")
        ax.legend()
        save(fig, "live_vs_retrieved.png")

    cc = _sizes("cross_layer_coverage")
    if cc:
        fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
        for s in _order(cc):
            ax.plot(cc[s]["layers"], cc[s]["cum_act"], "o-", color=COLORS[s], label=s)
            ax.plot(cc[s]["layers"], cc[s]["cum_base"], "o--", color=COLORS[s], alpha=.5)
        ax.set_ylim(0, 1)
        ax.set_xlabel("full-attention layer")
        ax.set_ylabel("coverage by earlier layers")
        ax.set_title("Cross-layer position reuse (solid); popularity baseline (dashed)")
        ax.legend()
        save(fig, "cross_layer_coverage.png")

    return written


def _inline(text: str) -> str:
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', text)
    return text


def _img(alt: str, src: str, fig_dir: Path) -> str:
    p = fig_dir / Path(src).name
    if not p.exists():
        return f"<p><em>[missing figure: {src}]</em></p>"
    b64 = base64.b64encode(p.read_bytes()).decode()
    return (
        f'<figure><img alt="{alt}" src="data:image/png;base64,{b64}">'
        f"<figcaption>{_inline(alt)}</figcaption></figure>"
    )


def _table(rows: list[str]) -> str:
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    head, body = cells[0], cells[2:]  # row 1 is the |---| separator
    h = "".join(f"<th>{_inline(c)}</th>" for c in head)
    b = "".join(
        "<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in r) + "</tr>" for r in body
    )
    return f"<div class='tbl'><table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>"


def _md_to_html(md: str, fig_dir: Path) -> str:
    lines = md.split("\n")
    out: list[str] = []
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        m = re.fullmatch(r"!\[([^\]]*)\]\(([^)]+)\)", line.strip())
        if m:
            out.append(_img(m.group(1), m.group(2), fig_dir))
            i += 1
        elif line.startswith("#"):
            lvl = len(line) - len(line.lstrip("#"))
            out.append(f"<h{lvl}>{_inline(line[lvl:].strip())}</h{lvl}>")
            i += 1
        elif line.startswith(">"):
            buf = []
            while i < n and lines[i].startswith(">"):
                buf.append(lines[i][1:].strip())
                i += 1
            out.append(f"<blockquote>{_inline(' '.join(buf))}</blockquote>")
        elif line.lstrip().startswith("- "):
            buf = []
            while i < n and lines[i].lstrip().startswith("- "):
                buf.append(f"<li>{_inline(lines[i].lstrip()[2:])}</li>")
                i += 1
            out.append("<ul>" + "".join(buf) + "</ul>")
        elif line.strip().startswith("|"):
            buf = []
            while i < n and lines[i].strip().startswith("|"):
                buf.append(lines[i])
                i += 1
            out.append(_table(buf))
        else:
            buf = []
            while i < n and lines[i].strip() and lines[i][:1] not in "#>|" \
                    and not lines[i].lstrip().startswith("- "):
                buf.append(lines[i])
                i += 1
            out.append(f"<p>{_inline(' '.join(buf))}</p>")
    return "\n".join(out)


def build_html(
    report_dir: Path = REPORT_DIR, fig_dir: Path = FIG_DIR, out: Path = OUT_HTML
) -> Path:
    """Compose report/*.md (sorted) + embedded figures into one standalone HTML file."""
    body = "\n".join(
        _md_to_html(f.read_text(), fig_dir) for f in sorted(Path(report_dir).glob("*.md"))
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_HTML.replace("%%BODY%%", body))
    return out


_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>KV-search report</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{--bg:#f6f8fa;--surface:#fff;--ink:#0f1720;--muted:#55657a;--line:#dfe6ee;--accent:#0d7d8a;--accent-soft:#0d7d8a14}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#0b1017;--surface:#121a24;--ink:#e6edf3;--muted:#8b9bb0;--line:#223040;--accent:#35c4c4;--accent-soft:#35c4c41f}}
:root[data-theme=dark]{--bg:#0b1017;--surface:#121a24;--ink:#e6edf3;--muted:#8b9bb0;--line:#223040;--accent:#35c4c4;--accent-soft:#35c4c41f}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--ink);font-family:"IBM Plex Sans",system-ui,sans-serif;line-height:1.62;margin:0}
.wrap{max-width:900px;margin:0 auto;padding:clamp(24px,5vw,64px) clamp(18px,4vw,32px) 96px}
h1{font-size:clamp(1.9rem,4vw,2.6rem);font-weight:700;letter-spacing:-.02em;margin:2.4em 0 .5em;line-height:1.1}
h1:first-child{margin-top:0}
h2{font-size:1.5rem;font-weight:700;margin:2em 0 .5em;padding-top:1em;border-top:1px solid var(--line)}
h3{font-size:1.15rem;font-weight:600;margin:1.6em 0 .4em}
p{max-width:68ch}
strong{font-weight:600}
code{font-family:"IBM Plex Mono",monospace;font-size:.88em;background:var(--accent-soft);color:var(--accent);padding:1px 5px;border-radius:5px}
a{color:var(--accent)}
blockquote{background:var(--accent-soft);border-left:3px solid var(--accent);border-radius:0 8px 8px 0;padding:12px 18px;margin:1.4em 0;max-width:68ch}
ul{max-width:68ch}
li{margin:.35em 0}
figure{margin:1.6em 0;background:#fff;border:1px solid var(--line);border-radius:10px;padding:12px;overflow-x:auto}
figure img{display:block;width:100%;height:auto;border-radius:4px}
figcaption{font-family:"IBM Plex Mono",monospace;font-size:.72rem;color:var(--muted);margin-top:8px}
.tbl{overflow-x:auto;margin:1.4em 0}
table{border-collapse:collapse;font-size:.92rem;font-variant-numeric:tabular-nums}
th{font-family:"IBM Plex Mono",monospace;font-size:.7rem;text-transform:uppercase;letter-spacing:.05em;color:var(--muted);text-align:right;padding:6px 14px;border-bottom:1px solid var(--line)}
th:first-child{text-align:left}
td{padding:7px 14px;border-bottom:1px solid var(--line);text-align:right}
td:first-child{text-align:left}
tr:last-child td{border-bottom:none}
</style></head>
<body><div class="wrap">
%%BODY%%
</div></body></html>
"""

