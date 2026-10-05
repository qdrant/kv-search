"""Terminal demo: one question, one answer on the whole screen; the answer streams as it is generated.

    python -m llm_memory tui --local-vectors

Tab switches between memory with text, memory only, RAG with bge-small, and RAG with ColBERT for the same question
(answers are kept, so switching back is instant); Ctrl+R shows the found places and what the model read; Ctrl+T cycles
the abstain threshold; Up and Down walk through the question history; Ctrl+C or typing exit quits.
Below the answer: the probability that the answer is not in memory, a map of what the model read in each found
document, and where the time went.
"""
import os
import re

from rich.console import Group
from rich.panel import Panel
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.widget import Widget
from textual.widgets import Footer, Input

MODES = (("ours", "MEMORY + TEXT"), ("mem", "MEMORY ONLY"), ("rag", "RAG · bge-small"), ("colbert", "RAG · ColBERT"))
OURS = ("ours", "mem")
SUBTITLE = {"ours": "tower search in Qdrant; the model reads {k} × {t} text tokens around the found places (as much text as "
                    "RAG) plus memory: token vectors of {p} tokens around each",
            "mem": "tower search in Qdrant; the model reads only memory: token vectors of {mp} tokens around each of {mk} "
                   "found places, no text at all",
            "rag": "bge-small search in Qdrant; the model reads the text of {r} × {c}-token chunks",
            "colbert": "ColBERT search with Qdrant multivectors; the model reads the text of {r} × {c}-token chunks"}
THRESHOLDS = (0.9, 0.8, 0.5, None)
SPIN = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
C_PIECE, C_WINDOW, C_CHUNK = "#5fafd7", "#ffd75f", "#ffaf5f"
C_SEG = ("#87afff", "#af87ff", "#d787af", "#5fafd7", "#87d787")
DIAGRAM = """\
  question ─▶ Qwen3-4B (frozen) ─▶ tower ─▶ a vector per question token
                                                  │ kNN
                                                  ▼
         Qdrant: a vector per corpus token ─▶ 64-token windows ─▶ MaxSim ─▶ {k} places
                                                                               │
                                                                               ▼
  answer ◀─ Qwen3-4B layers 13–35 ◀─ reader (trained copy of layer 12) ◀─ memory: vectors of {p} tokens around each place
                                                                         + text: {t} tokens around each place\
"""
EXAMPLES = ("What does `combine_min_should_estimations` return when the minimum count exceeds the number of estimations?",
            "Which field type supports filtering by geographic bounding boxes and radius conditions?",
            "How does Qdrant support multivectors for late-interaction models like ColBERT?")


def gauge(p, thr, w=36):
    """Probability that the answer is not in memory, with the threshold marked."""
    fill, mark = round(p * w), (None if thr is None else min(w - 1, int(thr * w)))
    over = thr is not None and p >= thr
    t = Text()
    t.append("not in memory  ▕", style="dim")
    for i in range(w):
        if i == mark:
            t.append("┃", style="bold white")
        else:
            t.append("█" if i < fill else "░", style=("#ff875f" if over else "#87d787") if i < fill else "grey35")
    t.append("▏ ", style="dim")
    t.append(f"p {p:.2f}", style="bold")
    t.append(" · threshold " + ("off (greedy)" if thr is None else f"{thr}"), style="dim")
    t.append(" → abstains" if over else " → answers", style="#ff875f" if over else "#87d787")
    return t


def strip(n, marks, w):
    """One line for a document of n tokens: · not read, █ memory piece, ▓ found window or text chunk."""
    t = Text()
    for i in range(w):
        a, b = i * n / w, (i + 1) * n / w
        kind = None
        for lo, hi, k in marks:
            if lo < b and hi > a:
                kind = k
                if k != "piece":
                    break
        t.append(*{"piece": ("█", C_PIECE), "window": ("▓", C_WINDOW), "chunk": ("▓", C_CHUNK)}.get(kind, ("·", "grey35")))
    return t


def squash(s, w):
    s = re.sub(r"\s+", " ", s).strip()
    return s if len(s) <= w else s[:w - 1] + "…"


def doc_map(r, w):
    """Panel: for each found document, its score, title, the token ranges, a strip, and the found text."""
    ours, sw = r["mode"] in OURS, max(10, w - 4)
    pieces = {di: (lo, hi) for di, lo, hi in r.get("pieces", [])}
    docs = {}
    for x in r["sources"]:
        docs.setdefault(x["doc"], []).append(x)
    lines = []
    for di, xs in docs.items():
        head, title = Text(), xs[0]["title"]
        head.append(f"{max(x['score'] for x in xs):.3f}  ", style="bold")
        if len(title) > sw - 7:
            title = "…" + title[-(sw - 8):]
        path, name = os.path.split(title)
        head.append(f"{path}/" if path else "", style="grey62")
        head.append(name, style="bold")
        n = xs[0]["n"]
        rng = ", ".join(f"{x['lo']:,}–{x['hi']:,}" for x in xs)
        if ours and di in pieces:
            info = (f"{'found' if r['mode'] == 'mem' else 'text'} {rng} · memory piece {pieces[di][0]:,}–{pieces[di][1]:,} · "
                    f"document {n:,} tokens")
            marks = [(*pieces[di], "piece")] + [(x["lo"], x["hi"], "window") for x in xs]
        else:
            info = f"chunk {rng} · document {n:,} tokens"
            marks = [(x["lo"], x["hi"], "chunk") for x in xs]
        lines += [head, Text(info, style="dim"), strip(n, marks, sw)]
        lines += [Text(squash(x["text"], sw), style="italic grey62") for x in xs]
        lines.append(Text())
    legend = Text()
    legend.append("▓ ", style=C_WINDOW if ours else C_CHUNK)
    legend.append({"ours": "text around the found place", "mem": "found place, not given as text"}.get(r["mode"],
                  "chunk read as text"), style="dim")
    if ours:
        legend.append("   █ ", style=C_PIECE); legend.append("memory piece read as vectors", style="dim")
    legend.append("   · ", style="grey35"); legend.append("not read", style="dim")
    lines.append(legend)
    title = "What the model read" + {"ours": " · memory and text", "mem": " · memory only"}.get(r["mode"], " · text")
    return Panel(Group(*lines), title=title, title_align="left", border_style="grey42", padding=(0, 1))


def timing(r, w):
    if r["mode"] in OURS:
        t = r["timing"]
        knn, win, mem = t.get("kNN in Qdrant", 0), t.get("window vectors", 0), t.get("memory", 0)
        segs = [("question → vectors", r["search"] - knn - win), ("kNN in Qdrant", knn), ("window vectors", win),
                ("memory", mem), ("reading", r["reading"] - mem)]
    else:
        segs = [("search", r["search"]), ("reading", r["reading"])]
    total = sum(max(0, s) for _, s in segs) or 1e-9
    bw = max(10, min(60, w - 12))
    bar = Text()
    bar.append("▕", style="dim")
    for i, (_, s) in enumerate(segs):
        bar.append("█" * max(1 if s > 0.05 else 0, round(max(0, s) / total * bw)), style=C_SEG[i % len(C_SEG)])
    bar.append("▏ ", style="dim")
    bar.append(f"{total:.1f} s", style="bold")
    legend = Text()
    for i, (name, s) in enumerate(segs):
        legend.append("■ ", style=C_SEG[i % len(C_SEG)])
        legend.append(f"{name} {max(0, s):.1f}   ", style="dim")
    return Group(bar, legend)


class Header(Widget):
    DEFAULT_CSS = "Header { height: 2; padding: 0 1; background: $panel; }"

    def __init__(self, info):
        super().__init__()
        self.info, self.mode, self.done = info, "ours", set()

    def render(self):
        t = Text()
        t.append(" llm_memory ", style="bold #1c1c1c on #87d787")
        t.append("   ")
        for m, label in MODES:
            if m == self.mode:
                t.append(f" {label} ", style="bold reverse")
            else:
                t.append(f" {label}{' •' if m in self.done else ''} ", style="" if m in self.done else "dim")
            t.append(" ")
        i = self.info
        sub = Text(f"{os.path.basename(i['model'].rstrip('/'))} + {i['ckpt']} (step {i['step']}) · Qdrant collection "
                   f"{i['collection']} · {i['docs']:,} documents, {i['tokens'] / 1e6:.2f}M tokens", style="dim")
        return Group(t, sub)


class Answer(Widget):
    DEFAULT_CSS = "Answer { height: auto; }"

    def __init__(self, info):
        super().__init__()
        self.info, self.q, self.mode, self.text, self.r, self.phase, self.frame, self.error = info, None, "ours", "", None, "", 0, ""
        self.sources = False                                        # the map of what the model read: hidden until Ctrl+R

    def render(self):
        w = max(20, self.size.width)
        if self.q is None:
            t = Text()
            t.append("Ask a question about Qdrant: its source code, documentation, and articles "
                     f"({self.info['docs']:,} files).\n\n", style="bold")
            diagram = DIAGRAM.format(k=self.info["places"], p=self.info["piece"], t=self.info["text_win"])
            if w >= len(max(diagram.splitlines(), key=len)):
                t.append(diagram + "\n\n", style="#87afd7")
            t.append("The model reads memory, token vectors from Qdrant, instead of the text of found chunks. "
                     "Tab shows the same question answered from memory alone, and by RAG with bge-small and with "
                     "ColBERT.\n\n", style="")
            t.append("For example:\n", style="dim")
            for e in EXAMPLES:
                t.append(f"  • {e}\n", style="italic grey62")
            t.append("\nCtrl+C or typing exit quits.", style="dim")
            return t
        i = self.info
        sub = SUBTITLE[self.mode].format(c=i["chunk"], r=i["rag_k"], k=i["places"], t=i["text_win"], p=i["piece"],
                                         mk=i["mem_k"], mp=i["mem_piece"])
        parts = [Text(self.q, style="bold"), Text(sub, style="dim"), Text()]
        if self.error:
            return Group(*parts, Text(self.error, style="bold red"))
        if self.r is None:
            spin = SPIN[self.frame % len(SPIN)]
            if self.phase == "search":
                parts.append(Text(f"{spin} searching Qdrant…", style="dim"))
            else:
                parts.append(Text(self.text, style="bold #ffffd7") + Text(f" {spin}", style="dim"))
            return Group(*parts)
        r = self.r
        abst = r["answer"].strip() == self.info["abstain"]
        parts.append(Text(r["answer"] or "(empty answer)", style="bold italic #ff875f" if abst else "bold #ffffd7"))
        parts.append(Text())
        if r["mode"] == "mem":
            parts.append(gauge(r["p_abs"], r["thr"]))
            parts.append(Text(f"{len(r['sources'])} places: no text, only memory, {r['memory_rows']:,} token vectors in "
                              f"{len(r['pieces'])} pieces (one per document)", style="dim"))
        elif r["mode"] == "ours":
            parts.append(gauge(r["p_abs"], r["thr"]))
            parts.append(Text(f"{len(r['sources'])} places: {len(r['txt']):,} text tokens + memory, {r['memory_rows']:,} token "
                              f"vectors in {len(r['pieces'])} pieces (one per document)", style="dim"))
        else:
            parts.append(Text(f"{len(r['sources'])} chunks, {len(r['txt'])} text tokens", style="dim"))
        if self.sources:
            parts += [Text(), doc_map(r, w)]
        else:
            parts.append(Text("Ctrl+R shows the found places and what the model read", style="dim italic"))
        parts += [Text(), timing(r, w)]
        return Group(*parts)


class QuestionInput(Input):
    BINDINGS = [Binding("up", "history(-1)", show=False), Binding("down", "history(1)", show=False)]

    def __init__(self, **kw):
        super().__init__(**kw)
        self.history, self.pos = [], 0

    def action_history(self, d):
        if not self.history:
            return
        self.pos = max(0, min(len(self.history), self.pos + d))
        self.value = self.history[self.pos] if self.pos < len(self.history) else ""
        self.cursor_position = len(self.value)


class MemoryApp(App):
    TITLE = "llm_memory"
    ENABLE_COMMAND_PALETTE = False
    CSS = """
    Screen { background: $surface; }
    QuestionInput { margin: 1 1 0 1; border: round $accent; }
    #body { padding: 1 2; }
    """
    BINDINGS = [Binding("tab", "mode(1)", "Next mode", key_display="Tab", priority=True),
                Binding("shift+tab", "mode(-1)", "Previous mode", key_display="Shift+Tab", priority=True),
                Binding("ctrl+r", "sources", "Sources", key_display="Ctrl+R", priority=True),
                Binding("ctrl+t", "threshold", "Abstain threshold", key_display="Ctrl+T", priority=True),
                Binding("ctrl+c", "quit", "Quit", key_display="Ctrl+C", priority=True),
                Binding("ctrl+q", "quit", show=False, priority=True)]

    def __init__(self, query, thr, info):
        super().__init__()
        self.query_fn, self.thr, self.info, self.cache, self.busy = query, thr, info, {}, False

    def compose(self) -> ComposeResult:
        self.header, self.answer = Header(self.info), Answer(self.info)
        yield self.header
        yield QuestionInput(placeholder="Ask about Qdrant: its code, docs, and articles. Enter to ask.", id="q")
        with VerticalScroll(id="body"):
            yield self.answer
        yield Footer()

    def on_mount(self):
        self.set_interval(0.1, self.tick)
        self.query_one(QuestionInput).focus()

    def tick(self):
        if self.busy:
            self.answer.frame += 1
            self.answer.refresh(layout=True)

    def key(self, q, mode):
        return q, mode, (self.thr[0] if mode in OURS else None)

    def on_input_submitted(self, ev):
        q = ev.value.strip()
        if q.lower() in ("exit", "quit", "/exit", "/quit", ":q"):
            self.exit()
            return
        if not q or self.busy:
            return
        inp = self.query_one(QuestionInput)
        inp.history.append(q); inp.pos = len(inp.history); inp.value = ""
        self.ask(q, self.header.mode)

    def ask(self, q, mode):
        self.header.mode = mode
        self.header.done = {m for m, _ in MODES if self.key(q, m) in self.cache}
        self.header.refresh()
        a = self.answer
        a.q, a.mode, a.text, a.error = q, mode, "", ""
        if self.key(q, mode) in self.cache:
            a.r = self.cache[self.key(q, mode)]
        else:
            a.r, a.phase, self.busy = None, "search", True
            self.run_query(q, mode)
        a.refresh(layout=True)
        self.query_one("#body").scroll_home(animate=False)

    @work(thread=True)
    def run_query(self, q, mode):
        try:
            r = self.query_fn(mode, q, lambda text: self.call_from_thread(self.stream, text))
        except Exception as e:                                      # show the error instead of leaving the app
            self.call_from_thread(self.failed, f"{type(e).__name__}: {e}")
            return
        self.call_from_thread(self.finish, q, mode, r)

    def stream(self, text):
        self.answer.phase, self.answer.text = "read", text
        self.answer.refresh(layout=True)

    def finish(self, q, mode, r):
        self.cache[self.key(q, mode)] = r
        self.busy = False
        if self.answer.q == q and self.answer.mode == mode:
            self.answer.r = r
            self.header.done.add(mode)
            self.header.refresh()
            self.answer.refresh(layout=True)

    def failed(self, msg):
        self.busy = False
        self.answer.error = msg
        self.answer.refresh(layout=True)

    def action_mode(self, d):
        if self.busy:
            self.notify("Still answering, wait a moment.")
            return
        order = [m for m, _ in MODES]
        mode = order[(order.index(self.header.mode) + d) % len(order)]
        if self.answer.q is None:
            self.header.mode = self.answer.mode = mode
            self.header.refresh()
            return
        self.ask(self.answer.q, mode)

    def action_sources(self):
        self.answer.sources = not self.answer.sources
        self.answer.refresh(layout=True)

    def action_threshold(self):
        if self.busy:
            self.notify("Still answering, wait a moment.")
            return
        self.thr[0] = THRESHOLDS[(THRESHOLDS.index(self.thr[0]) + 1) % len(THRESHOLDS)] if self.thr[0] in THRESHOLDS \
            else THRESHOLDS[0]
        self.notify("Abstain threshold: " + ("off (greedy)" if self.thr[0] is None else str(self.thr[0])))
        if self.answer.q is not None and self.header.mode in OURS:
            self.ask(self.answer.q, self.header.mode)


def run(query, thr, info):
    """query(mode, question, on_token) → result dict (see __main__); thr: one-element list with the abstain threshold."""
    import warnings

    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()
    warnings.filterwarnings("ignore")
    MemoryApp(query, thr, info).run()
