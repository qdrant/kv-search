"""Command line: build the index, upload it to Qdrant, build the baseline collections, run the demo.

    python -m llm_memory index      --ckpt checkpoints/s2c.pt --docs data/docs.jsonl --idx data/index
    python -m llm_memory upload     --ckpt checkpoints/s2c.pt --docs data/docs.jsonl --idx data/index
    python -m llm_memory baselines  --docs data/docs.jsonl
    python -m llm_memory ask        --ckpt checkpoints/s2c.pt --docs data/docs.jsonl --idx data/index --local-vectors
    python -m llm_memory tui        --ckpt checkpoints/s2c.pt --docs data/docs.jsonl --idx data/index --local-vectors
"""
import argparse
import json
import os
import time

from transformers import AutoTokenizer

NAMES = {"ours": "OURS (tower search in Qdrant + memory read)", "mem": "OURS, MEMORY ONLY (tower search, memory read, no text)",
         "rag": "RAG (bge-small, {c}-token chunks)",
         "colbert": "ColBERT (answerai-colbert-small, Qdrant multivectors, {c}-token chunks)"}


def threshold(v):
    return None if v in ("off", "none") else float(v)


def main():
    ap = argparse.ArgumentParser(prog="llm_memory")
    ap.add_argument("command", choices=["index", "upload", "baselines", "ask", "tui"])
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507", help="Hugging Face id or local path of the frozen LLM")
    ap.add_argument("--ckpt", default="checkpoints/s2c.pt", help="tower + reader checkpoint")
    ap.add_argument("--docs", default="data/docs.jsonl", help="corpus: jsonl with id, title, text")
    ap.add_argument("--idx", default="data/index", help="index directory: facts.npy, rows.npy, meta.json")
    ap.add_argument("--meta", default="data/meta.json", help="titles and rows of each document in the collection")
    ap.add_argument("--url", default="http://127.0.0.1:6333"); ap.add_argument("--collection", default="llm_memory")
    ap.add_argument("--enc-max", type=int, default=16384, help="index: longer documents are encoded in pieces of this size")
    ap.add_argument("--batch", type=int, default=256); ap.add_argument("--parallel", type=int, default=6)
    ap.add_argument("--chunk", type=int, default=256, help="baselines: chunk size in Qwen tokens")
    ap.add_argument("--chunks", default="data/chunks.json")
    ap.add_argument("--bge", default="BAAI/bge-small-en-v1.5"); ap.add_argument("--colbert", default="answerdotai/answerai-colbert-small-v1")
    ap.add_argument("--q", default="", help="ask: one question, then exit")
    ap.add_argument("--file", default="", help="ask: questions, one per line (optionally <question>\\t<reference>)")
    ap.add_argument("--mode", default="ours", choices=["ours", "mem", "rag", "colbert", "all"], help="ask: starting mode "
                    "(mem: our search and memory without any text)")
    ap.add_argument("--abstain-thr", type=threshold, default=0.8, help="ask: say 'not in the text' only if its probability "
                    "at the first step is at least this; 'off' for plain greedy decoding")
    ap.add_argument("--knn", type=int, default=256); ap.add_argument("--cand", type=int, default=128)
    ap.add_argument("--win", type=int, default=64); ap.add_argument("--ret-k", type=int, default=8, help="ours: places")
    ap.add_argument("--rag-k", type=int, default=4, help="RAG: chunks")
    ap.add_argument("--mem-k", type=int, default=4, help="memory only: places")
    ap.add_argument("--mem-piece", type=int, default=1024, help="memory only: memory piece around each place, tokens")
    ap.add_argument("--piece", type=int, default=512, help="memory piece around each found place, tokens")
    ap.add_argument("--text-win", type=int, default=128, help="ours: text tokens given around each found place (at least the "
                    "search window)")
    ap.add_argument("--max-new", type=int, default=96); ap.add_argument("--ef", type=int, default=256)
    ap.add_argument("--local-vectors", action="store_true", help="take window and memory vectors from the local index "
                    "(memmap) instead of retrieving them from Qdrant by id; kNN still runs in Qdrant")
    ap.add_argument("--compare-text", action="store_true", help="ours: also print the unmodified model's answer on the text "
                    "of the same windows")
    ap.add_argument("--snip", type=int, default=160); ap.add_argument("--dump", default="", help="ask: answers to jsonl")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    t0 = time.time()
    log = lambda s: print(f"{s} ({time.time() - t0:.0f}s)", flush=True)

    from qdrant_client import QdrantClient

    from .store import MemoryStore, build_index, load_docs, upload
    docs = load_docs(a.docs)

    if a.command == "baselines":
        from .baselines import build
        ch = build(QdrantClient(url=a.url, prefer_grpc=True, timeout=600), docs, AutoTokenizer.from_pretrained(a.model),
                   a.bge, a.colbert, a.chunk, log)
        os.makedirs(os.path.dirname(a.chunks) or ".", exist_ok=True)
        json.dump({"chunk": a.chunk, "chunks": ch}, open(a.chunks, "w"))
        log("baselines done")
        return
    if a.command == "upload":
        import torch
        mu = torch.load(a.ckpt, map_location="cpu")["mu"]
        upload(QdrantClient(url=a.url, prefer_grpc=True, timeout=600), a.collection, a.idx, docs, mu, a.meta, a.batch,
               a.parallel, log)
        return

    from .model import ABSTAIN, MemoryModel
    mm = MemoryModel(a.model, a.ckpt, a.device)
    log(f"model {a.model}, checkpoint {a.ckpt} (step {mm.step}), {len(docs)} documents")
    if a.command == "index":
        build_index(mm, docs, a.idx, a.enc_max, log)
        return

    # ---------- ask ----------
    from .baselines import TextRetrievers
    store = MemoryStore(a.url, a.collection, a.meta, mm.mu, a.device, a.idx if a.local_vectors else None)
    texts = TextRetrievers(store.qc, a.bge, a.colbert, a.rag_k)
    csize = json.load(open(a.chunks))["chunk"] if os.path.exists(a.chunks) else "?"
    tok, sep = mm.tok, mm.sep
    doc_ids = {}

    def doc_tok(di):
        if di not in doc_ids:
            doc_ids[di] = tok(docs[di]["text"], add_special_tokens=False).input_ids
        return doc_ids[di]

    thr, last = [a.abstain_thr], {}

    def text_span(di, lo, hi, c):
        """Text given to the model around a found place: the search window itself, or --text-win tokens centered on it."""
        if a.text_win <= a.win:
            return lo, hi
        n_ = len(doc_tok(di))
        lo = max(0, min(c - a.text_win // 2, n_ - a.text_win))
        return lo, min(n_, lo + a.text_win)

    def merge(src):
        """Overlapping text spans of one document become one span (best score), in order of first appearance."""
        out = []
        for di, lo, hi, s in src:
            for i, (dj, l2, h2, s2) in enumerate(out):
                if dj == di and lo <= h2 and hi >= l2:
                    out[i] = (di, min(lo, l2), max(hi, h2), max(s, s2))
                    break
            else:
                out.append((di, lo, hi, s))
        return out

    def query(mode, question, on_token=None):
        """One answer: search, memory, reading. Returns a dict for the command line and the terminal app."""
        store.timing.clear()
        t1 = time.time()
        r = {"mode": mode, "question": question, "thr": thr[0]}
        if mode in ("ours", "mem"):
            qv = mm.search_vectors(mm.token_vectors(tok(question, add_special_tokens=False).input_ids))
            places = store.search(qv, a.knn, a.cand, a.win, a.ret_k if mode == "ours" else a.mem_k, a.ef)
            t2 = time.time()
            if mode == "ours":
                src = [(di, *text_span(di, lo, hi, c), s) for di, lo, hi, c, s in places]
                src = merge(src) if a.text_win > a.win else src
                txt = [t_ for di, lo, hi, s in src for t_ in doc_tok(di)[lo:hi] + sep]
            else:                                                   # memory only: the found windows are shown, not read
                src, txt = [(di, lo, hi, s) for di, lo, hi, c, s in places], []
            mem, poss, span = store.memory(places, a.piece if mode == "ours" else a.mem_piece)
            r["answer"], r["p_abs"] = mm.answer(mem, poss, span, txt, question, thr[0], a.max_new, on_token)
            r["memory_rows"], r["pieces"] = span, list(store.pieces)
        else:
            src = texts.search(mode, question)
            t2 = time.time()
            txt = [t_ for di, lo, hi, s in src for t_ in doc_tok(di)[lo:hi] + sep]
            r["answer"] = mm.text_answer(txt, question, a.max_new, on_token)
        t3 = time.time()
        r.update(txt=txt, search=t2 - t1, reading=t3 - t2, timing=dict(store.timing) if mode in ("ours", "mem") else {},
                 sources=[{"doc": di, "title": store.titles[di], "lo": lo, "hi": hi, "score": s, "n": len(doc_tok(di)),
                           "text": tok.decode(doc_tok(di)[lo:hi])} for di, lo, hi, s in src])
        return r

    if a.command == "tui":
        from .tui import run
        for m_ in ("rag", "colbert"):
            texts.model(m_)
        run(query, thr, {"model": a.model, "ckpt": os.path.basename(a.ckpt), "step": mm.step, "collection": a.collection,
                         "docs": len(docs), "tokens": sum(n_ for r0, n_ in store.rows), "chunk": csize, "abstain": ABSTAIN,
                         "places": a.ret_k, "piece": a.piece, "text_win": max(a.text_win, a.win), "rag_k": a.rag_k,
                         "mem_k": a.mem_k, "mem_piece": a.mem_piece})
        return

    def answer(mode, question):
        r = query(mode, question)
        print(f"\n[{NAMES[mode].format(c=csize)}]\nANSWER: {r['answer']}")
        if mode in ("ours", "mem"):
            last["p_abs"] = r["p_abs"]
            print(f"p(answer not in memory) = {r['p_abs']:.2f}; abstain threshold: "
                  f"{'off (greedy)' if thr[0] is None else thr[0]}")
            extra = (f"; memory {r['memory_rows']} rows (vectors from {'local index' if a.local_vectors else 'Qdrant'}) + "
                     f"text {len(r['txt'])} tokens")
        else:
            extra = f"; text {len(r['txt'])} tokens"
        tim = r["timing"]
        print(f"search {r['search']:.1f} s, reading {r['reading']:.1f} s" + (" (" + ", ".join(f"{k} {v:.1f}" for k, v in tim.items()) + ")"
                                                                          if tim else "") + extra + "; sources:")
        for x in r["sources"]:
            print(f"  [{x['score']:.3f}] {x['title']}: {x['text'].replace(chr(10), ' ')[:a.snip]}")
        if mode == "ours" and a.compare_text:
            print(f"(the same model on the text of these windows: {mm.text_answer(r['txt'], question, a.max_new)})")
        return r["answer"], [x["title"] for x in r["sources"]]

    mode = [a.mode]

    def ask(question, ref=""):
        print(f"\nQUESTION: {question}" + (f"\nREFERENCE: {ref}" if ref else ""))
        rec = {"q": question, "ref": ref}
        for m_ in (["ours", "mem", "rag", "colbert"] if mode[0] == "all" else [mode[0]]):
            rec[m_], rec[m_ + "_src"] = answer(m_, question)
            if m_ in ("ours", "mem"):
                rec[m_ + "_p_abs"] = last["p_abs"]
        if a.dump:
            with open(a.dump, "a") as f_:
                f_.write(json.dumps(rec, ensure_ascii=False) + "\n")

    qs = [a.q] if a.q else ([x.rstrip("\n").split("\t") for x in open(a.file) if x.strip()] if a.file else [])
    for q in qs:
        ask(*q) if isinstance(q, list) else ask(q)
    if qs:
        return
    print("modes: /ours /mem /rag /colbert /all (all four on one question); abstain threshold: /thr 0.8, /thr off; "
          "exit: Ctrl-D")
    while True:
        try:
            q = input(f"\n[{mode[0]}] question> ").strip()
        except EOFError:
            break
        if q in ("/ours", "/mem", "/rag", "/colbert", "/all"):
            mode[0] = q[1:]
            print(f"mode: {mode[0]}")
        elif q.startswith("/thr"):
            thr[0] = threshold(q.split()[1] if len(q.split()) > 1 else "off")
            print(f"abstain threshold: {'off (greedy)' if thr[0] is None else thr[0]}")
        elif q:
            ask(q)


if __name__ == "__main__":
    main()
