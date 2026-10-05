"""Classic retrievers over the evaluation episodes, with dumps in the format of llm_memory.evaluate (for its `read` mode).

Ported from the nvme-llm research scripts exp120_bm25.py, exp121_dense.py and exp122_colbert.py. The corpus is cut into
windows of --win tokens of the reader tokenizer on the document grid (dense: chunks of --chunk tokens every --stride), the
same windows the tower search returns, so retrieval compares one to one. Search is an exact full scan.

  bm25     terms \\w+, lower-cased (code identifiers with _ are one term), k1 1.2, b 0.75; the centre of a window is the
           median query word found in it;
  dense    bge-small-en-v1.5 (CLS, BGE query instruction) or Qwen3-Embedding-0.6B (last token, left padding, query
           instruction); cosine;
  colbert  answerai-colbert-small-v1, MaxSim over each window. --impl pylate (pip install pylate) is what the published
           numbers used. --impl native is the encoder of the demo (llm_memory.baselines.Colbert) and needs no extra
           package: the same tokens as PyLate (4,982,724 on the Qdrant corpus), but an fp16 encoder instead of fp32, so a
           few rankings differ (Qdrant questions: 0.14/0.27/0.41 against 0.14/0.27/0.42 with PyLate).
The query is the question. Hit: a returned window overlaps the tokens of the episode's first anchor, at top-1/4/16.

    python -m llm_memory.retrievers bm25 --dump data/eval/bm25.json
    python -m llm_memory.retrievers dense --embedder BAAI/bge-small-en-v1.5 --kind bge --chunk 64 --dump data/eval/bge-64.json
    python -m llm_memory.retrievers dense --embedder Qwen/Qwen3-Embedding-0.6B --kind qwen3 --chunk 256 --stride 192 --bs 32 \\
        --dump data/eval/qwen3e-256.json
    python -m llm_memory.retrievers colbert --dump data/eval/colbert.json
"""
import argparse
import bisect
import collections
import json
import math
import os
import re
import time

import numpy as np
import torch
import torch.nn.functional as F

from .evaluate import DocTokens, groups, hit_line, load_docs_by_id, load_episodes

WORD = re.compile(r"\w+")
PREFIX = {"bge": "Represent this sentence for searching relevant passages: ",
          "qwen3": "Instruct: Given a question or a statement, retrieve passages that answer it or contain it\nQuery:"}


def chunks_of(docs, toks, size, stride=0):
    """Chunks (document id, first token, end token) of `size` tokens, every `stride` tokens (0: back to back), and their text."""
    chunks, texts = [], []
    for did in sorted(docs):
        _, st, en = toks.get(did)
        n = len(st)
        starts = list(range(0, max(1, n - size + stride), stride)) if stride else list(range(0, n, size))
        for s in starts:
            e = min(n, s + size)
            if e <= s:
                continue
            chunks.append((did, s, e)); texts.append(docs[did]["text"][st[s]:en[e - 1]])
    return chunks, texts


def query_text(e):
    return e["cloze"].replace("___", " ")


def score(eps, toks, ranked, method, win, log, grid=False):
    """Hit of the first anchor at top-1/4/16; ranked[i] = [(document id, first token, end token, centre)] best first.
    A hit is a chunk that overlaps the anchor tokens; with grid, a window of the grid that holds one of them (bm25)."""
    res = collections.defaultdict(lambda: [0, 0])
    dump = {}
    for e, rk in zip(eps, ranked):
        t0, t1 = toks.anchor(e["doc"], e["fact"]["a_start"], e["fact"]["a_end"])
        if grid:
            gold = {t // win for t in range(t0, t1)}
            hit = next((i for i, (d, s, x, c) in enumerate(rk) if d == e["doc"] and s // win in gold), 10 ** 9)
        else:
            hit = next((i for i, (d, s, x, c) in enumerate(rk) if d == e["doc"] and min(x, t1) > max(s, t0)), 10 ** 9)
        dump[e["id"]] = {"cloze": {method: [[d, s // win, c] for d, s, x, c in rk]}}
        for grp in ("all", e["kind"]):
            for k in (1, 4, 16):
                res[(grp, k)][0] += hit < k; res[(grp, k)][1] += 1
    log(f"{method}: anchor in top-1/4/16 windows of the whole corpus, {len(eps)} episodes: "
        + "; ".join(f"{g} {hit_line(res, g)}" for g in groups(res)))
    return dump


def bm25(a, docs, toks, eps, log):
    chunks, texts = chunks_of(docs, toks, a.win)
    vocab, tfs, dls = {}, [], []
    for text in texts:
        terms = WORD.findall(text.lower())
        tfs.append(collections.Counter(vocab.setdefault(t, len(vocab)) for t in terms)); dls.append(len(terms))
    N, avgdl = len(chunks), float(np.mean(dls))
    post = collections.defaultdict(lambda: ([], []))
    for i, cnt in enumerate(tfs):
        for t, c in cnt.items():
            post[t][0].append(i); post[t][1].append(c)
    dl = np.array(dls, dtype=np.float32)
    P = {}
    for t, (ids, cs) in post.items():
        ids, cs = np.array(ids, dtype=np.int64), np.array(cs, dtype=np.float32)
        idf = math.log(1 + (N - len(ids) + 0.5) / (len(ids) + 0.5))
        P[t] = (ids, idf * cs * (a.k1 + 1) / (cs + a.k1 * (1 - a.b + a.b * dl[ids] / avgdl)))
    del post, tfs
    log(f"BM25 index: {N} windows, {len(P)} terms, {avgdl:.1f} words per window on average")
    ranked = []
    for e in eps:
        text = query_text(e)
        sc = np.zeros(N, dtype=np.float32)
        for t in set(WORD.findall(text.lower())):
            if t in vocab and vocab[t] in P:
                ids, v = P[vocab[t]]
                sc[ids] += v
        qt = set(WORD.findall(text.lower()))
        rk = []
        for i in np.argsort(-sc)[:16]:                  # centre: the median query word found in the window (document token)
            did, lo, hi = chunks[i]
            _, st, en = toks.get(did)
            body = docs[did]["text"][st[lo]:en[hi - 1]]
            pos = [st[lo] + m.start() for m in WORD.finditer(body) if m.group().lower() in qt]
            c = bisect.bisect_right(st, sorted(pos)[len(pos) // 2]) - 1 if pos else (lo + hi) // 2
            rk.append((did, lo, hi, int(c)))
        ranked.append(rk)
    return score(eps, toks, ranked, "bm25", a.win, log, grid=True), a.win


def dense(a, docs, toks, eps, log):
    from transformers import AutoModel, AutoTokenizer
    dev, stride = a.device, a.stride or a.chunk
    etok = AutoTokenizer.from_pretrained(a.embedder, padding_side="left" if a.kind == "qwen3" else "right")
    emb = AutoModel.from_pretrained(a.embedder, dtype=torch.bfloat16 if dev != "cpu" else torch.float32).eval().to(dev)
    bs = a.bs or 128

    @torch.no_grad()
    def encode(texts):
        out = [None] * len(texts)
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        for b in range(0, len(order), bs):
            idx = order[b:b + bs]
            x = etok([texts[i] for i in idx], padding=True, truncation=True, max_length=a.max_len, return_tensors="pt").to(dev)
            h = emb(**x).last_hidden_state
            v = h[:, 0] if a.kind == "bge" else h[:, -1]                    # bge: CLS; qwen3: last token (left padding)
            v = F.normalize(v.float(), dim=-1).half()
            for j, i in enumerate(idx):
                out[i] = v[j]
        return torch.stack(out)

    chunks, texts = chunks_of(docs, toks, a.chunk, a.stride)
    log(f"{len(chunks)} chunks of {a.chunk} tokens every {stride}, {len(docs)} documents; encoding with {a.embedder}")
    E = encode(texts)
    del texts
    Q = encode([PREFIX[a.kind] + query_text(e) for e in eps])
    top = (Q @ E.T).float().topk(16, dim=-1).indices.cpu().numpy()
    ranked = [[chunks[i] + ((chunks[i][1] + chunks[i][2]) // 2,) for i in row] for row in top]
    return score(eps, toks, ranked, "dense", a.chunk, log), a.chunk


def colbert(a, docs, toks, eps, log):
    dev = a.device
    chunks, texts = chunks_of(docs, toks, a.win)
    log(f"{len(chunks)} windows of {a.win} tokens, {len(docs)} documents; encoding with {a.colbert} ({a.impl})")
    if a.impl == "pylate":
        from pylate import models
        cb = models.ColBERT(model_name_or_path=a.colbert, device=dev, query_length=a.qlen, document_length=a.dlen)
        D = cb.encode(texts, batch_size=a.bs or 256, is_query=False, show_progress_bar=False)
        enc_q = lambda qs: cb.encode(qs, batch_size=64, is_query=True, show_progress_bar=False)
    else:                                                               # same tokens as PyLate, fp16 encoder: see the docstring
        from .baselines import Colbert
        cb = Colbert(a.colbert, dev)
        D = cb.docs(texts, dlen=a.dlen, bs=a.bs or 64)
        enc_q = lambda qs: [cb.query(q, qlen=a.qlen) for q in qs]
    del texts
    lens = np.array([len(x) for x in D])
    order = np.argsort(lens, kind="stable")                             # packs of windows of similar length: little padding
    packs = []
    for c in range(0, len(order), a.wchunk):
        idx = order[c:c + a.wchunk]
        L = int(lens[idx].max())
        P = torch.zeros(len(idx), L, D[idx[0]].shape[1], dtype=torch.float16)
        M = torch.zeros(len(idx), L, dtype=torch.bool)
        for j, i in enumerate(idx):
            P[j, :lens[i]] = torch.as_tensor(np.asarray(D[i]), dtype=torch.float16); M[j, :lens[i]] = True
        packs.append((torch.as_tensor(idx, device=dev), P.to(dev), M.to(dev)))
    del D
    log(f"corpus encoded: {int(lens.sum())} ColBERT tokens ({lens.mean():.1f} per window), "
        f"{sum(p[1].numel() for p in packs) * 2 / 1e9:.2f} GB on the GPU")
    Q = torch.as_tensor(np.stack([np.asarray(x) for x in enc_q([query_text(e) for e in eps])]), dtype=torch.float16, device=dev)
    tops = []
    with torch.no_grad():
        for b in range(0, len(Q), a.qbatch):
            q = Q[b:b + a.qbatch]                                       # [nq, lq, d]
            sc = torch.full((len(q), len(chunks)), -1e4, device=dev)
            for idx, P, M in packs:
                s = torch.einsum("qid,wld->qwil", q, P)                 # [nq, nw, lq, L] fp16
                s = s.masked_fill_(~M[None, :, None, :], -1e4).amax(-1).float().sum(-1)
                sc[:, idx] = s
            tops.append(sc.topk(16, dim=-1).indices.cpu())
    top = torch.cat(tops).numpy()
    ranked = [[chunks[i] + ((chunks[i][1] + chunks[i][2]) // 2,) for i in row] for row in top]
    return score(eps, toks, ranked, "colbert", a.win, log), a.win


def main():
    ap = argparse.ArgumentParser(prog="python -m llm_memory.retrievers")
    ap.add_argument("method", choices=["bm25", "dense", "colbert"])
    ap.add_argument("--tok", default="Qwen/Qwen3-4B-Instruct-2507", help="reader tokenizer: the window grid")
    ap.add_argument("--docs", default="data/docs.jsonl"); ap.add_argument("--episodes", default="data/eval/episodes.jsonl")
    ap.add_argument("--n", type=int, default=400, help="episodes (shuffled with seed 115, first n), as llm_memory.evaluate")
    ap.add_argument("--max-doc", type=int, default=1000000, help="use only the first tokens of each document")
    ap.add_argument("--win", type=int, default=64, help="bm25, colbert: window size in reader tokens")
    ap.add_argument("--dump", default="", help="retrieval dump (default data/eval/<method>.json)")
    ap.add_argument("--k1", type=float, default=1.2); ap.add_argument("--b", type=float, default=0.75)
    ap.add_argument("--embedder", default="BAAI/bge-small-en-v1.5", help="dense: bge-small-en-v1.5 or Qwen3-Embedding-0.6B")
    ap.add_argument("--kind", default="bge", choices=["bge", "qwen3"], help="dense: pooling and query instruction")
    ap.add_argument("--chunk", type=int, default=64, help="dense: chunk size in reader tokens")
    ap.add_argument("--stride", type=int, default=0, help="dense: chunk step, 0 = back to back")
    ap.add_argument("--max-len", type=int, default=512, help="dense: embedder tokens per chunk")
    ap.add_argument("--bs", type=int, default=0, help="encoding batch (default: dense 128, colbert 256 pylate / 64 native)")
    ap.add_argument("--colbert", default="answerdotai/answerai-colbert-small-v1")
    ap.add_argument("--impl", default="pylate", choices=["pylate", "native"])
    ap.add_argument("--qlen", type=int, default=64, help="colbert: query length with [MASK] expansion")
    ap.add_argument("--dlen", type=int, default=256, help="colbert: tokens per window (64 Qwen tokens are up to ~150 BERT ones)")
    ap.add_argument("--wchunk", type=int, default=1024); ap.add_argument("--qbatch", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    t0 = time.time()
    log = lambda s: print(f"{s} ({time.time() - t0:.0f}s)", flush=True)
    if torch.version.hip:
        torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False)
    from transformers import AutoTokenizer
    docs = load_docs_by_id(a.docs)
    toks = DocTokens(AutoTokenizer.from_pretrained(a.tok), docs, a.max_doc, size=len(docs))
    eps = load_episodes(a.episodes, docs, a.n)
    dump, win = {"bm25": bm25, "dense": dense, "colbert": colbert}[a.method](a, docs, toks, eps, log)
    path = a.dump or f"data/eval/{a.method}.json"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    json.dump({"win": win, "max_doc": a.max_doc, "eps": dump}, open(path, "w"), ensure_ascii=False)
    log(f"dump → {path}")


if __name__ == "__main__":
    main()
