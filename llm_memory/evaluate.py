"""Evaluation on generated questions: episodes, retrieval hit@k, and answers for the judge.

Ported from the nvme-llm research script exp125_e2e.py without changes to the computation. Modes:

  episodes  verified generated questions → evaluation episodes, one per question with a short answer, and the token count of
            every document (index.json, used by the research baselines);
  search    question → tower → one vector per question token → exact kNN over a local index (store.build_index), top --knn
            per token → 64-token windows of the document grid by maximum score and by votes → MaxSim over each candidate
            window → top 16 with a centre (the row that best matches the median question token); hit of the anchor window
            at top-1/4/16 and a dump for `read`;
  read      answers from any retrieval dump (this search or llm_memory.retrievers). Reader `ours`: the vectors of a --piece
            token piece around the centre of each of the top --ret-k windows (one piece per document) as memory, plus text;
            reader `rag`: the unmodified model reading the text of the top --ret-k windows. Answers → jsonl for
            llm_memory.judge.

Episodes are shuffled with seed 115 and the first --n are used, so every retriever and reader sees the same questions.
Retrieval dump: {"win": W, "max_doc": N, "eps": {episode id: {"cloze": {method: [[document id, window, centre], ...]}}}},
best first; window w covers tokens [w·W, (w+1)·W) of the document in the reader tokenizer, the centre is a token position.

    python -m llm_memory.evaluate episodes --questions data/questions-v.jsonl --docs data/docs.jsonl --out data/eval
    python -m llm_memory.evaluate search --ckpt checkpoints/s2c.pt --idx data/index --dump data/eval/ours.json
    python -m llm_memory.evaluate read --reader ours --ckpt checkpoints/s2c.pt --retrieved data/eval/ours.json \\
        --win-all --abstain-curve --dump-answers data/eval/answers/ours.jsonl
    python -m llm_memory.evaluate read --reader rag --retrieved data/eval/bm25.json --ret-method bm25 \\
        --dump-answers data/eval/answers/rag-bm25.jsonl
"""
import argparse
import bisect
import collections
import gc
import json
import os
import random
import re
import string
import time

import numpy as np
import torch
import torch.nn.functional as F

from .model import ABSTAIN, rmsn

NORM_BLOCK = 65536      # rows normalized per batch when the index is prepared
KNN_CHUNK = 1 << 20     # index rows per kNN step when the index is streamed
Q_BATCH = 2048          # question tokens per kNN matmul when the index is streamed


def norm_ans(s):
    """Lower case, no punctuation, no articles: the containment measure ("ok") and the abstain check of the research code."""
    s = "".join(ch for ch in s.lower() if ch not in set(string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", s).split())


def contains(out, e):
    return any(norm_ans(x) and norm_ans(x) in norm_ans(out) for x in [e["answer"]] + list(e.get("aliases", [])))


def load_docs_by_id(path):
    docs = {}
    for line in open(path):
        dd = json.loads(line)
        docs[dd["id"]] = dd
    return docs


class DocTokens:
    """Reader-tokenizer ids and character offsets of documents (first max_doc tokens), with an LRU cache."""

    def __init__(self, tok, docs, max_doc=None, size=2000):
        self.tok, self.docs, self.max_doc, self.size = tok, docs, max_doc, size
        self.cache = collections.OrderedDict()

    def get(self, did):
        """→ (ids, start characters, end characters)."""
        if did not in self.cache:
            if len(self.cache) > self.size:
                self.cache.popitem(last=False)
            enc = self.tok(self.docs[did]["text"], add_special_tokens=False, return_offsets_mapping=True)
            off = enc["offset_mapping"][:self.max_doc]
            self.cache[did] = (enc["input_ids"][:self.max_doc], [o[0] for o in off], [o[1] for o in off])
        return self.cache[did]

    def ids(self, did):
        return self.get(did)[0]

    def anchor(self, did, c0, c1):
        """Characters [c0, c1) → tokens [t0, t1), at least one token."""
        _, st, en = self.get(did)
        i0 = bisect.bisect_left(st, c0)
        return i0, max(i0 + 1, bisect.bisect_right(en, c1))


def load_episodes(path, docs, n, seed=115):
    """Episodes with an answer whose document is in the corpus, shuffled with the seed, first n."""
    eps = [json.loads(x) for x in open(path)]
    eps = [e for e in eps if e["doc"] in docs and e["split"] == "held" and not e.get("absent")]
    random.Random(seed).shuffle(eps)
    return eps[:n]


def gold_windows(e, toks, win):
    """Windows (document, w) that hold the episode's anchors."""
    out = set()
    for c0, c1 in (e.get("anchors") or [[e["fact"]["a_start"], e["fact"]["a_end"]]]):
        t0, t1 = toks.anchor(e["doc"], c0, c1)
        out |= {(e["doc"], t // win) for t in range(t0, t1)}
    return out


def hit_line(res, grp, ks=(1, 4, 16)):
    return "/".join(f"{res[(grp, k)][0] / res[(grp, k)][1]:.2f}" for k in ks) + f" ({res[(grp, ks[0])][1]})"


def groups(res):
    return ["all"] + sorted({g for g, _ in res} - {"all"})


# ---------- episodes ----------

def make_episodes(a, tok, docs, toks, log):
    types = set(a.types.split(","))
    os.makedirs(a.out, exist_ok=True)
    n_out = 0
    with open(os.path.join(a.out, "episodes.jsonl"), "w") as f:
        for line in open(a.questions):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r["doc_id"] not in docs:
                continue
            for qi, q in enumerate(r["questions"]):
                if q["type"] not in types or not q.get("anchors") or not q.get("verify", {}).get("keep", False) or \
                        len(q["answer"].split()) > a.max_answer_words:
                    continue
                c0, c1 = q["anchors"][0]
                e = {"id": f"{r['doc_id']}#q{r['group']}.{qi}", "split": "held", "level": 1, "doc": r["doc_id"],
                     "kind": r["kind"], "qtype": q["type"], "fact": {"a_start": c0, "a_end": c1}, "query": q["question"],
                     "cloze": q["question"], "answer": q["answer"], "aliases": q.get("aliases", []),
                     "anchors": q["anchors"], "mem": [{"doc": r["doc_id"], "start": max(0, c0 - 160), "end": c1 + 160,
                                                       "role": "target"}], "text": [], "teacher": [], "absent": False,
                     "ctx_dep": False, "missing": []}
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
                n_out += 1
    index = {did: [0, len(toks.ids(did))] for did in docs}
    json.dump(index, open(os.path.join(a.out, "index.json"), "w"))
    log(f"{n_out} episodes → {a.out}/episodes.jsonl; {len(index)} documents, {sum(v[1] for v in index.values())} tokens "
        f"→ {a.out}/index.json")


# ---------- search ----------

class LocalIndex:
    """Search vectors norm(rmsn(f) − μ) of a local index (store.build_index): fp16 on the GPU when they fit, otherwise streamed
    from the memmap in chunks (the index need not fit in RAM either). The research code normalized rows in batches of
    NORM_BLOCK; on CUDA any batch of 16 rows or more gives the same bits (the reduction layout changes only below that), so
    rows fetched for MaxSim are normalized in batches of at least 16."""

    def __init__(self, path, mu, dev, where="auto", log=print):
        self.mu, self.dev = mu, dev
        self.meta = json.load(open(os.path.join(path, "meta.json")))["meta"]
        self.fm = np.load(os.path.join(path, "facts.npy"), mmap_mode="r")
        self.rpos = np.load(os.path.join(path, "rows.npy"))
        self.order = sorted(self.meta, key=lambda k_: self.meta[k_][0])
        self.starts = [self.meta[k_][0] for k_ in self.order]
        self.n, d = self.fm.shape
        if where == "auto":
            where = "gpu" if self.n * d * 2 <= 0.75 * torch.cuda.mem_get_info()[0] else "stream"
        self.C = None
        if where == "gpu":
            self.C = torch.empty((self.n, d), dtype=torch.float16, device=dev)
            for i in range(0, self.n, NORM_BLOCK):
                self.C[i:i + NORM_BLOCK] = self.norm(self.fm[i:i + NORM_BLOCK])
        log(f"index {path}: {self.n} rows, " + ("on the GPU" if self.C is not None else "streamed from disk, kNN in chunks"))

    def norm(self, blk):
        x = torch.from_numpy(np.ascontiguousarray(blk)).to(self.dev)
        return F.normalize(rmsn(x.float()) - self.mu, dim=-1).half()

    def row_doc(self, r):
        return self.order[bisect.bisect_right(self.starts, r) - 1]

    def knn(self, fqs, k, log=print):
        """Top k rows per question token → [(values [Tq, k], rows [Tq, k])] per question."""
        if self.C is not None:
            return [(fq @ self.C.T).float().topk(k, dim=-1) for fq in (f_.to(self.dev) for f_ in fqs)]
        lens = [f_.shape[0] for f_ in fqs]
        FQ = torch.cat(fqs).to(self.dev)
        bv = torch.full((FQ.shape[0], k), -9.0, device=self.dev)
        bi = torch.zeros((FQ.shape[0], k), dtype=torch.long, device=self.dev)
        for i in range(0, self.n, KNN_CHUNK):
            ch = torch.cat([self.norm(self.fm[j:min(self.n, i + KNN_CHUNK, j + NORM_BLOCK)])
                            for j in range(i, min(self.n, i + KNN_CHUNK), NORM_BLOCK)])
            for q0 in range(0, FQ.shape[0], Q_BATCH):
                v_, ix_ = (FQ[q0:q0 + Q_BATCH] @ ch.T).topk(min(k, ch.shape[0]), dim=-1)
                cv = torch.cat([bv[q0:q0 + Q_BATCH], v_.float()], 1); ci = torch.cat([bi[q0:q0 + Q_BATCH], ix_ + i], 1)
                t_, j_ = cv.topk(k, dim=-1)
                bv[q0:q0 + Q_BATCH] = t_; bi[q0:q0 + Q_BATCH] = ci.gather(1, j_)
            del ch
            log(f"  kNN over chunk {i // KNN_CHUNK + 1}/{-(-self.n // KNN_CHUNK)}")
        return list(zip(torch.split(bv, lens), torch.split(bi, lens)))

    def vectors(self, spans):
        """Search vectors of row ranges [(lo, hi)] → a list of [hi − lo, d] fp16 tensors on the device."""
        if self.C is not None:
            return [self.C[lo:hi] for lo, hi in spans]
        rows = np.concatenate([np.arange(lo, hi) for lo, hi in spans])
        pad = max(0, 16 - len(rows))                                    # fewer than 16 rows may round differently
        V = self.norm(self.fm[np.concatenate([rows, np.full(pad, rows[-1])])])[:len(rows)]
        return list(torch.split(V, [hi - lo for lo, hi in spans]))


def search(a, docs, toks, eps, log):
    from .model import MemoryModel
    mm = MemoryModel(a.model, a.ckpt, a.device)
    log(f"tower {a.ckpt} (step {mm.step})")
    fqs = [mm.search_vectors(mm.token_vectors(mm.tok(e["query"], add_special_tokens=False).input_ids)).half().cpu()
           for e in eps]
    mu, dev = mm.mu, mm.dev
    del mm                                                              # the GPU goes to the index
    gc.collect(); torch.cuda.empty_cache()
    idx = LocalIndex(a.idx, mu, dev, a.index_on, log)
    first = idx.knn(fqs, a.knn, log)
    res = collections.defaultdict(lambda: [0, 0])
    dump = {}
    for e, fq, (v, ix) in zip(eps, fqs, first):
        fq = fq.to(dev)
        mx, vote = collections.defaultdict(lambda: -2.0), collections.defaultdict(float)
        for vv, ii in zip(v.flatten().tolist(), ix.flatten().tolist()):
            w = (idx.row_doc(ii), int(idx.rpos[ii]) // a.win)
            mx[w] = max(mx[w], vv); vote[w] += max(0.0, vv)
        cand = list(dict.fromkeys(sorted(mx, key=mx.get, reverse=True)[:a.cand]
                                  + sorted(vote, key=vote.get, reverse=True)[:a.cand]))
        spans = {}
        for did, w in cand:
            r0, nr = idx.meta[did]
            j0, j1 = np.searchsorted(idx.rpos[r0:r0 + nr], [w * a.win, (w + 1) * a.win])   # positions of a document are sorted
            if j1 > j0:
                spans[(did, w)] = (r0 + int(j0), r0 + int(j1))
        ms, ctr = {}, {}
        for w, V in zip(spans, idx.vectors(list(spans.values()))):
            sim = fq.float() @ V.float().T
            ms[w] = float(sim.max(1).values.mean())
            ctr[w] = int(idx.rpos[spans[w][0] + int(sim.argmax(1).float().median())])
        rk = sorted(ms, key=ms.get, reverse=True)[:16]
        gold = gold_windows(e, toks, a.win)
        hit = next((i for i, w in enumerate(rk) if w in gold), 10 ** 9)
        in_cand = any(w in gold for w in cand)
        for grp in ("all", e["kind"]):
            for kk in (1, 4, 16):
                res[(grp, kk)][0] += hit < kk; res[(grp, kk)][1] += 1
            res[(grp, "cand")][0] += in_cand; res[(grp, "cand")][1] += 1
        dump[e["id"]] = {"cloze": {"r2": [[w[0], w[1], ctr[w]] for w in rk]}, "in_cand": in_cand, "n_cand": len(cand)}
    for grp in groups(res):
        if res[(grp, 1)][1]:
            log(f"   {grp}: anchor window in top-1/4/16 {hit_line(res, grp)}; gold window among candidates "
                f"{res[(grp, 'cand')][0] / res[(grp, 'cand')][1]:.2f}")
    os.makedirs(os.path.dirname(a.dump) or ".", exist_ok=True)
    json.dump({"win": a.win, "max_doc": 10 ** 9, "eps": dump}, open(a.dump, "w"))
    log(f"dump → {a.dump}")


# ---------- read ----------

def read(a, docs, toks, eps, log):
    from .model import MemoryModel
    rd = json.load(open(a.retrieved))
    W = rd["win"]
    mm = MemoryModel(a.model, a.ckpt if a.reader == "ours" else None, a.device)
    if a.reader == "ours":
        log(f"tower and reader {a.ckpt} (step {mm.step})")

    def windows(e):
        """Top --ret-k text windows of --win tokens around the centres, skipping ones that overlap a taken window by half."""
        got = rd["eps"].get(e["id"], {}).get("cloze", {}).get(a.ret_method) or []
        out, used = [], []
        for x in got:
            if len(out) >= a.ret_k:
                break
            did, w = x[0], int(x[1])
            n_ = len(toks.ids(did))
            c = int(x[2]) if len(x) > 2 else w * W + W // 2
            lo = max(0, min(c - a.win // 2, n_ - a.win)); hi = min(n_, lo + a.win)
            if any(dd == did and min(hi, h_) - max(lo, l_) > a.win // 2 for dd, l_, h_ in used):
                continue
            used.append((did, lo, hi)); out.append((did, lo, hi, c))
        return out

    n_ok, recs, hit_any, frecs = 0, [], 0, []
    for i_e, e in enumerate(eps):
        ws = windows(e)
        gold = gold_windows(e, toks, W)
        hit_any += any((did, t // W) in gold for did, lo, hi, c in ws for t in range(lo, hi))
        if a.reader == "rag":
            out = mm.text_answer([t_ for did, lo, hi, c in ws for t_ in toks.ids(did)[lo:hi] + mm.sep], e["query"], a.max_new)
        else:
            rows, poss, off, seen = [], [], 0, set()
            for did, lo, hi, c in ws:                                   # one piece per document, around the first centre
                if did in seen:
                    continue
                seen.add(did)
                ids = toks.ids(did)
                plo = max(0, min(c - a.piece // 2, len(ids) - a.piece)); phi = min(len(ids), plo + a.piece)
                rows.append(mm.token_vectors(ids[plo:phi]))
                poss.append(torch.arange(phi - plo, device=mm.dev) + off); off += phi - plo
            text = [] if a.no_text else [t_ for did, lo, hi, c in (ws if a.win_all else ws[:1])
                                         for t_ in toks.ids(did)[lo:hi] + mm.sep]
            out, p_abs = mm.answer(rows, poss, off, text, e["query"], None, a.max_new)
            if a.abstain_curve:             # the best real answer where the model abstained: abstain tokens banned at step 1
                forced = out
                if norm_ans(ABSTAIN) in norm_ans(out):
                    forced = mm.answer(rows, poss, off, text, e["query"], 2.0, a.max_new)[0]
                frecs.append({"id": e["id"], "kind": e["kind"], "mode": a.reader, "query": e["query"], "answer": e["answer"],
                              "out": forced, "ok": float(contains(forced, e)), "p_abs": p_abs, "greedy": out})
        ok = contains(out, e)
        n_ok += ok
        recs.append({"id": e["id"], "kind": e["kind"], "mode": a.reader, "query": e["query"], "answer": e["answer"],
                     "out": out, "ok": float(ok)})
        if i_e < 4:
            log(f"  [{a.reader}] {e['query'][:90]!r} → {out[:120]!r} (answer {e['answer']!r})")
    log(f"read {a.reader} over {a.ret_method}@{a.ret_k}: {len(eps)} episodes, answer place among the windows "
        f"{hit_any / max(1, len(eps)):.3f}, answer contained {n_ok / max(1, len(eps)):.3f}")
    if a.dump_answers:
        os.makedirs(os.path.dirname(a.dump_answers) or ".", exist_ok=True)
        with open(a.dump_answers, "w") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in recs)
        if a.abstain_curve:                                 # for the judge: out is the real answer, p_abs for the threshold
            with open(a.dump_answers.replace(".jsonl", ".forced.jsonl"), "w") as f:
                f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in frecs)
        log(f"answers → {a.dump_answers}")


def main():
    ap = argparse.ArgumentParser(prog="python -m llm_memory.evaluate")
    ap.add_argument("mode", choices=["episodes", "search", "read"])
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507", help="Hugging Face id or local path of the frozen LLM")
    ap.add_argument("--ckpt", default="checkpoints/s2c.pt", help="tower + reader checkpoint")
    ap.add_argument("--docs", default="data/docs.jsonl", help="corpus: jsonl with id, title, text")
    ap.add_argument("--questions", default="data/questions-v.jsonl", help="episodes: verified generated questions")
    ap.add_argument("--types", default="single,multi"); ap.add_argument("--max-answer-words", type=int, default=8)
    ap.add_argument("--out", default="data/eval", help="episodes: output directory (episodes.jsonl, index.json)")
    ap.add_argument("--episodes", default="data/eval/episodes.jsonl")
    ap.add_argument("--n", type=int, default=400, help="episodes to evaluate (shuffled with seed 115, first n)")
    ap.add_argument("--win", type=int, default=64, help="search: window grid; read: width of each text window")
    ap.add_argument("--idx", default="data/index", help="search: index directory (facts.npy, rows.npy, meta.json)")
    ap.add_argument("--knn", type=int, default=256, help="search: nearest index rows per question token")
    ap.add_argument("--cand", type=int, default=128, help="search: candidate windows by maximum and by votes")
    ap.add_argument("--index-on", default="auto", choices=["auto", "gpu", "stream"],
                    help="search: index on the GPU or streamed from disk; auto streams it when it does not fit the GPU")
    ap.add_argument("--dump", default="data/eval/ours.json", help="search: retrieval dump")
    ap.add_argument("--retrieved", default="data/eval/ours.json", help="read: retrieval dump")
    ap.add_argument("--ret-method", default="r2", help="read: method key in the dump (r2, bm25, dense, colbert)")
    ap.add_argument("--reader", default="ours", choices=["ours", "rag"])
    ap.add_argument("--ret-k", type=int, default=4, help="read: windows (places) to read")
    ap.add_argument("--piece", type=int, default=1024, help="read, ours: memory piece around each place, tokens")
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--no-text", action="store_true", help="read, ours: memory only, no window text")
    ap.add_argument("--win-all", action="store_true", help="read, ours: text of all --ret-k windows (as RAG), not only the best")
    ap.add_argument("--abstain-curve", action="store_true", help="read, ours: also the best real answer where the model "
                    "abstained and p(abstain) at the first step → <answers>.forced.jsonl (abstain-threshold curve in one pass)")
    ap.add_argument("--dump-answers", default="", help="read: answers jsonl")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    t0 = time.time()
    log = lambda s: print(f"{s} ({time.time() - t0:.0f}s)", flush=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    docs = load_docs_by_id(a.docs)
    toks = DocTokens(tok, docs)
    if a.mode == "episodes":
        make_episodes(a, tok, docs, toks, log)
        return
    eps = load_episodes(a.episodes, docs, a.n)
    log(f"{len(eps)} episodes from {a.episodes}")
    (search if a.mode == "search" else read)(a, docs, toks, eps, log)


if __name__ == "__main__":
    main()
