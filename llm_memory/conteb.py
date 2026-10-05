"""ConTEB: a public retrieval benchmark where a chunk can only be found with the context of its document.

Ported from the nvme-llm research script exp127_conteb.py (ConTEB: Conti et al. 2025, arXiv 2505.24782). Sets whose size on
Hugging Face matches table 1 of the paper: insurance, covid-qa, geography, football (in geography and football the names in
chunks are replaced by pronouns). A document is its chunks in order; every chunk of a set is scored for every query of the
set, exactly (no kNN preselection).

  bm25        terms \\w+, k1 1.2, b 0.75 (a check of the setup against the BM25 row of the paper);
  bge         bge-small-en-v1.5, one vector per chunk (llm_memory.baselines.Bge);
  colbert     answerai-colbert-small-v1 (llm_memory.baselines.Colbert), MaxSim over the chunk's tokens;
  ours        the tower encodes each document whole (in pieces of --enc-max tokens), MaxSim over the chunk's tokens;
  ours-chunk  the same tower encodes each chunk alone, without its document.
Measure: nDCG@10 (as in the paper), R@1, R@10; with <out>/overlap.json also on the queries whose documents are not in the
tower's training texts. The published table was computed with the stage-2b checkpoint (the tower before the last training
stage, s2c, which the demo uses).

Modes: prep downloads the sets (datasets-server JSON API, no pyarrow needed); overlap marks documents found in training
texts (at least 5 shared 10-word shingles); eval scores the methods → <out>/results-<sets><tag>.json.

    python -m llm_memory.conteb prep
    python -m llm_memory.conteb eval --ckpt checkpoints/s2c.pt --sets insurance
"""
import argparse
import collections
import json
import math
import os
import re
import time
import urllib.request

import numpy as np
import torch

API = "https://datasets-server.huggingface.co/rows"


def fetch(ds, config, path):
    rows, off = [], 0
    while True:
        u = f"{API}?dataset=illuin-conteb/{ds}&config={config}&split=train&offset={off}&length=100"
        for t in range(6):
            try:
                d = json.load(urllib.request.urlopen(u, timeout=60))
                break
            except Exception:
                if t == 5:
                    raise
                time.sleep(5 * (t + 1))
        rows += [r["row"] for r in d["rows"]]
        off += 100
        if off >= d["num_rows_total"]:
            break
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


def load_set(root, ds):
    """Chunks by document (in order) and queries → their relevant chunks."""
    docs = collections.OrderedDict()
    for line in open(os.path.join(root, ds, "documents.jsonl")):
        r = json.loads(line)
        did, n = r["chunk_id"].rsplit("_", 1)
        docs.setdefault(did, []).append((int(n), r["chunk_id"], r["chunk"], r.get("og_chunk") or r["chunk"]))
    for did in docs:
        docs[did].sort()
    qrels = collections.OrderedDict()
    for line in open(os.path.join(root, ds, "queries.jsonl")):
        r = json.loads(line)
        qrels.setdefault(r["query"], set()).add(r["chunk_id"])
    return docs, qrels


def shingles(text, n=10):
    w = re.findall(r"\w+", text.lower())
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def bm25_rank(texts, queries, k1=1.2, b=0.75, top=10):
    tz = lambda s: re.findall(r"\w+", s.lower())
    dt = [tz(t) for t in texts]
    N, avg = len(dt), sum(map(len, dt)) / len(dt)
    df = collections.Counter(w for x in dt for w in set(x))
    idf = {w: math.log(1 + (N - n + 0.5) / (n + 0.5)) for w, n in df.items()}
    inv = collections.defaultdict(list)
    for i, x in enumerate(dt):
        for w, c in collections.Counter(x).items():
            inv[w].append((i, c * (k1 + 1) / (c + k1 * (1 - b + b * len(x) / avg))))
    out = []
    for q in queries:
        sc = collections.defaultdict(float)
        for w in set(tz(q)):
            for i, v in inv.get(w, ()):
                sc[i] += idf[w] * v
        out.append(sorted(sc, key=sc.get, reverse=True)[:top])
    return out


def maxsim_rank(fqs, C, t2c, n_chunks, dev, top=10):
    """Exact MaxSim over all chunks: C — token vectors [N, d], t2c — chunk number of each token [N]."""
    C, t2c = C.to(dev), t2c.to(dev)
    out = []
    for fq in fqs:
        S = (fq.to(dev).to(C.dtype) @ C.T).float()
        M = torch.full((S.shape[0], n_chunks), float("-inf"), device=dev)
        M.scatter_reduce_(1, t2c.expand(S.shape[0], -1), S, reduce="amax")
        out.append(M.mean(0).topk(min(top, n_chunks)).indices.tolist())
    return out


def metrics(ranks, golds, cids):
    nd, r1, r10 = [], [], []
    for rk, g in zip(ranks, golds):
        rel = [1.0 if cids[i] in g else 0.0 for i in rk[:10]]
        dcg = sum(r / math.log2(j + 2) for j, r in enumerate(rel))
        nd.append(dcg / sum(1 / math.log2(j + 2) for j in range(min(len(g), 10))))
        r1.append(rel[0] if rel else 0.0)
        r10.append(sum(rel) / len(g))
    m = lambda x: 100 * sum(x) / max(1, len(x))
    return {"ndcg10": m(nd), "r1": m(r1), "r10": m(r10), "n": len(nd)}


def overlap(a, data, log):
    owner = collections.defaultdict(set)                                # shingle → ConTEB documents
    for ds, (docs, _) in data.items():
        for did, parts in docs.items():
            for s in shingles(" ".join(p[3] for p in parts)):
                owner[s].add((ds, did))
    hits = collections.Counter()
    for path in a.train_docs.split(","):
        for line in open(path):
            for s in shingles(json.loads(line)["text"]) & owner.keys():
                for k in owner[s]:
                    hits[k] += 1
    res = {}
    for ds, (docs, _) in data.items():
        res[ds] = sorted(did for did in docs if hits[(ds, did)] >= 5)
        log(f"{ds}: {len(res[ds])} of {len(docs)} documents found in the training texts (≥5 shared 10-word shingles)")
    json.dump(res, open(os.path.join(a.out, "overlap.json"), "w"))


def evaluate(a, data, log):
    ovl_path = os.path.join(a.out, "overlap.json")
    ovl = json.load(open(ovl_path)) if os.path.exists(ovl_path) else {}
    methods, dev = a.methods.split(","), a.device
    flat = {}                                                           # set → (chunk ids, texts, queries, relevant chunks)
    for ds, (docs, qrels) in data.items():
        cids = [p[1] for parts in docs.values() for p in parts]
        flat[ds] = (cids, [p[2] for parts in docs.values() for p in parts], list(qrels), list(qrels.values()))
        log(f"{ds}: {len(docs)} documents, {len(cids)} chunks, {len(qrels)} queries")
    ranks = {}
    if "bm25" in methods:
        for ds, (cids, texts, qs, _) in flat.items():
            ranks[(ds, "bm25")] = bm25_rank(texts, qs)
    if "bge" in methods:
        from .baselines import Bge
        B = Bge(a.bge, dev)
        for ds, (cids, texts, qs, _) in flat.items():
            D, Q = torch.from_numpy(B.enc(texts)), torch.from_numpy(B.enc(qs, query=True))
            ranks[(ds, "bge")] = (Q @ D.T).topk(min(10, len(cids)), dim=-1).indices.tolist()
        del B
    if "colbert" in methods:
        from .baselines import Colbert
        Cb = Colbert(a.colbert, dev)
        for ds, (cids, texts, qs, _) in flat.items():
            E = Cb.docs(texts)
            t2c = torch.cat([torch.full((len(e),), i, dtype=torch.long) for i, e in enumerate(E)])
            ranks[(ds, "colbert")] = maxsim_rank([torch.from_numpy(Cb.query(q)) for q in qs],
                                                 torch.from_numpy(np.concatenate(E)), t2c, len(cids), dev)
        del Cb
    torch.cuda.empty_cache()
    if "ours" in methods or "ours-chunk" in methods:
        from .model import MemoryModel
        mm = MemoryModel(a.model, a.ckpt, dev)
        tok = mm.tok
        log(f"tower {a.ckpt} (step {mm.step})")
        vecs = lambda ids: mm.search_vectors(mm.token_vectors(ids)).half()
        for ds, (docs, qrels) in data.items():
            fqs = [vecs(tok(q, add_special_tokens=False).input_ids).cpu() for q in qrels]
            log(f"{ds}: {len(fqs)} query vectors")
            pieces, sp = [], []                                         # document tokens and chunk bounds
            for parts in docs.values():
                ids, prev_ws = [], True
                for _, _, text, _ in parts:
                    if not prev_ws and text[:1] not in (" ", "\n", "\t"):
                        text = " " + text                               # glue chunks without merging words
                    t = tok(text, add_special_tokens=False).input_ids
                    sp.append((len(pieces), len(ids), len(ids) + len(t)))
                    ids += t
                    prev_ws = text[-1:] in (" ", "\n", "\t")
                pieces.append(ids)
            t2c = torch.cat([torch.full((e - s,), i, dtype=torch.long) for i, (_, s, e) in enumerate(sp)])
            for var in ("ours", "ours-chunk"):
                if var not in methods:
                    continue
                if var == "ours":                                       # whole document (in pieces of --enc-max)
                    C = torch.cat([vecs(ids[lo:lo + a.enc_max]).cpu() for ids in pieces
                                   for lo in range(0, len(ids), a.enc_max)])
                else:                                                   # each chunk alone: no document context
                    C = torch.cat([vecs(pieces[di][s:e]).cpu() for di, s, e in sp if e > s])
                assert C.shape[0] == t2c.shape[0], (C.shape, t2c.shape)
                log(f"{ds}/{var}: {C.shape[0]} tokens")
                ranks[(ds, var)] = maxsim_rank(fqs, C, t2c, len(sp), dev)
                del C
                torch.cuda.empty_cache()
    res = {}
    for ds, (cids, _, qs, golds) in flat.items():
        bad = set(ovl.get(ds, []))
        clean = [i for i, g in enumerate(golds) if not any(c.rsplit("_", 1)[0] in bad for c in g)]
        for m in methods:
            if (ds, m) not in ranks:
                continue
            rk = ranks[(ds, m)]
            r = {"all": metrics(rk, golds, cids)}
            if ovl:
                r["clean"] = metrics([rk[i] for i in clean], [golds[i] for i in clean], cids)
            res[f"{ds}/{m}"] = r
            log(f"{ds:10s} {m:11s} nDCG@10 {r['all']['ndcg10']:5.1f}  R@1 {r['all']['r1']:5.1f}  R@10 {r['all']['r10']:5.1f}"
                f"  ({r['all']['n']})" + (f" | without training overlap: nDCG@10 {r['clean']['ndcg10']:5.1f} "
                                          f"({r['clean']['n']})" if ovl else ""))
    path = os.path.join(a.out, f"results-{'-'.join(data)}{a.tag}.json")
    json.dump(res, open(path, "w"), indent=1)
    log(f"results → {path}")


def main():
    ap = argparse.ArgumentParser(prog="python -m llm_memory.conteb")
    ap.add_argument("mode", choices=["prep", "overlap", "eval"])
    ap.add_argument("--sets", default="insurance,covid-qa,geography,football")
    ap.add_argument("--root", default="data/conteb", help="downloaded sets: <root>/<set>/{documents,queries}.jsonl")
    ap.add_argument("--out", default="data/conteb", help="overlap.json and results-*.json")
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507"); ap.add_argument("--ckpt", default="checkpoints/s2c.pt")
    ap.add_argument("--train-docs", default="", help="overlap: training corpora of the tower (jsonl with text), comma-separated")
    ap.add_argument("--methods", default="bm25,bge,colbert,ours,ours-chunk")
    ap.add_argument("--bge", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--colbert", default="answerdotai/answerai-colbert-small-v1")
    ap.add_argument("--enc-max", type=int, default=16384); ap.add_argument("--device", default="cuda")
    ap.add_argument("--tag", default="", help="suffix of the results file name")
    a = ap.parse_args()
    t0 = time.time()
    log = lambda s: print(f"{s} ({time.time() - t0:.0f}s)", flush=True)
    sets = a.sets.split(",")
    os.makedirs(a.out, exist_ok=True)
    if a.mode == "prep":
        for ds in sets:
            os.makedirs(os.path.join(a.root, ds), exist_ok=True)
            for cfg, fn in (("documents", "documents.jsonl"), ("queries", "queries.jsonl")):
                log(f"{ds}/{cfg}: {fetch(ds, cfg, os.path.join(a.root, ds, fn))} rows")
        return
    if a.mode == "overlap" and not a.train_docs:
        ap.error("overlap needs --train-docs")
    data = {ds: load_set(a.root, ds) for ds in sets}
    (overlap if a.mode == "overlap" else evaluate)(a, data, log)


if __name__ == "__main__":
    main()
