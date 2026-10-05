"""Corpus index (one vector per token), its upload to Qdrant, search, and fetching memory for the reader.

Qdrant point = one token of the corpus.
  vector   ŷ = norm(rmsn(f) − μ), float16, Dot: exactly the search measure of the tower;
  payload  d (document number), p (token position), r = rms(f), n = |rmsn(f) − μ|.
Hence f = r · (n · ŷ + μ) is restored exactly (up to fp16), so the reader's memory can come straight from Qdrant
without re-encoding the document.
"""
import collections
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from .model import rmsn


def load_docs(path):
    """Corpus jsonl (id, title, text) sorted by id: the document number d is the position in this list."""
    docs = [json.loads(x) for x in open(path)]
    docs.sort(key=lambda d_: d_["id"])
    return docs


def build_index(mm, docs, out, enc_max=16384, log=print):
    """Encode every document whole (longer ones in pieces of enc_max tokens) → <out>/facts.npy (f, fp16, one row per token),
    rows.npy (token position), meta.json (document id → [first row, number of rows])."""
    os.makedirs(out, exist_ok=True)
    ids = {dd["id"]: mm.tok(dd["text"], add_special_tokens=False).input_ids for dd in docs}
    order = sorted(ids)
    total = sum(len(ids[did]) for did in order)
    FM = np.lib.format.open_memmap(os.path.join(out, "facts.npy"), mode="w+", dtype=np.float16, shape=(total, mm.d))
    rows_pos, meta, n_rows = [], {}, 0
    for i, did in enumerate(order):
        r0, t = n_rows, ids[did]
        for lo in range(0, len(t), enc_max):
            f = mm.token_vectors(t[lo:lo + enc_max])
            FM[n_rows:n_rows + f.shape[0]] = f.half().cpu().numpy()
            rows_pos.append(np.arange(f.shape[0], dtype=np.int32) + lo)
            n_rows += f.shape[0]
        meta[did] = [r0, n_rows - r0]
        if i % 200 == 0:
            log(f"  documents {i}/{len(order)}, rows {n_rows}")
    assert n_rows == total, (n_rows, total)
    FM.flush(); del FM
    np.save(os.path.join(out, "rows.npy"), np.concatenate(rows_pos))
    json.dump({"meta": meta, "k": 1, "win": 64}, open(os.path.join(out, "meta.json"), "w"))
    log(f"index: {n_rows} rows → {out}/facts.npy")


def upload(qc, collection, idx, docs, mu, meta_path, batch=256, parallel=6, log=print):
    """Index → Qdrant collection, one point per token; writes the demo meta (titles, rows of each document)."""
    from qdrant_client import models as qm
    fm = np.load(os.path.join(idx, "facts.npy"), mmap_mode="r")
    mi = json.load(open(os.path.join(idx, "meta.json")))["meta"]
    rpos = np.load(os.path.join(idx, "rows.npy"))
    id2i = {dd["id"]: i for i, dd in enumerate(docs)}
    N = fm.shape[0]
    rdoc = np.empty(N, dtype=np.int32)
    meta = {"docs": [{"id": dd["id"], "title": dd.get("title", dd["id"])} for dd in docs], "rows": [[0, 0]] * len(docs)}
    for did, (r0, n_) in mi.items():
        rdoc[r0:r0 + n_] = id2i[did]; meta["rows"][id2i[did]] = [r0, n_]
    os.makedirs(os.path.dirname(meta_path) or ".", exist_ok=True)
    json.dump(meta, open(meta_path, "w"))
    if qc.collection_exists(collection):
        qc.delete_collection(collection)
    qc.create_collection(
        collection,
        vectors_config=qm.VectorParams(size=fm.shape[1], distance=qm.Distance.DOT, datatype=qm.Datatype.FLOAT16, on_disk=True),
        quantization_config=qm.BinaryQuantization(binary=qm.BinaryQuantizationConfig(always_ram=True)),
        hnsw_config=qm.HnswConfigDiff(m=16, ef_construct=100),
        optimizers_config=qm.OptimizersConfigDiff(indexing_threshold=0),     # build the graph after the upload
        on_disk_payload=True)
    mu_np = mu.float().cpu().numpy()
    CH = 65536
    for i in range(0, N, CH):                                                # 2D array per call: the client takes ndarrays whole
        f = np.asarray(fm[i:i + CH], dtype=np.float32)
        r = np.sqrt((f * f).mean(-1)).clip(min=1e-6)
        y = f / r[:, None] - mu_np
        n = np.linalg.norm(y, axis=-1).clip(min=1e-6)
        pay = [{"d": int(rdoc[i + j]), "p": int(rpos[i + j]), "r": float(r[j]), "n": float(n[j])} for j in range(len(r))]
        qc.upload_collection(collection, vectors=y / n[:, None], payload=pay, ids=list(range(i, i + len(r))),
                             batch_size=batch, parallel=parallel, wait=False)
        if i % (CH * 8) == 0:
            log(f"  points {i + len(r)}/{N}")
    qc.update_collection(collection, optimizers_config=qm.OptimizersConfigDiff(indexing_threshold=20000))
    log(f"{N} points → {collection}; the HNSW graph is being built in the background")


class MemoryStore:
    """Search over the token vectors in Qdrant and memory for the reader.

    kNN always runs in Qdrant. Vectors of candidate windows and of memory pieces come either from Qdrant (retrieve by id)
    or, with local_idx, from a local memmap of the same index: fetching ~8K vectors by id takes seconds today."""

    def __init__(self, url, collection, meta_path, mu, device="cuda", local_idx=None):
        from qdrant_client import QdrantClient
        self.qc = QdrantClient(url=url, prefer_grpc=True, timeout=600)
        self.qc_rest = QdrantClient(url=url, prefer_grpc=False, timeout=600)   # retrieve by id: REST is ~2x faster than gRPC
        self.collection, self.mu, self.dev = collection, mu, device
        meta = json.load(open(meta_path))
        self.titles = [x["title"] for x in meta["docs"]]
        self.rows = meta["rows"]
        self.fm = np.load(os.path.join(local_idx, "facts.npy"), mmap_mode="r") if local_idx else None
        self.timing = {}

    def _fetch(self, ids, payload):
        pts = self.qc_rest.retrieve(self.collection, ids=ids, with_vectors=True, with_payload=payload)
        pts.sort(key=lambda p_: p_.id)
        return pts

    def _local(self, ids):
        return torch.from_numpy(np.asarray(self.fm[np.asarray(ids)], dtype=np.float32)).to(self.dev)

    def search(self, qv, knn=256, cand=128, win=64, ret_k=4, ef=256):
        """Question token vectors qv [Tq, d] (unit) → per-token kNN in Qdrant → 64-token windows (max score and votes)
        → MaxSim over each window's vectors → ret_k best places [(doc, lo, hi, center, score)]."""
        from qdrant_client import models as qm
        reqs = [qm.QueryRequest(query=v.tolist(), limit=knn, with_payload=["d", "p"],
                                params=qm.SearchParams(hnsw_ef=ef, quantization=qm.QuantizationSearchParams(rescore=True,
                                                                                                           oversampling=2.0)))
                for v in qv]
        tq = time.time()
        res = self.qc.query_batch_points(self.collection, requests=reqs)
        self.timing["kNN in Qdrant"] = time.time() - tq
        mx, vote = collections.defaultdict(lambda: -2.0), collections.defaultdict(float)
        for r_ in res:
            for p_ in r_.points:
                w = (p_.payload["d"], p_.payload["p"] // win)
                mx[w] = max(mx[w], p_.score); vote[w] += max(0.0, p_.score)
        cands = list(dict.fromkeys(sorted(mx, key=mx.get, reverse=True)[:cand] + sorted(vote, key=vote.get, reverse=True)[:cand]))
        want = {}
        for di, w in cands:
            r0, n_ = self.rows[di]
            for p in range(w * win, min(n_, (w + 1) * win)):
                want[r0 + p] = (di, w)
        tq = time.time()
        ms, ctr = {}, {}
        if self.fm is not None:
            ids_all = sorted(want)
            Mall = F.normalize(rmsn(self._local(ids_all)) - self.mu, dim=-1)
            by_w = collections.defaultdict(list)
            for j, i_ in enumerate(ids_all):
                by_w[want[i_]].append(j)
            self.timing["window vectors"] = time.time() - tq
            for w, js in by_w.items():
                sim = qv @ Mall[js].T
                ms[w] = float(sim.max(1).values.mean())
                ctr[w] = ids_all[js[int(sim.argmax(1).float().median())]] - self.rows[w[0]][0]
        else:
            pts = self._fetch(list(want), ["p"])
            self.timing["window vectors"] = time.time() - tq
            by_w = collections.defaultdict(list)
            for p_ in pts:
                by_w[want[p_.id]].append(p_)
            for w, ps in by_w.items():
                M = torch.tensor(np.array([p_.vector for p_ in ps], dtype=np.float32), device=self.dev)
                sim = qv @ M.T
                ms[w] = float(sim.max(1).values.mean())
                ctr[w] = ps[int(sim.argmax(1).float().median())].payload["p"]
        out, used = [], []
        for di, w in sorted(ms, key=ms.get, reverse=True):
            if len(out) >= ret_k:
                break
            n_ = self.rows[di][1]; c = ctr[(di, w)]
            lo = max(0, min(c - win // 2, n_ - win)); hi = min(n_, lo + win)
            if any(dd == di and min(hi, h_) - max(lo, l_) > win // 2 for dd, l_, h_ in used):
                continue
            used.append((di, lo, hi)); out.append((di, lo, hi, c, ms[(di, w)]))
        return out

    @torch.no_grad()
    def memory(self, places, piece=1024):
        """Vectors f of a piece of `piece` tokens around each found place (one piece per document) → (rows, positions, span).
        From Qdrant: f = r · (n · ŷ + μ)."""
        mem, poss, off, seen = [], [], 0, set()
        self.pieces = []                                            # (document, start, end) of each piece, for display
        tq = time.time()
        for di, lo, hi, c, _ in places:
            if di in seen:
                continue
            seen.add(di)
            r0, n_ = self.rows[di]
            plo = max(0, min(c - piece // 2, n_ - piece)); phi = min(n_, plo + piece)
            self.pieces.append((di, plo, phi))
            if self.fm is not None:
                mem.append(self._local(np.arange(r0 + plo, r0 + phi)))
            else:
                pts = self._fetch(list(range(r0 + plo, r0 + phi)), ["r", "n"])
                yh = torch.tensor(np.array([p_.vector for p_ in pts], dtype=np.float32), device=self.dev)
                rr = torch.tensor([p_.payload["r"] for p_ in pts], device=self.dev)
                nn_ = torch.tensor([p_.payload["n"] for p_ in pts], device=self.dev)
                mem.append(rr[:, None] * (nn_[:, None] * yh + self.mu))
            poss.append(torch.arange(phi - plo, device=self.dev) + off)
            off += phi - plo
        self.timing["memory"] = time.time() - tq
        return mem, poss, off
