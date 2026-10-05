"""Classic retrievers for side-by-side comparison in the demo: the same reader (unmodified Qwen3-4B) reads the text of the
4 best chunks.

  RAG      bge-small-en-v1.5: CLS pooling, normalized, BGE query instruction. Collection `rag_bge`, 384-d, cosine.
  ColBERT  answerai-colbert-small-v1 in the Stanford format: BERT + linear 384→96, per-token normalization; prefixes
           [unused0] (query) and [unused1] (document); the query is padded with [MASK] to 32 tokens (not attended, but
           their vectors take part in MaxSim); documents up to 300 tokens, punctuation dropped. Collection `rag_colbert`:
           native Qdrant multivectors, MaxSim on the server.
The corpus is cut into chunks of `size` Qwen tokens, consecutive within each document.
"""
import os
import string

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoModel, AutoTokenizer

BGE_Q = "Represent this sentence for searching relevant passages: "


def _local(path):
    if os.path.isdir(path):
        return path
    from huggingface_hub import snapshot_download
    return snapshot_download(path)


class Bge:
    def __init__(self, path, dev="cuda"):
        path = _local(path)
        self.tok = AutoTokenizer.from_pretrained(path)
        self.m = AutoModel.from_pretrained(path).eval().to(dev)
        self.dev = dev

    @torch.no_grad()
    def enc(self, texts, query=False, bs=128):
        out = []
        for i in range(0, len(texts), bs):
            b = self.tok([(BGE_Q + t) if query else t for t in texts[i:i + bs]], padding=True, truncation=True,
                         max_length=512, return_tensors="pt").to(self.dev)
            with torch.autocast("cuda", dtype=torch.float16):
                h = self.m(**b).last_hidden_state[:, 0]
            out.append(F.normalize(h.float(), dim=-1).cpu())
        return torch.cat(out).numpy()


class Colbert:
    def __init__(self, path, dev="cuda"):
        path = _local(path)
        self.tok = AutoTokenizer.from_pretrained(path)
        self.m = AutoModel.from_pretrained(path).eval().to(dev)
        w = {k: v for k, v in load_file(os.path.join(path, "model.safetensors")).items() if k.startswith("linear")}
        self.lin = w["linear.weight"].to(dev).float()                      # [96, 384]
        self.dev = dev
        self.mask_id = self.tok.mask_token_id
        self.punct = {i for i, t in enumerate(self.tok.convert_ids_to_tokens(range(self.tok.vocab_size)))
                      if t and all(c in string.punctuation for c in t)}

    @torch.no_grad()
    def _run(self, ids, att):
        with torch.autocast("cuda", dtype=torch.float16):
            h = self.m(input_ids=ids, attention_mask=att).last_hidden_state
        return F.normalize(h.float() @ self.lin.T, dim=-1)

    def query(self, q, qlen=32):
        ids = [self.tok.cls_token_id, 1] + self.tok(q, add_special_tokens=False).input_ids[:qlen - 3] + [self.tok.sep_token_id]
        att = [1] * len(ids) + [0] * (qlen - len(ids))
        ids = ids + [self.mask_id] * (qlen - len(ids))
        return self._run(torch.tensor([ids], device=self.dev), torch.tensor([att], device=self.dev))[0].cpu().numpy()

    def docs(self, texts, dlen=300, bs=64):
        out = []
        for i in range(0, len(texts), bs):
            seqs = [[self.tok.cls_token_id, 2] + self.tok(t, add_special_tokens=False).input_ids[:dlen - 3] + [self.tok.sep_token_id]
                    for t in texts[i:i + bs]]
            L = max(map(len, seqs))
            ids = torch.tensor([s + [0] * (L - len(s)) for s in seqs], device=self.dev)
            att = torch.tensor([[1] * len(s) + [0] * (L - len(s)) for s in seqs], device=self.dev)
            E = self._run(ids, att).cpu().numpy()
            for j, s in enumerate(seqs):
                keep = [k for k, t in enumerate(s) if t not in self.punct]
                out.append(E[j, keep])
        return out


def chunks_of(docs, tok, size):
    """Chunks of `size` Qwen tokens, consecutive within each document → [(document number, start, end)]."""
    out = []
    for i, dd in enumerate(docs):
        n = len(tok(dd["text"], add_special_tokens=False).input_ids)
        out += [(i, s, min(n, s + size)) for s in range(0, n, size)]
    return out


def build(qc, docs, qwen_tok, bge_path, colbert_path, size=256, log=print):
    """Chunk the corpus and fill the `rag_bge` and `rag_colbert` collections; returns the chunk list."""
    from qdrant_client import models as qm
    ch = chunks_of(docs, qwen_tok, size)
    texts = [qwen_tok.decode(qwen_tok(docs[di]["text"], add_special_tokens=False).input_ids[s:e]) for di, s, e in ch]
    log(f"{len(ch)} chunks of {size} tokens")
    pay = [{"d": di, "s": s, "e": e} for di, s, e in ch]

    bge = Bge(bge_path)
    V = bge.enc(texts)
    if qc.collection_exists("rag_bge"):
        qc.delete_collection("rag_bge")
    qc.create_collection("rag_bge", vectors_config=qm.VectorParams(size=V.shape[1], distance=qm.Distance.COSINE))
    qc.upload_collection("rag_bge", vectors=V, payload=pay, ids=list(range(len(ch))), batch_size=256, wait=True)
    log(f"rag_bge: {V.shape}")
    del bge
    torch.cuda.empty_cache()

    cb = Colbert(colbert_path)
    if qc.collection_exists("rag_colbert"):
        qc.delete_collection("rag_colbert")
    qc.create_collection("rag_colbert", vectors_config=qm.VectorParams(
        size=96, distance=qm.Distance.COSINE, multivector_config=qm.MultiVectorConfig(comparator=qm.MultiVectorComparator.MAX_SIM),
        hnsw_config=qm.HnswConfigDiff(m=0)))
    B = 2048
    for i in range(0, len(texts), B):
        E = cb.docs(texts[i:i + B])
        qc.upload_points("rag_colbert", points=[qm.PointStruct(id=i + j, vector=e.tolist(), payload=pay[i + j])
                                                for j, e in enumerate(E)], batch_size=64, wait=True)
        log(f"  rag_colbert: {i + len(E)}/{len(texts)}")
    return ch


class TextRetrievers:
    """Search in the baseline collections: 4 best chunks → [(document, start, end, score)]."""

    def __init__(self, qc, bge_path, colbert_path, k=4):
        self.qc, self.bge_path, self.colbert_path, self.k = qc, bge_path, colbert_path, k
        self.models = {}

    def model(self, mode):
        if mode not in self.models:
            self.models[mode] = Bge(self.bge_path) if mode == "rag" else Colbert(self.colbert_path)
        return self.models[mode]

    def search(self, mode, question):
        r_ = self.model(mode)
        if mode == "rag":
            q = r_.enc([question], query=True)[0].tolist()
            res = self.qc.query_points("rag_bge", query=q, limit=self.k, with_payload=True).points
        else:
            res = self.qc.query_points("rag_colbert", query=r_.query(question).tolist(), limit=self.k, with_payload=True).points
        return [(p_.payload["d"], p_.payload["s"], p_.payload["e"], p_.score) for p_ in res]
