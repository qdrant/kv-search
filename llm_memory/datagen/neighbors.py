"""Topic neighbors of every document: hard distractors for training episodes.

Document embedding: the mean over the first --max-tok tokens (title + text, the first token excluded as an attention sink) of
the RMS-normalized hidden state x/rms(x) after layer --layer of a small frozen LLM. Embeddings are centred over the whole corpus
and normalized; neighbors are the top --k by cosine among all documents of any kind. Output JSON: {ids, nbr, sim, args}.
The released data/src123/neighbors.json was computed with Qwen2.5-1.5B-Instruct, layer 20, 512 tokens, k = 8.

    python -m llm_memory.datagen.neighbors --docs data/train/docs.jsonl --out data/train/neighbors.json
"""
import argparse
import json
import random
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--docs", default="data/train/docs.jsonl"); ap.add_argument("--out", default="data/train/neighbors.json")
    ap.add_argument("--layer", type=int, default=20); ap.add_argument("--max-tok", type=int, default=512)
    ap.add_argument("--k", type=int, default=8); ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    t0 = time.time()
    log = lambda s: print(f"{s} ({time.time() - t0:.0f}s)", flush=True)
    docs = [json.loads(x) for x in open(a.docs)]
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).eval().to(a.device)
    emb = torch.zeros(len(docs), model.config.hidden_size)
    order = sorted(range(len(docs)), key=lambda i: len(docs[i]["text"]))                      # less padding
    with torch.no_grad():
        for b in range(0, len(order), a.bs):
            idx = order[b:b + a.bs]
            enc = tok([docs[i]["title"] + "\n" + docs[i]["text"][:6000] for i in idx], truncation=True,
                      max_length=a.max_tok, padding=True, return_tensors="pt").to(a.device)
            hs = model.model(**enc, output_hidden_states=True).hidden_states[a.layer].float()
            hs = hs / hs.pow(2).mean(-1, keepdim=True).sqrt().clamp(min=1e-6)
            m = enc["attention_mask"][..., None].float()
            m[:, 0] = 0.0                                                                       # the first token is a sink
            emb[idx] = ((hs * m).sum(1) / m.sum(1).clamp(min=1)).cpu()
            if (b // a.bs) % 50 == 0:
                log(f"  documents {b + len(idx)}/{len(docs)}")
    emb = emb - emb.mean(0)
    emb = torch.nn.functional.normalize(emb, dim=-1)
    nbr, sim = [], []
    for b in range(0, len(docs), 1024):
        s = emb[b:b + 1024] @ emb.T
        s[torch.arange(s.shape[0]), torch.arange(b, b + s.shape[0])] = -2.0
        v, i = s.topk(a.k, -1)
        nbr += i.tolist(); sim += v.tolist()
    ids = [d["id"] for d in docs]
    json.dump({"ids": ids, "nbr": [[ids[j] for j in r] for r in nbr], "sim": [[round(x, 4) for x in r] for r in sim],
               "args": vars(a)}, open(a.out, "w"))
    sims = torch.tensor(sim)
    log(f"similarity to the 1st neighbor: median {sims[:, 0].median():.3f}, to the {a.k}th: {sims[:, -1].median():.3f}")
    same = sum(docs[nbr[i][0]]["kind"] == docs[i]["kind"] for i in range(len(docs))) / len(docs)
    log(f"first neighbors of the same kind: {same:.2f}")
    for i in random.Random(0).sample(range(len(docs)), min(8, len(docs))):
        print(f"  {docs[i]['kind']}: {docs[i]['title'][:60]!r} → " + "; ".join(
            f"{docs[j]['kind']}: {docs[j]['title'][:40]!r} {sim[i][r]:.2f}" for r, j in enumerate(nbr[i][:3])))


if __name__ == "__main__":
    main()
