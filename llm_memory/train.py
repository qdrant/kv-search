"""Training: first the tower learns to search, then the tower and the reader R learn to read memory.

Stage `search` (only the tower learns):
  episode = a piece of the target document around a fact (or question) anchor + pieces of topic-neighbor documents;
  queries = the fact and other facts/questions inside the target piece, encoded by the same model and tower;
  loss = InfoNCE over MaxSim (question tokens vs 64-token windows of all pieces; positives = windows with the anchor),
         with windows of recent episodes as extra cross-document negatives (--bank), a leash ‖f − s‖² that keeps f near
         the reader's layer input s, and an auxiliary significance gate (BCE to fact anchors + an open-share budget).
Stage `reader` (tower, reader R and the mixing scalar learn together): the same losses plus
  cross-entropy of the answer through layers cut..35 and a match of the layer-cut input on the question and answer
  tokens to a teacher (the same frozen model reading all pieces as text).
  Episode kinds: cite (text window with the answer), know (window without it, the answer is only in memory), far
  (questions that need distant parts of the document), absent (only distractors; the answer is "not in the text").
  --app builds memory exactly as after retrieval: 4 equal pieces, the target one and distractors around the window of a
  neighbor document most similar to the question (current tower, MaxSim), ordered by that score; text = windows around
  the centres of all pieces.

Evaluation (--eval-every, held-out documents md5(id) % 1000 < 50): search — the anchor window is the top-1 among the
episode's windows; reading — the reference is contained in the answer, and the NLL of the reference, per kind; controls
(no memory; the unmodified model on the text window and on all pieces as text).

    python -m llm_memory.train --stage search --out runs/search ...
    python -m llm_memory.train --stage reader --init runs/search/last.pt --out runs/reader ...
"""
import argparse
import bisect
import collections
import hashlib
import json
import math
import os
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .model import ABSTAIN, MemoryModel, rmsn


def norm_ans(s):
    """Answer normalization for 'contained' checks: lowercase, no punctuation, no articles."""
    import re
    import string
    s = "".join(ch for ch in s.lower() if ch not in set(string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", s).split())


def held_out(doc_id):
    return int(hashlib.md5(doc_id.encode()).hexdigest(), 16) % 1000 < 50


class TrainData:
    """Corpus, synthetic facts (fill-in-the-blank items and anchors) and verified questions, split train/held."""

    def __init__(self, tok, docs_path, facts_paths, questions_paths="", neighbors_path=""):
        self.tok = tok
        self.docs = {}
        for line in open(docs_path):
            dd = json.loads(line)
            self.docs[dd["id"]] = dd
        self.by_src = collections.defaultdict(list)
        for did, dd in self.docs.items():
            self.by_src[self.src_key(did)].append(did)
        self.nbr = {}
        if neighbors_path:
            j_ = json.load(open(neighbors_path))
            self.nbr = {i_: [x for x in n_ if x in self.docs] for i_, n_ in zip(j_["ids"], j_["nbr"])}
        self.tcache = collections.OrderedDict()
        self.cloze, self.nat, self.far = ({"train": [], "held": []} for _ in range(3))
        self.anchors_of, self.cloze_by_doc, self.q_by_doc = {}, collections.defaultdict(list), collections.defaultdict(list)
        self._load_facts(facts_paths)
        self._load_questions(questions_paths)

    def src_key(self, did):
        dd = self.docs[did]
        return dd["source"] if dd["kind"] in ("code", "docs") else dd["kind"]

    def doc_tok(self, did):
        """(token ids, char start of each token, char end of each token) of a document, cached."""
        if did not in self.tcache:
            if len(self.tcache) > 3000:
                self.tcache.popitem(last=False)
            enc = self.tok(self.docs[did]["text"], add_special_tokens=False, return_offsets_mapping=True)
            self.tcache[did] = (enc["input_ids"], [o[0] for o in enc["offset_mapping"]], [o[1] for o in enc["offset_mapping"]])
        return self.tcache[did]

    def tok_span(self, did, c0, c1):
        _, st, en = self.doc_tok(did)
        i0 = bisect.bisect_left(st, c0)
        return i0, max(i0 + 1, bisect.bisect_right(en, c1))

    @staticmethod
    def _lines(paths):
        for p_ in paths.split(","):
            if p_ and os.path.exists(p_):
                yield from open(p_)

    def _load_facts(self, paths):
        for line in self._lines(paths):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r["doc_id"] not in self.docs:
                continue
            sp = "held" if held_out(r["doc_id"]) else "train"
            self.anchors_of[r["doc_id"]] = [(f["a_start"], f["a_end"]) for f in r["facts"] if f.get("ok")]
            for f in r["facts"]:
                v = str(f.get("value") or "")
                if f.get("ok") and f.get("value_ok") and len(v) >= 2 and v in f["statement"]:
                    self.cloze[sp].append({"doc": r["doc_id"], "q": "Fill in the blank: " + f["statement"].replace(v, "___", 1),
                                           "query": f["statement"].replace(v, " ", 1), "answer": v, "aliases": [],
                                           "anchors": [(f["a_start"], f["a_end"])]})
                    self.cloze_by_doc[r["doc_id"]].append(self.cloze[sp][-1])

    def _load_questions(self, paths):
        for line in self._lines(paths):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r["doc_id"] not in self.docs:
                continue
            sp = "held" if held_out(r["doc_id"]) else "train"
            for q in r["questions"]:
                if not q.get("verify", {"keep": True}).get("keep", True) or not q.get("anchors"):
                    continue
                it = {"doc": r["doc_id"], "q": q["question"], "query": q["question"], "answer": q["answer"],
                      "aliases": q.get("aliases", []), "anchors": [tuple(x) for x in q["anchors"]]}
                if q["type"] in ("single", "multi"):
                    self.nat[sp].append(it)
                elif q["type"] == "far":
                    self.far[sp].append(it)
                if q["type"] in ("single", "multi", "far"):
                    self.q_by_doc[r["doc_id"]].append(it)

    def summary(self):
        return (f"fill-in items train/held {len(self.cloze['train'])}/{len(self.cloze['held'])}, questions "
                f"{len(self.nat['train'])}/{len(self.nat['held'])}, far {len(self.far['train'])}/{len(self.far['held'])}, "
                f"documents with facts {len(self.anchors_of)}")


class Trainer:
    def __init__(self, a, mm, data, log):
        self.a, self.mm, self.data, self.log = a, mm, data, log
        self.dev = mm.dev
        self.rng = random.Random(a.seed)
        self.bank = collections.deque(maxlen=max(1, a.bank))         # [(document, unit rows of a piece, fp16)] per episode
        self.base_cache = {}                                         # controls do not depend on training: computed once

    # ---------- helpers over the frozen model ----------
    def nrm(self, x):
        return F.normalize(rmsn(x.float()) - self.mm.mu, dim=-1)

    def run(self, x, pos, lo, hi, ckpt=False):
        layers = self.mm.layers
        pe = self.mm.rot(x, pos[None])
        for l in range(lo, hi):
            if ckpt:
                x = checkpoint(lambda y, l=l: layers[l](y, position_embeddings=pe, position_ids=pos[None]), x, use_reentrant=False)
            else:
                x = layers[l](x, position_embeddings=pe, position_ids=pos[None])
        return x

    def query_vecs(self, text, grad):
        return self.mm.tower_vectors(self.mm.hidden_levels(self.mm.tok(text, add_special_tokens=False).input_ids), grad)[0]

    def query_vecs_batch(self, texts, grad):
        levs = self.mm.hidden_levels_batch([self.mm.tok(t_, add_special_tokens=False).input_ids for t_ in texts])
        return [self.mm.tower_vectors(lev, grad)[0] for lev in levs]

    # ---------- episodes ----------
    def piece_around(self, did, c0, c1, size, r_):
        ids = self.data.doc_tok(did)[0]
        n = len(ids)
        if n <= size:
            return 0, n
        i0, i1 = self.data.tok_span(did, c0, c1)
        lo = min(max(0, (i0 + i1) // 2 - r_.randrange(size // 8, size - size // 8)), n - size)
        return lo, lo + size

    def distractors(self, did, n_, r_):
        """Topic neighbors (hard negatives) from the same split; otherwise random documents of the same source."""
        near = [x for x in self.data.nbr.get(did, []) if x != did and held_out(x) == held_out(did)]
        if len(near) >= n_:
            return r_.sample(near, n_)
        pool = self.data.by_src[self.data.src_key(did)]
        out = []
        for _ in range(4 * n_):
            x = r_.choice(pool)
            if x != did and x not in out and held_out(x) == held_out(did):
                out.append(x)
            if len(out) == n_:
                break
        return out

    @torch.no_grad()
    def best_window(self, did, qn, lo, hi):
        """The 64-token window of [lo, hi) most similar to the question (MaxSim of the current tower) → (centre, score)."""
        a = self.a
        f, _ = self.mm.tower_vectors(self.mm.hidden_levels(self.data.doc_tok(did)[0][lo:hi]))
        S = qn @ self.nrm(f).T
        L = S.shape[1]; nw = -(-L // a.swin)
        S = F.pad(S, (0, nw * a.swin - L), value=float("-inf")).view(S.shape[0], nw, a.swin).max(-1).values.mean(0)
        w = int(S.argmax())
        return lo + min(L - 1, w * a.swin + a.swin // 2), float(S[w])

    def make_episode_app(self, kind, item, r_):
        a, data = self.a, self.data
        did = item["doc"]
        ids = data.doc_tok(did)[0]
        spans = [data.tok_span(did, c0, c1) for c0, c1 in item["anchors"]]
        a0, a1 = min(x for x, _ in spans), max(y for _, y in spans)
        if a1 - a0 > a.piece:
            return None
        qn = self.nrm(self.query_vecs(item["query"], False))
        pieces = []
        if kind != "absent":
            lo, hi = self.piece_around(did, min(c for c, _ in item["anchors"]), max(c for _, c in item["anchors"]), a.piece, r_)
            if not (lo <= a0 and a1 <= hi):
                lo = max(0, min(a0 - (a.piece - (a1 - a0)) // 2, len(ids) - a.piece)); hi = min(len(ids), lo + a.piece)
            _, sc = self.best_window(did, qn, lo, hi)
            if kind == "cite":
                wl = max(lo, min(a0 - r_.randrange(0, max(1, a.win - (a1 - a0))), hi - a.win))
            else:
                cands = [x for x in range(lo, max(lo + 1, hi - a.win), 16) if x + a.win <= a0 - 16 or x >= a1 + 16]
                if not cands:
                    return None
                wl = r_.choice(cands)
            pieces.append((sc, (did, lo, hi, "target"), (did, wl, min(hi, wl + a.win))))
        for x in self.distractors(did, a.app_pieces - len(pieces), r_):
            xn = len(data.doc_tok(x)[0])
            s0 = r_.randrange(0, max(1, xn - a.dscan)); s1 = min(xn, s0 + a.dscan)
            c, sc = self.best_window(x, qn, s0, s1)
            xl = max(0, min(c - a.piece // 2, xn - a.piece)); xh = min(xn, xl + a.piece)
            wl = max(xl, min(c - a.win // 2, xh - a.win))
            pieces.append((sc, (x, xl, xh, "dist"), (x, wl, min(xh, wl + a.win))))
        if not pieces or (kind != "absent" and len(pieces) < 2):
            return None
        pieces.sort(key=lambda t_: -t_[0])
        tok = self.mm.tok
        txt = " ".join(tok.decode(data.doc_tok(w_[0])[0][w_[1]:w_[2]]) for _, p_, w_ in pieces if kind != "cite" or p_[3] != "target")
        if kind != "cite" and norm_ans(item["answer"]) and norm_ans(item["answer"]) in norm_ans(txt):
            return None                                                  # the answer is visible in the text: not know/absent
        ans = ABSTAIN if kind == "absent" else item["answer"]
        queries = [(item["query"], (a0, a1))] if kind != "absent" else []
        return {"kind": kind, "item": item, "pieces": [p_ for _, p_, _ in pieces], "win": None,
                "wtxt": [w_ for _, _, w_ in pieces], "q": item["q"], "answer": ans, "anchor": (a0, a1), "queries": queries}

    def make_episode(self, kind, item, r_=None):
        """→ dict: pieces (document, lo, hi, role), text window or None, question, answer, queries; None if it did not work."""
        a, data = self.a, self.data
        r_ = r_ or self.rng
        if a.app:
            return self.make_episode_app(kind, item, r_)
        did = item["doc"]
        ids = data.doc_tok(did)[0]
        spans = [data.tok_span(did, c0, c1) for c0, c1 in item["anchors"]]
        a0, a1 = min(s for s, _ in spans), max(e for _, e in spans)
        if a1 - a0 > a.piece:
            return None
        lo, hi = self.piece_around(did, min(c for c, _ in item["anchors"]), max(c for _, c in item["anchors"]), a.piece, r_)
        if not (lo <= a0 and a1 <= hi):
            lo = max(0, min(a0 - (a.piece - (a1 - a0)) // 2, len(ids) - a.piece)); hi = min(len(ids), lo + a.piece)
        pieces = [] if kind == "absent" else [(did, lo, hi, "target")]
        for x in self.distractors(did, a.ndist, r_):
            xn = len(data.doc_tok(x)[0])
            xl = r_.randrange(0, max(1, xn - a.dpiece))
            pieces.append((x, xl, min(xn, xl + a.dpiece), "dist"))
        if not pieces:
            return None
        r_.shuffle(pieces)
        win = None
        if kind == "cite":
            wl = max(lo, min(a0 - r_.randrange(0, max(1, a.win - (a1 - a0))), hi - a.win))
            win = (did, wl, min(hi, wl + a.win))
        elif kind == "know":
            cands = [s for s in range(lo, max(lo + 1, hi - a.win), 16) if s + a.win <= a0 - 16 or s >= a1 + 16]
            if not cands:
                return None
            wl = r_.choice(cands)
            win = (did, wl, min(hi, wl + a.win))
            if norm_ans(item["answer"]) in norm_ans(self.mm.tok.decode(ids[wl:win[2]])):
                return None                                              # the answer is visible in the window: not know
        ans = ABSTAIN if kind == "absent" else item["answer"]
        queries = [(item["query"], (a0, a1))]
        if kind != "absent" and a.nq > 1 and r_ is self.rng:           # training only: more queries from the same piece
            def inside(pool):
                out = []
                for it in pool:
                    if it is item:
                        continue
                    sp_ = [data.tok_span(did, c0, c1) for c0, c1 in it["anchors"]]
                    s0, s1 = min(x for x, _ in sp_), max(y for _, y in sp_)
                    if lo <= s0 and s1 <= hi:
                        out.append((it["query"], (s0, s1)))
                return out
            qx, cx = inside(data.q_by_doc.get(did, [])), inside(data.cloze_by_doc.get(did, []))
            n_q = min(len(qx), round((a.nq - 1) * a.qnat))               # share --qnat of natural questions, rest fill-ins
            queries += r_.sample(qx, n_q) + r_.sample(cx, min(len(cx), a.nq - 1 - n_q))
        return {"kind": kind, "item": item, "pieces": pieces, "win": win, "q": item["q"], "answer": ans,
                "anchor": (a0, a1), "queries": queries}

    def sample(self, split):
        a, data, rng = self.a, self.data, self.rng
        r_ = rng.random()
        acc = 0.0
        for kv in a.mix.split(","):
            kind, p = kv.split(":")
            acc += float(p)
            if r_ < acc:
                break
        if kind == "far" and not data.far[split]:
            kind = "know"
        if kind == "far":
            item = rng.choice(data.far[split])
        elif data.nat[split] and rng.random() < a.qnat:
            item = rng.choice(data.nat[split])
        else:
            item = rng.choice(data.cloze[split])
        return self.make_episode(kind, item)

    # ---------- passes ----------
    def piece_vectors(self, ep, grad):
        """Tower over each piece → memory rows f, their positions, search windows, leash, gate logits and gate labels."""
        a, mm, data = self.a, self.mm, self.data
        rows, poss, wins, leash_l, glog, glab = [], [], [], [], [], []
        off = 0
        for did, lo, hi, role in ep["pieces"]:
            ids = data.doc_tok(did)[0][lo:hi]
            lev = mm.hidden_levels(ids)
            f, g = mm.tower_vectors(lev, grad)
            if grad:
                with torch.no_grad():
                    mm.mu.mul_(0.99).add_(0.01 * rmsn(f.detach()).mean(0))
            s = lev[mm.tower.ibase].float()
            leash_l.append(((f - s).pow(2).sum(-1) / s.pow(2).sum(-1).clamp(min=1e-6)).mean())
            lab = torch.zeros(len(ids), device=self.dev)
            _, st, en = data.doc_tok(did)
            for c0, c1 in data.anchors_of.get(did, []):
                k0, k1 = bisect.bisect_left(st, c0) - lo, bisect.bisect_right(en, c1) - lo
                k0, k1 = max(k0, 0), min(max(k1, k0 + 1), len(ids))
                if k1 > k0:
                    lab[k0:k1] = 1.0
            glog.append(g); glab.append(lab)
            rows.append(f); poss.append(torch.arange(len(ids), device=self.dev) + off)
            for w0 in range(0, len(ids), a.swin):                        # search windows over the piece
                wins.append((len(rows) - 1, role == "target", lo + w0))
            off += len(ids)
        return rows, poss, wins, off, torch.stack(leash_l).mean(), torch.cat(glog), torch.cat(glab)

    def positives(self, wins, span_):
        a0_, a1_ = span_
        return [j for j, (_, tgt, w0) in enumerate(wins) if tgt and w0 < a1_ and a0_ < w0 + self.a.swin]

    def search_scores(self, fq, rows, wins, span_, rn=None):
        """MaxSim of the question tokens over every window of every piece (windows tile each piece) / τ."""
        a = self.a
        qn = self.nrm(fq)
        sc_ = []
        for r in (rn if rn is not None else [self.nrm(r_) for r_ in rows]):
            S = qn @ r.T
            L = S.shape[1]; nw = -(-L // a.swin)
            S = F.pad(S, (0, nw * a.swin - L), value=float("-inf")).view(S.shape[0], nw, a.swin)
            sc_.append(S.max(-1).values.mean(0))
        return torch.cat(sc_) / a.tau, self.positives(wins, span_)

    def bank_matrix(self, exclude):
        """Rows of queued pieces from other documents, each padded to a whole number of windows, and the real-row mask."""
        rs, ms = [], []
        for ep_rows in self.bank:
            for did, r in ep_rows:
                if did in exclude:
                    continue
                pad = -r.shape[0] % self.a.swin
                rs.append(F.pad(r, (0, 0, 0, pad))); ms.append(torch.arange(r.shape[0] + pad, device=self.dev) < r.shape[0])
        if not rs:
            return None, None
        return torch.cat(rs), torch.cat(ms)

    def bank_scores(self, qn, B, M):
        S = (qn.half() @ B.T).float().masked_fill(~M[None], float("-inf"))
        return S.view(S.shape[0], -1, self.a.swin).max(-1).values.mean(0) / self.a.tau

    def text_ids(self, ep):
        w = []
        for did, wl, wh in (ep.get("wtxt") or ([ep["win"]] if ep["win"] else [])):
            w += self.data.doc_tok(did)[0][wl:wh] + self.mm.sep
        return w

    def tails(self, ep):
        post = self.mm.question_ids(ep["q"])
        ans = self.mm.tok(ep["answer"], add_special_tokens=False).input_ids + [self.mm.eos]
        return post, ans

    def student_h(self, rows, poss, mem_span, rest, grad):
        """Layer-cut input of the student: prompt and text through layers 0..cut−2, memory rows through R (layer cut−1)."""
        mm, dev, P0 = self.mm, self.dev, self.mm.P0
        mem = torch.cat(rows, 0) if rows else torch.zeros(0, mm.d, device=dev)
        mpos = (torch.cat(poss) + P0) if rows else torch.zeros(0, dtype=torch.long, device=dev)
        pos_rest = torch.arange(P0 + mem_span, P0 + mem_span + len(rest), device=dev)
        pos = torch.cat([torch.arange(P0, device=dev), mpos, pos_rest])
        with torch.no_grad():
            xt = self.run(mm.emb(mm._T([mm.pre + rest])), torch.cat([pos[:P0], pos_rest]), 0, mm.cut - 1)
        x = torch.cat([xt[:, :P0].float(), mem[None].float(), xt[:, P0:].float()], 1)
        with torch.set_grad_enabled(grad), torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev != "cpu"):
            y = mm.R(x, position_embeddings=mm.rot(x, pos[None]), position_ids=pos[None])
            M = mem.shape[0]
            m_in = x[:, P0:P0 + M]
            y = torch.cat([y[:, :P0], m_in + mm.gate * (y[:, P0:P0 + M] - m_in), y[:, P0 + M:]], 1)
        return y, pos

    def upper_logits(self, y, pos, sl, grad):
        mm = self.mm
        with torch.set_grad_enabled(grad), torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.dev != "cpu"):
            h = self.run(y.to(torch.bfloat16), pos, mm.cut, mm.nL, ckpt=grad)
            return mm.model.lm_head(mm.model.model.norm(h[0, sl])).float()

    @torch.no_grad()
    def teacher_cut(self, ep, post, ans):
        """The frozen model reading all pieces as text (target first): its layer-cut input on the question and answer."""
        mm = self.mm
        ctx = []
        for did, lo, hi, role in sorted(ep["pieces"], key=lambda p: p[3] != "target"):
            ctx += self.data.doc_tok(did)[0][lo:hi] + mm.sep
        ids = mm.pre + ctx + post + ans
        x = self.run(mm.emb(mm._T([ids])), torch.arange(len(ids), device=self.dev), 0, mm.cut)
        return x[0, -(len(post) + len(ans)):].float()

    @torch.no_grad()
    def base_answer(self, ctx_ids, post):
        mm, out = self.mm, []
        for _ in range(self.a.max_new):
            nx = int(mm.model(input_ids=mm._T([mm.pre + ctx_ids + post + out])).logits[0, -1].argmax())
            if nx == mm.eos:
                break
            out.append(nx)
        return mm.tok.decode(out, skip_special_tokens=True)

    def episode_loss(self, ep, lam):
        a = self.a
        rows, poss, wins, span, leash_l, glog, glab = self.piece_vectors(ep, True)
        loss, parts = 0.0, {}
        if ep["kind"] != "absent":
            lss, hits = [], []
            qs = [(qt, sp) for qt, sp in ep["queries"] if self.positives(wins, sp)]
            fqs = self.query_vecs_batch([qt for qt, _ in qs], True) if qs else []
            rn = [self.nrm(r_) for r_ in rows]
            B, M = self.bank_matrix({p_[0] for p_ in ep["pieces"]}) if a.bank else (None, None)
            for (_, span_), fq in zip(qs, fqs):
                sc, posw = self.search_scores(fq, rows, wins, span_, rn)
                if B is not None:
                    sc = torch.cat([sc, self.bank_scores(self.nrm(fq), B, M)])   # negatives from other documents
                lss.append(torch.logsumexp(sc, 0) - torch.logsumexp(sc[posw], 0))
                hits.append(float(int(sc.argmax()) in posw))
            if lss:
                ls = torch.stack(lss).mean()
                loss = loss + a.search_w * ls
                parts["search"] = ls.item(); parts["hit"] = sum(hits) / len(hits); parts["nq"] = float(len(lss))
                if B is not None:
                    parts["bank"] = float(M.shape[0] // a.swin)
            if a.bank:
                self.bank.append([(p_[0], r_.detach().half()) for p_, r_ in zip(ep["pieces"], rn)])
        gm = torch.sigmoid(glog).mean()
        gate_l = a.gate_w * F.binary_cross_entropy_with_logits(glog, glab) + lam * gm
        loss = loss + a.leash * leash_l + gate_l
        parts["leash"] = leash_l.item(); parts["open"] = gm.item()
        if a.stage == "reader":
            post, ans = self.tails(ep)
            y, pos = self.student_h(rows, poss, span, self.text_ids(ep) + post + ans, True)
            na, nt = len(ans), len(post) + len(ans)
            ce = F.cross_entropy(self.upper_logits(y, pos, slice(-na - 1, -1), True), self.mm._T(ans))
            t = self.teacher_cut(ep, post, ans)
            s = y[0, -nt:].float()
            mt = ((s - t).pow(2).sum(-1) / t.pow(2).sum(-1).clamp(min=1e-6)).mean()
            loss = loss + a.ce * ce + a.match * mt
            parts["ce"] = ce.item(); parts["match"] = mt.item()
        return loss, parts, gm.item()

    @torch.no_grad()
    def evaluate(self, step):
        a, mm, data = self.a, self.mm, self.data
        mm.tower.eval()
        res = collections.defaultdict(list)
        er = random.Random(7)
        kinds = ["cite", "know", "far", "absent"] if a.stage == "reader" else ["cite"]
        for kind in kinds:
            pools = data.far["held"] if kind == "far" else (data.nat["held"] + data.cloze["held"])
            if not pools:
                continue
            items = er.sample(pools, min(len(pools), a.eval_n if kind in ("cite", "know") else a.eval_n // 2))
            for i_it, it in enumerate(items):
                ep = self.make_episode(kind, it, er)
                if ep is None:
                    continue
                rows, poss, wins, span, _, _, _ = self.piece_vectors(ep, False)
                if kind != "absent" and self.positives(wins, ep["anchor"]):
                    sc, posw = self.search_scores(self.query_vecs(it["query"], False), rows, wins, ep["anchor"])
                    res[(kind, "search")].append(float(int(sc.argmax()) in posw))
                if a.stage == "search":
                    continue
                post, ans = self.tails(ep)
                out = mm.answer(rows, poss, span, self.text_ids(ep), ep["q"], None, a.max_new)[0]
                ok = norm_ans(ABSTAIN) in norm_ans(out) if kind == "absent" else self.correct(it, out)
                res[(kind, "student")].append(float(ok))
                y, pos = self.student_h(rows, poss, span, self.text_ids(ep) + post + ans, False)
                res[(kind, "nll")].append(float(F.cross_entropy(self.upper_logits(y, pos, slice(-len(ans) - 1, -1), False),
                                                                mm._T(ans))))
                if kind in ("know", "far") and a.controls and len(res[(kind, "nomem")]) < a.eval_n // 2:
                    res[(kind, "nomem")].append(float(self.correct(it, mm.answer([], [], 0, [], ep["q"], None, a.max_new)[0])))
                    if (kind, i_it) not in self.base_cache:
                        full = []
                        for did, lo, hi, role in ep["pieces"]:
                            full += data.doc_tok(did)[0][lo:hi] + mm.sep
                        wtxt = self.text_ids(ep)
                        self.base_cache[(kind, i_it)] = (float(self.correct(it, self.base_answer(wtxt, post))),
                                                         float(self.correct(it, self.base_answer(full + wtxt, post))))
                    bw, bf = self.base_cache[(kind, i_it)]
                    res[(kind, "text_window")].append(bw); res[(kind, "text_pieces")].append(bf)
        f = lambda key: f"{sum(res[key]) / len(res[key]):.3f} ({len(res[key])})" if res.get(key) else "-"
        line = " | ".join(f"{kind}: search {f((kind, 'search'))}" + (
            f", reader {f((kind, 'student'))}, NLL {f((kind, 'nll'))}, no memory {f((kind, 'nomem'))}, unmodified model on "
            f"the window {f((kind, 'text_window'))}, on all pieces {f((kind, 'text_pieces'))}" if a.stage == "reader" else "")
            for kind in kinds)
        self.log(f"== eval step {step}: {line}")
        mm.tower.train()

    @staticmethod
    def correct(item, out):
        o = norm_ans(out)
        return any(norm_ans(x) and norm_ans(x) in o for x in [item["answer"]] + list(item.get("aliases", [])))


def main():
    ap = argparse.ArgumentParser(prog="llm_memory.train")
    ap.add_argument("--stage", required=True, choices=["search", "reader"])
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--docs", default="data/train/docs.jsonl")
    ap.add_argument("--facts", default="data/train/facts-0.jsonl,data/train/facts-1.jsonl", help="comma-separated jsonl files")
    ap.add_argument("--questions", default="data/train/questions.jsonl", help="verified questions (empty: fill-ins only)")
    ap.add_argument("--neighbors", default="data/train/neighbors.json", help="topic neighbors for distractors (empty: random)")
    ap.add_argument("--out", required=True); ap.add_argument("--init", default="", help="checkpoint to start from")
    ap.add_argument("--resume", action="store_true", help="continue <out>/last.pt with its step and optimizer")
    ap.add_argument("--cut", type=int, default=13, help="memory enters the input of this layer; R is a copy of layer cut−1")
    ap.add_argument("--levels", default=",".join(str(i) for i in range(0, 35, 2)), help="tower input: these layer outputs")
    ap.add_argument("--heads", type=int, default=20); ap.add_argument("--mlp", type=int, default=6144)
    ap.add_argument("--tlayers", type=int, default=3)
    ap.add_argument("--tckpt", action="store_true", help="gradient checkpointing of the tower blocks (less memory)")
    ap.add_argument("--piece", type=int, default=2048, help="target piece, tokens")
    ap.add_argument("--dpiece", type=int, default=512, help="distractor piece (classic episodes)")
    ap.add_argument("--ndist", type=int, default=2, help="distractor pieces (classic episodes)")
    ap.add_argument("--app", action="store_true", help="reader: episodes as after retrieval (see module docstring)")
    ap.add_argument("--app-pieces", type=int, default=4); ap.add_argument("--dscan", type=int, default=2048,
                    help="--app: tokens of a neighbor document scanned for its most similar window")
    ap.add_argument("--win", type=int, default=64, help="text window"); ap.add_argument("--swin", type=int, default=64,
                                                                                      help="search window")
    ap.add_argument("--mix", default="cite:0.3,know:0.4,far:0.2,absent:0.1", help="reader: episode kinds and shares")
    ap.add_argument("--qnat", type=float, default=0.5, help="share of natural questions among items and extra queries")
    ap.add_argument("--nq", type=int, default=8, help="search queries per episode (other facts and questions of the piece)")
    ap.add_argument("--bank", type=int, default=0, help="queue of this many recent episodes as cross-document negatives")
    ap.add_argument("--ce", type=float, default=1.0); ap.add_argument("--match", type=float, default=0.3)
    ap.add_argument("--search-w", type=float, default=0.5); ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--leash", type=float, default=0.1); ap.add_argument("--gate-w", type=float, default=0.2)
    ap.add_argument("--budget", type=float, default=0.3, help="target mean gate openness")
    ap.add_argument("--lr", type=float, default=1e-4, help="tower"); ap.add_argument("--rlr", type=float, default=5e-5,
                                                                                   help="reader and mixing scalar")
    ap.add_argument("--steps", type=int, default=3000); ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=50); ap.add_argument("--hours", type=float, default=0.0)
    ap.add_argument("--eval-every", type=int, default=250); ap.add_argument("--eval-n", type=int, default=60)
    ap.add_argument("--no-controls", dest="controls", action="store_false", help="skip the no-memory and text controls")
    ap.add_argument("--log-every", type=int, default=10); ap.add_argument("--save-every", type=int, default=250)
    ap.add_argument("--max-new", type=int, default=48); ap.add_argument("--seed", type=int, default=124)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev, t0 = a.device, time.time()
    log = lambda s: print(f"{s} ({time.time() - t0:.0f}s)", flush=True)
    torch.manual_seed(a.seed)
    if torch.version.hip:                                                # ROCm: flash/mem-efficient SDPA failed in backward
        torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False)

    ck_path = a.init or (os.path.join(a.out, "last.pt") if a.resume else "")
    assert not ck_path or os.path.exists(ck_path), f"no checkpoint {ck_path}"
    mm = MemoryModel(a.model, ck_path or None, dev, {k_: getattr(a, k_) for k_ in ("cut", "levels", "heads", "mlp", "tlayers")},
                     grad_ckpt=a.tckpt)
    data = TrainData(mm.tok, a.docs, a.facts, a.questions, a.neighbors)
    log(f"{len(data.docs)} documents; {data.summary()}")
    mm.tower.train()
    if a.stage == "search":
        mm.R = None                                                       # the reader is not trained in this stage
    else:
        mm.R.requires_grad_(True)
        mm.gate = nn.Parameter(mm.gate.clone())
    step0 = mm.extra.get("step", 0) if a.resume else 0
    lam = mm.extra.get("lam", 0.0) if ck_path else 0.0
    if ck_path:
        log(f"loaded {ck_path} (step {mm.step})")
    groups = [{"params": list(mm.tower.parameters()), "base": a.lr}]
    if mm.R is not None:
        groups.append({"params": list(mm.R.parameters()) + [mm.gate], "base": a.rlr})
    fused = dev != "cpu" and not torch.version.hip
    opt = torch.optim.AdamW(groups, lr=a.lr, weight_decay=0.0, **({"fused": True} if fused else {"foreach": False}))
    if a.resume and "opt" in mm.extra:
        opt.load_state_dict(mm.extra["opt"])
    mm.extra = {}
    log(f"tower {sum(p_.numel() for p_ in mm.tower.parameters()) / 1e6:.0f}M, reader "
        f"{sum(p_.numel() for p_ in mm.R.parameters()) / 1e6 if mm.R is not None else 0:.0f}M; cut {mm.cut}")

    tr = Trainer(a, mm, data, log)

    def save(step):
        os.makedirs(a.out, exist_ok=True)
        torch.save(mm.checkpoint(lam=lam, step=step, opt=opt.state_dict(), train_args=vars(a)), os.path.join(a.out, "last.pt"))

    params = [p_ for g_ in groups for p_ in g_["params"]]
    if step0 == 0:
        tr.evaluate(0)
    step, run_parts = step0, collections.Counter()
    t_last = time.time()
    while step < a.steps:
        if a.hours and time.time() - t0 > a.hours * 3600:
            log("time is up")
            break
        sched = min(1.0, (step + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, step / a.steps)))
        for g_ in opt.param_groups:
            g_["lr"] = g_["base"] * sched
        opt.zero_grad(set_to_none=True)
        gms = []
        for _ in range(a.accum):
            ep = None
            while ep is None:
                ep = tr.sample("train")
            loss, parts, gm = tr.episode_loss(ep, lam)
            (loss / a.accum).backward()
            for k_, v in parts.items():
                run_parts[k_] += v
            run_parts["n"] += 1
            gms.append(gm)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        lam = min(10.0, max(0.0, lam + 0.05 * (sum(gms) / len(gms) - a.budget)))   # keeps mean gate openness near budget
        step += 1
        if step % a.log_every == 0:
            n_ = max(1, run_parts.pop("n"))
            dt = time.time() - t_last
            t_last = time.time()
            log(f"  step {step}: " + ", ".join(f"{k_} {v / n_:.4f}" for k_, v in run_parts.items()) +
                (f", alpha {float(mm.gate):.3f}" if mm.R is not None else "") +
                f", lambda {lam:.2f}, episodes/s {a.log_every * a.accum / dt:.2f}, memory {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
            run_parts.clear()
        if step % a.eval_every == 0:
            tr.evaluate(step)
        if step % a.save_every == 0:
            save(step)
    save(step)
    if step % a.eval_every:
        tr.evaluate(step)
    log("== TRAIN DONE")


if __name__ == "__main__":
    main()
