"""Frozen LLM, the tower that turns its hidden states into one vector per token, and the reader R.

The same vector of a corpus token is used twice: as a search key (MaxSim against question tokens) and as memory that the
reader inserts into the model in place of the layer-13 input. Ported from the nvme-llm research code (exp117, exp124,
demo126) without changes to the computation.
"""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

ENC_SYSTEM = "Answer the question using the text."          # system prompt used when encoding documents and questions
ABSTAIN = "not in the text"
READER_SYSTEM = "Answer the question using the text. If the text does not contain the answer, reply: not in the text."
READER_USER = "Text:\n@@MEM@@\n\nQuestion: {q}\n\nAnswer briefly."


def rmsn(x):
    return x / x.pow(2).mean(-1, keepdim=True).sqrt().clamp(min=1e-6)


class Block(nn.Module):
    """Bidirectional transformer block of the tower."""

    def __init__(self, d, heads, mlp):
        super().__init__()
        self.h = heads
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.o = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.up, self.down = nn.Linear(d, mlp), nn.Linear(mlp, d)
        for m in (self.o, self.down):                               # a new block starts as the identity
            nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x):                                           # x [T, d]
        T, d = x.shape
        q, k, v = self.qkv(self.n1(x)).view(T, 3, self.h, d // self.h).permute(1, 2, 0, 3)
        a = F.scaled_dot_product_attention(q[None], k[None], v[None])[0].transpose(0, 1).reshape(T, d)
        x = x + self.o(a)
        return x + self.down(F.gelu(self.up(self.n2(x))))


class Tower(nn.Module):
    """Input: hidden states of selected layers [nlev, T, d]; level ibase is the state the reader works on.
    Output: one vector per token f [T, d] (scaled like the ibase state) and a significance gate logit [T]."""

    def __init__(self, d, nlev, ibase, heads, mlp, layers, grad_ckpt=False):
        super().__init__()
        self.ibase, self.grad_ckpt = ibase, grad_ckpt
        self.proj = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(nlev)])
        for p_ in self.proj:                                        # at start the tower returns the ibase state as is
            nn.init.zeros_(p_.weight)
        self.blocks = nn.ModuleList([Block(d, heads, mlp) for _ in range(layers)])
        self.gn = nn.LayerNorm(d)
        self.gate = nn.Linear(d, 1)
        nn.init.zeros_(self.gate.weight); nn.init.constant_(self.gate.bias, 3.0)   # σ(3) ≈ 0.95: everything open

    def forward(self, lev):
        s = lev[self.ibase].float()
        h = rmsn(s)
        for i, p_ in enumerate(self.proj):
            h = h + p_(rmsn(lev[i].float()))
        for b in self.blocks:
            h = checkpoint(b, h, use_reentrant=False) if self.grad_ckpt and torch.is_grad_enabled() else b(h)
        return s.pow(2).mean(-1, keepdim=True).sqrt() * h, self.gate(self.gn(h)).squeeze(-1)


TOWER_DEFAULTS = {"cut": 13, "levels": ",".join(str(i) for i in range(0, 35, 2)), "heads": 20, "mlp": 6144, "tlayers": 3}


class _Stream:
    """Streamer for generate: passes the answer decoded so far to on_token after each new token."""

    def __init__(self, tok, eos, on_token):
        self.tok, self.eos, self.on_token, self.ids, self.prompt = tok, eos, on_token, [], True

    def put(self, value):
        if self.prompt:                                             # the first call carries the prompt
            self.prompt = False
            return
        self.ids += value.flatten().tolist()
        self.on_token(self.tok.decode(self.ids[:self.ids.index(self.eos)] if self.eos in self.ids else self.ids,
                                      skip_special_tokens=True))

    def end(self):
        pass


class MemoryModel:
    """Qwen3-4B (frozen) + tower + reader R.

    With a checkpoint (keys tower, R, gate, mu, args) everything is loaded from it. Without one, a fresh tower is built from
    `config` (TOWER_DEFAULTS) and the reader starts as an exact copy of layer cut−1 with a zero mixing scalar, so the model
    initially behaves as if the memory rows were the plain layer-cut inputs."""

    def __init__(self, model_path, ckpt_path=None, device="cuda", config=None, grad_ckpt=False):
        self.dev = dev = device
        self.tok = tok = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16).eval().to(dev)
        self.model.requires_grad_(False)
        m = self.model.model
        self.layers, self.rot, self.emb = m.layers, m.rotary_emb, m.embed_tokens
        self.nL, self.d = self.model.config.num_hidden_layers, self.model.config.hidden_size
        self.eos = tok.convert_tokens_to_ids("<|im_end|>")
        etm = tok.apply_chat_template([{"role": "system", "content": ENC_SYSTEM}, {"role": "user", "content": "Text:\n@@MEM@@"}],
                                      tokenize=False, add_generation_prompt=False)
        self.epre = tok(etm.split("@@MEM@@")[0], add_special_tokens=False).input_ids

        ck = torch.load(ckpt_path, map_location="cpu") if ckpt_path else None
        ca = dict(TOWER_DEFAULTS, **(config or {}))
        if ck is not None:
            ca.update({k_: ck["args"][k_] for k_ in TOWER_DEFAULTS})
        self.config = {k_: ca[k_] for k_ in TOWER_DEFAULTS}
        levels = sorted(set(int(x) for x in ca["levels"].split(",")))
        assert ca["cut"] - 1 in levels, "the reader's layer input (output of layer cut−1) must be one of the levels"
        self.lev_pos = {l: i + 1 for i, l in enumerate(levels)}         # position 0 is the embedding output
        self.cut, self.top = ca["cut"], max(levels)
        self.tower = Tower(self.d, len(levels) + 1, self.lev_pos[self.cut - 1], ca["heads"], ca["mlp"], ca["tlayers"],
                           grad_ckpt).to(dev)
        self.mu = torch.zeros(self.d, device=dev)                        # running mean of rmsn(f): centre of the search space
        # reader: a trainable copy of layer cut−1 with its own attention and its own slot in the KV cache
        self.R = copy.deepcopy(self.layers[self.cut - 1]).float()
        self.gate = torch.zeros((), device=dev)                          # scalar: how strongly R rewrites memory rows
        self.step, self.extra = None, {}                                 # extra: the rest of the checkpoint (step, optimizer, …)
        if ck is not None:
            self.tower.load_state_dict(ck["tower"])
            self.mu = ck["mu"].to(dev).float()
            if "R" in ck:
                self.R.load_state_dict(ck["R"]); self.gate = ck["gate"].to(dev).float()
            self.step = ck.get("step")
            self.extra = {k_: v for k_, v in ck.items() if k_ not in ("tower", "R", "gate", "mu")}
            del ck
        self.tower.eval(); self.R.eval()

        rtm = tok.apply_chat_template([{"role": "system", "content": READER_SYSTEM}, {"role": "user", "content": READER_USER}],
                                      tokenize=False, add_generation_prompt=True)
        pre_s, self.post_t = rtm.split("@@MEM@@")
        self.pre = tok(pre_s, add_special_tokens=False).input_ids
        self.P0 = len(self.pre)
        self.sep = tok("\n\n", add_special_tokens=False).input_ids
        self.ab_first = torch.tensor(sorted({tok(v, add_special_tokens=False).input_ids[0]
                                             for v in (ABSTAIN, ABSTAIN.capitalize(), " " + ABSTAIN)}), device=dev)

    def _T(self, v):
        return torch.tensor(v, device=self.dev)

    @torch.no_grad()
    def hidden_levels(self, ids):
        """Hidden states the tower reads: [nlev, T, d] (embeddings + outputs of the chosen layers), frozen model."""
        full = self.epre + ids
        x = self.emb(self._T([full]))
        pos = torch.arange(len(full), device=self.dev)
        pe = self.rot(x, pos[None])
        out = [x[0, len(self.epre):]]
        for l in range(self.top + 1):
            x = self.layers[l](x, position_embeddings=pe, position_ids=pos[None])
            if l in self.lev_pos:
                out.append(x[0, len(self.epre):])
        return torch.stack(out)

    @torch.no_grad()
    def hidden_levels_batch(self, id_lists):
        """hidden_levels for several sequences in one pass (right padding; attention is causal, so real tokens never see it)."""
        fulls = [self.epre + ids for ids in id_lists]
        L = max(len(f_) for f_ in fulls)
        x = self.emb(self._T([f_ + [self.eos] * (L - len(f_)) for f_ in fulls]))
        pos = torch.arange(L, device=self.dev)
        pe = self.rot(x, pos[None])
        out = [x[:, len(self.epre):]]
        for l in range(self.top + 1):
            x = self.layers[l](x, position_embeddings=pe, position_ids=pos[None])
            if l in self.lev_pos:
                out.append(x[:, len(self.epre):])
        st = torch.stack(out)
        return [st[:, b, :len(ids)] for b, ids in enumerate(id_lists)]

    def tower_vectors(self, lev, grad=False):
        """Tower over hidden levels → (f [T, d], gate logits [T]) in float32; with grad=True gradients reach the tower."""
        with torch.set_grad_enabled(grad), torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.dev != "cpu"):
            f, g = self.tower(lev)
        return f.float(), g.float()

    @torch.no_grad()
    def token_vectors(self, ids, with_gate=False):
        """Token ids of a document (or question) → f [T, d] float32, one vector per token."""
        f, g = self.tower_vectors(self.hidden_levels(ids))
        return (f, g) if with_gate else f

    def search_vectors(self, f):
        """Search space of the tower: unit vectors norm(rmsn(f) − μ); dot product = cosine."""
        return F.normalize(rmsn(f.float()) - self.mu, dim=-1)

    def checkpoint(self, **extra):
        """State to save: tower, μ, reader and mixing scalar (if any), tower config, plus extra (step, optimizer, …)."""
        ck = {"tower": self.tower.state_dict(), "mu": self.mu.detach().cpu(), "args": dict(self.config)}
        if self.R is not None:
            ck["R"], ck["gate"] = self.R.state_dict(), self.gate.detach().cpu()
        return dict(ck, **extra)

    def question_ids(self, question):
        return self.tok(self.post_t.format(q=question), add_special_tokens=False).input_ids

    @torch.no_grad()
    def answer(self, mem_rows, poss, span, text_ids, question, threshold=None, max_new=96, on_token=None):
        """Read memory rows (vectors f of pieces around found places) plus text, answer the question.

        Lower layers 0..cut−2 see only the text; R and the upper layers see the prompt, the memory, and the text; each new
        token passes one position through the KV cache. With a threshold the model says "not in the text" only if the
        probability of that answer at the first step is at least the threshold, otherwise it gives its best real answer.
        on_token(text) gets the answer decoded so far after each new token. Returns (answer, p_abstain)."""
        dev, layers, rot, P0 = self.dev, self.layers, self.rot, self.P0
        post = self.question_ids(question)
        mem = torch.cat(mem_rows) if mem_rows else torch.zeros(0, self.d, device=dev)
        M = mem.shape[0]
        mpos = torch.cat(poss) + P0 if mem_rows else torch.zeros(0, dtype=torch.long, device=dev)
        rest = text_ids + post
        pos_rest = torch.arange(P0 + span, P0 + span + len(rest), device=dev)
        pos = torch.cat([torch.arange(P0, device=dev), mpos, pos_rest])
        cache = DynamicCache()

        def lower(ids, p_):
            x = self.emb(self._T([ids])); pe = rot(x, p_[None])
            for l in range(self.cut - 1):
                x = layers[l](x, position_embeddings=pe, position_ids=p_[None], past_key_values=cache, use_cache=True)
            return x

        def upper(y, p_):
            pe = rot(y, p_[None])
            for l in range(self.cut, self.nL):
                y = layers[l](y, position_embeddings=pe, position_ids=p_[None], past_key_values=cache, use_cache=True)
            return self.model.lm_head(self.model.model.norm(y[0, -1:])).float()[0]

        xt = lower(self.pre + rest, torch.cat([pos[:P0], pos_rest]))
        x = torch.cat([xt[:, :P0].float(), mem[None].float(), xt[:, P0:].float()], 1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = self.R(x, position_embeddings=rot(x, pos[None]), position_ids=pos[None], past_key_values=cache, use_cache=True)
            y = torch.cat([y[:, :P0], x[:, P0:P0 + M] + self.gate * (y[:, P0:P0 + M] - x[:, P0:P0 + M]), y[:, P0 + M:]], 1)
            lg = upper(y.to(torch.bfloat16), pos)
        p_abs = float(lg.softmax(-1)[self.ab_first].sum())
        if threshold is not None:
            if p_abs >= threshold:
                return ABSTAIN, p_abs
            lg[self.ab_first] = float("-inf")
        nx = int(lg.argmax())
        out, p_last = [], int(pos[-1])
        while nx != self.eos and len(out) < max_new:
            out.append(nx); p_last += 1
            if on_token:
                on_token(self.tok.decode(out, skip_special_tokens=True))
            p1 = torch.tensor([p_last], device=dev)
            x1 = lower([nx], p1)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                y1 = self.R(x1.float(), position_embeddings=rot(x1, p1[None]), position_ids=p1[None], past_key_values=cache,
                            use_cache=True)
                nx = int(upper(y1.to(torch.bfloat16), p1).argmax())
        return self.tok.decode(out, skip_special_tokens=True), p_abs

    @torch.no_grad()
    def text_answer(self, text_ids, question, max_new=96, on_token=None):
        """The unmodified model reading plain text (the RAG baseline), same prompt and greedy decoding."""
        ids = self._T([self.pre + text_ids + self.question_ids(question)])
        stream = _Stream(self.tok, self.eos, on_token) if on_token else None
        g = self.model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new, do_sample=False,
                                temperature=None, top_p=None, top_k=None, repetition_penalty=1.0, eos_token_id=self.eos,
                                pad_token_id=self.eos, streamer=stream)[0, ids.shape[1]:].tolist()
        return self.tok.decode(g[:g.index(self.eos)] if self.eos in g else g, skip_special_tokens=True)
