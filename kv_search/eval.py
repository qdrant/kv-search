"""NIAH scoring."""

import re
import unicodedata

from pydantic import BaseModel

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]", flags=re.UNICODE)


def normalize(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).lower()
    s = _PUNCT.sub(" ", s)
    return _WS.sub(" ", s).strip()


def _tokens(s: str) -> list[str]:
    return normalize(s).split()


def containment(gen: str, ref: str) -> float:
    n_ref = normalize(ref)
    if not n_ref:
        return 0.0
    return float(n_ref in normalize(gen))


def token_f1(gen: str, ref: str) -> float:
    g, r = _tokens(gen), _tokens(ref)
    if not g or not r:
        return float(not g and not r)
    common = 0
    seen: dict[str, int] = {}
    for t in r:
        seen[t] = seen.get(t, 0) + 1
    for t in g:
        if seen.get(t, 0) > 0:
            seen[t] -= 1
            common += 1
    if common == 0:
        return 0.0
    prec, rec = common / len(g), common / len(r)
    return 2 * prec * rec / (prec + rec)


def rouge_l(gen: str, ref: str) -> float:
    """LCS-based F-measure over tokens."""
    g, r = _tokens(gen), _tokens(ref)
    if not g or not r:
        return float(not g and not r)
    prev = [0] * (len(r) + 1)
    for gi in g:
        cur = [0]
        for j, rj in enumerate(r):
            cur.append(prev[j] + 1 if gi == rj else max(prev[j + 1], cur[j]))
        prev = cur
    lcs = prev[-1]
    if lcs == 0:
        return 0.0
    prec, rec = lcs / len(g), lcs / len(r)
    return 2 * prec * rec / (prec + rec)


def recall_at_k(approx_ids: set[int], exact_ids: set[int]) -> float:
    """Fraction of exact ids also returned by the approximate search."""
    if not exact_ids:
        return 1.0
    return len(approx_ids & exact_ids) / len(exact_ids)


class EvalRow(BaseModel):
    bucket: int
    idx: int
    config: str
    containment: float
    token_f1: float
    rouge_l: float
    gen_len: int
    gen: str
    label: str
    agreement: float | None = None


def score_row(
    bucket: int,
    idx: int,
    config: str,
    gen: str,
    label: str,
    reference: str | None = None,
) -> EvalRow:
    # agreement is token-F1 vs the reference (exact-search) generation
    return EvalRow(
        bucket=bucket,
        idx=idx,
        config=config,
        containment=containment(gen, label),
        token_f1=token_f1(gen, label),
        rouge_l=rouge_l(gen, label),
        gen_len=len(gen),
        gen=gen,
        label=label,
        agreement=None if reference is None else token_f1(gen, reference),
    )
