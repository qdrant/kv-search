"""Verification pass over generated questions: the generator LLM checks each question against the document text.

Generated questions can depend on the document ("the book in this point") or have a wrong reference answer, which makes both
the measure and training noisy. Here the LLM reads the document text around the question's anchors (--pad characters on each
side) and answers in JSON: its own answer from the text, whether the reference answer is correct by the text, and whether the
question is self-contained for someone searching a large collection without seeing this document. A question is kept if the
reference is correct and the question is self-contained; absent and abstract questions are not checked (keep = true). Output:
the same JSONL with a "verify" object on every question. About 98% of the checked questions are kept.

    python -m llm_memory.datagen.verify --server http://127.0.0.1:8000/v1 --questions data/questions.jsonl \\
        --docs data/docs.jsonl --out data/questions-v.jsonl
"""
import argparse
import asyncio
import collections
import json
import os
import re
import time

SYSTEM = "You check questions for a question-answering benchmark against the source text. Be strict and exact."
USER = """Document: {title}

Excerpts from the document:
{text}

Question: {question}
Reference answer: {answer}

1. Answer the question using only the excerpts (a few words).
2. Is the reference answer correct according to the excerpts? (a paraphrase or a shorter form is fine)
3. Is the question self-contained: would a person searching a large collection of many documents, without seeing this
   document, understand exactly what is asked and expect this answer? It is NOT self-contained if it refers to "this",
   "the passage", "the example", "the function above", or leaves the subject ambiguous.

Return only JSON: {{"answer": "...", "reference_correct": true, "self_contained": true}}"""


def parse_obj(text):
    """JSON object from the model's answer, or None."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    a, b = text.find("{"), text.rfind("}")
    if a < 0 or b <= a:
        return None
    s = text[a:b + 1]
    for cand in (s, re.sub(r",\s*([\]}])", r"\1", s)):
        try:
            v = json.loads(cand)
            return v if isinstance(v, dict) else None
        except json.JSONDecodeError:
            continue
    return None


def excerpt(text, anchors, pad):
    spans = sorted([max(0, s - pad), min(len(text), e + pad)] for s, e in anchors)
    merged = []
    for s, e in spans:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return "\n...\n".join(text[s:e] for s, e in merged)


async def amain(a):
    from openai import AsyncOpenAI

    t0 = time.time()
    log = lambda s: print(f"{s} ({time.time() - t0:.0f}s)", flush=True)
    docs = {}
    for line in open(a.docs):
        d = json.loads(line)
        docs[d["id"]] = d
    recs = []
    for line in open(a.questions):
        try:
            recs.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    if a.limit:
        recs = recs[:a.limit]
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            try:
                r = json.loads(line)
                done.add((r["doc_id"], r["group"]))
            except (json.JSONDecodeError, KeyError):
                pass
    recs = [r for r in recs if (r["doc_id"], r["group"]) not in done and r["doc_id"] in docs]
    log(f"groups to check {len(recs)} (done before {len(done)})")
    client = AsyncOpenAI(base_url=a.server, api_key="none", timeout=900)
    sem = asyncio.Semaphore(a.sessions)
    st = collections.Counter()
    fout = open(a.out, "a")

    async def check(d, q):
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": USER.format(title=d["title"], text=excerpt(d["text"], q["anchors"], a.pad),
                                                        question=q["question"], answer=q["answer"])}]
        for attempt in range(4):
            try:
                async with sem:
                    resp = await client.chat.completions.create(
                        model=a.model_name, messages=msgs, temperature=0.0, max_tokens=a.max_tokens,
                        extra_body={"chat_template_kwargs": {"enable_thinking": False}})
                raw = resp.choices[0].message.content or ""
                v = parse_obj(raw)
                if v is None:
                    st["bad_json"] += 1
                    return {"keep": False, "error": "json", "raw": raw[:300]}
                v = {"answer": str(v.get("answer", "")), "reference_correct": bool(v.get("reference_correct")),
                     "self_contained": bool(v.get("self_contained"))}
                v["keep"] = v["reference_correct"] and v["self_contained"]
                return v
            except Exception:
                st["retry"] += 1
                await asyncio.sleep(3 * (attempt + 1))
        st["req_fail"] += 1
        return {"keep": False, "error": "request"}

    async def run(r):
        d = docs[r["doc_id"]]
        outs = await asyncio.gather(*(check(d, q) if q["type"] in ("single", "multi", "far") and q["anchors"] else
                                      asyncio.sleep(0, result={"keep": True, "skipped": True}) for q in r["questions"]))
        for q, v in zip(r["questions"], outs):
            q["verify"] = v
            if not v.get("skipped"):
                st["checked"] += 1
                st["keep"] += v["keep"]
                st["not_correct"] += not v.get("reference_correct", False)
                st["not_self"] += not v.get("self_contained", False)
        st["groups"] += 1
        fout.write(json.dumps(r, ensure_ascii=False) + "\n")
        fout.flush()

    async def reporter():
        while True:
            await asyncio.sleep(a.report)
            log(f"  groups {st['groups']}/{len(recs)}, questions checked {st['checked']}, kept {st['keep']} (reference wrong "
                f"{st['not_correct']}, not self-contained {st['not_self']}), bad JSON {st['bad_json']}")

    rep = asyncio.create_task(reporter())
    await asyncio.gather(*(run(r) for r in recs))
    rep.cancel()
    log(f"  total: checked {st['checked']}, kept {st['keep']} ({st['keep'] / max(1, st['checked']):.1%}); reference wrong "
        f"{st['not_correct']}, not self-contained {st['not_self']}, failed requests {st['req_fail']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model-name", default="qwen", help="model name served by the server")
    ap.add_argument("--questions", default="data/questions.jsonl"); ap.add_argument("--docs", default="data/docs.jsonl")
    ap.add_argument("--out", default="data/questions-v.jsonl")
    ap.add_argument("--pad", type=int, default=1500, help="characters of text around each anchor")
    ap.add_argument("--sessions", type=int, default=128); ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--report", type=int, default=60); ap.add_argument("--limit", type=int, default=0)
    asyncio.run(amain(ap.parse_args()))


if __name__ == "__main__":
    main()
