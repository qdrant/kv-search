"""LLM judge for the answers of llm_memory.evaluate, and CRAG-style tallies of judged answers.

Ported from the nvme-llm research script judge_answers.py. The judge is a model served by any OpenAI-compatible chat server
(vLLM); the published numbers used Qwen3-30B-A3B-Instruct-2507 (AWQ). For each answer it sees the question, the reference
(with aliases) and the system answer, and returns JSON {"correct": true|false}: correct if the answer states the same fact
as the reference (a paraphrase or extra correct detail is fine); incorrect if it gives a different value, contradicts the
reference, is too vague, or says it does not know.

  judge  answers jsonl → <file>.judged.jsonl next to each file (field "judge": true, false, or null when the judge failed)
         and a summary JSON: share judged correct per file and per document kind, plus the containment measure "ok";
  crag   judged files → right / wrong / "not in the text" / score (+1 right, 0 abstain, −1 wrong), as in CRAG;
  curve  a *.forced.judged.jsonl (evaluate read --abstain-curve) → the same tallies for abstain thresholds τ: the model
         abstains if p(abstain) at the first step is at least τ or its real answer itself says "not in the text".
Abstain: "not in the text" in the lower-cased answer; right: judged correct and not abstain; wrong: the rest.

    vllm serve stelterlab/Qwen3-30B-A3B-Instruct-2507-AWQ --served-model-name judge --port 8004 --max-model-len 4096
    python -m llm_memory.judge judge --server http://127.0.0.1:8004/v1 --answers a.jsonl,b.jsonl --out data/eval/judge.json
    python -m llm_memory.judge crag data/eval/answers/*.judged.jsonl
    python -m llm_memory.judge curve data/eval/answers/ours.forced.judged.jsonl --tau 0.5,0.8,never
"""
import argparse
import asyncio
import collections
import json
import os
import re
import time

ABSTAIN = "not in the text"          # model.ABSTAIN, not imported: the judge and the tallies run without torch
SYSTEM = ("You grade answers of a question-answering system against a reference answer. "
          "Be strict about facts, lenient about wording.")
USER = """Question: {question}
Reference answer: {reference}

System answer: {answer}

Is the system answer correct? It is correct if it states the same fact as the reference answer: a paraphrase, a shorter
or longer form, or extra correct detail is fine. It is incorrect if it gives a different value, contradicts the reference,
is too vague to contain the fact, or says it does not know.

Return only JSON: {{"correct": true}} or {{"correct": false}}"""


def parse_obj(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    a, b = text.find("{"), text.rfind("}")
    if a < 0 or b <= a:
        return None
    try:
        v = json.loads(text[a:b + 1])
        return v if isinstance(v, dict) else None
    except json.JSONDecodeError:
        return None


async def judge_files(a):
    from openai import AsyncOpenAI
    t0 = time.time()
    log = lambda s: print(f"{s} ({time.time() - t0:.0f}s)", flush=True)
    client = AsyncOpenAI(base_url=a.server, api_key="none", timeout=600)
    sem = asyncio.Semaphore(a.sessions)
    st = collections.Counter()

    async def judge(r):
        ref = r["answer"] + ("".join(f" / {x}" for x in r.get("aliases", [])) if r.get("aliases") else "")
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user",
                 "content": USER.format(question=r["query"], reference=ref, answer=r["out"].strip() or "(empty)")}]
        for attempt in range(4):
            try:
                async with sem:
                    resp = await client.chat.completions.create(
                        model=a.model_name, messages=msgs, temperature=0.0, max_tokens=a.max_tokens,
                        extra_body={"chat_template_kwargs": {"enable_thinking": False}})
                v = parse_obj(resp.choices[0].message.content or "")
                if v is None or "correct" not in v:
                    st["bad_json"] += 1
                    return None
                return bool(v["correct"])
            except Exception:
                st["retry"] += 1
                await asyncio.sleep(3 * (attempt + 1))
        st["req_fail"] += 1
        return None

    summary = {}
    for path in a.answers.split(","):
        recs = [json.loads(l) for l in open(path) if l.strip()]
        verdicts = await asyncio.gather(*(judge(r) for r in recs))
        by = collections.defaultdict(lambda: [0, 0])
        with open(os.path.splitext(path)[0] + ".judged.jsonl", "w") as f:
            for r, v in zip(recs, verdicts):
                r["judge"] = v
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                if v is None:
                    continue
                for g in ("all", r.get("kind", "?")):
                    by[g][0] += v; by[g][1] += 1
        contains = sum(r.get("ok", 0) for r in recs) / max(1, len(recs))
        summary[path] = {g: round(c / max(1, n), 3) for g, (c, n) in by.items()} | {"n": len(recs),
                                                                                     "contains": round(contains, 3)}
        log(f"{os.path.basename(path)}: judge {summary[path].get('all', 0.0):.3f} (n {by['all'][1]}), "
            f"contained {contains:.3f}; "
            + ", ".join(f"{g} {c / max(1, n):.3f}" for g, (c, n) in sorted(by.items()) if g != "all"))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(summary, open(a.out, "w"), ensure_ascii=False, indent=1)
    log(f"bad JSON {st['bad_json']}, retries {st['retry']}, failed requests {st['req_fail']}")


def abstains(out):
    return ABSTAIN in out.lower()


def tally(recs, tau=None):
    """CRAG tallies of judged records; with tau, a record abstains also when its p_abs ≥ tau (forced answers)."""
    n = max(1, len(recs))
    ab = [abstains(r["out"]) or (tau is not None and r["p_abs"] >= tau) for r in recs]
    right = sum(1 for r, x in zip(recs, ab) if r.get("judge") is True and not x)
    wrong = len(recs) - right - sum(ab)
    return {"n": len(recs), "right": round(right / n, 3), "wrong": round(wrong / n, 3), "abstain": round(sum(ab) / n, 3),
            "score": round((right - wrong) / n, 3)}


def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def print_rows(rows, key):
    print(f"{key:<48} {'n':>4} {'right':>6} {'wrong':>6} {'abstain':>8} {'score':>6}")
    for name, t in rows:
        print(f"{name:<48} {t['n']:>4} {t['right']:>6.3f} {t['wrong']:>6.3f} {t['abstain']:>8.3f} {t['score']:>+6.3f}")


def main():
    ap = argparse.ArgumentParser(prog="python -m llm_memory.judge")
    sub = ap.add_subparsers(dest="cmd", required=True)
    j = sub.add_parser("judge", help="judge answers with an LLM served by an OpenAI-compatible server")
    j.add_argument("--server", default="http://127.0.0.1:8004/v1"); j.add_argument("--model-name", default="judge")
    j.add_argument("--answers", required=True, help="answers jsonl, comma-separated (llm_memory.evaluate read --dump-answers)")
    j.add_argument("--out", required=True, help="summary JSON"); j.add_argument("--sessions", type=int, default=96)
    j.add_argument("--max-tokens", type=int, default=32)
    c = sub.add_parser("crag", help="right / wrong / abstain / score of judged files")
    c.add_argument("files", nargs="+"); c.add_argument("--out", default="", help="also write the tallies to this JSON")
    k = sub.add_parser("curve", help="abstain-threshold curve from a *.forced.judged.jsonl")
    k.add_argument("file"); k.add_argument("--tau", default="0.3,0.5,0.7,0.8,0.9,never",
                                           help="thresholds, comma-separated; 'never' = never abstain on p_abs")
    k.add_argument("--out", default="", help="also write the curve to this JSON")
    a = ap.parse_args()
    if a.cmd == "judge":
        asyncio.run(judge_files(a))
        return
    if a.cmd == "crag":
        rows = [(os.path.basename(p), tally(load(p))) for p in a.files]
        print_rows(rows, "file")
    else:
        recs = load(a.file)
        rows = [(f"tau {t}", tally(recs, None if t == "never" else float(t))) for t in a.tau.split(",")]
        print_rows(rows, os.path.basename(a.file))
    if a.out:
        json.dump(dict(rows), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
