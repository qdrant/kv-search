"""Natural questions over the generated facts, generated separately from the facts.

The facts of a document whose anchor was found (ok) go in groups of --group in document order. For each group the LLM writes
questions that cite the numbers of the facts they need:
  single    the answer is in one fact, asked in other words (not a fill-in-the-blank, not a copy of the statement);
  multi     needs two or three facts together (compare, combine, relate);
  abstract  the main point or purpose, a relationship across many facts; the answer is one or two sentences;
  absent    plausible for the document, but the facts do not hold the answer (the answer is "not in the text").
--far N adds one more group per document with N facts spread evenly over the whole document, and questions of type far that
need two or three facts from distant parts. Questions must be self-contained for someone searching a large collection, and
short answers come with acceptable variants (aliases). The anchors of the cited facts are the correct retrieval targets.
split is "held" for the held-out documents (md5(id) % 1000 < 50, the same split as training), else "train". The client is
async like facts.py; output is appended and finished groups are skipped on restart. Check the result with verify.py.

    python -m llm_memory.datagen.questions --server http://127.0.0.1:8000/v1 --facts data/train/facts.jsonl \\
        --out data/train/questions.jsonl
"""
import argparse
import asyncio
import collections
import hashlib
import json
import os
import time

from .facts import parse

SYSTEM = ("You write questions for evaluating question answering over a document collection. Questions sound like "
          "real users asking about the document; answers are exact and fully supported by the given facts.")

USER = """Document: {title}
Kind: {kind}

Facts stated in one part of this document (numbered):
{facts}

Write questions that a user of this document collection could ask:
- {n_single} "single": answerable from exactly one fact. Ask naturally, in your own words: not a fill-in-the-blank, and do \
not copy the wording of the fact.
- {n_multi} "multi": answering needs two or three of the facts together (compare, combine, relate them).
- 1 "abstract": about the main point, purpose, or a relationship across many of these facts.
- 1 "absent": plausible for this document and on the same topic, but the facts do NOT contain the answer.

Each question must make sense on its own to someone searching a large collection of many documents, without seeing this \
one: name the subject AND where it lives (for code: the function or type name together with its module, file or project; \
for prose: the person, organization, event or product by name). Never "it", "this", "the passage", "the document", "the \
function above", "this point", "the example". Bad: "What distance metric is configured in the vectors object?" Good: "What \
distance metric does the create_collection example in the Python client quickstart configure?"

Return only a JSON array. Each item:
{{"type": "single", "question": "...", "answer": "...", "aliases": ["other acceptable short forms of the answer"], \
"facts": [3]}}

Answers: a few words for single and multi, one or two sentences for abstract, exactly "not in the text" for absent (with \
"facts": [])."""

USER_FAR = """Document: {title}
Kind: {kind}

Facts from DIFFERENT, far apart parts of this document, in document order (numbered):
{facts}

Write {n_far} "far" questions. Each must need two or three facts whose numbers are at least 5 apart (so the answer combines
distant parts of the document: compare, combine, follow a reference from one part to another). Ask naturally, in your own
words. Each question must make sense on its own to someone searching a large collection of many documents: name the subject and
where it lives (module, file, project; or the person, organization, event by name); never "it", "this", "the document".
Then write 1 "absent": plausible for this document, but the facts do NOT contain the answer.

Return only a JSON array. Each item:
{{"type": "far", "question": "...", "answer": "...", "aliases": ["other acceptable short forms"], "facts": [2, 17]}}

Answers: a few words (at most one short sentence) for far, exactly "not in the text" for absent (with "facts": [])."""

TYPES = ("single", "multi", "abstract", "absent", "far")


def held_out(doc_id):
    return int(hashlib.md5(doc_id.encode()).hexdigest(), 16) % 1000 < 50


async def amain(a):
    from openai import AsyncOpenAI

    t0 = time.time()
    log = lambda s: print(f"{s} ({time.time() - t0:.0f}s)", flush=True)
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            try:
                r = json.loads(line)
                done.add((r["doc_id"], r["group"]))
            except (json.JSONDecodeError, KeyError):
                pass
    jobs = []
    for line in (x for p in a.facts.split(",") for x in open(p)):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        fs = [f for f in r["facts"] if f.get("ok")]
        for g0 in range(0, len(fs), a.group):
            grp = fs[g0:g0 + a.group]
            if len(grp) < a.min_facts or (r["doc_id"], g0) in done:
                continue
            jobs.append((r, g0, grp))
        if a.far and len(fs) >= 2 * a.group and (r["doc_id"], -1) not in done:        # far group: evenly over the document
            idx = sorted({round(j * (len(fs) - 1) / (a.far - 1)) for j in range(a.far)})
            jobs.append((r, -1, [dict(fs[j], _i=j) for j in idx]))
        if a.limit and len(jobs) >= a.limit:
            break
    log(f"groups to process {len(jobs)} (done before {len(done)})")
    client = AsyncOpenAI(base_url=a.server, api_key="none", timeout=900)
    sem = asyncio.Semaphore(a.sessions)
    st = collections.Counter()
    fout = open(a.out, "a")

    async def run_group(r, g0, grp):
        async with sem:
            n_single, n_multi = (3, 2) if len(grp) >= 12 else (2, 1)
            text = "\n".join(f"[{k}] {f['statement']}" for k, f in enumerate(grp))
            if g0 < 0:
                u = USER_FAR.format(title=r["title"], kind=r["kind"], facts=text, n_far=3)
            else:
                u = USER.format(title=r["title"], kind=r["kind"], facts=text, n_single=n_single, n_multi=n_multi)
            msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": u}]
            resp = None
            for attempt in range(4):
                try:
                    resp = await client.chat.completions.create(
                        model=a.model_name, messages=msgs, temperature=a.temperature, max_tokens=a.max_tokens,
                        extra_body={"chat_template_kwargs": {"enable_thinking": False}})
                    break
                except Exception as e:
                    st["retry"] += 1
                    if attempt == 3:
                        log(f"  request failed ({r['doc_id']} group {g0}): {e!r}"[:300])
                    await asyncio.sleep(3 * (attempt + 1))
            st["groups"] += 1
            if resp is None:
                st["req_fail"] += 1
                return
            ch = resp.choices[0]
            st["out_tok"] += resp.usage.completion_tokens if resp.usage else 0
            st["trunc"] += ch.finish_reason == "length"
            items, bad = parse(ch.message.content or "")
            st["bad_json"] += bad
            qs = []
            for it in items:
                if not isinstance(it, dict) or it.get("type") not in TYPES or \
                        not isinstance(it.get("question"), str) or not isinstance(it.get("answer"), (str, int, float)):
                    st["bad_item"] += 1
                    continue
                raw = it.get("facts") if isinstance(it.get("facts"), list) else []
                ids = [int(k) for k in raw if (isinstance(k, int) or (isinstance(k, str) and k.strip().isdigit()))
                       and 0 <= int(k) < len(grp)]
                if it["type"] in ("single", "multi", "far") and not ids:
                    st["no_facts"] += 1
                    continue
                if it["type"] == "absent":
                    ids = []
                al = [str(x).strip() for x in it.get("aliases", []) if isinstance(x, (str, int, float)) and str(x).strip()] \
                    if isinstance(it.get("aliases"), list) else []
                qs.append({"type": it["type"], "question": it["question"].strip(), "answer": str(it["answer"]).strip(),
                           "aliases": al, "facts": [grp[k]["_i"] if g0 < 0 else g0 + k for k in ids],
                           "anchors": [[grp[k]["a_start"], grp[k]["a_end"]] for k in ids]})
                st[it["type"]] += 1
            fout.write(json.dumps({"doc_id": r["doc_id"], "kind": r["kind"], "title": r["title"], "group": g0,
                                   "split": "held" if held_out(r["doc_id"]) else "train", "n_facts": len(grp),
                                   "questions": qs}, ensure_ascii=False) + "\n")
            fout.flush()

    async def reporter():
        while True:
            await asyncio.sleep(a.report)
            el = time.time() - t0
            log(f"  groups {st['groups']}/{len(jobs)} ({st['groups'] / el:.2f}/s), questions: "
                + ", ".join(f"{t} {st[t]}" for t in TYPES) + f"; without facts {st['no_facts']}, bad items "
                f"{st['bad_item']}, bad JSON {st['bad_json']}, truncated {st['trunc']}, output tokens/s {st['out_tok'] / el:.0f}")

    rep = asyncio.create_task(reporter())
    await asyncio.gather(*(run_group(*j) for j in jobs))
    rep.cancel()
    log(f"  total: groups {st['groups']}, questions " + ", ".join(f"{t} {st[t]}" for t in TYPES)
        + f", failed requests {st['req_fail']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model-name", default="qwen", help="model name served by the server")
    ap.add_argument("--facts", default="data/train/facts.jsonl", help="facts.py output; several files comma-separated")
    ap.add_argument("--out", default="data/train/questions.jsonl")
    ap.add_argument("--group", type=int, default=40, help="facts per group")
    ap.add_argument("--min-facts", type=int, default=4, help="smaller groups are skipped")
    ap.add_argument("--far", type=int, default=24, help="facts in the far group of a document (0 — no far questions)")
    ap.add_argument("--sessions", type=int, default=96); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=1600); ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--report", type=int, default=120)
    asyncio.run(amain(ap.parse_args()))


if __name__ == "__main__":
    main()
