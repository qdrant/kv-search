"""Synthetic facts: an LLM reads each document chunk by chunk and lists the facts it states.

A document is cut into chunks of about --chunk tokens of the reader's tokenizer, ending at a line end or, failing that, a
sentence end. For each chunk in order the LLM lists the NEW facts of the chunk, given the running list of the last --known
facts already found in this document (a refinement of a known fact is a new fact that refers to it). Per fact:
  statement  the fact in other words, self-contained (names instead of pronouns);
  anchor     a verbatim quote of the chunk where the fact is stated; its position is found by the program (exact, then with
             any whitespace, then ignoring case, then the longest line of a code anchor); facts without it get ok = false;
  value      the detail a question would ask for (number, name, date, term, identifier); value_ok = it occurs in the statement;
  refines    the number of the refined known fact or null.
The LLM is an OpenAI-compatible server with continuous batching (vLLM); every document is its own async session that sends its
next chunk as soon as the previous answer arrives, so the server batch stays full. Test and fixture files of code are skipped.
--part a/b/n takes the documents whose line number i has a ≤ i mod n < b, to split the work between servers. Output: one JSONL
line per document, appended; a restart skips finished documents.

    python -m llm_memory.datagen.facts --server http://127.0.0.1:8000/v1 --docs data/train/docs-facts.jsonl \\
        --out data/train/facts.jsonl
"""
import argparse
import asyncio
import collections
import json
import os
import re
import time

SYSTEM = ("You extract facts from documents to build a retrieval dataset. You are precise: anchors are copied "
          "character by character from the passage, and you never invent facts that the passage does not state.")

USER = """Document: {title}
Kind: {kind}

Known facts from earlier parts of this document:
{known}

Passage:
<<<
{passage}
>>>

List every NEW fact that this passage states and that is not already among the known facts. A refinement, detail or \
correction of a known fact counts as a new fact; set "refines" to the number of that known fact.

Return only a JSON array. Each item:
{{"statement": "...", "anchor": "...", "value": "...", "refines": null}}

Rules:
- statement: one atomic fact, self-contained: always name the subject (the person, the book's title, the function's or \
type's name), never "it", "he", "this", "the novel", "the function". Paraphrase: use different wording and sentence \
structure than the passage, but keep names, numbers and identifiers exact.
- anchor: an exact verbatim quote from the passage, 4 to 15 words, where the fact is stated. Copy it character by \
character. For code, the anchor is a single line or part of one line.
- value: the specific detail of the statement that a question would ask for: a number, name, date, term or identifier. \
Copy it exactly as written in your statement.
- For code: state what types, functions, methods, parameters, defaults, return values, errors and behaviours are and do, \
using the exact identifiers.
- Skip boilerplate: license headers, import lists, navigation and formatting text.
- If the passage states no new facts, return [].

Example. Passage: "Ferris, the mascot, was adopted in 2015 by the Rust community." Item:
{{"statement": "The Rust community chose Ferris as its mascot in 2015.", "anchor": "Ferris, the mascot, was adopted in \
2015", "value": "2015", "refines": null}}"""

TEST_PATH = re.compile(r"(^|/)(tests?|testing|fixtures?|testdata|test_data|benches|__tests__|snapshots)/|(^|/)test_[^/]*$|"
                       r"_test\.\w+$|\.test\.\w+$|_spec\.\w+$|(^|/)spec/", re.I)


def chunks_of(text, tok, size):
    """Character spans of chunks of about `size` tokens; a chunk ends at a line end, else a sentence end, near the target."""
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    offs = enc["offset_mapping"]
    n, out, i = len(offs), [], 0
    while i < n:
        j = min(n, i + size)
        if j < n:
            lo, hi = i + (3 * size) // 4, min(n - 1, j + size // 6)
            best = None
            for k in range(hi, lo, -1):
                if "\n" in text[offs[k][0]:offs[k][1]]:
                    best = k + 1
                    break
            if best is None:
                for k in range(hi, lo, -1):
                    if text[offs[k][0]:offs[k][1]].rstrip().endswith((".", "!", "?", ";", "}")):
                        best = k + 1
                        break
            j = best or j
        out.append((offs[i][0], offs[j - 1][1]))
        i = j
    return out


def locate(anchor, chunk):
    """Position of the anchor in the chunk: (start, end, how) or None."""
    anchor = (anchor or "").replace("\\n", "\n").replace("\\t", "\t").strip().strip('"`').strip()
    if len(anchor) < 8:
        return None
    p = chunk.find(anchor)
    if p >= 0:
        return p, p + len(anchor), "exact"
    words = anchor.split()
    pat = r"\s+".join(re.escape(w) for w in words)
    for flags, how in ((0, "space"), (re.I, "case")):
        m = re.search(pat, chunk, flags)
        if m:
            return m.start(), m.end(), how
    line = max(anchor.split("\n"), key=len).strip()                                  # the longest line of a code anchor
    if len(line) >= 12 and line != anchor:
        p = chunk.find(line)
        if p >= 0:
            return p, p + len(line), "line"
    return None


def parse(text):
    """JSON array from the model's answer: (list, parse_error)."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    a, b = text.find("["), text.rfind("]")
    if a < 0 or b <= a:
        return [], text.strip() not in ("", "[]")
    s = text[a:b + 1]
    for cand in (s, re.sub(r",\s*([\]}])", r"\1", s)):
        try:
            v = json.loads(cand)
            return (v if isinstance(v, list) else []), False
        except json.JSONDecodeError:
            continue
    return [], True


async def amain(a):
    from openai import AsyncOpenAI
    from transformers import AutoTokenizer

    t0 = time.time()
    log = lambda s: print(f"{s} ({time.time() - t0:.0f}s)", flush=True)
    tok = AutoTokenizer.from_pretrained(a.tok)
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            try:
                done.add(json.loads(line)["doc_id"])
            except (json.JSONDecodeError, KeyError):
                pass
    lo, hi, n = (int(x) for x in a.part.split("/"))
    docs, skipped = [], collections.Counter()
    for i, line in enumerate(open(a.docs)):
        if not lo <= i % n < hi:
            continue
        d = json.loads(line)
        if d["id"] in done:
            continue
        if d["kind"] in ("code", "docs") and TEST_PATH.search(d["title"]):
            skipped["test"] += 1
            continue
        docs.append(d)
        if a.limit and len(docs) >= a.limit:
            break
    log(f"documents to process {len(docs)} (done before {len(done)}, test files skipped {skipped['test']})")
    client = AsyncOpenAI(base_url=a.server, api_key="none", timeout=900)
    sem = asyncio.Semaphore(a.sessions)
    st = collections.Counter()
    fout = open(a.out, "a")

    def prompt(d, chunk, fs):
        lo_ = max(0, len(fs) - a.known)
        known = "\n".join(f"[{k}] {fs[k]['statement']}" for k in range(lo_, len(fs))) or "(none yet)"
        return [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": USER.format(title=d["title"], kind=d["kind"], known=known, passage=chunk)}]

    async def run_doc(d):
        async with sem:
            chunks = chunks_of(d["text"], tok, a.chunk)
            facts, bad = [], 0
            for ci, (c0, c1) in enumerate(chunks):
                chunk = d["text"][c0:c1]
                r = None
                for attempt in range(4):
                    try:
                        r = await client.chat.completions.create(
                            model=a.model_name, messages=prompt(d, chunk, facts), temperature=a.temperature,
                            max_tokens=a.max_tokens, extra_body={"chat_template_kwargs": {"enable_thinking": False}})
                        break
                    except Exception as e:
                        st["retry"] += 1
                        if attempt == 3:
                            log(f"  request failed ({d['id']} chunk {ci}): {e!r}"[:300])
                        await asyncio.sleep(3 * (attempt + 1))
                st["chunks"] += 1
                if r is None:
                    st["req_fail"] += 1
                    continue
                ch = r.choices[0]
                st["out_tok"] += r.usage.completion_tokens if r.usage else 0
                st["trunc"] += ch.finish_reason == "length"
                items, b = parse(ch.message.content or "")
                bad += b
                st["bad_json"] += b
                for it in items:
                    if not isinstance(it, dict) or not isinstance(it.get("statement"), str) or not it["statement"].strip():
                        st["bad_item"] += 1
                        continue
                    stmt = it["statement"].strip()
                    anchor = it.get("anchor") if isinstance(it.get("anchor"), str) else ""
                    value = it.get("value") if isinstance(it.get("value"), (str, int, float)) else ""
                    value = str(value).strip()
                    ref = it.get("refines")
                    ref = ref if isinstance(ref, int) and 0 <= ref < len(facts) else None
                    loc = locate(anchor, chunk)
                    f = {"statement": stmt, "anchor": anchor, "value": value,
                         "value_ok": bool(value) and value.lower() in stmt.lower(), "refines": ref, "chunk": ci,
                         "ok": loc is not None}
                    if loc:
                        f["a_start"], f["a_end"], f["how"] = c0 + loc[0], c0 + loc[1], loc[2]
                        st["anchor_found"] += 1
                        st["anchor_exact"] += loc[2] == "exact"
                    facts.append(f)
                    st["facts"] += 1
            fout.write(json.dumps({"doc_id": d["id"], "kind": d["kind"], "title": d["title"], "chunks": chunks,
                                   "facts": facts, "bad_json": bad}, ensure_ascii=False) + "\n")
            fout.flush()
            st["docs"] += 1

    async def reporter():
        while True:
            await asyncio.sleep(a.report)
            el = time.time() - t0
            f = max(1, st["facts"])
            log(f"  documents {st['docs']}/{len(docs)}, chunks {st['chunks']} ({st['chunks'] / el:.2f}/s), facts "
                f"{st['facts']} ({st['facts'] / max(1, st['chunks']):.1f} per chunk), anchor found "
                f"{st['anchor_found'] / f:.3f} (exact {st['anchor_exact'] / f:.3f}), bad JSON {st['bad_json']}, "
                f"truncated {st['trunc']}, request retries {st['retry']}, output tokens/s {st['out_tok'] / el:.0f}")

    rep = asyncio.create_task(reporter())
    await asyncio.gather(*(run_doc(d) for d in docs))
    rep.cancel()
    el = time.time() - t0
    log(f"  total: documents {st['docs']}, chunks {st['chunks']} ({st['chunks'] / el:.2f}/s), facts {st['facts']}, "
        f"anchor found {st['anchor_found'] / max(1, st['facts']):.3f}, failed requests {st['req_fail']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model-name", default="qwen", help="model name served by the server")
    ap.add_argument("--docs", default="data/train/docs-facts.jsonl"); ap.add_argument("--out", default="data/train/facts.jsonl")
    ap.add_argument("--part", default="0/1/1", help="a/b/n: documents whose line number i has a ≤ i mod n < b")
    ap.add_argument("--tok", default="Qwen/Qwen3-4B-Instruct-2507", help="tokenizer for chunking (the reader's)")
    ap.add_argument("--chunk", type=int, default=256); ap.add_argument("--known", type=int, default=24)
    ap.add_argument("--sessions", type=int, default=160, help="concurrent document sessions")
    ap.add_argument("--limit", type=int, default=0, help="at most this many documents (0 — all)")
    ap.add_argument("--max-tokens", type=int, default=2000); ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--report", type=int, default=120, help="seconds between progress lines")
    asyncio.run(amain(ap.parse_args()))


if __name__ == "__main__":
    main()
