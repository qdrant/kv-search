#!/usr/bin/env python3
"""Offset harness for the qdrant-summary needle set.

For every needle in cache/needles.json, locate its `anchor` in each
CODEBASE_SUMMARY_<size>.md and record the character offset and the Qwen token
offset of the fact within that summary. Writes cache/needles_offsets.json.

Because the summaries are exact byte-prefixes of each other, a needle present in
a small tier keeps the SAME offset in every larger tier -- so you can measure how
retrieval of a fixed position behaves as the surrounding context grows, and add
deep needles that only exist in the larger tiers for long-range probing.

NOTE on absolute KV position: these offsets are within the markdown content. The
model prefills the content wrapped by the chat template, so the absolute position
in the KV cache is `template_prefix_tokens + token_offset`. Pass the measured
prefix length via --template-prefix N (default 0) or add it at analysis time; the
value is constant for a given dataset/model and only shifts every needle equally.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

CACHE = Path(__file__).resolve().parent.parent / "cache"
SIZES = ["100k", "200k", "400k", "600k", "800k", "1M"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template-prefix", type=int, default=0,
                    help="constant #tokens the chat template inserts before the content")
    args = ap.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B")

    needles = json.loads((CACHE / "needles.json").read_text())["needles"]
    texts = {}
    for s in SIZES:
        p = CACHE / f"CODEBASE_SUMMARY_{s}.md"
        texts[s] = p.read_text(encoding="utf-8") if p.is_file() else None

    def toklen(s: str) -> int:
        return len(tok(s, add_special_tokens=False).input_ids)

    out = {}
    print(f"{'needle':<32} {'style':<6} " + " ".join(f"{s:>9}" for s in SIZES))
    for nd in needles:
        row = {}
        cells = []
        for s in SIZES:
            txt = texts[s]
            if txt is None:
                row[s] = {"present": False}
                cells.append(f"{'--':>9}")
                continue
            c = txt.count(nd["anchor"])
            if c == 0:
                row[s] = {"present": False}
                cells.append(f"{'.':>9}")
            elif c > 1:
                row[s] = {"present": True, "ambiguous": True, "count": c}
                cells.append(f"{'AMBIG!':>9}")
            else:
                ch = txt.index(nd["anchor"])
                tkn = toklen(txt[:ch]) + args.template_prefix
                row[s] = {"present": True, "char_offset": ch, "token_offset": tkn}
                cells.append(f"{tkn:>9,}")
        out[nd["id"]] = {"style": nd["style"], "first_tier": nd["first_tier"],
                         "question": nd["question"], "answer": nd["answer"],
                         "offsets": row}
        print(f"{nd['id']:<32} {nd['style']:<6} " + " ".join(cells))

    (CACHE / "needles_offsets.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {CACHE / 'needles_offsets.json'}  (token offsets; add "
          f"--template-prefix for absolute KV positions)")

    # sanity: any anchor missing from its declared first_tier, or ambiguous anywhere?
    problems = []
    for nid, v in out.items():
        ft = v["first_tier"]
        if not v["offsets"].get(ft, {}).get("present"):
            problems.append(f"{nid}: absent from declared first_tier {ft}")
        for s, cell in v["offsets"].items():
            if cell.get("ambiguous"):
                problems.append(f"{nid}: anchor ambiguous in {s} (count={cell['count']})")
    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print("  -", p)
    else:
        print("all anchors unique and present in their first tier ✓")


if __name__ == "__main__":
    main()
