import random
import uuid
from collections.abc import Iterator
from enum import StrEnum
from pathlib import Path
from typing import Any

import datasets
from pydantic import BaseModel


class Datasets(StrEnum):
    SQUAD = "squad"
    QDRANT = "qdrant"


class Message(BaseModel):
    prefill: list[dict[str, Any]]
    query: list[dict[str, Any]]


class EvalExample(BaseModel):
    prefill: list[dict[str, Any]]
    query: list[dict[str, Any]]
    label: str
    bucket: int
    lang: str
    idx: int


def _wrap_multimodal(messages: list[dict[str, Any]]) -> None:
    for message in messages:
        if isinstance(message["content"], str):
            message["content"] = [{"type": "text", "text": message["content"]}]


# this comes from RULER
_FILLER = (
    "The grass is green. The sky is blue. The sun is yellow. "
    "Here we go. There and back again. "
)


def generate_niah_examples(
    tokenizer: Any,
    length: int,
    n_keys: int = 1,
    depth: float = 0.5,
    n_examples: int = 5,
    seed: int = 0,
    multimodal: bool = False,
) -> Iterator[EvalExample]:
    """NIAH: n_keys same-template needles in filler, query asks one (n_keys=1 is
    single NIAH). length is the approx prefill token budget; depth the position."""

    unit = len(tokenizer.encode(_FILLER, add_special_tokens=False)) or 1
    for idx in range(n_examples):
        rng = random.Random(f"{seed}-{idx}")
        keys = [str(uuid.UUID(int=rng.getrandbits(128))) for _ in range(n_keys)]
        values = [rng.randint(1_000_000, 9_999_999) for _ in range(n_keys)]
        needles = [
            f"The special magic number for {k} is: {v}."
            for k, v in zip(keys, values)
        ]
        target = min(int(depth * n_keys), n_keys - 1)

        needle_tokens = sum(
            len(tokenizer.encode(nd, add_special_tokens=False)) for nd in needles
        )
        reps = max((length - needle_tokens) // (n_keys + 1) // unit, 1)
        gap = _FILLER * reps

        parts = [gap]
        for nd in needles:
            parts.append(nd + " ")
            parts.append(gap)

        prefill = [{"role": "user", "content": "".join(parts)}]
        query = [{
            "role": "user",
            "content": f"What is the special magic number for {keys[target]}? "
            "Answer with just the number.",
        }]
        if multimodal:
            _wrap_multimodal(prefill)
            _wrap_multimodal(query)
        yield EvalExample(
            prefill=prefill,
            query=query,
            label=str(values[target]),
            bucket=length,
            lang="niah",
            idx=idx,
        )


def generate_qa_examples(
    tokenizer: Any,
    length: int,
    depth: float = 0.5,
    n_examples: int = 5,
    seed: int = 0,
    multimodal: bool = False,
) -> Iterator[EvalExample]:
    """SQuAD QA in a haystack of distractor SQuAD paragraphs. The gold paragraph
    is injected at depth; the question is paraphrastic so retrieval is semantic.
    length is the approx prefill token budget."""

    ds = datasets.load_dataset("rajpurkar/squad", split="validation")
    n = len(ds)
    for idx in range(n_examples):
        rng = random.Random(f"qa-{seed}-{idx}")
        item = ds[rng.randrange(n)]
        answer = item["answers"]["text"][0]
        gold = item["context"]

        budget = length - len(tokenizer.encode(gold, add_special_tokens=False))
        distractors: list[str] = []
        used = 0
        while used < budget:
            ctx = ds[rng.randrange(n)]["context"]
            if answer in ctx or ctx == gold:  # keep the answer unique to gold
                continue
            distractors.append(ctx)
            used += len(tokenizer.encode(ctx, add_special_tokens=False))

        pos = int(depth * len(distractors))
        haystack = "\n\n".join(distractors[:pos] + [gold] + distractors[pos:])

        prefill = [{"role": "user", "content": haystack}]
        query = [{
            "role": "user",
            "content": f"{item['question']} Answer with a short phrase.",
        }]
        if multimodal:
            _wrap_multimodal(prefill)
            _wrap_multimodal(query)
        yield EvalExample(
            prefill=prefill,
            query=query,
            label=answer,
            bucket=length,
            lang="qa",
            idx=idx,
        )


def load_dataset(
    dataset_name: Datasets, multimodal: bool = False, qdrant_size: str = "100k"
) -> Message:
    """Build one context message for prefill and a corresponding query message.

    ``qdrant_size`` selects the qdrant codebase-summary tier
    (``cache/CODEBASE_SUMMARY_{qdrant_size}.md``): one of 100k/200k/400k/600k/800k/1M.
    """

    if dataset_name == Datasets.QDRANT:
        path = Path(f"./cache/CODEBASE_SUMMARY_{qdrant_size}.md")

        if not path.is_file():
            raise RuntimeError(f"missing summary tier: {path}")
        with path.open("rt") as f:
            data = f.read()

        return Message(
            prefill=[
                {
                    "role": "user",
                    "content": [{"type": "text", "text": data}] if multimodal else data,
                }
            ],
            query=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "The codebase of what project do you have in your context?",
                        }
                    ]
                    if multimodal
                    else "The codebase of what project do you have in your context?",
                }
            ],
        )
    else:
        raise ValueError
