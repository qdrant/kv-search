import json
import warnings
from collections.abc import Iterator
from enum import StrEnum
from pathlib import Path
from typing import Any

import datasets
from huggingface_hub import snapshot_download
from pydantic import BaseModel


class Datasets(StrEnum):
    SQUAD = "squad"
    NIAH = "niah"
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


def load_niah_examples(
    lang: str = "english",
    bucket: int = 71680,
    limit: int = 0,
    multimodal: bool = False,
) -> Iterator[EvalExample]:
    """Yield scorable examples from one MR-NIAH bucket. limit=0 means all."""

    local_dir = Path(snapshot_download("MiniMaxAI/MR-NIAH", repo_type="dataset"))
    data_file = local_dir / lang / f"{bucket}_tokens.jsonl"
    if not data_file.is_file():
        raise RuntimeError(f"missing MR-NIAH file: {data_file}")

    yielded = 0
    with data_file.open("rt", encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            if not line.strip():
                continue
            if limit and yielded >= limit:
                break
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                # shipped files have a truncated final record
                warnings.warn(f"{data_file.name}: skip malformed line {line_no}: {e}")
                continue

            messages = data["messages"]
            if multimodal:
                _wrap_multimodal(messages)
            yield EvalExample(
                prefill=messages[:-1],
                query=[messages[-1]],
                label=data["label"],
                bucket=bucket,
                lang=lang,
                idx=line_no,
            )
            yielded += 1


def load_dataset(
    dataset_name: Datasets, multimodal: bool = False, qdrant_size: str = "100k"
) -> Message:
    """Build one context message for prefill and a corresponding query message.

    ``qdrant_size`` selects the qdrant codebase-summary tier
    (``cache/CODEBASE_SUMMARY_{qdrant_size}.md``): one of 100k/200k/400k/600k/800k/1M.
    """

    if dataset_name == Datasets.SQUAD:
        ds = datasets.load_dataset("rajpurkar/squad", split="train")
        contexts: list[str] = list({ds[i]["context"] for i in range(len(ds))})
        return Message(
            # HACK: empirically enough to get to 100k tokens
            prefill=[
                {
                    "role": "user",
                    "content": [{"type": "text", "text": context}]
                    if multimodal
                    else context,
                }
                for context in contexts[:650]
            ],
            query=[
                {
                    "role": "user",
                    "content": [{"type": "text", "text": ds[128]["question"]}]
                    if multimodal
                    else ds[128]["question"],
                }
            ],
        )
    elif dataset_name == Datasets.NIAH:
        local_dir = Path(snapshot_download("MiniMaxAI/MR-NIAH", repo_type="dataset"))
        # data_file = local_dir / "english/102400_tokens.jsonl"
        data_file = local_dir / "english/71680_tokens.jsonl"

        if not data_file.is_file():
            raise RuntimeError
        with data_file.open("rt") as f:
            data = json.loads(next(f))

        messages = data["messages"]
        if multimodal:
            for message in messages:
                message["content"] = [{"type": "text", "text": message["content"]}]
        return Message(prefill=messages[:-1], query=[messages[-1]])
    elif dataset_name == Datasets.QDRANT:
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
