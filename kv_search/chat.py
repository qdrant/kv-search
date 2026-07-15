import compression.zstd
from pathlib import Path
from typing import Literal

import numpy as np
import rich
import torch
from qdrant_client import QdrantClient
from qdrant_client.models import Batch, Distance, PointStruct, VectorParams
from rich.progress import track
from safetensors.torch import load, save
from transformers import (
    AutoModelForCausalLM,
    AutoModelForMultimodalLM,
    AutoProcessor,
    BatchEncoding,
    Gemma3ForConditionalGeneration,
    Gemma3Processor,
    Gemma4ForConditionalGeneration,
    Gemma4Processor,
    Mistral3ForConditionalGeneration,
    PixtralProcessor,
    Qwen2ForCausalLM,
    Qwen2Tokenizer,
    Qwen3_5ForConditionalGeneration,
    Qwen3VLProcessor,
    TextStreamer,
)
from transformers.cache_utils import CacheLayerMixin

from kv_search.data import Datasets, Message, load_dataset
from kv_search.query_aware_cache import QdrantCache, bind_query_aware_cache, CutoffCache

IS_MULTIMODAL = {
    "Qwen/Qwen3.5-9B",
    "mistralai/Ministral-3-8B-Instruct-2512",
    "google/gemma-3-4b-it",
    "google/gemma-4-12B-it",
}


ModelType = (
    Qwen3_5ForConditionalGeneration
    | Qwen2ForCausalLM
    | Mistral3ForConditionalGeneration
    | Gemma3ForConditionalGeneration
    | Gemma4ForConditionalGeneration
)
ProcessorType = (
    Qwen3VLProcessor
    | Qwen2Tokenizer
    | PixtralProcessor
    | Gemma3Processor
    | Gemma4Processor
)

client = QdrantClient("localhost")


def upsert():
    for i in track(range(8), transient=True, description="Layers"):
        with compression.zstd.open(
            f"cache/qdrant/qwen3_5/layer_{i}_tensors.safetensors.zst",
            "rb",
        ) as f:
            tensors: dict[str, torch.Tensor] = load(f.read())

        for h in track(range(4), transient=True, description="Heads"):
            if client.collection_exists(f"layer={i};head={h}"):
                client.delete_collection(f"layer={i};head={h}")

            client.create_collection(
                collection_name=f"layer={i};head={h}",
                vectors_config=VectorParams(
                    size=tensors["keys"].shape[-1], distance=Distance.COSINE
                ),
            )

            length: int = tensors["prefill_length"]

            ids = torch.split(torch.arange(length, dtype=torch.int), 128)
            keys = torch.split(tensors["keys"][0, h, :length].to(torch.float), 128)
            values = torch.split(tensors["values"][0, h, :length].to(torch.float), 128)

            for idx, k, v in track(
                zip(ids, keys, values),
                total=len(keys),
                transient=True,
                description="Batch",
            ):
                client.upsert(
                    collection_name=f"layer={i};head={h}",
                    points=Batch(
                        ids=idx.tolist(),
                        vectors=k.numpy(),
                        payloads=[{"value": t} for t in v.tolist()],
                    ),
                )


def main(
    model_name: Literal[
        "Qwen/Qwen3.5-9B",
        "Qwen/Qwen2.5-7B",
        "mistralai/Ministral-3-8B-Instruct-2512",
        "google/gemma-3-4b-it",
        "google/gemma-4-12B-it",
    ],
    dataset_name: Datasets,
):
    processor: ProcessorType = AutoProcessor.from_pretrained(model_name)
    if model_name in IS_MULTIMODAL:
        model: ModelType = (
            AutoModelForMultimodalLM.from_pretrained(
                model_name,
                attn_implementation="sdpa",
                dtype=torch.bfloat16,
            )
            .cuda()
            .eval()
        )
    else:
        model: ModelType = (
            AutoModelForCausalLM.from_pretrained(
                model_name,
                attn_implementation="sdpa",
                dtype=torch.bfloat16,
            )
            .cuda()
            .eval()
        )  # ty:ignore[missing-argument]

    bind_query_aware_cache(model)
    # past_key_values = QdrantCache("localhost", config=model.config)
    past_key_values = CutoffCache(1024, config=model.config)

    streamer = TextStreamer(processor.tokenizer)

    messages = load_dataset(dataset_name, multimodal=model_name in IS_MULTIMODAL)

    rich.print(messages.query)
    inputs: BatchEncoding[torch.Tensor] = processor.apply_chat_template(
        messages.query,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
    )  # ty:ignore[invalid-assignment]
    inputs["attention_mask"] = torch.cat(
        [
            torch.ones(1, past_key_values.get_seq_length(), dtype=torch.long),
            inputs["attention_mask"],
        ],
        dim=1,
    )
    inputs = inputs.to(model.device)

    model.generate(
        **inputs,  # ty:ignore[invalid-argument-type]
        max_new_tokens=256,
        past_key_values=past_key_values,
        use_cache=True,
        streamer=streamer,
    )  # ty:ignore[invalid-argument-type]


if __name__ == "__main__":
    # upsert()
    main("Qwen/Qwen3.5-9B", dataset_name=Datasets.QDRANT)
