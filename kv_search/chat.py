import contextlib
import io
import os

os.environ.setdefault("HF_HUB_VERBOSITY", "error")

from pathlib import Path
from typing import Literal

import torch
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Batch,
    Distance,
    HnswConfigDiff,
    VectorParams,
)
from rich.progress import track
from rich.prompt import Prompt

import transformers.utils.logging

transformers.utils.logging.set_verbosity(transformers.utils.logging.CRITICAL)

# auto_docstring emits [ERROR] lines via print() at class-definition time
with contextlib.redirect_stdout(io.StringIO()):
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

    from kv_search.data import Datasets, Message, load_dataset
    from kv_search.query_aware_cache import (
        CacheState,
        CutoffCache,
        LayerState,
        QdrantCache,
        bind_query_aware_cache,
    )

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
    state = CacheState.load(Path(f"cache/qdrant/qwen3_5/"))
    for layer in track(state.layers, transient=True, description="Layers"):
        if not isinstance(layer, LayerState):
            continue

        for h in track(range(4), transient=True, description="Heads"):
            if client.collection_exists(f"layer={layer.idx};head={h}"):
                client.delete_collection(f"layer={layer.idx};head={h}")

            client.create_collection(
                collection_name=f"layer={layer.idx};head={h}",
                vectors_config={
                    "key": VectorParams(
                        size=layer.keys.shape[-1], distance=Distance.DOT
                    ),
                    "value": VectorParams(
                        size=layer.keys.shape[-1],
                        distance=Distance.DOT,
                        hnsw_config=HnswConfigDiff(m=0),
                    ),
                },
            )

            ids = torch.split(torch.arange(state.context_len, dtype=torch.int), 128)
            keys = torch.split(
                layer.keys[0, h, : state.context_len].to(torch.float), 128
            )
            values = torch.split(
                layer.values[0, h, : state.context_len].to(torch.float), 128
            )

            for idx, k, v in track(
                zip(ids, keys, values),
                total=len(keys),
                transient=True,
                description="Batch",
            ):
                client.upsert(
                    collection_name=f"layer={layer.idx};head={h}",
                    points=Batch(
                        ids=idx.tolist(),
                        vectors={
                            "key": k.numpy(),
                            "value": v.numpy(),
                        },
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
            .cuda()  # ty:ignore[missing-argument]
            .eval()
        )

    bind_query_aware_cache(model)
    past_key_values = QdrantCache("localhost", config=model.config)
    # past_key_values = CutoffCache(128, config=model.config)

    streamer = TextStreamer(processor.tokenizer, skip_prompt=True)

    # messages = load_dataset(dataset_name, multimodal=model_name in IS_MULTIMODAL)
    # rich.print(messages.query)
    while True:
        user = input("\n> ")
        print("\n")
        inputs: BatchEncoding[torch.Tensor] = processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "text", "text": user}]}],  # ty:ignore[invalid-argument-type]
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        )  # ty:ignore[invalid-assignment]
        inputs = inputs.to(model.device)

        model.generate(
            **inputs,  # ty:ignore[invalid-argument-type]
            max_new_tokens=256,
            past_key_values=past_key_values,
            use_cache=True,
            streamer=streamer,
        )  # ty:ignore[invalid-argument-type]
        past_key_values.reset()


if __name__ == "__main__":
    # upsert()
    main("Qwen/Qwen3.5-9B", dataset_name=Datasets.QDRANT)
