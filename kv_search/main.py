from enum import Enum, auto
from pathlib import Path
from typing import Literal

import torch
from pydantic import BaseModel
from pydantic_settings import CliApp, CliSubCommand
from rich.progress import track
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
)

from kv_search.data import Datasets, Message, load_dataset
from kv_search.query_aware_cache import (
    RecordingCache,
    bind_query_aware_cache,
    save_cache,
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


ModelName = Literal[
    "Qwen/Qwen3.5-9B",
    "Qwen/Qwen2.5-7B",
    "mistralai/Ministral-3-8B-Instruct-2512",
    "google/gemma-3-4b-it",
    "google/gemma-4-12B-it",
]


def _load_model(model_name: ModelName) -> tuple[ModelType, ProcessorType]:
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
    return model, processor


def _do_prefill(
    messages: Message,
    model: ModelType,
    processor: ProcessorType,
    past_key_values: RecordingCache,
):
    inputs: BatchEncoding[torch.Tensor] = processor.apply_chat_template(
        messages.prefill,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )  # ty:ignore[invalid-assignment]
    print(f"{inputs['input_ids'].shape=}")
    print(f"{len(past_key_values)=}")
    input_chunks = torch.split(inputs["input_ids"], 2048, -1)
    attention_masks = torch.split(inputs["attention_mask"], 2048, -1)
    if "mm_token_type_ids" in inputs:
        mm_token_type_chunks = torch.split(inputs["mm_token_type_ids"], 2048, -1)

    for i, (input_ids, attention_mask) in track(
        enumerate(zip(input_chunks, attention_masks)),
        total=len(input_chunks),
        description="Computing Prefill",
    ):
        input_ids = input_ids.to(model.device)
        attention_mask = attention_mask.to(model.device)

        additional_args = {}
        if "mm_token_type_ids" in inputs:
            additional_args["mm_token_type_ids"] = mm_token_type_chunks[i].to(
                model.device
            )

        with torch.no_grad():
            with torch.nn.attention.sdpa_kernel(
                [
                    torch.nn.attention.SDPBackend.FLASH_ATTENTION,
                    torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
                ]
            ):
                past_key_values = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **additional_args,
                    past_key_values=past_key_values,
                    logits_to_keep=1,
                ).past_key_values


class CacheImpl(Enum):
    TORCH = auto()
    QDRANT = auto()


class Prefill(BaseModel):
    model_name: ModelName = "Qwen/Qwen3.5-9B"
    dataset_name: Datasets = Datasets.QDRANT
    upsert: bool = False
    qdrant_url: str = "localhost"
    qdrant_api_key: str | None = None

    def cli_cmd(self) -> None:
        model, processor = _load_model(self.model_name)
        past_key_values = RecordingCache(config=model.config)
        messages = load_dataset(
            self.dataset_name, multimodal=self.model_name in IS_MULTIMODAL
        )

        _do_prefill(messages, model, processor, past_key_values)

        cache_dir = Path(f"cache/{self.dataset_name}/{model.config.model_type}")
        cache_dir.mkdir(exist_ok=True, parents=True)
        save_cache(
            past_key_values,
            cache_dir,
            past_key_values.get_seq_length(),
            past_key_values.queries,
        )


class Chat(BaseModel):
    model_name: ModelName = "Qwen/Qwen3.5-9B"
    dataset_name: Datasets = Datasets.QDRANT
    cache_impl: CacheImpl = CacheImpl.QDRANT
    n_retrieved: int | None = 128
    max_new_tokens: int = 256
    qdrant_url: str = "localhost"
    qdrant_api_key: str | None = None

    def cli_cmd(self) -> None:
        raise NotImplementedError


class KvSearch(BaseModel):
    prefill: CliSubCommand[Prefill]
    chat: CliSubCommand[Chat]

    def cli_cmd(self) -> None:
        CliApp.run_subcommand(self)


def main() -> None:
    CliApp.run(KvSearch)


if __name__ == "__main__":
    main()
