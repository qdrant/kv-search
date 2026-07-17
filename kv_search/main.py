import importlib.util
from enum import Enum, auto
from pathlib import Path
from typing import Any, Literal

import rich
import torch
from pydantic import BaseModel, Field
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
    PreTrainedTokenizerBase,
    Qwen2ForCausalLM,
    Qwen2Tokenizer,
    Qwen3_5ForConditionalGeneration,
    Qwen3VLProcessor,
    TextStreamer,
)

from kv_search.data import Datasets, Message, load_dataset
from kv_search.query_aware_cache import (
    QdrantRetriever,
    RecordingCache,
    RetrievalCache,
    RetrieverConfig,
    bind_query_aware_cache,
    load_cache,
    save_cache,
)
from kv_search.timer import timers

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

_ATTN_IMPL = (
    "flash_attention_2"
    if importlib.util.find_spec("flash_attn") is not None
    else "sdpa"
)


@timers.model_load
def _load_model(model_name: ModelName) -> tuple[ModelType, ProcessorType]:
    processor: ProcessorType = AutoProcessor.from_pretrained(model_name)
    if model_name in IS_MULTIMODAL:
        model: ModelType = AutoModelForMultimodalLM.from_pretrained(
            model_name,
            attn_implementation=_ATTN_IMPL,
            dtype=torch.bfloat16,
            device_map="cuda",
        ).eval()
    else:
        model: ModelType = AutoModelForCausalLM.from_pretrained(
            model_name,
            attn_implementation=_ATTN_IMPL,
            dtype=torch.bfloat16,
            device_map="cuda",
        ).eval()

    bind_query_aware_cache(model)
    return model, processor


@timers.prefill_gen
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

        with (
            torch.no_grad(),
            torch.nn.attention.sdpa_kernel(
                [
                    torch.nn.attention.SDPBackend.CUDNN_ATTENTION,
                    torch.nn.attention.SDPBackend.FLASH_ATTENTION,
                    torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
                ],
                set_priority=True,
            ),
        ):
            past_key_values = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **additional_args,
                past_key_values=past_key_values,
                logits_to_keep=1,
            ).past_key_values


class TimedStreamer(TextStreamer):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        skip_prompt: bool = False,
        **decode_kwargs: Any,
    ):
        super().__init__(tokenizer, skip_prompt, **decode_kwargs)

    def put(self, value):
        timers.token_gen.record()
        super().put(value)


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

        cache_dir = Path(f"cache/{self.dataset_name}/{model.config.model_type}")
        cache_dir.mkdir(exist_ok=True, parents=True)

        past_key_values = RecordingCache(path=cache_dir, config=model.config)
        messages = load_dataset(
            self.dataset_name, multimodal=self.model_name in IS_MULTIMODAL
        )

        _do_prefill(messages, model, processor, past_key_values)

        save_cache(
            past_key_values,
            cache_dir,
            past_key_values.get_seq_length(),
        )
        past_key_values.finalize()
        rich.print(timers)
        rich.print(
            f"peak allocated: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB"
        )
        rich.print(
            f"peak reserved:  {torch.cuda.max_memory_reserved() / 2**30:.2f} GiB"
        )


class Chat(BaseModel):
    model_name: ModelName = "Qwen/Qwen3.5-9B"
    dataset_name: Datasets = Datasets.QDRANT
    retriever: RetrieverConfig = Field(default_factory=QdrantRetriever)
    max_new_tokens: int = 256

    def cli_cmd(self) -> None:
        model, processor = _load_model(self.model_name)

        cache_dir = Path(f"cache/{self.dataset_name}/{model.config.model_type}")
        cache_dir.mkdir(exist_ok=True, parents=True)

        prefill, context_len = load_cache(cache_dir, model.config)

        past_key_values = RetrievalCache(
            retriever=self.retriever, prefill=prefill, config=model.config
        )

        streamer = TimedStreamer(processor.tokenizer, skip_prompt=True)

        # messages = load_dataset(dataset_name, multimodal=model_name in IS_MULTIMODAL)
        # rich.print(messages.query)
        while True:
            try:
                user = input("\n> ")
            except KeyboardInterrupt:
                break

            try:
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

                # offset positional embeddings to beyond prefill content
                prompt_len = inputs["input_ids"].shape[1]
                inputs["position_ids"] = torch.arange(
                    context_len, context_len + prompt_len, device=model.device
                ).unsqueeze(0)

                model.generate(
                    **inputs,  # ty:ignore[invalid-argument-type]
                    max_new_tokens=self.max_new_tokens,
                    past_key_values=past_key_values,
                    use_cache=True,
                    streamer=streamer,
                )  # ty:ignore[invalid-argument-type]
            except KeyboardInterrupt:
                continue
            finally:
                past_key_values.reset()

        rich.print(timers)
        rich.print(
            f"peak allocated: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB"
        )
        rich.print(
            f"peak reserved:  {torch.cuda.max_memory_reserved() / 2**30:.2f} GiB"
        )


class KvSearch(
    BaseModel,
    cli_shortcuts={
        "retriever.url": "url",
        "retriever.n-retrieved": "n",
        "max-new-tokens": "g",
        "model-name": "m",
        "dataset-name": "d",
        "retriever.api-key": "api-key",
        "retriever.type": "r",
    },
):
    prefill: CliSubCommand[Prefill]
    chat: CliSubCommand[Chat]

    def cli_cmd(self) -> None:
        CliApp.run_subcommand(self)


def main() -> None:
    CliApp.run(KvSearch)


if __name__ == "__main__":
    main()
