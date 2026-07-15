from pathlib import Path
from typing import Literal

import rich
import torch
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
    TextStreamer,
)
from transformers.cache_utils import CacheLayerMixin, LinearAttentionCacheLayerMixin

from kv_search.data import Datasets, Message, load_dataset
from kv_search.query_aware_cache import (
    CacheState,
    LayerState,
    LinearLayerState,
    QueryAwareCache,
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


def _do_prefill(
    messages: Message,
    model: ModelType,
    processor: ProcessorType,
    past_key_values: QueryAwareCache,
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
    past_key_values = QueryAwareCache(config=model.config)

    streamer = TextStreamer(processor.tokenizer)

    messages = load_dataset(dataset_name, multimodal=model_name in IS_MULTIMODAL)

    _do_prefill(messages, model, processor, past_key_values)

    print(f"{past_key_values.get_seq_length()=}")
    prefill_length = past_key_values.get_seq_length()

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

    outputs = model.generate(
        **inputs,  # ty:ignore[invalid-argument-type]
        max_new_tokens=256,
        past_key_values=past_key_values,
        use_cache=True,
        streamer=streamer,
    )  # ty:ignore[invalid-argument-type]
    print(processor.decode(outputs[0][inputs["input_ids"].shape[-1] :]))
    print(f"{past_key_values.get_seq_length()=}")

    state = CacheState(context_len=prefill_length, layers=[])
    for i, layer in enumerate(past_key_values.layers):
        match layer:
            case CacheLayerMixin(keys=keys, values=values):
                state.layers.append(
                    LayerState(
                        idx=i,
                        queries=past_key_values.queries[i],
                        keys=keys,
                        values=values,
                    )
                )
            case LinearAttentionCacheLayerMixin(
                conv_states=conv_states, recurrent_states=rec_states
            ):
                state.layers.append(
                    LinearLayerState(
                        idx=i, conv_states=conv_states, rec_states=rec_states
                    )
                )

    cache_dir = Path(f"cache/{dataset_name}/{model.config.model_type}")
    cache_dir.mkdir(exist_ok=True, parents=True)

    state.save(cache_dir)


if __name__ == "__main__":
    main("Qwen/Qwen3.5-9B", dataset_name=Datasets.QDRANT)
