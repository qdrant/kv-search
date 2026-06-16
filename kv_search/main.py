from typing import Literal
import rich
import torch

from kv_search.data import load_dataset, Datasets, Message
from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    Qwen3_5Model,
    Qwen2Model,
    Qwen3VLProcessor,
    Qwen2Tokenizer,
    BatchEncoding,
    Mistral3Model,
    Mistral3ForConditionalGeneration,
    Qwen3_5ForConditionalGeneration,
    Qwen2ForCausalLM,
    PixtralProcessor,
)
from transformers.cache_utils import CacheLayerMixin
from kv_search.query_aware_cache import QueryAwareCache, bind_query_aware_cache
from safetensors.torch import save_file

IS_MULTIMODAL = {"qwen3_5"}


ModelType = (
    Qwen3_5ForConditionalGeneration
    | Qwen2ForCausalLM
    | Mistral3ForConditionalGeneration
)
ProcessorType = Qwen3VLProcessor | Qwen2Tokenizer | PixtralProcessor


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
    input_chunks = torch.split(inputs["input_ids"], 1024, -1)
    attention_masks = torch.split(inputs["attention_mask"], 1024, -1)
    if "mm_token_type_ids" in inputs:
        mm_token_type_chunks = torch.split(inputs["mm_token_type_ids"], 1024, -1)

    for i, (input_ids, attention_mask) in enumerate(zip(input_chunks, attention_masks)):
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
        "Qwen/Qwen3.5-9B", "Qwen/Qwen2.5-7B", "mistralai/Ministral-3-8B-Reasoning-2512"
    ],
    dataset_name: Datasets,
):
    processor: ProcessorType = AutoProcessor.from_pretrained(model_name)
    if model_name == "mistralai/Ministral-3-8B-Reasoning-2512":
        model: ModelType = (
            Mistral3ForConditionalGeneration.from_pretrained(
                model_name, attn_implementation="sdpa", dtype=torch.bfloat16
            )
            .eval()
            .to("cuda")
        )
    else:
        model: ModelType = (
            AutoModelForCausalLM.from_pretrained(
                model_name, attn_implementation="sdpa", dtype=torch.bfloat16
            )
            .eval()
            .to("cuda")
        )

    bind_query_aware_cache(model)
    past_key_values = QueryAwareCache(config=model.config)

    messages = load_dataset(
        dataset_name, multimodal=model.config.model_type in ["qwen3_5", "mistral3"]
    )

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
    )  # ty:ignore[invalid-assignment]
    inputs["input_ids"] = torch.cat(
        [
            torch.zeros(1, past_key_values.get_seq_length(), dtype=torch.long),
            inputs["input_ids"],
        ],
        dim=1,
    )
    inputs["attention_mask"] = torch.cat(
        [
            torch.ones(1, past_key_values.get_seq_length(), dtype=torch.long),
            inputs["attention_mask"],
        ],
        dim=1,
    )
    if "mm_token_type_ids" in inputs:
        inputs["mm_token_type_ids"] = torch.cat(
            [
                torch.zeros(1, past_key_values.get_seq_length(), dtype=torch.long),
                inputs["mm_token_type_ids"],
            ],
            dim=1,
        )
    inputs = inputs.to(model.device)

    outputs = model.generate(
        **inputs,  # ty:ignore[invalid-argument-type]
        max_new_tokens=128,
        past_key_values=past_key_values,
        use_cache=True,
    )  # ty:ignore[invalid-argument-type]
    print(processor.decode(outputs[0][inputs["input_ids"].shape[-1] :]))
    print(f"{past_key_values.get_seq_length()=}")

    tensors = [
        {
            "prefill_length": torch.tensor(prefill_length),
            "queries": past_key_values.queries[i],
            "keys": past_key_values.layers[i].keys,
            "values": past_key_values.layers[i].values,
        }
        for i in range(len(past_key_values.layers))
        if isinstance(past_key_values.layers[i], CacheLayerMixin)
    ]

    for i, tensor_dict in enumerate(tensors):
        save_file(
            tensor_dict,
            f"layer_{i}_tensors.safetensors",
        )


if __name__ == "__main__":
    main("Qwen/Qwen2.5-7B", dataset_name=Datasets.NIAH)
