import rich
import time
import torch
from itertools import batched
from datasets import load_dataset
from transformers import AutoModelForImageTextToText, AutoProcessor, DynamicCache
from transformers.cache_utils import CacheLayerMixin, LinearAttentionCacheLayerMixin
from query_aware_cache import QueryAwareCache, bind_query_aware_cache


def main():
    # processor = AutoProcessor.from_pretrained("Qwen/Qwen3.5-0.8B")
    # model = AutoModelForImageTextToText.from_pretrained("Qwen/Qwen3.5-0.8B").eval().to("cuda")
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3.5-9B")
    model = (
        AutoModelForImageTextToText.from_pretrained(
            "Qwen/Qwen3.5-9B", attn_implementation="sdpa", dtype=torch.bfloat16
        )
        .eval()
        .to("cuda")
    )

    bind_query_aware_cache(model)
    past_key_values = QueryAwareCache(config=model.config)
    # past_key_values = DynamicCache(config=model.config, offloading=True)

    ds = load_dataset("rajpurkar/squad", split="train")

    total = 0
    i = 0
    seen = set()
    while total < 100000:
        if ds[i]["context"] in seen:
            i += 1
            continue

        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": ds[i]["context"]}],
            }
        ]
        seen += ds[i]["context"]
        i += 1
        # rich.print(messages)
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        print(inputs["input_ids"].shape[-1])
        total += inputs["input_ids"].shape[-1]
        print(f"{i=} {total=}")
        input_chunks = torch.split(inputs["input_ids"], 1024, -1)
        attention_masks = torch.split(inputs["attention_mask"], 1024, -1)
        mm_token_type_chunks = torch.split(inputs["mm_token_type_ids"], 1024, -1)
        for input_ids, attention_mask, mm_token_type_ids in zip(
            input_chunks, attention_masks, mm_token_type_chunks
        ):
            input_ids = input_ids.to(model.device)
            attention_mask = attention_mask.to(model.device)
            mm_token_type_ids = mm_token_type_ids.to(model.device)
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
                        mm_token_type_ids=mm_token_type_ids,
                        past_key_values=past_key_values,
                        logits_to_keep=1,
                    ).past_key_values

    print(f"{total=}")
    print(f"{past_key_values.get_seq_length()=}")

    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": ds[128]["question"]}],
        },
    ]
    rich.print(ds[128])
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
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
    inputs["mm_token_type_ids"] = torch.cat(
        [
            torch.zeros(1, past_key_values.get_seq_length(), dtype=torch.long),
            inputs["mm_token_type_ids"],
        ],
        dim=1,
    )
    inputs = inputs.to(model.device)

    outputs = model.generate(
        **inputs, max_new_tokens=512, past_key_values=past_key_values, use_cache=True
    )
    print(processor.decode(outputs[0][inputs["input_ids"].shape[-1] :]))
    print(f"{past_key_values.get_seq_length()=}")

    # for layer in past_key_values.layers:
    #    if isinstance(layer, CacheLayerMixin):
    #        assert layer.keys is not None and layer.values is not None
    #        rich.print(layer.keys.shape, layer.values.shape)
    #    if isinstance(layer, LinearAttentionCacheLayerMixin):
    #        assert layer.conv_states is not None and layer.recurrent_states is not None
    #        rich.print(layer.conv_states.shape, layer.recurrent_states.shape)
    # for layer_idx, queries in past_key_values.query_states.items():
    #    rich.print(f"layer {layer_idx}: {len(queries)} query snapshots, shape {queries[0].shape}")


if __name__ == "__main__":
    main()
