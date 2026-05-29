import rich
from datasets import load_dataset
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.cache_utils import CacheLayerMixin, LinearAttentionCacheLayerMixin
from query_aware_cache import QueryAwareCache, bind_query_aware_cache


def main():
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3.5-0.8B")
    model = AutoModelForImageTextToText.from_pretrained("Qwen/Qwen3.5-0.8B")

    bind_query_aware_cache(model)
    past_key_values = QueryAwareCache(config=model.config)

    ds = load_dataset("rajpurkar/squad", split="validation")

    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": ds[0]["context"]}],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": ds[0]["question"]}],
        },
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(
        **inputs, max_new_tokens=128, past_key_values=past_key_values, use_cache=True
    )
    print(processor.decode(outputs[0][inputs["input_ids"].shape[-1] :]))

    for layer in past_key_values.layers:
        if isinstance(layer, CacheLayerMixin):
            assert layer.keys is not None and layer.values is not None
            rich.print(layer.keys.shape, layer.values.shape)
        if isinstance(layer, LinearAttentionCacheLayerMixin):
            assert layer.conv_states is not None and layer.recurrent_states is not None
            rich.print(layer.conv_states.shape, layer.recurrent_states.shape)
    for layer_idx, queries in past_key_values.query_states.items():
        rich.print(f"layer {layer_idx}: {len(queries)} query snapshots, shape {queries[0].shape}")


if __name__ == "__main__":
    main()
