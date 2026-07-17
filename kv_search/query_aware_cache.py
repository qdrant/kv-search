import compression.zstd
import json
import types
from pathlib import Path
from typing import Callable, Protocol, Unpack

import torch
from qdrant_client import QdrantClient
from qdrant_client.models import QueryRequest, SearchParams
from safetensors.torch import load, save
from transformers import PreTrainedConfig, PretrainedConfig
from transformers.cache_utils import (
    Cache,
    CacheLayerMixin,
    DynamicCache,
    LinearAttentionCacheLayerMixin,
)
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.models.gemma3.modeling_gemma3 import Gemma3Attention
from transformers.models.gemma4_unified.modeling_gemma4_unified import (
    Gemma4UnifiedTextAttention,
)
from transformers.models.ministral3.modeling_ministral3 import Ministral3Attention
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Attention


class Retriever(Protocol):
    def retrieve(
        self, query_states: torch.Tensor, layer_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return topk (keys, values) from prefill context for given queries."""


class FullContextRetriever:
    pass


class TopKRetriever:
    def __init__(self, n_retrieved: int, prefill: DynamicCache):
        self.prefill = prefill
        self.n_retrieved = n_retrieved

    def _repeat_kv(self, hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """
        This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
        num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
        """
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(
            batch, num_key_value_heads, n_rep, slen, head_dim
        )
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

    def _eager_attention_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        scaling: float,
        attention_mask: torch.Tensor | None = None,
        softcap: float | None = None,
    ) -> torch.Tensor:
        key_states = self._repeat_kv(key, 4)

        attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling

        if softcap is not None:
            attn_weights = attn_weights / softcap
            attn_weights = torch.tanh(attn_weights)
            attn_weights = attn_weights * softcap
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # attn_weights = torch.nn.functional.softmax(
        #     attn_weights, dim=-1, dtype=torch.float32
        # ).to(query.dtype)

        return attn_weights

    def retrieve(
        self,
        query_states: torch.Tensor,
        layer_idx: int,
    ):
        layer = self.prefill.layers[layer_idx]
        assert isinstance(layer, CacheLayerMixin)
        assert layer.keys is not None
        assert layer.values is not None

        s = self._eager_attention_forward(query_states, layer.keys, scaling=1)
        _, idx = torch.topk(s, self.n_retrieved, dim=-1)
        idx = idx.reshape((1, 4, -1, 1))

        keys = layer.keys.take_along_dim(idx, dim=2)
        values = layer.values.take_along_dim(idx, dim=2)

        return keys, values


class QdrantRetriever:
    def __init__(self, client: QdrantClient, n_retrieved: int):
        self.n_retrieved = n_retrieved
        self.client = client

    def retrieve(
        self,
        query_states: torch.Tensor,
        layer_idx: int,
    ):
        cached_keys_per_head: list[torch.Tensor] = []
        cached_values_per_head: list[torch.Tensor] = []
        for head_idx in range(4):
            query_idx = head_idx * 4
            data = self.client.query_batch_points(
                collection_name=f"layer={layer_idx};head={head_idx}",
                requests=[
                    QueryRequest(
                        query=query_states[0, query_idx + i, token_idx]
                        .cpu()
                        .to(torch.float)
                        .numpy(),
                        params=SearchParams(exact=True),
                        with_payload=True,
                        with_vector=True,
                        using="key",
                        limit=self.n_retrieved,
                    )
                    for token_idx in range(query_states.shape[2])
                    for i in range(4)
                ],
            )
            cached_key_head: list[list[float]] = []
            cached_value_head: list[list[float]] = []
            for r in data:
                for p in r.points:
                    assert isinstance(p.vector, dict)
                    assert "key" in p.vector
                    assert "value" in p.vector
                    cached_key_head.append(p.vector["key"])
                    cached_value_head.append(p.vector["value"])
            cached_keys_per_head.append(
                torch.tensor(
                    cached_key_head,
                    dtype=query_states.dtype,
                    device=query_states.device,
                )
                .unsqueeze(0)
                .unsqueeze(0)
            )
            cached_values_per_head.append(
                torch.tensor(
                    cached_value_head,
                    dtype=query_states.dtype,
                    device=query_states.device,
                )
                .unsqueeze(0)
                .unsqueeze(0)
            )
        cached_keys = torch.cat(cached_keys_per_head, dim=1)
        cached_values = torch.cat(cached_values_per_head, dim=1)
        return cached_keys, cached_values


class RetrievalCache(DynamicCache):
    def __init__(
        self,
        retriever: Retriever,
        prefill: DynamicCache,
        config: PreTrainedConfig | None = None,
    ):
        super().__init__(config=config)
        self.retriever = retriever
        self.prefill = prefill
        self._restore_state()

    def _restore_state(self):
        for cached_layer, layer in zip(self.prefill.layers, self.layers):
            if isinstance(layer, LinearAttentionCacheLayerMixin):
                assert isinstance(cached_layer, LinearAttentionCacheLayerMixin)
                if not (
                    layer.is_conv_states_initialized
                    and layer.is_recurrent_states_initialized
                ):
                    layer.lazy_initialization(
                        cached_layer.conv_states, cached_layer.recurrent_states
                    )
                assert layer.conv_states is not None
                assert layer.recurrent_states is not None
                assert cached_layer.conv_states is not None
                assert cached_layer.recurrent_states is not None
                layer.conv_states.copy_(cached_layer.conv_states)
                layer.recurrent_states.copy_(cached_layer.recurrent_states)
                layer.has_previous_state = True
            elif layer.is_initialized:
                assert layer.keys is not None
                assert layer.values is not None
                layer.keys = torch.tensor(
                    [], dtype=layer.keys.dtype, device=layer.keys.device
                )
                layer.values = torch.tensor(
                    [], dtype=layer.values.dtype, device=layer.values.device
                )

    def reset(self):
        super().reset()
        self._restore_state()

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        *args,
        query_states: torch.Tensor | None = None,
        **kwargs,
    ):
        keys, values = super().update(
            key_states, value_states, layer_idx, *args, **kwargs
        )

        if query_states is not None:
            cached_keys, cached_values = self.retriever.retrieve(
                query_states, layer_idx
            )
            keys = torch.cat([cached_keys, keys], dim=2)
            values = torch.cat([cached_values, values], dim=2)

        return keys, values


class RecordingCache(DynamicCache):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.queries: list[torch.Tensor | None] = [None] * len(self.layers)

    def update(
        self, key_states, value_states, layer_idx, *args, query_states=None, **kwargs
    ):
        if query_states is not None:
            if layer_idx >= len(self.queries):
                self.queries.extend([None] * (layer_idx - len(self.queries) + 1))
            if self.queries[layer_idx] is None:
                self.queries[layer_idx] = query_states.detach().cpu()
            else:
                self.queries[layer_idx] = torch.cat(
                    [self.queries[layer_idx], query_states.detach().cpu()], dim=-2
                )

        keys, values = super().update(
            key_states, value_states, layer_idx, *args, **kwargs
        )
        return keys, values


def _layer_to_dict(
    layer: CacheLayerMixin | LinearAttentionCacheLayerMixin,
) -> dict[str, torch.Tensor]:
    tensors = {}
    if isinstance(layer, LinearAttentionCacheLayerMixin):
        assert layer.conv_states is not None
        assert layer.recurrent_states is not None
        tensors["conv_states"] = _shuffle_bf16(layer.conv_states)
        tensors["recurrent_states"] = _shuffle_bf16(layer.recurrent_states)
    else:
        assert layer.keys is not None
        assert layer.values is not None
        tensors["keys"] = _shuffle_bf16(layer.keys)
        tensors["values"] = _shuffle_bf16(layer.values)

    return tensors


# dirty byte tricks so it compresses better
def _shuffle_bf16(t: torch.Tensor) -> torch.Tensor:
    return (
        t.contiguous()
        .view(torch.uint8)
        .reshape(-1, 2)
        .t()
        .contiguous()
        .reshape(2, *t.shape)
    )


def _unshuffle_bf16(u: torch.Tensor) -> torch.Tensor:
    return (
        u.reshape(2, -1)
        .t()
        .contiguous()
        .flatten()
        .view(torch.bfloat16)
        .reshape(u.shape[1:])
    )


def save_cache(
    cache: Cache,
    path: Path,
    context_len: int,
    queries: list[torch.Tensor | None] | None = None,
) -> None:
    (path / "meta.json").write_text(json.dumps({"context_len": context_len}))
    for i, layer in enumerate(cache.layers):
        tensors = _layer_to_dict(layer)
        blob = compression.zstd.compress(save(tensors))
        (path / f"layer_{i:02d}.safetensors.zst").write_bytes(blob)

    if queries is not None:
        tensors = {
            str(i): _shuffle_bf16(v) for i, v in enumerate(queries) if v is not None
        }
        blob = compression.zstd.compress(save(tensors))
        (path / "queries.safetensors.zst").write_bytes(blob)


def load_cache(
    path: Path, config: PretrainedConfig, device="cuda"
) -> tuple[DynamicCache, int]:
    cache = DynamicCache(config=config)
    meta = json.loads((path / "meta.json").read_text())

    files = sorted(path.glob("layer_*.safetensors.zst"))
    assert len(files) == len(cache.layers)
    for layer, file in zip(cache.layers, files):
        tensors = load(compression.zstd.decompress(file.read_bytes()))
        for k, v in tensors.items():
            tensors[k] = _unshuffle_bf16(v).to(device, non_blocking=True)
        if isinstance(layer, LinearAttentionCacheLayerMixin):
            layer.update_conv_state(tensors["conv_states"])
            layer.update_recurrent_state(tensors["conv_states"])
        else:
            layer.lazy_initialization(tensors["keys"], tensors["values"])
            layer.keys = tensors["keys"]
            layer.values = tensors["values"]

    return cache, meta["context_len"]


def _make_attention_mask(
    attention_mask: torch.Tensor | None, q_len: int, k_len: int, dtype, device
):
    # Need to dynamically create our attention mask because transformers does not have sane defaults for causal/attention masks during inference
    # usually this is created beforehand, when the prompt is tokenized, but we don't know the context length then
    if attention_mask is None and q_len > 1:
        attention_mask = torch.zeros(q_len, k_len, dtype=dtype, device=device)
        attention_mask[:, k_len - q_len :] = torch.triu(
            torch.full(
                (q_len, q_len),
                torch.finfo(dtype).min,
                device=device,
            ),
            diagonal=1,
        )
        return attention_mask[None, None]


def _qwen_3_5_forward(
    self: Qwen3_5Attention,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    past_key_values: Cache | None = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Copy of Qwen3_5Attention.forward with query_states forwarded to cache.update()."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        ALL_ATTENTION_FUNCTIONS,
        apply_rotary_pos_emb,
        eager_attention_forward,
    )

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states, gate = torch.chunk(
        self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1
    )
    gate = gate.reshape(*input_shape, -1)

    query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(
        1, 2
    )
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        key_states, value_states = past_key_values.update(
            key_states, value_states, self.layer_idx, query_states=query_states
        )

    attention_mask = _make_attention_mask(
        attention_mask,
        query_states.shape[2],
        key_states.shape[2],
        query_states.dtype,
        query_states.device,
    )

    attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation, eager_attention_forward
    )

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = attn_output * torch.sigmoid(gate)
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def _qwen_2_5_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    past_key_values: Cache | None = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> tuple[torch.Tensor, torch.Tensor | None]:
    from transformers.models.qwen2.modeling_qwen2 import (
        ALL_ATTENTION_FUNCTIONS,
        apply_rotary_pos_emb,
        eager_attention_forward,
    )

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        key_states, value_states = past_key_values.update(
            key_states, value_states, self.layer_idx, query_states=query_states
        )

    attention_mask = _make_attention_mask(
        attention_mask,
        query_states.shape[2],
        key_states.shape[2],
        query_states.dtype,
        query_states.device,
    )

    attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation, eager_attention_forward
    )

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=self.sliding_window,  # main diff with Llama
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def _ministral3_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    position_ids: torch.Tensor,
    past_key_values: Cache | None = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> tuple[torch.Tensor, torch.Tensor | None]:
    from transformers.models.ministral3.modeling_ministral3 import (
        ALL_ATTENTION_FUNCTIONS,
        apply_rotary_pos_emb,
        eager_attention_forward,
        get_llama_4_attn_scale,
    )

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    query_states = query_states * get_llama_4_attn_scale(
        position_ids,
        self.config.rope_parameters.get("llama_4_scaling_beta"),
        self.config.rope_parameters.get("original_max_position_embeddings"),
    ).to(query_states.dtype)

    if past_key_values is not None:
        key_states, value_states = past_key_values.update(
            key_states, value_states, self.layer_idx, query_states=query_states
        )

    attention_mask = _make_attention_mask(
        attention_mask,
        query_states.shape[2],
        key_states.shape[2],
        query_states.dtype,
        query_states.device,
    )

    attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation, eager_attention_forward
    )

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=getattr(
            self.config, "sliding_window", None
        ),  # main diff with Llama
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def _gemma3_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: torch.Tensor = None,
    attention_mask: torch.Tensor | None = None,
    past_key_values: Cache | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
    from transformers.models.gemma3.modeling_gemma3 import (
        ALL_ATTENTION_FUNCTIONS,
        apply_rotary_pos_emb,
        eager_attention_forward,
    )

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    query_states = self.q_norm(query_states)
    key_states = self.k_norm(key_states)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        key_states, value_states = past_key_values.update(
            key_states, value_states, self.layer_idx, query_states=query_states
        )

    attention_mask = _make_attention_mask(
        attention_mask,
        query_states.shape[2],
        key_states.shape[2],
        query_states.dtype,
        query_states.device,
    )

    attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation, eager_attention_forward
    )

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=self.attention_dropout if self.training else 0.0,
        scaling=self.scaling,
        sliding_window=self.sliding_window,
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def _gemma4_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: torch.Tensor,
    attention_mask: torch.Tensor | None,
    shared_kv_states: dict[str, tuple[torch.Tensor, torch.Tensor]],
    past_key_values: Cache | None = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> tuple[torch.Tensor, torch.Tensor | None]:
    from transformers.models.gemma4_unified.modeling_gemma4_unified import (
        ALL_ATTENTION_FUNCTIONS,
        apply_rotary_pos_emb,
        eager_attention_forward,
    )

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    cos, sin = position_embeddings

    query_states = self.q_proj(hidden_states).view(hidden_shape)
    query_states = self.q_norm(query_states)
    query_states = apply_rotary_pos_emb(query_states, cos, sin, unsqueeze_dim=2)
    query_states = query_states.transpose(1, 2)

    # For layers with shared KV (from kv sharing point onwards), we reuse the same keys/values states as the last non-sharing layer.
    # We cannot simply reuse the cached state if we have a Cache, as sliding layers will not remember the full states in their Cache
    # once we are past the sliding window - so we always use `shared_kv_states` instead, even when past_key_values is not None
    if self.is_kv_shared_layer:
        key_states, value_states = shared_kv_states[self.layer_type]
        # Device of past layer may be different from current one
        key_states = key_states.to(query_states.device)
        value_states = value_states.to(query_states.device)
    else:
        key_states = self.k_proj(hidden_states).view(hidden_shape)
        value_states = (
            self.v_proj(hidden_states).view(hidden_shape)
            if self.v_proj is not None
            else key_states
        )

        key_states = self.k_norm(key_states)
        key_states = apply_rotary_pos_emb(key_states, cos, sin, unsqueeze_dim=2)
        key_states = key_states.transpose(1, 2)

        value_states = self.v_norm(value_states)
        value_states = value_states.transpose(1, 2)

    if past_key_values is not None and not self.is_kv_shared_layer:
        key_states, value_states = past_key_values.update(
            key_states, value_states, self.layer_idx, query_states=query_states
        )

    attention_mask = _make_attention_mask(
        attention_mask,
        query_states.shape[2],
        key_states.shape[2],
        query_states.dtype,
        query_states.device,
    )

    if self.store_full_length_kv:
        shared_kv_states[self.layer_type] = key_states, value_states

    attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation, eager_attention_forward
    )

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=self.attention_dropout if self.training else 0.0,
        scaling=self.scaling,
        sliding_window=self.sliding_window,
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def bind_query_aware_cache(model) -> None:
    """
    Patch every Qwen3_5Attention layer in the model to forward query states to
    the cache. Scoped to this model instance — no global state is modified.

    Call once after model loading, before creating the cache:
        bind_query_aware_cache(model)
        cache = QueryAwareCache(config=model.config)
    """
    for module in model.modules():
        if isinstance(module, Qwen3_5Attention):
            module.forward = types.MethodType(_qwen_3_5_forward, module)
        if isinstance(module, Qwen2Attention):
            module.forward = types.MethodType(_qwen_2_5_forward, module)
        if isinstance(module, Ministral3Attention):
            module.forward = types.MethodType(_ministral3_forward, module)
        if isinstance(module, Gemma3Attention):
            module.forward = types.MethodType(_gemma3_forward, module)
        if isinstance(module, Gemma4UnifiedTextAttention):
            module.forward = types.MethodType(_gemma4_forward, module)
