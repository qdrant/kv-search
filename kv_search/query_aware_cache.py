import compression.zstd
import types
from pathlib import Path
from typing import Callable, Unpack

import torch
from pydantic import BaseModel, Field, TypeAdapter
from qdrant_client import QdrantClient
from qdrant_client.models import QueryRequest, SearchParams
from safetensors.torch import load, save
from transformers import DynamicCache
from transformers.cache_utils import (
    Cache,
    CacheLayerMixin,
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


class LinearLayerState(BaseModel):
    idx: int
    conv_states: torch.Tensor
    rec_states: torch.Tensor


class LayerState(BaseModel):
    idx: int
    queries: torch.Tensor
    keys: torch.Tensor
    values: torch.Tensor


LayerAdapter: TypeAdapter[LayerState | LinearLayerState] = TypeAdapter(
    LayerState | LinearLayerState
)


class CacheState(BaseModel):
    context_len: int
    layers: list[LayerState | LinearLayerState] = Field(default_factory=list)

    def save(self, path: Path):
        with (path / "meta.json").open("wt") as f:
            f.write(self.model_dump_json(exclude={"layers"}))
        for layer in self.layers:
            with compression.zstd.open(
                path / f"layer_{layer.idx:02d}_tensors.safetensors.zst", "wb"
            ) as f:
                tmp = layer.model_dump()
                tmp["idx"] = torch.tensor(tmp["idx"], dtype=torch.long)
                f.write(save(tmp))

    @classmethod
    def load(cls, path: Path) -> CacheState:
        with (path / "meta.json").open("rt") as f:
            ret = cls.model_validate_json(f.read())

        for p in sorted(list(path.glob("layer_*_tensors.safetensors.zst"))):
            with compression.zstd.open(p, "rb") as f:
                tmp = load(f.read())
                tmp["idx"] = tmp["idx"].item()  # ty:ignore[invalid-assignment]
                ret.layers.append(LayerAdapter.validate_python(tmp))

        return ret


class CutoffCache(DynamicCache):
    cutoff: int | None
    state: CacheState
    layers: list[CacheLayerMixin | LinearAttentionCacheLayerMixin]

    def __init__(self, cutoff: int | None = None, **kwargs):
        super().__init__(**kwargs)
        self.cutoff = cutoff

        self.scaling = 256**-0.5

        self.state = CacheState.load(Path(f"cache/qdrant/qwen3_5/"))

        assert len(self.state.layers) == len(self.layers)
        for cached_layer, layer in zip(self.state.layers, self.layers):
            if isinstance(cached_layer, LinearLayerState):
                assert isinstance(layer, LinearAttentionCacheLayerMixin)
                layer.lazy_initialization(
                    cached_layer.conv_states, cached_layer.rec_states
                )

    def repeat_kv(self, hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
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

    def eager_attention_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        scaling: float,
        attention_mask: torch.Tensor | None = None,
        softcap: float | None = None,
    ) -> torch.Tensor:
        key_states = self.repeat_kv(key, 4)

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

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        *args,
        query_states: torch.Tensor | None = None,
        **kwargs,
    ):
        layer_state = self.state.layers[layer_idx]
        assert isinstance(layer_state, LayerState)
        layer_state.keys = layer_state.keys.to(key_states.device)
        layer_state.values = layer_state.values.to(value_states.device)

        cache_keys = layer_state.keys[:, :, : self.state.context_len, :]
        cache_values = layer_state.values[:, :, : self.state.context_len, :]

        keys, values = super().update(
            key_states, value_states, layer_idx, *args, **kwargs
        )

        if query_states is not None:
            if self.cutoff is None:
                return torch.cat([cache_keys, keys], dim=2), torch.cat(
                    [cache_values, values], dim=2
                )

            s = self.eager_attention_forward(
                query_states, cache_keys, scaling=self.scaling
            )
            # print(f"{self.cutoff=}")
            # print(f"{s.shape=}")
            _, idx = torch.topk(s, self.cutoff, dim=-1)
            idx, _ = idx.reshape((1, 4, -1, 1)).sort(dim=2)
            # idx = idx.to(keys.device)

            keys = torch.cat([cache_keys.take_along_dim(idx, dim=2), keys], dim=2)
            values = torch.cat(
                [cache_values.take_along_dim(idx, dim=2), values],
                dim=2,
            )

        return keys, values


class QdrantCache(DynamicCache):
    """
    Drop-in replacement for DynamicCache. Receives query states inside update()
    so they can eventually influence which K/V pairs are returned.

    Captured queries are stored in self.query_states[layer_idx] as a list of
    tensors (one per forward call), shape [batch, heads, seq_len, head_dim].
    """

    client: QdrantClient

    def __init__(self, url: str, **kwargs):
        super().__init__(**kwargs)
        self.client = QdrantClient(url)

    def update(
        self,
        key_states,
        value_states,
        layer_idx,
        *args,
        query_states: torch.Tensor | None = None,
        **kwargs,
    ):

        keys, values = super().update(
            key_states, value_states, layer_idx, *args, **kwargs
        )

        cached_keys: list[torch.Tensor] = []
        cached_values: list[torch.Tensor] = []
        if query_states is not None:
            for token_idx in range(query_states.shape[2]):
                cached_key_per_token: list[torch.Tensor] = []
                cached_value_per_token: list[torch.Tensor] = []
                for head_idx in range(key_states.shape[1]):
                    query_idx = head_idx * 4
                    data = self.client.query_batch_points(
                        collection_name=f"layer={(layer_idx - 3) // 4};head={head_idx}",
                        requests=[
                            QueryRequest(
                                query=query_states[0, query_idx + i, token_idx]
                                .cpu()
                                .to(torch.float)
                                .numpy(),
                                params=SearchParams(exact=True),
                                with_payload=True,
                                with_vector=True,
                                limit=4096,
                            )
                            for i in range(4)
                        ],
                    )
                    cached_key_head: list[list[float]] = []
                    cached_value_head: list[list[float]] = []
                    for r in data:
                        for p in r.points:
                            cached_key_head.append(p.vector)
                            cached_value_head.append(p.payload["value"])
                    cached_key_per_token.append(
                        torch.tensor(
                            cached_key_head, dtype=keys.dtype, device=keys.device
                        )
                        .unsqueeze(0)
                        .unsqueeze(0)
                    )
                    cached_value_per_token.append(
                        torch.tensor(
                            cached_value_head, dtype=keys.dtype, device=keys.device
                        )
                        .unsqueeze(0)
                        .unsqueeze(0)
                    )
                cached_keys.append(torch.cat(cached_key_per_token, dim=1))
                cached_values.append(torch.cat(cached_value_per_token, dim=1))
        keys = torch.cat(
            cached_keys + [keys],
            dim=2,
        )
        values = torch.cat(
            cached_values + [values],
            dim=2,
        )

        return keys, values


class QueryAwareCache(DynamicCache):
    """
    Drop-in replacement for DynamicCache. Receives query states inside update()
    so they can eventually influence which K/V pairs are returned.

    Captured queries are stored in self.query_states[layer_idx] as a list of
    tensors (one per forward call), shape [batch, heads, seq_len, head_dim].
    """

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

        # Bring current layer back to GPU if offloaded
        if layer_idx < len(self.layers):
            layer = self.layers[layer_idx]
            if layer.is_initialized and layer.keys.device != key_states.device:
                layer.keys = layer.keys.to(key_states.device)
                layer.values = layer.values.to(key_states.device)

        keys, values = super().update(
            key_states, value_states, layer_idx, *args, **kwargs
        )

        layer = self.layers[layer_idx]

        # Offload to CPU after update
        layer.keys = layer.keys.to("cpu")
        layer.values = layer.values.to("cpu")

        return keys, values


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
