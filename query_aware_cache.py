import types
from typing import Callable

import torch
from transformers import DynamicCache
from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    ALL_ATTENTION_FUNCTIONS,
    Qwen3_5Attention,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from typing import Unpack


class QueryAwareCache(DynamicCache):
    """
    Drop-in replacement for DynamicCache. Receives query states inside update()
    so they can eventually influence which K/V pairs are returned.

    Captured queries are stored in self.query_states[layer_idx] as a list of
    tensors (one per forward call), shape [batch, heads, seq_len, head_dim].
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.query_states: dict[int, list[torch.Tensor]] = {}

    def update(
        self, key_states, value_states, layer_idx, *args, query_states=None, **kwargs
    ):
        if query_states is not None:
            self.query_states.setdefault(layer_idx, []).append(query_states.detach())
        # Eventually: use query_states to select/filter returned K/V here
        return super().update(key_states, value_states, layer_idx, *args, **kwargs)


def _query_aware_attention_forward(
    self: Qwen3_5Attention,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    past_key_values: Cache | None = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Copy of Qwen3_5Attention.forward with query_states forwarded to cache.update()."""
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
            module.forward = types.MethodType(_query_aware_attention_forward, module)
