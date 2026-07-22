import compression.zstd
import json
import math
import types
from concurrent.futures import ThreadPoolExecutor
from functools import cached_property
from pathlib import Path
from typing import Annotated, BinaryIO, Callable, Literal, Protocol, Unpack

import numpy as np
import numpy.typing as npt
import qdrant_edge as edge
import torch
from pydantic import BaseModel, Field, PrivateAttr
from qdrant_client import QdrantClient
from qdrant_client.models import QueryRequest, SearchParams
from rich.progress import track
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

from kv_search.timer import timers
from kv_search._native import NativeEdgeRetriever


class Retriever(Protocol):
    def retrieve(
        self, query_states: torch.Tensor, layer_idx: int, prefill: DynamicCache
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return topk (keys, values) from prefill context for given queries."""


class FullContextRetriever(BaseModel):
    type: Literal["full"] = "full"

    def retrieve(
        self, query_states: torch.Tensor, layer_idx: int, prefill: DynamicCache
    ):
        layer = prefill.layers[layer_idx]
        assert isinstance(layer, CacheLayerMixin)
        assert layer.keys is not None
        assert layer.values is not None
        return layer.keys, layer.values


class TopKRetriever(BaseModel):
    type: Literal["topk"] = "topk"
    n_retrieved: int = 128

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
        self, query_states: torch.Tensor, layer_idx: int, prefill: DynamicCache
    ):
        layer = prefill.layers[layer_idx]
        assert isinstance(layer, CacheLayerMixin)
        assert layer.keys is not None
        assert layer.values is not None

        s = self._eager_attention_forward(query_states, layer.keys, scaling=1)
        _, idx = torch.topk(s, self.n_retrieved, dim=-1)
        idx = idx.reshape((1, 4, -1, 1))

        keys = layer.keys.take_along_dim(idx, dim=2)
        values = layer.values.take_along_dim(idx, dim=2)

        return keys, values


class QdrantRetriever(BaseModel):
    type: Literal["qdrant"] = "qdrant"
    n_retrieved: int = 128
    url: str = "localhost"
    api_key: str | None = None

    @cached_property
    def _client(self) -> QdrantClient:
        return QdrantClient(self.url, api_key=self.api_key, prefer_grpc=True)

    @cached_property
    def _pool(self) -> ThreadPoolExecutor:
        return ThreadPoolExecutor(max_workers=4)

    def _query_head(
        self, layer_idx: int, head_idx: int, query_states: npt.NDArray
    ) -> tuple[npt.NDArray, npt.NDArray]:
        query_idx = head_idx * 4
        data = self._client.query_batch_points(
            collection_name=f"layer={layer_idx};head={head_idx}",
            requests=[
                QueryRequest(
                    query=query_states[query_idx + i, token_idx],
                    params=SearchParams(exact=True),
                    with_payload=False,
                    with_vector=True,
                    using="key",
                    limit=self.n_retrieved,
                )
                for token_idx in range(query_states.shape[1])
                for i in range(4)
            ],
        )
        keys = np.array(
            [p.vector["key"] for r in data for p in r.points],  # ty:ignore[invalid-argument-type, not-subscriptable]
            dtype=np.float32,
        )
        values = np.array(
            [p.vector["value"] for r in data for p in r.points],  # ty:ignore[invalid-argument-type, not-subscriptable]
            dtype=np.float32,
        )
        return keys, values

    def retrieve(
        self, query_states: torch.Tensor, layer_idx: int, prefill: DynamicCache
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query = query_states[0].to(torch.float32).cpu().numpy()
        with timers.qdrant_retrieve:
            results = list(
                self._pool.map(
                    lambda h: self._query_head(layer_idx, h, query), range(4)
                )
            )
        keys = (
            torch.from_numpy(np.stack([k for k, _ in results]))
            .unsqueeze(0)
            .to(query_states.device, query_states.dtype)
        )
        values = (
            torch.from_numpy(np.stack([v for _, v in results]))
            .unsqueeze(0)
            .to(query_states.device, query_states.dtype)
        )
        return keys, values


class QdrantEdgeNativeRetriever(BaseModel):
    type: Literal["native"] = "native"
    n_retrieved: int = 128

    @cached_property
    def _engine(self) -> NativeEdgeRetriever:
        return NativeEdgeRetriever(
            [
                (
                    (layer_idx, head_idx),
                    f"cache/edge/layer{layer_idx:02d}_head{head_idx}",
                )
                for layer_idx in range(3, 32, 4)
                for head_idx in range(4)
            ]
        )

    def retrieve(
        self, query_states: torch.Tensor, layer_idx: int, prefill: DynamicCache
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query = query_states[0].to(torch.float32).cpu().numpy()
        with timers.qdrant_retrieve:
            keys, values = self._engine.retrieve(
                layer_idx,
                query,
                limit=self.n_retrieved,
            )
        keys = (
            torch.from_numpy(keys)
            .unsqueeze(0)
            .to(query_states.device, query_states.dtype)
        )
        values = (
            torch.from_numpy(values)
            .unsqueeze(0)
            .to(query_states.device, query_states.dtype)
        )
        return keys, values


class QdrantEdgeRetriever(BaseModel):
    type: Literal["edge"] = "edge"
    n_retrieved: int = 128

    _shards: dict[tuple[int, int], edge.EdgeShard] = PrivateAttr(default_factory=dict)

    def _shard(self, layer_idx: int, head_idx: int) -> edge.EdgeShard:
        key = (layer_idx, head_idx)
        if key in self._shards:
            return self._shards[key]
        shard = edge.EdgeShard.load(f"cache/edge/layer{layer_idx:02d}_head{head_idx}")
        self._shards[key] = shard
        return shard

    def _query_head(
        self, layer_idx: int, head_idx: int, query_states: npt.NDArray
    ) -> tuple[npt.NDArray, npt.NDArray]:
        query_idx = head_idx * 4
        data = [
            self._shard(layer_idx, head_idx).query(
                edge.QueryRequest(
                    query=edge.Query.Nearest(
                        query=query_states[query_idx + i, token_idx], using="key"
                    ),
                    params=edge.SearchParams(exact=True),
                    with_payload=False,
                    with_vector=True,
                    limit=self.n_retrieved,
                )
            )
            for token_idx in range(query_states.shape[1])
            for i in range(4)
        ]
        keys = np.array(
            [p.vector["key"] for r in data for p in r],  # ty:ignore[invalid-argument-type, not-subscriptable]
            dtype=np.float32,
        )
        values = np.array(
            [p.vector["value"] for r in data for p in r],  # ty:ignore[invalid-argument-type, not-subscriptable]
            dtype=np.float32,
        )
        return keys, values

    def retrieve(
        self, query_states: torch.Tensor, layer_idx: int, prefill: DynamicCache
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query = query_states[0].to(torch.float32).cpu().numpy()
        with timers.qdrant_retrieve:
            # results = list(
            #     self._pool.map(
            #         lambda h: self._query_head(layer_idx, h, query), range(4)
            #     )
            # )
            results = [self._query_head(layer_idx, h, query) for h in range(4)]
        keys = (
            torch.from_numpy(np.stack([k for k, _ in results]))
            .unsqueeze(0)
            .to(query_states.device, query_states.dtype)
        )
        values = (
            torch.from_numpy(np.stack([v for _, v in results]))
            .unsqueeze(0)
            .to(query_states.device, query_states.dtype)
        )
        return keys, values


RetrieverConfig = Annotated[
    TopKRetriever
    | FullContextRetriever
    | QdrantRetriever
    | QdrantEdgeRetriever
    | QdrantEdgeNativeRetriever,
    Field(discriminator="type"),
]


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

        # if query_states is not None:
        #     cached_keys, cached_values = self.retriever.retrieve(
        #         query_states, layer_idx, self.prefill
        #     )
        #     keys = torch.cat([cached_keys, keys], dim=2)
        #     values = torch.cat([cached_values, values], dim=2)

        return keys, values


class RecordingCache(DynamicCache):
    # See https://huggingface.co/docs/safetensors/en/index#format

    HEADER_RESERVED_BYTES = 256  # 8 bytes of header size + header

    def __init__(self, path: Path, **kwargs):
        super().__init__(**kwargs)
        self.path = path
        self._files: dict[int, BinaryIO] = {}
        self._shapes: dict[int, list[int]] = {}

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        *args,
        query_states: torch.Tensor | None = None,
        **kwargs,
    ):
        if query_states is not None:
            assert query_states.dtype == torch.bfloat16

            # [batch, head, seq_len, dim] -> [batch, seq_len, head, dim]
            # batch is always 1, so this way, in the file, all heads for one
            # token are stored next to eachother and we can just append new tokens
            chunk = query_states.transpose(1, 2).contiguous()
            if layer_idx not in self._files:
                f = (self.path / f"queries_{layer_idx:02d}.safetensors").open("wb")
                # Reserve some bytes for the header to be written to later
                f.write(b"\0" * self.HEADER_RESERVED_BYTES)

                self._files[layer_idx] = f
                self._shapes[layer_idx] = [
                    chunk.shape[0],
                    0,  # will grow this over time
                    chunk.shape[2],
                    chunk.shape[3],
                ]

            # write to file as bytes, the conversion to uint8 just makes sure it's a dtype numpy
            # can handle (not bf16)
            self._files[layer_idx].write(chunk.cpu().view(torch.uint8).numpy().data)
            self._shapes[layer_idx][1] += chunk.shape[1]

        keys, values = super().update(
            key_states, value_states, layer_idx, *args, **kwargs
        )
        return keys, values

    def finalize(self):
        for layer_idx, f in self._files.items():
            shape = self._shapes[layer_idx]

            # bf16 means 2 bytes per number
            nbytes = 2 * math.prod(shape)
            header = json.dumps(
                {
                    "queries": {
                        "dtype": "BF16",
                        "shape": shape,
                        "data_offsets": [0, nbytes],
                    }
                }
            ).encode()

            # make sure the header fits in its slot
            assert len(header) <= 256 - 8

            # pad header with space, which is valid json
            header += b" " * (256 - 8 - len(header))

            f.seek(0)

            # write header size
            f.write((256 - 8).to_bytes(8, "little"))

            f.write(header)
            f.close()

        self._files.clear()
        self._shapes.clear()


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


@timers.prefill_save
def save_cache(
    cache: Cache,
    path: Path,
    context_len: int,
) -> None:
    (path / "meta.json").write_text(json.dumps({"context_len": context_len}))
    for i, layer in track(
        enumerate(cache.layers), description="Writing to disk", total=len(cache.layers)
    ):
        tensors = _layer_to_dict(layer)
        blob = compression.zstd.compress(save(tensors))
        (path / f"layer_{i:02d}.safetensors.zst").write_bytes(blob)


@timers.prefill_load
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
            layer.update_recurrent_state(tensors["recurrent_states"])
        else:
            layer.lazy_initialization(tensors["keys"], tensors["values"])
            layer.keys = tensors["keys"]
            layer.values = tensors["values"]

    return cache, meta["context_len"]


def _causal_mask(q_len: int, k_len: int, dtype, device) -> torch.Tensor | None:
    if q_len <= 1:
        return None

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


def _make_attention_mask(
    attn_impl: str, q_len: int, k_len: int, dtype, device
) -> torch.Tensor | None:
    # Need to dynamically create our attention mask because transformers does not have sane defaults for causal/attention masks during inference
    # usually this is created beforehand, when the prompt is tokenized, but we don't know the context length then
    if attn_impl not in ("sdpa", "eager"):
        return None

    return _causal_mask(q_len, k_len, dtype, device)


# HACK: Most of this is somewhat specific to qwen3.5 and also implemented in the slowest possible way
# generalizing and improving is tbd
def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
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


def _partition_attend(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scaling: float,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    key_states = _repeat_kv(key, 4).to(torch.float32)
    value_states = _repeat_kv(value, 4).to(torch.float32)

    attn_weights = torch.matmul(query.to(torch.float32), key_states.transpose(2, 3)) * scaling

    if mask is not None:
        attn_weights = attn_weights + mask

    lse = torch.logsumexp(attn_weights, dim=-1)
    out = torch.matmul(
        torch.softmax(attn_weights, dim=-1), value_states
    )

    return out, lse


def _merge_partitions(
    out_a: torch.Tensor, lse_a: torch.Tensor, out_b: torch.Tensor, lse_b: torch.Tensor
) -> torch.Tensor:
    lse = torch.logaddexp(lse_a, lse_b)
    wa = torch.exp(lse_a - lse).unsqueeze(-1)
    wb = torch.exp(lse_b - lse).unsqueeze(-1)
    return wa * out_a + wb * out_b


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

    if isinstance(past_key_values, RetrievalCache):
        # NOTE: at least part of this should be happening in the retriever
        k_retrieved, v_retrieved = past_key_values.retriever.retrieve(
            query_states, self.layer_idx, prefill=past_key_values.prefill
        )

        retrieved_out, retrieved_lse = _partition_attend(
            query_states, k_retrieved, v_retrieved, self.scaling, None
        )

        live_mask = _causal_mask(
            query_states.shape[2],
            key_states.shape[2],
            query_states.dtype,
            query_states.device,
        )
        live_out, live_lse = _partition_attend(
            query_states, key_states, value_states, self.scaling, live_mask
        )

        attn_output = _merge_partitions(
            live_out, live_lse, retrieved_out, retrieved_lse
        )
        attn_output = attn_output.to(query_states.dtype).transpose(1, 2).contiguous()
    else:
        attention_mask = _make_attention_mask(
            self.config._attn_implementation,
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
        self.config._attn_implementation,
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
        self.config._attn_implementation,
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
    position_embeddings: torch.Tensor,
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
        self.config._attn_implementation,
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
        self.config._attn_implementation,
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
