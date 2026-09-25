import json
from collections.abc import Generator
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import safe_open
from transformers import AutoConfig, PreTrainedConfig
from transformers.cache_utils import CacheLayerMixin, LinearAttentionCacheLayerMixin

from kv_search.cache import load_cache


class CachedData:
    """Loads a recorded prefill/session cache and exposes its tensors (queries, recorded
    top-k indices/scores, prefill keys/values). Pass load_prefill=False to skip the
    multi-GB prefill load for index/score-only analyses."""

    def __init__(
        self,
        cache_dir: Path,
        model_name: str,
        device: str | None = None,
        load_prefill: bool = True,
        context_len: int | None = None,
    ):
        self.cache_dir = cache_dir
        # skip mps: several ops here are flaky on it; pass device= to override
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.config: PreTrainedConfig = AutoConfig.from_pretrained(model_name)
        text_config = getattr(self.config, "text_config", self.config)
        self.head_dim = getattr(
            text_config,
            "head_dim",
            text_config.hidden_size // text_config.num_attention_heads,
        )
        self.num_key_value_groups = (
            text_config.num_attention_heads // text_config.num_key_value_heads
        )
        self.scaling = self.head_dim**-0.5

        if load_prefill:
            self.prefill, self.context_len = load_cache(cache_dir, self.config, device)
        else:
            self.prefill = None
            self.context_len = context_len or json.loads(
                (cache_dir / "meta.json").read_text()
            )["context_len"]

    @property
    def full_layers(self) -> Generator[tuple[int, CacheLayerMixin]]:
        for idx, layer in enumerate(self.prefill.layers):
            if isinstance(layer, CacheLayerMixin) and not isinstance(
                layer, LinearAttentionCacheLayerMixin
            ):
                yield idx, layer

    @property
    def full_layer_indices(self) -> list[int]:
        """Full-attention layer indices, derived from recorded index files when prefill
        isn't loaded."""
        if self.prefill is not None:
            return [idx for idx, _ in self.full_layers]
        dirs = sorted(self.cache_dir.glob("indices*/"))
        if not dirs:
            raise RuntimeError(f"no index recordings under {self.cache_dir}")
        return sorted(
            int(f.name.removeprefix("indices_").removesuffix(".safetensors"))
            for f in dirs[0].glob("indices_*.safetensors")
        )

    def queries(
        self, layer_idx: int, positions: slice = slice(-1, None)
    ) -> torch.Tensor:
        path = self.cache_dir / f"queries_{layer_idx:02d}.safetensors"
        with safe_open(path, framework="pt", device=self.device) as f:
            q = f.get_slice("queries")[:, positions, :, :]
        return q.transpose(1, 2).contiguous()

    def indices(
        self, layer_idx: int, prompt_idx: int = 0
    ) -> tuple[int | None, torch.Tensor]:
        tmp = self.cache_dir / f"indices{prompt_idx:02d}/"
        with safe_open(tmp / f"indices_{layer_idx:02d}.safetensors", "pt", self.device) as f:
            indices = f.get_slice("indices")[:]
        meta: dict[str, int] = json.loads((tmp / "meta.json").read_text())
        return meta.get("prompt_len"), indices

    def scores(self, layer_idx: int, prompt_idx: int = 0) -> torch.Tensor:
        tmp = self.cache_dir / f"indices{prompt_idx:02d}/"
        with safe_open(tmp / f"indices_{layer_idx:02d}.safetensors", "pt", self.device) as f:
            return f.get_slice("scores")[:]

    def dynamic_scores(self, layer_idx: int, prompt_idx: int = 0) -> torch.Tensor:
        tmp = self.cache_dir / f"indices{prompt_idx:02d}/"
        with safe_open(tmp / f"indices_{layer_idx:02d}.safetensors", "pt", self.device) as f:
            return f.get_slice("dynamic_scores")[:]

    def aligned_records(
        self, layer_idx: int, prompt_idx: int = 0
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(indices, scores, dynamic_scores) as numpy. Older recordings accumulated
        scores/dynamic_scores across prompts while indices were per-prompt, so slice to
        the trailing indices.shape[1] steps (a no-op for correct recordings)."""
        _, idx_t = self.indices(layer_idx, prompt_idx)
        idx = idx_t.cpu().numpy()
        q = idx.shape[1]
        sc = self.scores(layer_idx, prompt_idx).cpu().numpy()[:, -q:, :]
        dyn = self.dynamic_scores(layer_idx, prompt_idx).cpu().numpy()[:, -q:, :]
        return idx, sc, dyn

    def retrieved_values(self, layer_idx: int, idx: np.ndarray) -> torch.Tensor:
        """Prefill value vectors at recorded top-k positions. idx [H, q, K] ->
        [H, q, K, dim]."""
        assert self.prefill is not None, "needs load_prefill=True"
        layer = self.prefill.layers[layer_idx]
        assert isinstance(layer, CacheLayerMixin) and layer.values is not None
        values = layer.values[0]  # [n_kv, ctx, dim]
        g = self.num_key_value_groups
        idx_t = torch.as_tensor(idx, device=values.device, dtype=torch.long)
        return torch.stack(
            [
                values[h // g].index_select(0, idx_t[h].reshape(-1)).reshape(*idx_t[h].shape, -1)
                for h in range(idx.shape[0])
            ]
        )

    def prefill_kv(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-kv-head prefill (keys, values), each [n_kv, ctx, dim] f32 on device."""
        assert self.prefill is not None, "needs load_prefill=True"
        layer = self.prefill.layers[layer_idx]
        assert layer.keys is not None and layer.values is not None
        c = self.context_len
        return (
            layer.keys[0, :, :c, :].to(self.device, torch.float32),
            layer.values[0, :, :c, :].to(self.device, torch.float32),
        )
