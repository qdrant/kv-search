import rich
from rich.table import Table
import math
from collections.abc import Generator
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from matplotlib.axes import Axes
from rich.progress import track
from safetensors.torch import safe_open
from transformers import AutoConfig, PreTrainedConfig
from transformers.cache_utils import CacheLayerMixin, LinearAttentionCacheLayerMixin
import numpy as np

from kv_search.cache import _repeat_kv, load_cache


class CachedData:
    def __init__(self, cache_dir: Path, model_name: str, device: str = "cuda"):
        self.cache_dir = cache_dir
        self.device = device

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

        self.prefill, self.context_len = load_cache(cache_dir, self.config, device)

    @property
    def full_layers(self) -> Generator[tuple[int, CacheLayerMixin]]:
        for idx, layer in enumerate(self.prefill.layers):
            if isinstance(layer, CacheLayerMixin) and not isinstance(
                layer, LinearAttentionCacheLayerMixin
            ):
                yield idx, layer

    def queries(
        self, layer_idx: int, positions: slice = slice(-1, None)
    ) -> torch.Tensor:
        path = self.cache_dir / f"queries_{layer_idx:02d}.safetensors"
        with safe_open(path, framework="pt", device=self.device) as f:
            q = f.get_slice("queries")[:, positions, :, :]
        return q.transpose(1, 2).contiguous()

    def indices(self, layer_idx: int) -> torch.Tensor:
        path = self.cache_dir / f"indices_{layer_idx:02d}.safetensors"
        with safe_open(path, framework="pt", device=self.device) as f:
            return f.get_slice("indices")[:]

    def plot_mse(self):
        layers = list(self.full_layers)
        fig = plt.figure(figsize=(12, 4 * len(layers)), layout="constrained")
        fig.suptitle(self.config.model_type, fontsize=24)

        axs: list[Axes] = fig.subplots(nrows=len(layers))

        for ax, (layer_idx, layer) in zip(axs, layers):
            assert layer.keys is not None and layer.values is not None

            with torch.no_grad():
                query = self.queries(layer_idx).to(torch.float32)
                keys = _repeat_kv(
                    layer.keys[:, :, : self.context_len, :], self.num_key_value_groups
                ).to(torch.float32)
                values = _repeat_kv(
                    layer.values[:, :, : self.context_len, :], self.num_key_value_groups
                ).to(torch.float32)

                logits = torch.matmul(query, keys.transpose(2, 3)) * self.scaling
                o_full = torch.matmul(torch.softmax(logits, dim=-1), values)

                ordered_idx = torch.argsort(logits, dim=-1, descending=True)
                rand_idx = torch.argsort(torch.rand_like(logits), dim=-1)

                ns = torch.unique(
                    torch.logspace(
                        0, math.log2(self.context_len - 1), 100, 2, dtype=torch.int
                    )
                ).tolist()

                mse = []
                mse_random = []
                for n in track(ns):
                    s_partial = logits.scatter(-1, ordered_idx[..., n:], float("-inf"))
                    w = torch.softmax(s_partial, dim=-1)
                    o = torch.matmul(w, values)
                    mse.append(torch.nn.functional.mse_loss(o_full, o))

                    s_partial = logits.scatter(-1, rand_idx[..., n:], float("-inf"))
                    w = torch.softmax(s_partial, dim=-1)
                    o = torch.matmul(w, values)
                    mse_random.append(torch.nn.functional.mse_loss(o_full, o))

                mse = torch.stack(mse).cpu().numpy()
                mse_random = torch.stack(mse_random).cpu().numpy()

                ax.semilogx(ns, mse, base=2, label="top-k")
                ax.semilogx(ns, mse_random, base=2, label="random")
                ax.set_xlabel("k in top-k")
                ax.set_ylabel("MSE $\\left|o - \\tilde{o}\\right|^2$")
                ax.set_title(f"Layer {layer_idx}")
                ax.legend()
        fig.savefig(self.cache_dir / "mse.png")

    def plot_indices(self) -> None:
        layers = list(self.full_layers)
        fig = plt.figure(figsize=(12, 2.5 * len(layers)), layout="constrained")
        fig.suptitle(self.config.model_type, fontsize=24)

        axs: list[Axes] = fig.subplots(nrows=len(layers))

        n_bins = 512

        for ax, (layer_idx, layer) in zip(axs, layers):
            indices = self.indices(layer_idx).cpu().numpy()  # [16, q_len, n]

            heat = np.zeros((indices.shape[1], n_bins))  # [q_len, c_len (binned)]

            bins = indices * n_bins // self.context_len

            for head_idx in range(indices.shape[0]):
                present = np.zeros(
                    (indices.shape[1], n_bins), dtype=bool
                )  # [q_len, c_len (binned)]
                np.put_along_axis(present, bins[head_idx], True, axis=1)
                heat += present

            ax.imshow(
                heat,
                cmap="magma",
                aspect="auto",
                vmin=0,
                vmax=indices.shape[0],
                interpolation="nearest",
            )

        fig.savefig(self.cache_dir / "index_heatmap.png")

    def index_stats(self) -> None:
        layers = list(self.full_layers)
        table = Table("Layer", "# touched", "Hit Ceiling", "Retention", "Redundancy")
        for layer_idx, layer in layers:
            indices: np.ndarray = (
                self.indices(layer_idx).cpu().numpy()
            )  # [16, q_len, n]
            num_touched_per_head = np.mean(
                [np.unique(indices[h]).size for h in range(indices.shape[0])],
                dtype=float,
            )
            hit_ceiling = 1 - num_touched_per_head / indices.shape[1] / indices.shape[2]
            retention = np.mean(
                [
                    len(set(indices[h, t]) & set(indices[h, t - 1]))
                    for h in range(indices.shape[0])
                    for t in range(1, indices.shape[1])
                ],
                dtype=float,
            )
            redundancy = np.mean(
                [
                    indices.shape[0]
                    * indices.shape[2]
                    / np.unique(indices[:, t, :]).size
                    for t in range(indices.shape[1])
                ]
            )

            table.add_row(
                str(layer_idx),
                f"{num_touched_per_head:.2f}",
                f"{hit_ceiling:.2f}",
                f"{retention:.2f}",
                f"{redundancy:.2f}",
            )

        rich.print(table)
