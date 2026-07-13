from qdrant_client.models import VectorParams, PointStruct, Batch, Distance
import compression.zstd
import torch
from safetensors.torch import load
from qdrant_client import QdrantClient
from pathlib import Path
import numpy as np
from rich.progress import track

client = QdrantClient("localhost")


def upsert():
    for i in track(range(8), transient=True, description="Layers"):
        with compression.zstd.open(
            f"cache/qdrant/qwen3_5/layer_{i}_tensors.safetensors.zst",
            "rb",
        ) as f:
            tensors: dict[str, torch.Tensor] = load(f.read())

        for h in track(range(4), transient=True, description="Heads"):
            if client.collection_exists(f"layer={i};head={h}"):
                client.delete_collection(f"layer={i};head={h}")

            client.create_collection(
                collection_name=f"layer={i};head={h}",
                vectors_config=VectorParams(
                    size=tensors["keys"].shape[-1], distance=Distance.COSINE
                ),
            )

            length: int = tensors["prefill_length"]

            ids = torch.split(torch.arange(length, dtype=torch.int), 128)
            keys = torch.split(tensors["keys"][0, h, :length].to(torch.float), 128)
            values = torch.split(tensors["values"][0, h, :length].to(torch.float), 128)

            for idx, k, v in track(zip(ids, keys, values), total=len(keys), transient=True, description="Batch"):
                client.upsert(
                    collection_name=f"layer={i};head={h}",
                    points=Batch(
                        ids=idx.tolist(),
                        vectors=k.numpy(),
                        payloads=[{"value": t} for t in v.tolist()],
                    ),
                )


if __name__ == "__main__":
    upsert()
