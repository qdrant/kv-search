"""Transport for the experimental Qdrant page-attention score channel."""
import math

import numpy as np
import requests
from qdrant_client import QdrantClient, grpc


class PageAttentionClient:
    def __init__(self, url="http://localhost:6433", grpc_port=6434,
                 collection="pages_100k", api_key=None, timeout=300):
        self.url = url.rstrip("/")
        self.collection = collection
        self.timeout = timeout
        self.headers = {"api-key": api_key} if api_key else {}
        self.client = QdrantClient(url, grpc_port=grpc_port, api_key=api_key,
                                  prefer_grpc=True, timeout=timeout)
        self._validated = set()

    def validate(self, layer, kv_heads, dim, context_len, *, rescore=False):
        """Fail before treating an ordinary point result as attention coordinates."""
        key = (layer, kv_heads, dim, context_len, rescore)
        if key in self._validated:
            return
        response = requests.get(
            f"{self.url}/collections/{self.collection}",
            headers=self.headers, timeout=self.timeout,
        )
        response.raise_for_status()
        info = response.json()["result"]
        config = info["config"]["params"]
        vectors = config["vectors"]
        if (config.get("shard_number") != 1 or info["status"] != "green"
                or info["points_count"] != context_len
                or info.get("indexed_vectors_count") != context_len * len(vectors)):
            raise ValueError("Page-attention collection must be fully indexed, single-shard, "
                             "and match the cached context length")
        for head in range(kv_heads):
            spec = vectors.get(f"l{layer:04}h{head:04}", {})
            index = spec.get("page_attention") or {}
            if (spec.get("size") != 2 * dim or spec.get("distance") != "Dot"
                    or index.get("head_dim") != dim or index.get("layer") != layer
                    or index.get("head") != head):
                raise ValueError("Collection is not the matching page-attention index; "
                                 "use the custom Qdrant image and matching cache")
            if rescore and not index.get("rescore", 0):
                raise ValueError("Rescore requested, but the collection's rescore budget is zero")
        self._validated.add(key)

    def query(self, queries, layer, kv_heads, *, ef=16, rescore=False,
              scaling=None, batch_size=64):
        """[q_heads, query_tokens, dim] -> (attention, LSE), both float32.

        The server integrates the whole immutable context and uses 1/sqrt(dim).
        Its returned scores are coordinates/LSE, not similarity scores or IDs.
        """
        queries = np.asarray(queries, dtype=np.float32)
        if queries.ndim != 3 or min(queries.shape) <= 0:
            raise ValueError("Expected nonempty [query_heads, query_tokens, dim]")
        heads, tokens, dim = queries.shape
        if kv_heads <= 0 or heads % kv_heads or ef <= 0 or batch_size <= 0 or layer < 0:
            raise ValueError("Invalid head grouping, layer, ef, or batch size")
        if not np.isfinite(queries).all():
            raise ValueError("Queries must be finite")
        if scaling is not None and not math.isclose(scaling, dim ** -0.5, rel_tol=1e-6):
            raise ValueError("The demo server only supports attention scaling 1/sqrt(head_dim)")
        group = heads // kv_heads
        out = np.empty_like(queries)
        lse = np.empty((heads, tokens), dtype=np.float32)
        total = heads * tokens
        for start in range(0, total, batch_size):
            indices = [(i % heads, i // heads) for i in range(start, min(start + batch_size, total))]
            requests = []
            for h, t in indices:
                padded = np.zeros(2 * dim, dtype=np.float32)
                padded[:dim] = queries[h, t]
                requests.append(grpc.QueryPoints(
                    collection_name=self.collection,
                    query=grpc.Query(nearest=grpc.VectorInput(
                        dense=grpc.DenseVector(data=padded.tolist()))),
                    using=f"l{layer:04}h{h // group:04}",
                    params=grpc.SearchParams(hnsw_ef=ef,
                        quantization=grpc.QuantizationSearchParams(rescore=rescore)),
                    limit=dim + 1,
                    with_payload=grpc.WithPayloadSelector(enable=False),
                    with_vectors=grpc.WithVectorsSelector(enable=False),
                ))
            results = self.client.grpc_points.QueryBatch(grpc.QueryBatchPoints(
                collection_name=self.collection, query_points=requests), timeout=self.timeout).result
            if len(results) != len(indices):
                raise RuntimeError("Incomplete page-attention query batch")
            for (h, t), result in zip(indices, results, strict=True):
                points = sorted(result.result, key=lambda p: p.id.num)
                if (any(not p.id.HasField("num") for p in points)
                        or [p.id.num for p in points] != list(range(dim + 1))):
                    raise RuntimeError("Invalid page-attention score channel; check index and segment count")
                scores = np.asarray([p.score for p in points], dtype=np.float32)
                if not np.isfinite(scores).all():
                    raise RuntimeError("Non-finite page-attention output")
                out[h, t] = scores[:dim]
                lse[h, t] = scores[dim]
        return out, lse

    def close(self):
        self.client.close()
