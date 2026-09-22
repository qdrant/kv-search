"""Client for the experimental typed Qdrant attention REST API."""
import math

import numpy as np
import requests


class PageAttentionClient:
    def __init__(self, url="http://localhost:6433", grpc_port=6434,
                 collection="pages_100k", api_key=None, timeout=300):
        self.url = url.rstrip("/")
        self.collection = collection
        self.timeout = timeout
        self.headers = {"api-key": api_key} if api_key else {}
        # grpc_port remains accepted for compatibility with retriever settings.
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        self._validated = set()

    def validate(self, layer, kv_heads, dim, context_len, *, rescore=False):
        """Validate the immutable collection against the model's cached context."""
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

    def attention(self, query, using, *, ef=16, rescore=False, return_top_k=0):
        """One Q vector or point ID -> dict with attention, LSE, optional token_ids.

        An ID uses the stored K as Q. Candidate IDs describe the best tokens in
        scanned pages, not all contributors to the approximate full attention.
        """
        if isinstance(query, np.ndarray):
            query = query.tolist()
        return self.attention_batch([dict(query=query, using=using, ef=ef,
                                         rescore=rescore, return_top_k=return_top_k)])[0]

    def attention_batch(self, queries):
        if not 1 <= len(queries) <= 64:
            raise ValueError("Attention batch must contain 1..64 queries")
        response = self.session.post(
            f"{self.url}/collections/{self.collection}/attention/batch",
            json={"queries": queries}, timeout=self.timeout,
        )
        response.raise_for_status()
        results = response.json()["result"]
        if not isinstance(results, list) or len(results) != len(queries):
            raise RuntimeError("Incomplete page-attention query batch")
        for request, result in zip(queries, results, strict=True):
            attention = np.asarray(result.get("attention"), dtype=np.float32)
            if (attention.ndim != 1 or not attention.size or not np.isfinite(attention).all()
                    or not math.isfinite(result.get("lse", float("nan")))):
                raise RuntimeError("Invalid page-attention output")
            k = request.get("return_top_k", 0)
            ids = result.get("token_ids")
            if k:
                if (not isinstance(ids, list) or len(ids) > k
                        or any(type(i) is not int or i < 0 for i in ids)
                        or len(set(ids)) != len(ids)):
                    raise RuntimeError("Invalid page-attention token IDs")
            elif "token_ids" in result:
                raise RuntimeError("Unexpected token IDs in attention-only response")
        return results

    def query(self, queries, layer, kv_heads, *, ef=16, rescore=False,
              scaling=None, batch_size=64, return_top_k=0):
        """[q_heads, query_tokens, dim] -> (attention, LSE), both float32.

        The server integrates the whole immutable context and uses 1/sqrt(dim).
        With return_top_k > 0, also returns IDs as [head][query_token][candidate].
        """
        queries = np.asarray(queries, dtype=np.float32)
        if queries.ndim != 3 or min(queries.shape) <= 0:
            raise ValueError("Expected nonempty [query_heads, query_tokens, dim]")
        heads, tokens, dim = queries.shape
        if (kv_heads <= 0 or heads % kv_heads or not 1 <= ef <= 8192
                or not 1 <= batch_size <= 64 or layer < 0 or not 0 <= return_top_k <= 8192):
            raise ValueError("Invalid head grouping, layer, ef, or batch size")
        if not np.isfinite(queries).all():
            raise ValueError("Queries must be finite")
        if scaling is not None and not math.isclose(scaling, dim ** -0.5, rel_tol=1e-6):
            raise ValueError("The demo server only supports attention scaling 1/sqrt(head_dim)")
        group = heads // kv_heads
        out = np.empty_like(queries)
        lse = np.empty((heads, tokens), dtype=np.float32)
        token_ids = [[None for _ in range(tokens)] for _ in range(heads)] if return_top_k else None
        total = heads * tokens
        for start in range(0, total, batch_size):
            indices = [(i % heads, i // heads) for i in range(start, min(start + batch_size, total))]
            batch = []
            for h, t in indices:
                batch.append(dict(
                    query=queries[h, t].tolist(),
                    using=f"l{layer:04}h{h // group:04}",
                    ef=ef, rescore=rescore, return_top_k=return_top_k,
                ))
            results = self.attention_batch(batch)
            for (h, t), result in zip(indices, results, strict=True):
                if len(result["attention"]) != dim:
                    raise RuntimeError("Invalid page-attention output dimension")
                out[h, t] = result["attention"]
                lse[h, t] = result["lse"]
                if token_ids is not None:
                    token_ids[h][t] = result["token_ids"]
        return (out, lse, token_ids) if return_top_k else (out, lse)

    def close(self):
        self.session.close()
