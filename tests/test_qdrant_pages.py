import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from qdrant_client import grpc

from kv_search.qdrant_pages import PageAttentionClient


class PageAttentionTests(unittest.TestCase):
    def client(self, transform=lambda points: points):
        client = PageAttentionClient.__new__(PageAttentionClient)
        client.collection, client.timeout = "demo", 10
        self.requests = []
        def query_batch(batch, timeout):
            self.requests.extend(batch.query_points)
            results = []
            for request in batch.query_points:
                values = list(request.query.nearest.dense.data)
                dim = len(values) // 2
                self.assertEqual(values[dim:], [0] * dim)
                scores = values[:dim] + [42.0]
                # Qdrant sorts the channel by score, never by coordinate ID.
                points = [grpc.ScoredPoint(id=grpc.PointId(num=i), score=s)
                          for i, s in enumerate(scores)]
                points.sort(key=lambda p: p.score, reverse=True)
                results.append(SimpleNamespace(result=transform(points)))
            return SimpleNamespace(result=results)
        client.client = SimpleNamespace(grpc_points=SimpleNamespace(QueryBatch=query_batch))
        return client

    def test_multi_token_grouping_chunking_and_negative_coordinates(self):
        client = self.client()
        query = (np.arange(6 * 3 * 16, dtype=np.float32) - 200).reshape(6, 3, 16)
        out, lse = client.query(query, 2, 2, ef=32, rescore=True, batch_size=7)
        np.testing.assert_array_equal(out, query)
        np.testing.assert_array_equal(lse, np.full((6, 3), 42.0))
        self.assertEqual([r.using for r in self.requests],
                         [f"l0002h{h // 3:04}" for _ in range(3) for h in range(6)])
        for request in self.requests:
            self.assertEqual(request.limit, 17)
            self.assertEqual(request.params.hnsw_ef, 32)
            self.assertTrue(request.params.quantization.rescore)
            self.assertFalse(request.with_vectors.enable)

    def test_rejects_invalid_channel(self):
        for transform in (lambda p: p[:-1], lambda p: p[:-1] + [p[0]],
                          lambda p: [grpc.ScoredPoint(id=grpc.PointId(uuid="bad"), score=0)] + p[1:],
                          lambda p: [grpc.ScoredPoint(id=p[0].id, score=float("nan"))] + p[1:]):
            with self.subTest(transform=transform), self.assertRaises(RuntimeError):
                self.client(transform).query(np.zeros((4, 1, 16)), 0, 1)

    def test_rejects_wrong_scaling_and_invalid_queries_before_rpc(self):
        client = self.client()
        for kwargs in ({"scaling": 0.1}, {"ef": 0}, {"batch_size": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                client.query(np.zeros((4, 1, 16)), 0, 1, **kwargs)
        with self.assertRaises(ValueError):
            client.query(np.full((4, 1, 16), np.nan), 0, 1)
        self.assertEqual(self.requests, [])

    def test_rejects_incomplete_batch(self):
        client = self.client()
        client.client.grpc_points.QueryBatch = Mock(return_value=SimpleNamespace(result=[]))
        with self.assertRaisesRegex(RuntimeError, "Incomplete"):
            client.query(np.zeros((4, 1, 16)), 0, 1)

    def test_rejects_ordinary_collection_and_zero_rescore_budget(self):
        client = self.client()
        client.url, client.headers, client._validated = "http://localhost", {}, set()
        vector = {"size": 32, "distance": "Dot"}
        info = {"status": "green", "points_count": 11, "indexed_vectors_count": 11,
                "config": {"params": {"shard_number": 1, "vectors": {"l0000h0000": vector}}}}
        response = Mock()
        response.json.return_value = {"result": info}
        with patch("kv_search.qdrant_pages.requests.get", return_value=response) as get:
            with self.assertRaisesRegex(ValueError, "not the matching"):
                client.validate(0, 1, 16, 11)
            vector["page_attention"] = {"head_dim": 16, "layer": 0, "head": 0, "rescore": 0}
            client.validate(0, 1, 16, 11)
            client.validate(0, 1, 16, 11)
            self.assertEqual(get.call_count, 2)
            with self.assertRaisesRegex(ValueError, "budget is zero"):
                client.validate(0, 1, 16, 11, rescore=True)
            with self.assertRaisesRegex(ValueError, "context length"):
                client.validate(0, 1, 16, 12)


if __name__ == "__main__":
    unittest.main()
