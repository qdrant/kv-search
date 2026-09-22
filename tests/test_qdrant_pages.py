import unittest
from unittest.mock import Mock, patch

import numpy as np

from kv_search.qdrant_pages import PageAttentionClient


class PageAttentionTests(unittest.TestCase):
    def client(self, transform=lambda result: result):
        client = PageAttentionClient("http://localhost", collection="demo", timeout=10)
        self.requests = []

        def post(url, json, timeout):
            self.assertTrue(url.endswith("/collections/demo/attention/batch"))
            self.requests.extend(json["queries"])
            results = []
            for request in json["queries"]:
                query = request["query"]
                result = {"attention": query if isinstance(query, list) else [float(query)] * 16,
                          "lse": 42.0}
                if request.get("return_top_k", 0):
                    result["token_ids"] = list(range(min(request["return_top_k"], 3)))
                results.append(transform(result))
            response = Mock()
            response.json.return_value = {"result": results}
            return response

        client.session.post = Mock(side_effect=post)
        self.addCleanup(client.close)
        return client

    def test_multi_token_grouping_chunking_and_negative_coordinates(self):
        client = self.client()
        query = (np.arange(6 * 3 * 16, dtype=np.float32) - 200).reshape(6, 3, 16)
        out, lse = client.query(query, 2, 2, ef=32, rescore=True, batch_size=7)
        np.testing.assert_array_equal(out, query)
        np.testing.assert_array_equal(lse, np.full((6, 3), 42.0))
        self.assertEqual([r["using"] for r in self.requests],
                         [f"l0002h{h // 3:04}" for _ in range(3) for h in range(6)])
        for request in self.requests:
            self.assertEqual(len(request["query"]), 16)
            self.assertEqual(request["ef"], 32)
            self.assertTrue(request["rescore"])
            self.assertEqual(request["return_top_k"], 0)

    def test_id_query_and_optional_candidates(self):
        client = self.client()
        self.assertNotIn("token_ids", client.attention(42, "l0000h0000"))
        self.assertEqual(self.requests[-1]["query"], 42)
        result = client.attention([1.0] * 16, "l0000h0000", return_top_k=5)
        self.assertEqual(result["token_ids"], [0, 1, 2])
        out, lse, ids = client.query(np.ones((4, 2, 16)), 0, 1, return_top_k=2)
        np.testing.assert_array_equal(out, np.ones((4, 2, 16)))
        self.assertEqual(ids, [[[0, 1], [0, 1]]] * 4)

    def test_rejects_malformed_typed_response(self):
        for transform in (lambda r: {**r, "attention": r["attention"][:-1]},
                          lambda r: {**r, "attention": [float("nan")] * 16},
                          lambda r: {**r, "lse": float("inf")},
                          lambda r: {**r, "token_ids": [0]}):
            with self.subTest(transform=transform), self.assertRaises(RuntimeError):
                self.client(transform).query(np.zeros((4, 1, 16)), 0, 1)
        for ids in ([1, 1], [-1], ["bad"], [1, 2, 3], None):
            with self.subTest(ids=ids), self.assertRaises(RuntimeError):
                self.client(lambda r: {**r, "token_ids": ids}).attention([0.] * 16, "h", return_top_k=2)

    def test_rejects_wrong_scaling_and_invalid_queries_before_rpc(self):
        client = self.client()
        for kwargs in ({"scaling": 0.1}, {"ef": 0}, {"batch_size": 0},
                       {"batch_size": 65}, {"return_top_k": -1}, {"return_top_k": 8193}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                client.query(np.zeros((4, 1, 16)), 0, 1, **kwargs)
        with self.assertRaises(ValueError):
            client.query(np.full((4, 1, 16), np.nan), 0, 1)
        self.assertEqual(self.requests, [])

    def test_rejects_incomplete_batch(self):
        client = self.client()
        response = Mock()
        response.json.return_value = {"result": []}
        client.session.post = Mock(return_value=response)
        with self.assertRaisesRegex(RuntimeError, "Incomplete"):
            client.query(np.zeros((4, 1, 16)), 0, 1)

    def test_rejects_ordinary_collection_and_zero_rescore_budget(self):
        client = self.client()
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
