"""Read-only integration checks against a running custom server and built collection.

QDRANT_ATTENTION_URL=http://localhost:6533 QDRANT_ATTENTION_COLLECTION=pages_from_cache_100k
python -m unittest discover -s tests -p test_qdrant_attention_api.py
"""
import os
import unittest

import numpy as np
import requests

from kv_search.qdrant_pages import PageAttentionClient


@unittest.skipUnless(os.environ.get("QDRANT_ATTENTION_URL"), "requires custom Qdrant and an indexed collection")
class AttentionAPIIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.url = os.environ["QDRANT_ATTENTION_URL"].rstrip("/")
        cls.collection = os.environ.get("QDRANT_ATTENTION_COLLECTION", "pages_from_cache_100k")
        cls.base = f"{cls.url}/collections/{cls.collection}"
        cls.http = requests.Session()
        key = os.environ.get("QDRANT_API_KEY")
        if key:
            cls.http.headers["api-key"] = key
        cls.client = PageAttentionClient(cls.url, collection=cls.collection, api_key=key)
        r = cls.http.get(cls.base, timeout=30)
        r.raise_for_status()
        info = r.json()["result"]
        cls.n = info["points_count"]
        vectors = info["config"]["params"]["vectors"]
        cls.using = next(name for name, spec in sorted(vectors.items()) if spec.get("page_attention"))
        cls.dim = vectors[cls.using]["page_attention"]["head_dim"]
        r = cls.http.post(cls.base + "/points", json={"ids": [0], "with_vector": [cls.using]}, timeout=30)
        r.raise_for_status()
        cls.q = r.json()["result"][0]["vector"][cls.using][:cls.dim]

    @classmethod
    def tearDownClass(cls):
        cls.client.close()
        cls.http.close()

    def call(self, body):
        r = self.http.post(self.base + "/attention", json={"using": self.using, **body}, timeout=60)
        r.raise_for_status()
        return r.json()["result"]

    def test_id_matches_stored_key_query_and_ids_are_omitted_by_default(self):
        for rescore in (False, True):
            by_id = self.call({"query": 0, "rescore": rescore})
            by_vector = self.call({"query": self.q, "rescore": rescore})
            self.assertEqual(by_id, by_vector)
            self.assertEqual(set(by_id), {"attention", "lse"})
            self.assertEqual(len(by_id["attention"]), self.dim)

    def test_top_k_does_not_change_attention_and_candidates_are_retrievable(self):
        for rescore in (False, True):
            baseline = self.call({"query": self.q, "ef": 64, "rescore": rescore})
            for k in (0, 1, 16, 128):
                answer = self.call({"query": self.q, "ef": 64, "rescore": rescore, "return_top_k": k})
                self.assertEqual(answer["attention"], baseline["attention"])
                self.assertEqual(answer["lse"], baseline["lse"])
                if not k:
                    self.assertNotIn("token_ids", answer)
                    continue
                ids = answer["token_ids"]
                self.assertEqual(len(ids), k)
                self.assertEqual(len(set(ids)), k)
                self.assertTrue(all(0 <= i < self.n for i in ids))
                r = self.http.post(self.base + "/points", json={"ids": ids, "with_vector": [self.using]}, timeout=60)
                r.raise_for_status()
                points = r.json()["result"]
                self.assertEqual({p["id"] for p in points}, set(ids))
                self.assertTrue(all(len(p["vector"][self.using]) == 2 * self.dim for p in points))

    def test_batch_mixed_id_and_vector_preserves_order(self):
        batch = [{"query": q, "using": self.using, "return_top_k": k}
                 for q, k in [(self.q, 0), (0, 8), (1, 1), ([0.] * self.dim, 0)]]
        answers = self.client.attention_batch(batch)
        for request, answer in zip(batch, answers):
            self.assertEqual(answer, self.call(request))

    def test_typed_result_matches_legacy_computation(self):
        for rescore in (False, True):
            for q in (self.q, [0.] * self.dim):
                typed = self.call({"query": q, "ef": 64, "rescore": rescore})
                r = self.http.post(self.base + "/points/query", json={
                    "query": q + [0.] * self.dim, "using": self.using,
                    "limit": self.dim + 1, "with_vector": False, "with_payload": False,
                    "params": {"hnsw_ef": 64, "quantization": {"rescore": rescore}},
                }, timeout=60)
                r.raise_for_status()
                old = sorted(r.json()["result"]["points"], key=lambda p: p["id"])
                np.testing.assert_array_equal(np.asarray(typed["attention"] + [typed["lse"]], dtype="f4"),
                                              np.asarray([p["score"] for p in old], dtype="f4"))

    def test_rejects_invalid_and_unsupported_requests(self):
        bad = [({"query": self.n + 1}, 404), ({"query": []}, 400),
               ({"query": self.q + [0.]}, 400), ({"query": 0, "using": "missing"}, 400),
               ({"query": 0, "ef": 0}, 422), ({"query": 0, "return_top_k": 8193}, 422),
               ({"query": 0, "return_top_k": -1}, 400), ({"query": 0, "exact": True}, 400),
               ({"query": 0, "filter": {}}, 400)]
        for body, status in bad:
            with self.subTest(body=body):
                r = self.http.post(self.base + "/attention", json={"using": self.using, **body}, timeout=60)
                self.assertEqual(r.status_code, status, r.text)
        for queries in ([], [{"using": self.using, "query": 0}] * 65,
                        [{"using": self.using, "query": 0, "ef": 0}]):
            r = self.http.post(self.base + "/attention/batch", json={"queries": queries}, timeout=60)
            self.assertEqual(r.status_code, 422, r.text)

    @unittest.skipUnless(os.environ.get("QDRANT_API_KEY"), "server must require authentication")
    def test_endpoint_requires_authentication(self):
        r = requests.post(self.base + "/attention", json={"using": self.using, "query": 0}, timeout=30)
        self.assertIn(r.status_code, (401, 403))


if __name__ == "__main__":
    unittest.main()
