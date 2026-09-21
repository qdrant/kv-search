import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import torch
from transformers.cache_utils import DynamicLayer

from kv_search.cache import QdrantPagesRetriever, _merge_partitions


class RetrieverTests(unittest.TestCase):
    def test_hybrid_layer_mapping_and_multi_token_attention_merge(self):
        dense = DynamicLayer()
        dense.keys = torch.zeros(1, 2, 11, 16)
        prefill = SimpleNamespace(layers=[object(), object(), object(), dense,
                                         object(), object(), object(), dense])
        retriever = QdrantPagesRetriever(rescore=True)
        out = np.arange(6 * 2 * 16, dtype=np.float32).reshape(6, 2, 16) / 256
        lse = np.full((6, 2), 2.345678, dtype=np.float32)
        transport = Mock()
        transport.query.return_value = (out, lse)
        retriever.__dict__["_client"] = transport
        query = torch.zeros(1, 6, 2, 16, dtype=torch.bfloat16)
        remote = retriever.retrieve(query, 7, prefill, 0.25)
        transport.validate.assert_called_once_with(1, 2, 16, 11, rescore=True)
        args, kwargs = transport.query.call_args
        self.assertEqual(args[1:], (1, 2))
        self.assertEqual(args[0].shape, (6, 2, 16))
        self.assertTrue(kwargs["rescore"])
        self.assertEqual(remote.out.shape, query.shape)
        self.assertEqual(remote.lse.dtype, torch.float32)
        torch.testing.assert_close(remote.lse, torch.from_numpy(lse).unsqueeze(0), rtol=0, atol=0)
        # A unit-mass live token combined with the returned remote partition.
        live = torch.ones_like(remote.out, dtype=torch.float32)
        merged = _merge_partitions(remote.out, remote.lse, live, torch.zeros_like(remote.lse))
        mass = torch.exp(remote.lse).unsqueeze(-1)
        expected = (mass * remote.out.float() + live) / (mass + 1)
        torch.testing.assert_close(merged, expected)

    def test_batch_size_guard(self):
        with self.assertRaisesRegex(ValueError, "batch size 1"):
            QdrantPagesRetriever().retrieve(torch.zeros(2, 4, 1, 16), 0, None, 0.25)


if __name__ == "__main__":
    unittest.main()
