import argparse
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

spec = importlib.util.spec_from_file_location(
    "qdrant_pages_ingest", Path(__file__).resolve().parents[1] / "scripts/qdrant_pages_ingest.py")
ingest = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ingest)


class PreparationTests(unittest.TestCase):
    def test_bf16_query_mapping_preserves_bits_and_rejects_truncation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "queries.safetensors"
            bits = np.arange(2 * 4 * 128, dtype="<u2").reshape(1, 2, 4, 128)
            header = json.dumps({"queries": {"dtype": "BF16", "shape": list(bits.shape),
                                             "data_offsets": [0, bits.nbytes]}}).encode()
            blob = struct.pack("<Q", len(header)) + header + bits.tobytes()
            path.write_bytes(blob)
            q = ingest.query_array(path)
            np.testing.assert_array_equal(q, bits[0])
            del q
            path.write_bytes(blob[:-2])
            with self.assertRaisesRegex(ValueError, "expected BF16"):
                ingest.query_array(path)

    def test_missing_queries_fails_before_build_or_work_directory_creation(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache, work = Path(tmp) / "cache", Path(tmp) / "work"
            cache.mkdir()
            (cache / "meta.json").write_text('{"context_len":513}')
            (cache / "layer_03.safetensors.zst").touch()
            args = argparse.Namespace(cache=cache, work_dir=work, stride=64)
            values = np.zeros((2, 513, 128), dtype="<u2")
            with patch.object(ingest, "layer_arrays", return_value=(values, values)):
                with self.assertRaisesRegex(ValueError, "Missing.*queries_03"):
                    ingest.prepare(args)
            self.assertFalse(work.exists())

    def test_manifest_binding_and_original_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            manifest = {"context_len": 129, "spec": {"head_dim": 128, "num_layers": 1, "num_kv_heads": 1}}
            ingest.write_json(work / "manifest.json", manifest)
            (work / "pages").mkdir()
            (work / "pages/source.fnv").write_text(ingest.fnv((work / "manifest.json").read_bytes()))
            stem = "l0000h0000"
            raw, head = work / "raw" / stem, work / "heads" / stem
            raw.mkdir(parents=True)
            head.mkdir(parents=True)
            page_hashes, original_hashes = {}, {}
            for ext in ingest.EXTENSIONS:
                path = work / "pages" / f"{stem}.{ext}"
                path.write_bytes(b"page-generation")
                page_hashes[path.name] = ingest.sha256(path)
            for name in ("keys.bf16", "values.bf16"):
                (raw / name).write_bytes(bytes(129 * 128 * 2))
                original_hashes[name] = ingest.sha256(raw / name)
            ingest.write_json(head / "complete.json", {"pages": page_hashes, "originals": original_hashes})
            self.assertEqual(ingest.validate_generation(work), manifest)
            with (raw / "keys.bf16").open("r+b") as f:
                f.write(b"\x01")
            with self.assertRaisesRegex(ValueError, "Original row checksum mismatch"):
                ingest.validate_generation(work)
            (work / "manifest.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "manifest checksum mismatch"):
                ingest.validate_generation(work)


if __name__ == "__main__":
    unittest.main()
