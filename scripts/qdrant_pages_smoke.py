"""CPU-only regression check using real cached queries and frozen reference outputs.

No model weights, GPU, prototype server, or native Python extension are needed.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kv_search.qdrant_pages import PageAttentionClient


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://localhost:6433")
    ap.add_argument("--grpc-port", type=int, default=6434)
    ap.add_argument("--collection", default="pages_100k")
    ap.add_argument("--api-key")
    data = Path(os.environ.get("KV_SEARCH_DATA", Path(__file__).resolve().parents[2] / "kv-search-data"))
    ap.add_argument("--fixture", type=Path, default=data / "tests/fixtures/qdrant-pages-100k.npz")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    if not args.fixture.is_file():
        ap.error(f"Missing external fixture: {args.fixture}; use --fixture or KV_SEARCH_DATA")
    rows = []
    client = PageAttentionClient(args.url, args.grpc_port, args.collection, args.api_key)
    try:
        with np.load(args.fixture, allow_pickle=False) as fixture:
            queries = fixture["queries"]
            for rescore in (False, True):
                for ef in (16, 64):
                    max_out = max_lse = 0.0
                    started = time.perf_counter()
                    for layer, query in enumerate(queries):
                        client.validate(layer, 4, query.shape[-1], int(fixture["context_len"]),
                                        rescore=rescore)
                        out, lse = client.query(query, layer, 4, ef=ef, rescore=rescore,
                                               batch_size=19)
                        expected_out = fixture[f"out_{ef}_{int(rescore)}"][layer]
                        expected_lse = fixture[f"lse_{ef}_{int(rescore)}"][layer]
                        # AVX2 and AVX512 have different weight-grid precision.
                        # Fixtures were frozen on AVX512; allow small SIMD drift.
                        np.testing.assert_allclose(out, expected_out, atol=3e-3, rtol=2e-2)
                        np.testing.assert_allclose(lse, expected_lse, atol=2e-3, rtol=2e-4)
                        max_out = max(max_out, float(np.max(np.abs(out - expected_out))))
                        max_lse = max(max_lse, float(np.max(np.abs(lse - expected_lse))))
                    row = {"ef": ef, "rescore": rescore, "heads_checked": int(np.prod(queries.shape[:3])),
                           "max_abs_out": max_out, "max_abs_lse": max_lse,
                           "elapsed_seconds": time.perf_counter() - started}
                    print(json.dumps(row), flush=True)
                    rows.append(row)
    finally:
        client.close()
    report = {"collection": args.collection, "checks": rows,
              "note": "CPU cached-query regression; not model E2E or a latency benchmark"}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
