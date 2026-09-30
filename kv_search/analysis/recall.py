"""Edge HNSW retrieval quality vs exact truth, per (layer, kv-head), for edge quantization.

Replays recorded decode sessions through a built edge shard (edge.EdgeShard.query_batch, HNSW at
`hnsw_ef`) and scores the returned ids against the exact f32 top-n (tailm_build.scan), with Jojii's
metrics: recall@n and weight recall (share of the exact top-n softmax weight the returned set
covers). Run per shard variant (f32 / f16 / uint8 / ... edge folders) to read off the quality cost
of each storage precision at a fixed graph and query set.

Queries the shard from Python (no server); needs the prefill layer files (exact truth) and the
recorded sessions, so it runs on spark."""

from __future__ import annotations

from pathlib import Path

import qdrant_edge as edge
import torch
from safetensors import safe_open

from kv_search.tailm_build import LayerReader, scan
from kv_search.tailm_validate import Session


def _layer_reader(cache_dir: Path, layer: int) -> LayerReader:
    for suffix in ("", ".zst"):
        p = cache_dir / f"layer_{layer:02d}.safetensors{suffix}"
        if p.exists():
            return LayerReader(p)
    raise FileNotFoundError(f"{cache_dir}: no layer_{layer:02d} file")


def _session_queries(
    sessions: list[Session], layer: int, kv_head: int, group: int, device: str
) -> torch.Tensor:
    """Decode queries of one KV head over all sessions, ordered (session, step, q-head): [R, d] f32.
    Reads only the queries (sessions recorded under -r native carry no attn_out)."""
    qs = []
    for s in sessions:
        with safe_open(str(s.path), framework="pt") as f:
            q = f.get_tensor(f"layer{layer:02d}/queries")[
                :, kv_head * group : (kv_head + 1) * group
            ]
        qs.append(q.reshape(q.shape[0] * group, -1))
    return torch.cat(qs).to(device, torch.float32)


def _hnsw_topn(
    shard: edge.EdgeShard,
    q,
    n: int,
    hnsw_ef: int | None,
    rescore: bool | None = None,
    oversampling: float | None = None,
) -> list[list[int]]:
    """Top-n ids per query row of `q` [B, d] (numpy f32), HNSW search at `hnsw_ef`. On a quantized
    shard, `rescore` toggles rescoring the candidates on the original vectors; None leaves the
    server default (no quantization params sent, so f32/f16 shards are unaffected)."""
    quant = (
        None
        if rescore is None
        else edge.QuantizationSearchParams(rescore=rescore, oversampling=oversampling)
    )
    out: list[list[int]] = []
    for i in range(len(q)):
        r = shard.query(
            edge.QueryRequest(
                query=edge.Query.Nearest(query=q[i], using="key"),
                params=edge.SearchParams(exact=False, hnsw_ef=hnsw_ef, quantization=quant),
                with_payload=False,
                with_vector=None,
                limit=n,
            )
        )
        out.append([int(p.id) for p in r])
    return out


def edge_recall(
    cache_dir: Path,
    edge_root: Path,
    cells: list[tuple[int, int]],
    sessions: list[Session],
    scaling: float,
    group: int,
    top_n: int,
    hnsw_ef: int | None,
    device: str,
    rescore: bool | None = None,
    oversampling: float | None = None,
    q_block: int = 2048,
    key_chunk: int = 16384,
) -> list[dict]:
    results = []
    for layer, head in cells:
        rd = _layer_reader(cache_dir, layer)
        K = rd.head("keys", head).to(device)  # bf16 [N, d]
        V = rd.head("values", head).to(device)
        q = _session_queries(sessions, layer, head, group, device)
        shard = edge.EdgeShard.load(str(edge_root / f"layer{layer:02d}_head{head}"))
        n_q = q.shape[0]
        rec_sum = wrec_sum = 0.0
        for a in range(0, n_q, q_block):
            qb = q[a : a + q_block]
            hs = scan(qb[None], K[None], V[None], scaling, top_n, key_chunk, moments=False).head(0)
            # exact top-n softmax weights over the whole prefill: exp(scaled - m) / D
            w = torch.exp(hs.top_sc - hs.m[:, None]) / hs.D[:, None]  # [B, n]
            ret = _hnsw_topn(
                shard, qb.detach().cpu().numpy(), top_n, hnsw_ef, rescore, oversampling
            )
            ex_ids = hs.top_ix.tolist()
            for j, ret_j in enumerate(ret):
                got = set(ret_j)
                hit = [k for k, i in enumerate(ex_ids[j]) if i in got]
                rec_sum += len(hit) / top_n
                denom = float(w[j].sum())
                wrec_sum += float(w[j, hit].sum()) / denom if denom > 0 else 0.0
        results.append(
            {
                "layer": layer,
                "head": head,
                "n_queries": int(n_q),
                "recall": rec_sum / n_q,
                "weight_recall": wrec_sum / n_q,
            }
        )
    return results
