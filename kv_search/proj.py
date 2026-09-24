"""Query-aware projection edges for the key HNSW graph (usage: `docs/proj-build.md`).

Plain HNSW links a key to the keys close to it. Attention queries do not look like keys: the
keys one query retrieves together can sit far apart in key space, so the graph has no short
path between them and a search with a small `ef` misses part of the top-n. The projection pass
rewires level 0 from the queries' point of view: for a set of training queries it takes the
exact top-`topn` keys, counts how often two keys are retrieved by the same query, and gives each
key up to `m` edges to its most frequently co-retrieved keys. Search is unchanged; only the
level-0 links differ, so the shards load with the upstream qdrant-edge crate.

The pass runs inside a Qdrant fork (`hnsw_config.projection` +
`PUT /collections/{name}/hnsw_training_vectors`), used as the build server only. This module
supplies what the fork needs from kv-search:

1. the training queries. Decode queries do not exist at build time, so the prompt's own queries
   stand in (recorded post-RoPE by `RecordingCache` into `queries_LL.safetensors`): the last
   `window` prompt positions of the kv-head's GROUP q-heads, each RoPE-shifted from its prompt
   position to a decode position `context_len + U[0, shift_span)`.
2. the REST calls: the chunked training-vector upload (the server caps requests at 32 MB) and
   the PATCH that turns on HNSW with the `projection` block (raw REST: qdrant-client's models
   forbid the unknown field).

Two recipes: `plain` (no excluded points, no repair) and `barred` (the attention sinks
`excluded_points` never receive a projected edge; a reachability repair runs after the pass).
A barred graph only pays off when the decode side scores the excluded points directly, which
this repo does not do yet, so `chat` refuses barred shards.
"""

import json
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import requests

from kv_search.tailm import (
    _query_file_layout,
    load_prompt_queries_at,
    rope_inv_freq,
    rope_shift,
)

GROUP = 4  # query heads per kv-head (GQA)
PROJ_JSON = "proj.json"

Recipe = Literal["plain", "barred"]
RECIPES: tuple[Recipe, ...] = ("plain", "barred")


@dataclass(frozen=True)
class ProjConfig:
    """Knobs of the fork's projection pass and of the training-query builder. The defaults are
    the plain recipe: m=16 projected edges, exact top-100 per training query, at most 64
    retrievers sampled per key, 200 co-retrieval candidates, an 8,192-position window."""

    m: int = 16
    topn: int = 100
    maxq: int = 64
    cands: int = 200
    seed: int = 0
    # training queries: the last `window` prompt positions x GROUP heads, each RoPE-shifted to
    # context_len + U[0, shift_span), the positions the decode queries will actually have
    window: int = 8192
    shift_span: int = 256
    # upper bound on the training vectors the server uses; larger uploads are sampled down to
    # it, silently (fork default 100,000). None = every uploaded vector.
    max_training_vectors: int | None = None
    # barred recipe: point ids (= token positions) that never receive a projected edge, and
    # the reachability repair after the pass with its per-point cap
    excluded_points: tuple[int, ...] = ()
    repair: bool = False
    repair_max_per_point: int = 32

    @property
    def recipe(self) -> Recipe:
        return "barred" if self.excluded_points or self.repair else "plain"

    @classmethod
    def from_recipe(
        cls, recipe: str, exclude: tuple[int, ...] = (0, 1, 2), **knobs: Any
    ) -> "ProjConfig":
        """`plain` ignores `exclude`; `barred` excludes it and turns the repair on."""
        if recipe == "plain":
            return cls(**knobs, excluded_points=(), repair=False)
        if recipe == "barred":
            return cls(**knobs, excluded_points=tuple(exclude), repair=True)
        raise ValueError(f"unknown proj recipe {recipe!r} ({' | '.join(RECIPES)})")

    def server_block(self, n_training: int) -> dict[str, Any]:
        """The `hnsw_config.projection` object the fork expects.

        A whitelist: `window`/`shift_span` only decide which vectors are uploaded. Every field
        is sent, because the fork's own defaults are the barred recipe. `n_training` is the
        number of vectors actually uploaded; with `max_training_vectors=None` it becomes the
        bound, so the server uses the whole set instead of its default cap.
        """
        cap = self.max_training_vectors
        if cap is None:
            cap = max(n_training, 1)
        elif cap < n_training:
            raise ValueError(
                f"max_training_vectors={cap} < {n_training} uploaded training vectors: the "
                "server would silently sample the set down. Raise the bound (or leave it "
                "unset) or upload fewer vectors."
            )
        return {
            "m": self.m,
            "topn": self.topn,
            "maxq": self.maxq,
            "cands": self.cands,
            "seed": self.seed,
            "max_training_vectors": cap,
            "excluded_points": list(self.excluded_points),
            "repair": self.repair,
            "repair_max_per_point": self.repair_max_per_point,
        }

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def save(root: Path, cfg: ProjConfig) -> None:
    """Record the recipe next to the shards, so chat knows what it loads."""
    root.mkdir(parents=True, exist_ok=True)
    (root / PROJ_JSON).write_text(json.dumps(cfg.to_json(), indent=1))


def load(root: Path) -> ProjConfig | None:
    """The recipe of a shard folder, or None for a folder without projection edges.

    Keys this version does not know (`positions`, `sample`, ...) are ignored, but a training
    source other than the prefill queries is refused: this version cannot rebuild it.
    """
    path = root / PROJ_JSON
    if not path.exists():
        return None
    doc = json.loads(path.read_text())
    source = doc.get("source", "prefill")
    if source != "prefill":
        raise ValueError(f"{path}: training source {source!r}, only 'prefill' is supported")
    known = {f.name for f in fields(ProjConfig)}
    cfg = ProjConfig(**{k: v for k, v in doc.items() if k in known})
    return replace(cfg, excluded_points=tuple(cfg.excluded_points))


# ------------------------------------------------------------------ training queries


def preflight(cache_dir: Path, layers: list[int], context_len: int, kv_heads: int) -> None:
    """Before any upload: every layer's `queries_LL.safetensors` exists, comes from the same
    prefill as the keys (one row per prompt position) and holds GROUP q-heads per kv-head."""
    for layer in layers:
        path = cache_dir / f"queries_{layer:02d}.safetensors"
        if not path.exists():
            raise SystemExit(
                f"error: --proj: {path} missing; the training queries are recorded by the "
                "prefill, run it again (without --skip-prefill)"
            )
        _, (_, seq, heads, _) = _query_file_layout(path)
        if seq != context_len:
            raise SystemExit(
                f"error: --proj: {path} holds {seq} positions, the cache {context_len}: the "
                "queries come from another prefill"
            )
        if heads < GROUP * kv_heads:
            raise SystemExit(
                f"error: --proj: {path} holds {heads} q-heads, need {GROUP * kv_heads}"
            )


def training_queries(
    cache_dir: Path,
    model_config: Any,
    layer: int,
    kv_head: int,
    context_len: int,
    cfg: ProjConfig,
) -> npt.NDArray[np.float32]:
    """The training set of one (layer, kv-head): f32 `[GROUP * window, dim]`, head-major.

    The last `window` prompt queries of each of the kv-head's q-heads, RoPE-shifted to
    `context_len + U[0, shift_span)`: one random target per position, shared by the GROUP
    heads, seeded by (`cfg.seed`, layer, kv_head) so a rebuild reproduces the set.
    `model_config` must be the tier's (YaRN-patched above the native window,
    `main.load_model_config`), so the shift uses the prefill's own frequencies.
    """
    window = min(cfg.window, context_len)
    pos = np.arange(context_len - window, context_len)
    q = load_prompt_queries_at(cache_dir, layer, kv_head, pos, GROUP)
    inv_freq = rope_inv_freq(model_config)
    rng = np.random.default_rng(cfg.seed + 1000 * layer + kv_head)
    target = context_len + rng.integers(0, cfg.shift_span, size=window)
    delta = (target - pos).astype(np.float64)
    return np.concatenate([rope_shift(q[g], delta, inv_freq) for g in range(GROUP)])


# ------------------------------------------------------------------ REST (the fork)


def _headers(api_key: str | None) -> dict[str, str] | None:
    return {"api-key": api_key} if api_key else None


def upload_training_vectors(
    rest_uri: str,
    collection: str,
    vectors: npt.ArrayLike,
    api_key: str | None = None,
    vector_name: str = "key",
    chunk_rows: int = 2048,
) -> dict[str, Any]:
    """PUT the training vectors in chunks (the server caps requests at 32 MB). The first chunk
    replaces what was there, the rest append. Returns the server's
    `{vector_name, dim, num_vectors}` after checking it holds every row."""
    url = f"{rest_uri}/collections/{collection}/hnsw_training_vectors"
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    for i in range(0, len(vectors), chunk_rows):
        body = {
            "vector_name": vector_name,
            "vectors": vectors[i : i + chunk_rows].tolist(),
            "append": i > 0,
        }
        r = requests.put(url, json=body, headers=_headers(api_key), timeout=600)
        if not r.ok:
            raise RuntimeError(f"training-vector upload failed: {r.status_code} {r.text}")
    r = requests.get(
        url, params={"vector_name": vector_name}, headers=_headers(api_key), timeout=60
    )
    r.raise_for_status()
    info = r.json()["result"]
    if info.get("num_vectors") != len(vectors):
        raise RuntimeError(
            f"server holds {info.get('num_vectors')} training rows, uploaded {len(vectors)}"
        )
    return info


def enable_hnsw_with_projection(
    rest_uri: str,
    collection: str,
    cfg: ProjConfig,
    n_training: int,
    hnsw_m: int,
    indexing_threshold: int,
    api_key: str | None = None,
    vector_name: str = "key",
) -> None:
    """PATCH the collection: HNSW on with `hnsw_m` + the projection block. The same change as
    `_upsert`'s `update_collection`, in raw REST because `HnswConfigDiff` forbids `projection`.
    `n_training` is the number of training vectors just uploaded (see `server_block`)."""
    body = {
        "vectors": {
            vector_name: {
                "hnsw_config": {"m": hnsw_m, "projection": cfg.server_block(n_training)}
            }
        },
        "optimizers_config": {"indexing_threshold": indexing_threshold},
    }
    r = requests.patch(
        f"{rest_uri}/collections/{collection}",
        json=body,
        headers=_headers(api_key),
        timeout=600,
    )
    if not r.ok:
        raise RuntimeError(f"collection PATCH failed: {r.status_code} {r.text}")
