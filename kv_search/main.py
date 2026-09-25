import contextlib
import hashlib
import io
import json
import math
import os
import shutil
import sys
import tempfile
import time

import qdrant_edge as edge
import requests
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    HnswConfigDiff,
    OptimizersConfigDiff,
    VectorParams,
    VectorParamsDiff,
)
from qdrant_client.qdrant_remote import QdrantRemote
from rich.live import Live
from rich.markdown import Markdown
from transformers.cache_utils import CacheLayerMixin, DynamicCache

os.environ.setdefault("HF_HUB_VERBOSITY", "error")

# without this unusable reserved memory goes crazy during prefill
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import importlib.util
from collections.abc import Callable
from enum import Enum, auto
from pathlib import Path
from typing import Any, Literal

import rich
import torch
import transformers.utils.logging
from pydantic import BaseModel, Field
from pydantic_settings import CliApp, CliPositionalArg, CliSubCommand
from rich.console import Console
from rich.progress import track
from rich.table import Table
from safetensors import SafetensorError

# auto_docstring emits [ERROR] lines via print() at class-definition time
with contextlib.redirect_stdout(io.StringIO()):
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoModelForMultimodalLM,
        AutoProcessor,
        BatchEncoding,
        Gemma3ForConditionalGeneration,
        Gemma3Processor,
        Gemma4ForConditionalGeneration,
        Gemma4Processor,
        Mistral3ForConditionalGeneration,
        PixtralProcessor,
        PreTrainedConfig,
        PreTrainedTokenizerBase,
        Qwen2ForCausalLM,
        Qwen2Tokenizer,
        Qwen3_5ForConditionalGeneration,
        Qwen3VLProcessor,
        TextStreamer,
    )
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Attention

from kv_search import proj as projmod
from kv_search._native import tailm_kernel
from kv_search.analysis import CachedData, plots
from kv_search.cache import (
    FullContextRetriever,
    QdrantEdgeNativeRetriever,
    QdrantEdgeRetriever,
    QdrantRetriever,
    RecordingCache,
    RetrievalCache,
    RetrieverConfig,
    SessionRecorder,
    TopKRetriever,
    _git_commit,
    bind_query_aware_cache,
    load_cache,
    save_cache,
)
from kv_search.data import (
    Datasets,
    EvalExample,
    generate_niah_examples,
    generate_qa_examples,
    load_dataset,
)
from kv_search.eval import EvalRow, GenerationResult, score_row
from kv_search.tailm import VERSION as TAILM_VERSION
from kv_search.tailm_runtime import TailmCheck, TailmRuntime
from kv_search.timer import timers

transformers.utils.logging.set_verbosity(transformers.utils.logging.CRITICAL)

IS_MULTIMODAL = {
    "Qwen/Qwen3.5-9B",
    "mistralai/Ministral-3-8B-Instruct-2512",
    "google/gemma-3-4b-it",
    "google/gemma-4-12B-it",
}


console = Console()

ModelType = (
    Qwen3_5ForConditionalGeneration
    | Qwen2ForCausalLM
    | Mistral3ForConditionalGeneration
    | Gemma3ForConditionalGeneration
    | Gemma4ForConditionalGeneration
)

ProcessorType = (
    Qwen3VLProcessor
    | Qwen2Tokenizer
    | PixtralProcessor
    | Gemma3Processor
    | Gemma4Processor
)


ModelName = Literal[
    "Qwen/Qwen3.5-9B",
    "Qwen/Qwen2.5-7B",
    "mistralai/Ministral-3-8B-Instruct-2512",
    "google/gemma-3-4b-it",
    "google/gemma-4-12B-it",
]

_ATTN_IMPL = (
    "flash_attention_2"
    if importlib.util.find_spec("flash_attn") is not None
    else "sdpa"
)


NATIVE_MAX_POSITIONS = 262_144  # Qwen3.5 native context window (no YaRN)


def _size_to_tokens(label: str) -> int:
    """'100k' -> 100_000, '1M' -> 1_000_000."""
    s = label.strip().lower()
    if s.endswith("m"):
        return int(float(s[:-1]) * 1_000_000)
    if s.endswith("k"):
        return int(float(s[:-1]) * 1_000)
    return int(s)


def _yarn_factor(context_tokens: int) -> int:
    """Smallest integer YaRN factor covering the context; 1 (no YaRN) at or below the native
    window. Per-tier so sub-native runs stay undistorted (static YaRN degrades short contexts)."""
    return max(1, math.ceil(context_tokens / NATIVE_MAX_POSITIONS))


def _apply_yarn(config: Any, context_tokens: int) -> int:
    """Flip `config` to YaRN above the native window (returns the factor, 1 = untouched). Every
    RoPE-aware path must go through here so prefill keys, decode queries and shifted prefill
    queries share frequencies."""
    factor = _yarn_factor(context_tokens)
    if factor > 1:
        # nested text_config for multimodal Qwen3.5; fall back to top-level
        text_cfg = getattr(config, "text_config", config)
        # keep rope_theta / mrope_* / partial_rotary_factor, only flip to YaRN
        rope = dict(getattr(text_cfg, "rope_parameters", None) or {})
        rope["rope_type"] = "yarn"
        rope["factor"] = float(factor)
        rope["original_max_position_embeddings"] = NATIVE_MAX_POSITIONS
        text_cfg.rope_parameters = rope
        console.print(
            f"[yellow]YaRN enabled: factor={factor} "
            f"(context ~{context_tokens:,} > native {NATIVE_MAX_POSITIONS:,}); "
            f"rope={rope}[/]"
        )
    return factor


def load_model_config(model_name: str, context_tokens: int = 0) -> Any:
    """The HF config for `model_name` at this tier: YaRN-patched above the native window, untouched
    below. Use this (not a bare `AutoConfig.from_pretrained`) wherever the config feeds anything
    RoPE-aware, so queries can't be shifted with frequencies the prefill didn't use."""
    config = AutoConfig.from_pretrained(model_name)
    _apply_yarn(config, context_tokens)
    return config


@timers.model_load
def _load_model(
    model_name: ModelName, context_tokens: int = 0
) -> tuple[ModelType, ProcessorType]:
    config = load_model_config(model_name, context_tokens)

    processor: ProcessorType = AutoProcessor.from_pretrained(model_name)
    load_kwargs = {
        "config": config,
        "attn_implementation": _ATTN_IMPL,
        "dtype": torch.bfloat16,
        "device_map": "cuda",
    }
    if model_name in IS_MULTIMODAL:
        model: ModelType = AutoModelForMultimodalLM.from_pretrained(
            model_name, **load_kwargs
        ).eval()
    else:
        model: ModelType = AutoModelForCausalLM.from_pretrained(
            model_name, **load_kwargs
        ).eval()

    bind_query_aware_cache(model)
    return model, processor


def _cache_dir(dataset_name: Datasets, qdrant_size: str, model_type: str) -> Path:
    """Namespace artifacts by tier so all qdrant sizes coexist on disk."""
    if dataset_name == Datasets.QDRANT:
        return Path(f"cache/{dataset_name}/{qdrant_size}/{model_type}")
    return Path(f"cache/{dataset_name}/{model_type}")


@timers.prefill_gen
def _do_prefill(
    inputs: BatchEncoding,
    model: ModelType,
    past_key_values: DynamicCache,
    batch_size: int = 4096,
):
    input_chunks = torch.split(inputs["input_ids"], batch_size, -1)
    attention_masks = torch.split(inputs["attention_mask"], batch_size, -1)
    if "mm_token_type_ids" in inputs:
        mm_token_type_chunks = torch.split(inputs["mm_token_type_ids"], batch_size, -1)

    for i, (input_ids, attention_mask) in track(
        enumerate(zip(input_chunks, attention_masks)),
        total=len(input_chunks),
        description="Computing Prefill",
    ):
        input_ids = input_ids.to(model.device)
        attention_mask = attention_mask.to(model.device)

        additional_args = {}
        if "mm_token_type_ids" in inputs:
            additional_args["mm_token_type_ids"] = mm_token_type_chunks[i].to(
                model.device
            )

        with (
            torch.no_grad(),
            torch.nn.attention.sdpa_kernel(
                [
                    torch.nn.attention.SDPBackend.CUDNN_ATTENTION,
                    torch.nn.attention.SDPBackend.FLASH_ATTENTION,
                    torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
                ],
                set_priority=True,
            ),
        ):
            past_key_values = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **additional_args,
                past_key_values=past_key_values,
                logits_to_keep=1,
            ).past_key_values


SHARD_ATTEMPTS = 3  # snapshot + verify rounds per shard before the build gives up


def _shard_name(layer: int, head: int) -> str:
    return f"layer{layer:02d}_head{head}"


def _upsert(
    cache: DynamicCache,
    url: str,
    batch_size: int = 1024,
    api_key: str | None = None,
    edge_root: Path = Path("cache/edge"),
    parallel: int = 4,
    proj: projmod.ProjConfig | None = None,
    model_config: Any = None,
    queries_dir: Path | None = None,
    only: set[str] | None = None,
    segments: int = 1,
    vectors_on_disk: bool = True,
):
    """Upload every (full-attention layer, KV head) into its own collection, let the server
    build the key HNSW graph, and unpack its snapshot into `edge_root/layerLL_headH`.

    With `proj` the graph is built with projection edges (a Qdrant fork, docs/proj-build.md):
    the training queries (from `queries_dir`, shifted with `model_config`'s RoPE) are uploaded
    before HNSW is turned on, and the recipe is written to `edge_root/proj.json`. `only` limits
    the build to these shard names (e.g. {"layer03_head3"}).
    """
    cells = [
        (i, h)
        for i, layer in enumerate(cache.layers)
        if isinstance(layer, CacheLayerMixin)
        for h in range(4)
    ]
    if not cells:
        raise SystemExit("error: the cache has no full-attention layer to upload")
    if only is not None:
        unknown = only - {_shard_name(i, h) for i, h in cells}
        if unknown:
            raise SystemExit(
                f"error: --edge-only: no such shard {', '.join(sorted(unknown))} "
                f"(names are like {_shard_name(*cells[0])})"
            )
        cells = [(i, h) for i, h in cells if _shard_name(i, h) in only]
    if proj is not None:
        assert model_config is not None and queries_dir is not None
        first = cache.layers[cells[0][0]]
        assert first.keys is not None
        context_len = first.keys.shape[2]
        projmod.preflight(
            queries_dir, sorted({i for i, _ in cells}), context_len, first.keys.shape[1]
        )
        try:
            # the training set's size is known now: a too small server bound fails here
            proj.server_block(projmod.GROUP * min(proj.window, context_len))
        except ValueError as e:
            raise SystemExit(f"error: --proj-max-training-vectors: {e}")
        proj_json = edge_root / projmod.PROJ_JSON
        if only is None:
            # a full build: the old recipe no longer describes the folder until the new is written
            proj_json.unlink(missing_ok=True)
        else:
            try:
                built = projmod.load(edge_root)
            except ValueError as e:
                raise SystemExit(f"error: --edge-only: {e}")
            if built is None:
                raise SystemExit(
                    f"error: --edge-only: no {proj_json} (no finished --proj build in "
                    f"{edge_root}); build all shards first"
                )
            if built != proj:
                raise SystemExit(
                    f"error: --edge-only: {edge_root} was built with another recipe "
                    f"({proj_json}); rebuild all shards or pass the same flags"
                )

    edge_root.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(url, api_key=api_key, prefer_grpc=True)
    rest = QdrantClient(url, api_key=api_key)
    assert isinstance(rest._client, QdrantRemote)
    rest_uri = rest._client.rest_uri
    for i, h in track(cells, description="Upserting"):
        layer = cache.layers[i]
        name = f"layer={i};head={h}"
        if client.collection_exists(name):
            client.delete_collection(name)

        assert layer.keys is not None
        assert layer.values is not None
        d = layer.keys.shape[-1]
        n = layer.keys.shape[2]

        client.create_collection(
            collection_name=name,
            vectors_config={
                "key": VectorParams(
                    size=d,
                    distance=Distance.DOT,
                    on_disk=vectors_on_disk,
                    hnsw_config=HnswConfigDiff(m=0, on_disk=vectors_on_disk),
                ),
                "value": VectorParams(
                    size=d,
                    distance=Distance.DOT,
                    on_disk=vectors_on_disk,
                    hnsw_config=HnswConfigDiff(m=0),
                ),
            },
            optimizers_config=OptimizersConfigDiff(
                indexing_threshold=0, default_segment_number=segments
            ),
        )

        # convert on CPU so no f32 temp lands in GPU/unified memory
        client.upload_collection(
            collection_name=name,
            ids=range(n),
            vectors={
                "key": layer.keys[0, h].cpu().float().numpy(),
                "value": layer.values[0, h].cpu().float().numpy(),
            },
            batch_size=batch_size,
            parallel=parallel,
            wait=False,
        )

        if proj is None:
            client.update_collection(
                collection_name=name,
                vectors_config={
                    "key": VectorParamsDiff(
                        hnsw_config=HnswConfigDiff(m=16, on_disk=vectors_on_disk)
                    )
                },
                # threshold below the single segment's size so the graph builds
                optimizers_config=OptimizersConfigDiff(
                    indexing_threshold=1000, default_segment_number=segments
                ),
            )
        else:
            assert queries_dir is not None
            train = projmod.training_queries(queries_dir, model_config, i, h, n, proj)
            projmod.upload_training_vectors(rest_uri, name, train, api_key=api_key)
            projmod.enable_hnsw_with_projection(
                rest_uri,
                name,
                proj,
                n_training=len(train),
                hnsw_m=16,
                indexing_threshold=20000,
                api_key=api_key,
            )

        shard_dir = edge_root / _shard_name(i, h)
        for attempt in range(1, SHARD_ATTEMPTS + 1):
            _wait_indexed(rest_uri, name, api_key, n_points=n)
            _snapshot_shard(rest_uri, name, api_key, shard_dir)
            try:
                _verify_shard(shard_dir, proj=proj is not None)
                break
            except RuntimeError as e:
                if attempt == SHARD_ATTEMPTS:
                    raise
                console.print(
                    f"[yellow]{name}: {e}; snapshot again ({attempt}/{SHARD_ATTEMPTS})[/]"
                )

        client.delete_collection(name)

    if proj is not None:
        projmod.save(edge_root, proj)


def _snapshot_shard(
    rest_uri: str, name: str, api_key: str | None, shard_dir: Path
) -> None:
    """Download the collection's shard snapshot and unpack it into `shard_dir` (replaced)."""
    with tempfile.TemporaryDirectory(dir=shard_dir.parent) as restore_dir:
        snapshot_path = Path(restore_dir) / "shard.snapshot"

        with requests.get(
            f"{rest_uri}/collections/{name}/shards/0/snapshot",
            headers={"api-key": api_key} if api_key else None,
            stream=True,
        ) as r:
            r.raise_for_status()
            with open(snapshot_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)

        if shard_dir.exists():
            shutil.rmtree(shard_dir)
        shard_dir.mkdir(parents=True, exist_ok=True)

        edge.EdgeShard.unpack_snapshot(str(snapshot_path), str(shard_dir))


def _wait_indexed(
    rest_uri: str,
    name: str,
    api_key: str | None,
    n_points: int,
    timeout: float = 3600.0,
    poll: float = 0.5,
    stable: int = 3,
) -> None:
    """Wait until the server has built the HNSW graph after the PATCH.

    `status == green` alone is not enough: right after the PATCH the collection is still green
    with nothing indexed, because the optimizer has not been scheduled yet. Done means: every
    point arrived (the upload runs with wait=False), the update queue is empty, the optimizer is
    idle without error, something is indexed, and (indexed vectors, segments) stayed the same
    over `stable` polls in a row. `_verify_shard` then checks the files themselves.
    """
    deadline = time.monotonic() + timeout
    seen: list[tuple[int, int]] = []
    while True:
        r = requests.get(
            f"{rest_uri}/collections/{name}",
            headers={"api-key": api_key} if api_key else None,
            timeout=60,
        )
        r.raise_for_status()
        info = r.json()["result"]
        opt = info.get("optimizer_status")
        if isinstance(opt, dict) and "error" in opt:
            raise RuntimeError(f"{name}: optimizer error: {opt['error']}")
        indexed = info.get("indexed_vectors_count") or 0
        ready = (
            info.get("status") == "green"
            and opt == "ok"
            and info.get("points_count") == n_points
            and (info.get("update_queue") or {}).get("length", 0) == 0
            and indexed > 0
        )
        seen = (seen + [(indexed, info.get("segments_count"))])[-stable:] if ready else []
        if len(seen) == stable and len(set(seen)) == 1:
            return
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"{name}: HNSW not built after {timeout:.0f} s (last info: status "
                f"{info.get('status')}, optimizer {opt}, {indexed} indexed, "
                f"{info.get('points_count')}/{n_points} points)"
            )
        time.sleep(poll)


def _verify_shard(shard_dir: Path, proj: bool) -> None:
    """Check an unpacked shard really holds the key graph.

    Every segment whose `segment.json` claims an HNSW key index needs a `vector_index-key/` with
    its `hnsw_config.json` and non-empty link files; at least one segment must have one (small
    appendable segments stay plain and are scanned in full). With `proj`, every key graph must
    carry the fork's `projection` block. Raises RuntimeError naming the first problem.
    """
    segments = shard_dir / "segments"
    if not segments.is_dir():
        raise RuntimeError(f"{shard_dir}: no segments/ folder")
    graphs = 0
    for seg in sorted(segments.iterdir()):
        config = json.loads((seg / "segment.json").read_text())["config"]
        if config["vector_data"]["key"]["index"]["type"] != "hnsw":
            continue
        index = seg / "vector_index-key"
        hnsw_config = index / "hnsw_config.json"
        if not hnsw_config.exists():
            raise RuntimeError(
                f"{seg.name}: segment.json claims an HNSW key index, but {index.name}/ "
                "holds no graph"
            )
        links = list(index.glob("links*.bin"))
        if not links or any(f.stat().st_size == 0 for f in links):
            raise RuntimeError(f"{seg.name}: {index.name}/ has empty or missing links")
        if proj and "projection" not in json.loads(hnsw_config.read_text()):
            raise RuntimeError(
                f"{seg.name}: key graph built without the projection block (plain Qdrant "
                "instead of the fork?)"
            )
        graphs += 1
    if graphs == 0:
        raise RuntimeError(f"{shard_dir}: no HNSW key graph in any segment")


def _print_stats():
    rich.print(timers)
    rich.print(f"peak allocated: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    rich.print(f"peak reserved:  {torch.cuda.max_memory_reserved() / 2**30:.2f} GiB")


class TimedStreamer(TextStreamer):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        skip_prompt: bool = False,
        **decode_kwargs: Any,
    ):
        super().__init__(tokenizer, skip_prompt, **decode_kwargs)
        self._buffer = ""
        self._live: Live | None = None
        self._render_live = True

    def put(self, value):
        if self.next_tokens_are_prompt:
            super().put(value)
            return
        timers.token_gen.record()
        super().put(value)

    def on_finalized_text(self, text, stream_end=False):
        if not self._render_live:
            super().on_finalized_text(text, stream_end)
            return

        if self._live is None:
            self._buffer = ""
            self._live = Live(
                console=console, auto_refresh=False, vertical_overflow="visible"
            )
            self._live.start()
        self._buffer += text
        self._live.update(Markdown(self._buffer), refresh=True)

    def end(self):
        timers.token_gen.reset_lap()
        super().end()
        if self._live is not None:
            self._live.update(Markdown(self._buffer), refresh=True)
            self._live.stop()

    def reset(self, render_live: bool = True):
        if self._live is not None:
            self._live.stop()
            self._live = None
        self._render_live = render_live


class CacheImpl(Enum):
    TORCH = auto()
    QDRANT = auto()


PROJ_RETRIEVERS = ("edge", "native")  # the retrievers that read the edge shard folders


class ProjDir(BaseModel):
    """Which edge shard folder a command uses: `edge/` (plain HNSW) or the projection-edge
    folder of a recipe (docs/proj-build.md)."""

    # key graph with query-aware projection edges (built through the Qdrant fork)
    proj: bool = False
    # plain | barred (barred: the attention sinks get no projected edge, plus a repair pass)
    proj_recipe: projmod.Recipe = "plain"

    def edge_folder(self, cache_dir: Path) -> Path:
        if not self.proj:
            return cache_dir / "edge"
        return cache_dir / ("edge_proj" if self.proj_recipe == "plain" else "edge_proj_barred")

    def proj_preflight(self, retriever_type: str) -> None:
        """Flag checks that need no model or cache."""
        if self.proj and retriever_type not in PROJ_RETRIEVERS:
            raise SystemExit(
                f"error: --proj needs -r edge or -r native (got -r {retriever_type})"
            )

    def proj_load(self, cache_dir: Path) -> Path:
        """The shard folder chat loads. With --proj: check it was built with projection edges of
        a recipe the decode side supports, and print what it holds."""
        folder = self.edge_folder(cache_dir)
        if not self.proj:
            return folder
        try:
            cfg = projmod.load(folder)
        except ValueError as e:
            raise SystemExit(f"error: --proj: {e}")
        if cfg is None:
            raise SystemExit(
                f"error: --proj: no {projmod.PROJ_JSON} in {folder}; build the shards with "
                f"`kv-search prefill --skip-prefill --proj --proj-recipe {self.proj_recipe}`"
            )
        if cfg.recipe == "barred":
            raise SystemExit(
                f"error: --proj: {folder} holds barred shards; the decode side does not score "
                f"the excluded points {list(cfg.excluded_points)} yet, and without that a barred "
                "graph is worse than a plain one"
            )
        _print_plain(
            [
                f"proj: {folder} ({cfg.recipe}: m={cfg.m}, topn={cfg.topn}, "
                f"window={cfg.window}, seed={cfg.seed})"
            ]
        )
        return folder


class ProjFlags(ProjDir):
    """`prefill` build knobs of the projection pass (ProjConfig; defaults = plain recipe)."""

    proj_m: int = 16
    proj_topn: int = 100
    proj_maxq: int = 64
    proj_cands: int = 200
    proj_seed: int = 0
    proj_window: int = 8192
    proj_shift_span: int = 256
    # server-side bound on the training vectors; unset = every uploaded vector
    proj_max_training_vectors: int | None = None
    # barred only: positions that never receive a projected edge
    proj_exclude: list[int] = [0, 1, 2]
    # barred only: per-point cap of the repair pass
    proj_repair_cap: int = 32

    def proj_config(self) -> projmod.ProjConfig | None:
        if not self.proj:
            return None
        return projmod.ProjConfig.from_recipe(
            self.proj_recipe,
            exclude=tuple(self.proj_exclude),
            m=self.proj_m,
            topn=self.proj_topn,
            maxq=self.proj_maxq,
            cands=self.proj_cands,
            seed=self.proj_seed,
            window=self.proj_window,
            shift_span=self.proj_shift_span,
            max_training_vectors=self.proj_max_training_vectors,
            repair_max_per_point=self.proj_repair_cap,
        )


class CmdPrefill(ProjFlags):
    model_name: ModelName = "Qwen/Qwen3.5-9B"
    dataset_name: Datasets = Datasets.QDRANT
    qdrant_size: str = "100k"
    upsert: bool = False
    url: str = "localhost"
    api_key: str | None = None
    upsert_batch_size: int = 1024
    prefill_batch_size: int = 4096
    # build the edge shards from the saved prefill cache, without the model (implies --upsert)
    skip_prefill: bool = False
    # rebuild only these shards, e.g. layer03_head3 (needs --upsert or --skip-prefill)
    edge_only: list[str] = []

    def preflight(self) -> None:
        """Flag checks that need no model or cache."""
        building = self.upsert or self.skip_prefill
        if self.proj and not building:
            raise SystemExit("error: --proj needs --upsert or --skip-prefill")
        if self.edge_only and not building:
            raise SystemExit("error: --edge-only needs --upsert or --skip-prefill")
        if self.proj and self.proj_recipe == "barred":
            console.print(
                "[yellow]warning: barred shards need the decode side to score the excluded "
                "points, which chat does not do yet: chat --proj refuses them[/]"
            )

    def cli_cmd(self) -> None:
        self.preflight()
        context_tokens = (
            _size_to_tokens(self.qdrant_size)
            if self.dataset_name == Datasets.QDRANT
            else 0
        )
        if self.skip_prefill:
            config = load_model_config(self.model_name, context_tokens)
            cache_dir = _cache_dir(self.dataset_name, self.qdrant_size, config.model_type)
            cache, context_len = load_cache(cache_dir, config, device="cpu")
            console.print(f"loaded {cache_dir}: context_len {context_len:,}")
        else:
            model, processor = _load_model(self.model_name, context_tokens)
            config = model.config

            cache_dir = _cache_dir(
                self.dataset_name, self.qdrant_size, model.config.model_type
            )
            cache_dir.mkdir(exist_ok=True, parents=True)

            cache = RecordingCache(path=cache_dir, config=model.config)
            messages = load_dataset(
                self.dataset_name,
                multimodal=self.model_name in IS_MULTIMODAL,
                qdrant_size=self.qdrant_size,
            )

            inputs: BatchEncoding = processor.apply_chat_template(
                messages.prefill,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )  # ty:ignore[invalid-assignment]
            _do_prefill(inputs, model, cache, self.prefill_batch_size)

            save_cache(
                cache,
                cache_dir,
                cache.get_seq_length(),
            )
            cache.finalize()
            del model, processor
            torch.cuda.empty_cache()

        if self.upsert or self.skip_prefill:
            _upsert(
                cache,
                self.url,
                self.upsert_batch_size,
                api_key=self.api_key,
                edge_root=self.edge_folder(cache_dir),
                proj=self.proj_config(),
                model_config=config,
                queries_dir=cache_dir,
                only=set(self.edge_only) or None,
            )

        if not self.skip_prefill:
            _print_stats()


_RETRIEVERS: dict[str, type[RetrieverConfig]] = {
    "topk": TopKRetriever,
    "full": FullContextRetriever,
    "qdrant": QdrantRetriever,
    "edge": QdrantEdgeRetriever,
    "native": QdrantEdgeNativeRetriever,
}


TAILM_MODEL = "Qwen/Qwen3.5-9B"  # only _qwen_3_5_forward calls a retriever
TAILM_RETRIEVERS = ("topk", "native")  # the retrievers that report the tail boundary


def _full_attention_cells(text_config: Any) -> list[tuple[int, int]]:
    """Every (full-attention layer, KV head) of the model: the cells a tailM file can belong to."""
    return [
        (i, h)
        for i, t in enumerate(text_config.layer_types)
        if t == "full_attention"
        for h in range(text_config.num_key_value_heads)
    ]


def _print_plain(lines: list[str]) -> None:
    """Print without rich markup, so bracketed text in gate reasons stays literal."""
    for line in lines:
        console.print(line, markup=False, highlight=False)


class TailmFlags(BaseModel):
    """Opt-in tail correction (tailM) for retrieval attention (docs/tailm-build.md). A
    subcommand that builds a RetrievalCache gets it by inheriting these flags and calling the
    methods below."""

    # add back the dropped prefill keys' softmax mass on the heads scripts/build_tailm.py gated on
    tailm: bool = False
    # the build's --out; default <cache_dir>/tailm
    tailm_dir: Path | None = None
    # after each answer, print today's vs tailM's attention error against exact attention
    tailm_check: bool = False

    def tailm_preflight(self, model_name: str, retriever_type: str) -> None:
        """Flag checks that need no model or cache."""
        if self.tailm_check and not self.tailm:
            raise SystemExit("error: --tailm-check needs --tailm")
        if not self.tailm:
            return
        if model_name != TAILM_MODEL:
            raise SystemExit(
                f"error: --tailm supports only {TAILM_MODEL} (got {model_name})"
            )
        if retriever_type not in TAILM_RETRIEVERS:
            raise SystemExit(
                f"error: --tailm needs -r topk or -r native (got -r {retriever_type})"
            )

    def tailm_folder(self, cache_dir: Path) -> Path:
        return self.tailm_dir if self.tailm_dir is not None else cache_dir / "tailm"

    def tailm_load(
        self,
        model_name: str,
        text_config: Any,
        scaling: float,
        device: Any,
        cache_dir: Path,
        context_len: int,
        n_retrieved: int,
        retriever_type: str | None = None,
    ) -> tuple[TailmRuntime | None, TailmCheck | None]:
        """Load the tailM files, print the startup message; on -r native
        it also names the Rust decode tail's kernel."""
        if not self.tailm:
            return None, None
        try:
            runtime = TailmRuntime.load(
                self.tailm_folder(cache_dir),
                model=model_name,
                context_len=context_len,
                n_retrieved=n_retrieved,
                head_dim=text_config.head_dim,
                scaling=scaling,
                cells=_full_attention_cells(text_config),
                group=text_config.num_attention_heads
                // text_config.num_key_value_heads,
                device=device,
            )
        except ValueError as e:
            msg = str(e)
            if (
                "tailM file version" in msg
            ):  # tailm.load_head: a file of another format version
                msg += f"; rebuild the files with scripts/build_tailm.py (current version {TAILM_VERSION})"
            raise SystemExit(f"error: --tailm: {msg}")
        except KeyError as e:
            raise SystemExit(
                f"error: --tailm: a head file lacks metadata key {e}; rebuild it with scripts/build_tailm.py"
            )
        except SafetensorError as e:
            raise SystemExit(
                f"error: --tailm: unreadable head file in {self.tailm_folder(cache_dir)}: {e}"
            )
        lines = runtime.describe()
        if retriever_type == "native":
            lines.append(
                f"  native: decode tail in Rust (bf16 weights, {tailm_kernel()})"
            )
        _print_plain(lines)
        return runtime, TailmCheck(runtime) if self.tailm_check else None

    @staticmethod
    def tailm_attach(retriever: Any, runtime: TailmRuntime | None) -> None:
        """-r native with --tailm: register the tail with the Rust engine once at startup, so a bad
        shard folder or tail state stops `chat` here, not mid-answer. `attend` still attaches per
        call (idempotent) for a /native switch in the REPL."""
        if runtime is None or not hasattr(retriever, "attach_tailm"):
            return
        try:
            retriever.attach_tailm(runtime)
        except Exception as e:
            raise SystemExit(f"error: --tailm: native decode tail: {e}")


class CmdChat(TailmFlags, ProjDir):
    model_name: ModelName = "Qwen/Qwen3.5-9B"
    dataset_name: Datasets = Datasets.QDRANT
    qdrant_size: str = "100k"
    retriever: RetrieverConfig = Field(default_factory=QdrantRetriever)
    max_new_tokens: int = 256
    render_live: bool = True
    record_indices: bool = False
    record_prompts: bool = False

    def cli_cmd(self) -> None:
        self.tailm_preflight(self.model_name, self.retriever.type)
        self.proj_preflight(self.retriever.type)
        context_tokens = (
            _size_to_tokens(self.qdrant_size)
            if self.dataset_name == Datasets.QDRANT
            else 0
        )
        # check the shard folder first: a --proj mistake stops chat before the model load
        if self.proj:
            model_type = AutoConfig.from_pretrained(self.model_name).model_type
            self.proj_load(_cache_dir(self.dataset_name, self.qdrant_size, model_type))
        model, processor = _load_model(self.model_name, context_tokens)

        cache_dir = _cache_dir(
            self.dataset_name, self.qdrant_size, model.config.model_type
        )
        cache_dir.mkdir(exist_ok=True, parents=True)

        prefill, context_len = load_cache(cache_dir, model.config)

        # point edge/native retrievers at this tier's shard folder (edge/, or --proj's)
        if hasattr(self.retriever, "edge_root"):
            self.retriever.edge_root = str(self.edge_folder(cache_dir))
        if self.proj and getattr(self.retriever, "exact", False):
            console.print(
                "[yellow]warning: --proj with exact search: the projection edges change nothing "
                "(add --no-retriever.exact)[/]"
            )

        if self.record_indices and isinstance(self.retriever, TopKRetriever):
            self.retriever.record_indices = self.record_indices

        tailm, check = None, None
        if self.tailm:
            # tailm_preflight admitted only Qwen3.5-9B with -r topk / -r native
            assert isinstance(
                self.retriever, (TopKRetriever, QdrantEdgeNativeRetriever)
            )
            attn = next(m for m in model.modules() if isinstance(m, Qwen3_5Attention))
            tailm, check = self.tailm_load(
                self.model_name,
                getattr(model.config, "text_config", model.config),
                attn.scaling,
                model.device,
                cache_dir,
                context_len,
                self.retriever.n_retrieved,
                retriever_type=self.retriever.type,
            )
            self.tailm_attach(self.retriever, tailm)

        recorder: SessionRecorder | None = None
        if self.record_prompts:
            recorder = SessionRecorder(
                cache_dir / "replay",
                meta={
                    "model": self.model_name,
                    "dataset": str(self.dataset_name),
                    "qdrant_size": self.qdrant_size,
                    "context_len": str(context_len),
                    "attn_implementation": model.config._attn_implementation,
                    "kv_search_commit": _git_commit(),
                },
            )
            if not isinstance(
                self.retriever, (FullContextRetriever, QdrantEdgeNativeRetriever)
            ):
                console.print(
                    "[yellow]record_prompts only records under `-r full` or `-r native`; "
                    "switch with /full or /native in the repl[/]"
                )

        cache = RetrievalCache(
            retriever=self.retriever,
            prefill=prefill,
            config=model.config,
            tailm=tailm,
            check=check,
            recorder=recorder,
        )

        streamer = TimedStreamer(processor.tokenizer, skip_prompt=True)

        try:
            self._repl(model, processor, cache, context_len, streamer, cache_dir)
        finally:
            _print_stats()

    def _repl(
        self,
        model: ModelType,
        processor: ProcessorType,
        cache: RetrievalCache,
        context_len: int,
        streamer: TimedStreamer,
        cache_dir: Path,
    ) -> None:

        instances = {cache.retriever.type: cache.retriever}
        n_retrieved = getattr(cache.retriever, "n_retrieved", 128)
        # the start retriever's search mode, for edge/native retrievers created by a switch
        search = {
            k: getattr(cache.retriever, k)
            for k in ("exact", "hnsw_ef")
            if hasattr(cache.retriever, k)
        }

        prompt_idx = 0

        while True:
            try:
                streamer.reset(render_live=self.render_live)

                print()
                console.rule(
                    f"user \\[retriever = {cache.retriever.type}]", style="bright_cyan"
                )
                user = console.input("[bold bright_cyan]> [/]").strip()
                if not sys.stdin.isatty():
                    console.print(user)
                console.rule(style="bright_cyan")
                print()
            except EOFError, KeyboardInterrupt:
                print()
                return

            if not user:
                continue

            if user.startswith("/"):
                cmd = user[1:].strip()
                if cmd in ("help", "?"):
                    print(
                        "commands: "
                        + ", ".join(f"/{t}" for t in _RETRIEVERS)
                        + ", /help"
                    )
                elif cmd == "live":
                    self.render_live = not self.render_live
                    print(f"[render_live = {self.render_live}]")
                elif cmd in _RETRIEVERS:
                    if cmd not in instances:
                        r = _RETRIEVERS[cmd]()
                        if not isinstance(r, FullContextRetriever):
                            r.n_retrieved = n_retrieved
                        for k, v in search.items():
                            if hasattr(r, k):
                                setattr(r, k, v)
                        if hasattr(r, "record_indices"):
                            r.record_indices = self.record_indices
                        if hasattr(r, "edge_root"):
                            r.edge_root = str(self.edge_folder(cache_dir))
                        instances[cmd] = r
                    cache.retriever = instances[cmd]
                    print(f"[retriever = {cmd}]")
                else:
                    print(f"unknown command '/{cmd}', see /help")
                continue

            record = self.record_indices and isinstance(cache.retriever, TopKRetriever)
            if record:
                cache.retriever.reset_indices()

            record_prompt = cache.recorder is not None and isinstance(
                cache.retriever, (FullContextRetriever, QdrantEdgeNativeRetriever)
            )
            if record_prompt:
                cache.recorder.reset()

            # isolate this prompt's timings; discard the first (cold) prompt when measuring
            timers.reset_generation()
            torch.cuda.reset_peak_memory_stats()
            try:
                prompt_len, out, logits = self._generate(
                    model, processor, cache, context_len, streamer, user
                )
                if record:
                    # save per prompt, store prompt length with it
                    tmp = cache_dir / f"indices{prompt_idx:02d}"
                    tmp.mkdir(exist_ok=True)
                    (tmp / "meta.json").write_text(
                        json.dumps({"prompt_len": prompt_len})
                    )
                    cache.retriever.save_indices(tmp)
                if record_prompt:
                    n_rows = cache.recorder.n_rows
                    token_ids = out[0, :n_rows]
                    positions = context_len + torch.arange(n_rows)
                    answer = processor.tokenizer.decode(
                        out[0, prompt_len:], skip_special_tokens=True
                    )
                    path = cache.recorder.save(
                        cache,
                        prompt_idx,
                        token_ids,
                        positions,
                        prompt_len,
                        user,
                        answer,
                        logits=logits,
                    )
                    console.print(
                        f"[green]recorded {n_rows} positions"
                        f"{f', logits {tuple(logits.shape)}' if logits is not None else ''}"
                        f" -> {path}[/]"
                    )
                prompt_idx += 1
            except KeyboardInterrupt:
                streamer.end()
            finally:
                cache.reset()

            _print_stats()
            if cache.check is not None:
                _print_plain(cache.check.report())
                cache.check.reset()

    def _generate(
        self,
        model: ModelType,
        processor: ProcessorType,
        cache: RetrievalCache,
        context_len: int,
        streamer: TimedStreamer,
        user: str,
    ) -> tuple[int, torch.Tensor, torch.Tensor | None]:
        inputs: BatchEncoding[torch.Tensor] = processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "text", "text": user}]}],  # ty:ignore[invalid-argument-type]
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        )  # ty:ignore[invalid-assignment]
        inputs = inputs.to(model.device)

        # offset positional embeddings to beyond prefill content
        prompt_len = inputs["input_ids"].shape[1]
        inputs["position_ids"] = torch.arange(
            context_len, context_len + prompt_len, device=model.device
        ).unsqueeze(0)

        gen_kwargs: dict[str, Any] = {}
        if self.record_prompts:
            # collect per-step next-token logits for the recording
            gen_kwargs["output_logits"] = True
            gen_kwargs["return_dict_in_generate"] = True

        out = model.generate(  # ty:ignore[invalid-argument-type]
            **inputs,  # ty:ignore[invalid-argument-type]
            max_new_tokens=self.max_new_tokens,
            past_key_values=cache,
            use_cache=True,
            streamer=streamer,
            do_sample=False,
            **gen_kwargs,
        )

        if self.record_prompts:
            # out.logits: tuple (len = num_generated) of [1, vocab] -> [num_generated, vocab]
            logits = torch.stack(out.logits, dim=0)[:, 0, :]  # ty:ignore[unresolved-attribute, invalid-argument-type]
            return prompt_len, out.sequences, logits  # ty:ignore[unresolved-attribute]
        return prompt_len, out, None  # ty:ignore[invalid-return-type]


class CmdFigures(BaseModel):
    """Exploratory figures/tables (index heatmap, scores, cross-layer, mse) into the cache dir."""

    model_name: ModelName = "Qwen/Qwen3.5-9B"
    dataset_name: Datasets = Datasets.QDRANT
    qdrant_size: str = "100k"

    def cli_cmd(self) -> None:
        config: PreTrainedConfig = AutoConfig.from_pretrained(self.model_name)
        cache_dir = _cache_dir(self.dataset_name, self.qdrant_size, config.model_type)
        cache_dir.mkdir(exist_ok=True, parents=True)
        data = CachedData(cache_dir, model_name=self.model_name)
        plots.analyze(data)


class CmdAnalyze(BaseModel):
    """Run registered analyses over sizes and write envelopes to cache/analysis/.
    names and sizes are comma-separated (e.g. `analyze sweep,reuse -s 100k,1M`)."""

    names: CliPositionalArg[str]
    sizes: str = "100k,200k,1M"
    n_prompts: int = 5
    model_name: ModelName = "Qwen/Qwen3.5-9B"
    dataset_name: Datasets = Datasets.QDRANT

    def cli_cmd(self) -> None:
        from kv_search.analysis import io, registry

        config: PreTrainedConfig = AutoConfig.from_pretrained(self.model_name)
        for name in self.names.split(","):
            a = registry.ANALYSES[name]
            sizes_out: dict = {}
            for size in self.sizes.split(","):
                cache_dir = _cache_dir(self.dataset_name, size, config.model_type)
                if a.per_prompt and len(sorted(cache_dir.glob("indices0*/"))) < self.n_prompts:
                    console.print(f"[yellow]skip {name}/{size}: <{self.n_prompts} prompts[/]")
                    continue
                data = CachedData(
                    cache_dir, model_name=self.model_name, load_prefill=a.needs_prefill
                )
                sizes_out[size] = a.run(data, self.n_prompts)
                console.print(f"[green]{name}/{size} done[/]")
                del data
                torch.cuda.empty_cache()
            path = io.write_envelope(
                name, self.model_name, {"n_prompts": self.n_prompts}, sizes_out
            )
            console.print(f"[green]wrote {path}[/]")


def _encode_eval(
    processor: ProcessorType, ex: EvalExample
) -> tuple[BatchEncoding, int]:
    # render once and split at the final user turn, so prefill is a true prefix
    # of the decode input (rendering prefill separately drifts on Qwen's per-turn
    # think handling)
    enc: BatchEncoding = processor.apply_chat_template(
        ex.prefill + ex.query,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
    )  # ty:ignore[invalid-assignment]
    ids = enc["input_ids"][0].tolist()
    im_start = processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
    user_tok = processor.tokenizer.encode("user", add_special_tokens=False)[0]
    starts = [
        i for i in range(len(ids) - 1) if ids[i] == im_start and ids[i + 1] == user_tok
    ]
    if not starts:
        raise RuntimeError("could not locate final user turn in render")
    return enc, starts[-1]


def _slice_inputs(enc: BatchEncoding, end: int) -> BatchEncoding:
    out = {
        "input_ids": enc["input_ids"][:, :end],
        "attention_mask": enc["attention_mask"][:, :end],
    }
    if "mm_token_type_ids" in enc:
        out["mm_token_type_ids"] = enc["mm_token_type_ids"][:, :end]
    return BatchEncoding(out)


@torch.no_grad()
def _eval_generate(
    model: ModelType,
    processor: ProcessorType,
    cache: RetrievalCache,
    context_len: int,
    enc_full: BatchEncoding,
    n: int,
    max_new_tokens: int,
) -> GenerationResult:
    query_ids = enc_full["input_ids"][:, n:].to(model.device)
    q_len = query_ids.shape[1]
    gen_kwargs: dict[str, Any] = {
        "input_ids": query_ids,
        "attention_mask": torch.ones((1, q_len), device=model.device),
        "position_ids": torch.arange(
            context_len, context_len + q_len, device=model.device
        ).unsqueeze(0),
    }
    if "mm_token_type_ids" in enc_full:
        gen_kwargs["mm_token_type_ids"] = enc_full["mm_token_type_ids"][:, n:].to(
            model.device
        )

    # warm-up (untimed): absorbs one-time costs - edge shard load/populate, cuda
    # kernel init, page cache - so the timed run is steady-state and full vs hnsw
    # compare fairly (full's KV is already resident from prefill).
    model.generate(  # ty:ignore[invalid-argument-type]
        **gen_kwargs,
        max_new_tokens=min(max_new_tokens, 2),
        past_key_values=cache,
        use_cache=True,
        do_sample=False,
    )
    cache.reset()

    # qdrant_retrieve is accumulated by the retriever; reset so we read this gen
    timers.qdrant_retrieve.reset()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = model.generate(  # ty:ignore[invalid-argument-type]
        **gen_kwargs,
        max_new_tokens=max_new_tokens,
        past_key_values=cache,
        use_cache=True,
        do_sample=False,
    )
    torch.cuda.synchronize()
    seconds = time.perf_counter() - t0

    new = out[0, q_len:]
    text = processor.tokenizer.decode(new, skip_special_tokens=True)
    return GenerationResult(
        text=text,
        tokens=int(new.shape[0]),
        seconds=seconds,
        retrieval_seconds=timers.qdrant_retrieve.total,
    )


# prefill KV is keyed by context only (no segments), so it survives a segment
# sweep; edge shards depend on segments and get their own cache
_PREFILL_CACHE = Path("cache/eval_prefill")
_SHARD_CACHE = Path("cache/eval_shards")


def _cache_key(*parts: object) -> str:
    return hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:16]


def _evict(root: Path, keep: int) -> None:
    """Drop all but the `keep` most-recently-used entries under root."""
    dirs = sorted(
        (p for p in root.iterdir() if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
    )
    for d in dirs[: max(0, len(dirs) - keep)]:
        shutil.rmtree(d, ignore_errors=True)


def _cached_dir(
    root: Path,
    key: str,
    keep: int,
    build: Callable[[Path], None],
    rebuild: bool = False,
) -> Path:
    """Return root/key, running build(tmp) into a temp dir on a miss and
    publishing it atomically. Keeps the `keep` most-recent entries."""
    d = root / key
    if rebuild:
        shutil.rmtree(d, ignore_errors=True)
    if d.exists():
        os.utime(d, None)  # mark recently used
        return d
    root.mkdir(parents=True, exist_ok=True)
    _evict(root, keep - 1)
    tmp = Path(tempfile.mkdtemp(dir=root))
    build(tmp)
    os.replace(tmp, d)
    return d


class CmdEval(BaseModel):
    model_name: ModelName = "Qwen/Qwen3.5-9B"
    task: Literal["niah", "qa"] = "qa"
    buckets: str = "71680"
    n_retrieved: int = 128
    hnsw_ef: int | None = None
    hnsw_segments: int = 1
    max_new_tokens: int = 128
    # task knobs (n_keys only used by niah)
    n_keys: int = 1
    depth: float = 0.5
    seed: int = 0
    n_examples: int = 5
    # which retrievers to score; exact/hnsw run on the same per-example edge shards
    full: bool = True
    exact: bool = True
    hnsw: bool = True
    topk: bool = False
    url: str = "localhost"
    api_key: str | None = None
    upsert_batch_size: int = 1024
    # build-time memory tier: True = on disk (Cold, paged on demand), False = in
    # RAM / Cached (edge pre-loads into page cache on open). Persisted in the shard.
    vectors_on_disk: bool = True
    # cache prefills + shards across runs; keep the N most-recent per cache.
    # rebuild_shards forces re-upsert (edge build constants aren't in the key);
    # prefill is fully keyed, so clear cache/eval_prefill by hand if code changes
    cache_max: int = 6
    rebuild_shards: bool = False
    prefill_batch_size: int = 4096
    out: str = ""

    def cli_cmd(self) -> None:
        buckets = sorted({int(b) for b in self.buckets.split(",") if b.strip()})
        multimodal = self.model_name in IS_MULTIMODAL

        rows: list[EvalRow] = []
        loaded_factor: int | None = None
        model: ModelType | None = None
        processor: ProcessorType | None = None

        for bucket in buckets:
            factor = _yarn_factor(bucket)
            if factor != loaded_factor:
                if model is not None:
                    del model, processor
                    torch.cuda.empty_cache()
                model, processor = _load_model(self.model_name, bucket)
                loaded_factor = factor
            assert model is not None and processor is not None

            if self.task == "niah":
                examples = generate_niah_examples(
                    processor.tokenizer,
                    bucket,
                    self.n_keys,
                    self.depth,
                    self.n_examples,
                    self.seed,
                    multimodal,
                )
            else:
                examples = generate_qa_examples(
                    processor.tokenizer,
                    bucket,
                    self.depth,
                    self.n_examples,
                    self.seed,
                    multimodal,
                )
            for ex in examples:
                rows.extend(self._eval_example(model, processor, ex))

        self._report(rows)

    def _cached_prefill(
        self, model: ModelType, enc: Any, n: int, ex: EvalExample
    ) -> DynamicCache:
        """Prefill the context KV, cached to disk and reused across runs."""
        key = _cache_key(
            self.model_name,
            self.task,
            ex.bucket,
            self.seed,
            ex.idx,
            self.n_keys,
            self.depth,
        )

        def build(tmp: Path) -> None:
            prefill = DynamicCache(config=model.config)
            _do_prefill(_slice_inputs(enc, n), model, prefill, self.prefill_batch_size)
            save_cache(prefill, tmp, n)

        pdir = _cached_dir(_PREFILL_CACHE, key, self.cache_max, build)
        return load_cache(pdir, model.config)[0]

    def _eval_example(
        self, model: ModelType, processor: ProcessorType, ex: EvalExample
    ) -> list[EvalRow]:
        enc, n = _encode_eval(processor, ex)
        prefill = self._cached_prefill(model, enc, n, ex)

        edge_root: Path | None = None
        if self.exact or self.hnsw:
            key = _cache_key(
                self.model_name,
                self.task,
                ex.bucket,
                self.seed,
                ex.idx,
                self.n_keys,
                self.depth,
                self.hnsw_segments,
                self.vectors_on_disk,
            )

            def build(tmp: Path, prefill: DynamicCache = prefill) -> None:
                _upsert(
                    prefill,
                    self.url,
                    self.upsert_batch_size,
                    api_key=self.api_key,
                    edge_root=tmp,
                    segments=self.hnsw_segments,
                    vectors_on_disk=self.vectors_on_disk,
                )

            edge_root = _cached_dir(
                _SHARD_CACHE, key, self.cache_max, build, self.rebuild_shards
            )

        gens: dict[str, GenerationResult] = {}
        try:
            cache = RetrievalCache(
                retriever=FullContextRetriever(), prefill=prefill, config=model.config
            )

            def run(name: str, retriever: Any) -> None:
                cache.retriever = retriever
                gens[name] = _eval_generate(
                    model, processor, cache, n, enc, n, self.max_new_tokens
                )
                cache.reset()

            if self.topk:
                run("topk", TopKRetriever(n_retrieved=self.n_retrieved))
            if self.full:
                run("full", FullContextRetriever())
            if edge_root is not None:
                # full/topk are done; exact/hnsw read the shards, not the prefill
                # KV, so free it here — lets the mmap'd shards fit in page cache
                # and avoids disk thrash at long context
                for layer in prefill.layers:
                    if isinstance(layer, CacheLayerMixin):
                        layer.keys = None
                        layer.values = None
                torch.cuda.empty_cache()
                # one engine holds an exclusive WAL lock on the shards, so reuse
                # it for exact and hnsw rather than opening a second
                native = QdrantEdgeNativeRetriever(
                    edge_root=str(edge_root), n_retrieved=self.n_retrieved
                )
                if self.exact:
                    native.exact, native.hnsw_ef = True, None
                    run("exact", native)
                if self.hnsw:
                    native.exact, native.hnsw_ef = False, self.hnsw_ef
                    run("hnsw", native)
        finally:
            del prefill
            torch.cuda.empty_cache()

        # exact/topk scored vs full; hnsw vs exact (the graph-quality gap)
        ref = {"topk": "full", "exact": "full", "hnsw": "exact"}
        rows = []
        for name, gen in gens.items():
            row = score_row(
                ex.bucket,
                ex.idx,
                name,
                gen,
                ex.label,
                reference=gens.get(ref.get(name, "")),
            )
            rows.append(row)
            ms = 1e3 * row.gen_seconds / max(row.gen_tokens, 1)
            console.print(
                f"[dim]{ex.bucket} #{ex.idx} {name}:[/] "
                f"contain={row.containment:.0f} f1={row.token_f1:.2f} "
                f"{ms:.0f}ms/tok"
            )
        return rows

    def _report(self, rows: list[EvalRow]) -> None:
        out_path = Path(
            self.out or f"cache/eval/niah_{time.strftime('%Y%m%d_%H%M%S')}.json"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps([r.model_dump() for r in rows], indent=2, ensure_ascii=False)
        )

        table = Table(title="NIAH accuracy")
        cols = (
            "bucket",
            "config",
            "n",
            "containment",
            "token_f1",
            "rouge_l",
            "ms/tok",
            "retr ms/tok",
        )
        for col in cols:
            table.add_column(col)
        seen: dict[tuple[int, str], list[EvalRow]] = {}
        for r in rows:
            seen.setdefault((r.bucket, r.config), []).append(r)
        for (bucket, config), group in sorted(seen.items()):
            n = len(group)

            def mean(attr: str, group=group, n=n) -> float:
                return sum(getattr(r, attr) for r in group) / n

            def per_tok(attr: str, group=group) -> float:
                toks = sum(r.gen_tokens for r in group) or 1
                return 1e3 * sum(getattr(r, attr) for r in group) / toks

            table.add_row(
                str(bucket),
                config,
                str(n),
                f"{mean('containment'):.2f}",
                f"{mean('token_f1'):.2f}",
                f"{mean('rouge_l'):.2f}",
                f"{per_tok('gen_seconds'):.0f}",
                f"{per_tok('retrieval_seconds'):.0f}",
            )
        console.print(table)
        console.print(f"wrote {out_path}")


class CmdRecord(BaseModel):
    """Record retrieval data across sizes by driving `chat` over a prompts file.
    kind=indices -> topk oracle (--record-indices); kind=queries -> native decode
    queries (--record-prompts)."""

    kind: Literal["indices", "queries"] = "indices"
    sizes: str = "100k,200k,1M"
    prompts: str = "prompts_sweep.txt"
    max_new_tokens: int = 256
    n_retrieved: int = 128
    model_name: ModelName = "Qwen/Qwen3.5-9B"
    dataset_name: Datasets = Datasets.QDRANT

    def cli_cmd(self) -> None:
        for size in self.sizes.split(","):
            retriever = (
                TopKRetriever(n_retrieved=self.n_retrieved)
                if self.kind == "indices"
                else QdrantEdgeNativeRetriever(n_retrieved=self.n_retrieved)
            )
            chat = CmdChat(
                model_name=self.model_name,
                dataset_name=self.dataset_name,
                qdrant_size=size,
                retriever=retriever,
                max_new_tokens=self.max_new_tokens,
                render_live=False,
                record_indices=(self.kind == "indices"),
                record_prompts=(self.kind == "queries"),
            )
            with open(self.prompts) as f:  # feed prompts to the (stdin-driven) repl
                orig, sys.stdin = sys.stdin, f
                try:
                    chat.cli_cmd()
                finally:
                    sys.stdin = orig


class CmdRebuildShards(BaseModel):
    """Rebuild edge shards as HNSW-indexed single-segment shards from the existing
    prefill (no re-prefill). Needs qdrant running at `url`."""

    sizes: str = "100k,200k,1M"
    url: str = "localhost"
    model_name: ModelName = "Qwen/Qwen3.5-9B"
    dataset_name: Datasets = Datasets.QDRANT

    def cli_cmd(self) -> None:
        config: PreTrainedConfig = AutoConfig.from_pretrained(self.model_name)
        for size in self.sizes.split(","):
            cache_dir = _cache_dir(self.dataset_name, size, config.model_type)
            cache, ctx = load_cache(cache_dir, config, "cpu")
            console.print(f"[{size}] ctx={ctx}; upserting -> {cache_dir}/edge")
            _upsert(cache, self.url, edge_root=cache_dir / "edge", parallel=1)
            del cache
            console.print(f"[green]{size} done[/]")


class CmdReport(BaseModel):
    """Regenerate report figures from the analysis envelopes and compose report/*.md
    into one standalone HTML file."""

    report_dir: str = "report"
    out: str = "cache/report/report.html"

    def cli_cmd(self) -> None:
        from kv_search import report as rp

        rp.make_figures()
        path = rp.build_html(report_dir=Path(self.report_dir), out=Path(self.out))
        console.print(f"[green]wrote {path}[/]")


class CmdKvSearch(
    BaseModel,
    cli_shortcuts={
        "retriever.url": "url",
        "retriever.n-retrieved": "n",
        "max-new-tokens": "g",
        "model-name": "m",
        "dataset-name": "d",
        "qdrant-size": "s",
        "retriever.api-key": "api-key",
        "retriever.type": "r",
    },
):
    prefill: CliSubCommand[CmdPrefill]
    chat: CliSubCommand[CmdChat]
    analyze: CliSubCommand[CmdAnalyze]
    figures: CliSubCommand[CmdFigures]
    record: CliSubCommand[CmdRecord]
    rebuild_shards: CliSubCommand[CmdRebuildShards]
    report: CliSubCommand[CmdReport]
    eval: CliSubCommand[CmdEval]

    def cli_cmd(self) -> None:
        CliApp.run_subcommand(self)


def main() -> None:
    CliApp.run(CmdKvSearch)


if __name__ == "__main__":
    main()
