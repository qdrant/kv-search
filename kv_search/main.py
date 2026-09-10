import contextlib
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
    CollectionStatus,
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
from enum import Enum, auto
from pathlib import Path
from typing import Any, Literal

import rich
import torch
import transformers.utils.logging
from pydantic import BaseModel, Field
from pydantic_settings import CliApp, CliSubCommand
from rich.console import Console
from rich.progress import track
from rich.table import Table

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

from kv_search.analysis import CachedData
from kv_search.cache import (
    FullContextRetriever,
    QdrantEdgeNativeRetriever,
    QdrantEdgeRetriever,
    QdrantRetriever,
    RecordingCache,
    RetrievalCache,
    RetrieverConfig,
    TopKRetriever,
    bind_query_aware_cache,
    load_cache,
    save_cache,
)
from kv_search.data import (
    Datasets,
    EvalExample,
    load_dataset,
    load_niah_examples,
)
from kv_search.eval import EvalRow, GenerationResult, score_row
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
    """Smallest integer YaRN factor covering the context; 1 (=> no YaRN) at or
    below the native window. Applied only per-tier so sub-native runs stay
    undistorted (static YaRN degrades short contexts)."""
    return max(1, math.ceil(context_tokens / NATIVE_MAX_POSITIONS))


@timers.model_load
def _load_model(
    model_name: ModelName, context_tokens: int = 0
) -> tuple[ModelType, ProcessorType]:
    config = AutoConfig.from_pretrained(model_name)

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

    processor: ProcessorType = AutoProcessor.from_pretrained(model_name)
    load_kwargs = dict(
        config=config,
        attn_implementation=_ATTN_IMPL,
        dtype=torch.bfloat16,
        device_map="cuda",
    )
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


def _upsert(
    cache: DynamicCache,
    url: str,
    batch_size: int = 1024,
    api_key: str | None = None,
    edge_root: Path = Path("cache/edge"),
    parallel: int = 4,
):
    edge_root.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(url, api_key=api_key, prefer_grpc=True)
    rest = QdrantClient(url, api_key=api_key)
    assert isinstance(rest._client, QdrantRemote)
    for i, layer in track(
        enumerate(cache.layers), description="Upserting", total=len(cache.layers)
    ):
        if not isinstance(layer, CacheLayerMixin):
            continue

        for h in track(range(4), transient=True, description="Head"):
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
                        on_disk=True,
                        hnsw_config=HnswConfigDiff(m=0, on_disk=True),
                    ),
                    "value": VectorParams(
                        size=d,
                        distance=Distance.DOT,
                        on_disk=True,
                        hnsw_config=HnswConfigDiff(m=0),
                    ),
                },
                optimizers_config=OptimizersConfigDiff(
                    indexing_threshold=0, default_segment_number=1
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

            client.update_collection(
                collection_name=name,
                vectors_config={
                    "key": VectorParamsDiff(hnsw_config=HnswConfigDiff(m=16))
                },
                # threshold below the single segment's size so the graph builds
                optimizers_config=OptimizersConfigDiff(
                    indexing_threshold=1000, default_segment_number=1
                ),
            )
            _wait_indexed(client, name)

            indexed = client.get_collection(name).indexed_vectors_count
            if not indexed:
                raise RuntimeError(f"{name}: HNSW index not built ({indexed}/{n})")

            shard_dir = edge_root / f"layer{i:02d}_head{h}"

            with tempfile.TemporaryDirectory(dir=shard_dir.parent) as restore_dir:
                snapshot_path = Path(restore_dir) / "shard.snapshot"

                with requests.get(
                    f"{rest._client.rest_uri}/collections/{name}/shards/0/snapshot",
                    headers={"api-key": api_key} if api_key else None,
                    stream=True,
                ) as r:
                    r.raise_for_status()
                    with open(snapshot_path, "wb") as f:
                        f.writelines(r.iter_content(chunk_size=1 << 20))

                if shard_dir.exists():
                    shutil.rmtree(shard_dir)
                shard_dir.mkdir(parents=True, exist_ok=True)

                edge.EdgeShard.unpack_snapshot(str(snapshot_path), str(shard_dir))

            client.delete_collection(name)


def _wait_indexed(client: QdrantClient, name: str) -> None:
    while client.get_collection(name).status != CollectionStatus.GREEN:
        time.sleep(0.5)


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


class CmdPrefill(BaseModel):
    model_name: ModelName = "Qwen/Qwen3.5-9B"
    dataset_name: Datasets = Datasets.QDRANT
    qdrant_size: str = "100k"
    upsert: bool = False
    url: str = "localhost"
    api_key: str | None = None
    upsert_batch_size: int = 1024
    prefill_batch_size: int = 4096

    def cli_cmd(self) -> None:
        context_tokens = (
            _size_to_tokens(self.qdrant_size)
            if self.dataset_name == Datasets.QDRANT
            else 0
        )
        model, processor = _load_model(self.model_name, context_tokens)

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

        if self.upsert:
            del model, processor
            torch.cuda.empty_cache()
            _upsert(
                cache,
                self.url,
                self.upsert_batch_size,
                api_key=self.api_key,
                edge_root=cache_dir / "edge",
            )

        _print_stats()


_RETRIEVERS: dict[str, type[RetrieverConfig]] = {
    "topk": TopKRetriever,
    "full": FullContextRetriever,
    "qdrant": QdrantRetriever,
    "edge": QdrantEdgeRetriever,
    "native": QdrantEdgeNativeRetriever,
}


class CmdChat(BaseModel):
    model_name: ModelName = "Qwen/Qwen3.5-9B"
    dataset_name: Datasets = Datasets.QDRANT
    qdrant_size: str = "100k"
    retriever: RetrieverConfig = Field(default_factory=QdrantRetriever)
    max_new_tokens: int = 256
    render_live: bool = True
    record_indices: bool = False

    def cli_cmd(self) -> None:
        context_tokens = (
            _size_to_tokens(self.qdrant_size)
            if self.dataset_name == Datasets.QDRANT
            else 0
        )
        model, processor = _load_model(self.model_name, context_tokens)

        cache_dir = _cache_dir(
            self.dataset_name, self.qdrant_size, model.config.model_type
        )
        cache_dir.mkdir(exist_ok=True, parents=True)

        prefill, context_len = load_cache(cache_dir, model.config)

        # point edge/native retrievers at this tier's shard folder
        if isinstance(
            self.retriever, (QdrantEdgeRetriever, QdrantEdgeNativeRetriever)
        ):
            self.retriever.edge_root = str(cache_dir / "edge")

        if self.record_indices and isinstance(self.retriever, TopKRetriever):
            self.retriever.record_indices = self.record_indices

        cache = RetrievalCache(
            retriever=self.retriever, prefill=prefill, config=model.config
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
                        if isinstance(r, TopKRetriever):
                            r.record_indices = self.record_indices
                        if isinstance(r, (QdrantEdgeRetriever, QdrantEdgeNativeRetriever)):
                            r.edge_root = str(cache_dir / "edge")
                        instances[cmd] = r
                    cache.retriever = instances[cmd]
                    print(f"[retriever = {cmd}]")
                else:
                    print(f"unknown command '/{cmd}', see /help")
                continue

            record = self.record_indices and isinstance(cache.retriever, TopKRetriever)
            if record:
                cache.retriever.reset_indices()

            # isolate this prompt's timings; discard the first (cold) prompt when measuring
            timers.reset_generation()
            torch.cuda.reset_peak_memory_stats()
            try:
                prompt_len = self._generate(
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
                prompt_idx += 1
            except KeyboardInterrupt:
                streamer.end()
            finally:
                cache.reset()

            _print_stats()

    def _generate(
        self,
        model: ModelType,
        processor: ProcessorType,
        cache: RetrievalCache,
        context_len: int,
        streamer: TimedStreamer,
        user: str,
    ) -> int:
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

        model.generate(
            **inputs,  # ty:ignore[invalid-argument-type]
            max_new_tokens=self.max_new_tokens,
            past_key_values=cache,
            use_cache=True,
            streamer=streamer,
        )  # ty:ignore[invalid-argument-type]

        return prompt_len


class CmdAnalyze(BaseModel):
    model_name: ModelName = "Qwen/Qwen3.5-9B"
    dataset_name: Datasets = Datasets.QDRANT
    qdrant_size: str = "100k"

    def cli_cmd(self) -> None:
        config: PreTrainedConfig = AutoConfig.from_pretrained(self.model_name)

        cache_dir = _cache_dir(self.dataset_name, self.qdrant_size, config.model_type)
        cache_dir.mkdir(exist_ok=True, parents=True)

        data = CachedData(cache_dir, model_name=self.model_name)
        data.analyze()


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
    starts = [i for i in range(len(ids) - 1) if ids[i] == im_start and ids[i + 1] == user_tok]
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


class CmdEval(BaseModel):
    model_name: ModelName = "Qwen/Qwen3.5-9B"
    lang: str = "english"
    buckets: str = "71680"
    n_retrieved: int = 128
    hnsw_ef: int | None = None
    max_new_tokens: int = 128
    limit_examples: int = 0
    # which retrievers to score; exact/hnsw run on the same per-example edge shards
    full: bool = True
    exact: bool = True
    hnsw: bool = True
    topk: bool = False
    url: str = "localhost"
    api_key: str | None = None
    upsert_batch_size: int = 1024
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

            for ex in load_niah_examples(
                self.lang, bucket, self.limit_examples, multimodal
            ):
                rows.extend(self._eval_example(model, processor, ex))

        self._report(rows)

    def _eval_example(
        self, model: ModelType, processor: ProcessorType, ex: EvalExample
    ) -> list[EvalRow]:
        enc, n = _encode_eval(processor, ex)
        prefill = DynamicCache(config=model.config)
        _do_prefill(_slice_inputs(enc, n), model, prefill, self.prefill_batch_size)

        edge_root = (
            Path(tempfile.mkdtemp(prefix="eval_edge_", dir="cache"))
            if self.exact or self.hnsw
            else None
        )
        gens: dict[str, GenerationResult] = {}
        try:
            if edge_root is not None:
                _upsert(prefill, self.url, self.upsert_batch_size,
                        api_key=self.api_key, edge_root=edge_root)
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
            if edge_root is not None:
                shutil.rmtree(edge_root, ignore_errors=True)

        # exact/topk scored vs full; hnsw vs exact (the graph-quality gap)
        ref = {"topk": "full", "exact": "full", "hnsw": "exact"}
        rows = []
        for name, gen in gens.items():
            row = score_row(ex.bucket, ex.idx, name, gen, ex.label,
                            reference=gens.get(ref.get(name, "")))
            rows.append(row)
            ms = 1e3 * row.gen_seconds / max(row.gen_tokens, 1)
            console.print(f"[dim]{ex.bucket} #{ex.idx} {name}:[/] "
                          f"contain={row.containment:.0f} f1={row.token_f1:.2f} "
                          f"{ms:.0f}ms/tok")
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
        cols = ("bucket", "config", "n", "containment", "token_f1", "rouge_l",
                "ms/tok", "retr ms/tok")
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
    eval: CliSubCommand[CmdEval]

    def cli_cmd(self) -> None:
        CliApp.run_subcommand(self)


def main() -> None:
    CliApp.run(CmdKvSearch)


if __name__ == "__main__":
    main()
