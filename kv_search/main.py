import contextlib
import io
import os
import shutil
import sys
import tempfile

import qdrant_edge as edge
import requests
from qdrant_client import QdrantClient
from qdrant_client.models import Batch, Distance, HnswConfigDiff, VectorParams
from qdrant_client.qdrant_remote import QdrantRemote
from rich.live import Live
from rich.markdown import Markdown
from transformers.cache_utils import CacheLayerMixin

os.environ.setdefault("HF_HUB_VERBOSITY", "error")

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

# auto_docstring emits [ERROR] lines via print() at class-definition time
with contextlib.redirect_stdout(io.StringIO()):
    from transformers import (
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
        PreTrainedTokenizerBase,
        Qwen2ForCausalLM,
        Qwen2Tokenizer,
        Qwen3_5ForConditionalGeneration,
        Qwen3VLProcessor,
        TextStreamer,
    )

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
from kv_search.data import Datasets, Message, load_dataset
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


@timers.model_load
def _load_model(model_name: ModelName) -> tuple[ModelType, ProcessorType]:
    processor: ProcessorType = AutoProcessor.from_pretrained(model_name)
    if model_name in IS_MULTIMODAL:
        model: ModelType = AutoModelForMultimodalLM.from_pretrained(
            model_name,
            attn_implementation=_ATTN_IMPL,
            dtype=torch.bfloat16,
            device_map="cuda",
        ).eval()
    else:
        model: ModelType = AutoModelForCausalLM.from_pretrained(
            model_name,
            attn_implementation=_ATTN_IMPL,
            dtype=torch.bfloat16,
            device_map="cuda",
        ).eval()

    bind_query_aware_cache(model)
    return model, processor


@timers.prefill_gen
def _do_prefill(
    messages: Message,
    model: ModelType,
    processor: ProcessorType,
    past_key_values: RecordingCache,
    batch_size: int = 4096,
):
    inputs: BatchEncoding[torch.Tensor] = processor.apply_chat_template(
        messages.prefill,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )  # ty:ignore[invalid-assignment]
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
    cache: RecordingCache, url: str, batch_size: int = 128, api_key: str | None = None
):
    client = QdrantClient(url, api_key=api_key)
    assert isinstance(client._client, QdrantRemote)
    for i, layer in track(
        enumerate(cache.layers), description="Upserting", total=len(cache.layers)
    ):
        if not isinstance(layer, CacheLayerMixin):
            continue

        for h in track(range(4), transient=True, description="Head"):
            if client.collection_exists(f"layer={i};head={h}"):
                client.delete_collection(f"layer={i};head={h}")

            assert layer.keys is not None
            assert layer.values is not None

            client.create_collection(
                collection_name=f"layer={i};head={h}",
                vectors_config={
                    "key": VectorParams(
                        size=layer.keys.shape[-1], distance=Distance.DOT
                    ),
                    "value": VectorParams(
                        size=layer.keys.shape[-1],
                        distance=Distance.DOT,
                        hnsw_config=HnswConfigDiff(m=0),
                    ),
                },
            )

            ids = torch.split(
                torch.arange(layer.keys.shape[2], dtype=torch.int), batch_size
            )
            keys = torch.split(layer.keys[0, h].cpu().to(torch.float), batch_size)
            values = torch.split(layer.values[0, h].cpu().to(torch.float), batch_size)

            for idx, k, v in track(
                zip(ids, keys, values),
                total=len(keys),
                transient=True,
                description="Batch",
            ):
                client.upsert(
                    collection_name=f"layer={i};head={h}",
                    points=Batch(
                        ids=idx.tolist(),
                        vectors={
                            "key": k.numpy(),
                            "value": v.numpy(),
                        },
                    ),
                )

            shard_dir = Path("cache") / "edge" / f"layer{i:02d}_head{h}"

            with tempfile.TemporaryDirectory(dir=shard_dir.parent) as restore_dir:
                snapshot_path = Path(restore_dir) / "shard.snapshot"

                with requests.get(
                    f"{client._client.rest_uri}/collections/layer={i};head={h}/shards/0/snapshot",
                    headers={"api-key": api_key} if api_key else None,
                    stream=True,
                ) as r:
                    r.raise_for_status()
                    with open(snapshot_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)

                if shard_dir.exists():
                    shutil.rmtree(shard_dir)
                shard_dir.mkdir(parents=True, exist_ok=True)

                edge.EdgeShard.unpack_snapshot(str(snapshot_path), str(shard_dir))


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
    upsert: bool = False
    url: str = "localhost"
    api_key: str | None = None
    upsert_batch_size: int = 128
    prefill_batch_size: int = 4096

    def cli_cmd(self) -> None:
        model, processor = _load_model(self.model_name)

        cache_dir = Path(f"cache/{self.dataset_name}/{model.config.model_type}")
        cache_dir.mkdir(exist_ok=True, parents=True)

        cache = RecordingCache(path=cache_dir, config=model.config)
        messages = load_dataset(
            self.dataset_name, multimodal=self.model_name in IS_MULTIMODAL
        )

        _do_prefill(messages, model, processor, cache, self.prefill_batch_size)

        save_cache(
            cache,
            cache_dir,
            cache.get_seq_length(),
        )
        cache.finalize()

        if self.upsert:
            _upsert(cache, self.url, self.upsert_batch_size, api_key=self.api_key)

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
    retriever: RetrieverConfig = Field(default_factory=QdrantRetriever)
    max_new_tokens: int = 256
    render_live: bool = True
    record_indices: bool = False

    def cli_cmd(self) -> None:
        model, processor = _load_model(self.model_name)

        cache_dir = Path(f"cache/{self.dataset_name}/{model.config.model_type}")
        cache_dir.mkdir(exist_ok=True, parents=True)

        prefill, context_len = load_cache(cache_dir, model.config)

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
                        if hasattr(r, "n_retrieved"):
                            r.n_retrieved = n_retrieved
                        if hasattr(r, "record_indices"):
                            r.record_indices = self.record_indices
                        instances[cmd] = r
                    cache.retriever = instances[cmd]
                    print(f"[retriever = {cmd}]")
                else:
                    print(f"unknown command '/{cmd}', see /help")
                continue

            record = self.record_indices and isinstance(cache.retriever, TopKRetriever)
            if record:
                cache.retriever.reset_indices()

            try:
                self._generate(model, processor, cache, context_len, streamer, user)
            except KeyboardInterrupt:
                streamer.end()
            finally:
                cache.reset()

            if record:
                cache.retriever.save_indices(cache_dir)

    def _generate(
        self,
        model: ModelType,
        processor: ProcessorType,
        cache: RetrievalCache,
        context_len: int,
        streamer: TimedStreamer,
        user: str,
    ):
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


class CmdKvSearch(
    BaseModel,
    cli_shortcuts={
        "retriever.url": "url",
        "retriever.n-retrieved": "n",
        "max-new-tokens": "g",
        "model-name": "m",
        "dataset-name": "d",
        "retriever.api-key": "api-key",
        "retriever.type": "r",
    },
):
    prefill: CliSubCommand[CmdPrefill]
    chat: CliSubCommand[CmdChat]

    def cli_cmd(self) -> None:
        CliApp.run_subcommand(self)


def main() -> None:
    CliApp.run(CmdKvSearch)


if __name__ == "__main__":
    main()
