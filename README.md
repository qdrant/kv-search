# kv-search

Query-aware KV-cache retrieval experiments. A prompt is prefilled once into a
KV cache; generation then attends over a top-k retrieval of that cache instead
of the full context.

## Setup

```sh
uv sync
```

CUDA is required for running. `flash-attn` builds are Linux-only; elsewhere it falls back to
SDPA automatically.

## Workflow

Shared flags: `-m` model, `-d` dataset (`qdrant`, `squad`, `niah`). The cache is
stored under `cache/{dataset}/{model_type}/`, so `prefill` must run before
`chat`/`analyze` for a given model+dataset.

### Prefill

Compute and save the KV cache for a dataset:

```sh
uv run kv-search prefill
```

Optionally push key/value vectors to Qdrant (needed only for the `qdrant`/`edge`/`native`
retrievers):

```sh
uv run kv-search prefill --upsert --url localhost --api-key <key>
```

### Chat

Interactive generation against the prefilled cache:

```sh
uv run kv-search chat -r native -g 512
```

- `-r` retriever: `native`, `edge`, `qdrant`, `topk`, `full`
- `-g` max new tokens, `-n` top-k retrieved per step
- `--record-indices` saves per-prompt retrieval indices/scores for `analyze`, only works with `-r topk`

In the REPL: `/full` switches to full-context generation, `/native` (or any
retriever name) switches back, `/live` toggles live rendering, `/help` lists
commands.

Pipe prompts instead of typing them; each line is a separate prompt and the
session exits at EOF:

```sh
uv run kv-search chat -r native -g 512 < prompt.txt
```

### Analyze

Generate some plots and tables (requires a `chat --record-indices` run
first):

```sh
uv run kv-search analyze
```

Writes plots into the cache directory.
