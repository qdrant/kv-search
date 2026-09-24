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

#### Tail correction (tailM)

`--tailm` puts back the softmax mass of the prefill keys the retrieval dropped, per head, from the
files `scripts/build_tailm.py` writes (`<cache>/tailm`, or `--tailm-dir`). Retrieval stays at `-n`
keys; only heads the build gated on recorded decode sessions are corrected, and the startup message
lists them. Works with `-r topk` and `-r native`. On `-r native` the decode steps compute the
correction in Rust, next to each KV head's search (bf16 weights; the startup message names the CPU
kernel); the prompt pass and `-r topk` compute it on the GPU.

```sh
uv run kv-search chat -d niah -r topk --tailm --tailm-check
```

`--tailm-check` prints after each answer how far today's and tailM's attention output are from
exact full attention.

Building the files takes two steps, once per prefill cache:

1. Record decode sessions to gate the heads on. The `record` subcommand lives in the research
   checkout of kv-search, not here. Use prompts other than the ones you evaluate with:
   ```sh
   # in the research checkout
   uv run kv-search record -d niah -g 400 --prompts-file <prompts.txt> --out <sessions-dir>
   ```
2. Fit, gate and write one file per (full-attention layer, KV head):
   ```sh
   uv run python scripts/build_tailm.py --cache cache/niah/qwen3_5 --validate <sessions-dir>
   ```
   Without `--validate` every head is written gated off, and `chat --tailm` refuses to start
   ("not validated"). `--validate-only --validate <dir>` gates existing files later. The files are
   built for one retrieval size (`--n-retrieved`, default 128); run `chat` with the same `-n`,
   otherwise every head stays off. `scripts/build_tailm.py --help` lists the other options.

### Analyze

Generate some plots and tables (requires a `chat --record-indices` run
first):

```sh
uv run kv-search analyze
```

Writes plots into the cache directory.
