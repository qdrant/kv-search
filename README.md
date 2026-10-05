# LLM Memory in Qdrant

A question-answering demo over the Qdrant codebase and documentation where the knowledge base is stored in Qdrant as the
LLM's own hidden states, one vector per token.

The same vectors do two jobs:

1. **Retrieval.** Questions are encoded by the same network, and each question token searches Qdrant for its nearest
   corpus tokens. There is no chunking and no separate embedding model.
2. **Memory.** The vectors around the places found are fed straight back into the LLM as knowledge, without re-reading
   the document text through the lower layers.

The model also learned to say "not in the text" when the answer is not in its memory. A single threshold lets you
trade answer coverage for reliability.

This is a research prototype built in about a week on consumer GPUs. It works end to end, and the sections below say
honestly where it wins and where it does not.

## How It Works

**Index.** A frozen Qwen3-4B-Instruct-2507 reads each document whole, and a small tower (298M parameters, 3
bidirectional transformer blocks) reads the hidden states of 19 layers and produces one 2560-d vector per token. Each
vector knows the context of its document. No text is dropped, but one pass reads at most 16K tokens: the 26 longer
documents of the Qdrant corpus (12% of its tokens) are encoded in consecutive 16K-token windows that do not see each
other. Every token becomes a Qdrant point: the vector is stored as float16 on disk, with binary quantization in RAM
and an HNSW graph.

**Search.** The same model and tower encode the question. Each question token runs a kNN query in Qdrant (top 256).
Hits are grouped into 64-token windows by maximum score and votes. The candidate windows are re-ranked with MaxSim
(the mean over question tokens of the best match in the window), and the 4 best places go to the reader.

**Reading.** For each place, the vectors of a 1024-token piece around it are inserted into the model as memory at the
input of layer 13. A trained copy of layer 12 (the reader, about 100M parameters, with its own attention and KV cache)
mixes the memory with the prompt. Layers 13 to 35 of the frozen model see the prompt, the memory, and the text of the
four 64-token windows, and they generate the answer.

**Abstaining.** The probability of the answer "not in the text" at the first generated token is compared with a
threshold. Above it, the model abstains. Below it, the model gives its best real answer.

## Results

All numbers come from data the tower and reader never saw in training: the Qdrant sources are excluded from the
training data.

> **Checkpoints behind these tables.** Qdrant retrieval is measured with `s2c`, the released checkpoint. The other two
> tables were measured during development: end-to-end answers with the tower after the search stage for search and
> `s2c` for reading, and ConTEB with the tower of the stage before `s2c`. They will be re-measured with `s2c` alone.

The first table shows retrieval on the Qdrant corpus: 400 generated questions, scored by whether the source 64-token
window is among the top 1, 4, or 16 results.

| Retriever | Top 1 | Top 4 | Top 16 |
|---|---|---|---|
| **Ours (tower, one vector per token)** | **0.20** | **0.38** | **0.49** |
| ColBERT (answerai-colbert-small-v1) | 0.14 | 0.27 | 0.42 |
| bge-small-en-v1.5 | 0.12 | 0.26 | 0.38 |
| BM25 | 0.10 | 0.18 | 0.29 |

The second table shows the public ConTEB benchmark (Conti et al., 2025). It tests whether a retriever can find a
chunk whose meaning depends on its document; in Football and Geography, names in chunks are replaced by pronouns. The
measure is nDCG@10, and all rows ran in one setup.

| Retriever | COVID-QA | Geography | Football | Insurance |
|---|---|---|---|---|
| **Ours, whole document** | **77.8** | **66.9** | **29.9** | **27.6** |
| Ours, each chunk encoded alone | 61.3 | 49.2 | 17.0 | 15.6 |
| ColBERT (answerai-colbert-small-v1) | 74.1 | 62.7 | 25.5 | 16.3 |
| bge-small-en-v1.5 | 52.9 | 54.6 | 18.8 | 9.5 |
| BM25 | 64.0 | 50.9 | 15.8 | 8.9 |

Document context adds 12 to 18 points: the tower and the scoring are the same, and the only difference is whether the
tower saw the whole document. Methods trained specifically for this benchmark (InSeNT, Anthropic Contextual
Retrieval) are far ahead on Geography and Football, at about 90 and 55 to 64.

The third table shows answers end to end on the same 400 Qdrant questions. The reader is Qwen3-4B, it gets 4 found
places, and the judge is Qwen3-30B-A3B. The score follows CRAG: +1 for a right answer, 0 for "not in the text", and
-1 for a wrong answer.

| System | Right | Wrong | "Not in the text" | Score |
|---|---|---|---|---|
| **Ours: memory + 4 × 64 text tokens, threshold 0.8** | **0.54** | **0.29** | 0.17 | **+0.26** |
| **Ours: same, threshold 0.5** | **0.52** | **0.20** | 0.28 | **+0.32** |
| Ours: memory only, no text (greedy) | 0.40 | 0.23 | 0.38 | +0.17 |
| ColBERT, text 4 × 64 | 0.52 | 0.35 | 0.13 | +0.18 |
| bge-small, text 4 × 64 | 0.48 | 0.33 | 0.20 | +0.15 |
| ColBERT, text 4 × 2048 (32x more text) | 0.63 | 0.29 | 0.08 | +0.34 |

At the same text budget, memory gives as many right answers as ColBERT with 40% fewer wrong ones. RAG that reads 32x
more text still gets more right answers. Re-judging the same answers moves the numbers by about 1 point.

## Requirements

- An NVIDIA GPU with 16 GB of memory. The demo was run on an RTX 5080 and an RTX 4090.
- 32 GB of RAM and about 75 GB of free disk: 33 GB for the local index and 36 GB for the Qdrant collection.
- Qdrant 1.10 or later. The demo was run with Qdrant 1.19.1.
- Python 3.10 or later, PyTorch 2.14, and Transformers 5.17. The code relies on the decoder-layer API of
  Transformers 5.
- Optional, for the steps that need them: `openai` (the LLM judge and data generation), `vllm` (serving the judge or
  the generator), `pyarrow` (building the training corpus), and `pylate` (the ColBERT evaluation baseline as
  published).

## Setup

1. Install PyTorch for your CUDA version, then the rest:

   ```bash
   pip install -r requirements.txt
   ```

2. Download the [`llm_memory` folder from Google Drive](https://drive.google.com/drive/folders/1g59BuvPFOL2FK1RSnOnMEVxzm002_iw4?usp=sharing)
   and put its contents into the repository root. The demo needs two of its files:

   ```
   checkpoints/s2c.pt   # tower + reader, 1.6 GB
   data/docs.jsonl      # Qdrant corpus, 4,856 files, 30 MB
   ```

   The folder also holds the evaluation questions (`data/questions-v.jsonl`), the training data (`data/train/`), and
   `SHA256SUMS` to check the download with `sha256sum -c SHA256SUMS`.

   Hugging Face downloads the models on first use: the frozen
   [`Qwen/Qwen3-4B-Instruct-2507`](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507), `BAAI/bge-small-en-v1.5`, and
   `answerdotai/answerai-colbert-small-v1`. To use local copies, pass `--model`, `--bge`, and `--colbert`.

3. Start Qdrant:

   ```bash
   docker run -d -p 6333:6333 -p 6334:6334 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant:v1.19.1
   ```

4. Build the index. This encodes 6.7M tokens and takes about 15 minutes on an RTX 5080:

   ```bash
   python -m llm_memory index
   ```

5. Upload it to Qdrant. This takes about 15 minutes, and the HNSW graph builds in the background a few minutes after:

   ```bash
   python -m llm_memory upload
   ```

6. Build the baseline collections for side-by-side comparison (`rag_bge` and `rag_colbert`, about 7 minutes):

   ```bash
   python -m llm_memory baselines
   ```

## Running the Demo

```bash
python -m llm_memory tui --local-vectors
```

This opens a full-screen terminal app, which also works over SSH. Type a question and press Enter: the answer appears
token by token as the model generates it. Below the answer you see the probability that the answer is not in memory,
a map of what the model read in each found document, and where the time went.

| Key | Effect |
|---|---|
| Tab, Shift+Tab | Switch between memory with text, memory only, RAG with bge-small, and RAG with ColBERT for the same question. Answers are kept, so switching back is instant |
| Ctrl+T | Cycle the abstain threshold: 0.8, 0.5, off. A lower threshold gives more "not in the text" and fewer wrong answers |
| Up, Down | Question history |
| Ctrl+Q | Quit |

By default the comparison is at an equal text budget. Our model gets 128 text tokens around each of 8 found places,
1,024 in all, which is as much text as RAG reads in 4 chunks of 256 tokens. On top of that it reads memory: the token
vectors of 512 tokens around the first place in each document. These settings were chosen on 44 questions written
the way people ask them; `--ret-k 4 --text-win 64 --piece 1024` restores the setup of the end-to-end table. The memory
only mode reads 4 pieces of 1,024 tokens around the first 4 places and no text at all, as in training.

With `--local-vectors`, the window and memory vectors come from the local index instead of being retrieved from Qdrant
by id. kNN still runs in Qdrant. Retrieving about 8K vectors by id from Qdrant takes several seconds today, while the
whole answer takes about 1.5 s with local vectors.

Some questions to start with, from everyday use to the internals of the code:

- How do I enable binary quantization for a collection?
- How can I filter search results by a payload field?
- What does the `on_disk_payload` option do?
- How does Qdrant support multivectors for late-interaction models like ColBERT?
- Which field type supports filtering by geographic bounding boxes and radius conditions?
- What does `combine_min_should_estimations` return when the minimum count exceeds the number of estimations?

For scripts, `ask` prints the same results as plain text. To ask one question in all three modes and save the
answers:

```bash
python -m llm_memory ask --local-vectors --mode all --q "How do I enable binary quantization for a collection?" \
    --dump answers.jsonl
```

`--file` takes a text file with one question per line instead. Without `--q` or `--file`, `ask` runs a plain-text
session with the commands `/ours`, `/mem`, `/rag`, `/colbert`, `/all`, `/thr 0.5`, and `/thr off`.

## Training

Training data lives in `data/train/`: the corpus `docs.jsonl`, synthetic facts `facts-0.jsonl` and `facts-1.jsonl`,
generated questions `questions.jsonl`, and topic neighbors `neighbors.json`. It is in the same Google Drive folder as the checkpoint, or you can
regenerate it as described in [the data generation guide](llm_memory/datagen/README.md).

[`scripts/train.sh`](scripts/train.sh) runs the recipe in three stages on one 24 GB GPU:

1. **Search.** Only the tower learns. The query tokens are matched by MaxSim against 64-token windows of the target
   piece and of topic-neighbor documents, with InfoNCE and windows of recent episodes as extra cross-document
   negatives. About 1.3 hours per 1,000 steps on an RTX 4090.
2. **Reader.** The tower, the reader, and the mixing scalar learn together. The losses are the answer's cross-entropy,
   a match of the layer-13 input to a teacher that reads all pieces as text, and the search loss. Episodes have the
   answer in the text window, only in memory, far away in the document, or absent. About 2.4 hours per 1,000 steps.
3. **Reader as applied.** Memory is assembled exactly as after retrieval: 4 pieces of 1,024 tokens around the best
   windows, ordered by search score, with 30% of episodes that have no answer. About 3.7 hours per 1,000 steps.

Each stage evaluates on held-out documents during training: search top-1 and, for the reader, answer accuracy and the
NLL of the reference per episode kind. Start a stage from a checkpoint with `--init`, and continue an interrupted run
with `--resume`.

The released checkpoint reached stage 3 by a longer development path, so this recipe from scratch has not been run end
to end yet. The training code was checked against the research code it was ported from: step-by-step losses match
exactly for the search stage and within the research code's own run-to-run GPU noise for the reader stages.

## Evaluation

The numbers in [Results](#results) come from these steps. All commands take `--help`.

```bash
# evaluation episodes from the verified Qdrant questions
python -m llm_memory.evaluate episodes --questions data/questions-v.jsonl --docs data/docs.jsonl --out data/eval

# our retrieval over the local index (python -m llm_memory index), hit at 1, 4, and 16
python -m llm_memory.evaluate search --ckpt checkpoints/s2c.pt --idx data/index --knn 256 --dump data/eval/ours.json

# baseline retrieval on the same windows
python -m llm_memory.retrievers bm25 --dump data/eval/bm25.json
python -m llm_memory.retrievers dense --embedder BAAI/bge-small-en-v1.5 --kind bge --chunk 64 --dump data/eval/bge-64.json
python -m llm_memory.retrievers colbert --dump data/eval/colbert.json

# answers: ours (memory + text of 4 windows, with the abstain curve) and RAG over any retrieval dump
python -m llm_memory.evaluate read --reader ours --ckpt checkpoints/s2c.pt --retrieved data/eval/ours.json \
    --piece 1024 --win-all --abstain-curve --dump-answers data/eval/answers/ours.jsonl
python -m llm_memory.evaluate read --reader rag --retrieved data/eval/colbert.json --ret-method colbert \
    --dump-answers data/eval/answers/rag-colbert.jsonl

# LLM judge, then the right / wrong / "not in the text" tallies and the abstain-threshold curve
vllm serve stelterlab/Qwen3-30B-A3B-Instruct-2507-AWQ --served-model-name judge --port 8004 --max-model-len 4096
python -m llm_memory.judge judge --server http://127.0.0.1:8004/v1 --answers data/eval/answers/ours.forced.jsonl,data/eval/answers/rag-colbert.jsonl
python -m llm_memory.judge crag data/eval/answers/*.judged.jsonl
python -m llm_memory.judge curve data/eval/answers/ours.forced.judged.jsonl --tau 0.5,0.8,never

# ConTEB, the public benchmark
python -m llm_memory.conteb prep
python -m llm_memory.conteb eval --ckpt checkpoints/s2c.pt --sets insurance,covid-qa,geography,football
```

## Data

The corpora, facts, and questions are produced by `llm_memory.datagen`; see
[the data generation guide](llm_memory/datagen/README.md) for the exact commands. Everything was generated by
Qwen3-30B-A3B-Instruct-2507 (AWQ) served with vLLM. The Qdrant evaluation questions were checked by the same model
for correctness and self-containment. The training questions were used without that check.

## Files

| Path | Contents |
|---|---|
| `llm_memory/model.py` | Frozen LLM, tower, reader, abstain threshold |
| `llm_memory/store.py` | Index build, Qdrant upload, search, memory fetch |
| `llm_memory/baselines.py` | bge-small and ColBERT baselines of the demo |
| `llm_memory/__main__.py` | Demo command line: `index`, `upload`, `baselines`, `ask`, `tui` |
| `llm_memory/tui.py` | Terminal app for the demo |
| `llm_memory/train.py` | Training of the tower and the reader |
| `llm_memory/evaluate.py` | Evaluation episodes, retrieval hit@k, answers for the judge |
| `llm_memory/retrievers.py` | BM25, dense, and ColBERT retrieval for evaluation |
| `llm_memory/judge.py` | LLM judge, right / wrong / abstain tallies, abstain-threshold curve |
| `llm_memory/conteb.py` | ConTEB benchmark |
| `llm_memory/datagen/` | Corpora, topic neighbors, synthetic facts and questions, question checks |
| `scripts/train.sh` | Training recipe |

## Limitations

- The index has one 2560-d vector per token, about 5 KB per token, so the 6.7M-token corpus takes 36 GB in Qdrant.
  Compact memory is the main open question. First naive attempts did not work: a gate trained to drop tokens pruned
  worse than random, and averaging token vectors per chunk collapsed retrieval.
- Reading costs about 4K memory positions through 24 layers (the reader and layers 13 to 35) per question, so it is
  not cheaper than RAG yet.
- RAG with large windows (4 × 2048 tokens) still gets more right answers.
- In training, the tower only saw pieces of 1K to 2K tokens, while the index feeds it whole documents of up to 16K
  tokens. It generalizes (document context adds 12 to 18 points on ConTEB), but training on whole documents is an
  untested lever.
- The model is Qwen3-4B; heavier models are expected to read memory better but were not tested.

## Provenance

- Corpus: `qdrant/qdrant` at 6ab21ca (September 3, 2026), `qdrant/qdrant-client` at cf747f4 (September 16, 2026), and
  the Qdrant documentation and articles. It has 4,856 files and 6.73M Qwen tokens.
- Training data: synthetic facts and questions generated over Wikipedia, CNN/DailyMail, GitHub code (search engines,
  vector databases, and the Rust ecosystem), and SQuAD. Everything that mentions Qdrant was removed.
- Checkpoint `s2c.pt`: stage 3 of the recipe, trained on October 4, 2026 on memory assembled exactly as after
  retrieval, plus episodes where the answer is absent.
