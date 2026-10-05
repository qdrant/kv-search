#!/bin/bash
# Training recipe: three stages on one 24 GB GPU (RTX 4090 class). Data layout: see llm_memory/datagen/README.md.
#
# The released checkpoint s2c.pt went through a longer research path before stages 1 to 3: an earlier 2-block tower
# trained with 8x token compression, grown to 3 blocks and fine-tuned without compression. Stage 1 then ran for 600 steps
# on top of it. This script starts the tower from scratch; it has not been run end to end yet, so the step count of
# stage 1 is an estimate.
set -e
M=${MODEL:-Qwen/Qwen3-4B-Instruct-2507}; D=${DATA:-data/train}; R=${RUNS:-runs}
COMMON="--model $M --docs $D/docs.jsonl --facts $D/facts-0.jsonl,$D/facts-1.jsonl --questions $D/questions.jsonl \
  --neighbors $D/neighbors.json"

# 1. Search: only the tower learns (InfoNCE over MaxSim with cross-document negatives from a queue of recent episodes).
python -m llm_memory.train --stage search $COMMON --out $R/search \
  --ndist 3 --dpiece 1024 --piece 2048 --nq 8 --bank 24 --steps 6000 --accum 4 --lr 1e-4 --eval-every 500 --eval-n 100

# 2. Reader: the tower, the reader R, and the mixing scalar learn together on classic episodes.
python -m llm_memory.train --stage reader $COMMON --init $R/search/last.pt --out $R/reader \
  --ndist 3 --dpiece 512 --piece 2048 --nq 8 --bank 16 --steps 2300 --accum 8 --lr 1e-4 --rlr 5e-5 --eval-every 500 --eval-n 40

# 3. Reader as applied: memory assembled exactly as after retrieval, 30% of episodes without the answer.
python -m llm_memory.train --stage reader $COMMON --init $R/reader/last.pt --out $R/reader-app \
  --app --app-pieces 4 --piece 1024 --dscan 2048 --mix cite:0.3,know:0.35,far:0.05,absent:0.3 --qnat 0.5 \
  --bank 8 --tckpt --steps 1900 --accum 8 --lr 3e-5 --rlr 5e-5 --eval-every 400 --eval-n 40
