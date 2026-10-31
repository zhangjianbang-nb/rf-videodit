#!/usr/bin/env bash
# Launch training on GPU 1 (GPU 0 hosts the Bonsai llama-server and must not be touched).
#
#   scripts/train.sh                 # default run
#   scripts/train.sh --steps 50000   # extra args are forwarded to train.py
set -euo pipefail
cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN_DIR="${RUN_DIR:-runs/base}"
mkdir -p "$RUN_DIR"

nohup .venv/bin/python -u -m vidgen.train \
    --cache-dir data/cache \
    --run-dir "$RUN_DIR" \
    --crop 64 --frames 16 \
    --hidden 384 --depth 12 --heads 6 --patch 2,4,4 \
    --batch-size 8 --accum 2 \
    --lr 3e-4 --warmup 500 --ema-decay 0.999 \
    --t-mode logit_normal --loss-weight none \
    --steps 200000 --max-hours 6 \
    --num-workers 8 \
    --log-every 50 --ckpt-every 2000 --sample-every 5000 \
    --sample-count 8 --sample-steps 25 --cfg-scale 4 \
    "$@" > "$RUN_DIR/train.log" 2>&1 &

echo "training started (pid $!); log: $RUN_DIR/train.log"
