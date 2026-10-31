#!/usr/bin/env bash
# Generate clips and compute FVD against held-out real clips.
#
#   scripts/eval.sh runs/base/ckpt_final.pt [extra args for evaluate.py]
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

CKPT="${1:?usage: scripts/eval.sh <ckpt.pt> [extra args]}"
shift || true
.venv/bin/python -m vidgen.evaluate --ckpt "$CKPT" --cache-dir data/cache \
    --n-classes 32 --per-class 8 --steps 50 --cfg-scale 4 "$@"
