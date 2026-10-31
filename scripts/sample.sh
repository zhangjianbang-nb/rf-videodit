#!/usr/bin/env bash
# Sample from a checkpoint.
#
#   scripts/sample.sh runs/base/ckpt_final.pt --classes "BasketballDunk,Surfing" --per-class 4
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

CKPT="${1:?usage: scripts/sample.sh <ckpt.pt> [extra args for sample.py]}"
shift || true
.venv/bin/python -m vidgen.sample --ckpt "$CKPT" --cache-dir data/cache "$@"
