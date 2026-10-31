#!/usr/bin/env bash
# Launch the local inference web UI (GPU 1 only).
#
#   scripts/webapp.sh [ckpt] [--port 8093]
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

CKPT="${1:-runs/base/ckpt_best.pt}"
shift || true

NOTES="模型仅训练约 8.8 万步（单卡 4090 约 6 小时，与其它服务共享 GPU），样本仍偏噪点：\`std\` 0.72（真实 0.51）、帧间抖动 0.82（真实 0.15），FVD ≈ 990（噪声地板 275）。这是早期检查点，不是最终质量。"

exec .venv/bin/python -u -m vidgen.webapp \
    --ckpt "$CKPT" --cache-dir data/cache --out-dir webapp/videos \
    --frames 16 --size 64 --steps 25 --cfg-scale 3.0 \
    --notes "$NOTES" "$@"
