#!/usr/bin/env bash
# Create the venv and install dependencies.
set -euo pipefail
cd "$(dirname "$0")/.."

uv venv --python 3.12 .venv
VIRTUAL_ENV="$PWD/.venv" uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
VIRTUAL_ENV="$PWD/.venv" uv pip install numpy opencv-python-headless

echo "verifying:"
.venv/bin/python -c "import torch, torchvision, cv2, numpy; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'cuda_available', torch.cuda.is_available())"
