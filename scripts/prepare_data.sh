#!/usr/bin/env bash
# Download UCF-101 (+ official train/test splits) and build the cached clip store.
# 6.9 GB download, 8 parallel connections through the local proxy.
set -euo pipefail
cd "$(dirname "$0")/.."

PROXY="${PROXY:-http://127.0.0.1:9981}"
mkdir -p data data/splits

if [ ! -f data/UCF-101.zip ]; then
  echo "== downloading UCF-101.zip (~6.9 GB) =="
  aria2c -x8 -s8 -k1M --all-proxy="$PROXY" --file-allocation=none \
         --console-log-level=warn --summary-interval=60 \
         -d data -o UCF-101.zip \
         "https://huggingface.co/datasets/quchenyuan/UCF101-ZIP/resolve/main/UCF-101.zip"
else
  echo "== data/UCF-101.zip already present =="
fi

if [ ! -f data/splits/ucfTrainTestlist/trainlist01.txt ]; then
  echo "== downloading official split lists =="
  curl -sL --proxy "$PROXY" -o data/splits.zip \
    "https://huggingface.co/datasets/quchenyuan/UCF101-ZIP/resolve/main/UCF101TrainTestSplits-RecognitionTask.zip"
  unzip -q -o data/splits.zip -d data/splits
fi

echo "== building cache (decode -> filter -> 16 frames @ 72x72 uint8) =="
.venv/bin/python prepare_data.py --zip data/UCF-101.zip --out data/cache \
    --test-list data/splits/ucfTrainTestlist/testlist01.txt \
    --frames 16 --size 72 --workers 16 \
    --min-std 6.0 --min-motion 1.0

echo "done: data/cache/{train,val}.npy (see data/cache/filter_report.json)"
