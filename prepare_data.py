"""CLI: unzip UCF-101, filter the clips and build the cached tensor store.

    python prepare_data.py --zip data/UCF-101.zip --out data/cache
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from vidgen.data import build_cache


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--zip", default="data/UCF-101.zip")
    p.add_argument("--video-root", default="data/ucf101")
    p.add_argument("--out", default="data/cache")
    p.add_argument("--test-list", default="data/splits/ucfTrainTestlist/testlist01.txt")
    p.add_argument("--frames", type=int, default=16)
    p.add_argument("--size", type=int, default=72)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--skip-extract", action="store_true")
    # data filters
    p.add_argument("--min-std", type=float, default=6.0,
                   help="drop clips whose pixel std is below this (flat/black footage)")
    p.add_argument("--min-motion", type=float, default=1.0,
                   help="drop clips whose mean frame-to-frame |diff| is below this (static)")
    p.add_argument("--max-per-class", type=int, default=0,
                   help="cap clips per class (0 = no cap)")
    return p.parse_args()


def main():
    args = parse_args()
    root = Path(args.video_root)
    if not args.skip_extract:
        if root.exists() and any(root.rglob("*.avi")):
            print(f"[prep] {root} already contains videos, skipping extraction")
        else:
            root.mkdir(parents=True, exist_ok=True)
            print(f"[prep] extracting {args.zip} -> {root} (this takes a few minutes)")
            subprocess.run(["unzip", "-q", "-o", args.zip, "-d", str(root)], check=True)
            print("[prep] extraction done")

    test_list = args.test_list if Path(args.test_list).exists() else None
    if test_list is None:
        print(f"[prep] test list {args.test_list} not found; falling back to a deterministic 10% split")
    build_cache(str(root / "UCF-101") if (root / "UCF-101").is_dir() else str(root),
                args.out, num_frames=args.frames, size=args.size,
                workers=args.workers, test_list=test_list,
                min_std=args.min_std, min_motion=args.min_motion,
                max_per_class=args.max_per_class)


if __name__ == "__main__":
    main()
