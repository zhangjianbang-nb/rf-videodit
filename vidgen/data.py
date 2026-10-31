"""Data pipeline: decode UCF-101 clips, filter them, and cache as uint8 tensors.

Filters (applied per clip, with a report of what was dropped and why)
--------------------------------------------------------------------
* too_short    -- fewer frames than `num_frames`, so the uniform 16-frame sample
                  would repeat frames and the clip would look frozen
* flat         -- near-uniform clip (std below `min_std`): black frames, credits,
                  colour bars -- nothing for a generative model to learn
* no_motion    -- mean |frame-to-frame difference| below `min_motion`: static
                  footage, which teaches the model that video does not move

The cache is a set of plain .npy files opened with mmap_mode='r', so the page
cache does the caching for us and no decoding happens during training.

Layout
------
    data/cache/{train,val}.npy          (N, T, H, W, 3) uint8   H=W=72 (crop-augmented at train time)
    data/cache/{train,val}_labels.npy   (N,) int64
    data/cache/classes.txt              one class name per line
    data/cache/filter_report.json       thresholds, drop counts, stat percentiles
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

cv2.setNumThreads(1)


# --------------------------------------------------------------------------- #
# decoding + filtering
# --------------------------------------------------------------------------- #
def decode_clip(path: str, num_frames: int, size: int = 72):
    """Uniformly sample `num_frames` frames from a video, resize the shorter
    side to `size` and centre-crop to size x size.

    Returns (clip, stats) where clip is (T, size, size, 3) uint8 and stats is
    {'frames': raw frame count, 'motion': mean |dt|, 'std': pixel std}; clip is
    None if the file cannot be decoded."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None, {"frames": 0}
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if len(frames) < 2:
        return None, {"frames": len(frames)}

    idx = np.linspace(0, len(frames) - 1, num_frames).round().astype(int)
    out = np.empty((num_frames, size, size, 3), dtype=np.uint8)
    for i, j in enumerate(idx):
        f = frames[j]
        h, w = f.shape[:2]
        scale = size / min(h, w)
        nh, nw = max(size, int(round(h * scale))), max(size, int(round(w * scale)))
        f = cv2.resize(f, (nw, nh), interpolation=cv2.INTER_AREA)
        y0, x0 = (nh - size) // 2, (nw - size) // 2
        out[i] = f[y0:y0 + size, x0:x0 + size]

    x = out.astype(np.float32)
    stats = {
        "frames": len(frames),
        "motion": float(np.abs(np.diff(x, axis=0)).mean()) if num_frames > 1 else 0.0,
        "std": float(x.std()),
    }
    return out, stats


def _worker(args):
    path, num_frames, size, min_std, min_motion = args
    try:
        clip, stats = decode_clip(path, num_frames, size)
    except Exception as exc:                      # corrupt container, codec error, ...
        return path, None, {"frames": 0}, f"decode_error:{type(exc).__name__}"
    if clip is None:
        return path, None, stats, "decode_failed"
    if stats["frames"] < num_frames:
        return path, clip, stats, "too_short"
    if stats["std"] < min_std:
        return path, clip, stats, "flat"
    if stats["motion"] < min_motion:
        return path, clip, stats, "no_motion"
    return path, clip, stats, None


def _pct(values, qs=(1, 5, 25, 50, 75, 95, 99)):
    if not values:
        return {}
    a = np.asarray(values, dtype=np.float64)
    return {f"p{q}": round(float(np.percentile(a, q)), 3) for q in qs}


def build_cache(video_root: str, out_dir: str, *, num_frames: int = 16, size: int = 72,
                workers: int = 16, test_list: str | None = None,
                min_std: float = 6.0, min_motion: float = 1.0,
                max_per_class: int = 0, log=print):
    """Decode, filter and cache every video under `video_root` (class sub-folders).

    `test_list`: optional UCF-101 *test* list ('Class/video.avi' per line); those
    entries form the held-out split.  Without it a deterministic 10% split is used.
    `max_per_class`: if > 0, keep at most that many clips per class in train+val
    (deterministic: first in sorted order), to flatten class imbalance.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(video_root)

    test_set = set()
    if test_list and Path(test_list).exists():
        for line in Path(test_list).read_text().splitlines():
            line = line.strip()
            if line:
                test_set.add(line)

    classes = sorted([d.name for d in root.iterdir() if d.is_dir()])
    jobs, labels, is_test = [], [], []
    for ci, cname in enumerate(classes):
        vids = sorted((root / cname).glob("*.avi"))
        for v in vids:
            jobs.append((str(v), num_frames, size, min_std, min_motion))
            labels.append(ci)
            rel = f"{cname}/{v.name}"
            is_test.append(rel in test_set if test_set else (hash(rel) % 10 == 0))

    log(f"[cache] {len(jobs)} videos across {len(classes)} classes; decoding with {workers} workers")
    clips = [None] * len(jobs)
    stats = [None] * len(jobs)
    reasons = [None] * len(jobs)
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, (path, clip, st, reason) in enumerate(ex.map(_worker, jobs, chunksize=8)):
            clips[i], stats[i], reasons[i] = clip, st, reason
            done += 1
            if done % 1000 == 0:
                log(f"[cache]   decoded {done}/{len(jobs)}")

    # ---- filter report: what the raw distribution looked like before thresholding
    all_stats = [s for s in stats if s]
    report = {
        "num_videos": len(jobs),
        "thresholds": {"min_std": min_std, "min_motion": min_motion,
                       "num_frames": num_frames, "max_per_class": max_per_class},
        "stats_percentiles": {
            "frames": _pct([s.get("frames", 0) for s in all_stats]),
            "motion": _pct([s.get("motion") for s in all_stats if "motion" in s]),
            "std": _pct([s.get("std") for s in all_stats if "std" in s]),
        },
    }
    drop_counts: dict[str, int] = {}
    for r in reasons:
        if r:
            drop_counts[r] = drop_counts.get(r, 0) + 1
    report["dropped"] = drop_counts
    log(f"[cache] filter thresholds: std>={min_std}, motion>={min_motion}, frames>={num_frames}")
    log(f"[cache] stat percentiles: frames {report['stats_percentiles']['frames']}")
    log(f"[cache]                   motion {report['stats_percentiles']['motion']}")
    log(f"[cache]                   std    {report['stats_percentiles']['std']}")
    log(f"[cache] dropped: {drop_counts if drop_counts else 'nothing'}")

    # ---- keep survivors
    keep = [i for i, r in enumerate(reasons) if r is None]
    kept_labels = np.array([labels[i] for i in keep], dtype=np.int64)
    kept_is_test = np.array([is_test[i] for i in keep], dtype=bool)

    if max_per_class > 0:
        sel = []
        for c in range(len(classes)):
            idx = np.nonzero(kept_labels == c)[0]
            sel.extend(idx[:max_per_class].tolist())
        sel = np.array(sorted(sel), dtype=np.int64)
        report["capped_from"] = int(len(keep))
        keep = [keep[i] for i in sel]
        kept_labels = kept_labels[sel]
        kept_is_test = kept_is_test[sel]
        log(f"[cache] capped to {max_per_class}/class: {len(keep)} clips kept")

    log(f"[cache] kept {len(keep)}/{len(jobs)} clips "
        f"({100 * len(keep) / max(1, len(jobs)):.1f}%)")

    report["kept"] = int(len(keep))
    report["train"] = int((~kept_is_test).sum())
    report["val"] = int(kept_is_test.sum())
    for name, mask in (("train", ~kept_is_test), ("val", kept_is_test)):
        idx = np.nonzero(mask)[0]
        if len(idx) == 0:
            raise ValueError(f"split '{name}' is empty -- check the split list / filters")
        arr = np.stack([clips[keep[i]] for i in idx])
        np.save(out / f"{name}.npy", arr)
        np.save(out / f"{name}_labels.npy", kept_labels[idx])
        log(f"[cache] {name}: {arr.shape}  ({arr.nbytes / 1e9:.2f} GB)")

    (out / "classes.txt").write_text("\n".join(classes))
    (out / "filter_report.json").write_text(json.dumps(report, indent=2))
    (out / "meta.json").write_text(json.dumps(
        {"num_frames": num_frames, "size": size, "num_classes": len(classes),
         "video_root": str(root), "test_list": test_list}, indent=2))
    log(f"[cache] report -> {out / 'filter_report.json'}")


# --------------------------------------------------------------------------- #
# training dataset
# --------------------------------------------------------------------------- #
class ClipDataset(Dataset):
    """Reads clips from the memmap cache.  Returns float32 (C, T, H, W) in [-1, 1].

    train mode: random crop from the stored `size` canvas (spatial augmentation).
    """

    def __init__(self, cache_dir: str, split: str, crop: int = 64, train: bool = True):
        self.data = np.load(os.path.join(cache_dir, f"{split}.npy"), mmap_mode="r")
        self.labels = np.load(os.path.join(cache_dir, f"{split}_labels.npy"))
        self.crop = crop
        self.train = train
        self.stored = self.data.shape[2]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        clip = np.asarray(self.data[i])                      # (T, H, W, 3) uint8
        c = self.crop
        if c < self.stored:
            if self.train:
                y0 = np.random.randint(0, self.stored - c + 1)
                x0 = np.random.randint(0, self.stored - c + 1)
            else:
                y0 = x0 = (self.stored - c) // 2
            clip = clip[:, y0:y0 + c, x0:x0 + c]
        x = torch.from_numpy(np.ascontiguousarray(clip)).permute(3, 0, 1, 2).float()
        x = x / 127.5 - 1.0
        return x, int(self.labels[i])


def load_classes(cache_dir: str) -> list[str]:
    return (Path(cache_dir) / "classes.txt").read_text().splitlines()
