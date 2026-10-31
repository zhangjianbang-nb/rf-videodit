"""Evaluate a checkpoint: generate clips for a set of classes, compare against
held-out real clips with FVD, and write side-by-side montages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .data import ClipDataset, load_classes
from .flow import sample
from .metrics import FeatureExtractor, frechet_distance, pixel_stats
from .sample import load_model
from .viz import save_video_grid


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--cache-dir", default="data/cache")
    p.add_argument("--out-dir", default="")
    p.add_argument("--n-classes", type=int, default=32)
    p.add_argument("--per-class", type=int, default=8)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--solver", default="heun")
    p.add_argument("--cfg-scale", type=float, default=4.0)
    p.add_argument("--gen-batch", type=int, default=16)
    p.add_argument("--frames", type=int, default=16)
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


@torch.no_grad()
def generate(model, y, *, args, device):
    g = torch.Generator(device=device).manual_seed(args.seed)
    outs = []
    for i in range(0, len(y), args.gen_batch):
        yb = y[i:i + args.gen_batch]
        x0 = torch.randn((len(yb), 3, args.frames, args.size, args.size), device=device, generator=g)
        outs.append(sample(model, x0.shape, yb, steps=args.steps, cfg_scale=args.cfg_scale,
                           solver=args.solver, device=device, x0=x0))
    return torch.cat(outs)


def main():
    args = parse_args()
    device = torch.device(args.device)
    model, train_args, step = load_model(args.ckpt, device, use_ema=True)
    classes = load_classes(args.cache_dir)
    out_dir = Path(args.out_dir or (Path(args.ckpt).parent / "eval"))
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    class_ids = rng.choice(len(classes), size=min(args.n_classes, len(classes)), replace=False)
    class_ids.sort()
    y = torch.tensor(np.repeat(class_ids, args.per_class), device=device)
    labels = [classes[c] for c in class_ids for _ in range(args.per_class)]
    print(f"[eval] generating {len(y)} clips for {len(class_ids)} classes "
          f"(steps={args.steps}, cfg={args.cfg_scale}, solver={args.solver})")

    gen = generate(model, y, args=args, device=device)

    # real clips of the same classes, from the held-out split
    val = ClipDataset(args.cache_dir, "val", crop=args.size, train=False)
    val_labels = np.load(str(Path(args.cache_dir) / "val_labels.npy"))
    real_idx = []
    for c in class_ids:
        idx = np.nonzero(val_labels == c)[0]
        take = rng.choice(idx, size=min(args.per_class, len(idx)), replace=False)
        real_idx.extend(take.tolist())
    real = torch.stack([val[i][0] for i in real_idx])
    real_labels = [classes[int(val_labels[i])] for i in real_idx]
    print(f"[eval] comparing against {len(real)} held-out real clips")

    extractor = FeatureExtractor(device=args.device)
    f_gen = extractor.features(gen)
    f_real = extractor.features(real)
    fvd = frechet_distance(f_gen, f_real)

    # noise floor: FVD between two halves of the real set (finite-sample bias)
    half = len(f_real) // 2
    floor = frechet_distance(f_real[:half], f_real[half:])

    results = {
        "checkpoint": args.ckpt, "step": step, "n_gen": int(len(gen)), "n_real": int(len(real)),
        "fvd": fvd, "fvd_floor_real_vs_real": floor,
        "steps": args.steps, "cfg_scale": args.cfg_scale, "solver": args.solver,
        "gen_pixel_stats": pixel_stats(gen), "real_pixel_stats": pixel_stats(real),
        "classes": [classes[c] for c in class_ids],
    }
    (out_dir / "fvd.json").write_text(json.dumps(results, indent=2))

    ncols = 8
    save_video_grid(gen[:64], out_dir / "generated.mp4", ncols=ncols, fps=8, labels=labels[:64])
    save_video_grid(real[:64], out_dir / "real.mp4", ncols=ncols, fps=8, labels=real_labels[:64])

    print(f"[eval] FVD = {fvd:.2f}   (real-vs-real noise floor {floor:.2f})")
    print(f"[eval] generated pixel stats: {results['gen_pixel_stats']}")
    print(f"[eval] real      pixel stats: {results['real_pixel_stats']}")
    print(f"[eval] wrote {out_dir}/generated.mp4, {out_dir}/real.mp4, {out_dir}/fvd.json")


if __name__ == "__main__":
    main()
