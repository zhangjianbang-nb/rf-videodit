"""Sampling / inference CLI for a trained RF-VideoDiT checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .data import load_classes
from .flow import sample
from .model import build_model
from .viz import save_frame_strip, save_video_grid


def load_model(ckpt_path: str, device="cuda", use_ema: bool = True):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ck["args"]
    cfg_kwargs = dict(
        num_classes=args.get("num_classes", 101),
        patch=tuple(int(x) for x in args["patch"].split(",")),
        hidden=args["hidden"], depth=args["depth"], heads=args["heads"],
        mlp_ratio=args["mlp_ratio"], label_dropout=0.0,
    )
    model = build_model(**cfg_kwargs).to(device).eval()
    if use_ema and ck.get("ema") is not None:
        sd = {k: v.to(device).float() for k, v in ck["ema"].items()}
    else:
        sd = ck["model"]
    model.load_state_dict({k: v.to(model.state_dict()[k].dtype) for k, v in sd.items()})
    return model, args, ck["step"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--cache-dir", default="data/cache")
    p.add_argument("--out", default="samples/out.mp4")
    p.add_argument("--classes", default="", help="comma-separated class ids or names; empty = unconditional")
    p.add_argument("--per-class", type=int, default=2)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--solver", default="heun", choices=["euler", "heun"])
    p.add_argument("--cfg-scale", type=float, default=4.0)
    p.add_argument("--frames", type=int, default=16)
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-ema", action="store_true")
    p.add_argument("--also-strip", action="store_true", help="also write a PNG frame strip of sample 0")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda")
    model, train_args, step = load_model(args.ckpt, device, use_ema=not args.no_ema)
    classes = load_classes(args.cache_dir)
    print(f"[sample] checkpoint step {step}, ema={not args.no_ema}")

    if args.classes.strip():
        ids = []
        for tok in args.classes.split(","):
            tok = tok.strip()
            ids.append(int(tok) if tok.isdigit() else classes.index(tok))
        y = torch.tensor(ids, device=device).repeat_interleave(args.per_class)
        labels = [classes[i] for i in ids for _ in range(args.per_class)]
    else:
        y = None
        labels = None
        n = 8
        args.per_class = n

    n = y.shape[0] if y is not None else args.per_class
    g = torch.Generator(device=device).manual_seed(args.seed)
    x0 = torch.randn((n, 3, args.frames, args.size, args.size), device=device, generator=g)

    def progress(i, total):
        if i % max(1, total // 5) == 0:
            print(f"[sample] {i}/{total} steps", flush=True)

    clips = sample(model, x0.shape, y, steps=args.steps, cfg_scale=args.cfg_scale,
                   solver=args.solver, device=device, x0=x0, progress=progress)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_video_grid(clips, out, ncols=4, fps=args.fps, labels=labels)
    print(f"[sample] wrote {out} ({n} clips, {args.steps} steps, cfg {args.cfg_scale})")
    if args.also_strip:
        save_frame_strip(clips[0], out.with_suffix(".png"))
        print(f"[sample] wrote {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
