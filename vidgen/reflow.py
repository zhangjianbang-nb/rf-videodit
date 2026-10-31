"""Optional stage 2: reflow.  Straighten the learned ODE by retraining on
(x0, ODE(x0)) pairs produced by the stage-1 model, so sampling needs far fewer
steps.  Usage mirrors train.py but reads a stage-1 checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import ClipDataset
from .flow import sample, sample_reflow_pairs
from .model import EMA
from .sample import load_model
from .viz import save_video_grid


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="stage-1 checkpoint to straighten")
    p.add_argument("--cache-dir", default="data/cache")
    p.add_argument("--crop", type=int, default=64)
    p.add_argument("--steps-gen", type=int, default=25, help="ODE steps used to build pairs")
    p.add_argument("--train-steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--cfg-scale", type=float, default=4.0)
    p.add_argument("--run-dir", default="runs/reflow")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda")
    model, _, step = load_model(args.ckpt, device, use_ema=True)
    print(f"[reflow] loaded stage-1 checkpoint at step {step}")

    ds = ClipDataset(args.cache_dir, "train", crop=args.crop, train=True)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95))
    ema = EMA(model, decay=0.999)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    data_iter = iter(dl)
    for i in range(1, args.train_steps + 1):
        try:
            x1, y = next(data_iter)
        except StopIteration:
            data_iter = iter(dl)
            x1, y = next(data_iter)
        x1, y = x1.to(device), y.to(device)
        # pairs: start from the same noise the model would use at sampling time
        x0 = torch.randn_like(x1)
        x1_pred = sample_reflow_pairs(model, x1, y, steps=args.steps_gen,
                                      cfg_scale=args.cfg_scale, device=device)[1]
        t = torch.rand(x1.shape[0], device=device)
        t_ = t.view(-1, 1, 1, 1, 1)
        x_t = (1 - t_) * x0 + t_ * x1_pred
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = model(x_t, t, y)
            loss = ((pred - (x1_pred - x0)) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        ema.update(model)
        if i % 50 == 0:
            print(f"[reflow] step {i}/{args.train_steps} loss {loss.item():.4f}", flush=True)

    torch.save({"model": model.state_dict(), "ema": ema.shadow, "step": step,
                "stage": "reflow", "args": vars(args)}, run_dir / "ckpt_reflow.pt")
    print(f"[reflow] saved {run_dir}/ckpt_reflow.pt")

    ema.copy_to(model)
    g = torch.Generator(device=device).manual_seed(0)
    y = torch.arange(8, device=device)
    x0 = torch.randn((8, 3, 16, args.crop, args.crop), device=device, generator=g)
    clips = sample(model, x0.shape, y, steps=6, cfg_scale=args.cfg_scale,
                   solver="euler", device=device, x0=x0)
    save_video_grid(clips, run_dir / "reflow_6step.mp4", ncols=4, fps=8)
    print(f"[reflow] wrote {run_dir}/reflow_6step.mp4 (6-step samples)")


if __name__ == "__main__":
    main()
