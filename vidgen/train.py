"""Training driver for RF-VideoDiT.

Features: bf16 autocast, gradient accumulation, cosine LR with warmup, EMA
weights, resumable checkpoints, periodic sample previews and a CSV log.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import signal
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import ClipDataset, load_classes
from .flow import flow_matching_loss, sample
from .model import EMA, RFVideoDiTConfig, build_model
from .viz import save_video_grid


def parse_args():
    p = argparse.ArgumentParser()
    # data
    p.add_argument("--cache-dir", default="data/cache")
    p.add_argument("--crop", type=int, default=64)
    p.add_argument("--frames", type=int, default=16)
    # model
    p.add_argument("--hidden", type=int, default=384)
    p.add_argument("--depth", type=int, default=12)
    p.add_argument("--heads", type=int, default=6)
    p.add_argument("--patch", type=str, default="2,4,4")
    p.add_argument("--mlp-ratio", type=float, default=4.0)
    p.add_argument("--cfg-dropout", type=float, default=0.1)
    p.add_argument("--grad-checkpoint", action="store_true",
                   help="recompute blocks in backward (slower, ~40%% less activation memory)")
    # optimisation
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--t-mode", default="logit_normal", choices=["logit_normal", "uniform"])
    p.add_argument("--loss-weight", default="none", choices=["none", "snr"])
    # schedule / io
    p.add_argument("--steps", type=int, default=200_000)
    p.add_argument("--max-hours", type=float, default=0.0, help="0 = no time limit")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--ckpt-every", type=int, default=2000)
    p.add_argument("--ckpt-keep", type=int, default=8,
                   help="how many numbered checkpoints to retain (for rollback after a loss spike)")
    p.add_argument("--sample-every", type=int, default=5000)
    p.add_argument("--sample-count", type=int, default=8)
    p.add_argument("--sample-steps", type=int, default=25)
    p.add_argument("--cfg-scale", type=float, default=4.0)
    p.add_argument("--run-dir", default="runs/base")
    p.add_argument("--resume", default="", help="'auto' or path to .pt")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--compile", action="store_true")
    return p.parse_args()


def lr_at(step: int, args) -> float:
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    prog = (step - args.warmup) / max(1, args.steps - args.warmup)
    return args.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))


def save_ckpt(path: Path, model, ema, opt, step, args, best_loss, keep_opt=True):
    tmp = path.with_suffix(".tmp")
    payload = {
        "model": model.state_dict(),
        "ema": ema.shadow if ema is not None else None,
        "step": step,
        "args": vars(args),
        "best_loss": best_loss,
    }
    if keep_opt and opt is not None:
        payload["opt"] = opt.state_dict()
    torch.save(payload, tmp)
    os.replace(tmp, path)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    run_dir = Path(args.run_dir)
    (run_dir / "samples").mkdir(parents=True, exist_ok=True)

    # a stop flag lets us save a checkpoint before exiting on SIGINT/SIGTERM
    stop_flag = {"stop": False, "reason": "signal"}

    def _handler(signum, frame):
        stop_flag["stop"] = True
        print(f"[train] received signal {signum}; will checkpoint and exit", flush=True)

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)

    classes = load_classes(args.cache_dir)
    num_classes = len(classes)
    args.num_classes = num_classes          # persisted into every checkpoint
    print(f"[train] {num_classes} classes: {classes[:5]} ...")

    train_ds = ClipDataset(args.cache_dir, "train", crop=args.crop, train=True)
    val_ds = ClipDataset(args.cache_dir, "val", crop=args.crop, train=False)
    print(f"[train] train clips {len(train_ds)}, val clips {len(val_ds)}")

    dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
                    num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0)

    cfg = RFVideoDiTConfig(
        num_classes=num_classes,
        patch=tuple(int(x) for x in args.patch.split(",")),
        hidden=args.hidden, depth=args.depth, heads=args.heads,
        mlp_ratio=args.mlp_ratio, label_dropout=args.cfg_dropout,
        grad_checkpoint=args.grad_checkpoint,
    )
    model = build_model(**vars(cfg)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] RF-VideoDiT: {n_params / 1e6:.1f}M params, patch={cfg.patch}, hidden={cfg.hidden}, depth={cfg.depth}")

    ema = EMA(model, decay=args.ema_decay)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=args.weight_decay)
    start_step, best_loss = 0, float("inf")

    if args.resume:
        ckpt_path = None
        if args.resume == "auto":
            cands = sorted(run_dir.glob("ckpt_*.pt"))
            if cands:
                ckpt_path = cands[-1]
        else:
            ckpt_path = Path(args.resume)
        if ckpt_path and ckpt_path.exists():
            print(f"[train] resuming from {ckpt_path}")
            ck = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ck["model"])
            if ck.get("ema") is not None:
                ema.shadow = {k: v.to(device).float() for k, v in ck["ema"].items()}
            if "opt" in ck:
                opt.load_state_dict(ck["opt"])
            start_step = ck["step"]
            best_loss = ck.get("best_loss", float("inf"))
        else:
            print("[train] no checkpoint found, starting fresh")

    # a second model instance used only for previews, so training weights are untouched
    preview_model = build_model(**vars(cfg)).to(device).eval()

    log_path = run_dir / "log.csv"
    new_log = not log_path.exists() or start_step == 0
    log_file = open(log_path, "a", newline="")
    writer = csv.writer(log_file)
    if new_log:
        writer.writerow(["step", "loss", "lr", "sec_per_step", "samples_per_sec", "elapsed_min"])

    data_iter = iter(dl)
    t_start = time.time()
    running = torch.zeros((), device=device)
    running_n = 0
    last_log = t_start

    def next_batch():
        nonlocal data_iter
        try:
            return next(data_iter)
        except StopIteration:
            data_iter = iter(dl)
            return next(data_iter)

    step = start_step
    stop_reason = "max steps reached"
    ckpt_loss_sum, ckpt_loss_n = 0.0, 0          # accumulate loss over a checkpoint window
    best_window_loss = float("inf")
    while step < args.steps:
        lr = lr_at(step, args)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        micro_losses = []
        for _ in range(args.accum):
            x, y = next_batch()
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = flow_matching_loss(model, x, y, t_mode=args.t_mode, weight=args.loss_weight)
            (loss / args.accum).backward()
            micro_losses.append(loss.detach())
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        ema.update(model)

        loss_val = torch.stack(micro_losses).mean().item()
        running += loss_val
        running_n += 1
        ckpt_loss_sum += loss_val
        ckpt_loss_n += 1
        step += 1

        if step % args.log_every == 0:
            now = time.time()
            sps = args.log_every * args.batch_size * args.accum / (now - last_log)
            avg = running.item() / running_n
            elapsed = (now - t_start) / 60
            print(f"[train] step {step}/{args.steps} loss {avg:.4f} lr {lr:.2e} "
                  f"{sps:.1f} samp/s elapsed {elapsed:.1f}m", flush=True)
            writer.writerow([step, f"{avg:.5f}", f"{lr:.3e}", f"{(now - last_log) / args.log_every:.3f}",
                             f"{sps:.2f}", f"{elapsed:.2f}"])
            log_file.flush()
            running = torch.zeros((), device=device)
            running_n = 0
            last_log = now

        if args.ckpt_every and step % args.ckpt_every == 0:
            best_loss = min(best_loss, loss_val)
            window_loss = ckpt_loss_sum / max(1, ckpt_loss_n)   # mean loss over this checkpoint window
            save_ckpt(run_dir / "ckpt_last.pt", model, ema, opt, step, args, best_loss)
            for old in sorted(run_dir.glob("ckpt_[0-9]*.pt"))[:-max(1, args.ckpt_keep)]:
                old.unlink()
            save_ckpt(run_dir / f"ckpt_{step:07d}.pt", model, ema, None, step, args, best_loss, keep_opt=False)
            # a rolling "best by training loss" copy: the checkpoint to fall back on
            # if a later loss spike damages the weights
            if window_loss < best_window_loss:
                best_window_loss = window_loss
                save_ckpt(run_dir / "ckpt_best.pt", model, ema, None, step, args, best_loss, keep_opt=False)
                print(f"[train] new best checkpoint at step {step} (window loss {window_loss:.4f})", flush=True)
            ckpt_loss_sum, ckpt_loss_n = 0.0, 0
            print(f"[train] checkpoint saved at step {step}", flush=True)

        if args.sample_every and step % args.sample_every == 0:
            preview_model.load_state_dict({k: v.to(preview_model.state_dict()[k].dtype)
                                           for k, v in ema.shadow.items()})
            gen_classes = torch.arange(min(args.sample_count, num_classes), device=device)
            g = torch.Generator(device=device).manual_seed(1234)
            x0 = torch.randn((len(gen_classes), 3, args.frames, args.crop, args.crop),
                             device=device, generator=g)
            clips = sample(preview_model, x0.shape, gen_classes, steps=args.sample_steps,
                           cfg_scale=args.cfg_scale, solver="heun", device=device, x0=x0)
            out = run_dir / "samples" / f"step_{step:07d}.mp4"
            save_video_grid(clips, out, ncols=min(4, len(gen_classes)), fps=8)
            labels = [classes[i] for i in gen_classes.tolist()]
            (run_dir / "samples" / f"step_{step:07d}.txt").write_text("\n".join(labels))
            print(f"[train] wrote preview {out} ({', '.join(labels)})", flush=True)
            del clips, x0
            torch.cuda.empty_cache()

        if args.max_hours and (time.time() - t_start) / 3600 >= args.max_hours:
            stop_reason = f"time limit {args.max_hours}h reached"
            break

        if stop_flag["stop"]:
            stop_reason = "stopped by signal"
            break

    best_loss = min(best_loss, loss_val)
    save_ckpt(run_dir / "ckpt_last.pt", model, ema, opt, step, args, best_loss)
    save_ckpt(run_dir / f"ckpt_{step:07d}.pt", model, ema, None, step, args, best_loss, keep_opt=False)
    save_ckpt(run_dir / "ckpt_final.pt", model, ema, None, step, args, best_loss, keep_opt=False)
    (run_dir / "DONE").write_text(json.dumps(
        {"step": step, "best_loss": best_loss, "reason": stop_reason,
         "minutes": (time.time() - t_start) / 60}, indent=2))
    print(f"[train] finished at step {step} ({stop_reason}); total {(time.time() - t_start) / 60:.1f} min")
    print(f"[train] weights saved to {run_dir}/ckpt_final.pt (model + ema)")
    log_file.close()


if __name__ == "__main__":
    main()
