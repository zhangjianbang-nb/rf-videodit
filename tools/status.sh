#!/usr/bin/env bash
# Compact training status: loss trend, throughput, ETA, GPU, checkpoints.
set -euo pipefail
cd "$(dirname "$0")/.."

python3 - <<'PY'
import csv, json, os, time
from pathlib import Path

run = Path("runs/base")
rows = list(csv.DictReader(open(run / "log.csv")))
steps = [float(r["step"]) for r in rows]
loss = [float(r["loss"]) for r in rows]
lr = [float(r["lr"]) for r in rows]
sps = [float(r["samples_per_sec"]) for r in rows]
mins = [float(r["elapsed_min"]) for r in rows]

step = int(steps[-1]); cur = loss[-1]
# mean of the last 10 logged points (~500 steps) and of the previous 10
recent = sum(loss[-10:]) / min(10, len(loss))
prev = sum(loss[-20:-10]) / max(1, len(loss[-20:-10]))
print(f"step        {step}")
print(f"loss        {cur:.4f}   (last 500-step mean {recent:.4f}, previous {prev:.4f}, "
      f"delta {recent - prev:+.4f})")
print(f"best loss   {min(loss):.4f} at step {int(steps[loss.index(min(loss))])}")
print(f"lr          {lr[-1]:.3e}")
print(f"throughput  {sps[-1]:.1f} samples/s")
print(f"elapsed     {mins[-1]:.1f} min")

done = run / "DONE"
if done.exists():
    d = json.loads(done.read_text())
    print(f"status      finished at step {d['step']} ({d['reason']}), {d['minutes']:.1f} min")
else:
    ck = sorted(run.glob("ckpt_[0-9]*.pt"))
    print(f"status      running; checkpoints {[c.name for c in ck]}")
    prevs = sorted(run.glob("samples/*.mp4"))
    if prevs:
        print(f"previews    {len(prevs)} (latest {prevs[-1].name})")

# rough ETA from the log itself (steps/min over the last logged window)
if len(steps) > 1:
    dsteps = steps[-1] - steps[0]
    dmins = mins[-1] - mins[0]
    if dmins > 0:
        print(f"rate        {dsteps / dmins:.0f} steps/min (over this process's log)")
PY

nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | sed 's/^/gpu         /'
