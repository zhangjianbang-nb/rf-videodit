"""Local inference web app for a trained RF-VideoDiT checkpoint.

Standard library only (no FastAPI/Flask needed): a threaded HTTP server that
serves a single-page UI, accepts generation jobs, runs them one at a time on the
GPU, and streams back an mp4 montage.

    python -m vidgen.webapp --ckpt runs/base/ckpt_final.pt --port 8093
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import torch

from .data import load_classes
from .flow import sample
from .sample import load_model
from .viz import save_video_grid

HERE = Path(__file__).resolve().parent
STATIC = HERE.parent / "webapp"

STATE: dict = {}
JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()
GPU_LOCK = threading.Lock()


# --------------------------------------------------------------------------- #
# job execution
# --------------------------------------------------------------------------- #
def run_job(job_id: str, params: dict):
    job = JOBS[job_id]
    try:
        job.update(status="running", progress=0.0, started=time.time())
        cfg = STATE["config"]
        device = STATE["device"]
        model = STATE["model"]

        n = int(params.get("num_samples", 4))
        frames = int(params.get("frames", cfg["frames"]))
        size = int(params.get("size", cfg["size"]))
        steps = int(params.get("steps", cfg["steps"]))
        cfg_scale = float(params.get("cfg_scale", cfg["cfg_scale"]))
        solver = str(params.get("solver", "heun"))
        seed = int(params.get("seed", 0))
        cls = params.get("class_id", None)

        if cls in ("", None):
            y = None
            labels = None
        else:
            y = torch.full((n,), int(cls), device=device, dtype=torch.long)
            labels = [STATE["classes"][int(cls)] for _ in range(n)]

        g = torch.Generator(device=device).manual_seed(seed)
        x0 = torch.randn((n, 3, frames, size, size), device=device, generator=g)

        def progress(i, total):
            job["progress"] = i / total

        with GPU_LOCK:
            clips = sample(model, x0.shape, y, steps=steps, cfg_scale=cfg_scale,
                           solver=solver, device=device, x0=x0, progress=progress)

        out = STATE["out_dir"] / f"{job_id}.mp4"
        save_video_grid(clips, out, ncols=min(4, n), fps=int(params.get("fps", 8)), labels=labels)
        job.update(status="done", progress=1.0, video=f"/videos/{out.name}",
                   elapsed=time.time() - job["started"])
        del clips, x0
        torch.cuda.empty_cache()
    except Exception as exc:                      # surface failures to the UI
        job.update(status="error", error=f"{type(exc).__name__}: {exc}")


class Handler(BaseHTTPRequestHandler):
    server_version = "RFVideoDiT/0.1"

    def log_message(self, fmt, *args):            # keep the console readable
        if "/api/job" not in self.path:
            print(f"[web] {self.address_string()} {fmt % args}", flush=True)

    # ---- helpers
    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode(), "application/json; charset=utf-8")

    # ---- routes
    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            html = (STATIC / "index.html").read_bytes()
            self._send(200, html, "text/html; charset=utf-8")
        elif path == "/api/meta":
            self._json(STATE["meta"])
        elif path == "/api/job":
            qs = parse_qs(urlparse(self.path).query)
            jid = (qs.get("id") or [""])[0]
            with JOBS_LOCK:
                job = JOBS.get(jid)
            self._json(job or {"status": "unknown"}, 200 if job else 404)
        elif path.startswith("/videos/"):
            name = path.split("/")[-1]
            f = STATE["out_dir"] / name
            if f.exists() and f.suffix == ".mp4":
                self._send(200, f.read_bytes(), "video/mp4")
            else:
                self._json({"error": "not found"}, 404)
        elif path == "/api/health":
            self._json({"ok": True, "jobs": len(JOBS)})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        path = urlparse(self.path).path
        if path != "/api/generate":
            self._json({"error": "not found"}, 404)
            return
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            params = json.loads(raw.decode() or "{}")
        except json.JSONDecodeError:
            self._json({"error": "invalid json"}, 400)
            return
        jid = uuid.uuid4().hex[:12]
        with JOBS_LOCK:
            JOBS[jid] = {"status": "queued", "progress": 0.0, "params": params}
        threading.Thread(target=run_job, args=(jid, params), daemon=True).start()
        self._json({"job_id": jid})


def build_state(args):
    device = torch.device("cuda")
    model, train_args, step = load_model(args.ckpt, device, use_ema=not args.no_ema)
    classes = load_classes(args.cache_dir)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "checkpoint": args.ckpt,
        "trained_steps": step,
        "num_classes": len(classes),
        "classes": classes,
        "params_millions": round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
        "defaults": {"frames": args.frames, "size": args.size, "steps": args.steps,
                     "cfg_scale": args.cfg_scale, "num_samples": 4, "solver": "heun"},
        "notes": args.notes,
    }
    STATE.update(model=model, device=device, classes=classes, meta=meta,
                 config=meta["defaults"], out_dir=out_dir)
    print(f"[web] loaded checkpoint {args.ckpt} (step {step}), {len(classes)} classes")
    return meta


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="runs/base/ckpt_final.pt")
    p.add_argument("--cache-dir", default="data/cache")
    p.add_argument("--out-dir", default="webapp/videos")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8093)
    p.add_argument("--frames", type=int, default=16)
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--steps", type=int, default=25)
    p.add_argument("--cfg-scale", type=float, default=3.0)
    p.add_argument("--no-ema", action="store_true")
    p.add_argument("--notes", default="")
    return p.parse_args()


def main():
    args = parse_args()
    build_state(args)
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[web] serving on http://{args.host}:{args.port}  (Ctrl+C to stop)", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
