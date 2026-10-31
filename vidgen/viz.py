"""Small visualisation helpers: video montages as browser-playable mp4.

Encoding note: browsers cannot decode the MPEG-4 Part 2 stream that OpenCV's
`mp4v` fourcc produces (Chromium reports DEMUXER_ERROR_NO_SUPPORTED_STREAMS).
The OpenCV wheels also ship no H.264 encoder, so we encode with PyAV/libx264
(yuv420p, even dimensions, +faststart) and keep OpenCV only as a last-resort
fallback for files that will be inspected offline.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

try:                       # PyAV is the preferred encoder
    import av
    HAVE_AV = True
except ImportError:        # pragma: no cover
    HAVE_AV = False


def _to_uint8(clip: torch.Tensor) -> np.ndarray:
    """(C, T, H, W) in [-1,1] -> (T, H, W, 3) uint8 RGB."""
    x = ((clip.float().clamp(-1, 1) + 1) * 127.5).round().byte()
    return x.permute(1, 2, 3, 0).cpu().numpy()


def _write_h264(frames_bgr: np.ndarray, path: str | Path, fps: int, crf: int = 20) -> bool:
    """Encode (T, H, W, 3) uint8 BGR frames as H.264/yuv420p mp4.  False if PyAV is absent."""
    if not HAVE_AV:
        return False
    T, H, W = frames_bgr.shape[:3]
    # yuv420p needs even dimensions
    if H % 2 or W % 2:
        pad_h, pad_w = H % 2, W % 2
        frames_bgr = np.pad(frames_bgr, ((0, 0), (0, pad_h), (0, pad_w), (0, 0)), constant_values=16)
        H, W = H + pad_h, W + pad_w

    container = av.open(str(path), mode="w")
    stream = container.add_stream("libx264", rate=fps)
    stream.width, stream.height = W, H
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": str(crf), "preset": "veryfast", "movflags": "+faststart"}
    try:
        for f in frames_bgr:
            for packet in stream.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(f), format="bgr24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()
    return True


def _write_mp4v(frames_bgr: np.ndarray, path: str | Path, fps: int) -> None:
    """OpenCV fallback (MPEG-4 Part 2; not playable in browsers)."""
    T, H, W = frames_bgr.shape[:3]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer for {path}")
    for f in frames_bgr:
        writer.write(f)
    writer.release()


def write_video(frames_bgr: np.ndarray, path: str | Path, fps: int = 8) -> str:
    """Write frames (T, H, W, 3) uint8 BGR to `path`; returns the codec used."""
    if _write_h264(frames_bgr, path, fps):
        return "h264"
    _write_mp4v(frames_bgr, path, fps)
    return "mp4v"


def save_video_grid(clips: torch.Tensor, path: str | Path, ncols: int = 4, fps: int = 8,
                    labels: list[str] | None = None, scale: int = 1) -> str:
    """Write a (B, C, T, H, W) batch of clips as one mp4 montage, B tiles laid out
    row-major with `ncols` columns; every tile plays its own frames in sync."""
    clips = clips.detach()
    B, C, T, H, W = clips.shape
    ncols = max(1, min(ncols, B))
    nrows = int(np.ceil(B / ncols))
    tiles = [_to_uint8(clips[i]) for i in range(B)]

    if scale != 1:
        tiles = [np.stack([cv2.resize(f, (W * scale, H * scale), interpolation=cv2.INTER_NEAREST)
                           for f in t]) for t in tiles]
        H, W = H * scale, W * scale

    pad = 2
    canvas_h = nrows * H + (nrows + 1) * pad
    canvas_w = ncols * W + (ncols + 1) * pad
    frames = np.empty((T, canvas_h, canvas_w, 3), dtype=np.uint8)

    for t in range(T):
        canvas = np.full((canvas_h, canvas_w, 3), 24, dtype=np.uint8)
        for i in range(B):
            r, c = divmod(i, ncols)
            y = pad + r * (H + pad)
            x = pad + c * (W + pad)
            canvas[y:y + H, x:x + W] = cv2.cvtColor(tiles[i][t], cv2.COLOR_RGB2BGR)
            if labels and i < len(labels):
                cv2.putText(canvas, labels[i][:24], (x + 4, y + 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        frames[t] = canvas

    return write_video(frames, path, fps)


def save_video(clip: torch.Tensor, path: str | Path, fps: int = 8) -> str:
    """Single clip -> mp4."""
    frames = np.stack([cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in _to_uint8(clip)])
    return write_video(frames, path, fps)


def save_frame_strip(clip: torch.Tensor, path: str | Path, every: int = 2) -> None:
    """One PNG with frames side by side (handy for quick inspection)."""
    frames = _to_uint8(clip)[::every]
    strip = np.concatenate([cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in frames], axis=1)
    cv2.imwrite(str(path), strip)


def transcode_to_h264(src: str | Path, dst: str | Path | None = None, fps: int | None = None) -> str:
    """Re-encode an existing mp4 (e.g. an old mp4v montage) as browser-playable H.264."""
    src, dst = Path(src), Path(dst) if dst else Path(src).with_suffix(".h264.mp4")
    if not HAVE_AV:
        raise RuntimeError("PyAV is required for transcoding")
    with av.open(str(src)) as inp:
        stream_in = inp.streams.video[0]
        rate = fps or int(round(float(stream_in.average_rate or 8)))
        frames = [f.to_ndarray(format="bgr24") for f in inp.decode(stream_in)]
    _write_h264(np.stack(frames), dst, rate)
    return str(dst)
