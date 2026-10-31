"""Evaluation metrics for generated video: FVD (Fréchet Video Distance) with an
R(2+1)D-18 feature extractor pretrained on Kinetics-400, plus simple pixel
statistics.  FVD is the standard video-generation metric: lower is better."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights

KINETICS_MEAN = torch.tensor([0.43216, 0.394666, 0.37645]).view(1, 3, 1, 1, 1)
KINETICS_STD = torch.tensor([0.22803, 0.22145, 0.216989]).view(1, 3, 1, 1, 1)


class FeatureExtractor:
    """R(2+1)D-18 trunk (up to the 512-d pooled embedding) as an FVD feature net."""

    def __init__(self, device="cuda", size: int = 112):
        self.device = torch.device(device)
        weights = R2Plus1D_18_Weights.KINETICS400_V1
        net = r2plus1d_18(weights=weights)
        net.fc = torch.nn.Identity()
        self.net = net.to(self.device).eval()
        self.size = size
        self.mean = KINETICS_MEAN.to(self.device)
        self.std = KINETICS_STD.to(self.device)

    @torch.no_grad()
    def features(self, clips: torch.Tensor, batch: int = 16) -> np.ndarray:
        """clips: (N, 3, T, H, W) in [-1, 1] -> (N, 512) float64 features."""
        feats = []
        for i in range(0, len(clips), batch):
            x = clips[i:i + batch].to(self.device).float()
            if x.shape[-1] != self.size:
                # resize spatially only: fold time into the batch dimension
                B, C, T, H, W = x.shape
                x = F.interpolate(x.reshape(B * T, C, H, W), size=(self.size, self.size),
                                  mode="bilinear", align_corners=False).reshape(B, C, T, self.size, self.size)
            x = (x - self.mean) / self.std
            feats.append(self.net(x).float().cpu().numpy())
        return np.concatenate(feats, axis=0).astype(np.float64)


def _sqrtm_psd(mat: np.ndarray) -> np.ndarray:
    """Symmetric PSD square root via eigendecomposition."""
    vals, vecs = np.linalg.eigh(mat)
    vals = np.clip(vals, 0, None)
    return (vecs * np.sqrt(vals)) @ vecs.T


def frechet_distance(feat_a: np.ndarray, feat_b: np.ndarray) -> float:
    mu_a, mu_b = feat_a.mean(0), feat_b.mean(0)
    ca = np.cov(feat_a, rowvar=False)
    cb = np.cov(feat_b, rowvar=False)
    ca_sqrt = _sqrtm_psd(ca)
    prod = ca_sqrt @ cb @ ca_sqrt
    tr = np.trace(_sqrtm_psd(prod))
    return float(np.sum((mu_a - mu_b) ** 2) + np.trace(ca) + np.trace(cb) - 2 * tr)


def pixel_stats(clips: torch.Tensor) -> dict:
    """Cheap sanity numbers: brightness spread and frame-to-frame motion."""
    x = clips.float()
    return {
        "mean": x.mean().item(),
        "std": x.std().item(),
        "temporal_diff": (x[:, :, 1:] - x[:, :, :-1]).abs().mean().item(),
    }
