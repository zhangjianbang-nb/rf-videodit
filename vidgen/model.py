"""RF-VideoDiT: rectified-flow video diffusion transformer in pixel space.

Design notes
------------
* A video clip is a tensor x in R^{C x T x H x W}.  It is patchified with a
  3D convolution (kernel == stride == (pt, ph, pw)), giving a token grid of
  shape (T/pt, H/ph, W/ph) laid out frame-major.
* Every block factorises attention into two cheaper passes:
    - *spatial* attention: all tokens of one frame attend to each other
      (span S = (H/ph)*(W/ph) per frame),
    - *temporal* attention: the tokens at the same spatial location attend
      across frames (span F = T/pt).
  This costs O(F*S^2 + S*F^2) instead of O((F*S)^2) for full 3D attention.
* Positional information comes only from factorised 3D RoPE: a 2D RoPE on the
  (row, col) grid inside a frame and a 1D RoPE on the frame index.  Because the
  encoding is relative, the same weights can be sampled at a different number
  of frames than were seen in training.
* Timestep and class conditioning enter through AdaLN-Zero: shift/scale for
  each sub-layer and a zero-initialised residual gate, so the network starts
  out as the identity (predicts zero velocity), which stabilises training.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# positional encoding: factorised 3D RoPE
# --------------------------------------------------------------------------- #
def rope_freqs(positions: torch.Tensor, dim: int, base: float = 10000.0):
    """Cos/sin tables for `dim//2` rotary pairs at the given positions.

    positions: (N,) integer positions.  Returns (cos, sin) each of (N, dim//2).
    """
    half = dim // 2
    inv = 1.0 / (base ** (torch.arange(0, half, dtype=torch.float32, device=positions.device) / half))
    ang = positions.float()[:, None] * inv[None, :]           # (N, half)
    return ang.cos(), ang.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate even/odd dimension pairs of x by (cos, sin).

    x:   (..., N, D) where D = 2 * cos.shape[-1]
    cos/sin: (N, D//2) -- broadcasts over leading dims.
    """
    d = cos.shape[-1]
    x = x.view(*x.shape[:-1], d, 2)
    x_even, x_odd = x.unbind(-1)
    cos = cos.view(*([1] * (x.dim() - 3)), cos.shape[0], d)
    sin = sin.view(*([1] * (x.dim() - 3)), sin.shape[0], d)
    out = torch.stack([x_even * cos - x_odd * sin, x_even * sin + x_odd * cos], dim=-1)
    return out.flatten(-2)


def spatial_rope(grid_h: int, grid_w: int, head_dim: int, device, base: float = 10000.0):
    """2D RoPE over an (h, w) grid: first half of the rotary pairs encodes h,
    the second half encodes w."""
    half = head_dim // 2                       # number of rotary pairs
    quarter = half // 2
    hs = torch.arange(grid_h, device=device).repeat_interleave(grid_w)   # (S,)
    ws = torch.arange(grid_w, device=device).repeat(grid_h)              # (S,)
    inv = 1.0 / (base ** (torch.arange(0, quarter, dtype=torch.float32, device=device) / quarter))
    ang = torch.cat([hs.float()[:, None] * inv[None, :],
                     ws.float()[:, None] * inv[None, :]], dim=-1)        # (S, half)
    return ang.cos(), ang.sin()


def temporal_rope(num_frames: int, head_dim: int, device, base: float = 10000.0):
    """1D RoPE over the frame index."""
    pos = torch.arange(num_frames, device=device)
    return rope_freqs(pos, head_dim, base)


# --------------------------------------------------------------------------- #
# conditioning embeddings
# --------------------------------------------------------------------------- #
class TimestepEmbedder(nn.Module):
    def __init__(self, hidden: int, freq_dim: int = 256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(nn.Linear(freq_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B,) in [0, 1] -> (B, hidden)"""
        half = self.freq_dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
        ang = t.float()[:, None] * 1000.0 * freqs[None, :]
        emb = torch.cat([ang.sin(), ang.cos()], dim=-1)
        return self.mlp(emb.to(self.mlp[0].weight.dtype))


class LabelEmbedder(nn.Module):
    """Class embedding with a dedicated null slot used for classifier-free guidance."""

    def __init__(self, num_classes: int, hidden: int, dropout: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.dropout = dropout
        self.table = nn.Embedding(num_classes + 1, hidden)   # last row = null class

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        if self.training and self.dropout > 0:
            drop = torch.rand(y.shape[0], device=y.device) < self.dropout
            y = torch.where(drop, torch.full_like(y, self.num_classes), y)
        return self.table(y)

    @property
    def null_index(self) -> int:
        return self.num_classes


# --------------------------------------------------------------------------- #
# attention + block
# --------------------------------------------------------------------------- #
class Attention(nn.Module):
    def __init__(self, hidden: int, heads: int, qk_norm: bool = False):
        super().__init__()
        assert hidden % heads == 0
        self.heads = heads
        self.head_dim = hidden // heads
        self.qkv = nn.Linear(hidden, 3 * hidden)
        self.proj = nn.Linear(hidden, hidden)
        # QK-norm bounds the attention logits.  Without it a single unlucky batch
        # can produce a huge gradient and Adam then takes a full-size step in a bad
        # direction -- the loss spikes observed at steps 37.2k and 88.4k.  RMSNorm
        # initialises to identity, so it can be added to an already-trained model.
        self.q_norm = nn.RMSNorm(self.head_dim) if qk_norm else None
        self.k_norm = nn.RMSNorm(self.head_dim) if qk_norm else None

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """x: (B, N, D); cos/sin: (N, head_dim//2)."""
        B, N, D = x.shape
        qkv = self.qkv(x).view(B, N, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)                              # (B, H, N, Dh)
        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        out = F.scaled_dot_product_attention(q, k, v)        # flash / mem-efficient
        out = out.transpose(1, 2).reshape(B, N, D)
        return self.proj(out)


class Block(nn.Module):
    """Factorised spatial + temporal attention block with AdaLN-Zero."""

    def __init__(self, hidden: int, heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(hidden, elementwise_affine=False)
        self.spatial = Attention(hidden, heads)
        self.temporal = Attention(hidden, heads)
        self.mlp = nn.Sequential(nn.Linear(hidden, int(hidden * mlp_ratio)), nn.GELU(),
                                 nn.Linear(int(hidden * mlp_ratio), hidden))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 9 * hidden))
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)

    def forward(self, x, c, spatial_cos, spatial_sin, temporal_cos, temporal_sin,
                num_frames: int, tokens_per_frame: int):
        B, N, D = x.shape
        (s1, sc1, g1, s2, sc2, g2, s3, sc3, g3) = self.ada(c).chunk(9, dim=-1)

        # ---- spatial attention: batch frames together
        h = self.norm1(x) * (1 + sc1.unsqueeze(1)) + s1.unsqueeze(1)
        h = h.view(B * num_frames, tokens_per_frame, D)
        h = self.spatial(h, spatial_cos, spatial_sin).view(B, N, D)
        x = x + g1.unsqueeze(1) * h

        # ---- temporal attention: batch spatial locations together
        h = self.norm2(x) * (1 + sc2.unsqueeze(1)) + s2.unsqueeze(1)
        h = h.view(B, num_frames, tokens_per_frame, D).transpose(1, 2).reshape(B * tokens_per_frame, num_frames, D)
        h = self.temporal(h, temporal_cos, temporal_sin)
        h = h.view(B, tokens_per_frame, num_frames, D).transpose(1, 2).reshape(B, N, D)
        x = x + g2.unsqueeze(1) * h

        # ---- mlp
        h = self.norm3(x) * (1 + sc3.unsqueeze(1)) + s3.unsqueeze(1)
        x = x + g3.unsqueeze(1) * self.mlp(h)
        return x


# --------------------------------------------------------------------------- #
# the model
# --------------------------------------------------------------------------- #
@dataclass
class RFVideoDiTConfig:
    in_channels: int = 3
    num_classes: int = 101
    patch: tuple = (2, 4, 4)          # (temporal, height, width)
    hidden: int = 384
    depth: int = 12
    heads: int = 6
    mlp_ratio: float = 4.0
    rope_base: float = 10000.0
    label_dropout: float = 0.1
    grad_checkpoint: bool = False     # recompute blocks in backward to save memory


class RFVideoDiT(nn.Module):
    def __init__(self, cfg: RFVideoDiTConfig):
        super().__init__()
        self.cfg = cfg
        pt, ph, pw = cfg.patch
        self.pt, self.ph, self.pw = pt, ph, pw
        self.hidden = cfg.hidden

        self.patch_embed = nn.Conv3d(cfg.in_channels, cfg.hidden, kernel_size=cfg.patch, stride=cfg.patch)
        self.t_embed = TimestepEmbedder(cfg.hidden)
        self.y_embed = LabelEmbedder(cfg.num_classes, cfg.hidden, cfg.label_dropout)

        self.blocks = nn.ModuleList([Block(cfg.hidden, cfg.heads, cfg.mlp_ratio) for _ in range(cfg.depth)])
        self.norm_out = nn.LayerNorm(cfg.hidden, elementwise_affine=False)
        self.ada_out = nn.Sequential(nn.SiLU(), nn.Linear(cfg.hidden, 2 * cfg.hidden))
        nn.init.zeros_(self.ada_out[-1].weight)
        nn.init.zeros_(self.ada_out[-1].bias)
        self.head = nn.Linear(cfg.hidden, pt * ph * pw * cfg.in_channels)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv3d):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # re-apply the zero-init that makes the block a no-op at step 0
        for blk in self.blocks:
            nn.init.zeros_(blk.ada[-1].weight)
            nn.init.zeros_(blk.ada[-1].bias)
        nn.init.zeros_(self.ada_out[-1].weight)
        nn.init.zeros_(self.ada_out[-1].bias)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def _grid(self, T: int, H: int, W: int):
        F = T // self.pt
        gh, gw = H // self.ph, W // self.pw
        assert T % self.pt == 0 and H % self.ph == 0 and W % self.pw == 0, \
            f"T/H/W must be divisible by patch {self.cfg.patch}, got {(T, H, W)}"
        return F, gh, gw

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T, H, W) noisy clip; t: (B,) in [0,1]; y: (B,) class ids.
        Returns predicted velocity, same shape as x."""
        B, C, T, H, W = x.shape
        F, gh, gw = self._grid(T, H, W)
        S = gh * gw
        D = self.hidden

        h = self.patch_embed(x)                        # (B, D, F, gh, gw)
        h = h.flatten(3).transpose(2, 3).reshape(B, F * S, D)

        head_dim = D // self.cfg.heads
        s_cos, s_sin = spatial_rope(gh, gw, head_dim, x.device, self.cfg.rope_base)
        t_cos, t_sin = temporal_rope(F, head_dim, x.device, self.cfg.rope_base)

        c = self.t_embed(t) + self.y_embed(y)
        for blk in self.blocks:
            if self.cfg.grad_checkpoint and self.training:
                h = torch.utils.checkpoint.checkpoint(
                    blk, h, c, s_cos, s_sin, t_cos, t_sin, F, S, use_reentrant=False)
            else:
                h = blk(h, c, s_cos, s_sin, t_cos, t_sin, F, S)

        shift, scale = self.ada_out(c).chunk(2, dim=-1)
        h = self.norm_out(h) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        h = self.head(h)                                # (B, N, pt*ph*pw*C)

        # unpatchify: (B, F, gh, gw, pt, ph, pw, C) -> (B, C, T, H, W)
        h = h.view(B, F, gh, gw, self.pt, self.ph, self.pw, C)
        h = h.permute(0, 7, 1, 4, 2, 5, 3, 6).reshape(B, C, F * self.pt, gh * self.ph, gw * self.pw)
        return h


class EMA:
    """Exponential moving average of model weights, kept in fp32 on device."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        d = self.decay
        for k, v in model.state_dict().items():
            s = self.shadow[k]
            if v.dtype.is_floating_point:
                s.mul_(d).add_(v.detach().float(), alpha=1 - d)
            else:
                s.copy_(v)

    def copy_to(self, model: nn.Module):
        model.load_state_dict({k: v.to(model.state_dict()[k].dtype) for k, v in self.shadow.items()})


def build_model(**overrides) -> RFVideoDiT:
    return RFVideoDiT(RFVideoDiTConfig(**overrides))


if __name__ == "__main__":   # tiny self-test
    m = build_model(num_classes=101).cuda()
    x = torch.randn(2, 3, 16, 64, 64, device="cuda")
    t = torch.rand(2, device="cuda")
    y = torch.randint(0, 101, (2,), device="cuda")
    v = m(x, t, y)
    n = sum(p.numel() for p in m.parameters())
    print("output", tuple(v.shape), "params %.1fM" % (n / 1e6))
    # zero-init check: at step 0 the model must output exactly zero
    print("max|v| at init:", v.abs().max().item())
