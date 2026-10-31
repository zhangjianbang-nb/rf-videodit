"""Rectified-flow (flow matching) objective and ODE samplers.

Convention used everywhere in this project
------------------------------------------
    t = 0  ->  pure noise        x_0 ~ N(0, I)
    t = 1  ->  real video        x_1
    x_t = (1 - t) * x_0 + t * x_1
    target velocity u = x_1 - x_0

The network v_theta(x_t, t, y) regresses u.  Sampling integrates the ODE
dx/dt = v_theta backwards from t = 1 (noise) to t = 0 (data), i.e.
x_{t-dt} = x_t - dt * v_theta(x_t, t).

Why rectified flow: the interpolation is a straight line, so with a perfectly
fitted velocity field a single Euler step would suffice.  In practice the
learned field is curved, and the sampler trades step count for fidelity.
"""

from __future__ import annotations

import torch


def sample_timesteps(bs: int, device, mode: str = "logit_normal", uniform_frac: float = 0.1):
    """Timestep distribution.  logit-normal (SD3) concentrates capacity on the
    mid-range t where the velocity field is hardest; a uniform tail keeps the
    endpoints well behaved."""
    if mode == "uniform":
        return torch.rand(bs, device=device)
    if mode == "logit_normal":
        n = torch.randn(bs, device=device)
        t = torch.sigmoid(n)
        mask = torch.rand(bs, device=device) < uniform_frac
        t = torch.where(mask, torch.rand(bs, device=device), t)
        return t
    raise ValueError(mode)


def flow_matching_loss(model, x1: torch.Tensor, y: torch.Tensor, *,
                       t_mode: str = "logit_normal", x0: torch.Tensor | None = None,
                       weight: str = "none"):
    """MSE between predicted and target velocity.

    x1: (B, C, T, H, W) real clips in [-1, 1].  `weight='snr'` down-weights the
    noisy end (t -> 0) where the target velocity is dominated by noise."""
    bs = x1.shape[0]
    if x0 is None:
        x0 = torch.randn_like(x1)
    t = sample_timesteps(bs, x1.device, t_mode)
    t_ = t.view(bs, *([1] * (x1.dim() - 1)))
    x_t = (1 - t_) * x0 + t_ * x1
    target = x1 - x0
    pred = model(x_t, t, y)
    err = (pred - target) ** 2
    if weight == "snr":
        # SNR(t) = t^2 / (1-t)^2  ->  weight = SNR / (SNR + 1) = t^2
        err = err * (t_ ** 2)
    return err.mean()


@torch.no_grad()
def sample(model, shape, y: torch.Tensor | None, *, steps: int = 50,
           cfg_scale: float = 1.0, solver: str = "heun", device=None,
           x0: torch.Tensor | None = None, progress=None):
    """Generate clips by integrating the learned ODE from noise to data.

    Convention: t=0 is pure noise, t=1 is data, so sampling starts at t=0 with
    x ~ N(0, I) and integrates FORWARD (dt > 0) up to t=1.  Getting this backwards
    (starting at t=1 and integrating negatively) applies the velocity field with
    the wrong sign and yields noise-like output.

    shape: (B, C, T, H, W).  y: (B,) class ids or None for unconditional.
    cfg_scale > 1 applies classifier-free guidance (requires the model to have
    been trained with label dropout).  solver: 'euler' (1 NFE/step) or 'heun'
    (2 NFE/step, better quality per step).
    """
    device = device or next(model.parameters()).device
    B = shape[0]
    # unconditional sampling = every sample conditioned on the null class, so the
    # class embedding always receives a tensor
    if y is None:
        y = torch.full((B,), model.y_embed.null_index, device=device, dtype=torch.long)
    x = torch.randn(shape, device=device) if x0 is None else x0.clone()
    dt = 1.0 / steps                       # positive: marching from t=0 (noise) up to t=1 (data)

    def velocity(x_t, t_scalar):
        t_batch = torch.full((B,), t_scalar, device=device)
        v = model(x_t, t_batch, y)
        if cfg_scale != 1.0:
            null = torch.full_like(y, model.y_embed.null_index)
            v_u = model(x_t, t_batch, null)
            v = v_u + cfg_scale * (v - v_u)
        return v

    for i in range(steps):
        t = i / steps
        v = velocity(x, t)
        if solver == "euler":
            x = x + dt * v
        elif solver == "heun":
            x_pred = x + dt * v                       # Euler predictor
            v2 = velocity(x_pred, min(1.0, t + dt))
            x = x + 0.5 * dt * (v + v2)               # trapezoidal corrector
        else:
            raise ValueError(solver)
        if progress is not None:
            progress(i + 1, steps)
    return x.clamp(-1, 1)


@torch.no_grad()
def sample_reflow_pairs(model, x1: torch.Tensor, y: torch.Tensor, *, steps: int = 25,
                        cfg_scale: float = 1.0, solver: str = "euler", device=None):
    """Reflow data construction: run the model from the *same* noise x0 that
    produced x1 during training, record (x0, x_final).  Retraining on these
    pairs straightens the trajectory, so far fewer steps are needed."""
    device = device or next(model.parameters()).device
    x0 = torch.randn_like(x1, device=device)
    x_final = sample(model, tuple(x1.shape), y, steps=steps, cfg_scale=cfg_scale,
                     solver=solver, device=device, x0=x0)
    return x0, x_final
