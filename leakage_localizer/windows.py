"""Differentiable Gaussian windows and pooled features."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def constrain_mu(raw: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    """Map unconstrained raw → (lo, hi).

    Saturation at 0/1 is a trap: init *inside* the band (see front_mu), not at lo.
    """
    return lo + (hi - lo) * torch.sigmoid(raw)


# Start h_ψ's σ-raw near the wide end of annealed_sigma (sigmoid ≈ 0.95).
WIDE_SIGMA_RAW = 2.944439  # logit(0.95)


def front_mu(lo: float, hi: float, frac: float = 0.10) -> float:
    """Start of a left-to-right search: a bit inside the band so ∂μ/∂raw ≠ 0."""
    f = min(max(float(frac), 1e-3), 1.0 - 1e-3)
    return float(lo) + f * (float(hi) - float(lo))


def positive_sigma(raw: torch.Tensor, sigma_min: float = 1.0) -> torch.Tensor:
    """Legacy: σ ≥ sigma_min via softplus. Prefer annealed_sigma in the model."""
    return float(sigma_min) + F.softplus(raw)


def annealed_sigma(raw: torch.Tensor, sigma_min: float, sigma_max: float) -> torch.Tensor:
    """σ ∈ (sigma_min, sigma_max). Start raw large so σ is near the current max (wide)."""
    lo = float(sigma_min)
    hi = max(float(sigma_max), lo + 1e-4)
    return lo + (hi - lo) * torch.sigmoid(raw)


def sigma_max_at_epoch(epoch: int, epochs: int, init_sigma: float, sigma_min: float) -> float:
    """Linear cap: epoch 0 → init_sigma (wide), last epoch → sigma_min."""
    if epochs <= 1:
        return float(sigma_min)
    t = float(epoch) / float(epochs - 1)
    return float(init_sigma) + t * (float(sigma_min) - float(init_sigma))


def gaussian_mask(length: int, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Soft Gaussian over time. mu, sigma: (B,) or scalar → (B, T)."""
    mu = mu.reshape(-1)
    sigma = sigma.reshape(-1)
    t = torch.arange(length, device=mu.device, dtype=mu.dtype)
    z = (t.unsqueeze(0) - mu.unsqueeze(1)) / sigma.unsqueeze(1).clamp_min(1e-4)
    return torch.softmax(-0.5 * z.square(), dim=-1)


def extract_features(
    x: torch.Tensor,
    weights: torch.Tensor,
    n_moments: int = 8,
    grid: int = 16,
) -> torch.Tensor:
    """x, weights: (B, T) → (B, n_moments + grid)."""
    w = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
    mean = (w * x).sum(-1, keepdim=True)
    xc = x - mean
    moments = [mean.squeeze(-1)]
    for k in range(2, n_moments + 1):
        m_k = (w * xc.pow(k)).sum(-1)
        # signed k-th root keeps scale O(1) for the heads
        moments.append(torch.sign(m_k) * m_k.abs().clamp_min(1e-12).pow(1.0 / k))
    mom = torch.stack(moments, dim=-1)
    qs = torch.linspace(0.05, 0.95, grid, device=x.device, dtype=x.dtype)
    cdf = torch.cumsum(w, dim=-1)
    dist = (cdf.unsqueeze(1) - qs.view(1, grid, 1)).abs()
    kern = torch.softmax(-dist * 50.0, dim=-1)
    grid_x = (kern * x.unsqueeze(1)).sum(-1)
    return torch.cat([mom, grid_x], dim=-1)


def default_bands(trace_len: int, front_frac: float = 0.10):
    """Allowed search intervals. Init μ at the *front* (left), not at SNR peaks.

    SNR peaks (156 / 517 on T=700) are the *success target*, not the start.
    """
    t = float(trace_len)
    scale = t / 700.0
    r_band = (40.0 * scale, 280.0 * scale)
    c_band = (400.0 * scale, min(t - 1.0, 640.0 * scale))
    if c_band[0] >= c_band[1]:
        c_band = (0.55 * t, 0.95 * t)
    init_mu_r = front_mu(*r_band, frac=front_frac)
    init_mu_c = front_mu(*c_band, frac=front_frac)
    return r_band, c_band, init_mu_r, init_mu_c


def full_bands(trace_len: int):
    """No r/C prior: both windows may put mass on any sample."""
    t = float(trace_len)
    band = (0.0, t - 1.0)
    return band, band, 0.25 * t, 0.75 * t
