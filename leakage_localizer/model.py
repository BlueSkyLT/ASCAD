"""Models: per-trace proposer + Boolean share heads (default Variant B)."""

from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
import torch.nn as nn

from .combine import xor_combine
from .windows import (
    WIDE_SIGMA_RAW,
    annealed_sigma,
    constrain_mu,
    default_bands,
    extract_features,
    gaussian_mask,
)

NETWORK_TYPES = ("poi_searcher", "poi_searcher_v1", "leakage_localizer")


def is_poi_searcher_network(network_type: Optional[str]) -> bool:
    if not network_type:
        return False
    return str(network_type).lower() in NETWORK_TYPES


def _inv_mu(mu: float, lo: float, hi: float) -> float:
    p = (float(mu) - lo) / max(hi - lo, 1e-6)
    p = min(max(p, 1e-4), 1.0 - 1e-4)
    return float(torch.log(torch.tensor(p / (1.0 - p))))


def _raw_sigma_wide() -> float:
    """Init σ-raw so annealed_sigma starts near the current sigma_max."""
    return float(WIDE_SIGMA_RAW)


class _Head(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 256),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        logits = self.net(self.norm(feat))
        return torch.softmax(logits, dim=-1)


class WindowProposer(nn.Module):
    """X (B, T) → raw (B, 4) = (μ_r, σ_r, μ_C, σ_C) unconstrained."""

    def __init__(self, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=9, stride=4),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16),
            nn.Flatten(),
            nn.Linear(16 * 16, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.unsqueeze(1))


class SoftWindowBooleanSCA(nn.Module):
    """Variant B: two windows, share heads, XOR-combine P(S)."""

    def __init__(
        self,
        trace_len: int = 700,
        n_moments: int = 8,
        grid: int = 16,
        hidden: int = 128,
        r_band: Optional[Sequence[float]] = None,
        c_band: Optional[Sequence[float]] = None,
        sigma_min: float = 1.0,
        init_mu_r: Optional[float] = None,
        init_mu_c: Optional[float] = None,
        init_sigma: float = 15.0,
    ):
        super().__init__()
        self.T = int(trace_len)
        auto_r, auto_c, auto_mr, auto_mc = default_bands(self.T)
        self.r_band = tuple(r_band) if r_band is not None else auto_r
        self.c_band = tuple(c_band) if c_band is not None else auto_c
        self.sigma_min = float(sigma_min)
        self.init_sigma = float(init_sigma)
        self.register_buffer("sigma_max", torch.tensor(float(init_sigma)))
        self.n_moments = int(n_moments)
        self.grid = int(grid)
        feat_dim = self.n_moments + self.grid
        # Default μ is band *front* (search left→right). μ is an output of h_ψ, never pinned.
        if init_mu_r is None:
            init_mu_r = auto_mr
        if init_mu_c is None:
            init_mu_c = auto_mc
        self.proposer = WindowProposer()
        with torch.no_grad():
            self.proposer.net[-1].bias.copy_(
                torch.tensor(
                    [
                        _inv_mu(init_mu_r, *self.r_band),
                        _raw_sigma_wide(),
                        _inv_mu(init_mu_c, *self.c_band),
                        _raw_sigma_wide(),
                    ]
                )
            )
        self.head_r = _Head(feat_dim, hidden)
        self.head_c = _Head(feat_dim, hidden)

    def set_sigma_max(self, value: float):
        self.sigma_max.fill_(max(float(value), self.sigma_min))

    def window_params(self, x: torch.Tensor):
        raw = self.proposer(x)
        mu_r = constrain_mu(raw[:, 0], *self.r_band)
        sig_r = annealed_sigma(raw[:, 1], self.sigma_min, float(self.sigma_max))
        mu_c = constrain_mu(raw[:, 2], *self.c_band)
        sig_c = annealed_sigma(raw[:, 3], self.sigma_min, float(self.sigma_max))
        return mu_r, mu_c, sig_r, sig_c

    def forward(self, x: torch.Tensor) -> dict:
        if x.ndim == 3:
            x = x.squeeze(1) if x.shape[1] == 1 else x.squeeze(-1)
        B, T = x.shape
        mu_r, mu_c, sig_r, sig_c = self.window_params(x)
        w_r = gaussian_mask(T, mu_r, sig_r)
        w_c = gaussian_mask(T, mu_c, sig_c)
        feat_r = extract_features(x, w_r, self.n_moments, self.grid)
        feat_c = extract_features(x, w_c, self.n_moments, self.grid)
        p_r = self.head_r(feat_r)
        p_c = self.head_c(feat_c)
        p_s = xor_combine(p_r, p_c)
        return {
            "p_r": p_r,
            "p_c": p_c,
            "p_s": p_s,
            "mu_r": mu_r,
            "mu_c": mu_c,
            "sigma_r": sig_r,
            "sigma_c": sig_c,
            "w_r": w_r,
            "w_c": w_c,
        }


class DirectWindowSHead(nn.Module):
    """Variant A: two windows, concat features, one S-head (no XOR)."""

    def __init__(
        self,
        trace_len: int = 700,
        n_moments: int = 8,
        grid: int = 16,
        hidden: int = 128,
        r_band: Optional[Sequence[float]] = None,
        c_band: Optional[Sequence[float]] = None,
        sigma_min: float = 1.0,
        init_mu_r: Optional[float] = None,
        init_mu_c: Optional[float] = None,
        init_sigma: float = 15.0,
    ):
        super().__init__()
        self.T = int(trace_len)
        auto_r, auto_c, auto_mr, auto_mc = default_bands(self.T)
        self.r_band = tuple(r_band) if r_band is not None else auto_r
        self.c_band = tuple(c_band) if c_band is not None else auto_c
        self.sigma_min = float(sigma_min)
        self.init_sigma = float(init_sigma)
        self.register_buffer("sigma_max", torch.tensor(float(init_sigma)))
        self.n_moments = int(n_moments)
        self.grid = int(grid)
        feat_dim = 2 * (self.n_moments + self.grid)
        if init_mu_r is None:
            init_mu_r = auto_mr
        if init_mu_c is None:
            init_mu_c = auto_mc
        self.proposer = WindowProposer()
        with torch.no_grad():
            self.proposer.net[-1].bias.copy_(
                torch.tensor(
                    [
                        _inv_mu(init_mu_r, *self.r_band),
                        _raw_sigma_wide(),
                        _inv_mu(init_mu_c, *self.c_band),
                        _raw_sigma_wide(),
                    ]
                )
            )
        self.head_s = _Head(feat_dim, hidden)

    def set_sigma_max(self, value: float):
        self.sigma_max.fill_(max(float(value), self.sigma_min))

    def window_params(self, x: torch.Tensor):
        raw = self.proposer(x)
        mu_r = constrain_mu(raw[:, 0], *self.r_band)
        sig_r = annealed_sigma(raw[:, 1], self.sigma_min, float(self.sigma_max))
        mu_c = constrain_mu(raw[:, 2], *self.c_band)
        sig_c = annealed_sigma(raw[:, 3], self.sigma_min, float(self.sigma_max))
        return mu_r, mu_c, sig_r, sig_c

    def forward(self, x: torch.Tensor) -> dict:
        if x.ndim == 3:
            x = x.squeeze(1) if x.shape[1] == 1 else x.squeeze(-1)
        B, T = x.shape
        mu_r, mu_c, sig_r, sig_c = self.window_params(x)
        w_r = gaussian_mask(T, mu_r, sig_r)
        w_c = gaussian_mask(T, mu_c, sig_c)
        feat = torch.cat(
            [
                extract_features(x, w_r, self.n_moments, self.grid),
                extract_features(x, w_c, self.n_moments, self.grid),
            ],
            dim=-1,
        )
        p_s = self.head_s(feat)
        return {
            "p_s": p_s,
            "mu_r": mu_r,
            "mu_c": mu_c,
            "sigma_r": sig_r,
            "sigma_c": sig_c,
            "w_r": w_r,
            "w_c": w_c,
        }


class GlobalPoolSHead(nn.Module):
    """Variant 0: no windows; moments of the full trace → S-head."""

    def __init__(self, trace_len: int = 700, n_moments: int = 8, grid: int = 16, hidden: int = 128):
        super().__init__()
        self.T = int(trace_len)
        self.n_moments = int(n_moments)
        self.grid = int(grid)
        self.head_s = _Head(self.n_moments + self.grid, hidden)

    def forward(self, x: torch.Tensor) -> dict:
        if x.ndim == 3:
            x = x.squeeze(1) if x.shape[1] == 1 else x.squeeze(-1)
        B, T = x.shape
        w = torch.full((B, T), 1.0 / T, device=x.device, dtype=x.dtype)
        feat = extract_features(x, w, self.n_moments, self.grid)
        return {"p_s": self.head_s(feat)}


class GlobalThetaSCA(nn.Module):
    """Variant G: four global (μ, σ) parameters shared across traces."""

    def __init__(
        self,
        trace_len: int = 700,
        n_moments: int = 8,
        grid: int = 16,
        hidden: int = 128,
        r_band: Optional[Sequence[float]] = None,
        c_band: Optional[Sequence[float]] = None,
        sigma_min: float = 1.0,
        init_mu_r: Optional[float] = None,
        init_mu_c: Optional[float] = None,
        init_sigma: float = 15.0,
    ):
        super().__init__()
        self.T = int(trace_len)
        auto_r, auto_c, auto_mr, auto_mc = default_bands(self.T)
        self.r_band = tuple(r_band) if r_band is not None else auto_r
        self.c_band = tuple(c_band) if c_band is not None else auto_c
        self.sigma_min = float(sigma_min)
        self.init_sigma = float(init_sigma)
        self.register_buffer("sigma_max", torch.tensor(float(init_sigma)))
        self.n_moments = int(n_moments)
        self.grid = int(grid)
        if init_mu_r is None:
            init_mu_r = auto_mr
        if init_mu_c is None:
            init_mu_c = auto_mc
        self.raw_mu_r = nn.Parameter(torch.tensor(_inv_mu(init_mu_r, *self.r_band)))
        self.raw_mu_c = nn.Parameter(torch.tensor(_inv_mu(init_mu_c, *self.c_band)))
        self.raw_sigma_r = nn.Parameter(torch.tensor(_raw_sigma_wide()))
        self.raw_sigma_c = nn.Parameter(torch.tensor(_raw_sigma_wide()))
        feat_dim = self.n_moments + self.grid
        self.head_r = _Head(feat_dim, hidden)
        self.head_c = _Head(feat_dim, hidden)

    def set_sigma_max(self, value: float):
        self.sigma_max.fill_(max(float(value), self.sigma_min))

    def forward(self, x: torch.Tensor) -> dict:
        if x.ndim == 3:
            x = x.squeeze(1) if x.shape[1] == 1 else x.squeeze(-1)
        B, T = x.shape
        mu_r = constrain_mu(self.raw_mu_r, *self.r_band).expand(B)
        mu_c = constrain_mu(self.raw_mu_c, *self.c_band).expand(B)
        sig_r = annealed_sigma(self.raw_sigma_r, self.sigma_min, float(self.sigma_max)).expand(B)
        sig_c = annealed_sigma(self.raw_sigma_c, self.sigma_min, float(self.sigma_max)).expand(B)
        w_r = gaussian_mask(T, mu_r, sig_r)
        w_c = gaussian_mask(T, mu_c, sig_c)
        p_r = self.head_r(extract_features(x, w_r, self.n_moments, self.grid))
        p_c = self.head_c(extract_features(x, w_c, self.n_moments, self.grid))
        return {
            "p_r": p_r,
            "p_c": p_c,
            "p_s": xor_combine(p_r, p_c),
            "mu_r": mu_r,
            "mu_c": mu_c,
            "sigma_r": sig_r,
            "sigma_c": sig_c,
            "w_r": w_r,
            "w_c": w_c,
        }


class BandAttentionProposer(nn.Module):
    """Per-trace h_ψ: conv scores over time, softmax inside each band.

    Uniform-in-band at init (last layer ~0). Can put mass on a peak without
    sliding a Gaussian from the front.
    """

    def __init__(self, trace_len: int, r_band, c_band, hidden: int = 32):
        super().__init__()
        self.T = int(trace_len)
        self.score = nn.Sequential(
            nn.Conv1d(1, hidden, kernel_size=11, padding=5),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=11, padding=5),
            nn.GELU(),
            nn.Conv1d(hidden, 2, kernel_size=1),
        )
        t = torch.arange(self.T)
        r_lo, r_hi = int(r_band[0]), int(math.ceil(r_band[1]))
        c_lo, c_hi = int(c_band[0]), int(math.ceil(c_band[1]))
        r_mask = (t >= r_lo) & (t <= min(self.T - 1, r_hi))
        c_mask = (t >= c_lo) & (t <= min(self.T - 1, c_hi))
        self.register_buffer("r_mask", r_mask)
        self.register_buffer("c_mask", c_mask)
        self.register_buffer("t_idx", t.float())
        self.temperature = 1.0
        # Soft "r earlier than C": added to scores as ±time_bias * (t-mid)/mid.
        self.time_bias = 0.0

    def _moments(self, w: torch.Tensor):
        t = self.t_idx.to(device=w.device, dtype=w.dtype)
        mu = (w * t).sum(-1)
        var = (w * (t - mu.unsqueeze(-1)).square()).sum(-1)
        sig = var.clamp_min(1e-6).sqrt()
        return mu, sig

    def forward(self, x: torch.Tensor):
        if x.ndim == 3:
            x = x.squeeze(1) if x.shape[1] == 1 else x.squeeze(-1)
        scores = self.score(x.unsqueeze(1)) / max(float(self.temperature), 1e-3)
        if float(self.time_bias) != 0.0:
            mid = max((self.T - 1) * 0.5, 1e-6)
            tilt = float(self.time_bias) * (self.t_idx.to(device=scores.device, dtype=scores.dtype) - mid) / mid
            scores = torch.stack((scores[:, 0] - tilt, scores[:, 1] + tilt), dim=1)
        neg = torch.finfo(scores.dtype).min / 4
        sr = scores[:, 0].masked_fill(~self.r_mask, neg)
        sc = scores[:, 1].masked_fill(~self.c_mask, neg)
        w_r = torch.softmax(sr, dim=-1)
        w_c = torch.softmax(sc, dim=-1)
        mu_r, sig_r = self._moments(w_r)
        mu_c, sig_c = self._moments(w_c)
        return w_r, w_c, mu_r, mu_c, sig_r, sig_c


class AttentionBooleanSCA(nn.Module):
    """Variant B with band-attention h_ψ instead of Gaussian-from-4-scalars."""

    def __init__(
        self,
        trace_len: int = 700,
        n_moments: int = 8,
        grid: int = 16,
        hidden: int = 128,
        r_band: Optional[Sequence[float]] = None,
        c_band: Optional[Sequence[float]] = None,
        sigma_min: float = 1.0,
        init_mu_r: Optional[float] = None,
        init_mu_c: Optional[float] = None,
        init_sigma: float = 15.0,
        attn_hidden: int = 32,
    ):
        super().__init__()
        self.T = int(trace_len)
        auto_r, auto_c, _, _ = default_bands(self.T)
        self.r_band = tuple(r_band) if r_band is not None else auto_r
        self.c_band = tuple(c_band) if c_band is not None else auto_c
        self.sigma_min = float(sigma_min)
        self.init_sigma = float(init_sigma)
        self.register_buffer("sigma_max", torch.tensor(float(init_sigma)))
        self.n_moments = int(n_moments)
        self.grid = int(grid)
        self.proposer = BandAttentionProposer(self.T, self.r_band, self.c_band, hidden=attn_hidden)
        feat_dim = self.n_moments + self.grid
        self.head_r = _Head(feat_dim, hidden)
        self.head_c = _Head(feat_dim, hidden)

    def set_sigma_max(self, value: float):
        # Attention width is data-driven; keep buffer for logging only.
        self.sigma_max.fill_(max(float(value), self.sigma_min))

    def forward(self, x: torch.Tensor) -> dict:
        if x.ndim == 3:
            x = x.squeeze(1) if x.shape[1] == 1 else x.squeeze(-1)
        w_r, w_c, mu_r, mu_c, sig_r, sig_c = self.proposer(x)
        p_r = self.head_r(extract_features(x, w_r, self.n_moments, self.grid))
        p_c = self.head_c(extract_features(x, w_c, self.n_moments, self.grid))
        return {
            "p_r": p_r,
            "p_c": p_c,
            "p_s": xor_combine(p_r, p_c),
            "mu_r": mu_r,
            "mu_c": mu_c,
            "sigma_r": sig_r,
            "sigma_c": sig_c,
            "w_r": w_r,
            "w_c": w_c,
        }


class TwoOrderWindows(nn.Module):
    """Only h_ψ: two windows, pooled power, no byte heads. For L_2O."""

    def __init__(self, trace_len: int = 700, r_band=None, c_band=None, attn_hidden: int = 32, **_ignored):
        super().__init__()
        self.T = int(trace_len)
        auto_r, auto_c, _, _ = default_bands(self.T)
        self.r_band = tuple(r_band) if r_band is not None else auto_r
        self.c_band = tuple(c_band) if c_band is not None else auto_c
        self.proposer = BandAttentionProposer(self.T, self.r_band, self.c_band, hidden=attn_hidden)

    def forward(self, x: torch.Tensor) -> dict:
        if x.ndim == 3:
            x = x.squeeze(1) if x.shape[1] == 1 else x.squeeze(-1)
        w_r, w_c, mu_r, mu_c, sig_r, sig_c = self.proposer(x)
        return {
            "ell_r": (w_r * x).sum(-1),
            "ell_c": (w_c * x).sum(-1),
            "mu_r": mu_r,
            "mu_c": mu_c,
            "sigma_r": sig_r,
            "sigma_c": sig_c,
            "w_r": w_r,
            "w_c": w_c,
        }


class ThreeOrderWindows(nn.Module):
    """Three per-trace windows for ASCADv2: alpha, beta, and C.

    The proposer emits a separate softmax on each band. On 15k traces the
    alpha/beta windows are constrained to the first 5k samples while C is forced
    to the second band.
    """

    def __init__(self, trace_len: int = 15000, attn_hidden: int = 32):
        super().__init__()
        self.T = int(trace_len)
        self.a_band = (0.0, 5000.0)
        self.b_band = (0.0, 5000.0)
        self.c_band = (5000.0, float(self.T - 1))
        self.proposer = BandAttentionProposer(self.T, self.a_band, self.c_band, hidden=attn_hidden)
        self.a_head = nn.Sequential(
            nn.Conv1d(1, attn_hidden, kernel_size=11, padding=5),
            nn.GELU(),
            nn.Conv1d(attn_hidden, 1, kernel_size=1),
        )
        self.b_head = nn.Sequential(
            nn.Conv1d(1, attn_hidden, kernel_size=11, padding=5),
            nn.GELU(),
            nn.Conv1d(attn_hidden, 1, kernel_size=1),
        )
        self.c_head = nn.Sequential(
            nn.Conv1d(1, attn_hidden, kernel_size=11, padding=5),
            nn.GELU(),
            nn.Conv1d(attn_hidden, 1, kernel_size=1),
        )

    def _band_softmax(self, scores: torch.Tensor, lo: int, hi: int):
        t = torch.arange(self.T, device=scores.device, dtype=scores.dtype)
        mask = (t >= lo) & (t <= hi)
        neg = torch.finfo(scores.dtype).min / 4
        masked = scores.masked_fill(~mask, neg)
        return torch.softmax(masked, dim=-1)

    def forward(self, x: torch.Tensor) -> dict:
        if x.ndim == 3:
            x = x.squeeze(1) if x.shape[1] == 1 else x.squeeze(-1)
        B, T = x.shape
        scores_a = self.a_head(x.unsqueeze(1)).squeeze(1)
        scores_b = self.b_head(x.unsqueeze(1)).squeeze(1)
        scores_c = self.c_head(x.unsqueeze(1)).squeeze(1)
        w_a = self._band_softmax(scores_a, 0, min(self.T - 1, 5000))
        w_b = self._band_softmax(scores_b, 0, min(self.T - 1, 5000))
        w_c = self._band_softmax(scores_c, 5000, self.T - 1)
        t = torch.arange(self.T, device=x.device, dtype=x.dtype)
        mu_a = (w_a * t).sum(-1)
        mu_b = (w_b * t).sum(-1)
        mu_c = (w_c * t).sum(-1)
        sig_a = ((w_a * (t - mu_a.unsqueeze(-1)).square()).sum(-1)).sqrt().clamp_min(1e-6)
        sig_b = ((w_b * (t - mu_b.unsqueeze(-1)).square()).sum(-1)).sqrt().clamp_min(1e-6)
        sig_c = ((w_c * (t - mu_c.unsqueeze(-1)).square()).sum(-1)).sqrt().clamp_min(1e-6)
        return {
            "ell_a": (w_a * x).sum(-1),
            "ell_b": (w_b * x).sum(-1),
            "ell_c": (w_c * x).sum(-1),
            "mu_a": mu_a,
            "mu_b": mu_b,
            "mu_c": mu_c,
            "sigma_a": sig_a,
            "sigma_b": sig_b,
            "sigma_c": sig_c,
            "w_a": w_a,
            "w_b": w_b,
            "w_c": w_c,
        }


def build_poi_searcher(trace_len: int, config: Optional[dict] = None) -> nn.Module:
    cfg = dict(config or {})
    variant = str(cfg.get("variant", "boolean")).lower()
    kwargs = dict(
        trace_len=int(trace_len),
        n_moments=int(cfg.get("n_moments", 8)),
        grid=int(cfg.get("grid", 16)),
        hidden=int(cfg.get("hidden", 128)),
        r_band=cfg.get("r_band"),
        c_band=cfg.get("c_band"),
        sigma_min=float(cfg.get("sigma_min", 1.0)),
        init_mu_r=cfg.get("init_mu_r"),
        init_mu_c=cfg.get("init_mu_c"),
        init_sigma=float(cfg.get("init_sigma", 15.0)),
    )
    if variant in ("two_order", "2o", "l2o"):
        return TwoOrderWindows(
            trace_len=int(trace_len),
            r_band=cfg.get("r_band"),
            c_band=cfg.get("c_band"),
            attn_hidden=int(cfg.get("attn_hidden", 32)),
        )
    proposer = str(cfg.get("proposer", "attention")).lower()
    if variant in ("attention",) or (
        variant in ("boolean", "b", "share", "default") and proposer == "attention"
    ):
        return AttentionBooleanSCA(**kwargs)
    if variant in ("boolean", "b", "share", "default"):
        return SoftWindowBooleanSCA(**kwargs)
    if variant in ("direct", "a"):
        return DirectWindowSHead(**kwargs)
    if variant in ("global_pool", "0", "none"):
        return GlobalPoolSHead(
            trace_len=kwargs["trace_len"],
            n_moments=kwargs["n_moments"],
            grid=kwargs["grid"],
            hidden=kwargs["hidden"],
        )
    if variant in ("global_theta", "g", "global"):
        return GlobalThetaSCA(**kwargs)
    raise ValueError(f"Unknown poi_searcher variant: {variant}")
