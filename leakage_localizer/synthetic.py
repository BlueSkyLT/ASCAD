"""Planted two-Gaussian Boolean traces for S0 (centres jitter per trace)."""

from __future__ import annotations

import numpy as np
import torch

HW = np.array([bin(i).count("1") for i in range(256)], dtype=np.float32)


def make_synthetic(
    n: int = 2048,
    T: int = 400,
    mu_r: float = 80.0,
    mu_c: float = 280.0,
    jitter: float = 40.0,
    sigma_leak: float = 6.0,
    amp: float = 2.0,
    noise: float = 1.0,
    seed: int = 0,
    independent_c_jump: bool = False,
) -> dict:
    rng = np.random.default_rng(seed)
    s = rng.integers(0, 256, size=n, dtype=np.uint8)
    r = rng.integers(0, 256, size=n, dtype=np.uint8)
    c = np.bitwise_xor(s, r)
    delta = rng.uniform(-jitter, jitter, size=n).astype(np.float32)
    mu_r_i = (mu_r + delta).astype(np.float32)
    if independent_c_jump:
        delta_c = rng.uniform(-jitter, jitter, size=n).astype(np.float32)
        mu_c_i = (mu_c + delta_c).astype(np.float32)
    else:
        mu_c_i = (mu_c + delta).astype(np.float32)
    t = np.arange(T, dtype=np.float32)
    x = rng.normal(0.0, noise, size=(n, T)).astype(np.float32)
    gr = np.exp(-0.5 * ((t[None, :] - mu_r_i[:, None]) / sigma_leak) ** 2)
    gc = np.exp(-0.5 * ((t[None, :] - mu_c_i[:, None]) / sigma_leak) ** 2)
    x += amp * HW[r][:, None] * gr
    x += amp * HW[c][:, None] * gc
    return {
        "x": torch.from_numpy(x),
        "s": torch.from_numpy(s.astype(np.int64)),
        "r": torch.from_numpy(r.astype(np.int64)),
        "c": torch.from_numpy(c.astype(np.int64)),
        "mu_r": torch.from_numpy(mu_r_i),
        "mu_c": torch.from_numpy(mu_c_i),
        "T": T,
        "base_mu_r": mu_r,
        "base_mu_c": mu_c,
    }
