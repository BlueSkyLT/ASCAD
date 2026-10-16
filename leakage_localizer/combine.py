"""XOR convolution of two 256-way share distributions."""

from __future__ import annotations

import torch


def xor_table(device=None, dtype=torch.long) -> torch.Tensor:
    """xor_table[s, m] = s ^ m, shape (256, 256)."""
    x = torch.arange(256, device=device, dtype=dtype)
    return x[:, None] ^ x[None, :]


def xor_combine(p_r: torch.Tensor, p_c: torch.Tensor) -> torch.Tensor:
    """P(S=s) = sum_m p_r[m] * p_c[s ^ m].

    p_r, p_c: (B, 256) probabilities (or unnormalized non-negatives).
    Returns (B, 256).
    """
    if p_r.shape != p_c.shape or p_r.ndim != 2 or p_r.shape[-1] != 256:
        raise ValueError(
            f"expected (B,256) and (B,256), got {tuple(p_r.shape)} {tuple(p_c.shape)}"
        )
    table = xor_table(device=p_c.device)
    p_c_perm = p_c[:, table]
    return (p_r.unsqueeze(1) * p_c_perm).sum(-1)
