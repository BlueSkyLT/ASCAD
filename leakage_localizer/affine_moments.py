"""Affine 3-way moment helpers for ASCADv2.

The relevant 3-way moment is
    m3(s) = E[(HW(alpha)-4)*(HW(beta)-4)*(HW(C)-4) | S=s]
with C = (alpha * S) xor beta under GF(256).
This module exposes the table and tensor helpers used by the v2 localizer.
"""

from __future__ import annotations

import numpy as np
import torch

from ASCAD_generate import multGF256

_HW = np.array([bin(i).count("1") for i in range(256)], dtype=np.float64)


def m3_table() -> torch.Tensor:
    """Return m3(s) for s in 0..255 as a float32 tensor."""
    out = np.empty(256, dtype=np.float64)
    alpha_values = np.arange(1, 256, dtype=np.uint16)
    for s in range(256):
        vals = []
        for a in alpha_values:
            vals.append((float(_HW[a]) - 4.0) * (4.0 - float(_HW[multGF256(int(a), int(s))])))
        out[s] = 0.5 * float(np.mean(vals))
    return torch.tensor(out, dtype=torch.float32)


def centered_product3(ell_a: torch.Tensor, ell_b: torch.Tensor, ell_c: torch.Tensor) -> torch.Tensor:
    """Centered 3-way product over the current batch."""
    a = ell_a - ell_a.mean()
    b = ell_b - ell_b.mean()
    c = ell_c - ell_c.mean()
    return a * b * c


def prod3_from_labels(ell_a: torch.Tensor, ell_b: torch.Tensor, ell_c: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Convenience helper: centered product3 then stage against the affine moment table."""
    prod = centered_product3(ell_a, ell_b, ell_c)
    target = m3_table().to(device=s.device, dtype=prod.dtype)[s.long()]
    return prod, target
