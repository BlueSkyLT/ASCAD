"""Permutation-aware pi routing for the ASCADv2 multi-task attack (T-041).

The 16 `permind_{s}_output` heads give R[n, slot, byte] = p(slot s carries byte b), one softmax per slot.
They ignore that the true routing is a permutation (each byte occurs in exactly one slot). The attack
decoders therefore route with the EXACT marginals of

    P(sigma | x)  proportional to  prod_s R[s, sigma(s)],     sigma in S_16,

    mu[s, b] = P(sigma(s) = b | x),

computed by a forward/backward DP over the 2^16 subsets of already-used bytes (the lattice of any
16-item assignment problem; no permutation dictionary is used). Rows and columns of mu sum to 1.

T-041 P0 (2026-10-17): +0.08 to +0.55 median z on 7/7 cached models, never fewer rank-0 bytes.

Mode: `ASCAD_PI_ROUTING=exact` (default) or `raw` (reproduces every number recorded before 2026-10-17).
"""
import os
import warnings

import numpy as np

N_SLOTS = 16
_LAYOUT = {}


def pi_routing_mode():
    mode = os.environ.get("ASCAD_PI_ROUTING", "exact").strip().lower()
    if mode not in ("exact", "raw"):
        raise ValueError(f"ASCAD_PI_ROUTING must be 'exact' or 'raw', got {mode!r}")
    return mode


def _layout(n, device):
    """Per popcount layer k: masks, and for each byte b the positions/sources of the DP updates."""
    key = (n, str(device))
    if key in _LAYOUT:
        return _LAYOUT[key]
    import torch
    masks = np.arange(1 << n)
    pc = np.array([bin(m).count("1") for m in masks])
    layers = []
    for k in range(n + 1):
        M = masks[pc == k]
        has = [np.nonzero((M >> b) & 1)[0] for b in range(n)]
        free = [np.nonzero(((M >> b) & 1) == 0)[0] for b in range(n)]
        t = lambda a: torch.as_tensor(a, dtype=torch.long, device=device)
        layers.append({
            "M": t(M),
            "has": [t(h) for h in has], "has_src": [t(M[h] ^ (1 << b)) for b, h in enumerate(has)],
            "free": [t(f) for f in free], "free_dst": [t(M[f] | (1 << b)) for b, f in enumerate(free)],
        })
    _LAYOUT[key] = layers
    return layers


def exact_perm_marginals(PI, chunk=1024, device=None):
    """Exact slot->byte marginals under the permutation constraint. PI: (n, 16, 16), rows = slots.

    Uses CUDA when available (float64). Traces whose permanent underflows fall back to their raw rows.
    """
    import torch
    PI = np.asarray(PI, dtype=np.float64)
    n_tr, n, n2 = PI.shape
    assert n == n2, PI.shape
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    L = _layout(n, device)
    full = (1 << n) - 1
    out = np.empty_like(PI)
    bad = 0
    for c0 in range(0, n_tr, chunk):
        A = torch.as_tensor(PI[c0:c0 + chunk], device=device).clamp_min(1e-300)
        A = A / A.amax(dim=2, keepdim=True)       # per-row rescale: P(sigma) unchanged
        m = A.shape[0]
        f = torch.zeros(m, 1 << n, dtype=torch.float64, device=device)
        f[:, 0] = 1.0
        for k in range(1, n + 1):
            lay = L[k]
            acc = torch.zeros(m, len(lay["M"]), dtype=torch.float64, device=device)
            for b in range(n):
                acc[:, lay["has"][b]] += f[:, lay["has_src"][b]] * A[:, k - 1, b:b + 1]
            f[:, lay["M"]] = acc
        g = torch.zeros_like(f)
        g[:, full] = 1.0
        for k in range(n - 1, -1, -1):
            lay = L[k]
            acc = torch.zeros(m, len(lay["M"]), dtype=torch.float64, device=device)
            for b in range(n):
                acc[:, lay["free"][b]] += A[:, k, b:b + 1] * g[:, lay["free_dst"][b]]
            g[:, lay["M"]] = acc
        Z = f[:, full]
        mu = torch.empty(m, n, n, dtype=torch.float64, device=device)
        for s in range(n):
            lay = L[s]
            for b in range(n):
                src = lay["M"][lay["free"][b]]
                mu[:, s, b] = (f[:, src] * g[:, lay["free_dst"][b]]).sum(1) * A[:, s, b]
        mu = mu / Z[:, None, None]
        ok = torch.isfinite(mu).all(dim=(1, 2)) & (Z > 0)
        mu = mu.cpu().numpy()
        okn = ok.cpu().numpy()
        if not okn.all():
            bad += int((~okn).sum())
            mu[~okn] = PI[c0:c0 + chunk][~okn]
        out[c0:c0 + chunk] = mu
    if bad:
        warnings.warn(f"exact_perm_marginals: {bad}/{n_tr} traces underflowed; raw routing kept for them")
    return out


def apply_pi_routing(PI, mode=None):
    """The routing every attack decoder uses: exact permutation marginals unless ASCAD_PI_ROUTING=raw."""
    mode = mode or pi_routing_mode()
    return np.asarray(PI, dtype=np.float64) if mode == "raw" else exact_perm_marginals(PI)
