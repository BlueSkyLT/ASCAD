#!/usr/bin/env python3
"""T-041 P0: permutation-aware pi routing at attack time (no retraining).

Reads a cached 34-head npz (softmax probabilities, `w6_nochead_eval --save-npz`) and writes one npz per
routing in which only the 16 `permind_*` rows are replaced:
  raw    as trained (copy)
  hung   Hungarian MAP permutation, one-hot rows
  sink   Sinkhorn normalisation (doubly stochastic), log space
  exact  exact marginals of P(sigma | x) prop. to prod_s R[s, sigma(s)] (subset DP over used-byte sets)
Uses only "pi is a permutation of 16 bytes"; no permutation dictionary. Scoring is left to the
unchanged `experiments/c4_oracle_swaps.py` (arm A0 on each written npz).
Pre-registration: docs/tasks/T041-permutation-aware-pi-head.md (P0).
"""
import argparse
import itertools
import os

import numpy as np
from scipy.optimize import linear_sum_assignment

N = 16


def routing(z):
    return np.stack([z[f"permind_{s}_output"] for s in range(N)], 1)  # (n, slot, byte)


def hungarian(R):
    logR = np.log(np.clip(R, 1e-300, None))
    out = np.zeros_like(R)
    for i in range(len(R)):
        out[i, np.arange(R.shape[1]), linear_sum_assignment(-logR[i])[1]] = 1.0
    return out


def sinkhorn(R, iters=50):
    L = np.log(np.clip(R, 1e-300, None))
    for _ in range(iters):
        L = L - np.logaddexp.reduce(L, axis=2, keepdims=True)
        L = L - np.logaddexp.reduce(L, axis=1, keepdims=True)
    L = L - np.logaddexp.reduce(L, axis=2, keepdims=True)  # end on rows: exact row sums for the decoder
    return np.exp(L)


def exact_marginals(R, chunk=256):
    """mu[s, b] = P(sigma(s) = b) under P(sigma) prop. to prod_s R[s, sigma(s)].

    Forward f[M]: total weight of assigning slots 0..|M|-1 to the byte set M. Backward g[M]: weight of
    assigning slots |M|..n-1 to the complement of M. mu[s, b] = sum_{|M|=s, b not in M}
    f[M] R[s, b] g[M | b] / f[full]. 2^n * n terms per trace, all positive.
    """
    n = R.shape[1]
    full = (1 << n) - 1
    masks = np.arange(1 << n)
    pc = np.array([bin(m).count("1") for m in masks])
    layers = [masks[pc == k] for k in range(n + 1)]
    out = np.empty_like(R)
    for c0 in range(0, len(R), chunk):
        A = R[c0:c0 + chunk]
        A = A / A.max(axis=2, keepdims=True)          # per-row rescale: distribution over sigma unchanged
        m = len(A)
        f = np.zeros((m, 1 << n))
        f[:, 0] = 1.0
        for k in range(1, n + 1):
            M = layers[k]
            acc = np.zeros((m, len(M)))
            for b in range(n):
                has = (M >> b) & 1 == 1
                acc[:, has] += f[:, M[has] ^ (1 << b)] * A[:, k - 1, b][:, None]
            f[:, M] = acc
        g = np.zeros((m, 1 << n))
        g[:, full] = 1.0
        for k in range(n - 1, -1, -1):
            M = layers[k]
            acc = np.zeros((m, len(M)))
            for b in range(n):
                free = (M >> b) & 1 == 0
                acc[:, free] += A[:, k, b][:, None] * g[:, M[free] | (1 << b)]
            g[:, M] = acc
        Z = f[:, full]
        mu = np.zeros((m, n, n))
        for s in range(n):
            M = layers[s]
            for b in range(n):
                free = (M >> b) & 1 == 0
                mu[:, s, b] = (f[:, M[free]] * g[:, M[free] | (1 << b)]).sum(1) * A[:, s, b]
        out[c0:c0 + chunk] = mu / Z[:, None, None]
    return out


def brute_marginals(R):
    n = R.shape[1]
    mu = np.zeros_like(R)
    for i in range(len(R)):
        for sig in itertools.permutations(range(n)):
            w = np.prod(R[i, np.arange(n), sig])
            mu[i, np.arange(n), sig] += w
        mu[i] /= mu[i].sum(1, keepdims=True)
    return mu


def self_test():
    rng = np.random.default_rng(0)
    R = rng.dirichlet(np.ones(6) * 0.5, size=(20, 6))
    err = np.abs(exact_marginals(R) - brute_marginals(R)).max()
    mu = exact_marginals(rng.dirichlet(np.ones(N) * 0.3, size=(8, N)))
    rows, cols = np.abs(mu.sum(2) - 1).max(), np.abs(mu.sum(1) - 1).max()
    print(f"self-test: DP vs brute force (6x6) max err {err:.2e}; 16x16 row/col sum dev {rows:.2e}/{cols:.2e}")
    assert err < 1e-10 and rows < 1e-9 and cols < 1e-9
    return {"dp_vs_brute": float(err), "row_dev": float(rows), "col_dev": float(cols)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out-dir", default="experiments/out/t041")
    ap.add_argument("--perm-labels", default=None, help="optional: report pi metrics against true perm_index")
    args = ap.parse_args()
    self_test()
    os.makedirs(args.out_dir, exist_ok=True)
    z = dict(np.load(args.npz))
    R = routing(z)
    variants = {"raw": R, "hung": hungarian(R), "sink": sinkhorn(R), "exact": exact_marginals(R)}
    T = None
    if args.perm_labels:
        import h5py
        with h5py.File(args.perm_labels, "r") as f:
            T = np.asarray(f["Attack_traces/labels"].fields("perm_index")[:len(R)])
    for name, Rv in variants.items():
        zz = dict(z)
        for s in range(N):
            zz[f"permind_{s}_output"] = Rv[:, s, :]
        path = os.path.join(args.out_dir, f"{args.label}_{name}_heads.npz")
        np.savez(path, **zz)
        msg = f"{args.label} {name:5s} -> {path}"
        if T is not None:
            idx = (np.arange(len(R))[:, None], np.arange(N)[None, :], T)
            msg += (f" | elem {(Rv.argmax(2) == T).mean():.3f}  p(true) mean {Rv[idx].mean():.3f}"
                    f"  |col-1| {np.abs(Rv.sum(1) - 1).mean():.3f}")
        print(msg, flush=True)


if __name__ == "__main__":
    main()
