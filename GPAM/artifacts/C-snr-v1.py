#!/usr/bin/env python3
"""Per-sample SNR for ASCADv1's r, C and S, on the 700-point crop and the raw trace.

v1's stored `labels` column was identified empirically, not assumed: it equals
AES_SBOX[plaintext[2] ^ key[2]] on 200/200 checked rows, and no mask index
reproduces it from that, so `labels` is the UNMASKED S. Hence S = labels,
r = masks[0] (its SNR peaks at 156/181, matching docs/notes/v1-poi-faq.md), and
C = S ^ r. The crop run asserts S == labels as a guard.

SNR(t) = Var_between_classes(t) / Var_within_classes(t), grouping traces by class.
"""
import sys
import time
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, "GPAM")
from run_ascadv2_gpam_baseline import AES_SBOX  # noqa: E402

D = Path("ATMEGA_AES_v1/ATM_AES_v1_fixed_key/ASCAD_data/ASCAD_databases")
CROP = D / "ASCAD.h5"
RAW = D / "ATMega8515_raw_traces.h5"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 16384
SBYTE, MIDX = 2, 0          # verified: labels == SB[pt[2]^k[2]]; r = masks[0]

TARGETS = [("r", 256), ("C", 256), ("S", 256)]


def labels_for(meta_rows):
    pt = np.asarray(meta_rows["plaintext"]).astype(np.int64)
    k = np.asarray(meta_rows["key"]).astype(np.int64)
    S = AES_SBOX[pt[:, SBYTE] ^ k[:, SBYTE]].astype(np.int64)
    r = np.asarray(meta_rows["masks"]).astype(np.int64)[:, MIDX]
    return {"r": r, "C": S ^ r, "S": S}


def accumulate(path, group, is_raw, block):
    tr = f"{group}/traces" if not is_raw else "traces"
    me = f"{group}/metadata" if not is_raw else "metadata"
    with h5py.File(path, "r") as f:
        T = f[tr].shape[1]
        # identity check first
        m5 = f[me][:8]
        lf = labels_for(m5)
        if not is_raw:
            stored = f[f"{group}/labels"][:8]
            print(f"stored labels[:8]     = {stored.tolist()}")
            for nm in ("S", "C", "r"):
                print(f"computed {nm:14s}= {lf[nm][:8].tolist()}"
                      f"   MATCH={bool(np.array_equal(stored, lf[nm]))}")
        acc = {n: {"cnt": np.zeros(K, np.int64),
                   "s1": np.zeros((K, T), np.float32),
                   "s2": np.zeros((K, T), np.float32)} for n, K in TARGETS}
        tot, t0 = 0, time.time()
        for s in range(0, N, block):
            e = min(s + block, N)
            X = f[tr][s:e].astype(np.float32)
            y = labels_for(f[me][s:e])
            for (name, K) in TARGETS:
                yy = y[name]
                order = np.argsort(yy, kind="stable")
                Xs, ys = X[order], yy[order]
                bounds = np.searchsorted(ys, np.arange(K), side="left")
                A = acc[name]
                for c in range(K):
                    a = int(bounds[c]); b = int(bounds[c + 1]) if c + 1 < K else len(ys)
                    if b <= a:
                        continue
                    sl = Xs[a:b]
                    A["cnt"][c] += b - a
                    A["s1"][c] += sl.sum(axis=0)
                    A["s2"][c] += np.square(sl, out=sl).sum(axis=0)
            tot += e - s
            if tot % (block * 40) == 0:
                print(f"  [{time.time()-t0:6.1f}s] {tot}/{N}", flush=True)
    return acc, T


def report(acc, T, tag, source):
    print(f"\n=== {tag}: per-sample SNR over {T} samples, N={N} traces ===")
    print(f"{'target':7s} {'K':>4s} {'maxSNR':>9s} {'at':>8s} {'>1.0':>6s} {'>0.1':>7s} "
          f"{'>0.01':>8s} {'mean':>9s} {'sumSNR':>10s}")
    for name, K in TARGETS:
        A = acc[name]
        n = A["cnt"].astype(np.float64); Ntot = n.sum()
        mu_c = A["s1"] / np.maximum(n[:, None], 1)
        mu = A["s1"].sum(0) / max(Ntot, 1)
        vb = ((np.maximum(n, 1)[:, None] * (mu_c - mu[None, :]) ** 2) * (n > 0)[:, None]).sum(0) / max(Ntot, 1)
        vw = (A["s2"] - np.maximum(n[:, None], 1) * mu_c ** 2).sum(0) / max(Ntot - K, 1)
        snr = vb / np.maximum(vw, 1e-12)
        i = int(np.argmax(snr))
        print(f"{name:7s} {K:>4d} {snr[i]:>9.4f} {i:>8d} {int((snr>1.0).sum()):>6d} "
              f"{int((snr>0.1).sum()):>7d} {int((snr>0.01).sum()):>8d} {snr.mean():>9.5f} {snr.sum():>10.1f}")
        np.save(f"GPAM/artifacts/snr_{source}_{name}.npy", snr.astype(np.float32))


def main():
    which = sys.argv[1]
    if which == "crop":
        acc, T = accumulate(CROP, "Profiling_traces", False, 512)
        report(acc, T, "v1-crop700", "v1crop")
    else:
        acc, T = accumulate(RAW, None, True, 64)
        report(acc, T, "v1-raw100k", "v1raw")


main()
