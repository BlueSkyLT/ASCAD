#!/usr/bin/env python3
"""Per-sample SNR for GPAM's four targets, on the 15k crop and on the 1M raw trace.

SNR(t) = Var_between_classes(t) / Var_within_classes(t), grouping traces by the
class value of each target. This is the quantity that decides whether a head can
learn at all: a head at chance with SNR ~ 0 is unlearnable, not under-trained.

Validated first on the extracted crop, where SNR(c) is documented as ~0.016, then
applied unchanged to the raw 1,000,000-point traces.

Raw accumulation is single-pass with per-class contiguous slices (block sorted by
label, then summed) so the 1M-point data is read from disk exactly once.
"""
import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
sys.path.insert(0, str(REPO_ROOT / "GPAM"))
from raw_dataset import RawMultiShardLoader  # noqa: E402

DB = REPO_ROOT / "STM32_AES_v2/ASCAD_data/ASCAD_databases"
CROP = DB / "ascadv2-extracted.h5"
RAW = DB / "ascadv2-stm32-conso-raw-traces1.h5"

TARGETS = [("rm", 256), ("rout", 256), ("c", 256), ("p", 16)]


def crop_labels(f, n, byte):
    lab = f["Profiling_traces/labels"][:n]
    return {
        "rm": np.asarray(lab["alpha_mask"])[:, 0].astype(np.int32),
        "rout": np.asarray(lab["beta_mask"])[:, 0].astype(np.int32),
        "c": np.asarray(lab["sbox_masked"])[:, byte].astype(np.int32),
        "p": np.asarray(lab["perm_index"])[:, byte].astype(np.int32),
    }


def accumulate(path, n_traces, block, byte, is_raw, loader):
    """Single pass: per-class count, sum, sum-of-squares for each of 4 targets."""
    with h5py.File(path, "r") as f:
        T = f["Profiling_traces/traces"].shape[1] if not is_raw else f["traces"].shape[1]
        acc = {name: {"cnt": np.zeros(K, np.int64),
                      "s1": np.zeros((K, T), np.float32),
                      "s2": np.zeros((K, T), np.float32)} for name, K in TARGETS}
        tot = 0
        t0 = time.time()
        for s in range(0, n_traces, block):
            e = min(s + block, n_traces)
            if is_raw:
                X = f["traces"][s:e].astype(np.float32)
                ry = loader.compute_labels(f["metadata"][s:e])
                y = {"rm": ry["rm"], "rout": ry["rout"],
                     "c": ry[f"c_{byte}"], "p": ry[f"p_{byte}"]}
            else:
                X = f["Profiling_traces/traces"][s:e].astype(np.float32)
                lb = crop_labels(f, e, byte)
                y = {k: lb[k][s:e] for k, _ in TARGETS}
            for (name, K) in TARGETS:
                yy = np.asarray(y[name]).astype(np.int64).ravel()
                order = np.argsort(yy, kind="stable")
                Xs = X[order]
                ys = yy[order]
                bounds = np.searchsorted(ys, np.arange(K), side="left")
                A = acc[name]
                for c in range(K):
                    a, b = int(bounds[c]), (int(bounds[c + 1]) if c + 1 < K else len(ys))
                    if b <= a:
                        continue
                    m = b - a
                    sl = Xs[a:b]
                    A["cnt"][c] += m
                    A["s1"][c] += sl.sum(axis=0)
                    A["s2"][c] += np.square(sl, dtype=np.float32, out=sl).sum(axis=0)
            tot += e - s
            print(f"  [{time.time()-t0:7.1f}s] {tot:6d}/{n_traces} traces "
                  f"({100*tot/n_traces:.0f}%)", flush=True)
    return acc, T


def report(acc, T, tag):
    print(f"\n=== {tag}: per-sample SNR over {T} samples ===")
    print(f"{'target':8s} {'K':>4s} {'maxSNR':>9s} {'at sample':>10s} {'>0.1':>7s} "
          f"{'>0.01':>8s} {'meanSNR':>9s}")
    for name, K in TARGETS:
        A = acc[name]
        n = A["cnt"].astype(np.float64)
        N = n.sum()
        mu_c = A["s1"] / np.maximum(n[:, None], 1)
        mu = A["s1"].sum(0) / max(N, 1)
        vb = ((np.maximum(n, 1)[:, None] * (mu_c - mu[None, :]) ** 2) * (n > 0)[:, None]).sum(0) / max(N, 1)
        within = (A["s2"] - np.maximum(n[:, None], 1) * mu_c ** 2).sum(0)
        vw = within / max(N - K, 1)
        snr = vb / np.maximum(vw, 1e-12)
        i = int(np.argmax(snr))
        print(f"{name:8s} {K:>4d} {snr[i]:>9.4f} {i:>10d} "
              f"{int((snr > 0.1).sum()):>7d} {int((snr > 0.01).sum()):>8d} {snr.mean():>9.5f}")
        top = np.argsort(-snr)[:8]
        print(f"         top-8 sample indices: {top.tolist()}")
        print(f"         SNR at those: {np.round(snr[top], 4).tolist()}")
        np.save(HERE / f"snr_{tag}_{name}.npy", snr.astype(np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["crop", "raw"], required=True)
    ap.add_argument("--n-traces", type=int, default=16384)
    ap.add_argument("--block", type=int, default=64)
    ap.add_argument("--byte", type=int, default=0)
    a = ap.parse_args()

    loader = RawMultiShardLoader(split="train", target_bytes=[a.byte],
                                 batch_size=1, chunk_size=a.block, shuffle=False)
    if a.source == "crop":
        acc, T = accumulate(CROP, a.n_traces, a.block, a.byte, False, loader)
        report(acc, T, "crop15k")
    else:
        acc, T = accumulate(RAW, a.n_traces, a.block, a.byte, True, loader)
        report(acc, T, "raw1m")


if __name__ == "__main__":
    main()
