#!/usr/bin/env python3
"""How much of the trace does ASCADv2 mask prediction actually need?

Context. A parallel session's four-head GPAM run tonight reproduces `rm` (the multiply
hide alpha) at 92.95% top-1 on the 15,000-point extracted crop, while `LLM/amp_check.py`
reads the single best 100-sample window at ~16% and `LLM/snr_check.py` shows the leak is
spread over many bands (11 of 60 scanned windows clear 4x chance, and window-mean SNR
predicts learned accuracy at Pearson r = 0.939). So the open question is not "can a model
read alpha" -- it can -- but **how many samples the label needs**.

This script answers it with a coverage curve. Bands are chosen by SNR only, top-N
non-overlapping windows of fixed width, then one shared MLP is trained on their union:

    python -u -m LLM.band_scan --curve docs/2026-09-23/out/SNR-alpha-curve.npy \
        --width 100 --bands 1 2 4 8 16 32 --repeats 3

Every number is reported as a mean over repetitions plus the spread, because a single
unseeded draw of this probe varied 5x on identical settings (MATCH-3750-600 = 0.1640 vs
SCAN 3750-3850 = 0.0300).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from LLM.amp_check import CHANCE, load, rescale
from LLM.patch_transformer_sca import set_seed

ROOT = Path(__file__).resolve().parents[1]


def pick_bands(curve: np.ndarray, width: int, n_bands: int, min_gap: int = 0,
               select: str = "snr", draw: int = 0) -> list[list[int]]:
    """Choose N non-overlapping windows of `width`.

    snr        - greedy top-N by mean SNR inside the window (the main experiment).
    random     - N random non-overlapping windows, `draw` gives different draws.
    spread     - N evenly spaced windows across the whole crop.
    contiguous - the first N windows, samples 0..N*width.
    low_snr    - the N LOWEST-SNR windows, so the ordering itself gets tested.
    """
    if width <= 0 or n_bands <= 0:
        return []
    kmax = max(1, len(curve) // width)
    n_bands = min(n_bands, kmax)
    means = np.array([curve[i * width:(i + 1) * width].mean() for i in range(kmax)])
    if select == "snr":
        order = np.argsort(-means)
    elif select == "low_snr":
        order = np.argsort(means)
    elif select == "contiguous":
        order = np.arange(kmax)
    elif select == "spread":
        order = np.linspace(0, kmax - 1, n_bands).round().astype(int)
    elif select == "random":
        rng = np.random.default_rng(1234 + draw)
        order = rng.permutation(kmax)
    else:
        raise ValueError(select)
    chosen: list[int] = []
    for idx in order:
        if len(chosen) >= n_bands:
            break
        if all(abs(int(idx) - c) * width >= width + min_gap for c in chosen):
            chosen.append(int(idx))
    return [[c * width, (c + 1) * width] for c in sorted(chosen)]


def train_probe(xp, xa, sp, sa, bands, args, device, rep: int):
    set_seed(args.seed + 1000 * rep)
    cols = [xp[:, lo:hi] for lo, hi in bands] or [xp]
    cols_a = [xa[:, lo:hi] for lo, hi in bands] or [xa]
    tr = torch.from_numpy(np.ascontiguousarray(np.concatenate(cols, axis=1))).to(device)
    at = torch.from_numpy(np.ascontiguousarray(np.concatenate(cols_a, axis=1))).to(device)
    if args.block > 1:
        pool = lambda t: t[:, :t.shape[1] // args.block * args.block].reshape(
            t.shape[0], -1, args.block).mean(2)
        tr, at = pool(tr), pool(at)
    mu, sd = tr.mean(0), tr.std(0).clamp_min(1e-6)
    tr, at = (tr - mu) / sd, (at - mu) / sd          # profiling statistics only
    net = nn.Sequential(nn.Linear(tr.shape[1], 256), nn.GELU(),
                        nn.Dropout(args.dropout), nn.Linear(256, 256)).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-2, weight_decay=1e-5)
    ytr = torch.from_numpy(sp).to(device)
    for _ in range(args.steps):
        opt.zero_grad()
        F.cross_entropy(net(tr), ytr).backward()
        opt.step()
    yat = torch.from_numpy(sa).to(device)
    with torch.no_grad():
        p = net(at)
        top1 = float((p.argmax(-1) == yat).float().mean())
        rank = (p.argsort(dim=-1, descending=True) == yat[:, None]).float().argmax(-1)
        res = {"n_features": int(tr.shape[1]), "top1": top1,
               "top5": float((rank < 5).float().mean()),
               "top25": float((rank < 25).float().mean()),
               "median_rank": float(rank.median()),
               "ce": float(F.cross_entropy(p, yat))}
        if args.controls:
            rng = np.random.default_rng(args.seed + 7 * rep)
            yperm = torch.from_numpy(rng.permutation(sp)).to(device)
            net2 = nn.Sequential(nn.Linear(tr.shape[1], 256), nn.GELU(),
                                 nn.Linear(256, 256)).to(device)
            opt2 = torch.optim.AdamW(net2.parameters(), lr=1e-2, weight_decay=1e-5)
            with torch.enable_grad():        # we are inside a no_grad block for the metrics
                for _ in range(args.steps):
                    opt2.zero_grad()
                    F.cross_entropy(net2(tr), yperm).backward()
                    opt2.step()
            with torch.no_grad():
                res["perm_null_top1"] = float((net2(at).argmax(-1) == yat).float().mean())
                res["const_input_top1"] = float(
                    (net(tr.mean(0, keepdim=True).expand_as(at)).argmax(-1) == yat).float().mean())
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5", default=str(ROOT / "STM32_AES_v2/ASCAD_data/ASCAD_databases/ascadv2-extracted.h5"))
    ap.add_argument("--curve", default=str(ROOT / "docs/2026-09-23/out/SNR-alpha-curve.npy"),
                    help="per-sample SNR curve from LLM/snr_check.py --curve-out")
    ap.add_argument("--field", default="alpha_mask")
    ap.add_argument("--norm", default="ztrace", choices=["raw", "ztrace", "global", "amp"])
    ap.add_argument("--sub-avg", action="store_true",
                    help="subtract the profiling mean trace before anything else")
    ap.add_argument("--width", type=int, default=100)
    ap.add_argument("--bands", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    ap.add_argument("--min-gap", type=int, default=0)
    ap.add_argument("--select", default="snr",
                    choices=["snr", "random", "spread", "contiguous", "low_snr"],
                    help="how bands are chosen; anything but snr is a control")
    ap.add_argument("--draws", type=int, default=4,
                    help="independent random band sets when --select random")
    ap.add_argument("--block", type=int, default=1)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--n-traces", type=int, default=20000)
    ap.add_argument("--n-atk", type=int, default=2000)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--controls", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    curve = np.load(args.curve)
    xp0, xa0, sp, sa = load(Path(args.h5), args.n_traces, args.n_atk, args.field)
    xp, xa = rescale(xp0, xa0, args.norm)
    if args.sub_avg:
        # Subtract the PROFILING mean trace: the trick the parallel GPAM/TimesNet work
        # credits with removing most of the static clock energy (T-020 Phase 5). Fitted
        # on profiling only, so the attack split stays untouched.
        mu = xp.mean(0, keepdims=True)
        xp, xa = xp - mu, xa - mu
    print(f"field={args.field} curve={curve.shape} prof={xp.shape} atk={xa.shape} "
          f"norm={args.norm} sub_avg={args.sub_avg} chance={CHANCE:.5f}", flush=True)
    out = {"field": args.field, "norm": args.norm, "width": args.width,
           "chance_acc": CHANCE, "repeats": args.repeats, "curve": args.curve, "curve_stats": {}}
    for nb in args.bands:
        draws = max(1, args.draws) if args.select == "random" else 1
        band_sets = [pick_bands(curve, args.width, nb, args.min_gap, args.select, d)
                     for d in range(draws)]
        band_sets = [b for b in band_sets if b]
        if not band_sets:
            continue
        rs = [train_probe(xp, xa, sp, sa, bands, args, device, rep)
              for bands in band_sets for rep in range(args.repeats)]
        t1 = [r["top1"] for r in rs]
        bands = band_sets[0]
        agg = {"n_bands": nb, "select": args.select, "n_draws": len(band_sets),
               "samples_used": sum(h - l for l, h in bands),
               "frac_of_trace": round(sum(h - l for l, h in bands) / curve.shape[0], 4),
               "bands": band_sets if len(band_sets) > 1 else bands,
               "top1_mean": float(np.mean(t1)), "top1_std": float(np.std(t1)),
               "top1_min": float(min(t1)), "top1_max": float(max(t1)),
               "top1_x_chance": float(np.mean(t1)) / CHANCE,
               "top5_mean": float(np.mean([r["top5"] for r in rs])),
               "top25_mean": float(np.mean([r["top25"] for r in rs])),
               "median_rank_mean": float(np.mean([r["median_rank"] for r in rs])),
               "all": rs}
        out["curve_stats"][f"{args.select}-{nb}"] = agg
        ctl = ""
        if args.controls:
            ctl = (f" perm={np.mean([r.get('perm_null_top1', np.nan) for r in rs]):.4f}"
                   f" const={np.mean([r.get('const_input_top1', np.nan) for r in rs]):.4f}")
        print(f"{args.select:>10s} {nb:3d} bands ({agg['samples_used']:6d} samples, "
              f"{agg['frac_of_trace']*100:5.2f}% of crop, {len(band_sets)} draws)  "
              f"top1={agg['top1_mean']:.4f}+-{agg['top1_std']:.4f} "
              f"[{agg['top1_min']:.4f}..{agg['top1_max']:.4f}] = "
              f"{agg['top1_x_chance']:5.1f}x  top5={agg['top5_mean']:.3f} "
              f"top25={agg['top25_mean']:.3f} medrank={agg['median_rank_mean']:5.1f}{ctl}",
              flush=True)
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
