#!/usr/bin/env python3
"""Does per-trace normalization erase the ASCADv2 multiply mask alpha?

Every loader in `LLM/data.py` applies `zscore_per_trace` (data.py:124-128), and the
patch-transformer line that produced FAIL_CHANCE on `alpha_mask` used it. alpha is a
GF(256) MULTIPLY hide, so if its leakage rides on trace amplitude, dividing each trace
by its own std removes it before the model ever sees it. This script measures the same
linear probe under three scalings and nothing else changes.

    python -u -m LLM.amp_check --n-traces 20000 --n-atk 2000
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from LLM.patch_transformer_sca import set_seed

ROOT = Path(__file__).resolve().parents[1]
CHANCE = 1.0 / 255.0     # class 0 of alpha_mask never occurs


def load(h5: Path, n_prof: int, n_atk: int, field: str):
    with h5py.File(h5, "r") as f:
        xp = np.array(f["Profiling_traces/traces"][:n_prof], dtype=np.float32)
        xa = np.array(f["Attack_traces/traces"][:n_atk], dtype=np.float32)
        sp = np.array(f["Profiling_traces/labels"][:n_prof][field][:, 0], dtype=np.int64)
        sa = np.array(f["Attack_traces/labels"][:n_atk][field][:, 0], dtype=np.int64)
    return xp, xa, sp, sa


def rescale(xp: np.ndarray, xa: np.ndarray, mode: str):
    """Fit every transform on the PROFILING split only."""
    if mode == "raw":
        return xp.copy(), xa.copy()
    if mode == "ztrace":
        f = lambda a: (a - a.mean(1, keepdims=True)) / np.maximum(a.std(1), 1e-6)[:, None]
        return f(xp), f(xa)
    if mode == "global":                       # one mean, one std for the dataset
        mu, sd = xp.mean(), xp.std() + 1e-6
        return (xp - mu) / sd, (xa - mu) / sd
    if mode == "amp":                          # keep amplitude, remove baseline only
        mu = xp.mean(1, keepdims=True)
        return xp - mu, xa - xp.mean(1, keepdims=True).mean(0, keepdims=True)
    raise ValueError(mode)


def probe(xp, xa, sp, sa, args, device, rep: int = 0) -> dict:
    """Linear or small-MLP probe on a sample window. Also reports amplitude stats.

    Seeded per repetition. Unseeded this produced 0.0300 and 0.1640 for the *identical*
    window and step count (MATCH-3750-600 vs SCAN-alpha-ztrace), so a single number is
    mostly luck of initialisation. Report the spread across repetitions.
    """
    set_seed(args.seed + 1000 * rep)
    stats = {}
    for name, fn in {"std": lambda a: a.std(1), "max": lambda a: a.max(1),
                     "absmean": lambda a: np.abs(a).mean(1),
                     "p95": lambda a: np.percentile(a, 95, axis=1),
                     "range": lambda a: a.max(1) - a.min(1)}.items():
        v_tr, v_at = fn(xp), fn(xa)
        stats[name] = {"pearson_alpha_tr": float(np.corrcoef(v_tr, sp)[0, 1]),
                       "pearson_alpha_at": float(np.corrcoef(v_at, sa)[0, 1])}

    lo, hi = args.window
    xp, xa = xp[:, lo:hi], xa[:, lo:hi]
    xp_t = torch.from_numpy(np.ascontiguousarray(xp)).to(device)
    xa_t = torch.from_numpy(np.ascontiguousarray(xa)).to(device)
    sp_t = torch.from_numpy(sp).to(device)
    sa_t = torch.from_numpy(sa).to(device)
    b = args.block
    def pool(x):
        if b <= 1:
            return x
        return x[:, :x.shape[1] // b * b].reshape(x.shape[0], -1, b).mean(2)
    xp_p, xa_p = pool(xp_t), pool(xa_t)
    mu, sd = xp_p.mean(0), xp_p.std(0).clamp_min(1e-6)
    xp_p, xa_p = (xp_p - mu) / sd, (xa_p - mu) / sd

    if args.mlp:
        net = torch.nn.Sequential(torch.nn.Linear(xp_p.shape[1], 256), torch.nn.GELU(),
                                  torch.nn.Linear(256, 256)).to(device)
        params, flat = net.parameters(), False
    else:
        W = torch.zeros(xp_p.shape[1], 256, device=device, requires_grad=True)
        net, params, flat = None, [W], True
    opt = torch.optim.AdamW(params, lr=1e-2, weight_decay=1e-5)
    for _ in range(args.steps):
        opt.zero_grad()
        logits = xp_p @ W if flat else net(xp_p)
        F.cross_entropy(logits, sp_t).backward()
        opt.step()
    with torch.no_grad():
        probs = (xa_p @ W) if flat else net(xa_p)
        acc = float((probs.argmax(-1) == sa_t).float().mean())
        ce = float(F.cross_entropy(probs, sa_t))
        # Top-k hit rates and the rank of the true alpha: 12x chance is a weak claim,
        # "the true alpha sits at median rank N of 255" is one a meeting can use.
        # Blind reference for 255 classes: top-5 2.0%, top-25 9.8%, median rank 127.
        srt = probs.argsort(dim=-1, descending=True)
        pos = (srt == sa_t[:, None]).float().argmax(dim=-1)
        topk = {k: float((pos < k).float().mean()) for k in (1, 5, 25, 64)}
    res = {"linear_acc": acc, "linear_ce": ce, "n_features": int(xp_p.shape[1]),
           "window": [lo, hi], "mlp": bool(args.mlp), "topk": topk,
           "median_rank_of_true": float(pos.median()), "amplitude_stats": stats}

    if args.controls:
        # (i) label-permutation null: same pipeline, shuffled training labels.
        # (ii) constant input: every attack trace replaced by the training mean.
        rng = np.random.default_rng(args.seed + 1)
        sp_perm = torch.from_numpy(rng.permutation(sp)).to(device)
        if flat:
            W2 = torch.zeros_like(W, requires_grad=True)
            net2, params2 = None, [W2]
        else:
            net2 = torch.nn.Sequential(
                torch.nn.Linear(xp_p.shape[1], 256), torch.nn.GELU(),
                torch.nn.Linear(256, 256)).to(device)
            params2, W2 = net2.parameters(), None
        opt = torch.optim.AdamW(params2, lr=1e-2, weight_decay=1e-5)
        for _ in range(args.steps):
            opt.zero_grad()
            F.cross_entropy((xp_p @ W2) if flat else net2(xp_p), sp_perm).backward()
            opt.step()
        with torch.no_grad():
            p2 = (xa_p @ W2) if flat else net2(xa_p)
            res["perm_null_acc"] = float((p2.argmax(-1) == sa_t).float().mean())
            xbar = xp_p.mean(0, keepdim=True).expand_as(xa_p)
            p3 = (xbar @ W) if flat else net(xbar)
            res["const_input_acc"] = float((p3.argmax(-1) == sa_t).float().mean())
    return res


def scan(args, device) -> dict:
    """Readability profile: train the probe separately in every window.

    The window that worked was picked from an SNR scan on profiling labels, so the
    honest objection is cherry-picking. Here nothing is chosen: every window gets its
    own probe, and the accuracy profile is compared with LLM/snr_check.py's variance
    profile. If learned accuracy peaks where SNR peaks, the SNR map is doing real work.
    """
    import h5py
    xp, xa, sp, sa = load(Path(args.h5), args.n_traces, args.n_atk, args.field)
    xp, xa = rescale(xp, xa, args.norm)
    out = {"field": args.field, "norm": args.norm, "width": args.width,
           "step": args.step, "chance_acc": CHANCE, "windows": []}
    for lo in range(0, xp.shape[1] - args.width + 1, args.step):
        args.window = [lo, lo + args.width]
        r = probe(xp, xa, sp, sa, args, device)
        out["windows"].append({"start": lo, "acc": round(r["linear_acc"], 5),
                              "x_chance": round(r["linear_acc"] / CHANCE, 2)})
        print(f"{lo:6d}-{lo + args.width:<6d} acc={r['linear_acc']:.4f} "
              f"({r['linear_acc'] / CHANCE:5.2f}x chance)", flush=True)
    best = max(out["windows"], key=lambda w: w["acc"])
    out["best_window"] = best
    print(f"BEST window {best['start']}-{best['start'] + args.width} "
          f"acc={best['acc']:.4f} ({best['x_chance']}x); SNR peak was 867", flush=True)
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(out, indent=2))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5", default=str(ROOT / "STM32_AES_v2/ASCAD_data/ASCAD_databases/ascadv2-extracted.h5"))
    ap.add_argument("--field", default="alpha_mask",
                    choices=["alpha_mask", "beta_mask", "sbox_masked"],
                    help="sbox_masked is read at column 0, i.e. masked S-box of byte 0")
    ap.add_argument("--n-traces", type=int, default=20000)
    ap.add_argument("--n-atk", type=int, default=2000)
    ap.add_argument("--block", type=int, default=10)
    ap.add_argument("--window", type=int, nargs=2, default=[0, 15000],
                    help="sample range to keep; LLM/snr_check.py found alpha peaking at 867")
    ap.add_argument("--mlp", action="store_true", help="256-256 MLP instead of one linear map")
    ap.add_argument("--controls", action="store_true",
                    help="also run a label-permutation null and a constant-input null")
    ap.add_argument("--scan", action="store_true",
                    help="readability profile: one independent probe per window")
    ap.add_argument("--norm", default="ztrace", choices=["raw", "ztrace", "global", "amp"])
    ap.add_argument("--width", type=int, default=100)
    ap.add_argument("--step", type=int, default=250)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--repeats", type=int, default=1,
                    help="independent initialisations; report the spread, not one draw")
    ap.add_argument("--only-mode", default=None, choices=["raw", "ztrace", "global", "amp"],
                    help="run a single scaling instead of all four")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.scan:
        scan(args, device)
        return
    xp, xa, sp, sa = load(Path(args.h5), args.n_traces, args.n_atk, args.field)
    print(f"loaded {xp.shape} / {xa.shape} field={args.field} "
          f"label range {sp.min()}-{sp.max()} classes={len(np.unique(sp))}", flush=True)
    out = {"field": args.field, "chance_acc": CHANCE, "chance_ce": math.log(256.0),
           "repeats": args.repeats, "modes": {}}
    for mode in ("raw", "ztrace", "global", "amp"):
        if args.only_mode:
            modes = [mode] if mode == args.only_mode else []
            if not modes:
                continue
        m_xp, m_xa = rescale(xp, xa, mode)
        rs = [probe(m_xp, m_xa, sp, sa, args, device, rep=r) for r in range(args.repeats)]
        accs = [r["linear_acc"] for r in rs]
        r = dict(rs[0])
        r.update({"acc_all": accs, "acc_mean": float(np.mean(accs)),
                  "acc_std": float(np.std(accs)), "acc_min": float(min(accs)),
                  "acc_max": float(max(accs))})
        out["modes"][mode] = r
        print(f"{mode:7s} acc_mean={r['acc_mean']:.4f}+-{r['acc_std']:.4f} "
              f"[{r['acc_min']:.4f}..{r['acc_max']:.4f}] n={args.repeats} "
              f"({r['acc_mean']/CHANCE:.1f}x chance) "
              f"top25={r.get('topk', {}).get('25', -1):.3f} "
              f"medrank={r.get('median_rank_of_true', -1):.0f}", flush=True)
    print(json.dumps(out, indent=2), flush=True)
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
