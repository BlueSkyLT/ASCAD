#!/usr/bin/env python3
"""T-031 T31-N2 — score a C-deletion (or control) MultiResNet against the baseline.

The N2 gate is "do the alpha/beta/pi heads improve when C's 38.3% trunk-gradient sink is removed?"
(W1 measured the sink; W2 measured that a C-less DECODE is flat, so the gate cannot be about rank 0).
This script reads one model's 34 head posteriors on the same 5,000 ASCADv2 attack traces that W1, W2
and T31-N1 used, and reports that question in three currencies:

  1. per-head accuracy and likelihood (the gate's own terms: alpha, beta, the 16 pi heads),
  2. key recovery under an ORACLE C head -- the only decoder that can show trunk progress in a model
     that has no usable C head -- plus the project's own multi-attack guessing entropy,
  3. GE-vs-traces curves, from ASCAD_test_models.guessing_entropy (not an improvised scorer).

Two modes, so the baseline goes through exactly the same metric code as the new checkpoints:
  --model-file PATH   GPU inference on a checkpoint written by MultiResNet/train.py
  --npz PATH          score cached head posteriors instead (CPU; the baseline's head_preds_5k.npz)

Ceiling warning, measured rather than assumed: z saturates at sqrt(255) = 15.97 and the baseline's
oracle-C decode is already at +15.78 / 10 traces, so currency 2 has almost no headroom in z and the
discriminating number there is traces-to-rank-0 on the *held* heads, not z.

Usage:
    conda run -n py10 python -u -m experiments.w6_nochead_eval --label baseline \
        --npz experiments/out/head_preds_5k.npz --out-json experiments/out/w6_eval_baseline.json
    conda run -n py10 python -u -m experiments.w6_nochead_eval --label nochead \
        --model-file experiments/out/w6_multiresnet_nochead.pth \
        --out-json experiments/out/w6_eval_nochead.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import h5py
import numpy as np

from ASCAD_test_models import guessing_entropy
from experiments.w1_timesnet_old_head_accs import head_metrics, load_attack_split
from experiments.w2_decoders import (
    EPS_ORACLE,
    check_pi_orientation,
    one_hot,
    route_marginal,
    summarise,
    uniform_head,
)
from experiments.w5_c_lut import H5, log_picked
from experiments.w7_joint_ab_decode import exact_byte_posterior, per_byte_rows

KEYS_256 = [f"sbox_{i}_output" for i in range(16)]
KEYS_PI = [f"permind_{i}_output" for i in range(16)]


def collect_from_checkpoint(model_file, n, device_str="cuda", batch=64):
    """34 head posteriors on the first n attack traces, exactly as MultiResNet/evaluate.py does it."""
    import torch

    from MultiResNet.model import MultiResNet, CompactMultiResNet

    X, lab = load_attack_split(n)
    mean_path = model_file + ".mean_wf.npy"
    if os.path.exists(mean_path):
        X = X - np.load(mean_path).astype(np.float32)
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    sidecar_path = model_file + ".config.json"
    sidecar = {}
    if os.path.exists(sidecar_path):
        with open(sidecar_path, "r", encoding="utf-8") as f:
            sidecar = json.load(f)

    if sidecar.get("compact", False) or "compact" in sidecar.get("network_type", ""):
        bot_dim = sidecar.get("bottleneck_dim", 512)
        h_dim = sidecar.get("head_hidden_dim", 256)
        model = CompactMultiResNet(
            input_dim=X.shape[1], depth=19, without_permind=False,
            bottleneck_dim=bot_dim, head_hidden_dim=h_dim
        ).to(device)
    else:
        model = MultiResNet(input_dim=X.shape[1], depth=19, without_permind=False).to(device)
    model.load_state_dict(torch.load(model_file, map_location=device, weights_only=True))
    model.eval()
    out = {}
    with torch.no_grad():
        for i in range(0, n, batch):
            xb = torch.from_numpy(X[i:i + batch]).to(device)
            o = model(xb)                      # dict of head logits, the trainer's own layout
            for k, v in o.items():
                out.setdefault(k, []).append(torch.softmax(v, dim=-1).float().cpu().numpy())
    heads = {k: np.concatenate(v, axis=0).astype(np.float64) for k, v in out.items()}
    return heads, lab, str(device)


def load_npz(path, n):
    z = np.load(path)
    _, lab = load_attack_split(n)
    return {k: np.asarray(z[k], dtype=np.float64)[:n] for k in z.files}, lab, "cpu (cached)"


def head_block(heads, lab, n):
    a = head_metrics(heads["alpha_output"], lab["alpha"])
    b = head_metrics(heads["beta_output"], lab["beta"])
    c = [head_metrics(heads[k], lab["sbox_masked"][:, i]) for i, k in enumerate(KEYS_256)]
    p = [head_metrics(heads[k], lab["perm_index"][:, i]) for i, k in enumerate(KEYS_PI)]
    pi_exact = float(np.mean(np.all(np.stack([heads[k].argmax(1) for k in KEYS_PI], axis=1)
                                    == lab["perm_index"][:n], axis=1)))
    return {
        "alpha": a, "beta": b,
        "C_mean_top1": float(np.mean([h["top1_acc"] for h in c])),
        "C_mean_gain": float(np.mean([h["gain_over_uniform_nats"] for h in c])),
        "C_per_slot_top1": {str(i): c[i]["top1_acc"] for i in range(16)},
        "C_mean_max_prob": float(np.mean([heads[k].max(1).mean() for k in KEYS_256])),
        "pi_element_top1": float(np.mean([h["top1_acc"] for h in p])),
        "pi_element_gain": float(np.mean([h["gain_over_uniform_nats"] for h in p])),
        "pi_exact_permutation": pi_exact,
        "beta_mean_p_true": float(np.mean(heads["beta_output"][
            np.arange(n), lab["beta"][:n]])),
        "alpha_mean_p_true": float(np.mean(heads["alpha_output"][
            np.arange(n), lab["alpha"][:n]])),
    }


def decode_table(heads, lab, n, rank_step):
    """as-trained / uniform-C / oracle-C decodes, on the tie-safe convention T31-N1 used."""
    A, B = heads["alpha_output"], heads["beta_output"]
    C = np.stack([heads[k] for k in KEYS_256], axis=1)
    PI = np.stack([heads[k] for k in KEYS_PI], axis=1)
    variants = {
        "as_trained": C,
        "uniform_C": uniform_head(C.shape, 256),
        "oracle_C": one_hot(lab["sbox_masked"][:n], eps=EPS_ORACLE),
    }
    table = {}
    for name, Cv in variants.items():
        Cmix = route_marginal(Cv, PI)
        Q = np.stack([exact_byte_posterior(Cmix[:, b], A, B) for b in range(16)], axis=1)
        rows = per_byte_rows(Q, lab, n, rank_step)
        table[name] = {"summary": summarise(rows),
                       "per_byte": {str(b): rows[b] for b in range(16)}, "Q": Q}
    return table


def ge_block(byte_posterior, meta, n, rank_step, n_attacks, ge_seed):
    """The project's own multi-attack GE per byte, plus its order-invariance cross-check."""
    rows = {}
    for b in range(16):
        r = guessing_entropy(predictions=byte_posterior[:, b, :], metadata=meta, num_traces=n,
                             rank_step=rank_step, target_byte=b, n_attacks=n_attacks,
                             rng=ge_seed, simulated_key=1, return_curve=False)
        rows[str(b)] = {k: v for k, v in r.items() if k != "ge_at"}
        rows[str(b)]["ge_curve_ge_at"] = r["ge_at"]          # GE vs trace count, the repo's grid
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", required=True, help="short name for this model in the JSON")
    ap.add_argument("--model-file", default="", help="checkpoint to run inference on (GPU)")
    ap.add_argument("--npz", default="", help="cached head posteriors instead of a checkpoint")
    ap.add_argument("--num-traces", type=int, default=5000)
    ap.add_argument("--rank-step", type=int, default=10)
    ap.add_argument("--ge-attacks", type=int, default=100)
    ap.add_argument("--ge-seed", type=int, default=12345)
    ap.add_argument("--save-npz", default="", help="write the collected posteriors here")
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()
    if bool(args.model_file) == bool(args.npz):
        raise SystemExit("give exactly one of --model-file / --npz")

    t0 = time.time()
    n = args.num_traces
    if args.model_file:
        heads, lab, device = collect_from_checkpoint(args.model_file, n)
        if args.save_npz:
            np.savez_compressed(args.save_npz, **heads)
            print(f"cached head posteriors -> {args.save_npz}", flush=True)
    else:
        heads, lab, device = load_npz(args.npz, n)

    missing = [k for k in ["alpha_output", "beta_output", *KEYS_256, *KEYS_PI] if k not in heads]
    if missing:
        raise SystemExit(f"model exposes no head(s): {missing}")
    PI = np.stack([heads[k] for k in KEYS_PI], axis=1)
    pi_acc = check_pi_orientation(PI, lab["perm_index"][:n], 0.6703) \
        if args.label.startswith("baseline") else float(np.mean(PI.argmax(2) == lab["perm_index"][:n]))
    print(f"[{args.label}] device={device} n={n} pi element acc {pi_acc:.4f}", flush=True)

    tbl = decode_table(heads, lab, n, args.rank_step)
    with h5py.File(H5, "r") as f:
        meta = f["Attack_traces/metadata"][:n]

    ge = {}
    for name in ("as_trained", "uniform_C", "oracle_C"):
        rows = ge_block(tbl[name]["Q"], meta, n, args.rank_step, args.ge_attacks, args.ge_seed)
        finals = [rows[str(b)]["ge_final"] for b in range(16)]
        mine = [tbl[name]["per_byte"][str(b)]["final_rank_worst"] for b in range(16)]
        ge[name] = {"per_byte": rows,
                    "median_ge_final": float(np.median(finals)),
                    "max_ge_final": float(np.max(finals)),
                    "cross_check_ge_final_vs_this_scripts_rank_max_abs_diff":
                        float(np.max(np.abs(np.array(finals) - np.array(mine, dtype=float))))}
        print(f"[{args.label}] GE {name}: median ge_final {np.median(finals):.2f} "
              f"cross-check vs own ranks "
              f"{ge[name]['cross_check_ge_final_vs_this_scripts_rank_max_abs_diff']:.3e}", flush=True)

    history = None
    if args.model_file:
        hp = args.model_file + ".history.json"
        if os.path.exists(hp):
            with open(hp, encoding="utf-8") as f:
                h = json.load(f)
            history = {
                "config": h.get("config"), "epochs": h["epochs"],
                "epoch_seconds": h["epoch_seconds"], "train_loss": h["train_loss"],
                "val_loss": h["val_loss"],
                "val_head_top1_mean_C": [float(np.mean([h["val_head_top1"][f"sbox_{i}_output"][e]
                                                        for i in range(16)]))
                                         for e in range(len(h["epochs"]))],
                "val_head_top1_mean_pi": [float(np.mean([h["val_head_top1"][f"permind_{i}_output"][e]
                                                          for i in range(16)]))
                                          for e in range(len(h["epochs"]))],
                "val_head_top1_alpha": h["val_head_top1"]["alpha_output"],
                "val_head_top1_beta": h["val_head_top1"]["beta_output"],
                "val_head_ce_alpha": h["val_head_ce"]["alpha_output"],
                "val_head_ce_beta": h["val_head_ce"]["beta_output"],
                "val_head_ce_C_mean": [float(np.mean([h["val_head_ce"][f"sbox_{i}_output"][e]
                                                      for i in range(16)]))
                                       for e in range(len(h["epochs"]))],
            }

    out = {
        "meta": {"label": args.label, "step": "T-031 T31-N2 head-quality evaluation",
                 "model_file": args.model_file or None, "npz": args.npz or None,
                 "device": device, "n_traces": int(n), "rank_step": args.rank_step,
                 "tie_break": "worst", "ge_attacks": args.ge_attacks, "ge_seed": args.ge_seed,
                 "eps_oracle": EPS_ORACLE, "numpy": np.__version__,
                 "git": os.popen("git -C %s rev-parse --short HEAD" % repo_root).read().strip(),
                 "gpu_used": bool(args.model_file), "wall_s": round(time.time() - t0, 1)},
        "pi_element_acc": pi_acc,
        "heads": head_block(heads, lab, n),
        "decoders": {k: {"summary": v["summary"], "per_byte": v["per_byte"]}
                     for k, v in tbl.items()},
        "guessing_entropy": ge,
        "training_history": history,
    }
    s = out["decoders"]["oracle_C"]["summary"]
    st = out["decoders"]["as_trained"]["summary"]
    print(f"[{args.label}] heads: alpha {out['heads']['alpha']['top1_acc']:.4f} "
          f"(gain {out['heads']['alpha']['gain_over_uniform_nats']:+.4f})  "
          f"beta {out['heads']['beta']['top1_acc']:.4f} "
          f"(gain {out['heads']['beta']['gain_over_uniform_nats']:+.4f})  "
          f"C {out['heads']['C_mean_top1']:.4f} ({out['heads']['C_mean_gain']:+.4f})  "
          f"pi {out['heads']['pi_element_top1']:.4f}", flush=True)
    print(f"[{args.label}] as-trained z {st['median_z']:+.4f} r0 {st['n_rank0_and_not_flat']}/16 "
          f"flat {st['n_flat_bytes']} | oracle-C z {s['median_z']:+.4f} "
          f"traces->r0 {s['median_traces_to_rank0']}", flush=True)
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {args.out_json} ({out['meta']['wall_s']}s)", flush=True)


if __name__ == "__main__":
    main()
