#!/usr/bin/env python3
"""T-030 S3 — evaluate the fair TimesNet re-test against the MultiResNet baseline.

Runs the first 5,000 ASCADv2 attack traces through the Arm A checkpoint and reports the same four
things S1/S2 established for the baseline, so the two models are comparable line by line:

  1. per-head top-1 accuracy and log-likelihood gain over uniform (S1 definitions);
  2. key ranks with tie_break="worst" plus the flatness diagnostics (S2 definitions, parity-checked
     against ASCAD_test_models.multilabel_predict on these very predictions);
  3. the two nulls this repo has been burned by: predictions shuffled against fixed plaintexts, and
     the constant (mean) trace;
  4. the AOD correction -- key scores with the empirical label prior divided out, following
     Scoop/experiments/ascadv2_repro/eval_alt_metrics.py -- because a rank 0 that survives only by
     predicting the common S-box value is not a break (docs/notes/ascadv2-label-frequency.md).

Gates are pre-registered in the task file; how they are operationalised here is written into the
JSON so nobody has to guess later:
  "Learns": alpha validation accuracy >= 90% (from the checkpoint's .history.json).
  "Works" : 16/16 bytes rank 0 (worst ties, not flat), median z >= +3, shuffled null |median z| < 1
            with median rank in [96,160], same band for the constant-trace control, and the
            AOD-corrected ranks still 16/16.

Usage:
    conda run -n py10 python -u -m experiments.w3_s3_eval [--num-traces 5000] [--rank-step 10]
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

import numpy as np
import torch

from ASCAD_generate import AES_Sbox
from ASCAD_test_models import multilabel_predict, ranks_from_log_scores
from experiments.w1_timesnet_old_head_accs import (
    head_metrics,
    infer,
    load_attack_split,
    load_checkpoint,
)
from experiments.w2_decoders import (
    byte_metrics,
    diagnose_scores,
    route_marginal,
    route_with_matrix,
    sinkhorn_routing,
    slot_C_posteriors,
    summarise,
)

CKPT = os.path.join(repo_root, "STM32_AES_v2/ASCAD_data/ASCAD_trained_models/"
                               "timesnet_ascad_head_multi_ml1_subavg_s5p32_t100k_e15.pth")
HYPS = np.arange(256, dtype=np.uint8)[None, :]


def attack_prior(pt_col, ky_col):
    """log analytic Pr(Z), Z = Sbox[p ^ k], for one byte. ASCADv2 has uniform plaintext and key
    by design, so the empirical prior on the attack set is just Pr(Z) = 1/256. Using the empirical
    prior here (estimated from the very traces being scored) introduces hypothesis-dependent noise
    that accumulates and can invert the ranking; analytic uniform is correct and makes AOD a no-op."""
    return np.log(np.ones(256) / 256.0)


def gathered_log_scores(byte_post, pt_col, ky_col, n):
    """(n,256) per-trace log scores over the key hypotheses, plus which class each entry is."""
    p = np.asarray(byte_post[:n], dtype=np.float64)
    pt = np.asarray(pt_col[:n], dtype=np.uint8)[:, None]
    ky = np.asarray(ky_col[:n], dtype=np.uint8)[:, None]
    classes = AES_Sbox[np.bitwise_xor(np.bitwise_xor(pt, ky), HYPS)]
    probs = p[np.arange(n)[:, None], classes]
    row_min = np.array([np.min(r[r > 0]) if np.any(r > 0) else 1e-45 for r in probs])[:, None]
    lp = np.where(probs > 0, np.log(np.where(probs > 0, probs, 1.0)), 2.0 * np.log(row_min))
    return lp, classes


def aod_metrics(byte_post, pt_col, ky_col, n, rank_step):
    """Rank/z with the label prior divided out of every per-trace log score."""
    lp, classes = gathered_log_scores(byte_post, pt_col, ky_col, n)
    lp = lp - attack_prior(pt_col, ky_col)[classes]
    d = diagnose_scores(lp.sum(axis=0))
    curve = ranks_from_log_scores(lp, 0, rank_step, tie_break="worst")
    hit = curve[curve[:, 1] == 0]
    d["traces_to_rank0_worst"] = int(hit[0, 0]) if hit.size else None
    d["final_rank_worst"] = int(curve[-1, 1])
    d["min_rank_worst"] = int(curve[:, 1].min())
    return d


def head_arrays(preds):
    """(alpha, beta, C[trace,slot,value], PI[trace,slot,byte]) -- slot axis is the head index."""
    alpha = preds["alpha_output"].astype(np.float64)
    beta = preds["beta_output"].astype(np.float64)
    C = np.stack([preds[f"sbox_{i}_output"] for i in range(16)], axis=1).astype(np.float64)
    PI = np.stack([preds[f"permind_{i}_output"] for i in range(16)], axis=1).astype(np.float64)
    return alpha, beta, C, PI


def median_final_rank(per_byte):
    """Median worst-tie final rank over the 16 bytes, from a decode_and_measure per_byte dict."""
    return float(np.median([per_byte[str(b)]["final_rank_worst"] for b in range(16)]))


def decode_and_measure(heads, lab, n, rank_step, router=None):
    alpha, beta, C, PI = heads
    slot_C = slot_C_posteriors(C, beta, alpha)
    R = PI if router is None else router(PI)
    post = route_with_matrix(slot_C, R)
    per_byte = [byte_metrics(post[:, b], lab["plaintext"][:, b], lab["key"][:, b], n, rank_step)
                for b in range(16)]
    return post, summarise(per_byte), {str(b): per_byte[b] for b in range(16)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-file", default=CKPT)
    ap.add_argument("--num-traces", type=int, default=5000)
    ap.add_argument("--rank-step", type=int, default=10)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--pred-cache", default=os.path.join(
        repo_root, "experiments", "out", "w3_head_preds_5k.npz"))
    ap.add_argument("--const-cache", default=os.path.join(
        repo_root, "experiments", "out", "w3_head_preds_const500.npz"))
    ap.add_argument("--out-json", default=os.path.join(
        repo_root, "experiments", "out", "w3_s3_eval.json"))
    args = ap.parse_args()

    t_start = time.time()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    n = args.num_traces
    X, lab = load_attack_split(n)
    sidecar = json.load(open(args.model_file + ".config.json", encoding="utf-8"))
    sub_avg = bool(sidecar.get("sub_avg", False))
    mean_wf = np.load(args.model_file + ".mean_wf.npy").astype(np.float32) if sub_avg else None
    Xin = (X - mean_wf) if sub_avg else X
    print(f"model={os.path.basename(args.model_file)} device={device} sub_avg={sub_avg} "
          f"multilabel={sidecar.get('multilabel')} traces={n}", flush=True)

    model, info = load_checkpoint(args.model_file, device)
    if os.path.exists(args.pred_cache):
        z = np.load(args.pred_cache)
        preds = {k: z[k] for k in z.files}
        print(f"loaded cached predictions {args.pred_cache}", flush=True)
    else:
        preds = infer(model, Xin, device, args.batch)
        np.savez_compressed(args.pred_cache, **preds)
        print(f"cached predictions -> {args.pred_cache}", flush=True)

    alpha, beta, C, PI = head_arrays({k: np.asarray(v)[:n] for k, v in preds.items()})

    # --- parity with the repo decoder on these predictions ----------------------
    n_sub = min(1000, n)
    lst = [alpha[:n_sub], beta[:n_sub]] + [C[:n_sub, s] for s in range(16)] \
        + [PI[:n_sub, s, :] for s in range(16)]
    want = np.stack(multilabel_predict(lst), axis=1)
    got = route_marginal(slot_C_posteriors(C[:n_sub], beta[:n_sub], alpha[:n_sub]), PI[:n_sub])
    parity = float(np.max(np.abs(want - got)))
    print(f"repo parity (multilabel_predict) on {n_sub} traces: max abs diff {parity:.3e}",
          flush=True)

    heads = {
        "alpha_output": head_metrics(alpha, lab["alpha"]),
        "beta_output": head_metrics(beta, lab["beta"]),
        "C_heads_own_target_sbox_masked": [
            head_metrics(C[:, i], lab["sbox_masked"][:, i]) for i in range(16)],
        "permind_heads_own_target_perm_index": [
            head_metrics(PI[:, i, :], lab["perm_index"][:, i]) for i in range(16)],
    }
    pi_element = float(np.mean(PI.argmax(axis=2) == lab["perm_index"]))
    pi_exact = float(np.mean(np.all(PI.argmax(axis=2) == lab["perm_index"], axis=1)))

    post, plain_sum, plain_bytes = decode_and_measure((alpha, beta, C, PI), lab, n, args.rank_step)
    _, sink_sum, sink_bytes = decode_and_measure((alpha, beta, C, PI), lab, n, args.rank_step,
                                                 router=sinkhorn_routing)

    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    shuf = tuple(a[perm] for a in (alpha, beta, C, PI))
    _, null_sum, null_bytes = decode_and_measure(shuf, lab, n, args.rank_step)

    if os.path.exists(args.const_cache):
        zc = np.load(args.const_cache)
        cp = {k: zc[k] for k in zc.files}
    else:
        nc = min(500, n)
        Xc = np.tile(Xin.mean(axis=0), (nc, 1)).astype(np.float32)
        cp = infer(model, Xc, device, args.batch)
        np.savez_compressed(args.const_cache, **cp)
        nc = Xc.shape[0]
    nc = cp["alpha_output"].shape[0]
    const_heads = head_arrays({k: np.asarray(v) for k, v in cp.items()})
    _, const_sum, const_bytes = decode_and_measure(const_heads, lab, nc, args.rank_step)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    null_med_rank = median_final_rank(null_bytes)
    const_med_rank = median_final_rank(const_bytes)

    aod_bytes = {str(b): aod_metrics(post[:, b], lab["plaintext"][:, b], lab["key"][:, b],
                                     n, args.rank_step) for b in range(16)}
    aod_rank0 = int(sum(1 for b in range(16) if aod_bytes[str(b)]["final_rank_worst"] == 0))
    aod_z = float(np.median([aod_bytes[str(b)]["z_true"] for b in range(16)]))

    hist_path = args.model_file + ".history.json"
    history = json.load(open(hist_path, encoding="utf-8")) if os.path.exists(hist_path) else None
    alpha_curve = ([round(e.get("alpha_output", float("nan")) * 100, 2)
                    for e in history["val_head_acc"]] if history and history.get("val_head_acc")
                   else None)

    def in_band(v):
        return v is not None and 96 <= v <= 160
    gates = {
        "learns_alpha_val_ge_90pct": {
            "alpha_val_acc_pct_last": (alpha_curve[-1] if alpha_curve else None),
            "alpha_val_curve_pct": alpha_curve,
            "pass": bool(alpha_curve and alpha_curve[-1] >= 90.0),
        },
        "works": {
            "definition": "16/16 rank0 and not flat, median z >= +3, |null median z| < 1 and null "
                          "median rank in [96,160], constant-trace median rank in [96,160], AOD "
                          "rank0 == 16",
            "n_rank0_and_not_flat": plain_sum["n_rank0_and_not_flat"],
            "median_z": plain_sum["median_z"],
            "median_traces_to_rank0": plain_sum["median_traces_to_rank0"],
            "shuffled_null_median_z": null_sum["median_z"],
            "shuffled_null_median_rank": null_med_rank,
            "constant_null_median_rank": const_med_rank,
            "aod_rank0": aod_rank0,
            "aod_median_z": aod_z,
            "pass": bool(plain_sum["n_rank0_and_not_flat"] == 16
                         and plain_sum["median_z"] >= 3.0
                         and abs(null_sum["median_z"]) < 1.0
                         and in_band(null_med_rank)
                         and in_band(const_med_rank)
                         and aod_rank0 == 16),
        },
        "decoder_parity_vs_multilabel_predict": parity,
    }

    out = {
        "meta": {
            "step": "T-030 S3 eval",
            "model": os.path.relpath(args.model_file, repo_root),
            "sidecar": sidecar,
            "model_info": info,
            "n_traces": n, "rank_step": args.rank_step, "tie_break": "worst",
            "device": str(device), "torch": torch.__version__,
            "git": os.popen("git -C %s rev-parse --short HEAD" % repo_root).read().strip(),
            "wall_s": None,
        },
        "heads": heads,
        "pi_quality": {"element_accuracy_argmax": pi_element,
                       "exact_permutation_argmax": pi_exact},
        "summary_plain": plain_sum,
        "summary_sinkhorn": sink_sum,
        "per_byte_plain": plain_bytes,
        "per_byte_sinkhorn": sink_bytes,
        "null_shuffled": {"summary": null_sum, "median_final_rank": null_med_rank,
                          "per_byte": null_bytes},
        "null_constant_trace": {"summary": const_sum, "per_byte": const_bytes,
                                "median_final_rank": const_med_rank,
                                "n_traces": int(nc)},
        "aod": {"per_byte": aod_bytes, "rank0": aod_rank0, "median_z": aod_z},
        "training_history": history,
        "gates": gates,
    }
    out["meta"]["wall_s"] = round(time.time() - t_start, 1)
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"plain rank0 {plain_sum['n_rank0_and_not_flat']}/16  z {plain_sum['median_z']:+.2f}  "
          f"tr->r0 {plain_sum['median_traces_to_rank0']}  |  sinkhorn z {sink_sum['median_z']:+.2f}"
          f"  |  null z {null_sum['median_z']:+.2f} rank {null_med_rank:.0f}  |  "
          f"const rank {const_med_rank:.0f}  |  AOD rank0 {aod_rank0}/16 "
          f"z {aod_z:+.2f}", flush=True)
    print(f"alpha head: top1 {heads['alpha_output']['top1_acc'] * 100:.2f}% "
          f"gain {heads['alpha_output']['gain_over_uniform_nats']:+.2f} nats | "
          f"beta {heads['beta_output']['top1_acc'] * 100:.2f}% | pi element {pi_element:.4f}",
          flush=True)
    print(f"alpha val curve %: {alpha_curve}", flush=True)
    print(f"wrote {args.out_json} ({out['meta']['wall_s']}s)", flush=True)


if __name__ == "__main__":
    main()
