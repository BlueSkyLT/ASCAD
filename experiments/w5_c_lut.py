#!/usr/bin/env python3
"""T-031 W2 — the C-head lookup-table (LUT) replacement, measured on the cached baseline.

The task file asks whether the 16 masked-S-box heads can be replaced by deterministic algebra,
`sbox_masked[slot] = multGF256(alpha, Sbox[p XOR k]) XOR beta`, through a pre-built lookup table
instead of being learned. Two things have to be separated to answer that, and both are measured here
on `experiments/out/head_preds_5k.npz` (34 heads x 5,000 ASCADv2 attack traces). Nothing is retrained
and no GPU is used.

  A. THE ALGEBRA IS ALREADY IN THE DECODER. `proba_dissect_beta` (XOR by beta) followed by
     `proba_dissect_alpha` (multiply by alpha in GF(2^8)) ARE this LUT, pushed forward through the
     alpha and beta posteriors and marginalised. T-030 S2 already showed the sequential dissection
     equals an explicit 256x256 joint sum to 3.5e-16. So "use a LUT" only changes anything if the
     marginalisation is also replaced by a point mass at the predicted (alpha_hat, beta_hat) -- which
     is what D1b below measures.

  B. THE OBSERVATION QUESTION. The masked value V is the only quantity in the decode path that is
     both a function of the key and measured from the trace. alpha, beta and perm_index are read off
     the mask vector (`ASCAD_generate.py:99-121`: masks[18], masks[17], masks[0..3]) and are
     key-independent by construction, so a model with no C head has no V observation at all -- and a
     lookup table is a function, not a measurement. D1a measures what is left in that case, D2 what a
     C-less model can decode, D1c the best any mask-only shortcut could ever do.

Variants:
  D0   repo dissection                 LUT marginalised over p(alpha|t), p(beta|t)   [reference]
  D1a  LUT, no V observation           probability of the key-implied masked value from the mask
                                       posteriors alone -- the literal reading of step 1
  D1b  LUT at (alpha_hat,beta_hat)     C-head posterior kept, masks plugged in at their argmax
  D1c  same with TRUE alpha,beta       upper bound on any mask-only algebraic shortcut
  D2   C head uniform                  what a model trained without C heads would decode
  D3   oracle C                        true masked value, near-hard one-hot (S2's +6.28 z bound)

Gates (pre-registered in docs/tasks/T031-c-head-investigation.md, unmovable):
  sanity   baseline reproduces 16/16 rank 0 at median z +9.50
  improve  "the LUT model reproduces the baseline's 16/16 rank-0, or improves it"

Usage:
    conda run -n py10 python -u -m experiments.w5_c_lut [--num-traces 5000] [--rank-step 10]
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

from ASCAD_generate import AES_Sbox
from ASCAD_test_models import multGF256, multilabel_predict, ranks_from_log_scores
from experiments.w1_timesnet_old_head_accs import head_metrics, load_attack_split
from experiments.w2_decoders import (
    EPS_ORACLE,
    diagnose_scores,
    one_hot,
    route_marginal,
    slot_C_posteriors,
    summarise,
)

H5 = os.path.join(repo_root, "STM32_AES_v2/ASCAD_data/ASCAD_databases/ascadv2-extracted.h5")
CACHE = os.path.join(repo_root, "experiments", "out", "head_preds_5k.npz")
DEFAULT_OUT = os.path.join(repo_root, "experiments", "out", "w5_c_lut.json")
HYPS = np.arange(256, dtype=np.uint8)[None, :]
VALUES = np.arange(256, dtype=np.uint8)[None, :]


def build_lut():
    """LUT[a, c] = multGF256(a, c). V = LUT[alpha, Sbox[p^k]] ^ beta."""
    lut = np.zeros((256, 256), dtype=np.uint8)
    for a in range(256):
        for c in range(256):
            lut[a, c] = multGF256(a, c)
    return lut


LUT = None  # built in main()


def log_picked(head_p, values):
    """log head_p[t, values[t,k]] with rank()'s zero floor: (row min non-zero)^2."""
    p = np.asarray(head_p, dtype=np.float64)
    n = p.shape[0]
    picked = p[np.arange(n)[:, None], np.asarray(values, dtype=np.int64)]
    row_min = np.array([np.min(r[r > 0]) if np.any(r > 0) else 1e-45 for r in p])[:, None]
    return np.where(picked > 0, np.log(np.where(picked > 0, picked, 1.0)), 2.0 * np.log(row_min))


def byte_scores_from_posterior(posterior, pt_col, ky_col, n):
    """(n,256) per-trace log scores over key hypotheses, from a posterior over the S-box value."""
    idx = AES_Sbox[np.bitwise_xor(np.bitwise_xor(pt_col[:n, None].astype(np.uint8),
                                                 ky_col[:n, None].astype(np.uint8)), HYPS)]
    return log_picked(posterior[:n], idx)


def pushforward_density_at(alpha_p, beta_p, c, want, n):
    """m[t, want[t,k]] = sum_a A[t,a] * B[t, want[t,k] ^ (a * c_t)] for one hypothesis k."""
    base = LUT[:, c].T.astype(np.int64)                    # (n,256) over a
    idx = (want[:, None] ^ base)                           # (n,256)
    w = alpha_p[:n]                                        # (n,256) weights over a
    return np.einsum("na,na->n", w, np.take_along_axis(beta_p[:n], idx, axis=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--num-traces", type=int, default=5000)
    ap.add_argument("--rank-step", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0, help="smoke test on the first N traces")
    ap.add_argument("--out-json", default=DEFAULT_OUT)
    args = ap.parse_args()

    global LUT
    t_start = time.time()
    LUT = build_lut()
    assert all(int(LUT[a, b]) == multGF256(a, b) for a in (0, 1, 3, 57, 255)
               for b in (0, 1, 2, 88, 255)), "LUT mismatch"

    n = args.limit or args.num_traces
    z = np.load(CACHE)
    if n > z["alpha_output"].shape[0]:
        raise SystemExit(f"cache holds {z['alpha_output'].shape[0]} traces")
    _, lab = load_attack_split(n)
    preds = {k: np.asarray(z[k])[:n] for k in z.files}
    alpha = preds["alpha_output"].astype(np.float64)
    beta = preds["beta_output"].astype(np.float64)
    C = np.stack([preds[f"sbox_{i}_output"] for i in range(16)], axis=1).astype(np.float64)
    PI = np.stack([preds[f"permind_{i}_output"] for i in range(16)], axis=1).astype(np.float64)
    alpha_hat = alpha.argmax(axis=1).astype(np.int64)
    beta_hat = beta.argmax(axis=1).astype(np.int64)
    alpha_true = lab["alpha"].astype(np.int64)
    beta_true = lab["beta"].astype(np.int64)

    # --- the algebra itself, checked against the stored labels ------------------
    c_true_byte = AES_Sbox[np.bitwise_xor(lab["plaintext"], lab["key"])]          # (n, byte)
    V_byte = LUT[lab["alpha"][:, None], c_true_byte] ^ lab["beta"][:, None]       # byte-indexed V
    V_slot = np.take_along_axis(V_byte, lab["perm_index"], axis=1)                # slot-indexed
    lut_matches = float(np.mean(V_slot == lab["sbox_masked"][:n]))
    print(f"traces={n}  LUT algebra reproduces stored sbox_masked on {lut_matches * 100:.4f}% of "
          f"{n * 16} values", flush=True)
    if lut_matches != 1.0:
        raise SystemExit("the LUT/algebra does not match the stored labels -- stop")

    cases = {}

    def record(name, score_list, desc):
        rows = []
        for lp in score_list:
            d = diagnose_scores(lp.sum(axis=0))
            curve = ranks_from_log_scores(lp, 0, args.rank_step, tie_break="worst")
            hit = curve[curve[:, 1] == 0]
            d["traces_to_rank0_worst"] = int(hit[0, 0]) if hit.size else None
            d["final_rank_worst"] = int(curve[-1, 1])
            rows.append(d)
        cases[name] = {"description": desc, "summary": summarise(rows),
                       "per_byte": {str(b): rows[b] for b in range(16)}}
        s = cases[name]["summary"]
        print(f"{name:26s} z {s['median_z']:+8.2f}  rank0 {s['n_rank0_final']:2d}/16 "
              f"genuine {s['n_rank0_and_not_flat']:2d}/16  flat {s['n_flat_bytes']:2d} "
              f"near_flat {s['n_near_flat_bytes']:2d}  spread {s['median_spread']:.4g}  "
              f"traces->r0 {s['median_traces_to_rank0']}", flush=True)

    # ---- D0 repo dissection ----------------------------------------------------
    lst = [alpha, beta] + [C[:, s] for s in range(16)] + [PI[:, s, :] for s in range(16)]
    sbox_d0 = multilabel_predict(lst)
    d0 = [byte_scores_from_posterior(sbox_d0[b], lab["plaintext"][:, b], lab["key"][:, b], n)
          for b in range(16)]
    record("D0_repo_dissection", d0, "LUT marginalised over the mask posteriors "
                                     "(= multilabel_predict, the current decoder)")
    fast = route_marginal(slot_C_posteriors(C, beta, alpha), PI)
    parity = float(np.max(np.abs(np.stack(sbox_d0, axis=1)[:n] - fast)))
    print(f"D0 parity vs the T-030 S2 fast path: max abs diff {parity:.3e}", flush=True)

    # ---- the masked-value mixture per byte, used by D1b/D1c/D1a ---------------
    Cmix = route_marginal(C, PI)                                  # posterior over V, per byte
    rows = np.arange(n)

    # ---- D1a: mask posteriors only, no V observation ---------------------------
    d1a = np.empty((n, 16, 256))
    for b in range(16):
        pt, ky = lab["plaintext"][:, b], lab["key"][:, b]
        for k in range(256):
            c = AES_Sbox[np.bitwise_xor(pt.astype(np.uint8), np.bitwise_xor(ky.astype(np.uint8),
                                                                           np.uint8(k)))]
            want = (LUT[alpha_hat, c] ^ beta_hat).astype(np.int64)
            d1a[:, b, k] = pushforward_density_at(alpha, beta, c, want, n)
    record("D1a_no_V_observation", [np.log(np.clip(d1a[:, b], 1e-300, None)) for b in range(16)],
           "probability of the key-implied masked value from the mask posteriors alone")
    # and the strictly literal reading: point mass at the implied value -> score is 0 for every k
    record("D1a2_point_mass", [np.zeros((n, 256)) for _ in range(16)],
           "hard one-hot LUT with no observation: every hypothesis scores identically")

    # ---- D1b / D1c: C-head posterior kept, masks plugged in -------------------
    d1b, d1c = [], []
    for b in range(16):
        pt, ky = lab["plaintext"][:, b], lab["key"][:, b]
        lp_b, lp_c = [], []
        for k in range(256):
            c = AES_Sbox[np.bitwise_xor(pt.astype(np.uint8), np.bitwise_xor(ky.astype(np.uint8),
                                                                           np.uint8(k)))]
            v_b = (LUT[alpha_hat, c] ^ beta_hat).astype(np.int64)
            v_c = (LUT[alpha_true, c] ^ beta_true).astype(np.int64)
            lp_b.append(np.log(np.clip(Cmix[:n, b][rows, v_b], 1e-300, None)))
            lp_c.append(np.log(np.clip(Cmix[:n, b][rows, v_c], 1e-300, None)))
        d1b.append(np.stack(lp_b, axis=1))
        d1c.append(np.stack(lp_c, axis=1))
    record("D1b_LUT_hard_masks", d1b, "C posterior kept; mask marginalisation replaced by the "
                                      "argmax masks")
    record("D1c_LUT_oracle_masks", d1c, "D1b with the TRUE masks: upper bound for a mask-only "
                                        "algebraic decoder")

    # ---- D2 / D3 ---------------------------------------------------------------
    C_uni = np.full(C.shape, 1.0 / 256)
    sbox_d2 = multilabel_predict([alpha, beta] + [C_uni[:, s] for s in range(16)]
                                 + [PI[:, s, :] for s in range(16)])
    record("D2_no_C_head", [byte_scores_from_posterior(sbox_d2[b], lab["plaintext"][:, b],
                                                       lab["key"][:, b], n) for b in range(16)],
           "C head uniform: what a model trained without C heads can decode")
    C_oracle = one_hot(lab["sbox_masked"][:n], eps=EPS_ORACLE)
    sbox_d3 = multilabel_predict([alpha, beta] + [C_oracle[:, s] for s in range(16)]
                                 + [PI[:, s, :] for s in range(16)])
    record("D3_oracle_C", [byte_scores_from_posterior(sbox_d3[b], lab["plaintext"][:, b],
                                                      lab["key"][:, b], n) for b in range(16)],
           f"true masked value, one-hot eps={EPS_ORACLE}: upper bound")

    # ---- null on the strongest LUT-with-head case ------------------------------
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    Cmix_s = Cmix[perm]
    d1b_null = []
    for b in range(16):
        pt, ky = lab["plaintext"][:, b], lab["key"][:, b]
        lp = []
        for k in range(256):
            c = AES_Sbox[np.bitwise_xor(pt.astype(np.uint8), np.bitwise_xor(ky.astype(np.uint8),
                                                                           np.uint8(k)))]
            v = (LUT[alpha_hat, c] ^ beta_hat).astype(np.int64)
            lp.append(np.log(np.clip(Cmix_s[:n, b][rows, v], 1e-300, None)))
        d1b_null.append(np.stack(lp, axis=1))
    record("D1b_null_shuffled", d1b_null, "D1b with head rows shuffled against fixed plaintexts")

    base = cases["D0_repo_dissection"]["summary"]
    lut_gate = {
        name: {"median_z": c["summary"]["median_z"],
               "n_rank0_and_not_flat": c["summary"]["n_rank0_and_not_flat"],
               "median_traces_to_rank0": c["summary"]["median_traces_to_rank0"],
               "passes_gate": bool(c["summary"]["n_rank0_and_not_flat"] == 16
                                   and c["summary"]["median_z"] >= base["median_z"] - 0.05)}
        for name, c in cases.items()
    }
    out = {
        "meta": {
            "step": "T-031 W2 LUT replacement of the C head",
            "cache": os.path.relpath(CACHE, repo_root), "h5": os.path.relpath(H5, repo_root),
            "n_traces": int(n), "rank_step": args.rank_step, "tie_break": "worst",
            "eps_oracle": EPS_ORACLE, "numpy": np.__version__, "python": sys.version.split()[0],
            "git": os.popen("git -C %s rev-parse --short HEAD" % repo_root).read().strip(),
            "gpu_used": False, "wall_s": None,
        },
        "algebra_checks": {
            "LUT_reproduces_stored_sbox_masked": lut_matches,
            "D0_parity_vs_S2_fast_path_max_abs_diff": parity,
            "note": "proba_dissect_beta then proba_dissect_alpha already IS the LUT pushforward, "
                    "marginalised over the mask posteriors (S2 measured that marginalisation equal "
                    "to the explicit 256x256 joint sum to 3.5e-16)",
        },
        "head_reference_recomputed": {
            "alpha": head_metrics(alpha, lab["alpha"]),
            "beta": head_metrics(beta, lab["beta"]),
            "C_mean_top1": float(np.mean([head_metrics(C[:, i], lab["sbox_masked"][:, i])["top1_acc"]
                                          for i in range(16)])),
            "C_mean_gain": float(np.mean([head_metrics(C[:, i], lab["sbox_masked"][:, i])
                                          ["gain_over_uniform_nats"] for i in range(16)])),
            "pi_mean_top1": float(np.mean([head_metrics(PI[:, i, :], lab["perm_index"][:, i])
                                           ["top1_acc"] for i in range(16)])),
            "alpha_hat_acc": float(np.mean(alpha_hat == lab["alpha"])),
            "beta_hat_acc": float(np.mean(beta_hat == lab["beta"])),
        },
        "cases": cases,
        "gates": {
            "sanity_reproduce_baseline": {
                "measured_median_z": base["median_z"], "documented": 9.50, "tolerance": 0.05,
                "n_rank0_and_not_flat": base["n_rank0_and_not_flat"],
                "pass": bool(abs(base["median_z"] - 9.50) <= 0.05
                             and base["n_rank0_and_not_flat"] == 16)},
            "improve_reproduce_or_beat_16_of_16": lut_gate,
        },
    }
    out["meta"]["wall_s"] = round(time.time() - t_start, 1)
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {args.out_json} ({out['meta']['wall_s']}s)", flush=True)


if __name__ == "__main__":
    main()
