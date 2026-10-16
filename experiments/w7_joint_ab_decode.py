#!/usr/bin/env python3
"""T-031 T31-N1 — does the one-(alpha,beta)-per-trace constraint unlock any decoding gain?

W2 (2026-10-06) bracketed the cost of mask uncertainty on the same weak C head: true masks
z +12.93 / 95 traces, argmax masks plugged in z +5.11 / 1,065, marginalised (the repo decoder)
+9.50 / 170. This step tests the task file's hypothesis that the constraint "one (alpha, beta) pair
serves all 16 slots of a trace" can be exploited by the decoder, and measures mask calibration.
CPU only, on `experiments/out/head_preds_5k.npz`.

Q1 IS THE CONSTRAINT INFORMATIVE FOR THE MARGINAL DECODER?  No, for a closed-form reason. With
   W_b(v|a,c) = Pr(V_b = a*S[p_b^v] ^ c | t), the joint marginal needs the other 15 bytes only
   through Z_b'(a,c) = sum_v W_b'(v|a,c). For a != 0 the map v -> a*S[p^v] ^ c is a bijection over
   all 256 values, so Z = 1 identically: marginalising a byte's own key makes that byte's slot say
   nothing about (a,c), for any (a,c). E1 measures the identity and prices the one branch where it
   fails (a = 0), which is also an unnormalised term in `proba_dissect_alpha`.

Q2 CAN THE CONSTRAINT BE MADE INFORMATIVE BY PLUGGING KEY ESTIMATES?  Yes, but it is then
   self-confirming, and the mask is redrawn every trace, so nothing can be borrowed across traces
   (E3, with its ceiling run using the true key and its null run on shuffled head rows). The
   ceiling case is worth having on its own terms: it asks what the 16 C heads together know about
   the MASK, which is the mask-conditional visibility question W1 said per-target SNR cannot see.

Q3 IS THE MASK HEADPOSTERIOR MISCALIBRATED?  E4 measures the confidence-error curve of the alpha,
   beta and C heads and sweeps a temperature on the mask posteriors. tau -> 0 is W2's argmax
   plug-in, tau = 1 is the head as trained, tau -> inf is uniform, so the sweep interpolates
   exactly between the two cases W2 measured, and prices each tau in attack terms.

Gates (pre-registered in docs/tasks/T031-c-head-investigation.md, unmovable):
  N1      median z >= +10.0 AND traces-to-rank-0 <= 150 AND 16/16 non-flat, "or a clear measurement
          of why the constraint does not help"
  sanity  the repo baseline decode reproduces 16/16 rank 0 at median z +9.50

Usage:
    conda run -n py10 python -u -m experiments.w7_joint_ab_decode --limit 200
    conda run -n py10 python -u experiments/w7_joint_ab_decode.py --grid-k 128
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
from ASCAD_test_models import multilabel_predict, ranks_from_log_scores
from experiments.w1_timesnet_old_head_accs import head_metrics, load_attack_split
from experiments.w2_decoders import (
    MUL_TABLE,
    check_pi_orientation,
    diagnose_scores,
    head_list_to_arrays,
    mul_dissect,
    route_marginal,
    summarise,
    xor_dissect,
)
from experiments.w5_c_lut import log_picked

CACHE = os.path.join(repo_root, "experiments", "out", "head_preds_5k.npz")
DEFAULT_OUT = os.path.join(repo_root, "experiments", "out", "w7_joint_ab_decode.json")
BASELINE_Z = 9.50
Z_TOL = 0.05
GATE_Z = 10.0
GATE_TRACES = 150
EXPECTED_PI_ACC = 0.6703          # S1's measurement for these heads; guards the read orientation
VALS = np.arange(256, dtype=np.int64)


# ------------------------------------------------------------------ algebra


def mul_row(a):
    """a * x for every x in 0..255 at once: one row of MUL_TABLE per alpha, shape (..., 256)."""
    return MUL_TABLE[np.asarray(a, dtype=np.int64)]


def hypothesis_index(px):
    """(..., ) array of p^k -> (..., 256) Sbox[(p^k) ^ h]. The hypothesis axis is h, h = 0 is the
    true key, which is the convention `rank()` and `experiments/rank_degeneracy.py` use."""
    return AES_Sbox[np.bitwise_xor(np.asarray(px, dtype=np.uint8)[..., None],
                                   np.arange(256, dtype=np.uint8))]


def sbox_under_hypothesis(pt_col, ky_col, n):
    """(n,256) for one byte: Sbox[p_b ^ k_b ^ h]."""
    return hypothesis_index(np.asarray(pt_col[:n], dtype=np.uint8)
                            ^ np.asarray(ky_col[:n], dtype=np.uint8))


def exact_byte_posterior(Cmix_b, A, B):
    """sum_{a,c} A[a] B[c] Cmix[a*x ^ c]: the repo dissection on one byte's masked posterior."""
    return mul_dissect(xor_dissect(Cmix_b, B), A)


def renorm_byte_posterior(Cmix_b, A, B):
    """The same mixture with every (a,c) component normalised as a distribution over the key.

    a != 0 is already normalised (bijection, sum_v W = 1). For a = 0 every hypothesis is equally
    likely given the observation, so the component must contribute A[0] * sum_c B[c] * (1/256)
    = A[0]/256, constant over x, instead of the unnormalised A[0] * sum_c B[c] * Cmix[c].
    """
    A2 = np.array(A, dtype=np.float64, copy=True)
    A2[:, 0] = 0.0
    out = exact_byte_posterior(Cmix_b, A2, B)
    out += A[:, 0:1] / 256.0
    return out


# ------------------------------------------------------------------ scoring


def per_byte_rows(Q, lab, n, rank_step):
    """Per-byte diagnostics from a (n,16,256) posterior over the unmasked S-box value."""
    rows = []
    for b in range(16):
        idx = sbox_under_hypothesis(lab["plaintext"][:, b], lab["key"][:, b], n)
        lp = log_picked(Q[:n, b], idx)
        d = diagnose_scores(lp.sum(axis=0))
        curve = ranks_from_log_scores(lp, 0, rank_step, tie_break="worst")
        hit = curve[curve[:, 1] == 0]
        d["traces_to_rank0_worst"] = int(hit[0, 0]) if hit.size else None
        d["final_rank_worst"] = int(curve[-1, 1])
        rows.append(d)
    return rows


def record(table, name, desc, Q, lab, n, rank_step, verbose=True):
    rows = per_byte_rows(Q, lab, n, rank_step)
    s = summarise(rows)
    table[name] = {"description": desc, "summary": s,
                   "per_byte": {str(b): rows[b] for b in range(16)}}
    if verbose:
        print(f"{name:24s} z {s['median_z']:+8.4f}  rank0 {s['n_rank0_final']:2d}/16 "
              f"genuine {s['n_rank0_and_not_flat']:2d}/16  flat {s['n_flat_bytes']:2d} "
              f"spread {s['median_spread']:.4g}  traces->r0 {s['median_traces_to_rank0']}",
              flush=True)
    return s


# ------------------------------------------------------------------ EM machinery


def em_cells(Cmix_log, A, B, xhat, n, k, chunk, use_prior=True):
    """Per-trace top-k (alpha,beta) cells under the cross-slot agreement evidence.

    Chunked so the 256x256 grid never exists for more than `chunk` traces at a time: at 5,000
    traces the full grid would be 2.6 GB per array.
    """
    a_idx = np.empty((n, k), dtype=np.int64)
    c_idx = np.empty((n, k), dtype=np.int64)
    w = np.empty((n, k), dtype=np.float64)
    best_a = np.empty(n, dtype=np.int64)
    best_c = np.empty(n, dtype=np.int64)
    kept = np.empty(n, dtype=np.float64)
    ev_best = np.empty(n, dtype=np.float64)
    prior = A[:, :, None] * B[:, None, :]
    cgrid = VALS[None, None, :]
    for s0 in range(0, n, chunk):
        s1 = min(s0 + chunk, n)
        m = s1 - s0
        E = np.empty((m, 256, 256), dtype=np.float64)
        xh = xhat[s0:s1].astype(np.int64)
        sl, sb = np.arange(m)[:, None, None], np.arange(16)[None, :, None]
        for a in range(256):
            ia = MUL_TABLE[a][xh]
            E[:, a] = Cmix_log[s0:s1][sl, sb, np.bitwise_xor(ia[:, :, None], cgrid)].sum(1)
        q = np.exp(np.clip(E - E.max(axis=(1, 2), keepdims=True), -700.0, 0.0))
        if use_prior:
            q *= prior[s0:s1]
        q /= np.maximum(q.sum(axis=(1, 2), keepdims=True), 1e-300)
        flat = q.reshape(m, -1)
        part = np.argpartition(-flat, k - 1, axis=1)[:, :k]
        ww = np.take_along_axis(flat, part, axis=1)
        kept[s0:s1] = ww.sum(axis=1)
        w[s0:s1] = ww / np.maximum(kept[s0:s1], 1e-300)[:, None]
        a_idx[s0:s1], c_idx[s0:s1] = part // 256, part % 256
        arg = np.argmax(flat, axis=1)
        best_a[s0:s1], best_c[s0:s1] = arg // 256, arg % 256
        ev_best[s0:s1] = flat[np.arange(m), arg]
    return {"a_idx": a_idx, "c_idx": c_idx, "w": w, "best_a": best_a, "best_c": best_c,
            "mean_grid_mass_kept": float(kept.mean()),
            "mean_prob_of_best_cell": float(ev_best.mean())}


def grid_decode(Cmix, a_idx, c_idx, w, n, chunk=64):
    """Q[t,b,x] = sum_i w[t,i] * Cmix[t,b, a_i*x ^ c_i] over a per-trace pruned (a,c) grid."""
    Q = np.zeros((n, 16, 256), dtype=np.float64)
    for s0 in range(0, n, chunk):
        s1 = min(s0 + chunk, n)
        m = s1 - s0
        rows = np.arange(m)[:, None]
        for i in range(a_idx.shape[1]):
            idx = np.bitwise_xor(mul_row(a_idx[s0:s1, i]),
                                 c_idx[s0:s1, i][:, None])
            for b in range(16):
                Q[s0:s1, b] += w[s0:s1, i][:, None] * Cmix[s0:s1, b][rows, idx]
    return Q


def hard_plug_decode(Cmix, a_hat, c_hat, n, chunk=64):
    """Q[t,b,x] = Cmix[t,b, a_hat*x ^ c_hat] -- W2's D1b scoring rule with different masks."""
    Q = np.zeros((n, 16, 256), dtype=np.float64)
    for s0 in range(0, n, chunk):
        s1 = min(s0 + chunk, n)
        rows = np.arange(s1 - s0)[:, None]
        idx = np.bitwise_xor(mul_row(a_hat[s0:s1]), c_hat[s0:s1][:, None])
        for b in range(16):
            Q[s0:s1, b] = Cmix[s0:s1, b][rows, idx]
    return Q


# ------------------------------------------------------------------ calibration


def tempered(probs, tau, floor=1e-300):
    """softmax(log p / tau): tau=1 is the head as trained, tau->0 the argmax plug-in, tau->inf uniform."""
    lp = np.log(np.clip(np.asarray(probs, dtype=np.float64), floor, None)) / float(tau)
    lp -= lp.max(axis=1, keepdims=True)
    e = np.exp(lp)
    return e / e.sum(axis=1, keepdims=True)


def true_prob(probs, labels):
    return probs[np.arange(probs.shape[0]), np.asarray(labels, dtype=np.int64)]


def ece(probs, labels, n_bins=15):
    """Expected calibration error: |mean confidence - mean accuracy|, weighted over confidence bins."""
    conf = probs.max(axis=1)
    acc = (probs.argmax(axis=1) == np.asarray(labels)).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    tot = err = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (conf > lo) & (conf <= hi) if lo > 0 else (conf <= hi)
        if not np.any(sel):
            continue
        m = int(sel.sum())
        tot += m
        err += m * abs(acc[sel].mean() - conf[sel].mean())
    return float(err / max(tot, 1))


# ------------------------------------------------------------------ main


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--num-traces", type=int, default=5000)
    ap.add_argument("--rank-step", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0, help="smoke test on the first N traces")
    ap.add_argument("--grid-k", type=int, default=128, help="top-K (a,c) cells kept per trace")
    ap.add_argument("--chunk", type=int, default=128, help="EM grid chunk in traces")
    ap.add_argument("--taus", default="0.05,0.1,0.2,0.35,0.5,0.7,1.0,1.5,2,3,5,10,20")
    ap.add_argument("--out-json", default=DEFAULT_OUT)
    args = ap.parse_args()

    t0 = time.time()
    n = args.limit or args.num_traces
    z = np.load(CACHE)
    if n > z["alpha_output"].shape[0]:
        raise SystemExit(f"cache holds {z['alpha_output'].shape[0]} traces")
    _, lab = load_attack_split(n)
    preds = {k: np.asarray(z[k])[:n] for k in z.files}
    A, B, C, PI = head_list_to_arrays(preds)
    pi_acc = check_pi_orientation(PI, lab["perm_index"][:n], EXPECTED_PI_ACC)

    alpha_true = lab["alpha"].astype(np.int64)
    beta_true = lab["beta"].astype(np.int64)
    alpha_hat_acc = float(np.mean(A.argmax(1) == alpha_true))
    beta_hat_acc = float(np.mean(B.argmax(1) == beta_true))
    true_a0 = alpha_true == 0
    print(f"traces={n}  pi element acc {pi_acc:.4f}  alpha_hat {alpha_hat_acc:.4f} "
          f"beta_hat {beta_hat_acc:.4f}  true alpha==0 in {true_a0.mean():.4%} of traces", flush=True)

    Cmix = route_marginal(C, PI)                     # (n,16,256) masked-value posterior per byte
    Cmix_log = np.log(np.clip(Cmix, 1e-300, None))
    cases = {}

    def decode(Ap, Bp):
        """(n,16,256) unmasked-value posterior for any mask posteriors, via the repo dissection."""
        return np.stack([exact_byte_posterior(Cmix[:, b], Ap, Bp) for b in range(16)], axis=1)

    # ---- E0 the reference decode, with its parity against the repo function -----------------
    lst = [A, B] + [C[:, s] for s in range(16)] + [PI[:, s, :] for s in range(16)]
    sbox_d0 = np.stack(multilabel_predict(lst), axis=1)[:n]   # (n,16,256) as Q0 is
    Q0 = decode(A, B)
    parity_d0 = float(np.max(np.abs(sbox_d0 - Q0)))
    e0 = record(cases, "E0_repo_dissection", "reference: per-slot marginalisation "
                                             "(= multilabel_predict)", Q0, lab, n, args.rank_step)
    print(f"E0 parity vs multilabel_predict: max abs diff {parity_d0:.3e}", flush=True)

    # ---- E1 the shared-mask constraint, marginalised correctly ------------------------------
    Q1 = np.stack([renorm_byte_posterior(Cmix[:, b], A, B) for b in range(16)], axis=1)
    e1 = record(cases, "E1_renorm_joint", "joint over the shared (alpha,beta), every (a,c) "
                                          "component normalised as a key distribution",
                Q1, lab, n, args.rank_step)

    # the bijection identity, measured rather than asserted
    rng = np.random.default_rng(0)
    Ptest = rng.random((7, 256)); Ptest /= Ptest.sum(-1, keepdims=True)
    ident = []
    for a in (0, 1, 2, 3, 5, 84, 255):
        for c in (0, 1, 99, 255):
            idx = np.bitwise_xor(mul_row(a), c)
            s = Ptest[:, idx].sum(axis=1)
            expect = 256.0 * Ptest[:, c] if a == 0 else np.ones(7)
            ident.append({"a": a, "c": c, "sum_max": float(s.max()), "sum_min": float(s.min()),
                          "expected_max": float(expect.max()),
                          "deviation": float(np.max(np.abs(s - expect)))})
    max_dev = float(max(d["deviation"] for d in ident))
    a0_unnorm = float(np.mean(np.abs(Q0[:, 0, :] - Q1[:, 0, :]).max(axis=1)))
    a0_effect = {
        "bijection_identity_max_deviation": max_dev,
        "bijection_identity_cases": len(ident),
        "bijection_identity_examples": ident[:8],
        "mean_head_prob_alpha_0": float(np.mean(A[:, 0])),
        "mean_head_prob_alpha_0_given_true_alpha_0":
            float(np.mean(A[true_a0, 0])) if np.any(true_a0) else None,
        "true_alpha_is_0_frequency": float(true_a0.mean()),
        "max_abs_posterior_delta": float(np.max(np.abs(Q0 - Q1))),
        "mean_over_traces_of_max_abs_posterior_delta": float(
            np.mean(np.abs(Q0 - Q1).reshape(n, -1).max(axis=1))),
        "byte0_worst_abs_delta": a0_unnorm,
        "note": "for a given trace and byte the a=0 term is constant over the 256 hypotheses, so it "
                "cannot reorder that trace; it only re-weights how much that trace contributes to "
                "the summed log score, which is what E1 vs E0 prices",
    }
    print(f"bijection identity max deviation {max_dev:.3e}; "
          f"E0 vs E1 max posterior delta {a0_effect['max_abs_posterior_delta']:.3e}", flush=True)

    # ---- E2 what the mask posterior actually costs ------------------------------------------
    px = np.bitwise_xor(lab["plaintext"][:n].astype(np.uint8), lab["key"][:n].astype(np.uint8))
    idx0 = hypothesis_index(px)                                   # (n,16,256), h = 0 is the true key
    Q0_pick = np.take_along_axis(Q0[:n], idx0, axis=2)
    k_hat_rel = Q0_pick.argmax(axis=2)                       # (n,16) relative guesses, 0 = true
    pass1_by_byte = {str(b): float(np.mean(k_hat_rel[:, b] == 0)) for b in range(16)}
    pass1_acc = float(np.mean(k_hat_rel == 0))
    print(f"pass-1 per-byte top-1 key accuracy: mean {pass1_acc:.4f} "
          f"(range {min(pass1_by_byte.values()):.4f}-{max(pass1_by_byte.values()):.4f})", flush=True)

    # how much of the +3.4 z gap is explained by the mask heads alone, before any joint decoding
    xhat_plugin = AES_Sbox[np.bitwise_xor(lab["plaintext"][:n].astype(np.uint8),
                                          np.bitwise_xor(lab["key"][:n].astype(np.uint8),
                                                         k_hat_rel.astype(np.uint8)))]
    xhat_true = AES_Sbox[np.bitwise_xor(lab["plaintext"][:n], lab["key"][:n])].astype(np.int64)

    # ---- E3 the plug-in-key EM --------------------------------------------------------------
    em_acc, grids = {}, {}
    for tag, xh, oracle in (("plug1", xhat_plugin, False), ("oraclekey", xhat_true, True)):
        cells = em_cells(Cmix_log, A, B, xh, n, args.grid_k, args.chunk, use_prior=True)
        grids[tag] = cells
        record(cases, f"E3_soft_em_{tag}", f"EM marginal decode over the top-{args.grid_k} "
              f"(alpha,beta) cells, slot evidence under the {tag} key",
               grid_decode(Cmix, cells["a_idx"], cells["c_idx"], cells["w"], n),
               lab, n, args.rank_step)
        ba, bc = cells["best_a"], cells["best_c"]
        em_acc[tag] = {
            "alpha_top1": float(np.mean(ba == alpha_true)),
            "beta_top1": float(np.mean(bc == beta_true)),
            "both_top1": float(np.mean((ba == alpha_true) & (bc == beta_true))),
            "grid_mass_kept": cells["mean_grid_mass_kept"],
            "mean_prob_of_best_cell": cells["mean_prob_of_best_cell"],
            "head_only_alpha_top1": alpha_hat_acc, "head_only_beta_top1": beta_hat_acc,
            "oracle_assisted": oracle,
        }
        print(f"EM[{tag}] mask acc alpha {em_acc[tag]['alpha_top1']:.4f} beta "
              f"{em_acc[tag]['beta_top1']:.4f} both {em_acc[tag]['both_top1']:.4f} "
              f"(head-only {alpha_hat_acc:.4f}/{beta_hat_acc:.4f})", flush=True)
        record(cases, f"E3_hard_em_{tag}", "EM argmax masks plugged in (W2's D1b rule with EM "
                                          "masks instead of head argmaxes)",
               hard_plug_decode(Cmix, ba, bc, n), lab, n, args.rank_step)

    # the same evidence WITHOUT the mask heads: what the 16 slots alone say about (alpha,beta)
    cells0 = em_cells(Cmix_log, A, B, xhat_true, n, args.grid_k, args.chunk, use_prior=False)
    em_acc["slots_only_oraclekey"] = {
        "alpha_top1": float(np.mean(cells0["best_a"] == alpha_true)),
        "beta_top1": float(np.mean(cells0["best_c"] == beta_true)),
        "both_top1": float(np.mean((cells0["best_a"] == alpha_true)
                                   & (cells0["best_c"] == beta_true))),
        "chance_top1": 1.0 / 256.0,
        "mean_prob_of_best_cell": cells0["mean_prob_of_best_cell"],
        "chance_prob_of_best_cell": 1.0 / 65536.0,
        "note": "argmax of the 65,536-cell grid under the TRUE key and no mask head: a mask "
                "estimator built from the C heads alone, which is the mask-conditional visibility "
                "question W1 said per-target SNR cannot measure. Oracle-assisted by construction "
                "(it needs the key), so it is a bound, not an attack",
    }
    print(f"slots-only (no mask heads, true key): alpha "
          f"{em_acc['slots_only_oraclekey']['alpha_top1']:.4f} beta "
          f"{em_acc['slots_only_oraclekey']['beta_top1']:.4f}", flush=True)

    # null: the same grid machinery on head rows shuffled against fixed plaintexts and keys
    perm = rng.permutation(n)
    g1 = grids["plug1"]
    record(cases, "E3_null_shuffled", "E3 plug1 grid applied to shuffled Cmix rows (evidence "
                                      "decoupled from the data) -- self-confirmation check",
           grid_decode(Cmix[perm], g1["a_idx"], g1["c_idx"], g1["w"], n), lab, n, args.rank_step)
    record(cases, "E3_null_random_masks", "uniform-random (alpha,beta) cells, marginalised: the "
                                          "no-information version of the joint decode",
           grid_decode(Cmix, rng.integers(0, 256, (n, 16)), rng.integers(0, 256, (n, 16)),
                       np.full((n, 16), 1.0 / 16), n), lab, n, args.rank_step)

    # ---- E4 temperature of the mask posteriors ---------------------------------------------
    taus = [float(t) for t in args.taus.split(",")]
    half = n // 2
    lab_first = {k: v[:half] for k, v in lab.items()}
    lab_second = {k: v[half:] for k, v in lab.items()}

    heads_tbl = {"alpha": (A, alpha_true), "beta": (B, beta_true)}
    cal = {}
    for name, (P, y) in heads_tbl.items():
        nll1 = [float(-np.mean(np.log(np.clip(true_prob(tempered(P[:half], t), y[:half]),
                                             1e-300, None)))) for t in taus]
        nll2 = [float(-np.mean(np.log(np.clip(true_prob(tempered(P[half:], t), y[half:]),
                                             1e-300, None)))) for t in taus]
        cal[name] = {
            "ece_tau1": ece(P, y), "acc_tau1": float(np.mean(P.argmax(1) == y)),
            "mean_true_prob_tau1": float(np.mean(true_prob(P, y))),
            "tau_star_fit_on_first_half": taus[int(np.argmin(nll1))],
            "tau_star_fit_on_second_half": taus[int(np.argmin(nll2))],
            "nll_by_tau_first_half": {str(t): round(v, 6) for t, v in zip(taus, nll1)},
            "nll_by_tau_second_half": {str(t): round(v, 6) for t, v in zip(taus, nll2)},
        }
    Cflat = np.concatenate([C[:, s] for s in range(16)], axis=0)
    yflat = np.concatenate([lab["sbox_masked"][:n, s] for s in range(16)], axis=0)
    cal["C_pooled"] = {"ece_tau1": ece(Cflat, yflat),
                       "acc_tau1": float(np.mean(Cflat.argmax(1) == yflat)),
                       "mean_true_prob_tau1": float(np.mean(true_prob(Cflat, yflat)))}
    for name, c in cal.items():
        print(f"{name}: ECE {c['ece_tau1']:.4f} acc {c['acc_tau1']:.4f} "
              f"mean p(true) {c['mean_true_prob_tau1']:.3e}"
              + (f" tau* {c.get('tau_star_fit_on_first_half')} / "
                 f"{c.get('tau_star_fit_on_second_half')}" if "tau_star_fit_on_first_half" in c
                 else ""), flush=True)

    def sweep(which):
        curve = {}
        for t in taus:
            P = tempered(B, t) if which == "beta" else tempered(A, t)
            Q = decode(P if which == "alpha" else A, P if which == "beta" else B)
            full = summarise(per_byte_rows(Q, lab, n, args.rank_step))
            h1 = summarise(per_byte_rows(Q[:half], lab_first, half, args.rank_step))
            h2 = summarise(per_byte_rows(Q[half:], lab_second, n - half, args.rank_step))
            curve[str(t)] = {
                "median_z": full["median_z"], "median_traces_to_rank0": full["median_traces_to_rank0"],
                "n_rank0_and_not_flat": full["n_rank0_and_not_flat"],
                "half1_median_z": h1["median_z"], "half2_median_z": h2["median_z"],
                "half1_traces_to_rank0": h1["median_traces_to_rank0"],
                "half2_traces_to_rank0": h2["median_traces_to_rank0"],
                "mask_top_prob_mean": float(P.max(axis=1).mean()),
                "mask_argmax_equals_head_argmax": float(np.mean(P.argmax(1) == (B if which == "beta"
                                                                               else A).argmax(1))),
            }
            print(f"{which} tau {t:>6g}: z {full['median_z']:+8.4f} traces->r0 "
                  f"{full['median_traces_to_rank0']}  halves z {h1['median_z']:+.3f}/"
                  f"{h2['median_z']:+.3f}", flush=True)
        return curve

    beta_sweep = sweep("beta")
    alpha_sweep = sweep("alpha")
    assert abs(beta_sweep["1.0"]["median_z"] - e0["median_z"]) < 1e-9, \
        "tau=1 must reproduce E0 exactly"
    assert abs(alpha_sweep["1.0"]["median_z"] - e0["median_z"]) < 1e-9, \
        "tau=1 must reproduce E0 exactly"

    # cross-fitted temperature: fit on one half, attack the other (no in-sample flattering)
    crossfit = {}
    for which, sweep_tbl in (("beta", beta_sweep), ("alpha", alpha_sweep)):
        t_first = cal[which]["tau_star_fit_on_first_half"]     # fitted on half 1, used on half 2
        t_second = cal[which]["tau_star_fit_on_second_half"]   # fitted on half 2, used on half 1
        for fit_tag, tau, sl, eval_lab, ref_key in (
                ("fit_first_eval_second", t_first, slice(half, n), lab_second,
                 "half2"), ("fit_second_eval_first", t_second, slice(0, half), lab_first, "half1")):
            P = tempered(B if which == "beta" else A, tau)
            Q = decode(P if which == "alpha" else A, P if which == "beta" else B)[sl]
            m = Q.shape[0]
            got = summarise(per_byte_rows(Q, eval_lab, m, args.rank_step))
            crossfit[f"{which}_{fit_tag}"] = {
                "tau": tau, "n_eval_traces": int(m),
                "median_z": got["median_z"],
                "median_traces_to_rank0": got["median_traces_to_rank0"],
                "n_rank0_and_not_flat": got["n_rank0_and_not_flat"],
                "tau1_median_z_on_same_half": sweep_tbl["1.0"][f"{ref_key}_median_z"],
                "tau1_traces_to_rank0_on_same_half": sweep_tbl["1.0"][f"{ref_key}_traces_to_rank0"],
            }
            print(f"crossfit {which}_{fit_tag}: tau {tau:g} z {got['median_z']:+.4f} "
                  f"(tau=1 {crossfit[f'{which}_{fit_tag}']['tau1_median_z_on_same_half']:+.4f}) "
                  f"traces->r0 {got['median_traces_to_rank0']}", flush=True)

    # tau chosen ON the attack metric is only an honest number if it is fitted and scored on
    # disjoint halves; fitting it on the same traces it is reported on would be selection bias.
    attack_tau = {}
    for which, sweep_tbl in (("beta", beta_sweep), ("alpha", alpha_sweep)):
        for fit_key, eval_key in (("half1", "half2"), ("half2", "half1")):
            tau_hat = max(sweep_tbl.items(),
                          key=lambda kv: (kv[1][f"{fit_key}_median_z"],
                                          -(kv[1][f"{fit_key}_traces_to_rank0"] or 1e9)))[0]
            attack_tau[f"{which}_fit_{fit_key}"] = {
                "tau_selected_on_attack_metric": float(tau_hat),
                "z_on_eval_half": sweep_tbl[tau_hat][f"{eval_key}_median_z"],
                "z_on_eval_half_at_tau1": sweep_tbl["1.0"][f"{eval_key}_median_z"],
                "traces_to_rank0_on_eval_half": sweep_tbl[tau_hat][f"{eval_key}_traces_to_rank0"],
                "traces_to_rank0_on_eval_half_at_tau1":
                    sweep_tbl["1.0"][f"{eval_key}_traces_to_rank0"],
            }
            print(f"attack-selected tau {which}/{fit_key}: {tau_hat} -> eval z "
                  f"{sweep_tbl[tau_hat][f'{eval_key}_median_z']:+.4f} (tau=1 "
                  f"{sweep_tbl['1.0'][f'{eval_key}_median_z']:+.4f})", flush=True)

    # ---- gates ------------------------------------------------------------------------------
    def role(name):
        if name.endswith("oraclekey"):
            return "oracle_bound"
        if "null" in name:
            return "control"
        return "candidate"

    cand = {k: v["summary"] for k, v in cases.items() if k != "E0_repo_dissection"}
    per_case = {}
    for k, s in cand.items():
        per_case[k] = {
            "role": role(k), "median_z": s["median_z"],
            "median_traces_to_rank0": s["median_traces_to_rank0"],
            "n_rank0_and_not_flat": s["n_rank0_and_not_flat"],
            "passes": bool(s["median_z"] >= GATE_Z and s["median_traces_to_rank0"] is not None
                           and s["median_traces_to_rank0"] <= GATE_TRACES
                           and s["n_rank0_and_not_flat"] == 16),
        }
    legit_pass = [v["passes"] for v in per_case.values() if v["role"] == "candidate"]
    best_tau = max(beta_sweep.items(),
                   key=lambda kv: (kv[1]["median_z"], -(kv[1]["median_traces_to_rank0"] or 1e9)))
    out = {
        "meta": {
            "step": "T-031 T31-N1 joint (alpha,beta) decoding + mask calibration",
            "cache": os.path.relpath(CACHE, repo_root), "n_traces": int(n),
            "rank_step": args.rank_step, "tie_break": "worst", "grid_k": args.grid_k,
            "chunk": args.chunk, "taus": taus, "numpy": np.__version__,
            "python": sys.version.split()[0],
            "git": os.popen("git -C %s rev-parse --short HEAD" % repo_root).read().strip(),
            "gpu_used": False, "wall_s": round(time.time() - t0, 1),
        },
        "heads": {"pi_element_acc": pi_acc, "alpha_hat_acc": alpha_hat_acc,
                  "beta_hat_acc": beta_hat_acc,
                  "alpha_head": head_metrics(A, lab["alpha"]),
                  "beta_head": head_metrics(B, lab["beta"]),
                  "C_mean_top1": float(np.mean([head_metrics(C[:, i], lab["sbox_masked"][:, i])
                                                ["top1_acc"] for i in range(16)])),
                  "C_mean_gain": float(np.mean([head_metrics(C[:, i], lab["sbox_masked"][:, i])
                                                ["gain_over_uniform_nats"] for i in range(16)])),
                  "pi_mean_top1": float(np.mean([head_metrics(PI[:, i, :], lab["perm_index"][:, i])
                                                 ["top1_acc"] for i in range(16)]))},
        "checks": {"E0_parity_vs_multilabel_predict_max_abs_diff": parity_d0,
                   "a0_and_bijection": a0_effect,
                   "pass1_top1_key_accuracy_by_byte": pass1_by_byte,
                   "pass1_top1_key_accuracy_mean": pass1_acc},
        "em_mask_accuracy": em_acc,
        "calibration": cal,
        "beta_temperature_sweep": beta_sweep,
        "alpha_temperature_sweep": alpha_sweep,
        "crossfitted_temperature": crossfit,
        "attack_selected_temperature": attack_tau,
        "best_beta_tau_by_z": {"tau": float(best_tau[0]), **best_tau[1]},
        "cases": cases,
        "gates": {
            "sanity_reproduce_baseline": {
                "measured_median_z": e0["median_z"], "documented": BASELINE_Z, "tolerance": Z_TOL,
                "n_rank0_and_not_flat": e0["n_rank0_and_not_flat"],
                "pass": bool(abs(e0["median_z"] - BASELINE_Z) <= Z_TOL
                             and e0["n_rank0_and_not_flat"] == 16)},
            "N1_joint_decode_beats_baseline": {
                "criterion": f"median z >= {GATE_Z} and traces-to-rank-0 <= {GATE_TRACES} "
                             "and 16/16 non-flat rank 0, judged on attack-legitimate decoders "
                             "(oracle bounds and shuffled nulls are classified but excluded)",
                "per_case": per_case,
                "any_legitimate_pass": bool(any(legit_pass)),
                "n_legitimate_candidates": len(legit_pass),
                "reference_E0": {"median_z": e0["median_z"],
                                 "median_traces_to_rank0": e0["median_traces_to_rank0"]}},
        },
    }
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {args.out_json} ({out['meta']['wall_s']}s)  "
          f"N1 gate any_legitimate_pass="
          f"{out['gates']['N1_joint_decode_beats_baseline']['any_legitimate_pass']}", flush=True)


if __name__ == "__main__":
    main()
