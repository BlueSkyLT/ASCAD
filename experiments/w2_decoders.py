#!/usr/bin/env python3
"""T-030 S2 — can a better decoder beat the MultiResNet baseline, using no GPU?

Everything here reads the cached head predictions of the working worst-case model
(`experiments/out/head_preds_5k.npz`, 34 heads x 5,000 ASCADv2 attack traces, produced by
`experiments/ablation_heads.py` in T-028) and the attack labels/metadata from the HDF5. No
training, no forward pass, no checkpoint is touched. Mask labels are used (this is the
worst-case line; the non-worst-case line is T-028).

The three S2 sub-questions:

  (a) pi routing. The baseline routes slot->byte with the *independent* per-slot marginals
      `Pr(pi_slot == b)`, which cannot satisfy the permutation constraint (exactly one byte per
      slot, exactly one slot per byte). Variants: Hungarian MAP assignment and Sinkhorn
      doubly-stochastic coupling, against that marginal path and against the true permutation.
  (b) joint (alpha, beta). The baseline dissections are a sequential marginalisation, beta first
      then alpha. `joint_alpha_beta_reference` computes the explicit 256 x 256 double sum, so the
      claim "per-slot dissection == joint marginalisation under p(alpha,beta|t)=p(alpha|t)p(beta|t)"
      is checked numerically instead of assumed.
  (c) oracle swaps and negation controls. Replace one head by its true label (alpha, beta, C, pi)
      or by the uniform distribution, and see what the key score does. This is how the two open
      anomalies are re-checked: "substituting the true beta lowers key z", and "the C heads score
      below uniform yet carry the key signal".

Gates (pre-registered in docs/tasks/T030-worstcase-multitask-improvement.md, not movable):
  sanity  - reproduce the baseline median z +9.50 with tie/flatness clean; stop if not.
  improve - median z over the 16 bytes rises above +9.50, OR traces-to-rank-0 falls.

Ranks use `tie_break="worst"` throughout, and every byte additionally reports the score spread,
the number of hypotheses tied with the true key, and `flat`/`near_flat`, because rank 0 on a flat
vector is a tie and not a break (T-028 Phase 0).

Usage:
    conda run -n py10 python -u -m experiments.w2_decoders [--num-traces 5000] [--rank-step 10]
        [--variants V0,V1,...] [--out-json experiments/out/w2_decoder_results.json]
    conda run -n py10 python -u -m experiments.w2_decoder_tests      # known-answer tests
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
from scipy.optimize import linear_sum_assignment

from ASCAD_generate import AES_Sbox
from perm_routing import apply_pi_routing
from ASCAD_test_models import (
    multGF256,
    multilabel_predict,
    ranks_from_log_scores,
)

H5 = os.path.join(repo_root, "STM32_AES_v2/ASCAD_data/ASCAD_databases/ascadv2-extracted.h5")
CACHE = os.path.join(repo_root, "experiments", "out", "head_preds_5k.npz")
DEFAULT_OUT = os.path.join(repo_root, "experiments", "out", "w2_decoder_results.json")

# An oracle head is a near-hard one-hot. Exact zeros would be floored to (row min non-zero)^2 by
# the rank() convention, i.e. a zero entry would score as the *best* hypothesis -- so oracles must
# not be hard, and the smoothing level is reported with every oracle number.
EPS_ORACLE = 1e-6
PI_CLASSES = 16
VALUE_CLASSES = 256
BASELINE_Z = 9.50          # docs/tasks/T030: median z of the working MultiResNet baseline
BASELINE_Z_VERIFIED = 9.5047   # S1's independent rank_degeneracy.py reproduction of it
Z_TOL = 0.05               # the sanity gate is "reproduces +9.50", not "reproduces 9.5047"


# ------------------------------------------------------------------ GF(2^8) tables


def gf_tables():
    """Multiplication table plus exp/log tables for a primitive element of GF(256)."""
    mul = np.zeros((256, 256), dtype=np.uint8)
    for a in range(256):
        for b in range(256):
            mul[a, b] = multGF256(a, b)
    for g in range(2, 256):
        seen = set()
        v = 1
        for _ in range(255):
            v = int(mul[v, g])
            seen.add(v)
        if len(seen) == 255:
            exp_t = np.zeros(255, dtype=np.int16)
            log_t = np.full(256, -1, dtype=np.int16)
            v = 1
            for u in range(255):
                exp_t[u] = v
                log_t[v] = u
                v = int(mul[v, g])
            return mul, exp_t, log_t, g
    raise RuntimeError("no primitive element found for GF(256)")


MUL_TABLE, GF_EXP, GF_LOG, GF_GENERATOR = gf_tables()


# ------------------------------------------------------------------ exact dissections


def fwht(a):
    """Unnormalised fast Walsh-Hadamard transform along the last axis (H^2 = n*I)."""
    a = np.array(a, dtype=np.float64, copy=True)
    n = a.shape[-1]
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            s1 = slice(i, i + h)
            s2 = slice(i + h, i + 2 * h)
            x = a[..., s1].copy()
            y = a[..., s2].copy()
            a[..., s1] = x + y
            a[..., s2] = x - y
        h *= 2
    return a


def xor_dissect(P, q):
    """proba[j] = sum_b q[..., b] * P[..., j ^ b]  -- equals proba_dissect_beta per row.

    XOR convolution along the last axis, diagonalised by the Hadamard transform: with the
    unnormalised transform H (H^2 = n*I), P (*) q = H( H(P) . H(q) ) / n.
    """
    n = P.shape[-1]
    return fwht(fwht(P) * fwht(q)) / n


def mul_dissect(Q, pa):
    """proba[j] = sum_a pa[..., a] * Q[..., multGF256(a, j)] -- equals proba_dissect_alpha.

    Over GF(256)^* this is a cyclic correlation in the log domain; the alpha=0 and value=0 cases
    are handled separately because 0 has no logarithm and kills everything it multiplies.
    """
    shape = Q.shape[:-1]
    out = np.zeros(Q.shape, dtype=np.float64)
    j0 = Q[..., 0]                                    # multGF256(a, 0) = 0 for every a
    out[..., 0] = j0 * pa[..., :].sum(axis=-1)        # every alpha maps j=0 to 0
    A = pa[..., GF_EXP[:]]                            # (..., 255) posterior over log alpha
    Z = Q[..., GF_EXP[:]]                             # (..., 255) over log value
    Ar = np.empty_like(A)                             # reversed for correlation-as-convolution
    Ar[..., 0] = A[..., 0]
    Ar[..., 1:] = A[..., 254:0:-1]
    corr = np.fft.irfft(np.fft.rfft(Ar, n=255, axis=-1) * np.fft.rfft(Z, n=255, axis=-1),
                        n=255, axis=-1)               # corr[l] = sum_u A[u] Z[u + l]
    flat_view = out.reshape(-1, VALUE_CLASSES)
    flat_view[:, GF_EXP[:].astype(np.int64)] = corr.reshape(-1, 255)
    out[..., 1:256] += pa[..., 0:1] * j0[..., None]   # the alpha = 0 term, for j != 0
    return out.reshape(shape + (VALUE_CLASSES,))


def joint_alpha_beta_reference(C_slot, pa, pb):
    """Explicit double sum over (alpha, beta) for one slot -- the (b) reference, small N only.

    The head gives Pr(V = v) for the masked value V = alpha*C ^ beta, and alpha, beta are trace
    global, so Pr(C = c) = sum_a sum_b pa[a] pb[b] Pr(V = a*c ^ b), which is what the sequential
    xor-then-multiply dissection computes. This brute-force form exists only to prove that.
    """
    n = C_slot.shape[0]
    out = np.zeros((n, 256), dtype=np.float64)
    vals = np.arange(256)
    for a in range(256):
        pa_a = pa[:, a]
        if not np.any(pa_a):
            continue
        scaled = MUL_TABLE[a][vals].astype(np.int64)      # a*c for every c
        for b in range(256):
            w = pa_a * pb[:, b]
            if not np.any(w):
                continue
            out += w[:, None] * C_slot[:, scaled ^ b]
    return out


def slot_C_posteriors(C_slot_p, beta_p, alpha_p):
    """Per-slot masked-S-box posterior -> per-slot posterior over the S-box value C.

    alpha_p/beta_p are trace-level (n, 256) heads shared by all 16 slots, so they gain a slot axis
    when the C tensor is (n, slot, value).
    """
    if C_slot_p.ndim == 3:
        b, a = beta_p[:, None, :], alpha_p[:, None, :]
    else:
        b, a = beta_p, alpha_p
    return mul_dissect(xor_dissect(C_slot_p, b), a)


# ------------------------------------------------------------------ pi routing


def route_marginal(slot_C, PI):
    """proba[byte b, value v] = sum_slot slot_C[slot, v] * mu[slot, b]  (= proba_dissect_permind).

    mu = exact permutation marginals of PI (T-041); ASCAD_PI_ROUTING=raw uses PI itself."""
    return np.einsum("nsv,nsb->nbv", slot_C, apply_pi_routing(PI))


def route_with_matrix(slot_C, R):
    """Same contraction with an arbitrary slot->byte routing/weight matrix R (n, slot, byte)."""
    return np.einsum("nsv,nsb->nbv", slot_C, R)


def hungarian_routing(PI):
    """MAP permutation per trace: maximise sum_slot PI[slot, sigma(slot)] over bijections."""
    n = PI.shape[0]
    R = np.zeros_like(PI)
    rows = np.arange(PI_CLASSES)
    for t in range(n):
        cols = linear_sum_assignment(-PI[t][rows])[1]
        R[t, rows, cols] = 1.0
    return R


def sinkhorn_routing(PI, iters=60, eps=1e-12):
    """Doubly-stochastic scaling of the pi posterior: the soft version of the permutation
    constraint, so a slot the model is unsure about borrows mass from its competitors."""
    K = np.maximum(np.asarray(PI, dtype=np.float64), eps)
    for _ in range(iters):
        K = K / K.sum(axis=1, keepdims=True)
        K = K / K.sum(axis=2, keepdims=True)
    return K


def oracle_routing(perm_index):
    """True routing: slot s carries byte perm_index[s]."""
    n = perm_index.shape[0]
    R = np.zeros((n, PI_CLASSES, PI_CLASSES), dtype=np.float64)
    R[np.arange(n)[:, None], np.arange(PI_CLASSES)[None, :], perm_index] = 1.0
    return R


# ------------------------------------------------------------------ metrics


def per_trace_log_scores(byte_post, plaintexts, keys, n):
    """(n, 256) per-trace log scores; hypothesis i is Sbox[p ^ k ^ i] (simulated-key layout).

    Zero handling copies ASCAD_test_models.rank(): a zero probability contributes
    log(row_min_nonzero ** 2), not -inf. The 256 gathered entries are a permutation of the head's
    own row, so the row minimum is the same either way.
    """
    p = np.asarray(byte_post[:n], dtype=np.float64)
    pt = np.asarray(plaintexts[:n], dtype=np.uint8)[:, None]
    ky = np.asarray(keys[:n], dtype=np.uint8)[:, None]
    hyp = np.arange(256, dtype=np.uint8)[None, :]
    sbox_idx = AES_Sbox[np.bitwise_xor(np.bitwise_xor(pt, ky), hyp)]
    probs = p[np.arange(n)[:, None], sbox_idx]
    nzmin = np.array([np.min(r[r > 0]) if np.any(r > 0) else 1e-45 for r in probs])[:, None]
    return np.where(probs > 0, np.log(np.where(probs > 0, probs, 1.0)), 2.0 * np.log(nzmin))


def diagnose_scores(scores, real_key=0):
    """Same fields as experiments/rank_degeneracy.diagnose, so numbers are comparable across tasks."""
    s = np.asarray(scores, dtype=np.float64)
    true = s[real_key]
    spread = float(s.max() - s.min())
    return {"spread": spread,
            "n_tied_with_true": int(np.sum(np.isclose(s, true, rtol=0, atol=1e-9))),
            "rank_strict": int(np.sum(s > true)),
            "rank_worst_tie": int(np.sum(s >= true) - 1),
            "z_true": float((true - s.mean()) / s.std()) if s.std() > 0 else 0.0,
            "flat": bool(spread < 1e-6),
            "near_flat": bool(spread < 1e-3)}


def byte_metrics(byte_post, plaintexts, keys, n, rank_step):
    lp = per_trace_log_scores(byte_post, plaintexts, keys, n)
    d = diagnose_scores(lp.sum(axis=0))
    curve = ranks_from_log_scores(lp, 0, rank_step, tie_break="worst")
    hit = curve[curve[:, 1] == 0]
    d["traces_to_rank0_worst"] = int(hit[0, 0]) if hit.size else None
    d["final_rank_worst"] = int(curve[-1, 1])
    d["min_rank_worst"] = int(curve[:, 1].min())
    return d


def summarise(per_byte):
    z = [d["z_true"] for d in per_byte]
    tr = [d["traces_to_rank0_worst"] for d in per_byte]
    got = [t for t in tr if t is not None]
    return {
        "median_z": float(np.median(z)),
        "min_z": float(np.min(z)),
        "max_z": float(np.max(z)),
        "median_spread": float(np.median([d["spread"] for d in per_byte])),
        "n_rank0_final": int(sum(1 for d in per_byte if d["final_rank_worst"] == 0)),
        "n_rank0_and_not_flat": int(sum(1 for d in per_byte
                                        if d["final_rank_worst"] == 0 and not d["flat"])),
        "n_flat_bytes": int(sum(1 for d in per_byte if d["flat"])),
        "n_near_flat_bytes": int(sum(1 for d in per_byte if d["near_flat"])),
        "worst_final_rank": int(max(d["final_rank_worst"] for d in per_byte)),
        "median_traces_to_rank0": float(np.median(got)) if got else None,
        "max_traces_to_rank0": int(max(got)) if got else None,
        "n_bytes_reach_rank0": len(got),
        "traces_to_rank0_per_byte": tr,
    }


# ------------------------------------------------------------------ heads


def head_list_to_arrays(preds):
    """The 34 cached heads as (alpha, beta, C[trace, slot, value], PI[trace, slot, byte]).

    permind_i_output is the posterior for SLOT i over which BYTE it carries, so the stack axis is
    the slot axis -- getting this the other way round silently degrades the routing (it showed up as
    10.8% pi element accuracy instead of the 67.03% S1 measured, and z 2.12 instead of 9.50).
    """
    alpha = preds["alpha_output"].astype(np.float64)
    beta = preds["beta_output"].astype(np.float64)
    C = np.stack([preds[f"sbox_{i}_output"] for i in range(16)], axis=1).astype(np.float64)
    PI = np.stack([preds[f"permind_{i}_output"] for i in range(16)], axis=1).astype(np.float64)
    return alpha, beta, C, PI


def check_pi_orientation(PI, perm_index, expected_element_accuracy):
    """Guard against the transposed-read failure mode: the argmax must match the labels at the
    accuracy S1 measured for these heads, on the axis we think it is."""
    got = float(np.mean(PI.argmax(axis=2) == perm_index))
    if abs(got - expected_element_accuracy) > 0.02:
        raise RuntimeError(
            f"pi head orientation looks wrong: element accuracy {got:.4f} vs expected "
            f"{expected_element_accuracy:.4f} (S1 measured 0.6703 for permind_*. "
            "Do not report any decoder number until this is explained.)")
    return got


def one_hot(labels, k=VALUE_CLASSES, eps=EPS_ORACLE):
    """Near-hard posterior for an oracle head; works for (n,) and (n, m) label arrays."""
    labels = np.asarray(labels, dtype=np.int64)
    out = np.full(labels.shape + (k,), eps / k, dtype=np.float64)
    np.put_along_axis(out.reshape(-1, k), labels.reshape(-1, 1), 1.0 - eps + eps / k, axis=1)
    return out / out.sum(axis=-1, keepdims=True)


def uniform_head(shape, k):
    return np.full(shape[:-1] + (k,), 1.0 / k, dtype=np.float64)


# ------------------------------------------------------------------ variants


def build_variants(names):
    """Each entry: (description, head replacements, whether a shuffled null is meaningful)."""
    table = {
        "V0_repo_marginal": ("Baseline: repo path, independent pi marginals", {}, "model"),
        "V1_hungarian": ("Hungarian MAP permutation per trace", {}, "model"),
        "V2_sinkhorn": ("Sinkhorn doubly-stochastic pi coupling", {}, "model"),
        "V3_oracle_pi": ("True permutation routing", {"pi": "oracle"}, "oracle"),
        "V4_oracle_beta": ("True beta replaces the beta head", {"beta": "oracle"}, "oracle"),
        "V5_oracle_alpha": ("True alpha replaces the alpha head", {"alpha": "oracle"}, "oracle"),
        "V6_oracle_alpha_beta": ("True alpha and beta", {"alpha": "oracle", "beta": "oracle"},
                                 "oracle"),
        "V7_oracle_C": ("True masked S-box value replaces the C heads", {"C": "oracle"}, "oracle"),
        "V8_oracle_C_pi": ("True C and true routing", {"C": "oracle", "pi": "oracle"}, "oracle"),
        "V9_uniform_C": ("Negation control: C heads carry nothing", {"C": "uniform"}, "model"),
        "V10_uniform_pi": ("Negation control: routing carries nothing", {"pi": "uniform"}, "model"),
        "V11_uniform_alpha_beta": ("Negation control: alpha and beta carry nothing",
                                   {"alpha": "uniform", "beta": "uniform"}, "model"),
        "V12_uniform_alpha_only": ("Negation control: alpha head only", {"alpha": "uniform"},
                                  "model"),
        "V13_uniform_beta_only": ("Negation control: beta head only", {"beta": "uniform"}, "model"),
        "V14_map_alpha_beta": ("Plug-in decoder: use the argmax of alpha and beta instead of "
                               "marginalising over their posteriors",
                               {"alpha": "map", "beta": "map"}, "model"),
        "V15_map_alpha_beta_sinkhorn": ("MAP alpha/beta with Sinkhorn routing",
                                        {"alpha": "map", "beta": "map"}, "model"),
    }
    return {k: table[k] for k in names}


ALL_VARIANTS = ["V0_repo_marginal", "V1_hungarian", "V2_sinkhorn", "V3_oracle_pi",
                "V4_oracle_beta", "V5_oracle_alpha", "V6_oracle_alpha_beta", "V7_oracle_C",
                "V8_oracle_C_pi", "V9_uniform_C", "V10_uniform_pi", "V11_uniform_alpha_beta",
                "V12_uniform_alpha_only", "V13_uniform_beta_only", "V14_map_alpha_beta",
                "V15_map_alpha_beta_sinkhorn"]

ROUTING = {"V1_hungarian": "hungarian", "V2_sinkhorn": "sinkhorn",
           "V15_map_alpha_beta_sinkhorn": "sinkhorn"}


def run_variant(kind, needs, n, heads, lab, routing=None, seed_perm=None):
    """Build the head set for one variant, decode it, return (n, 16 bytes, 256 values), notes."""
    alpha, beta, C, PI = heads
    a, b, c, pi = alpha[:n], beta[:n], C[:n], PI[:n]
    notes = {}
    for head, mode in needs.items():
        if mode == "oracle":
            truth = {"alpha": lambda: one_hot(lab["alpha"][:n]),
                     "beta": lambda: one_hot(lab["beta"][:n]),
                     "C": lambda: one_hot(lab["sbox_masked"][:n]),
                     "pi": lambda: one_hot(lab["perm_index"][:n], k=PI_CLASSES)}[head]()
            notes[head] = f"oracle: true labels, one-hot with eps={EPS_ORACLE}"
        elif mode == "map":
            continue                                  # applied after the shuffle, see below
        else:
            k = PI_CLASSES if head == "pi" else VALUE_CLASSES
            shape = {"alpha": a, "beta": b, "C": c, "pi": pi}[head].shape
            truth = uniform_head(shape, k)
            notes[head] = "replaced by the uniform distribution"
        if head == "alpha":
            a = truth
        elif head == "beta":
            b = truth
        elif head == "C":
            c = truth
        else:
            pi = truth
    if seed_perm is not None:                          # shuffle the MODEL heads, labels stay put
        a, b, c, pi = a[seed_perm], b[seed_perm], c[seed_perm], pi[seed_perm]
    for head, mode in needs.items():
        if mode != "map":
            continue
        if head == "alpha":
            a = one_hot(a.argmax(axis=-1))
        else:
            b = one_hot(b.argmax(axis=-1))
        notes[head] = f"MAP plug-in: argmax of the {head} head, eps={EPS_ORACLE}"

    slot_C = slot_C_posteriors(c, b, a)
    how = ROUTING.get(kind)
    R = apply_pi_routing(pi) if how is None else (hungarian_routing(pi) if how == "hungarian"
                                else sinkhorn_routing(pi))
    if routing is not None:
        R = routing
    return route_with_matrix(slot_C, R), notes


def c_head_target_table(C, lab, n):
    """What does a C head actually predict? Top-1 and log-lik gain against candidate targets.

    The heads are trained on V_slot = alpha*C ^ beta, score BELOW uniform on that label
    (-0.09 nats, S1), and yet uniform-C kills the whole attack (V9). This asks what their
    256-vector is aligned with instead.
    """
    pi = lab["perm_index"][:n]
    a = lab["alpha"][:n][:, None]
    b = lab["beta"][:n][:, None]
    px = np.bitwise_xor(lab["plaintext"][:n], lab["key"][:n])
    Cbyte = AES_Sbox[px]
    Vbyte = MUL_TABLE[a, Cbyte] ^ b
    Cslot = np.take_along_axis(Cbyte, pi, axis=1)
    Vslot = np.take_along_axis(Vbyte, pi, axis=1)
    arg = C[:n].argmax(axis=2).astype(np.int64)
    targets = {
        "V_slot__own_label__alpha_mul_C_xor_beta": Vslot,
        "C_slot__unmasked_sbox_value_of_the_byte_it_carries": Cslot,
        "alpha_mul_C_slot__V_xor_beta": Vslot ^ b,
        "C_slot_xor_beta": Cslot ^ b,
        "plaintext_xor_key_of_the_byte_it_carries": np.take_along_axis(px, pi, axis=1),
    }
    out = {"own_label_reconstructs_stored_sbox_masked":
               bool(np.array_equal(Vslot.astype(np.int64), lab["sbox_masked"][:n]))}
    head = C[:n]
    rowmin = head.min(axis=2)
    floor = np.where(rowmin > 0, rowmin ** 2, 1e-45)
    for name, y in targets.items():
        y = y.astype(np.int64)
        pv = np.take_along_axis(head, y[:, :, None], axis=2)[:, :, 0]
        out[name] = {
            "top1_acc": float(np.mean(arg == y)),
            "gain_over_uniform_nats": float(np.mean(np.log(np.where(pv > 0, pv, floor)))
                                             + np.log(VALUE_CLASSES)),
        }
    return out


def run_on_indices(kind, idx, heads, lab, rank_step):
    """Decode one variant on an arbitrary trace subset and return its summary."""
    needs = build_variants([kind])[kind][1]
    m = len(idx)
    hs = tuple(h[idx] for h in heads)
    sub = {k: np.asarray(v)[idx] for k, v in lab.items()}
    post, _ = run_variant(kind, needs, m, hs, sub)
    per_byte = [byte_metrics(post[:, b], sub["plaintext"][:, b], sub["key"][:, b], m, rank_step)
                for b in range(16)]
    return summarise(per_byte)


def split_half_stability(kinds, heads, lab, n_traces, rank_step):
    """Median z on two disjoint halves: is a small gain over V0 consistent, or one half's noise?"""
    half = n_traces // 2
    first, second = np.arange(0, half), np.arange(half, n_traces)
    return {k: {"first_half": run_on_indices(k, first, heads, lab, rank_step)["median_z"],
                "second_half": run_on_indices(k, second, heads, lab, rank_step)["median_z"]}
            for k in kinds}


# ------------------------------------------------------------------ main


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--num-traces", type=int, default=5000)
    ap.add_argument("--rank-step", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N traces")
    ap.add_argument("--variants", default="")
    ap.add_argument("--out-json", default=DEFAULT_OUT)
    args = ap.parse_args()

    t_start = time.time()
    n_traces = args.limit or args.num_traces

    if not os.path.exists(CACHE):
        raise SystemExit(f"missing cached predictions {CACHE} (produced by experiments/ablation_heads.py)")
    z = np.load(CACHE)
    if n_traces > z["alpha_output"].shape[0]:
        raise SystemExit(f"cache holds {z['alpha_output'].shape[0]} traces, asked for {n_traces}")
    with h5py.File(H5, "r") as f:
        lab_raw = f["Attack_traces/labels"][:n_traces]
        meta = f["Attack_traces/metadata"][:n_traces]
    lab = {
        "alpha": lab_raw["alpha_mask"].reshape(-1).astype(np.int64),
        "beta": lab_raw["beta_mask"].reshape(-1).astype(np.int64),
        "sbox_masked": np.asarray(lab_raw["sbox_masked"], dtype=np.int64),
        "perm_index": np.asarray(lab_raw["perm_index"], dtype=np.int64),
        "plaintext": np.asarray(meta["plaintext"], dtype=np.uint8),
        "key": np.asarray(meta["key"], dtype=np.uint8),
    }
    preds = {k: z[k][:n_traces] for k in z.files}
    heads = head_list_to_arrays(preds)
    alpha, beta, C, PI = heads
    print(f"traces={n_traces} heads=alpha(256),beta(256),C[16](256),pi[16](16) "
          f"GF primitive element={GF_GENERATOR}", flush=True)

    # --- (a) how good is the routing information itself? --------------------------
    pi_argmax = PI.argmax(axis=2)
    elem_argmax = check_pi_orientation(PI, lab["perm_index"], 0.6703)
    exact_argmax = float(np.mean(np.all(pi_argmax == lab["perm_index"], axis=1)))
    exact_hung = float(np.mean(
        np.all(np.stack([linear_sum_assignment(-PI[t])[1] for t in range(n_traces)])
               == lab["perm_index"], axis=1)))
    print(f"pi element accuracy argmax={elem_argmax:.4f}; exact permutation: "
          f"argmax={exact_argmax:.4f} hungarian={exact_hung:.4f}", flush=True)

    # --- (b) sequential dissection vs the explicit joint double sum ---------------
    n_joint = min(8, n_traces)
    fast = slot_C_posteriors(C[:n_joint], beta[:n_joint], alpha[:n_joint])
    ref = np.empty((n_joint, 16, 256))
    worst = 0.0
    for s in range(16):
        ref[:, s] = joint_alpha_beta_reference(C[:n_joint, s], alpha[:n_joint], beta[:n_joint])
        worst = max(worst, float(np.max(np.abs(ref[:, s] - fast[:, s]))))
    print(f"(b) joint 256x256 double sum vs sequential dissection: max abs diff {worst:.3e} "
          f"on {n_joint} traces x 16 slots", flush=True)

    # --- parity with the repo decoder on the REAL cached predictions --------------
    n_sub = min(1000, n_traces)
    lst = ([alpha[:n_sub], beta[:n_sub]] + [C[:n_sub, s] for s in range(16)]
           + [PI[:n_sub, s, :] for s in range(16)])
    want = np.stack(multilabel_predict(lst), axis=1)
    got = route_marginal(slot_C_posteriors(C[:n_sub], beta[:n_sub], alpha[:n_sub]), PI[:n_sub])
    parity = float(np.max(np.abs(want - got)))
    print(f"repo parity on real preds ({n_sub} traces): max abs diff {parity:.3e}", flush=True)

    # --- (c) what the C heads actually represent, and a stability check on small gains ---
    c_table = c_head_target_table(C, lab, n_traces)
    for k, v in c_table.items():
        if k.startswith("V_") or k.startswith("C_") or k.startswith("alpha") or k.startswith("plai"):
            print(f"  C head vs {k:52s} top1 {v['top1_acc'] * 100:6.2f}%  "
                  f"gain {v['gain_over_uniform_nats']:+.3f} nats", flush=True)
    stability = split_half_stability(
        ["V0_repo_marginal", "V1_hungarian", "V2_sinkhorn", "V4_oracle_beta"],
        heads, lab, n_traces, args.rank_step)
    print("split-half median z:", json.dumps(stability), flush=True)

    names = [v.strip() for v in args.variants.split(",") if v.strip()] or ALL_VARIANTS
    unknown = [v for v in names if v not in ALL_VARIANTS]
    if unknown:
        raise SystemExit(f"unknown variants {unknown}; available: {ALL_VARIANTS}")
    variants = build_variants(names)
    rng = np.random.default_rng(0)
    perm = rng.permutation(n_traces)

    results = {
        "meta": {
            "step": "T-030 S2",
            "cache": os.path.relpath(CACHE, repo_root),
            "h5": os.path.relpath(H5, repo_root),
            "n_traces": int(n_traces),
            "rank_step": args.rank_step,
            "tie_break": "worst",
            "eps_oracle": EPS_ORACLE,
            "numpy": np.__version__,
            "scipy": __import__("scipy").__version__,
            "python": sys.version.split()[0],
            "git": os.popen("git -C %s rev-parse --short HEAD" % repo_root).read().strip(),
            "gain_and_z": "z = (score[true] - mean) / std over the 256 key hypotheses, summed "
                          "log-likelihood over all traces; same definition as "
                          "experiments/rank_degeneracy.diagnose",
        },
        "pi_quality": {
            "element_accuracy_argmax": elem_argmax,
            "exact_permutation_argmax": exact_argmax,
            "exact_permutation_hungarian": exact_hung,
        },
        "joint_alpha_beta_check": {
            "traces": n_joint, "max_abs_diff_vs_sequential": worst,
            "meaning": "0 => the per-slot sequential dissection already IS the joint (alpha,beta) "
                       "marginalisation under p(alpha,beta|t)=p(alpha|t)p(beta|t)",
        },
        "repo_parity_on_real_preds": {
            "traces": n_sub, "max_abs_diff_vs_multilabel_predict": parity,
            "meaning": "V0's decode is the repo pipeline, so baseline numbers stay comparable",
        },
        "c_head_target_table": c_table,
        "split_half_median_z": stability,
        "variants": {},
    }

    base_z = None
    for kind, (desc, needs, null_kind) in variants.items():
        t0 = time.time()
        post, notes = run_variant(kind, needs, n_traces, heads, lab)
        per_byte = [byte_metrics(post[:, b], lab["plaintext"][:, b], lab["key"][:, b],
                                 n_traces, args.rank_step) for b in range(16)]
        entry = {"description": desc, "oracle_or_uniform": notes,
                 "summary": summarise(per_byte),
                 "per_byte": {str(b): per_byte[b] for b in range(16)},
                 "wall_s": round(time.time() - t0, 1)}
        if null_kind == "model":
            npost, _ = run_variant(kind, needs, n_traces, heads, lab, seed_perm=perm)
            nper = [byte_metrics(npost[:, b], lab["plaintext"][:, b], lab["key"][:, b],
                                 n_traces, args.rank_step) for b in range(16)]
            entry["null_shuffled_model_heads"] = summarise(nper)
        else:
            entry["null_shuffled_model_heads"] = {
                "not_applicable": "an oracle head carries true labels, so a shuffled null of the "
                                  "other heads is not a chance-level reference"}
        results["variants"][kind] = entry
        s = entry["summary"]
        print(f"{kind:24s} z_med {s['median_z']:+9.2f} rank0 {s['n_rank0_final']:2d}/16 "
              f"genuine {s['n_rank0_and_not_flat']:2d}/16 near_flat {s['n_near_flat_bytes']:2d} "
              f"med_traces->r0 {s['median_traces_to_rank0']} "
              f"null_z {(entry.get('null_shuffled_model_heads') or {}).get('median_z', 'n/a')}"
              f"  ({entry['wall_s']}s)", flush=True)
        if kind == "V0_repo_marginal":
            base_z = s["median_z"]
            if abs(base_z - BASELINE_Z) > Z_TOL:
                results["gates"] = {"sanity_reproduce_baseline_z_9_50": {
                    "measured_median_z": base_z, "documented": BASELINE_Z,
                    "tolerance": Z_TOL, "pass": False,
                    "action": "STOP: the decoder does not reproduce the baseline, so no variant "
                              "number here is comparable (S2 spec: stop and report)"}}
                with open(args.out_json, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2)
                raise SystemExit(f"SANITY GATE FAILED: V0 median z {base_z:+.4f} vs baseline "
                                 f"+{BASELINE_Z} (+/-{Z_TOL}). Wrote partial results to "
                                 f"{args.out_json}; stopping before the other variants.")

    base = results["variants"].get("V0_repo_marginal")
    if base:
        bz = np.array([base["per_byte"][str(b)]["z_true"] for b in range(16)])
        btr = [t for t in base["summary"]["traces_to_rank0_per_byte"] if t is not None]
        for kind, e in results["variants"].items():
            vz = np.array([e["per_byte"][str(b)]["z_true"] for b in range(16)])
            d = vz - bz
            vtr = [t for t in e["summary"]["traces_to_rank0_per_byte"] if t is not None]
            e["vs_V0_paired"] = {
                "median_z_delta": float(np.median(vz) - np.median(bz)),
                "mean_paired_delta": float(d.mean()),
                "se_across_16_bytes": float(d.std(ddof=1) / np.sqrt(len(d))),
                "bytes_better": int((d > 0).sum()),
                "median_traces_to_rank0": e["summary"]["median_traces_to_rank0"],
                "traces_delta_vs_V0": (None if not vtr or not btr
                                       else float(np.median(vtr) - np.median(btr))),
            }

    results["gates"] = {
        "sanity_reproduce_baseline_z_9_50": {
            "measured_median_z": base_z,
            "documented": BASELINE_Z,
            "s1_independent_reproduction": BASELINE_Z_VERIFIED,
            "tolerance": Z_TOL,
            "pass": bool(base_z is not None and abs(base_z - BASELINE_Z) <= Z_TOL),
        },
        "improvement": "a decoder variant passes if its median z exceeds V0's, or its median "
                       "traces-to-rank-0 is smaller, with no flat/near-flat bytes and its null "
                       "sitting near z=0 / rank 127",
        "decoder_parity": {
            "max_abs_diff_vs_multilabel_predict": parity,
            "max_abs_diff_joint_vs_sequential": worst,
            "pass": bool(parity < 1e-9 and worst < 1e-9),
        },
    }
    results["meta"]["total_wall_s"] = round(time.time() - t_start, 1)
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {args.out_json} (total {results['meta']['total_wall_s']}s)", flush=True)


if __name__ == "__main__":
    main()
