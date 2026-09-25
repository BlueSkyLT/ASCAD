#!/usr/bin/env python3
"""GPAM on ASCADv2: reproduce the quantities GPAM's Table 11 actually reports,
then attempt the separate ~80-trace GE claim from section 7.1.

Table 11 is per-head Accuracy / MeanRank / MaxRank of the predicted VALUE on the
holdout -- it is not GE. The ~80-trace number is prose in 7.1 and is obtained by
simulating a fixed all-zero key (replace pt by pt ^ key), taking rm by argmax,
and accumulating log-probabilities across traces.

Everything here ranks with model-predicted masks. ASCADv2 ships true alpha /
beta / sbox / perm in Attack_traces/labels, so reading masks from labels would
silently turn this into a worst-case attack.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
from tensorflow import keras

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from run_ascadv2_gpam_baseline import AES_SBOX, gf256_mul, perm_indices  # noqa: E402
from scaaml.models import get_gpam_model  # noqa: E402

DEFAULT_DB = str(REPO_ROOT / "STM32_AES_v2/ASCAD_data/ASCAD_databases/ascadv2-extracted.h5")
ARTIFACTS = str(REPO_ROOT / "GPAM/artifacts")

# Paper name -> stored ASCADv2 label field, and class count.
HEADS = {
    "rm": ("alpha_mask", 256),          # multiply hide alpha
    "rout": ("beta_mask", 256),         # add hide beta
    "c": ("sbox_masked", 256),          # masked S-box byte
    "p": ("perm_index", 16),            # shuffle index for this byte
}

MUL = np.array([[gf256_mul(a, z) for z in range(256)] for a in range(256)], dtype=np.uint8)


def rank_worst(score, target):
    """Rank of `target` counting ties AGAINST it, so a flat vector ranks last.

    Strict `>` ranking reports rank 0 for a constant score vector, which lets an
    entirely uninformative head look like a key recovery. For a head with no
    exact ties the two conventions agree, so this only changes degenerate cases.
    """
    return int((score >= score[target]).sum() - 1)


def value_rank(probs, y):
    """Rank of the true class, 0 = best. This is Table 11's MeanRank/MaxRank."""
    s = np.sort(probs, axis=1)[:, ::-1]
    true_p = probs[np.arange(len(y)), y]
    return (s >= true_p[:, None]).sum(1).astype(np.int32) - 1


def topk_acc(probs, y, k):
    idx = np.argpartition(-probs, k - 1, axis=1)[:, :k]
    return float((idx == np.asarray(y)[:, None]).any(1).mean())


def load(path, n_prof, n_attack, byte, holdout_slice):
    with h5py.File(path, "r") as f:
        tr = f["Profiling_traces/traces"][holdout_slice[0]:holdout_slice[1]].astype(np.float32)
        lab = f["Profiling_traces/labels"][holdout_slice[0]:holdout_slice[1]]
        meta = f["Profiling_traces/metadata"][holdout_slice[0]:holdout_slice[1]]
        at = f["Attack_traces/traces"][:n_attack].astype(np.float32)
        alab = f["Attack_traces/labels"][:n_attack]
        ameta = f["Attack_traces/metadata"][:n_attack]
        pt_all = f["Profiling_traces/traces"][:n_prof].astype(np.float32)

    def cols(struct):
        return {
            "rm": np.asarray(struct["alpha_mask"])[:, 0].astype(np.int32),
            "rout": np.asarray(struct["beta_mask"])[:, 0].astype(np.int32),
            "c": np.asarray(struct["sbox_masked"])[:, byte].astype(np.int32),
            "p": np.asarray(struct["perm_index"])[:, byte].astype(np.int32),
        }

    return pt_all, cols(lab), meta, at, cols(alab), ameta, tr


def build(x_train, patch_size):
    trace_len = int(x_train.shape[1])
    if trace_len % patch_size:
        raise ValueError(f"trace_len {trace_len} not divisible by patch {patch_size}")
    steps = trace_len // patch_size
    if steps % 2:
        raise ValueError(f"GPAM RoPE needs an even step count, got {steps}")
    tmin, tmax = float(x_train.min()), float(x_train.max())
    model = get_gpam_model(
        inputs={"trace1": {"min": tmin, "delta": tmax - tmin}},
        outputs={h: {"max_val": HEADS[h][1]} for h in HEADS},
        output_relations=[],
        trace_len=trace_len,
        merge_filter_1=0,
        merge_filter_2=0,
        patch_size=patch_size,
    )
    model.compile(
        optimizer=keras.optimizers.Adafactor(learning_rate=1e-3),
        loss={h: "sparse_categorical_crossentropy" for h in HEADS},
        metrics={h: ["accuracy"] for h in HEADS},
        run_eagerly=True,
    )
    return model


def table11(probs, y, n_classes):
    r = value_rank(probs, y)
    return {
        "acc_pct": round(100.0 * float((probs.argmax(1) == y).mean()), 4),
        "mean_rank": round(float(r.mean()), 4),
        "max_rank": int(r.max()),
        "top5_acc_pct": round(100.0 * topk_acc(probs, y, 5), 4),
        "chance_acc_pct": round(100.0 / n_classes, 4),
    }


def ge_fixed_key_simulation(probs, pt_eff, perm_hat, rm_hat, byte, n, n_trace_curve):
    """Paper 7.1: all-zero key by pt_eff = pt ^ key; rm by argmax; rout marginalised.

    score[g] accumulates log P_c( rm_hat * SBOX[pt_eff[pos] ^ g] ^ rout ).
    GE is the rank of hypothesis g = 0, the simulated true key byte.
    """
    pc, prout = probs["c"], probs["rout"]
    pos = np.asarray(perm_hat)[:n]
    rm = np.asarray(rm_hat)[:n]
    curve_T = sorted(set(list(n_trace_curve) + [n]))
    out = {}
    logpc = np.log(np.clip(pc[:n], 1e-30, 1.0))
    for T in curve_T:
        score = np.zeros(256)
        g = np.arange(256, dtype=np.uint8)
        for i in range(T):
            ind = pos[i]
            s_all = MUL[rm[i]][AES_SBOX[pt_eff[i][ind] ^ g]].astype(np.uint8)
            # XOR-convolution of the c head with the rout head, at 256 offsets
            idx = np.bitwise_xor.outer(s_all.astype(np.uint16), np.arange(256, dtype=np.uint16))
            lik = np.exp(logpc[i][idx] + np.log(np.clip(prout[i], 1e-30, 1.0))[None, :]).sum(1)
            score += np.log(np.clip(lik, 1e-30, 1.0))
        rank0 = rank_worst(score, 0)
        out[T] = rank0
    return out


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--dataset-path", default=DEFAULT_DB)
    a.add_argument("--n-train", type=int, default=20000)
    a.add_argument("--n-val", type=int, default=2000)
    a.add_argument("--n-attack", type=int, default=2000)
    a.add_argument("--target-byte", type=int, default=0)
    a.add_argument("--patch-size", type=int, default=150)
    a.add_argument("--epochs", type=int, default=6)
    a.add_argument("--batch-size", type=int, default=64)
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--ge-traces", type=int, nargs="+", default=[1, 10, 25, 50, 80, 100, 200, 500])
    a.add_argument("--out-json", default=f"{ARTIFACTS}/C-gpam-eval.json")
    a = a.parse_args()

    np.random.seed(a.seed)
    keras.utils.set_random_seed(a.seed)
    t0 = time.time()
    b = a.target_byte

    # Holdout = the tail of Profiling, so evaluation data is not training data.
    hold_start = 400000
    x_prof_all, _, _, x_atk, y_atk, ameta, x_hold = load(
        a.dataset_path, a.n_train + a.n_val, a.n_attack, b, (hold_start, hold_start + max(a.n_val, 2000))
    )
    xtr = x_prof_all[: a.n_train]
    with h5py.File(a.dataset_path, "r") as f:
        ytr_struct = f["Profiling_traces/labels"][: a.n_train]
        yva_struct = f["Profiling_traces/labels"][hold_start: hold_start + x_hold.shape[0]]
        meta_hold = f["Profiling_traces/metadata"][hold_start: hold_start + x_hold.shape[0]]

    def col(struct, h):
        raw = np.asarray(struct[HEADS[h][0]])
        return raw[:, b].astype(np.int32) if raw.ndim == 2 else raw[:, 0].astype(np.int32)

    ytr = {h: col(ytr_struct, h) for h in HEADS}
    yva = {h: col(yva_struct, h) for h in HEADS}
    print(f"[{time.time()-t0:6.1f}s] train {xtr.shape} holdout {x_hold.shape} atk {x_atk.shape}", flush=True)

    model = build(xtr, a.patch_size)
    print(f"[{time.time()-t0:6.1f}s] built (steps={xtr.shape[1]//a.patch_size}) -- fitting", flush=True)
    hist = model.fit(xtr, ytr, validation_data=(x_hold, yva), batch_size=a.batch_size,
                     epochs=a.epochs, verbose=2)
    print(f"[{time.time()-t0:6.1f}s] trained {len(hist.history['loss'])} epochs", flush=True)

    res = {"config": vars(a), "wall_seconds": round(time.time() - t0, 1),
           "epochs_run": len(hist.history["loss"]), "n_holdout": int(x_hold.shape[0]),
           "paper_table11": {"rm": [100.0, 0, 0], "rout": [18.25, 4.78, 65],
                             "c_avg": [1.18, 80.65, 255], "p_avg": [95.07, 0.055, 4.44]}}

    ph = model.predict(x_hold, batch_size=256, verbose=0)
    res["table11_holdout"] = {h: table11(ph[h], yva[h], HEADS[h][1]) for h in HEADS}
    for h in HEADS:
        m = res["table11_holdout"][h]
        print(f"HOLDOUT {h:5s} acc={m['acc_pct']:7.3f}% (chance {m['chance_acc_pct']:.3f}%) "
              f"mean_rank={m['mean_rank']:8.3f} max_rank={m['max_rank']} top5={m['top5_acc_pct']:.3f}%", flush=True)

    pa = model.predict(x_atk, batch_size=256, verbose=0)
    res["table11_attack"] = {h: table11(pa[h], y_atk[h], HEADS[h][1]) for h in HEADS}
    for h in HEADS:
        m = res["table11_attack"][h]
        print(f"ATTACK  {h:5s} acc={m['acc_pct']:7.3f}% mean_rank={m['mean_rank']:8.3f} max_rank={m['max_rank']}", flush=True)

    n = min(a.n_attack, x_atk.shape[0])
    amasks = np.stack(ameta["masks"])[:n]
    akey = np.stack(ameta["key"])[:n]
    aplain = np.stack(ameta["plaintext"])[:n]
    pt_eff = np.bitwise_xor(aplain, akey).astype(np.uint8)  # 7.1 fixed-key simulation
    perm_hat = pa["p"].argmax(1)[:n]
    rm_hat = pa["rm"].argmax(1)[:n]
    perm_true = np.array([perm_indices(b, m[0], m[1], m[2], m[3]) for m in amasks], dtype=np.int32)
    res["perm_head_agrees_with_recompute_pct"] = round(100.0 * float((perm_hat == perm_true[:n]).mean()), 3)
    res["rm_argmax_equals_true_pct"] = round(100.0 * float((rm_hat == y_atk["rm"][:n]).mean()), 3)
    print(f"p-head argmax == recomputed perm: {res['perm_head_agrees_with_recompute_pct']}% "
          f"(label alpha==masks[18] argmax match: {res['rm_argmax_equals_true_pct']}%)", flush=True)

    print(f"[{time.time()-t0:6.1f}s] GE curve (fixed-key simulation, predicted masks)...", flush=True)
    curve = ge_fixed_key_simulation(pa, pt_eff, perm_hat, rm_hat, b, n, a.ge_traces)
    res["ge_fixed_key_sim"] = {str(k): v for k, v in curve.items()}
    for T, r in curve.items():
        print(f"GE byte {b} @ {T:5d} traces -> rank {r}", flush=True)

    # Trace-independent prior on the c head: can frequency alone fake a low rank?
    cnt = np.bincount(ytr["c"], minlength=256).astype(np.float64)
    logprior = np.log((cnt + 1.0) / (cnt.sum() + 256))
    g = np.arange(256, dtype=np.uint8)
    score = np.zeros(256)
    for i in range(n):
        s_all = MUL[rm_hat[i]][AES_SBOX[pt_eff[i][perm_hat[i]] ^ g]]
        score += logprior[s_all]
    res["ge_prior_only"] = rank_worst(score, 0)
    print(f"GE from trace-independent prior alone @ {n} traces -> rank {res['ge_prior_only']}", flush=True)

    Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
    res["wall_seconds_total"] = round(time.time() - t0, 1)
    with open(a.out_json, "w") as f:
        json.dump(res, f, indent=2)
    model.save(Path(a.out_json).with_suffix(".keras"))
    print(f"[{time.time()-t0:6.1f}s] wrote {a.out_json}", flush=True)


if __name__ == "__main__":
    main()
