#!/usr/bin/env python3
"""Which GPAM head blocks the 7.1 GE chain? Ablate mask knowledge.

Same trained model and attack set as C-gpam-eval; only how much mask truth the
ranker is handed changes. Each regime is a different threat model, so the gap
between them localises the bottleneck head rather than the architecture.
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

DEFAULT_DB = str(REPO_ROOT / "STM32_AES_v2/ASCAD_data/ASCAD_databases/ascadv2-extracted.h5")
ARTIFACTS = str(REPO_ROOT / "GPAM/artifacts")

MUL = np.array([[gf256_mul(a, z) for z in range(256)] for a in range(256)], dtype=np.uint8)


def rank_worst(score, target):
    """Rank of `target` counting ties AGAINST it, so a flat vector ranks last.

    Strict `>` ranking reports rank 0 for a constant score vector, which is how a
    uniform head appeared to recover the key in the E and F controls. With no
    exact ties the conventions agree, so real rankings are unchanged.
    """
    return int((score >= score[target]).sum() - 1)


def ge_curve(pc, prout, pos, rm, pt_eff, Ts):
    """Rank of simulated-zero key byte at each trace budget. rm is a hard value."""
    g = np.arange(256, dtype=np.uint8)
    maxT = max(Ts)
    score = np.zeros(256)
    hit = {}
    for i in range(maxT):
        ind = pos[i]
        s_all = MUL[rm[i]][AES_SBOX[pt_eff[i][ind] ^ g]]
        li = np.log(np.clip(pc[i], 1e-30, 1.0))
        lr = np.log(np.clip(prout[i], 1e-30, 1.0))
        idx = np.bitwise_xor.outer(s_all.astype(np.uint16), np.arange(256, dtype=np.uint16))
        lik = np.exp(li[idx] + lr[None, :]).sum(1)
        score += np.log(np.clip(lik, 1e-30, 1.0))
        if (i + 1) in Ts:
            hit[i + 1] = rank_worst(score, 0)
    return hit


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--model", default=f"{ARTIFACTS}/C-gpam-eval.keras")
    a.add_argument("--dataset-path", default=DEFAULT_DB)
    a.add_argument("--n-attack", type=int, default=2000)
    a.add_argument("--target-byte", type=int, default=0)
    a.add_argument("--ts", type=int, nargs="+", default=[80, 200, 500, 1000, 2000])
    a.add_argument("--out-json", default=f"{ARTIFACTS}/C-gpam-ablation.json")
    a = a.parse_args()

    b, n = a.target_byte, a.n_attack
    t0 = time.time()
    model = keras.models.load_model(a.model, compile=False)
    with h5py.File(a.dataset_path, "r") as f:
        x = f["Attack_traces/traces"][:n].astype(np.float32)
        lab = f["Attack_traces/labels"][:n]
        meta = f["Attack_traces/metadata"][:n]
    print(f"[{time.time()-t0:.1f}s] loaded model + {n} attack traces", flush=True)

    p = model.predict(x, batch_size=256, verbose=0)
    pc, prout = p["c"], p["rout"]
    y_true = {
        "rm": np.asarray(lab["alpha_mask"])[:, 0].astype(np.int32),
        "rout": np.asarray(lab["beta_mask"])[:, 0].astype(np.int32),
        "c": np.asarray(lab["sbox_masked"])[:, b].astype(np.int32),
        "p": np.asarray(lab["perm_index"])[:, b].astype(np.int32),
    }
    masks = np.stack(meta["masks"])[:n]
    pt = np.stack(meta["plaintext"])[:n].astype(np.uint8)
    key = np.stack(meta["key"])[:n].astype(np.uint8)
    pt_eff = np.bitwise_xor(pt, key).astype(np.uint8)

    rm_hat = p["rm"].argmax(1)[:n].astype(np.int32)
    pos_hat = p["p"].argmax(1)[:n].astype(np.int32)
    pos_true = y_true["p"][:n]
    rm_true = y_true["rm"][:n]
    print(f"rm argmax match true: {100*(rm_hat==rm_true).mean():.2f}%  "
          f"p argmax match true: {100*(pos_hat==pos_true).mean():.2f}%  "
          f"(recomputed-perm match: {100*(np.array([perm_indices(b,m[0],m[1],m[2],m[3]) for m in masks]) == pos_true).mean():.2f}%)",
          flush=True)

    uniform = np.full((n, 256), 1.0 / 256)
    flat = np.tile(np.full((1, 256), 1.0 / 256), (n, 1))
    d_rout = np.zeros((n, 256))
    d_rout[np.arange(n), y_true["rout"][:n]] = 1.0

    # Regimes differ only in how much mask truth the ranker is handed.
    regimes = {
        "A_all_predicted": dict(pos=pos_hat, rm=rm_hat, prout=prout, pc=pc),
        "B_true_perm": dict(pos=pos_true, rm=rm_hat, prout=prout, pc=pc),
        "C_true_perm_rm": dict(pos=pos_true, rm=rm_true, prout=prout, pc=pc),
        "D_all_masks_true": dict(pos=pos_true, rm=rm_true, prout=d_rout, pc=pc),
        "E_c_head_chance": dict(pos=pos_true, rm=rm_true, prout=d_rout, pc=flat),
        "F_rout_head_chance": dict(pos=pos_true, rm=rm_true, prout=uniform, pc=pc),
    }
    res = {"config": vars(a), "rm_match_pct": round(100 * float((rm_hat == rm_true).mean()), 3),
           "perm_match_pct": round(100 * float((pos_hat == pos_true).mean()), 3), "ge": {}}

    for name, r in regimes.items():
        hit = ge_curve(r["pc"], r["prout"], r["pos"], r["rm"], pt_eff, a.ts)
        res["ge"][name] = hit
        print(f"{name:22s} " + "  ".join(f"@{T}={hit[T]:3d}" for T in a.ts), flush=True)

    Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
    res["wall_seconds"] = round(time.time() - t0, 1)
    with open(a.out_json, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[{time.time()-t0:.1f}s] wrote {a.out_json}", flush=True)


if __name__ == "__main__":
    main()
