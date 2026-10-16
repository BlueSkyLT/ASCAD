#!/usr/bin/env python3
"""T-030 S1 — did the old TimesNet multi-task trunk ever learn alpha?

The old ASCADv2 TimesNet multi-task runs (100k profiling traces, multilabel=2, no perm heads)
ended at ~random ranks (T-009) and their per-head accuracy was never measured. S1 settles the
first question of T-030: was that a trunk failure (alpha never learned) or a downstream one
(C heads / decoding)?

GATE (pre-registered in docs/tasks/T030-worstcase-multitask-improvement.md, may not move):
  alpha top-1 > 50%  => the old trunk learned alpha, so its failure is downstream.
  alpha at chance    => the trunk never got going.
Both outcomes are reported plainly; neither is a "break".

Head-to-label mapping is read from ASCAD_train_models.build_multilabel_targets + the checkpoint
sidecars, not assumed: multilabel=2 trains sbox_i on `sbox_masked_with_perm[:, i]` (byte-indexed)
and has no permind heads, so none are invented here.

Controls that ride along, because this repo has been burned by both:
  * ranks use tie_break="worst" and are reported with a flatness check (spread / ties at top),
    since a constant posterior scores rank 0 under the old tie rule;
  * shuffled-prediction and constant-trace nulls.

Reads the HDF5 attack split read-only (never the profiling traces — that would cost 7.5 GB RAM).

Usage:
    conda run -n py10 python -u -m experiments.w1_timesnet_old_head_accs \
        [--num-traces 5000] [--batch 64] [--limit N] [--only SUBSTR] [--device cuda]
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import sys
import time

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import h5py
import numpy as np
import torch

from ASCAD_generate import AES_Sbox
from ASCAD_test_models import (
    multGF256,
    multilabel_without_permind_predict,
    ranks_from_log_scores,
)
from ASCAD_train_models import build_timesnet_for_network

H5 = os.path.join(repo_root, "STM32_AES_v2/ASCAD_data/ASCAD_databases/ascadv2-extracted.h5")
MODELS_DIR = os.path.join(repo_root, "STM32_AES_v2/ASCAD_data/ASCAD_trained_models")
BASELINE_CACHE = os.path.join(repo_root, "experiments", "out", "head_preds_5k.npz")
DEFAULT_OUT = os.path.join(repo_root, "experiments", "out", "timesnet_old_head_accs.json")

# S1 scope: the multi-task checkpoints listed in the T-030 spec. `t100k_e22` have no sidecar;
# their config is inferred from the state-dict shapes and must load strictly.
CKPT_PATTERNS = [
    "timesnet_ascad_head_multi_noperm_s5p32_t100k_e30*.pth",
    "timesnet_ascad_head_multi_noperm_s5p32_t100k_e60*.pth",
    "timesnet_multi_noperm_s5p32_t100k_e30*.pth",
    "timesnet_multi_noperm_s5p32_t100k_e60*.pth",
    "timesnet_ascad_head_multi_noperm_t100k_e22.pth",
    "timesnet_multi_noperm_t100k_e22.pth",
]

GF = np.vectorize(multGF256, otypes=[np.uint8])  # only used for the label-algebra check


# ---------------------------------------------------------------- data


def load_attack_split(n, path=H5):
    """First n attack traces (float32) + the label columns and metadata S1 needs."""
    with h5py.File(path, "r") as f:
        X = np.array(f["Attack_traces/traces"][:n], dtype=np.float32)
        lab = f["Attack_traces/labels"][:n]
        meta = f["Attack_traces/metadata"][:n]
    out = {
        "alpha": lab["alpha_mask"].reshape(-1).astype(np.int64),
        "beta": lab["beta_mask"].reshape(-1).astype(np.int64),
        "sbox_masked": np.asarray(lab["sbox_masked"], dtype=np.int64),
        "sbox_masked_with_perm": np.asarray(lab["sbox_masked_with_perm"], dtype=np.int64),
        "perm_index": np.asarray(lab["perm_index"], dtype=np.int64),
        "plaintext": np.asarray(meta["plaintext"], dtype=np.uint8),
        "key": np.asarray(meta["key"], dtype=np.uint8),
        "masks": np.asarray(meta["masks"], dtype=np.uint8),
    }
    return X, out


def label_algebra_check(lab, n_check=2000):
    """Verify what the C label fields actually are, from the data (checked 2026-10-05)."""
    sl = slice(0, n_check)
    c = AES_Sbox[np.bitwise_xor(lab["plaintext"][sl], lab["key"][sl])]
    a = lab["alpha"][sl][:, None]
    b = lab["beta"][sl][:, None]
    masked = GF(a, c) ^ b
    pi = lab["perm_index"][sl]
    inv = np.argsort(pi, axis=1)
    return {
        "alpha_mask_equals_masks_col_18": bool((lab["alpha"] == lab["masks"][:, 18]).all()),
        "beta_mask_equals_masks_col_17": bool((lab["beta"] == lab["masks"][:, 17]).all()),
        "sbox_masked_with_perm_eq_gf(alpha)*xor(beta)_Sbox[p^k]": float(
            (lab["sbox_masked_with_perm"][sl] == masked).mean()
        ),
        "sbox_masked_eq_byte_value_permuted_to_slots": float(
            (lab["sbox_masked"][sl] == np.take_along_axis(masked, pi, axis=1)).mean()
        ),
        "sbox_masked_with_perm_eq_sbox_masked_unpermuted": float(
            (lab["sbox_masked_with_perm"][sl]
             == np.take_along_axis(lab["sbox_masked"][sl], inv, axis=1)).mean()
        ),
        "sbox_masked_with_perm_equals_raw_Sbox_p_xor_k": float(
            (lab["sbox_masked_with_perm"][sl] == c).mean()
        ),
    }


# ---------------------------------------------------------------- metrics


def head_metrics(probs, labels):
    """Top-1 accuracy and mean log-likelihood gain over the uniform distribution.

    gain = mean_n log p(label_n | trace n) + log K, so a uniform head scores 0 nats and the
    ceiling is log K. Zeros are floored with the convention ASCAD_test_models.rank() uses
    (row minimum non-zero, squared) so this number and the rank pipeline agree; the count of
    rows containing an exact zero is reported next to it.
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    n, k = probs.shape
    p_true = probs[np.arange(n), labels]
    nz = probs[probs > 0]
    n_zero_rows = int(np.sum(np.any(probs == 0, axis=1)))
    row_min = np.array([np.min(r[r > 0]) if np.any(r > 0) else 1.0 for r in probs])
    floored = np.where(p_true > 0, p_true, row_min ** 2)
    hist = np.bincount(labels, minlength=k) / float(n)
    return {
        "n_traces": n,
        "n_classes": k,
        "top1_acc": float(np.mean(np.argmax(probs, axis=1) == labels)),
        "mean_log_p_true": float(np.mean(np.log(floored))),
        "gain_over_uniform_nats": float(np.mean(np.log(floored)) + np.log(k)),
        "uniform_gain_nats_by_construction": 0.0,
        "max_gain_nats": float(np.log(k)),
        "chance_top1_uniform": float(1.0 / k),
        "constant_predictor_top1": float(hist.max()),
        "label_distinct_values": int(np.count_nonzero(hist)),
        "rows_with_exact_zero": n_zero_rows,
        "min_nonzero_prob_overall": float(nz.min()) if nz.size else None,
    }


def byte_key_ranks(sbox_probs, plaintext, key, rank_step, tie_break="worst"):
    """Per-byte final/min rank of the key through the repo's log-domain rank twin.

    Layout matches ASCAD_test_models.rank() with simulated_key=1: hypothesis i is scored as
    Sbox[p ^ k_true ^ i], so the true key is index 0.
    """
    hyp = np.arange(256, dtype=np.uint8)[None, :]
    ranks, flats = [], []
    for b in range(16):
        idx = AES_Sbox[np.bitwise_xor(
            (np.asarray(plaintext[:, b], dtype=np.uint8) ^ np.asarray(key[:, b], dtype=np.uint8))[:, None],
            hyp,
        )]
        p = np.asarray(sbox_probs[b], dtype=np.float64)
        n = p.shape[0]
        vals = p[np.arange(n)[:, None], idx]
        row_min = np.array([np.min(r[r > 0]) if np.any(r > 0) else 1.0 for r in p])[:, None]
        log_p = np.log(np.where(vals > 0, vals, row_min ** 2))
        out = ranks_from_log_scores(log_p, 0, rank_step, tie_break=tie_break)
        final = int(out[-1][1])
        cum = np.cumsum(log_p, axis=0)[-1]
        true_score = cum[0]
        ranks.append({"final_rank": final, "min_rank": int(out[:, 1].min())})
        flats.append({
            "spread": float(cum.max() - cum.min()),
            "ties_at_top": int(np.sum(cum == cum.max())),
            "rank_of_true_worst_tie": int(np.sum(cum > true_score) + np.sum(cum == true_score) - 1),
            "z_true": float((true_score - cum.mean()) / cum.std()) if cum.std() > 0 else 0.0,
        })
    fin = [r["final_rank"] for r in ranks]
    fl = [f["spread"] for f in flats]
    return {
        "per_byte": {str(b): {**ranks[b], **flats[b]} for b in range(16)},
        "n_rank0_final": int(sum(1 for r in fin if r == 0)),
        "median_final_rank": int(np.median(fin)),
        "worst_final_rank": int(max(fin)),
        "median_spread_nats": float(np.median(fl)),
        "flat_bytes": int(sum(1 for s in fl if s < 1e-9)),
    }


def infer(model, X, device, batch):
    out = {}
    with torch.no_grad():
        for i in range(0, X.shape[0], batch):
            # (B, L) -> (B, L, 1), the same reshape training used (reshape_inputs_for_network)
            xb = torch.from_numpy(X[i:i + batch]).unsqueeze(-1).to(device)
            for k, v in model(xb).items():
                out.setdefault(k, []).append(torch.softmax(v, dim=-1).cpu().numpy())
    return {k: np.concatenate(v, axis=0) for k, v in out.items()}


# ---------------------------------------------------------------- checkpoints


def d_model_of(sd):
    return int(sd["enc_embedding.value_embedding.token_conv.weight"].shape[0])


def infer_timesnet_config(sd):
    """Recover a TimesNetConfig dict from state-dict shapes (checkpoints with no sidecar)."""
    d = d_model_of(sd)
    cfg = {
        "d_model": d,
        "d_ff": int(sd["model.0.conv.0.kernels.1.weight"].shape[0]),
        "e_layers": 1 + max(int(m.group(1)) for m in
                            (re.match(r"model\.(\d+)\.", k) for k in sd) if m),
        "embedding_kernel_size": int(sd["enc_embedding.value_embedding.token_conv.weight"].shape[2]),
        "simple_heads": True,
        "without_permind": not any(k.startswith("permind_") for k in sd),
        "input_stride": 1,
        "pool_segments": 0,
    }
    pe_len = int(sd["enc_embedding.position_embedding.pe"].shape[1])
    if pe_len <= 5000:
        # With stride>1 the effective seq_len is <=3000 and pe is clamped to 5000, so the table
        # length no longer identifies seq_len. Refuse to guess; the sidecar path covers those.
        raise RuntimeError(f"cannot infer seq_len from positional table of length {pe_len}")
    cfg["seq_len"] = pe_len
    if "segment_proj.weight" in sd:
        cfg["pool_segments"] = int(sd["segment_proj.weight"].shape[1]) // d
    if "trunk.0.weight" in sd:
        cfg["head_hidden_dim"] = int(sd["trunk.0.weight"].shape[0])
        cfg["head_hidden_dim_2"] = int(sd["trunk.3.weight"].shape[0])
    return cfg


def network_type_for(path, sd):
    if os.path.exists(path + ".config.json"):
        return json.load(open(path + ".config.json", encoding="utf-8"))["network_type"]
    return ("timesnet_ascad_head_multi" if "trunk.0.weight" in sd else "timesnet_multi")


def weights_sha256(sd):
    """Hash of the stored tensors, so 'best' and 'last' checkpoints can be told apart."""
    h = hashlib.sha256()
    for k in sorted(sd):
        v = sd[k]
        h.update(k.encode())
        if torch.is_tensor(v):
            h.update(str(tuple(v.shape)).encode())
            h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def load_checkpoint(path, device):
    sidecar_path = path + ".config.json"
    if os.path.exists(sidecar_path):
        sidecar = json.load(open(sidecar_path, encoding="utf-8"))
        cfg = dict(sidecar["timesnet_config"])
        network_type = sidecar["network_type"]
        source = sidecar_path
    else:
        sd = torch.load(path, map_location="cpu", weights_only=True)
        cfg = infer_timesnet_config(sd)
        network_type = network_type_for(path, sd)
        source = "inferred from state-dict shapes (no sidecar)"
    sd = torch.load(path, map_location=device, weights_only=True)
    model = build_timesnet_for_network(network_type, cfg.get("input_dim", cfg["seq_len"]), cfg)
    model.load_state_dict(sd)
    model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    return model, {"network_type": network_type, "timesnet_config": cfg,
                   "config_source": source, "params": int(n_params),
                   "weights_sha256": weights_sha256(sd)}


# ---------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--num-traces", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--rank-step", type=int, default=10)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0, help="pilot: first N traces only")
    ap.add_argument("--only", default="", help="substring filter on checkpoint names")
    ap.add_argument("--no-ranks", action="store_true", help="skip the secondary rank decode")
    ap.add_argument("--out-json", default=DEFAULT_OUT)
    args = ap.parse_args()

    t_start = time.time()
    n_traces = args.limit or args.num_traces
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    paths = []
    for pat in CKPT_PATTERNS:
        paths.extend(sorted(glob.glob(os.path.join(MODELS_DIR, pat))))
    if args.only:
        paths = [p for p in paths if args.only in os.path.basename(p)]
    if not paths:
        raise SystemExit(f"no checkpoints matched {CKPT_PATTERNS}")

    print(f"device={device} traces={n_traces} checkpoints={len(paths)}", flush=True)
    X, lab = load_attack_split(n_traces)
    print(f"loaded {X.shape} attack traces", flush=True)

    results = {
        "meta": {
            "step": "T-030 S1",
            "h5": os.path.relpath(H5, repo_root),
            "model_dir": os.path.relpath(MODELS_DIR, repo_root),
            "n_traces": int(n_traces),
            "rank_step": args.rank_step,
            "tie_break": "worst",
            "device": str(device),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "python": sys.version.split()[0],
            "git": os.popen("git -C %s rev-parse --short HEAD" % repo_root).read().strip(),
            "gain_definition": "mean_n log p_true + log K; zeros floored as rank() does",
        },
        "label_algebra": label_algebra_check(lab, min(2000, n_traces)),
        "checkpoints": {},
    }

    # --- reference: the working MultiResNet, recomputed here from its cached predictions ------
    ref = {}
    if os.path.exists(BASELINE_CACHE):
        z = np.load(BASELINE_CACHE)
        n_ref = min(n_traces, z["alpha_output"].shape[0])
        alpha, beta = lab["alpha"][:n_ref], lab["beta"][:n_ref]
        ref["alpha_output"] = head_metrics(z["alpha_output"][:n_ref], alpha)
        ref["beta_output"] = head_metrics(z["beta_output"][:n_ref], beta)
        ref["C_head_mean"] = [
            head_metrics(z[f"sbox_{i}_output"][:n_ref], lab["sbox_masked"][:n_ref, i])
            for i in range(16)
        ]
        ref["C_head_cross_byte_indexed"] = [
            head_metrics(z[f"sbox_{i}_output"][:n_ref], lab["sbox_masked_with_perm"][:n_ref, i])
            for i in range(16)
        ]
        ref["permind_head_mean"] = [
            head_metrics(z[f"permind_{i}_output"][:n_ref], lab["perm_index"][:n_ref, i])
            for i in range(16)
        ]
        results["baseline_reference"] = {
            "source": os.path.relpath(BASELINE_CACHE, repo_root),
            "head_layout": "multilabel=1: 34 heads, C heads on sbox_masked (slot-indexed) + 16 perm heads",
            "alpha": ref["alpha_output"],
            "beta": ref["beta_output"],
            "C_mean_top1": float(np.mean([h["top1_acc"] for h in ref["C_head_mean"]])),
            "C_mean_gain": float(np.mean([h["gain_over_uniform_nats"] for h in ref["C_head_mean"]])),
            "C_per_head_top1": [h["top1_acc"] for h in ref["C_head_mean"]],
            "C_per_head_gain": [h["gain_over_uniform_nats"] for h in ref["C_head_mean"]],
            "C_cross_byte_indexed_mean_top1": float(
                np.mean([h["top1_acc"] for h in ref["C_head_cross_byte_indexed"]])),
            "C_cross_byte_indexed_mean_gain": float(
                np.mean([h["gain_over_uniform_nats"] for h in ref["C_head_cross_byte_indexed"]])),
            "permind_mean_top1": float(np.mean([h["top1_acc"] for h in ref["permind_head_mean"]])),
            "permind_mean_gain": float(np.mean([h["gain_over_uniform_nats"] for h in ref["permind_head_mean"]])),
        }
        print("baseline reference: alpha top1 "
              f"{results['baseline_reference']['alpha']['top1_acc']:.4f} "
              f"beta {results['baseline_reference']['beta']['top1_acc']:.4f}", flush=True)
    else:
        results["baseline_reference"] = {"error": f"missing {BASELINE_CACHE}"}
        print(f"WARNING: baseline cache missing: {BASELINE_CACHE}", flush=True)

    # --- the old TimesNet checkpoints ---------------------------------------------------------
    for path in paths:
        name = os.path.basename(path)
        st = os.stat(path)
        t0 = time.time()
        model, info = load_checkpoint(path, device)
        preds = infer(model, X, device, args.batch)
        # score each head against the field it was trained on
        n_run = preds["alpha_output"].shape[0]
        heads = {
            "alpha_output": head_metrics(preds["alpha_output"], lab["alpha"][:n_run]),
            "beta_output": head_metrics(preds["beta_output"], lab["beta"][:n_run]),
        }
        heads["C_heads_own_target_sbox_masked_with_perm"] = [
            head_metrics(preds[f"sbox_{i}_output"], lab["sbox_masked_with_perm"][:n_run, i])
            for i in range(16)
        ]
        heads["C_heads_cross_slot_indexed_sbox_masked"] = [
            head_metrics(preds[f"sbox_{i}_output"], lab["sbox_masked"][:n_run, i])
            for i in range(16)
        ]
        # Label-column guard: T-024 published wrong alpha/beta figures from a mis-mapped
        # column, so score each of the two mask heads against the other label too.
        heads["alpha_head_scored_against_beta_mask"] = head_metrics(
            preds["alpha_output"], lab["beta"][:n_run])
        heads["beta_head_scored_against_alpha_mask"] = head_metrics(
            preds["beta_output"], lab["alpha"][:n_run])
        c_top1 = [h["top1_acc"] for h in heads["C_heads_own_target_sbox_masked_with_perm"]]
        c_gain = [h["gain_over_uniform_nats"]
                  for h in heads["C_heads_own_target_sbox_masked_with_perm"]]
        entry = {
            "file": name,
            "bytes": st.st_size,
            "mtime_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(st.st_mtime)),
            **info,
            "has_permind_heads": "permind_0_output" in preds,
            "alpha_top1": heads["alpha_output"]["top1_acc"],
            "alpha_gain_nats": heads["alpha_output"]["gain_over_uniform_nats"],
            "beta_top1": heads["beta_output"]["top1_acc"],
            "beta_gain_nats": heads["beta_output"]["gain_over_uniform_nats"],
            "C_mean_top1": float(np.mean(c_top1)),
            "C_mean_gain_nats": float(np.mean(c_gain)),
            "C_cross_slot_mean_top1": float(np.mean([
                h["top1_acc"] for h in heads["C_heads_cross_slot_indexed_sbox_masked"]])),
            "C_cross_slot_mean_gain_nats": float(np.mean([
                h["gain_over_uniform_nats"] for h in heads["C_heads_cross_slot_indexed_sbox_masked"]])),
            "heads": heads,
            "inference_wall_s": round(time.time() - t0, 1),
        }

        if not args.no_ranks and n_run >= 256:
            lst = [preds["alpha_output"], preds["beta_output"]]
            lst += [preds[f"sbox_{i}_output"] for i in range(16)]
            sbox = multilabel_without_permind_predict(lst)
            pt, ky = lab["plaintext"][:n_run], lab["key"][:n_run]
            entry["ranks_c_alpha_beta_decode"] = byte_key_ranks(
                sbox, pt, ky, args.rank_step)

            rng = np.random.default_rng(0)
            perm = rng.permutation(n_run)
            shuf = [lst[i][perm] for i in range(len(lst))]
            entry["NULL_shuffled_pairs"] = byte_key_ranks(
                multilabel_without_permind_predict(shuf), pt, ky, args.rank_step)

            X_const = np.tile(X.mean(axis=0), (min(500, n_run), 1)).astype(np.float32)
            cp = infer(model, X_const, device, args.batch)
            cl = [cp["alpha_output"], cp["beta_output"]]
            cl += [cp[f"sbox_{i}_output"] for i in range(16)]
            entry["NULL_constant_trace"] = byte_key_ranks(
                multilabel_without_permind_predict(cl), pt[:X_const.shape[0]],
                ky[:X_const.shape[0]], args.rank_step)

        results["checkpoints"][name] = entry
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        rk = entry.get("ranks_c_alpha_beta_decode", {})
        print(f"{name:52s} alpha_top1={entry['alpha_top1'] * 100:6.2f}% "
              f"gain={entry['alpha_gain_nats']:+.2f}  beta={entry['beta_top1'] * 100:5.2f}%  "
              f"C_mean={entry['C_mean_top1'] * 100:5.2f}% gain={entry['C_mean_gain_nats']:+.2f} "
              f"rank0={rk.get('n_rank0_final', '-')}/16 med={rk.get('median_final_rank', '-')}"
              f"  ({entry['inference_wall_s']}s)", flush=True)

    alpha_vals = {k: v["alpha_top1"] for k, v in results["checkpoints"].items()}
    results["gate"] = {
        "rule": "alpha top-1 > 50% => trunk learned alpha (failure is downstream); at chance => trunk never got going",
        "chance_level": 1.0 / 256.0,
        "max_alpha_top1": float(max(alpha_vals.values())),
        "n_checkpoints_alpha_above_50pct": int(sum(1 for v in alpha_vals.values() if v > 0.50)),
        "n_checkpoints_alpha_above_chance_2pct": int(sum(1 for v in alpha_vals.values() if v > 0.02)),
        "verdict_if_any_above": "the old trunk learned alpha -> its failure is downstream (C heads or decoding)",
        "verdict_if_none_above": "no old trunk reached 50% alpha -> the trunk never got going",
    }
    print(f"\nGATE: max alpha top-1 = {results['gate']['max_alpha_top1'] * 100:.2f}% over "
          f"{len(alpha_vals)} checkpoints; "
          f"{results['gate']['n_checkpoints_alpha_above_50pct']} above 50%", flush=True)

    results["meta"]["total_wall_s"] = round(time.time() - t_start, 1)
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {args.out_json}  (total {results['meta']['total_wall_s']}s)", flush=True)


if __name__ == "__main__":
    main()
