#!/usr/bin/env python3
"""Evaluation script for MultiResNet on ASCADv2 (STM32F303).

Evaluates:
- Individual head accuracies (alpha, beta, masked S-boxes, permutation indices)
- AES Key Rank curves across all 16 key bytes via multilabel_predict
"""

import argparse
import json
import os
import sys

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import h5py
import numpy as np
import torch

from ASCAD_generate import AES_Sbox
from ASCAD_test_models import (
    full_ranks,
    multilabel_predict,
    multilabel_without_permind_predict,
    load_ascad,
)
from MultiResNet.model import MultiResNet, SingleHeadResNet, CompactMultiResNet


def main():
    parser = argparse.ArgumentParser(description="Evaluate MultiResNet on ASCADv2")
    parser.add_argument("--model-file", required=True, help="Path to .pth checkpoint")
    parser.add_argument("--h5", default="STM32_AES_v2/ASCAD_data/ASCAD_databases/ascadv2-extracted.h5")
    parser.add_argument("--num-traces", type=int, default=3000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load sidecar metadata
    sidecar_path = args.model_file + ".config.json"
    sidecar = {}
    if os.path.exists(sidecar_path):
        with open(sidecar_path, "r", encoding="utf-8") as f:
            sidecar = json.load(f)

    single_head = (sidecar.get("multilabel", 1) == 0)
    without_permind = (sidecar.get("multilabel", 1) == 2)
    sub_avg = sidecar.get("sub_avg", True)

    print(f"Loading {args.num_traces} attack traces from {args.h5}...")
    (_, _), (X_attack, _), (_, Metadata_attack) = load_ascad(args.h5, load_metadata=True)
    X_attack = X_attack[:args.num_traces].astype(np.float32)

    mean_path = args.model_file + ".mean_wf.npy"
    if sub_avg and os.path.exists(mean_path):
        print(f"Applying average waveform subtraction from {mean_path}...")
        mean_wf = np.load(mean_path)
        X_attack = X_attack - mean_wf

    input_dim = X_attack.shape[1]
    if single_head:
        model = SingleHeadResNet(input_dim=input_dim, depth=19, num_classes=256)
    elif sidecar.get("compact", False) or "compact" in sidecar.get("network_type", ""):
        bot_dim = sidecar.get("bottleneck_dim", 512)
        h_dim = sidecar.get("head_hidden_dim", 256)
        model = CompactMultiResNet(
            input_dim=input_dim, depth=19, without_permind=without_permind,
            bottleneck_dim=bot_dim, head_hidden_dim=h_dim,
        )
    else:
        model = MultiResNet(input_dim=input_dim, depth=19, without_permind=without_permind)

    model.load_state_dict(torch.load(args.model_file, map_location=device))
    model.to(device)
    model.eval()

    print("\nRunning inference...")
    batch_size = 64
    all_preds = {}

    with torch.no_grad():
        for i in range(0, args.num_traces, batch_size):
            xb = torch.from_numpy(X_attack[i : i + batch_size]).to(device)
            out = model(xb)
            if single_head:
                if "sbox" not in all_preds:
                    all_preds["sbox"] = []
                all_preds["sbox"].append(torch.softmax(out, dim=-1).cpu().numpy())
            else:
                for k, v in out.items():
                    if k not in all_preds:
                        all_preds[k] = []
                    all_preds[k].append(torch.softmax(v, dim=-1).cpu().numpy())

    for k in all_preds:
        all_preds[k] = np.concatenate(all_preds[k], axis=0)

    if single_head:
        target_byte = sidecar.get("target_byte", 0)
        print(f"\nSingle-Head evaluation for Byte {target_byte}:")
        ranks = full_ranks(all_preds["sbox"], X_attack, Metadata_attack, 0, args.num_traces, 10, target_byte, simulated_key=1)
        print(f"Final Key Rank: {ranks[-1][1]}, Min Key Rank: {min(r[1] for r in ranks)}")
        return

    # Multi-label recombination
    # Format predictions list matching ASCAD_test_models expectations:
    # 34 elements: [alpha(N,256), beta(N,256), sbox_0..15(N,256), permind_0..15(N,16)]
    print("\nRecombining multi-task heads...")
    pred_list = [all_preds["alpha_output"], all_preds["beta_output"]]
    for i in range(16):
        pred_list.append(all_preds[f"sbox_{i}_output"])
    if not without_permind:
        for i in range(16):
            pred_list.append(all_preds[f"permind_{i}_output"])

    if not without_permind:
        predictions_sbox = multilabel_predict(pred_list)
    else:
        predictions_sbox = multilabel_without_permind_predict(pred_list)

    print("\n=== MultiResNet Key Rank Across All 16 Bytes ===")
    results = {}
    for b in range(16):
        ranks_i = full_ranks(predictions_sbox[b], X_attack, Metadata_attack, 0, args.num_traces, 10, b, simulated_key=1)
        final_rk = int(ranks_i[-1][1])
        min_rk = int(min(r[1] for r in ranks_i))
        results[b] = {"final_rank": final_rk, "min_rank": min_rk}
        print(f"Byte {b:2d}: Final Rank = {final_rk:3d}, Min Rank = {min_rk:3d}")

    out_json = os.path.splitext(args.model_file)[0] + "_eval.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved evaluation results to {out_json}")


if __name__ == "__main__":
    main()
