#!/usr/bin/env python3
"""Train SingleHeadResNet on ASCADv2 (Arm 2A: Shuffle ON, Black-Box).

In Arm 2A:
- The network directly attacks an unmasked S-box intermediate: Y = AES_Sbox[p_0 ^ k_0].
- The trace X is the full 15,000-sample raw trace with temporal shuffling ENABLED.
- Zero mask labels: alpha, beta, and pi are completely omitted from loss and architecture.
- Objective: Determine whether a 1D Conv ResNet can break ASCADv2 end-to-end under both
  shuffling and affine masking without any auxiliary supervision.
"""

import argparse
import json
import os
import random
import sys
import time
from typing import Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, REPO_ROOT)

from MultiResNet.model import SingleHeadResNet

AES_Sbox = np.array([
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16
], dtype=np.uint8)
if int(np.unique(AES_Sbox).size) != 256:
    raise RuntimeError("AES_Sbox is not a permutation")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_dataset_2a(h5_path: str, train_len: int, target_byte: int = 0, sub_avg: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    print(f"Loading ASCADv2 dataset from {h5_path} (train_len={train_len}, target_byte={target_byte})...")
    with h5py.File(h5_path, "r") as f:
        X = np.asarray(f["Profiling_traces/traces"][:train_len], dtype=np.float32)
        meta = f["Profiling_traces/metadata"][:train_len]

    mean_wf = None
    if sub_avg:
        print("Computing mean trace across profiling set...")
        mean_wf = np.mean(X, axis=0, keepdims=True)
        X -= mean_wf

    pt = np.asarray(meta["plaintext"][:, target_byte], dtype=np.uint8)
    key = np.asarray(meta["key"][:, target_byte], dtype=np.uint8)
    Y = AES_Sbox[pt ^ key].astype(np.int64)

    return X, Y, mean_wf


def main():
    parser = argparse.ArgumentParser(description="Train SingleHeadResNet on ASCADv2 (Arm 2A)")
    parser.add_argument("--h5", default="STM32_AES_v2/ASCAD_data/ASCAD_databases/ascadv2-extracted.h5")
    parser.add_argument("--out-model", default="STM32_AES_v2/ASCAD_data/ASCAD_trained_models/single_head_2a_t300k_b0_seed1_sboxfix.pth")
    parser.add_argument("--train-len", type=int, default=300000)
    parser.add_argument("--target-byte", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--sub-avg", action="store_true", default=True)
    parser.add_argument("--val-split", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--save-last", action="store_true", default=True)
    args = parser.parse_args()

    if args.seed is not None:
        set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    X, Y, mean_wf = load_dataset_2a(args.h5, args.train_len, target_byte=args.target_byte, sub_avg=args.sub_avg)
    input_dim = X.shape[1]

    model = SingleHeadResNet(input_dim=input_dim, depth=19, num_classes=256).to(device)
    nparams = sum(p.numel() for p in model.parameters())
    print(f"SingleHeadResNet instantiated: {nparams:,} parameters ({nparams/1e6:.2f}M)")

    # Data split
    n = X.shape[0]
    n_val = int(n * args.val_split)
    n_tr = n - n_val

    X_tr, X_va = X[:n_tr], X[n_tr:]
    Y_tr, Y_va = torch.from_numpy(Y[:n_tr]), torch.from_numpy(Y[n_tr:])

    train_ds = TensorDataset(torch.from_numpy(X_tr), Y_tr)
    val_ds = TensorDataset(torch.from_numpy(X_va), Y_va)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    optimizer = torch.optim.RAdam(model.parameters(), lr=args.lr)
    scaler = GradScaler("cuda")
    criterion = nn.CrossEntropyLoss()

    out_dir = os.path.dirname(args.out_model)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if mean_wf is not None:
        mean_path = args.out_model + ".mean_wf.npy"
        np.save(mean_path, mean_wf.squeeze(0))
        print(f"Saved mean waveform to {mean_path}")

    history = {
        "config": {
            "network_type": "single_head_resnet_2a",
            "train_len": args.train_len,
            "target_byte": args.target_byte,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "sub_avg": args.sub_avg,
            "val_split": args.val_split,
            "seed": args.seed,
            "chance_ce": float(np.log(256)),
            "chance_top1": float(1.0 / 256.0),
        },
        "epochs": [],
        "epoch_seconds": [],
        "train_loss": [],
        "val_loss": [],
        "val_top1": [],
        "val_gain_nats": [],
    }
    history_path = args.out_model + ".history.json"

    best_loss = float("inf")
    chance_ce = np.log(256)
    print(f"\n--- Starting Arm 2A Training (Shuffle ON, Target Byte {args.target_byte}) ---")
    print(f"Chance cross-entropy = {chance_ce:.4f} nats, Chance top-1 = {100.0/256.0:.2f}%\n")

    t_start = time.time()
    for epoch in range(args.epochs):
        model.train()
        total_loss, seen = 0.0, 0
        t0 = time.time()

        for batch in train_loader:
            xb = batch[0].to(device)
            yb = batch[1].to(device)
            optimizer.zero_grad(set_to_none=True)

            with autocast("cuda"):
                logits = model(xb)
                loss = criterion(logits, yb)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += float(loss.detach()) * xb.shape[0]
            seen += xb.shape[0]

        train_loss = total_loss / max(seen, 1)

        # Validation
        model.eval()
        v_total, v_seen, v_correct = 0.0, 0, 0

        with torch.no_grad():
            for batch in val_loader:
                xb = batch[0].to(device)
                yb = batch[1].to(device)
                with autocast("cuda"):
                    logits = model(xb)
                    v_loss = criterion(logits, yb)

                v_total += float(v_loss.detach()) * xb.shape[0]
                v_correct += int((logits.argmax(dim=1) == yb).sum().item())
                v_seen += xb.shape[0]

        val_loss = v_total / max(v_seen, 1)
        val_top1 = v_correct / max(v_seen, 1)
        val_gain = float(chance_ce - val_loss)
        el = time.time() - t0

        history["epochs"].append(epoch + 1)
        history["epoch_seconds"].append(round(el, 1))
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_top1"].append(val_top1)
        history["val_gain_nats"].append(val_gain)

        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        print(f"Epoch {epoch+1:2d}/{args.epochs:2d}  train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  val_top1={val_top1*100:.3f}%  "
              f"info_gain={val_gain:+.4f} nats  ({el:.1f}s)  [chance={100/256:.2f}%]")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), args.out_model)
            print(f"  -> Saved new best model to {args.out_model} (val_loss={val_loss:.4f})")

    if args.save_last:
        base, ext = os.path.splitext(args.out_model)
        last_path = f"{base}_last{ext}"
        torch.save(model.state_dict(), last_path)
        print(f"Saved last-epoch checkpoint to {last_path}")

    total_time = time.time() - t_start
    print(f"\nTraining completed in {total_time:.1f}s ({total_time/60:.1f} min). Best val loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()
