#!/usr/bin/env python3
"""Training script for MultiResNet on ASCADv2 (STM32F303).

Supports:
- Multi-task supervision with permutation indices (34 heads, ~137M params)
- Multi-task supervision without permutation indices (18 heads, ~78M params)
- Single-head direct S-box prediction (~75M params)
- Mean-waveform subtraction (--sub-avg)
"""

import argparse
import json
import math
import os
import sys
import time

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import autocast, GradScaler

from ASCAD_generate import AES_Sbox
from MultiResNet.model import MultiResNet, SingleHeadResNet, CompactMultiResNet
from MultiResNet.decoder_loss import GF_MUL, decoder_nll, head_probs


MULTILABEL_KEYS_ALL = [
    "alpha_output", "beta_output",
    *[f"sbox_{i}_output" for i in range(16)],
    *[f"permind_{i}_output" for i in range(16)],
]
MULTILABEL_KEYS_NO_PERM = [
    "alpha_output", "beta_output",
    *[f"sbox_{i}_output" for i in range(16)],
]


DROP_GROUPS = {"alpha": ("alpha_output",), "beta": ("beta_output",),
               "C": tuple(f"sbox_{i}_output" for i in range(16)),
               "pi": tuple(f"permind_{i}_output" for i in range(16))}


def weight_of(key, c_weight):
    """Loss weight for one head: the 16 masked S-box (C) terms scale, everything else stays at 1."""
    return c_weight if key.startswith("sbox_") else 1.0


def weighted_multilabel_loss(outputs, targets, criterion, keys, c_weight):
    """Sum of per-head cross-entropies with the C terms scaled by `c_weight`.

    c_weight=1.0 is exactly the sum MultiResNet/train.py had before T-031 N4; c_weight=0.0 is N2's
    C-deletion objective. Kept as a free function so the arithmetic is testable without a GPU run.
    """
    loss = 0.0
    for k in keys:
        loss = loss + weight_of(k, c_weight) * criterion(outputs[k], targets[k])
    return loss


def load_dataset(h5_path, train_len, target_byte=0, single_head=False, without_permind=False, sub_avg=True,
                 with_sbox_out=False):
    print(f"Loading {train_len} traces from {h5_path}...")
    with h5py.File(h5_path, "r") as f:
        prof_grp = f["Profiling_traces"]
        X_prof = prof_grp["traces"][:train_len].astype(np.float32)
        meta_prof = prof_grp["metadata"][:train_len]
        labels_prof = prof_grp["labels"][:train_len]

    mean_wf = None
    if sub_avg:
        print("Subtracting profiling mean waveform...")
        mean_wf = np.mean(X_prof, axis=0, keepdims=True)
        X_prof -= mean_wf  # in place: bit-identical, avoids a second full copy (300k = 18 GB)

    if single_head:
        pt = np.asarray(meta_prof["plaintext"][:, target_byte], dtype=np.uint8)
        k = np.asarray(meta_prof["key"][:, target_byte], dtype=np.uint8)
        Y_prof = AES_Sbox[pt ^ k].astype(np.int64)
        return X_prof, Y_prof, mean_wf

    # Multi-label packaging from structured array
    targets = {
        "alpha_output": torch.as_tensor(labels_prof["alpha_mask"], dtype=torch.long).reshape(-1),
        "beta_output": torch.as_tensor(labels_prof["beta_mask"], dtype=torch.long).reshape(-1),
    }
    sbox_field = "sbox_masked_with_perm" if without_permind else "sbox_masked"
    for i in range(16):
        targets[f"sbox_{i}_output"] = torch.as_tensor(labels_prof[sbox_field][:, i], dtype=torch.long)
    if not without_permind:
        for i in range(16):
            targets[f"permind_{i}_output"] = torch.as_tensor(labels_prof["perm_index"][:, i], dtype=torch.long)
    if with_sbox_out:  # T-039 E6: unmasked S = Sbox(p xor k), the only label non-worst-case has
        S = AES_Sbox[np.asarray(meta_prof["plaintext"]) ^ np.asarray(meta_prof["key"])]
        targets["S"] = torch.as_tensor(S.astype(np.int64))

    return X_prof, targets, mean_wf


def main():
    parser = argparse.ArgumentParser(description="Train MultiResNet on ASCADv2")
    parser.add_argument("--h5", default="STM32_AES_v2/ASCAD_data/ASCAD_databases/ascadv2-extracted.h5")
    parser.add_argument("--out-model", default="STM32_AES_v2/ASCAD_data/ASCAD_trained_models/multiresnet_v2.pth")
    parser.add_argument("--train-len", type=int, default=20000)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--single-head", action="store_true", help="Train single unmasked S-box head instead of multi-task")
    parser.add_argument("--target-byte", type=int, default=0)
    parser.add_argument("--without-permind", action="store_true", help="Omit permutation index heads (18 heads instead of 34)")
    parser.add_argument("--sub-avg", action="store_true", default=True, help="Subtract mean waveform across profiling traces")
    parser.add_argument("--val-split", type=float, default=0.05)
    parser.add_argument("--no-c-heads", action="store_true",
                        help="T-031 T31-N2 interference run: drop the 16 masked S-box (C) heads from "
                             "the LOSS while keeping the 34-head architecture, so the shared trunk "
                             "receives no C gradient. The C head parameters stay built and untrained.")
    parser.add_argument("--c-loss-weight", type=float, default=1.0,
                        help="T-031 T31-N4: multiplier applied ONLY to the 16 sbox_* cross-entropy "
                             "terms; alpha/beta/permind keep weight 1. Default 1.0 reproduces every "
                             "previous run exactly. 0.0 is equivalent to --no-c-heads. Accepted range "
                             "is [0, 1].")
    parser.add_argument("--save-last", action="store_true",
                        help="Also write <out-model>_last.pth after the final epoch. The default "
                             "checkpoint is the best-val one, which for a 15-epoch budget is an "
                             "arbitrary epoch; reporting both removes that ambiguity.")
    parser.add_argument("--seed", type=int, default=None,
                        help="T-031 T31-N4b: seed torch/numpy/python RNGs before building the model "
                             "and iterating the loaders, so two runs differ only by this number. The "
                             "trainer set no seed before this flag, which is why T31-N2 needed a whole "
                             "control run to estimate run-to-run spread. Note AMP/cuDNN atomics can "
                             "still make two same-seed runs differ slightly; that gap is measured, not "
                             "assumed.")
    parser.add_argument("--compact", action="store_true",
                        help="T-033: train CompactMultiResNet (~15M params) with shared bottleneck and slim heads")
    parser.add_argument("--bottleneck-dim", type=int, default=512,
                        help="T-033: shared linear bottleneck dimension for CompactMultiResNet (default 512; set 0 for direct unshared slim heads)")
    parser.add_argument("--head-hidden-dim", type=int, default=256,
                        help="T-033: hidden dimension for slim BranchHead (default 256)")
    parser.add_argument("--drop-labels", default="",
                        help="T-039 E6: comma list from {alpha,beta,C,pi}. Those heads leave the "
                             "cross-entropy loss (still built) and learn only through --decoder-loss-weight.")
    parser.add_argument("--decoder-loss-weight", type=float, default=0.0,
                        help="T-039 E6: weight of the differentiable affine decoder NLL of the true "
                             "S = Sbox(p xor k), summed over 16 bytes (MultiResNet/decoder_loss.py). "
                             "0 (default) reproduces every previous run.")
    parser.add_argument("--heads-fp32", action="store_true",
                        help="T-039 E6b: run the output heads in float32 under AMP (CompactMultiResNet). "
                             "Unlabelled heads trained only via the decoder overflowed fp16 at 300k.")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.seed is not None:
        import random
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        print(f"Seed {args.seed} applied to python/numpy/torch before model init and loader order")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    X, Y, mean_wf = load_dataset(
        args.h5,
        args.train_len,
        target_byte=args.target_byte,
        single_head=args.single_head,
        without_permind=args.without_permind,
        sub_avg=args.sub_avg,
        with_sbox_out=args.decoder_loss_weight > 0,
    )

    input_dim = X.shape[1]
    if args.single_head:
        model = SingleHeadResNet(input_dim=input_dim, depth=19, num_classes=256).to(device)
    elif args.compact:
        model = CompactMultiResNet(
            input_dim=input_dim,
            depth=19,
            without_permind=args.without_permind,
            bottleneck_dim=args.bottleneck_dim,
            head_hidden_dim=args.head_hidden_dim,
        ).to(device)
    else:
        model = MultiResNet(input_dim=input_dim, depth=19, without_permind=args.without_permind).to(device)

    if args.heads_fp32:
        if not args.compact:
            raise SystemExit("--heads-fp32 is implemented for CompactMultiResNet only")
        model.heads_fp32 = True
        print("Output heads run in float32 under AMP (--heads-fp32)")

    nparams = sum(p.numel() for p in model.parameters())
    model_name = "CompactMultiResNet" if args.compact else ("SingleHeadResNet" if args.single_head else "MultiResNet")
    print(f"{model_name} instantiated: {nparams:,} parameters ({nparams/1e6:.2f}M)")

    # Data split
    n = X.shape[0]
    n_val = int(n * args.val_split)
    n_tr = n - n_val

    X_tr, X_va = X[:n_tr], X[n_tr:]
    if args.single_head:
        Y_tr, Y_va = torch.from_numpy(Y[:n_tr]), torch.from_numpy(Y[n_tr:])
        train_ds = TensorDataset(torch.from_numpy(X_tr), Y_tr)
        val_ds = TensorDataset(torch.from_numpy(X_va), Y_va)
    else:
        Y_tr = {k: v[:n_tr] for k, v in Y.items()}
        Y_va = {k: v[n_tr:] for k, v in Y.items()}
        train_ds = TensorDataset(torch.from_numpy(X_tr), *[Y_tr[k] for k in Y_tr])
        val_ds = TensorDataset(torch.from_numpy(X_va), *[Y_va[k] for k in Y_va])

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

    if not (0.0 <= args.c_loss_weight <= 1.0):
        raise SystemExit(f"--c-loss-weight must be in [0, 1], got {args.c_loss_weight}")
    c_weight = 0.0 if args.no_c_heads else args.c_loss_weight
    if args.no_c_heads and args.c_loss_weight != 1.0:
        print("--no-c-heads overrides --c-loss-weight: C terms are dropped entirely (weight 0)")

    full_keys = list(Y_tr.keys()) if not args.single_head else []
    all_keys = [k for k in full_keys if k != "S"]  # heads only; "S" is the decoder target
    # A head contributes to the loss iff its weight is non-zero. The 16 sbox heads are always BUILT;
    # only their contribution to the objective changes, so trunk capacity and parameter count stay
    # identical across every arm of the sweep.
    loss_keys = [k for k in all_keys if weight_of(k, c_weight) > 0.0] if not args.single_head else []
    drop = [g for g in args.drop_labels.split(",") if g]
    for g in drop:
        if g not in DROP_GROUPS:
            raise SystemExit(f"--drop-labels: unknown group {g!r}; choose from {sorted(DROP_GROUPS)}")
    dropped_keys = {k for g in drop for k in DROP_GROUPS[g]}
    loss_keys = [k for k in loss_keys if k not in dropped_keys]
    dec_w = args.decoder_loss_weight
    if dec_w > 0 and (args.single_head or args.without_permind):
        raise SystemExit("--decoder-loss-weight needs the 34-head model")
    if drop or dec_w > 0:
        print(f"E6: dropped label groups {drop or 'none'}; CE over {len(loss_keys)}/{len(all_keys)} "
              f"heads; decoder NLL weight {dec_w:g}")
    gf_mul = torch.as_tensor(GF_MUL, device=device)

    def decoder_term(outputs, S):
        with autocast("cuda", enabled=False):
            return decoder_nll(*head_probs(outputs), S, gf_mul).mean()
    if c_weight != 1.0:
        print(f"C loss weight = {c_weight:g}: loss over {len(loss_keys)}/{len(all_keys)} heads "
              f"(sbox terms scaled by {c_weight:g}); the 34-head architecture is unchanged")
    elif args.no_c_heads:
        print(f"--no-c-heads: loss over {len(loss_keys)}/{len(all_keys)} heads "
              "(alpha, beta, permind_0..15); the 16 sbox heads stay built and untrained")
    # Column index of each head inside the TensorDataset. Derived from the full key order, so
    # dropping heads from the loss cannot shift a batch column onto the wrong head.
    col = {k: i + 1 for i, k in enumerate(full_keys)}

    history = {
        "config": {
            "network_type": "compact_multiresnet" if args.compact else ("single_head_resnet" if args.single_head else "multi_resnet"),
            "compact": args.compact,
            "bottleneck_dim": args.bottleneck_dim if args.compact else None,
            "head_hidden_dim": args.head_hidden_dim if args.compact else None,
            "train_len": args.train_len, "epochs": args.epochs, "batch_size": args.batch_size,
            "lr": args.lr, "sub_avg": args.sub_avg, "val_split": args.val_split,
            "without_permind": args.without_permind, "no_c_heads": args.no_c_heads,
            "c_loss_weight": c_weight, "loss_heads": loss_keys, "seed": args.seed,
            "drop_labels": drop, "decoder_loss_weight": dec_w,
            "note": "seed=None means the trainer set no RNG seed, so run-to-run spread is unmeasured",
        },
        "epochs": [], "epoch_seconds": [], "train_loss": [], "val_loss": [],
        "val_head_ce": {k: [] for k in all_keys}, "val_head_top1": {k: [] for k in all_keys},
        "val_decoder_nll": [],
    }
    history_path = args.out_model + ".history.json"

    best_loss = float("inf")
    print("\n--- Starting Training ---")
    for epoch in range(args.epochs):
        model.train()
        total_loss, seen = 0.0, 0
        t0 = time.time()

        for batch in train_loader:
            xb = batch[0].to(device)
            optimizer.zero_grad(set_to_none=True)

            with autocast("cuda"):
                if args.single_head:
                    yb = batch[1].to(device)
                    logits = model(xb)
                    loss = criterion(logits, yb)
                else:
                    outputs = model(xb)
                    loss = weighted_multilabel_loss(
                        outputs, {k: batch[col[k]].to(device) for k in loss_keys}, criterion,
                        loss_keys, c_weight)
            if dec_w > 0:
                loss = loss + dec_w * decoder_term(outputs, batch[col["S"]].to(device))

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += float(loss.detach()) * xb.shape[0]
            seen += xb.shape[0]

        train_loss = total_loss / max(seen, 1)

        # Validation. Per-head CE and top-1 are recorded for EVERY head, including ones dropped
        # from the loss, so an interference run can show that its C heads really stayed untrained.
        model.eval()
        v_total, v_seen, v_dec = 0.0, 0, 0.0
        v_correct = {k: 0 for k in all_keys}
        v_ce_sum = {k: 0.0 for k in all_keys}
        with torch.no_grad():
            for batch in val_loader:
                xb = batch[0].to(device)
                with autocast("cuda"):
                    if args.single_head:
                        yb = batch[1].to(device)
                        v_loss = criterion(model(xb), yb)
                    else:
                        outputs = model(xb)
                        terms = {k: criterion(outputs[k], batch[col[k]].to(device))
                                 for k in all_keys}
                        # checkpoint selection uses the SAME weighted objective as training;
                        # the per-head val CE below stays unweighted so it is comparable across arms
                        v_loss = sum(weight_of(k, c_weight) * terms[k] for k in loss_keys)
                        for k in all_keys:
                            v_ce_sum[k] += float(terms[k].detach()) * xb.shape[0]
                            v_correct[k] += int((outputs[k].argmax(dim=1)
                                                 == batch[col[k]].to(device)).sum().item())
                if dec_w > 0:
                    d = decoder_term(outputs, batch[col["S"]].to(device))
                    v_loss = v_loss + dec_w * d
                    v_dec += float(d) * xb.shape[0]
                v_total += float(v_loss.detach()) * xb.shape[0]
                v_seen += xb.shape[0]

        val_loss = v_total / max(v_seen, 1)
        if not math.isfinite(val_loss) or not math.isfinite(train_loss):
            print(f"!!! non-finite loss at epoch {epoch + 1}: train {train_loss} val {val_loss} "
                  "(a NaN val loss never becomes 'best', so no checkpoint would be saved)", flush=True)
        el = time.time() - t0
        history["epochs"].append(epoch + 1)
        history["epoch_seconds"].append(round(el, 1))
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_decoder_nll"].append(v_dec / max(v_seen, 1) if dec_w > 0 else None)
        for k in all_keys:
            history["val_head_ce"][k].append(v_ce_sum[k] / max(v_seen, 1))
            history["val_head_top1"][k].append(v_correct[k] / max(v_seen, 1))
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        if args.single_head:
            print(f"Epoch {epoch+1:2d}/{args.epochs:2d}  train_loss={train_loss:.4f}  "
                  f"val_loss={val_loss:.4f}  ({el:.1f}s)")
        else:
            head = {k: history["val_head_top1"][k][-1] for k in all_keys}
            pi_mean = float(np.mean([head[f"permind_{i}_output"] for i in range(16)]))
            c_mean = float(np.mean([head[f"sbox_{i}_output"] for i in range(16)]))
            print(f"Epoch {epoch+1:2d}/{args.epochs:2d}  train_loss={train_loss:.4f}  "
                  f"val_loss={val_loss:.4f}  ({el:.1f}s)  val head top-1: "
                  f"alpha={head['alpha_output']:.4f} beta={head['beta_output']:.4f} "
                  f"C_mean={c_mean:.4f} pi_mean={pi_mean:.4f}"
                  + (f" dec_nll={history['val_decoder_nll'][-1]:.3f}" if dec_w > 0 else ""))

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), args.out_model)
            sidecar = {
                "network_type": "compact_multiresnet" if args.compact else ("single_head_resnet" if args.single_head else ("multi_resnet_without_permind" if args.without_permind else "multi_resnet")),
                "compact": args.compact,
                "bottleneck_dim": (args.bottleneck_dim if (args.compact and args.bottleneck_dim > 0) else None),
                "head_hidden_dim": args.head_hidden_dim if args.compact else None,
                "input_dim": input_dim,
                "train_len": args.train_len,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "sub_avg": args.sub_avg,
                "target_byte": args.target_byte if args.single_head else None,
                "multilabel": 0 if args.single_head else (2 if args.without_permind else 1),
                "val_loss": val_loss,
                "no_c_heads": args.no_c_heads,
                "c_loss_weight": c_weight,
                "seed": args.seed,
                "loss_heads": loss_keys,
                "drop_labels": drop,
                "decoder_loss_weight": dec_w,
                "heads_fp32": args.heads_fp32,
                "best_epoch": epoch + 1,
            }
            with open(args.out_model + ".config.json", "w", encoding="utf-8") as f:
                json.dump(sidecar, f, indent=2)

    if args.save_last and not args.single_head:
        last_path = os.path.splitext(args.out_model)[0] + "_last.pth"
        torch.save(model.state_dict(), last_path)
        print(f"Saved last-epoch checkpoint: {last_path}")

    print(f"\nTraining complete. Best model saved to {args.out_model}")
    print(f"Per-head validation history: {history_path}")


if __name__ == "__main__":
    main()
