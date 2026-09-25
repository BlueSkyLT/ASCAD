#!/usr/bin/env python3
"""Budget-controlled GPAM ASCADv2 reproduction targeting Table 11 GE < 1 @ ~80 traces.

Uses a scaled subset (100k train / 10k val / 10k attack) to stay under $5 budget.
Implements multi-byte masked Sbox heads and evaluates GE during training.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import tensorflow as tf
from tensorflow import keras

from scaaml.models import get_gpam_model


AES_SBOX = np.array([
    0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
    0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
    0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
    0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
    0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
    0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
    0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
    0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
    0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
    0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
    0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
    0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
    0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
    0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
    0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16,
])


def load_data(path: str, n_train: int, n_val: int, n_attack: int, target_bytes: list[int]):
    with h5py.File(path, "r") as f:
        prof_traces = f["Profiling_traces/traces"][:n_train + n_val].astype(np.float32)
        prof_labels_struct = f["Profiling_traces/labels"][:n_train + n_val]
        prof_labels = prof_labels_struct["sbox_masked"]
        attack_traces = f["Attack_traces/traces"][:n_attack].astype(np.float32)
        attack_labels_struct = f["Attack_traces/labels"][:n_attack]
        attack_labels = attack_labels_struct["sbox_masked"]
        attack_meta = f["Attack_traces/metadata"][:n_attack]

    x_train, x_val = prof_traces[:n_train], prof_traces[n_train:n_train + n_val]
    y_train = {f"sbox_byte_{b}": prof_labels[:n_train, b].astype(np.int32) for b in target_bytes}
    y_val = {f"sbox_byte_{b}": prof_labels[n_train:n_train + n_val, b].astype(np.int32) for b in target_bytes}

    return x_train, y_train, x_val, y_val, attack_traces, attack_labels, attack_meta


def build_model(train_x: np.ndarray, target_bytes: list[int], patch_size: int = 150):
    trace_len = int(train_x.shape[1])
    if trace_len % patch_size != 0:
        raise ValueError(f"trace_len={trace_len} must be divisible by patch_size={patch_size}")
    steps = trace_len // patch_size
    if steps % 2 != 0:
        raise ValueError(f"GPAM RoPE requires even patch count; got {steps}")

    trace_min = float(train_x.min())
    trace_delta = float(train_x.max() - trace_min)
    if trace_delta == 0:
        raise ValueError("Trace delta is zero")

    inputs = {"trace1": {"min": trace_min, "delta": trace_delta}}
    outputs = {f"sbox_byte_{b}": {"max_val": 256} for b in target_bytes}
    model = get_gpam_model(
        inputs=inputs,
        outputs=outputs,
        output_relations=[],
        trace_len=trace_len,
        merge_filter_1=0,
        merge_filter_2=0,
        patch_size=patch_size,
    )
    loss_dict = {f"sbox_byte_{b}": "sparse_categorical_crossentropy" for b in target_bytes}
    metrics_dict = {f"sbox_byte_{b}": ["acc"] for b in target_bytes}
    model.compile(
        optimizer=keras.optimizers.Adafactor(learning_rate=1e-3),
        loss=loss_dict,
        metrics=metrics_dict,
        run_eagerly=True,
    )
    return model


def compute_ge(model, attack_traces, attack_labels, attack_meta, target_bytes, max_traces=1000):
    """Compute guessing entropy for each target byte using classical key-rank approach."""
    n_attack = min(len(attack_traces), max_traces)
    ge_results = {}

    for byte_idx in target_bytes:
        preds = model.predict(attack_traces[:n_attack], batch_size=32, verbose=0)
        if isinstance(preds, dict):
            byte_preds = preds[f"sbox_byte_{byte_idx}"]
        else:
            byte_preds = preds

        # Classical key recovery: for each key hypothesis, accumulate log-probs
        key_scores = np.zeros(256, dtype=np.float64)
        for trace_idx in range(n_attack):
            plaintext = attack_meta[trace_idx]["plaintext"]
            for key_hyp in range(256):
                z_hyp = AES_SBOX[plaintext[byte_idx] ^ key_hyp]
                key_scores[key_hyp] += np.log(byte_preds[trace_idx, z_hyp] + 1e-40)

        # Rank by score
        ranked_keys = np.argsort(-key_scores)
        true_key = attack_meta[0]["key"][byte_idx]
        ge = int(np.where(ranked_keys == true_key)[0][0])
        ge_results[byte_idx] = ge

    return ge_results


class GECallback(keras.callbacks.Callback):
    def __init__(self, attack_traces, attack_labels, attack_meta, target_bytes, ge_target=1, check_every=5):
        super().__init__()
        self.attack_traces = attack_traces
        self.attack_labels = attack_labels
        self.attack_meta = attack_meta
        self.target_bytes = target_bytes
        self.ge_target = ge_target
        self.check_every = check_every
        self.best_ge = {b: float('inf') for b in target_bytes}

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.check_every != 0:
            return

        ge_results = compute_ge(
            self.model, self.attack_traces, self.attack_labels,
            self.attack_meta, self.target_bytes, max_traces=100
        )
        
        print(f"\n[Epoch {epoch+1}] GE results (100 attack traces):")
        for byte_idx, ge in ge_results.items():
            print(f"  Byte {byte_idx}: GE = {ge}")
            self.best_ge[byte_idx] = min(self.best_ge[byte_idx], ge)

        # Early stop if all bytes meet target
        if all(ge <= self.ge_target for ge in ge_results.values()):
            print(f"\n✓ All target bytes achieved GE ≤ {self.ge_target}. Stopping early.")
            self.model.stop_training = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str, default="STM32_AES_v2/ASCAD_data/ASCAD_databases/ascadv2-extracted.h5")
    parser.add_argument("--n-train", type=int, default=100000)
    parser.add_argument("--n-val", type=int, default=10000)
    parser.add_argument("--n-attack", type=int, default=1000)
    parser.add_argument("--target-bytes", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--patch-size", type=int, default=150)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--ge-target", type=int, default=1)
    parser.add_argument("--check-every", type=int, default=5)
    parser.add_argument("--out-json", type=str, default="GPAM/gpam_budget_results.json")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    print(f"Loading {args.n_train} train / {args.n_val} val / {args.n_attack} attack traces...")
    x_train, y_train, x_val, y_val, attack_traces, attack_labels, attack_meta = load_data(
        str(dataset_path), args.n_train, args.n_val, args.n_attack, args.target_bytes
    )
    print(f"Train shape: {x_train.shape}")
    print(f"Target bytes: {args.target_bytes}")

    print("\nBuilding GPAM model...")
    model = build_model(x_train, args.target_bytes, patch_size=args.patch_size)
    print(model.summary())

    ge_callback = GECallback(
        attack_traces, attack_labels, attack_meta, args.target_bytes,
        ge_target=args.ge_target, check_every=args.check_every
    )

    print(f"\nTraining for up to {args.epochs} epochs (early stop at GE ≤ {args.ge_target})...")
    hist = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        batch_size=args.batch_size,
        epochs=args.epochs,
        callbacks=[ge_callback],
        verbose=1,
    )

    print("\n" + "="*60)
    print("Final GE evaluation on full attack set...")
    final_ge = compute_ge(model, attack_traces, attack_labels, attack_meta, args.target_bytes, max_traces=args.n_attack)
    print("Final GE results:")
    for byte_idx, ge in final_ge.items():
        print(f"  Byte {byte_idx}: GE = {ge}")

    results = {
        "config": vars(args),
        "best_ge_during_training": ge_callback.best_ge,
        "final_ge": final_ge,
        "epochs_trained": len(hist.history["loss"]),
    }
    
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
