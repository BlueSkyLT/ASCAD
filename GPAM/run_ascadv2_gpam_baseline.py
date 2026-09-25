#!/usr/bin/env python3
"""Smoke GPAM baseline on ASCADv2 using upstream scaaml.get_gpam_model.

This script is intentionally small: it validates environment and data contract,
then trains a tiny GPAM model on a limited ASCADv2 subset using a single
masked-Sbox output head.

The primary goal is correctness and reproducibility, not a paper-faithful long
run. If this script succeeds, it gives a clean basis for a larger paper-tied
run or a more exact adapter for the full raw dataset.
"""

from __future__ import annotations

import argparse
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

G = np.array([0x0C, 0x05, 0x06, 0x0b, 0x09, 0x00, 0x0a, 0x0d, 0x03, 0x0e, 0x0f, 0x08, 0x04, 0x07, 0x01, 0x02])

def perm_indices(byte_index: int, m0: int, m1: int, m2: int, m3: int) -> int:
    x0, x1, x2, x3 = m0 & 0x0F, m1 & 0x0F, m2 & 0x0F, m3 & 0x0F
    return G[G[G[G[(15 - byte_index) ^ x0] ^ x1] ^ x2] ^ x3]


# GF(256) helpers taken from ASCAD_generate.py.
LOG_TABLE = [
    0, 0, 25, 1, 50, 2, 26, 198, 75, 199, 27, 104, 51, 238, 223, 3,
    100, 4, 224, 14, 52, 141, 129, 239, 76, 113, 8, 200, 248, 105, 28, 193,
    125, 194, 29, 181, 249, 185, 39, 106, 77, 228, 166, 114, 154, 201, 9, 120,
    101, 47, 138, 5, 33, 15, 225, 36, 18, 240, 130, 69, 53, 147, 218, 142,
    150, 143, 219, 189, 54, 208, 206, 148, 19, 92, 210, 241, 64, 70, 131, 56,
    102, 221, 253, 48, 191, 6, 139, 98, 179, 37, 226, 152, 34, 136, 145, 16,
    126, 110, 72, 195, 163, 182, 30, 66, 58, 107, 40, 84, 250, 133, 61, 186,
    43, 121, 10, 21, 155, 159, 94, 202, 78, 212, 172, 229, 243, 115, 167, 87,
    175, 88, 168, 80, 244, 234, 214, 116, 79, 174, 233, 213, 231, 230, 173, 232,
    44, 215, 117, 122, 235, 22, 11, 245, 89, 203, 95, 176, 156, 169, 81, 160,
    127, 12, 246, 111, 23, 196, 73, 236, 216, 67, 31, 45, 164, 118, 123, 183,
    204, 187, 62, 90, 251, 96, 177, 134, 59, 82, 161, 108, 170, 85, 41, 157,
    151, 178, 135, 144, 97, 190, 220, 252, 188, 149, 207, 205, 55, 63, 91, 209,
    83, 57, 132, 60, 65, 162, 109, 71, 20, 42, 158, 93, 86, 242, 211, 171,
    68, 17, 146, 217, 35, 32, 46, 137, 180, 124, 184, 38, 119, 153, 227, 165,
    103, 74, 237, 222, 197, 49, 254, 24, 13, 99, 140, 128, 192, 247, 112, 7,
]

ALOG_TABLE = [
    1, 3, 5, 15, 17, 51, 85, 255, 26, 46, 114, 150, 161, 248, 19, 53,
    95, 225, 56, 72, 216, 115, 149, 164, 247, 2, 6, 10, 30, 34, 102, 170,
    229, 52, 92, 228, 55, 89, 235, 38, 106, 190, 217, 112, 144, 171, 230, 49,
    83, 245, 4, 12, 20, 60, 68, 204, 79, 209, 104, 184, 211, 110, 178, 205,
    76, 212, 103, 169, 224, 59, 77, 215, 98, 166, 241, 8, 24, 40, 120, 136,
    131, 158, 185, 208, 107, 189, 220, 127, 129, 152, 179, 206, 73, 219, 118, 154,
    181, 196, 87, 249, 16, 48, 80, 240, 11, 29, 39, 105, 187, 214, 97, 163,
    254, 25, 43, 125, 135, 146, 173, 236, 47, 113, 147, 174, 233, 32, 96, 160,
    251, 22, 58, 78, 210, 109, 183, 194, 93, 231, 50, 86, 250, 21, 63, 65,
    195, 94, 226, 61, 71, 201, 64, 192, 91, 237, 44, 116, 156, 191, 218, 117,
    159, 186, 213, 100, 172, 239, 42, 126, 130, 157, 188, 223, 122, 142, 137, 128,
    155, 182, 193, 88, 232, 35, 101, 175, 234, 37, 111, 177, 200, 67, 197, 84,
    252, 31, 33, 99, 165, 244, 7, 9, 27, 45, 119, 153, 176, 203, 70, 202,
    69, 207, 74, 222, 121, 139, 134, 145, 168, 227, 62, 66, 198, 81, 243, 14,
    18, 54, 90, 238, 41, 123, 141, 140, 143, 138, 133, 148, 167, 242, 13, 23,
    57, 75, 221, 124, 132, 151, 162, 253, 28, 36, 108, 180, 199, 82, 246, 1,
]


def gf256_mul(a: int, b: int) -> int:
    if a == 0 or b == 0:
        return 0
    return ALOG_TABLE[(LOG_TABLE[a] + LOG_TABLE[b]) % 255]


def masked_sbox_label_for_metadata(metadata_row: np.ndarray, target_byte: int) -> int:
    masks = metadata_row["masks"]
    plaintext = metadata_row["plaintext"]
    key = metadata_row["key"]
    ind = perm_indices(target_byte, masks[0], masks[1], masks[2], masks[3])
    alpha = masks[18]
    beta = masks[17]
    masked_s = AES_SBOX[plaintext[ind] ^ key[ind]]
    return gf256_mul(alpha, masked_s) ^ beta


def load_ascadv2_subset(path: str, split: str, n_train: int, n_val: int, target_byte: int, dataset_kind: str):
    with h5py.File(path, "r") as f:
        if dataset_kind == "extracted":
            traces = f[f"{split}_traces/traces"]
            metadata = f[f"{split}_traces/metadata"]
        else:
            traces = f["traces"]
            metadata = f["metadata"]

        if dataset_kind == "raw":
            X = traces[: n_train + n_val].astype(np.float32)
            meta = metadata[: n_train + n_val]
            y = np.array([masked_sbox_label_for_metadata(row, target_byte) for row in meta], dtype=np.int32)
        else:
            labels = f[f"{split}_traces/labels"]
            X = traces[: n_train + n_val].astype(np.float32)
            y = labels["sbox_masked"][: n_train + n_val, target_byte].astype(np.int32)

    x_train, x_val = X[:n_train], X[n_train:n_train + n_val]
    y_train, y_val = y[:n_train], y[n_train:n_train + n_val]
    return x_train, y_train, x_val, y_val


def build_model(train_x: np.ndarray, target_byte: int, patch_size: int = 150):
    trace_len = int(train_x.shape[1])
    if trace_len % patch_size != 0:
        raise ValueError(f"trace_len={trace_len} must be divisible by patch_size={patch_size}")
    steps = trace_len // patch_size
    if steps % 2 != 0:
        raise ValueError(
            f"GPAM RoPE requires an even number of patches; got steps={steps} for trace_len={trace_len}, patch_size={patch_size}. "
            "Choose a patch_size that yields an even steps count (e.g. 150 for 15000-point traces)."
        )

    trace_min = float(train_x.min())
    trace_delta = float(train_x.max() - trace_min)
    if trace_delta == 0:
        raise ValueError("Trace delta is zero; cannot normalize.")

    inputs = {"trace1": {"min": trace_min, "delta": trace_delta}}
    outputs = {f"sbox_byte_{target_byte}": {"max_val": 256}}
    model = get_gpam_model(
        inputs=inputs,
        outputs=outputs,
        output_relations=[],
        trace_len=trace_len,
        merge_filter_1=0,
        merge_filter_2=0,
        patch_size=patch_size,
    )
    model.compile(
        optimizer=keras.optimizers.Adafactor(learning_rate=1e-3),
        loss={f"sbox_byte_{target_byte}": "sparse_categorical_crossentropy"},
        metrics={f"sbox_byte_{target_byte}": ["acc"]},
        run_eagerly=True,
    )
    return model


def parse_args():
    parser = argparse.ArgumentParser(description="GPAM ASCADv2 smoke baseline")
    parser.add_argument("--dataset-path", type=str, default="STM32_AES_v2/ASCAD_data/ASCAD_databases/ascadv2-extracted.h5")
    parser.add_argument("--dataset-kind", choices=["extracted", "raw"], default="extracted")
    parser.add_argument("--split", choices=["Profiling", "Attack"], default="Profiling")
    parser.add_argument("--target-byte", type=int, default=0)
    parser.add_argument("--n-train", type=int, default=2048)
    parser.add_argument("--n-val", type=int, default=512)
    parser.add_argument("--patch-size", type=int, default=150)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    split_name = args.split
    split_key = "Profiling" if split_name == "Profiling" else "Attack"
    x_train, y_train, x_val, y_val = load_ascadv2_subset(
        str(dataset_path),
        split=split_key,
        n_train=args.n_train,
        n_val=args.n_val,
        target_byte=args.target_byte,
        dataset_kind=args.dataset_kind,
    )
    print(f"Loaded {x_train.shape[0]} train / {x_val.shape[0]} val traces")
    print(f"Train target unique count: {len(np.unique(y_train))}")
    print(f"Val target unique count: {len(np.unique(y_val))}")

    model = build_model(x_train, args.target_byte, patch_size=args.patch_size)
    print(model.summary())

    hist = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        batch_size=args.batch_size,
        epochs=args.epochs,
        verbose=1,
    )
    print("Final accuracy:", hist.history.get("val_acc", hist.history.get("val_sbox_byte_0_acc", []))[-1])


if __name__ == "__main__":
    main()
