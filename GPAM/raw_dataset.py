#!/usr/bin/env python3
"""Streaming multi-shard HDF5 data loader for full raw ASCADv2 (800k traces x 1M samples).

Handles the official 8 shards:
  Shards 1-6 (600,000 traces) + Shard 7 [0:40,000]       -> Train (640,000 traces)
  Shard 7 [40,000:100,000] + Shard 8 [0:20,000]          -> Validation (80,000 traces)
  Shard 8 [20,000:100,000]                               -> Holdout (80,000 traces)

Dynamically extracts labels from metadata:
  rm: alpha mask (multiply hide, masks[18])
  rout: beta mask (additive hide, masks[17])
  p_{b}: permutation index for byte b
  c_{b}: masked S-box output for byte b
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

import h5py
import numpy as np
import tensorflow as tf

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
], dtype=np.uint8)

G_PERM = np.array([0x0C, 0x05, 0x06, 0x0b, 0x09, 0x00, 0x0a, 0x0d, 0x03, 0x0e, 0x0f, 0x08, 0x04, 0x07, 0x01, 0x02], dtype=np.uint8)

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

# Precomputed multiplication table for vectorized speed
MUL_TABLE = np.array([[gf256_mul(a, b) for b in range(256)] for a in range(256)], dtype=np.uint8)


def perm_indices_vec(byte_index: int, m0: np.ndarray, m1: np.ndarray, m2: np.ndarray, m3: np.ndarray) -> np.ndarray:
    x0, x1, x2, x3 = m0 & 0x0F, m1 & 0x0F, m2 & 0x0F, m3 & 0x0F
    return G_PERM[G_PERM[G_PERM[G_PERM[(15 - byte_index) ^ x0] ^ x1] ^ x2] ^ x3]


class RawMultiShardLoader:
    """Manages multi-shard streaming across the 8 ASCADv2 raw files."""

    def __init__(
        self,
        db_dir: str = "STM32_AES_v2/ASCAD_data/ASCAD_databases",
        split: str = "train",
        target_bytes: Optional[List[int]] = None,
        chunk_size: int = 256,
        batch_size: int = 64,
        shuffle: bool = True,
        seed: int = 42,
    ):
        self.db_dir = Path(db_dir)
        self.split = split
        self.target_bytes = target_bytes if target_bytes is not None else [0]
        self.chunk_size = chunk_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)

        # Plan the segment slices per shard for this split
        # Each shard has 100,000 traces.
        self.shard_slices: List[Tuple[int, int, int]] = []
        if split == "train":
            # 640k traces: Shards 1-6 full (100k each) + Shard 7 [0:40000]
            for s in range(1, 7):
                self.shard_slices.append((s, 0, 100000))
            self.shard_slices.append((7, 0, 40000))
            self.total_traces = 640000
        elif split == "val":
            # 80k traces: Shard 7 [40000:100000] (60k) + Shard 8 [0:20000] (20k)
            self.shard_slices.append((7, 40000, 100000))
            self.shard_slices.append((8, 0, 20000))
            self.total_traces = 80000
        elif split == "holdout":
            # 80k traces: Shard 8 [20000:100000]
            self.shard_slices.append((8, 20000, 100000))
            self.total_traces = 80000
        else:
            raise ValueError(f"Unknown split: {split}")

    def compute_labels(self, metadata: np.ndarray) -> Dict[str, np.ndarray]:
        """Vectorized extraction of targets from metadata chunk."""
        masks = metadata["masks"]
        plaintext = metadata["plaintext"]
        key = metadata["key"]

        rm = masks[:, 18].astype(np.int32)
        rout = masks[:, 17].astype(np.int32)

        labels = {
            "rm": rm,
            "rout": rout,
        }

        m0, m1, m2, m3 = masks[:, 0], masks[:, 1], masks[:, 2], masks[:, 3]

        for b in self.target_bytes:
            ind = perm_indices_vec(b, m0, m1, m2, m3)
            labels[f"p_{b}"] = ind.astype(np.int32)

            # Vectorized sbox masked calculation:
            # pt[i, ind[i]] ^ key[i, ind[i]]
            row_idx = np.arange(len(metadata))
            p_val = plaintext[row_idx, ind]
            k_val = key[row_idx, ind]
            s_val = AES_SBOX[p_val ^ k_val]

            c_val = MUL_TABLE[rm, s_val] ^ rout
            labels[f"c_{b}"] = c_val.astype(np.int32)

        return labels

    def generator(self) -> Generator[Tuple[np.ndarray, Dict[str, np.ndarray]], None, None]:
        """Yields batches of (X, y_dict)."""
        while True:
            # If shuffle, randomly order the shards in this epoch
            slices = list(self.shard_slices)
            if self.shuffle:
                self.rng.shuffle(slices)

            for shard_id, start_idx, end_idx in slices:
                shard_path = self.db_dir / f"ascadv2-stm32-conso-raw-traces{shard_id}.h5"
                with h5py.File(shard_path, "r") as f:
                    traces_ds = f["traces"]
                    meta_ds = f["metadata"]

                    # Chunk through this shard
                    indices = np.arange(start_idx, end_idx)
                    if self.shuffle:
                        self.rng.shuffle(indices)

                    for chunk_start in range(0, len(indices), self.chunk_size):
                        chunk_idx = indices[chunk_start : chunk_start + self.chunk_size]
                        # HDF5 sorted read optimization
                        sort_order = np.argsort(chunk_idx)
                        sorted_idx = chunk_idx[sort_order]

                        raw_x = traces_ds[sorted_idx].astype(np.float32)
                        raw_meta = meta_ds[sorted_idx]

                        # Restore original chunk order
                        inv_order = np.empty_like(sort_order)
                        inv_order[sort_order] = np.arange(len(chunk_idx))
                        chunk_x = raw_x[inv_order]
                        chunk_meta = raw_meta[inv_order]

                        chunk_y = self.compute_labels(chunk_meta)

                        # Yield in batch_size slices
                        for b_start in range(0, len(chunk_x), self.batch_size):
                            bx = chunk_x[b_start : b_start + self.batch_size]
                            by = {k: v[b_start : b_start + self.batch_size] for k, v in chunk_y.items()}
                            if len(bx) == self.batch_size:
                                yield bx, by

            if not self.shuffle:
                break


def build_raw_tf_dataset(
    db_dir: str = "STM32_AES_v2/ASCAD_data/ASCAD_databases",
    split: str = "train",
    target_bytes: Optional[List[int]] = None,
    batch_size: int = 64,
    chunk_size: int = 256,
    shuffle: bool = True,
    seed: int = 42,
) -> Tuple[tf.data.Dataset, int]:
    """Creates a tf.data.Dataset from RawMultiShardLoader."""
    loader = RawMultiShardLoader(
        db_dir=db_dir,
        split=split,
        target_bytes=target_bytes,
        chunk_size=chunk_size,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
    )

    y_sig = {
        "rm": tf.TensorSpec(shape=(batch_size,), dtype=tf.int32),
        "rout": tf.TensorSpec(shape=(batch_size,), dtype=tf.int32),
    }
    for b in loader.target_bytes:
        y_sig[f"c_{b}"] = tf.TensorSpec(shape=(batch_size,), dtype=tf.int32)
        y_sig[f"p_{b}"] = tf.TensorSpec(shape=(batch_size,), dtype=tf.int32)

    output_signature = (
        tf.TensorSpec(shape=(batch_size, 1000000), dtype=tf.float32),
        y_sig,
    )

    dataset = tf.data.Dataset.from_generator(
        loader.generator,
        output_signature=output_signature,
    )
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    total_steps = loader.total_traces // batch_size
    return dataset, total_steps


if __name__ == "__main__":
    import time
    print("Testing RawMultiShardLoader...")
    t0 = time.time()
    ds, steps = build_raw_tf_dataset(split="train", target_bytes=[0], batch_size=64, chunk_size=256)
    print(f"Dataset created with {steps} steps per epoch.")
    for i, (x, y) in enumerate(ds.take(2)):
        print(f"Batch {i}: X shape={x.shape}, rm shape={y['rm'].shape}, c_0 shape={y['c_0'].shape}")
    print(f"Test completed in {time.time() - t0:.2f} s")
