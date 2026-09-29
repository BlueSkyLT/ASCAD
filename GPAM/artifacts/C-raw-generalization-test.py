#!/usr/bin/env python3
"""Does GPAM GENERALISE on raw 1M-point ASCADv2 traces?

Every positive number so far is either train=eval memorisation (512 traces, where
a 17.8M-param model hits 94-98% even on SHUFFLED labels) or comes from the 15k POI
crop. This is the missing test: train on shard 1, evaluate on shard 2.

Shard 2 is a different capture block, so no row can be shared. Labels come from
GPAM/raw_dataset.compute_labels, already validated against the extracted file's
stored labels on 40,000 joined rows (0 mismatches).

Training batches are streamed from HDF5 so peak RAM stays ~4 GiB (the eval slab);
materialising 4,000 x 1,000,000 float32 would need 16 GiB before TF copies it.
"""
import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import tensorflow as tf
from tensorflow import keras

HERE = Path(__file__).resolve().parent          # GPAM/artifacts
GPAM_DIR = HERE.parent                          # GPAM
REPO_ROOT = GPAM_DIR.parent                     # repo root
sys.path.insert(0, str(GPAM_DIR))
from raw_dataset import RawMultiShardLoader  # noqa: E402
from scaaml.models import get_gpam_model  # noqa: E402

DB = REPO_ROOT / "STM32_AES_v2/ASCAD_data/ASCAD_databases"
HEADS = ["rm", "rout", "c_0", "p_0"]
NCLASSES = {"rm": 256, "rout": 256, "c_0": 256, "p_0": 16}
CHANCE_TOTAL = 3 * float(np.log(256)) + float(np.log(16))

_L = RawMultiShardLoader(split="train", target_bytes=[0], batch_size=1, shuffle=False)


def read_slab(shard, start, n):
    with h5py.File(DB / f"ascadv2-stm32-conso-raw-traces{shard}.h5", "r") as f:
        X = f["traces"][start:start + n].astype(np.float32)
        meta = f["metadata"][start:start + n]
    return X, _L.compute_labels(meta)


def train_ds(n_train, batch_size):
    """Streams shard-1 rows one batch at a time; re-read each epoch via from_generator."""
    def gen():
        with h5py.File(DB / "ascadv2-stm32-conso-raw-traces1.h5", "r") as f:
            for s in range(0, n_train, batch_size):
                e = min(s + batch_size, n_train)
                X = f["traces"][s:e].astype(np.float32)
                y = _L.compute_labels(f["metadata"][s:e])
                yield X, y
    sig = (tf.TensorSpec([None, 1_000_000], tf.float32),
           {h: tf.TensorSpec([None], tf.int32) for h in HEADS})
    return tf.data.Dataset.from_generator(gen, output_signature=sig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-eval", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--patch-size", type=int, default=400)
    a = ap.parse_args()

    t0 = time.time()
    Xev, Yev = read_slab(2, 0, a.n_eval)
    steps_per_epoch = int(np.ceil(a.n_train / a.batch_size))
    print(f"[{time.time()-t0:5.1f}s] eval slab shard2 {Xev.shape} resident; "
          f"train shard1 streamed {a.n_train} traces x {a.epochs} epochs "
          f"({steps_per_epoch} steps/epoch)", flush=True)

    model = get_gpam_model(inputs={"trace1": {"min": -128.0, "delta": 255.0}},
                           outputs={h: {"max_val": NCLASSES[h]} for h in HEADS},
                           output_relations=[], trace_len=1_000_000,
                           merge_filter_1=0, merge_filter_2=0, patch_size=a.patch_size)
    model.compile(optimizer=keras.optimizers.Adafactor(learning_rate=a.lr),
                  loss={h: "sparse_categorical_crossentropy" for h in HEADS},
                  metrics={h: ["acc"] for h in HEADS}, run_eagerly=True)

    ds = train_ds(a.n_train, a.batch_size).prefetch(2)
    for ep in range(1, a.epochs + 1):
        h = model.fit(ds, steps_per_epoch=steps_per_epoch, epochs=1, verbose=0,
                      validation_data=(Xev, Yev)).history
        print(f"[{time.time()-t0:6.1f}s] lr={a.lr:g} ep {ep:2d}/{a.epochs} | "
              f"train {float(h['loss'][0]):8.4f} | VAL(shard2) {float(h['val_loss'][0]):8.4f} "
              f"| chance {CHANCE_TOTAL:.4f} | val gap {CHANCE_TOTAL - float(h['val_loss'][0]):+.4f} "
              f"| val rm {float(h['val_rm_acc'][0])*100:6.3f}% (ch 0.391%) "
              f"rout {float(h['val_rout_acc'][0])*100:6.3f}% "
              f"c_0 {float(h['val_c_0_acc'][0])*100:6.3f}% "
              f"p_0 {float(h['val_p_0_acc'][0])*100:5.2f}% (ch 6.25%)", flush=True)

    print(f"\n=== lr={a.lr:g} : GENERALISATION verdict ===")
    print(f"  shard-2 held-out rm acc {float(h['val_rm_acc'][0])*100:.3f}% vs chance 0.391%")
    print(f"  crop-run reference (extracted 15k, 40k train, 8 ep, lr 1e-3): rm 92.95% held out")
    print(f"  {'PASSES' if float(h['val_rm_acc'][0]) > 0.02 else 'FAILS'}: "
          f"raw 1M-point traces {'can' if float(h['val_rm_acc'][0]) > 0.02 else 'cannot'} "
          f"be learned at this scale/LR (threshold 2% = 5x chance)")
    print(f"[{time.time()-t0:.1f}s] done")


if __name__ == "__main__":
    main()
