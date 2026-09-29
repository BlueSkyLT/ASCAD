#!/usr/bin/env python3
"""Paper-faithful GPAM training on full raw ASCADv2 (800k traces x 1M samples).

Matches Burzstein et al. (TCHES 2024):
  - Architecture: get_gpam_model (3 GAU blocks, hidden dim 768)
  - Inputs: trace_len=1_000_000, patch_size=400 (2,500 patches)
  - Optimizer: Adafactor (lr=5e-5)
  - Batch size: 64, steps per epoch: 1000, epochs: 150
  - Multi-task targets: rm (256), rout (256), c_{b} (256), p_{b} (16)
  - Real-time Telegram monitoring & periodic reporting
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional

# Ensure project root is on PYTHONPATH
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import tensorflow as tf
from tensorflow import keras

from scaaml.models import get_gpam_model
from GPAM.raw_dataset import build_raw_tf_dataset
from scripts.send_telegram import send_message


class TelegramMonitorCallback(keras.callbacks.Callback):
    """Callback that sends periodic status updates to Telegram."""

    def __init__(
        self,
        experiment_name: str,
        total_epochs: int,
        steps_per_epoch: int,
        report_interval_sec: float = 3600.0,  # 1 hour
        chance_total: float = 0.0,
    ):
        super().__init__()
        self.chance_total = chance_total
        self.experiment_name = experiment_name
        self.total_epochs = total_epochs
        self.steps_per_epoch = steps_per_epoch
        self.report_interval_sec = report_interval_sec
        self.start_time = 0.0
        self.last_report_time = 0.0
        self.epoch_start_time = 0.0

    def on_train_begin(self, logs=None):
        self.start_time = time.time()
        self.last_report_time = self.start_time
        msg = (
            f"🚀 [ASCAD GPAM] Training Started!\n"
            f"Experiment: {self.experiment_name}\n"
            f"Epochs: {self.total_epochs} | Steps/epoch: {self.steps_per_epoch}\n"
            f"Trace len: 1,000,000 | Patch size: 400 (2,500 tokens)\n"
            f"Device: RTX 5090 (Blackwell)\n"
            f"Start time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        print(f"[TelegramCallback] Sending start message...")
        send_message(msg)

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        now = time.time()
        elapsed = now - self.start_time
        epoch_dur = now - self.epoch_start_time

        # Calculate progress and ETA
        completed_epochs = epoch + 1
        epochs_remaining = self.total_epochs - completed_epochs
        avg_epoch_time = elapsed / completed_epochs
        eta_sec = epochs_remaining * avg_epoch_time
        eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
        elapsed_str = str(datetime.timedelta(seconds=int(elapsed)))

        # Format loss & accuracy summary
        train_loss = logs.get("loss", 0.0)
        val_loss = logs.get("val_loss", 0.0)

        # Printed every epoch, unconditionally. The 2026-09-25 campaign looked like
        # it was making progress ("19.410 -> 19.4094") while the 4-head chance sum
        # was 19.4081, i.e. still worse than guessing. A loss means nothing without
        # its chance reference next to it.
        if self.chance_total > 0:
            gap = self.chance_total - val_loss
            print(f"[chance-check] epoch {completed_epochs}: val_loss {val_loss:.4f} "
                  f"vs chance {self.chance_total:.4f} -> gap {gap:+.4f} "
                  f"({'BELOW chance = learning' if gap > 0.05 else 'AT/ABOVE chance = NOT learning'})",
                  flush=True)

        # Collect accuracy metrics for heads
        acc_items = []
        for k, v in sorted(logs.items()):
            if "acc" in k:
                acc_items.append(f"  {k}: {v*100:.2f}%")
        acc_text = "\n".join(acc_items) if acc_items else "  N/A"

        # Check if 1 hour has elapsed since last report, or if this is the final epoch
        if (now - self.last_report_time >= self.report_interval_sec) or (completed_epochs == self.total_epochs):
            self.last_report_time = now
            report_msg = (
                f"📊 [ASCAD GPAM] Hourly Progress Report\n"
                f"Epoch: {completed_epochs}/{self.total_epochs} ({completed_epochs/self.total_epochs*100:.1f}%)\n"
                f"Elapsed: {elapsed_str} | ETA: {eta_str}\n"
                f"Last epoch time: {epoch_dur:.1f}s\n"
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}\n"
                + (f"Chance CE: {self.chance_total:.4f} -> gap {self.chance_total - val_loss:+.4f}\n"
                   if self.chance_total > 0 else "")
                + f"Accuracies:\n{acc_text}"
            )
            print(f"[TelegramCallback] Sending hourly report (Epoch {completed_epochs})...")
            send_message(report_msg)

    def on_train_end(self, logs=None):
        elapsed = time.time() - self.start_time
        elapsed_str = str(datetime.timedelta(seconds=int(elapsed)))
        msg = (
            f"🎉 [ASCAD GPAM] Training Finished!\n"
            f"Experiment: {self.experiment_name}\n"
            f"Completed {self.total_epochs} epochs in {elapsed_str}.\n"
            f"Checkpoints and logs saved."
        )
        print(f"[TelegramCallback] Sending training complete message...")
        send_message(msg)


class PeriodicModelCheckpoint(keras.callbacks.Callback):
    """Saves checkpoint every N epochs."""

    def __init__(self, filepath_pattern: str, period: int = 10):
        super().__init__()
        self.filepath_pattern = filepath_pattern
        self.period = period

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.period == 0:
            path = self.filepath_pattern.format(epoch=epoch + 1)
            self.model.save(path)
            print(f"[PeriodicCheckpoint] Saved model checkpoint to {path}")


def build_gpam_raw(
    target_bytes: List[int],
    patch_size: int = 400,
    lr: float = 5e-5,
) -> keras.Model:
    trace_len = 1_000_000
    if trace_len % patch_size != 0:
        raise ValueError(f"trace_len={trace_len} not divisible by patch_size={patch_size}")
    steps = trace_len // patch_size
    if steps % 2 != 0:
        raise ValueError(f"Number of patches ({steps}) must be even for RoPE.")

    # int8 raw traces are in [-128, 127] -> min: -128.0, delta: 255.0
    inputs = {"trace1": {"min": -128.0, "delta": 255.0}}

    outputs = {
        "rm": {"max_val": 256},
        "rout": {"max_val": 256},
    }
    loss_dict = {
        "rm": "sparse_categorical_crossentropy",
        "rout": "sparse_categorical_crossentropy",
    }
    metrics_dict = {
        "rm": ["acc"],
        "rout": ["acc"],
    }

    for b in target_bytes:
        outputs[f"c_{b}"] = {"max_val": 256}
        outputs[f"p_{b}"] = {"max_val": 16}
        loss_dict[f"c_{b}"] = "sparse_categorical_crossentropy"
        loss_dict[f"p_{b}"] = "sparse_categorical_crossentropy"
        metrics_dict[f"c_{b}"] = ["acc"]
        metrics_dict[f"p_{b}"] = ["acc"]

    print(f"Building GPAM model: trace_len={trace_len}, patch_size={patch_size}, {len(outputs)} heads...")
    model = get_gpam_model(
        inputs=inputs,
        outputs=outputs,
        output_relations=[],
        trace_len=trace_len,
        merge_filter_1=0,
        merge_filter_2=0,
        patch_size=patch_size,
    )

    optimizer = keras.optimizers.Adafactor(learning_rate=lr)

    # CRITICAL: run_eagerly=True is required on RTX 5090 (Blackwell) to bypass broken XLA Triton GEMM autotuner
    model.compile(
        optimizer=optimizer,
        loss=loss_dict,
        metrics=metrics_dict,
        run_eagerly=True,
    )
    return model


def evaluate_holdout(
    model: keras.Model,
    holdout_ds: tf.data.Dataset,
    eval_batches: int,
    target_bytes: List[int],
) -> Dict[str, Dict[str, float]]:
    """Evaluates Table 11 metrics on holdout dataset."""
    print(f"\nEvaluating on holdout dataset ({eval_batches} batches = {eval_batches * 64} traces)...")
    head_ranks: Dict[str, List[int]] = {"rm": [], "rout": []}
    for b in target_bytes:
        head_ranks[f"c_{b}"] = []
        head_ranks[f"p_{b}"] = []

    batch_idx = 0
    for x_batch, y_batch in holdout_ds.take(eval_batches):
        preds = model(x_batch, training=False)
        for head_name in head_ranks.keys():
            prob = preds[head_name].numpy()  # [B, num_classes]
            true_label = y_batch[head_name].numpy()  # [B]
            for i in range(len(true_label)):
                tgt_p = prob[i, true_label[i]]
                # Count ties AGAINST the target: with strict `>` a flat head (all
                # probabilities equal) scores rank 0 and looks like a perfect
                # Table 11 result. Identical whenever probabilities are distinct.
                rank = int(np.sum(prob[i] >= tgt_p)) - 1
                head_ranks[head_name].append(rank)
        batch_idx += 1
        if batch_idx % 20 == 0:
            print(f"  Holdout eval progress: {batch_idx}/{eval_batches} batches done.")

    results = {}
    for head_name, ranks in head_ranks.items():
        arr = np.array(ranks)
        acc = float(np.mean(arr == 0)) * 100.0
        mean_rank = float(np.mean(arr))
        max_rank = int(np.max(arr)) if len(arr) > 0 else 0
        results[head_name] = {
            "acc_pct": round(acc, 2),
            "mean_rank": round(mean_rank, 3),
            "max_rank": max_rank,
            "total_traces": len(arr),
        }
        print(f"  Head {head_name:8s}: Acc={acc:6.2f}%, MeanRank={mean_rank:7.3f}, MaxRank={max_rank}")

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Full Raw ASCADv2 GPAM Training Campaign")
    parser.add_argument("--db-dir", type=str, default="STM32_AES_v2/ASCAD_data/ASCAD_databases")
    parser.add_argument("--target-bytes", type=int, nargs="+", default=[0])
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--steps-per-epoch", type=int, default=1000)
    parser.add_argument("--val-steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--patch-size", type=int, default=400)
    parser.add_argument("--checkpoint-dir", type=str, default="GPAM/artifacts/checkpoints_raw")
    parser.add_argument("--log-dir", type=str, default="GPAM/artifacts/logs_raw")
    parser.add_argument("--report-interval-sec", type=float, default=3600.0)
    parser.add_argument("--eval-holdout-batches", type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()

    # Create artifact directories
    ckpt_dir = Path(args.checkpoint_dir)
    log_dir = Path(args.log_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"gpam_raw_b{'_'.join(map(str, args.target_bytes))}_{timestamp}"

    # Configure GPU memory growth
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Configured memory growth for {len(gpus)} GPU(s).")
        except RuntimeError as e:
            print(f"Memory growth config warning: {e}")

    print("=" * 70)
    print(f"ASCADv2 Full Raw GPAM Reproduction Campaign: {run_name}")
    print(f"Target Bytes: {args.target_bytes} | Epochs: {args.epochs} | Steps/epoch: {args.steps_per_epoch}")
    print(f"Batch Size: {args.batch_size} | LR: {args.lr} | Patch Size: {args.patch_size}")
    print(f"Checkpoints: {ckpt_dir} | Logs: {log_dir}")
    print("=" * 70)

    # Build datasets
    print("Initializing Multi-Shard Streaming Datasets...")
    train_ds, train_steps_avail = build_raw_tf_dataset(
        db_dir=args.db_dir,
        split="train",
        target_bytes=args.target_bytes,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        shuffle=True,
    )
    val_ds, val_steps_avail = build_raw_tf_dataset(
        db_dir=args.db_dir,
        split="val",
        target_bytes=args.target_bytes,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        shuffle=False,
    )
    print(f"Train split available steps: {train_steps_avail} | Val split available steps: {val_steps_avail}")

    # Build model
    model = build_gpam_raw(
        target_bytes=args.target_bytes,
        patch_size=args.patch_size,
        lr=args.lr,
    )
    model.summary()

    # Callbacks
    csv_logger = keras.callbacks.CSVLogger(
        filename=str(log_dir / f"{run_name}_metrics.csv"),
        append=True,
    )
    ckpt_callback = PeriodicModelCheckpoint(
        filepath_pattern=str(ckpt_dir / f"{run_name}_epoch_{{epoch:03d}}.keras"),
        period=10,  # every 10 epochs
    )
    best_ckpt_callback = keras.callbacks.ModelCheckpoint(
        filepath=str(ckpt_dir / f"{run_name}_best.keras"),
        monitor="val_loss",
        save_best_only=True,
    )
    # Chance cross-entropy of this exact head set, so every epoch can be read as
    # "better than guessing" or not. Heads: rm + rout + one c and one p per byte.
    nb = len(args.target_bytes)
    chance_total = (2 + nb) * float(np.log(256)) + nb * float(np.log(16))
    print(f"Chance CE reference for {2 + 2*nb} heads: {chance_total:.4f}")

    telegram_callback = TelegramMonitorCallback(
        experiment_name=run_name,
        total_epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        report_interval_sec=args.report_interval_sec,
        chance_total=chance_total,
    )

    callbacks = [
        csv_logger,
        ckpt_callback,
        best_ckpt_callback,
        telegram_callback,
        keras.callbacks.TerminateOnNaN(),
    ]

    try:
        print("\nStarting model.fit() on RTX 5090...")
        history = model.fit(
            train_ds,
            epochs=args.epochs,
            steps_per_epoch=args.steps_per_epoch,
            validation_data=val_ds,
            validation_steps=args.val_steps,
            callbacks=callbacks,
        )

        # Final save
        final_model_path = ckpt_dir / f"{run_name}_final.keras"
        model.save(str(final_model_path))
        print(f"\nFinal model saved to {final_model_path}")

        # Evaluate on Holdout
        if args.eval_holdout_batches > 0:
            print("\nEvaluating Table 11 metrics on holdout dataset...")
            holdout_ds, holdout_steps_avail = build_raw_tf_dataset(
                db_dir=args.db_dir,
                split="holdout",
                target_bytes=args.target_bytes,
                batch_size=args.batch_size,
                chunk_size=args.chunk_size,
                shuffle=False,
            )
            holdout_results = evaluate_holdout(
                model=model,
                holdout_ds=holdout_ds,
                eval_batches=args.eval_holdout_batches,
                target_bytes=args.target_bytes,
            )

            # Save holdout results
            holdout_json_path = log_dir / f"{run_name}_holdout_eval.json"
            with open(holdout_json_path, "w", encoding="utf-8") as f:
                json.dump(holdout_results, f, indent=2)
            print(f"Holdout evaluation results saved to {holdout_json_path}")

            # Send holdout summary to Telegram
            holdout_summary_lines = [f"🏆 [ASCAD GPAM] Holdout Table 11 Evaluation ({args.eval_holdout_batches * 64} traces):"]
            for h_name, h_res in holdout_results.items():
                holdout_summary_lines.append(
                    f"  {h_name}: Acc={h_res['acc_pct']}%, MeanRank={h_res['mean_rank']}, MaxRank={h_res['max_rank']}"
                )
            send_message("\n".join(holdout_summary_lines))

    except Exception as e:
        err_msg = f"❌ [ASCAD GPAM] Training Failed with Error:\n{e}\n\nTraceback:\n{traceback.format_exc()[-500:]}"
        print(err_msg, file=sys.stderr)
        send_message(err_msg)
        raise


if __name__ == "__main__":
    main()
