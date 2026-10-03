#!/usr/bin/env python3
"""Table 11 evaluation of a trained 34-head raw GPAM on the 80,000-trace holdout.

Paper format (Burzstein et al., CHES 2024): per-head Accuracy | MeanRank | MaxRank.
No training happens here -- the shards are read once, in order, from shard 8
[20000:100000], which was never seen by the optimizer.

Usage:
  python C-table11-holdout-eval.py --model <ckpt.keras> [--max-batches 2500]
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import tensorflow as tf
from tensorflow import keras
from scaaml.models.gpam import GAU, Rescale, ScaledNorm

HERE = Path(__file__).resolve().parent          # GPAM/artifacts
GPAM_DIR = HERE.parent                            # GPAM
REPO_ROOT = GPAM_DIR.parent                       # repo
sys.path.insert(0, str(REPO_ROOT))

from GPAM.raw_dataset import build_raw_tf_dataset  # noqa: E402
from scripts.send_telegram import send_message  # noqa: E402

CHANCE_ACC = {256: 100.0 / 256, 16: 100.0 / 16}
# Mean rank of a uniform-random score vector over C classes (ties counted against
# the target) is (C-1)/2.
CHANCE_MEAN_RANK = {256: 127.5, 16: 7.5}


def head_n_classes(target_bytes: List[int]) -> Dict[str, int]:
    n = {"rm": 256, "rout": 256}
    for b in target_bytes:
        n[f"c_{b}"] = 256
        n[f"p_{b}"] = 16
    return n


def evaluate(model, ds, n_batches: int, n_classes: Dict[str, int], tag: str = ""):
    heads = list(n_classes.keys())
    # Streaming accumulators: rank histogram is cheap and lets MaxRank/MeanRank/
    # Accuracy all come off the same pass without storing 34 x 80000 ints.
    hist = {h: np.zeros(n_classes[h] + 1, dtype=np.int64) for h in heads}
    # Per-head top-k accuracy (k = 1, 2, 4, 8, 16) for the paper-adjacent story.
    TOPK = (1, 2, 4, 8, 16)
    topk = {h: np.zeros(len(TOPK), dtype=np.int64) for h in heads}
    seen = {h: 0 for h in heads}

    t0 = time.time()
    done = 0
    for x_batch, y_batch in ds.take(n_batches):
        preds = model(x_batch, training=False)
        for h in heads:
            prob = np.asarray(preds[h])
            y = np.asarray(y_batch[h])
            tgt_p = prob[np.arange(len(y)), y]
            # Worst-case rank: count every class scoring >= the true one. A flat
            # head then lands at rank C-1, not rank 0.
            rank = (prob >= tgt_p[:, None]).sum(1) - 1
            np.add.at(hist[h], rank, 1)
            kth = np.argpartition(-prob, TOPK[-1] - 1, axis=1)[:, :TOPK[-1]]
            order = np.take_along_axis(kth, np.argsort(-prob[np.arange(len(y))[:, None], kth], axis=1), axis=1)
            hit = (order == y[:, None])
            for i, k in enumerate(TOPK):
                topk[h][i] += int(hit[:, :k].any(1).sum())
            seen[h] += len(y)
        done += 1
        if done % 100 == 0:
            el = time.time() - t0
            eta = el / done * (n_batches - done)
            print(f"  {tag}{done}/{n_batches} batches "
                  f"({seen['rm']} traces) {el / 60:.1f} min elapsed, "
                  f"ETA {eta / 60:.1f} min", flush=True)

    results = {}
    for h in heads:
        ranks = np.arange(len(hist[h]))
        cnt = hist[h]
        n = max(seen[h], 1)
        mean_rank = float((ranks * cnt).sum() / n)
        max_rank = int(np.max(ranks[cnt > 0])) if n else 0
        c = CHANCE_MEAN_RANK[n_classes[h]]
        results[h] = {
            "acc_pct": round(float(cnt[0]) / n * 100.0, 3),
            "mean_rank": round(mean_rank, 3),
            "max_rank": max_rank,
            "chance_acc_pct": round(CHANCE_ACC[n_classes[h]], 3),
            "chance_mean_rank": c,
            "mean_rank_ratio_vs_chance": round(mean_rank / c, 4),
            "acc_over_chance": round((float(cnt[0]) / n * 100.0) / CHANCE_ACC[n_classes[h]], 2),
            "topk_pct": {str(k): round(float(v) / n * 100.0, 3) for k, v in zip(TOPK, topk[h])},
            "n_traces": n,
        }
    return results


def group_summary(res: Dict[str, dict], target_bytes: List[int]) -> Dict[str, dict]:
    """Collapse 34 heads into the paper's four quantities."""
    out = {}
    for name, keys in [
        ("rm", ["rm"]),
        ("rout", ["rout"]),
        ("p (shuffle index)", [f"p_{b}" for b in target_bytes]),
        ("c (masked S-box)", [f"c_{b}" for b in target_bytes]),
    ]:
        vals = [res[k] for k in keys]
        nc = res[keys[0]]["chance_acc_pct"]
        out[name] = {
            "n_heads": len(keys),
            "acc_mean_pct": round(float(np.mean([v["acc_pct"] for v in vals])), 3),
            "acc_min_pct": round(float(np.min([v["acc_pct"] for v in vals])), 3),
            "acc_max_pct": round(float(np.max([v["acc_pct"] for v in vals])), 3),
            "mean_rank_mean": round(float(np.mean([v["mean_rank"] for v in vals])), 3),
            "max_rank_worst": int(np.max([v["max_rank"] for v in vals])),
            "chance_acc_pct": nc,
            "chance_mean_rank": res[keys[0]]["chance_mean_rank"],
            "best_head": max(keys, key=lambda k: res[k]["acc_pct"]),
            "worst_head": min(keys, key=lambda k: res[k]["acc_pct"]),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--target-bytes", type=int, nargs="+", default=list(range(16)))
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--chunk-size", type=int, default=128)
    ap.add_argument("--max-batches", type=int, default=2500,
                    help="2500 x 32 = the full 80,000-trace holdout")
    ap.add_argument("--db-dir", type=str,
                    default=str(REPO_ROOT / "STM32_AES_v2/ASCAD_data/ASCAD_databases"))
    ap.add_argument("--out", type=str, default=str(HERE / "C-table11-holdout.json"))
    ap.add_argument("--label", type=str, default="")
    ap.add_argument("--no-telegram", action="store_true")
    args = ap.parse_args()

    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except RuntimeError as e:
            print(e)

    label = args.label or Path(args.model).stem
    print(f"[load] {args.model}")
    model = keras.models.load_model(
        args.model,
        custom_objects={"GAU": GAU, "Rescale": Rescale, "ScaledNorm": ScaledNorm},
    )
    n_classes = head_n_classes(args.target_bytes)
    print(f"[load] ok -- {len(model.outputs)} heads, inputs={model.input_shape}")

    ds, steps = build_raw_tf_dataset(
        db_dir=args.db_dir,
        split="holdout",
        target_bytes=args.target_bytes,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        shuffle=False,
    )
    n_batches = min(args.max_batches, steps)
    print(f"[data] holdout = shard8[20000:100000], {n_batches} batches x {args.batch_size} "
          f"= {n_batches * args.batch_size:,} traces")

    t0 = time.time()
    res = evaluate(model, ds, n_batches, n_classes, tag="[eval] ")
    elapsed = time.time() - t0
    summary = group_summary(res, args.target_bytes)

    payload = {
        "model": str(args.model),
        "label": label,
        "split": "holdout (shard 8 [20000:100000], never trained on)",
        "n_batches": n_batches,
        "n_traces": n_batches * args.batch_size,
        "batch_size": args.batch_size,
        "tie_rule": "rank = #{classes with prob >= true-class prob} - 1 (ties counted against target)",
        "elapsed_min": round(elapsed / 60.0, 1),
        "finished_utc": datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "group_summary": summary,
        "per_head": res,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(f"[out] wrote {args.out}")

    print(f"\n===== TABLE 11-STYLE HELD-OUT RESULTS :: {label} "
          f"({payload['n_traces']:,} traces) =====")
    print(f"{'quantity':22s} {'Acc%':>18s} {'MeanRank':>10s} {'MaxRank':>8s}   {'chance':>16s}")
    for name, s in summary.items():
        print(f"{name:22s} {s['acc_mean_pct']:>7.3f} avg "
              f"[{s['acc_min_pct']:.2f},{s['acc_max_pct']:.2f}] "
              f"{s['mean_rank_mean']:>10.3f} {s['max_rank_worst']:>8d}   "
              f"{s['chance_acc_pct']:6.3f}% / mr{s['chance_mean_rank']:.1f}")

    if not args.no_telegram:
        lines = [f"🏆 [ASCAD GPAM] Table 11 holdout eval done :: {label}",
                 f"{payload['n_traces']:,} never-seen traces | {elapsed / 60:.1f} min"]
        for name, s in summary.items():
            lines.append(
                f"  {name}: Acc {s['acc_mean_pct']:.2f}% "
                f"[{s['acc_min_pct']:.2f}-{s['acc_max_pct']:.2f}] | "
                f"MeanRank {s['mean_rank_mean']:.1f} | MaxRank {s['max_rank_worst']}"
                f"  (chance {s['chance_acc_pct']:.2f}%/{s['chance_mean_rank']:.1f})")
        send_message("\n".join(lines))


if __name__ == "__main__":
    main()
