#!/usr/bin/env python3
"""Can GPAM learn ANYTHING from raw ASCADv2 traces? A memorization test.

The full run sat at exactly chance for 33 epochs on 640k traces. Label streaming
is proven correct, so this isolates the two remaining suspects by trying to
overfit 512 traces (a set small enough that any working pipeline should drive
training loss well below chance):

  --norm fixed : inputs min=-128 delta=255, exactly what train_gpam_raw_full.py used
  --norm data  : min/delta measured from the training batch (what the extracted-crop
                 run that reached rm 93% did)
  --lr X       : the dead run used 5e-5, which is Table 1's *target* learning rate
                 applied as a constant; the crop run that worked used 1e-3

Model construction mirrors build_gpam_raw() in train_gpam_raw_full.py.
"""
import argparse
import os
import sys
import time

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
from tensorflow import keras  # noqa: E402

sys.path.insert(0, "GPAM")
from raw_dataset import RawMultiShardLoader  # noqa: E402
from scaaml.models import get_gpam_model  # noqa: E402

HEADS_4 = ["rm", "rout", "c_0", "p_0"]


def heads_for(all_bytes):
    if not all_bytes:
        return list(HEADS_4)
    hs = ["rm", "rout"]
    for b in range(16):
        hs += [f"c_{b}", f"p_{b}"]
    return hs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--norm", choices=["fixed", "data"], default="fixed")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--n-traces", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--patch-size", type=int, default=400)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--warmup-frac", type=float, default=0.0,
                    help="fraction of steps to ramp LR up to --lr (paper 5.3.1 schedule)")
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--all-bytes", action="store_true",
                    help="34 heads (rm, rout, c[0..15], p[0..15]) as in the paper, "
                         "instead of the 4-head byte-0 pilot")
    ap.add_argument("--shuffle-labels", action="store_true",
                    help="randomise every label; the rote-memorisation control")
    a = ap.parse_args()

    HEADS = heads_for(a.all_bytes)
    NCLASSES = {h: (16 if h.startswith("p_") else 256) for h in HEADS}
    tbytes = list(range(16)) if a.all_bytes else [0]

    t0 = time.time()
    loader = RawMultiShardLoader(split="train", target_bytes=tbytes,
                                 chunk_size=a.n_traces, batch_size=a.batch_size,
                                 shuffle=False)
    X, Y = [], {h: [] for h in HEADS}
    for x, y in loader.generator():
        X.append(x)
        for h in HEADS:
            Y[h].append(y[h])
        if sum(len(b) for b in X) >= a.n_traces:
            break
    X = np.concatenate(X)[: a.n_traces].astype(np.float32)
    Y = {h: np.concatenate(Y[h])[: a.n_traces] for h in HEADS}
    if a.shuffle_labels:
        # Destroys every trace->label relation while keeping each head's marginal
        # identical. Anything memorised here is rote fitting, not leakage -- this
        # is the control that makes the other numbers interpretable.
        rl = np.random.default_rng(1234)
        Y = {k: rl.permutation(v) for k, v in Y.items()}
        print("LABELS SHUFFLED: no trace carries its own label; accuracy above "
              "chance here would be pure memorisation of noise", flush=True)
    print(f"[{time.time()-t0:5.1f}s] memorization set: X{X.shape} "
          f"range=[{X.min():.0f},{X.max():.0f}] mean={X.mean():.2f} std={X.std():.2f}", flush=True)

    if a.norm == "fixed":
        tmin, tdelta = -128.0, 255.0
    else:
        tmin = float(X.min())
        tdelta = float(X.max() - tmin)
    print(f"norm={a.norm}: min={tmin} delta={tdelta}  "
          f"(NB: int8 traces fill [-128,127], so both branches coincide here)", flush=True)

    outputs = {h: {"max_val": NCLASSES[h]} for h in HEADS}
    model = get_gpam_model(inputs={"trace1": {"min": tmin, "delta": tdelta}},
                           outputs=outputs, output_relations=[], trace_len=1_000_000,
                           merge_filter_1=0, merge_filter_2=0, patch_size=a.patch_size)
    if a.warmup_frac > 0:
        # The mechanism paper 5.3.1 describes but never parameterises: start low,
        # ramp steadily to the target, then cosine-decay as the model converges.
        warm = max(1, int(a.steps * a.warmup_frac))
        sched = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=a.lr * 0.1, decay_steps=max(1, a.steps - warm),
            alpha=0.05, warmup_target=a.lr, warmup_steps=warm)
        optimizer = keras.optimizers.Adafactor(sched)
        lr_desc = f"{a.lr:g} warmup {a.warmup_frac:.0%} then cosine"
    else:
        optimizer = keras.optimizers.Adafactor(learning_rate=a.lr)
        lr_desc = f"{a.lr:g} constant"
    model.compile(optimizer=optimizer,
                  loss={h: "sparse_categorical_crossentropy" for h in HEADS},
                  metrics={h: ["acc"] for h in HEADS}, run_eagerly=True)
    nparam = model.count_params()

    # fit() records metrics per EPOCH, so make one epoch = one pass over the
    # memorization set; that yields a real trajectory instead of one mean.
    ds = tf.data.Dataset.from_tensor_slices((X, Y)).repeat().batch(a.batch_size)
    spe = max(1, a.n_traces // a.batch_size)
    n_epochs = max(1, a.steps // spe)
    hist = model.fit(ds, steps_per_epoch=spe, epochs=n_epochs, verbose=0,
                     callbacks=keras.callbacks.TerminateOnNaN())
    h = hist.history
    chance = {k: float(np.log(v)) for k, v in NCLASSES.items()}
    print(f"\n=== norm={a.norm} lr={lr_desc} params={nparam:,} "
          f"{n_epochs} epochs x {spe} steps = {n_epochs*spe} steps ===")
    print(f"{'epoch':>6} {'steps':>6} {'rm_loss':>8} {'rm_acc':>8} {'rout_loss':>9} "
          f"{'c_0_loss':>9} {'p_0_loss':>8} {'total':>8}")
    every = max(1, min(a.log_every, n_epochs // 12 or 1))
    marks = sorted(set(list(range(1, n_epochs + 1, every)) + [n_epochs]))
    for e in marks:
        i = e - 1
        print(f"{e:>6} {e*spe:>6} {h['rm_loss'][i]:>8.4f} {h['rm_acc'][i]:>8.4f} "
              f"{h['rout_loss'][i]:>9.4f} {h['c_0_loss'][i]:>9.4f} {h['p_0_loss'][i]:>8.4f} "
              f"{h['loss'][i]:>8.4f}")
    print(f"chance CE: rm/rout/c_0 {chance['rm']:.4f}  p_0 {chance['p_0']:.4f}  "
          f"total {sum(chance.values()):.4f}; chance acc rm {1/256:.5f} p_0 {1/16:.4f}")
    print(f"MEMORISE: final rm_acc {h['rm_acc'][-1]*100:.1f}% and final total loss "
          f"{h['loss'][-1]:.4f} on only {a.n_traces} traces -- a working pipeline "
          f"should approach 100%/0 here")

    if a.all_bytes:
        def fin(key):
            return h[key][-1]
        c_acc = np.mean([fin(f"c_{b}_acc") for b in range(16)])
        p_acc = np.mean([fin(f"p_{b}_acc") for b in range(16)])
        c_loss = np.mean([fin(f"c_{b}_loss") for b in range(16)])
        p_loss = np.mean([fin(f"p_{b}_loss") for b in range(16)])
        print("\n=== 34-head aggregate vs paper Table 11 (held-out values there; "
              "here train=eval memorisation, so NOT comparable as an accuracy) ===")
        print(f"{'kind':6s} {'ours acc':>9s} {'ours mean CE':>13s} {'chance acc':>11s} "
              f"{'paper acc':>10s} {'paper meanrank':>15s}")
        print(f"{'rm':6s} {fin('rm_acc')*100:>8.2f}% {fin('rm_loss'):>13.4f} {100/256:>10.3f}% "
              f"{'100.00':>10s} {'0':>15s}")
        print(f"{'rout':6s} {fin('rout_acc')*100:>8.2f}% {fin('rout_loss'):>13.4f} "
              f"{100/256:>10.3f}% {'18.25':>10s} {'4.78':>15s}")
        print(f"{'c[i]':6s} {c_acc*100:>8.2f}% {c_loss:>13.4f} {100/256:>10.3f}% "
              f"{'1.18':>10s} {'80.65':>15s}")
        print(f"{'p[i]':6s} {p_acc*100:>8.2f}% {p_loss:>13.4f} {100/16:>10.3f}% "
              f"{'95.07':>10s} {'0.055':>15s}")
        print(f"per-p-head acc: " + " ".join(f"{fin(f'p_{b}_acc')*100:.1f}" for b in range(16)))
        print(f"per-c-head acc: " + " ".join(f"{fin(f'c_{b}_acc')*100:.1f}" for b in range(16)))


if __name__ == "__main__":
    main()
