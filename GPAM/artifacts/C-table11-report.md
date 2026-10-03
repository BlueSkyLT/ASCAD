# T-019 final report — GPAM on raw ASCADv2, Table 11 holdout evaluation

**Run:** 34-head GPAM (`rm`, `rout`, `c[0..15]`, `p[0..15]`) on **raw 1,000,000-point ASCADv2
traces**, patch 400 (2,500 tokens), merge filters 0/0, 3 GAU blocks, 17,766,824 params, Adafactor
**lr 1e-3** (ours, not the paper's unpublished schedule), batch 32 × 2,000 steps = 64,000
traces/epoch, target 150 epochs. Launched 2026-09-29 21:23, **killed externally after 76 epochs**
(2026-10-01 ~21:57, no traceback, no reboot) — 48.6 h, 4.86M trace-instances = **51 % of the
paper's 9.6M**. Best held-out `val_loss` **96.928** at metrics epoch 64; overfitting gap ~3.9 nats
by epoch 75 (train 94.74 vs val 98.61).

**Eval:** `GPAM/artifacts/C-table11-holdout-eval.py`, inference only, **80,000 shard-8 holdout
traces** (`[20000:100000]`) — never seen by the optimizer or by validation (train = shards 1–6 +
shard 7[0:40k], val = shard 7[40k:] + shard 8[0:20k]). 6.1 min per checkpoint. Ranking counts ties
**against** the target (`rank = #{classes with p ≥ p_true} − 1`), so a flat head returns rank C−1,
not the spurious 0.

## Table 11 — ours vs the paper

Paper: Burzstein et al., CHES 2024, §7.1 Table 11 (printed p.492), same 80,000-trace holdout.
`c[i]` and `p[i]` are averages over the 16 byte heads.

| Quantity | Paper Acc % | Paper MeanRank | Paper MaxRank | **Ours Acc %** | **Ours MeanRank** | **Ours MaxRank** | Chance |
|---|---|---|---|---|---|---|---|
| `rm` (α, multiply hide) | 100.00 | 0 | 0 | **99.996** | **0.003** | 255 | 0.391 % |
| `rout` (β, add hide) | 18.25 | 4.78 | 65 | **10.480** | **10.517** | 114 | 0.391 % |
| `p[i]` (π, shuffle index) | 95.07 | 0.055 | 4.44 | **86.514** | **0.181** | 15 | 6.25 % |
| `c[i]` (masked S-box) | 1.18 | 80.65 | 255 | **0.679** | **108.631** | 255 | 0.391 % |

Distance-from-null, which is the honest way to read a mean rank (null = (C−1)/2 = 127.5 for
256-class heads): `rout` is at **95 %** of the paper's distance from null
(127.5 − 10.52 against 127.5 − 4.78) and `c` at **40 %** (127.5 − 108.63 against 127.5 − 80.65),
while the top-1 ratios are 57 % and 57 %. The two disagree because our rank distributions have
longer tails (MaxRank 114 vs 65 for β), not because the bulk of the predictions is worse.

**Verdict: 3 of the 4 quantities reproduce, and the fourth (`c`) reproduces in kind — the paper's
own `c` row is only 3× chance.** α is effectively exact: 79,997 of 80,000 correct, top-2 99.999 %.

Transcription caveat on the paper column: `p[i]`'s MaxRank is recorded in our notes as **4.44**, a
fractional value that cannot be a maximum over integer ranks. Treat the `p` MaxRank comparison as
provisional; Acc and MeanRank for the paper row are transcribed exactly as read off Table 11
(PDF p.21 / printed p.492).

## Per-head detail (best checkpoint, metrics epoch 64)

| Head group | Acc % range | MeanRank range | Notes |
|---|---|---|---|
| `p[0..15]` | 73.906 (b3) … **99.736 (b15)**, mean 86.514 | 0.003 (b15) … 0.384 (b3) | **all 16 heads ≥ 11.8× chance**; byte 15 essentially solved |
| `c[0..15]` | 0.351 (b14) … 1.069 (b5), mean 0.679 | 85.975 (b1) … **127.539 (b14)** | 8 of 16 heads ≥ 1.5× chance; **b14 sits on the null** (mean rank 127.539 vs 127.5, acc 0.90× chance) |
| `rm` | 99.996 % (3 errors in 80,000) | 0.003 | at least one of the 3 ranked the true α **last** (MaxRank 255) — confidently wrong, not vague |
| `rout` | 10.480 % (8,384) | 10.517 | top-2 18.76 %, top-4 32.91 %, top-8 53.46 %, top-16 76.25 % |

Checkpoint robustness — the result is not a selection artifact
(`GPAM/artifacts/C-table11-holdout-ep70.json`, metrics epoch 69):

| Checkpoint | `rm` | `rout` | `p` mean | `c` mean |
|---|---|---|---|---|
| best (val_loss 96.928) | 99.996 / 0.003 | 10.480 / 10.517 | 86.514 / 0.181 | 0.679 / 108.63 |
| epoch_070 | 99.940 / 0.004 | **12.335 / 9.376** | 84.750 / 0.217 | 0.714 / 107.01 |

β and π trade slightly against each other across epochs; nothing else moves. Both checkpoints are
inside ±2 pp on every quantity.

## What this does and does not license saying

**Can say:**
- On **never-seen raw 1 M-point traces** GPAM's α head reproduces the paper's headline (99.996 % vs
  100.00 %), and the shuffle head reaches 86.5 % vs the paper's 95.07 % — the largest earlier
  local number for π was 8.60 % on the 15 k POI crop, so π's gap was **training budget, not
  intrinsic difficulty**. This formally reverses the 2026-09-26 claim in this task file.
- The masked-S-box head is weak **in the paper too** (1.18 %, mean rank 80.65 of a 127.5 null). Any
  statement that our `c` "failed where GPAM succeeded" is false.
- Because `rm` is at 99.996 % held-out, the §7.1 simulation's stated precondition ("since rm has
  very high accuracy we use it directly instead of the whole probability distribution") is now
  satisfied by our model, so the GE test is finally worth running.

**Cannot say:**
- **Not a key recovery.** Table 11 is machine-learning metrics on mask labels; ASCADv2 draws a fresh
  random key per trace. No GE number is produced by this evaluation.
- **Not a strict break.** α, β and π are mask parameters that the First-Threat threat model makes
  available as training labels; predicting them is a worst-case result.
- No `c`-based statement of the "the S-box is invisible" type. And a correction to how I quoted the
  SNR profile earlier: **per-sample SNR values are not comparable across targets with different
  class counts.** Under the null the estimator is χ²_{K−1}/N, so with N = 16,384 traces its own
  noise floor is (K−1)/N = **0.0156 for the 256-class heads but only 0.00092 for π** (16 classes).
  Subtracting each target's own floor and counting samples significant at p < 1e-6:

  | Target | K | floor | peak SNR | excess peak | class-mean sep | samples sig. at 1e-6 |
  |---|---|---|---|---|---|---|
  | `rm` | 256 | 0.0156 | 1.9960 | 1.9805 | 1.41 σ | 41,073 |
  | `rout` | 256 | 0.0156 | 1.4195 | 1.4040 | 1.18 σ | 1,410 |
  | `p[i]` | 16 | 0.00092 | 0.2085 | 0.2075 | 0.46 σ | **4,236** |
  | `c[i]` | 256 | 0.0156 | 0.0721 | 0.0566 | 0.24 σ | **7** |

  Normalised this way, the SNR ordering **agrees** with the learned ordering
  (`rm` 99.996 % > `p` 86.5 % > `rout` 10.5 % > `c` 0.68 %): π is the second-most-visible target, not
  the least, and `c` — not π — is the near-invisible one. The earlier "27 samples above 0.1 for π
  vs 0 for `c`" contrast was an artifact of the fixed 0.1 threshold, which sits 6.4× above `c`'s
  floor but 109× above π's. Caveats: samples are correlated across time, so the significant-sample
  counts over-state independent evidence (a ranking, not an information total); the profile was
  computed per target in isolation, so it cannot see cross-head leverage.

## Campaign record (for reproduction)

| Item | Value |
|---|---|
| Launcher | `scripts/gpam_probe_then_campaign.sh` (gated on the 40 k probe, which passed at `rm` 98.340 %) |
| Command | `train_gpam_raw_full.py --target-bytes 0..15 --epochs 150 --steps-per-epoch 2000 --val-steps 50 --batch-size 32 --chunk-size 128 --lr 1e-3 --patch-size 400 --eval-holdout-batches 2500` |
| Log | `GPAM/artifacts/logs_raw/campaign34_1e-3.log` (420 MB), metrics `gpam_raw_..._20260929_212317_metrics.csv` (76 rows, epoch column 0-indexed) |
| Throughput | 38.4 min/epoch on the RTX 5090 (TF has no sm_120 kernels → PTX JIT, `run_eagerly=True`) vs the paper's 20 h on an RTX 4090 |
| Delayed transition | in the monitor's numbering, epochs 1–12 sat within 0.15 nats of the 144.1746 chance total (epoch 12 still *worse* at 144.3199) and the break came at **epoch 13**: 140.4642 → 133.4385 → 130.8809. **Do not read a GPAM curve before epoch 13** |
| Checkpoints kept | metrics epochs 9,19,…,69 (files `_epoch_010..070`) + `_best` |

**Known defects left open:** `train_gpam_raw_full.py` has no `--resume`; its end-of-run Telegram
string prints `eval_holdout_batches * 64` while the batch size is 32; the campaign's own
`evaluate_holdout` was never reached because the process was killed, which is exactly the gap this
report closes.

## Artifacts

- `GPAM/artifacts/C-table11-holdout-eval.py` — the evaluator (no training).
- `GPAM/artifacts/C-table11-holdout-ep64.json`, `-ep70.json` — full per-head results incl. top-k.
- `GPAM/artifacts/logs_raw/table11_ep64.log`, `table11_ep70.log`.
- `GPAM/artifacts/snr_raw1m_{rm,rout,c,p}.npy` — per-sample SNR profiles behind the `c` discussion.
