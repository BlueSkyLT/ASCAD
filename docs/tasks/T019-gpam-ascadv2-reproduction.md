# T-019: GPAM ASCADv2 reproduction (paper-faithful)

## Status

**2026-09-30 ~02:00 — IN FLIGHT, epoch 14/150: the 34-head raw campaign is generalising across
shards.** Held-out `val_loss` went 144.3199 (ep12, *worse* than chance) → 140.4642 (ep13) →
**133.4385** (ep14); gap vs the 144.1746 chance total: −0.1453 → +3.7104 → **+10.7361** in three
epochs. Train-side per-head at epoch 14: `rm`(α) **82.30 %**, `rout`(β) 2.18 % (5.6× chance),
`p_0`(π) 15.96 % (2.6× chance), `c_0`(masked S) 0.46 % — still flat. **π moving is new**; the
crop run had `p` at 8.60 % held-out and `T019:45` called the shuffle head "the hard one, not
merely undertrained." Not yet citable — mid-run, and the final held-out eval is the number to
record. Log: `GPAM/artifacts/logs_raw/campaign34_1e-3.log`.

**Per-head HELD-OUT metrics, from the metrics CSV**
(`gpam_raw_b0_..._20260929_212317_metrics.csv`; note the CSV `epoch` column is **0-indexed**, so
CSV epoch 14 is the monitor's "epoch 15"):

| CSV epoch | `val_rm` | `val_rout` | `val_p_0` | `val_c_0` | `val_loss` |
|---|---|---|---|---|---|
| 9 | 0.63 % | 0.31 % | 5.94 % | 0.31 % | 144.1792 |
| 10 | 0.44 % | 0.50 % | 5.87 % | 0.44 % | 144.1739 |
| 11 | 0.12 % | 0.50 % | 5.31 % | 0.19 % | 144.3199 |
| 12 | **18.63 %** | 2.19 % | 6.44 % | 0.63 % | 140.4642 |
| 13 | **91.00 %** | 2.69 % | 17.50 % | 0.31 % | 133.4385 |
| 14 | 78.31 % | 1.56 % | **24.25 %** | 0.44 % | 130.8809 |
| 15 | 77.94 % | 1.19 % | 23.44 % | 0.37 % | 132.3935 |
| 16 | **93.94 %** | 2.69 % | **31.56 %** | 0.31 % | 126.4043 |

**At CSV epoch 16 the raw run overtakes the crop run on both interesting heads**: `val_rm`
93.94 % vs the crop's 92.95 %, and `val_p_0` 31.56 % vs the crop's 8.60 % — 5.05× chance on the
head `T019:45` declared intrinsically hard. The epoch-15/16 dip in the total (+13.29 → +11.78 →
+17.77) came from the still-unlearned `c_b` heads oscillating, **not** from `rm` or `p_0`
degrading, which is why the aggregate gap is a poor tripwire here.

### CSV epoch 19 (monitor epoch 20) — the dissociation is now clean

`val_loss` 120.7839, gap **+23.3908**. All 32 per-head held-out accuracies:

| head group | held-out acc | chance | verdict |
|---|---|---|---|
| `rm` (α multiply mask) | **98.19 %** | 0.391 % | 251× — at the 40 k probe's 98.34 % |
| `rout` (β) | 3.31 % | 0.391 % | 8.5×, weak but real |
| `p_0 … p_15` (shuffle index, 16 heads) | **23.94 – 66.62 %, mean 35.2 %** | 6.25 % | **all 16 learning**, 3.8–10.7× chance |
| `c_0 … c_15` (masked S-box, 16 heads) | 0.06 – 0.81 %, **mean 0.397 %** | 0.391 % | **indistinguishable from chance, all 16** |

**This reverses `T019:45`.** That entry concluded "the shuffle head is the hard one, not merely
undertrained" from `p` barely moving in every earlier configuration. At 34 heads on raw 1 M
traces, **all sixteen** permutation heads are held-out above chance while **none** of the sixteen
masked-S-box heads is — mean 0.397 % against a 0.391 % bar, i.e. exactly guessing. `p` was
undertrained, not hard; the genuinely hard target is the masked S-box, which is the same
asymmetry brief A found from the opposite direction.

Still epoch 20/150 and climbing; the citable artifact remains the campaign's final holdout eval.

### Epochs 21–22 — α effectively solved, π still climbing, C immovable

| CSV epoch | `val_rm` | `val_rout` | `val_p` mean (16 heads) | `val_c` mean (16 heads) | `val_loss` |
|---|---|---|---|---|---|
| 19 | 98.81 % | 3.62 % | 34.97 % (28.0–66.9) | 0.434 % | 121.1434 |
| 21 | **99.81 %** | 3.75 % | **41.16 %** (27.9–75.6) | **0.367 %** | 116.5875 |
| 22 | 99.94 % | 3.69 % | 38.50 % (30.1–51.2) | 0.449 % | 118.1294 |
| 23 | **100.00 %** | 4.25 % | **46.45 %** (37.8–70.2) | 0.375 % | 114.3156 |

Bars: 0.391 % (`rm`/`rout`/`c_b`), 6.25 % (`p_b`). The epoch-22 total improvement (+4.56) came
from π's +6.2 pp across 16 heads plus α's last 1 pp — α had no headroom left, which is why the
aggregate had to move elsewhere. **α is now at 99.81 % held out on shard 2 from raw 1 M-point
traces.** π is 6.6× chance on all sixteen bytes with none below 27.9 %. The masked S-box has not
moved in 22 epochs and its mean sits *at* the guessing rate — it is the run's negative control,
and its stillness is what makes the α/π movement credible rather than an artifact of the loss
budget being shared.

Bars: `rm`/`rout`/`c_b` chance 0.391 %, `p_b` chance 6.25 %.
**The permutation head is the finding.** `val_p_0` 24.25 % = **3.9× chance held out**, against
8.60 % for the crop run — so `T019:45`'s "the shuffle head is the hard one, not merely
undertrained" needs revisiting: on raw 1 M traces with 34 heads it is learning, and still
climbing at epoch 14. `val_rm` 91.00 % (epoch 13) is at parity with the crop run's 92.95 % while
reading 1 M unselected samples instead of 15 k POI-selected ones. β is weakly real (1.6–2.7 %,
4–7× chance). Masked-S-box heads remain at chance throughout, as expected.
`val_rm` oscillated 91.00 % → 78.31 % between epochs 13 and 14, so these are not yet stable
readings and the peak is not the last row.

**Process note:** epochs 1–11 sat within ±0.01 of chance and epoch 12 looked like divergence. It
was the pre-breakthrough dip — same shape the 40 k probe showed at its epoch 5 before jumping
0.29 % → 17.87 %. A worsening val epoch in this model class is not evidence of failure.

**2026-09-29 21:23 — RAW 1M-POINT GPAM GENERALISES. The 40k-trace probe PASSED, and it did so via a delayed phase transition.** Train 40k traces from shard 1, held-out eval 1,024 traces from **shard 2** (disjoint capture), 4 heads, lr 1e-3, 8 epochs (chance total 19.4081):

| epoch | train | val(shard 2) | val `rm` |
|---|---|---|---|
| 1-5 | 19.4154 -> 19.3623 | 19.4079 -> 19.4944 (at chance) | 0.293 -> 0.293% |
| 6 | 18.7306 | **17.0840** | **17.871%** |
| 7 | 15.5788 | 14.6365 | **76.270%** |
| 8 | 13.9707 | **13.4763** | **98.340%** |

Five dead epochs then a cliff. **This reinterprets every earlier "no learning" observation**: the 4k-trace test failed because it never reached the transition (too little data), and the lr 3e-3 full campaign degraded before it could. It is also a live demonstration of the weight-initialisation plateau that paper 5.3.1 says its warmup schedule exists to escape - we reproduced the plateau without the schedule, and the transition still came.
**Headline: `rm` 98.34% on never-seen raw 1M-point traces**, vs the paper's 100.00% - using 6% of their data and 5% of their epochs. Eval resolution is 1/1024 = 0.098%.
**34-head campaign relaunched automatically at 21:23 on the gate passing**, lr 1e-3, batch 32, 2,000 steps/epoch x 150 epochs (~83 h). At 22:37 it is in epoch 2 with all heads at chance (val_loss 144.1826 vs chance 144.1746), which given the probe is the expected dead phase, not a failure: the analogue of probe epoch 6 arrives around epoch 6 here, ~3-4 h out.
**Outstanding gap in our own reporting:** the probe saved no checkpoint, so it produced accuracies but not Table 11's MeanRank/MaxRank; the campaign computes those only in its end-of-run holdout evaluation. Checkpoints land every 10 epochs, so a proper Acc|MeanRank|MaxRank triplet for the raw run is available after epoch 10 (~5.5 h) without waiting 83 h.


**2026-09-29 21:23 — SUPERSEDED AT 40 k TRACES: raw 1 M-point traces DO generalise.** The
matched-budget probe (`C-raw-generalization-test.py`, 40 k train from shard 1, held-out eval on
shard 2 rows 0–1023, 4 heads, 8 epochs, lr 1e-3) reached **`val rm` 98.340 %** against a 0.391 %
chance bar, val loss **13.4763** vs chance 19.4081, and printed
`PROBE GATE VERDICT: PASSES` (`GPAM/artifacts/logs_raw/probe40k_20260929_202123.log`). The
~83 h 34-head raw campaign auto-launched at 21:23:16 as pid 1083690.
**The budget, not the data, was the binding constraint**: the 19:00 finding below was correct at
4 k traces (0.6 % of the paper's 640 k) and simply had not reached the regime where transfer
appears. Only `rm` learned — `rout` 0.879 %, `c_0` 0.391 %, `p_0` 6.84 % (chance 6.25 %) stay at
chance, the same α-only hierarchy as the crop run, now better (98.34 % vs 92.95 %). α is a mask
label, so this is **worst-case / First Threat, not a strict break**, and Table 11's "~80 traces"
claim is still unreproduced. Controls: 0 overlap of `(key, plaintext, 38-byte masks)` signatures
across the exact train/eval slices; frequency prior caps a trace-ignoring model at ~0.48 %.
**Process lesson: do not read a GPAM curve before epoch 6** — this one looked dead for five
epochs (`val rm` 0.098–0.488 %, val gap −0.0863 at epoch 5) then jumped 37× in one epoch.

**2026-09-29 19:00 — raw 1M-point traces DO NOT GENERALISE at 4k traces (`C-raw-generalization-test.py`, train shard 1 -> held-out eval shard 2, disjoint captures):**

| lr | train loss ep1 -> ep10 | shard-2 val loss ep1 -> ep10 | held-out `rm` acc |
|---|---|---|---|
| 1e-3 | 19.4323 -> **19.0323** | 19.4077 -> **19.6892** | 0.879% -> 0.195% (chance 0.391%) |
| 3e-3 | 19.4195 -> **17.6454** | 19.4088 -> **20.6235** | 0.684% -> 0.684% |

Textbook overfitting: the model fits shard 1 hard (train falls well below chance) and transfers **nothing** to shard 2, where val loss ends *above* chance. **Do not over-read this as "raw is unlearnable"**: 4,000 traces is 0.6% of the paper's 640,000, and needing that much data is precisely why the paper crops nothing. The crop run reached rm 92.95% with 40k traces because its 15,000 samples are pre-selected informative points, so the signal is not diluted across 1M samples.
**Reading across all three LR experiments: 5e-5 inert, 3e-3 unstable (diverges on val), 1e-3 the only regime that fits train smoothly.** So the full campaign should run at **1e-3**, not 3e-3.
**Sequenced plan now running:** matched-budget probe (40k raw traces, 8 epochs, 4 heads, lr 1e-3, held-out eval) - the identical budget to the crop run that hit 92.95%, differing only in input length, ~1.3 h at 0.44 s/step. If held-out `rm` clears 2% (5x chance), launch the 34-head 150-epoch campaign at lr 1e-3 automatically; if not, the raw campaign is unjustified and the crop result is our answer.


**2026-09-29 18:08 — the 34-head raw campaign at lr 3e-3 was STOPPED after 3 epochs (~1.6 h) because it was diverging, not converging.** Logged train/val totals (chance = 144.1746):

| epoch | train loss | val loss | val gap |
|---|---|---|---|
| 1 | 144.1939 | 144.1985 | -0.024 |
| 2 | 144.1835 | 144.4070 | -0.232 |
| 3 | 144.1817 | **144.8397** | **-0.665** |

Train stayed pinned at chance while val degraded monotonically and super-linearly (gap ~x3 per epoch). Two corrections to notes earlier in this file: (a) the "monotone decline" seen during epoch 1 was a **running-average artifact** - reconstructing instantaneous loss from successive running means gives a flat 144.21 -> 144.20 across 1,750 steps, so it was never evidence of learning; (b) my "let it earn 20 epochs, maybe warmup" suggestion is refuted by epochs 2-3.
**So lr 3e-3 is a second, distinct failure mode**: 5e-5 was inert (could not even memorise 512 traces), 3e-3 fits nothing and actively degrades held-out loss. Learning rate alone is therefore not the remaining lever. Throughput also measured: **1 s/step at batch 32 with 34 heads**, so the paper's 150k-step schedule is ~82 h here (TF has no sm_120 Blackwell kernels and JITs from PTX, plus run_eagerly=True) versus the 20 h on an RTX 4090 the paper reports.
**Open question that the probes have NOT answered: does ANY configuration generalise on raw 1M-point traces?** Every positive number so far is either memorisation (train=eval) or comes from the 15k POI crop. The decisive cheap test is train on ~4k raw traces / evaluate on a disjoint ~2k raw traces at a few LRs - ~20 min per config. Until that passes, relaunching a multi-day campaign is unjustified.


**2026-09-29 17:00 — campaign running but NOT learning at scale; ETA corrected to ~82 h.** Epoch 1 finished in ~33 min (1 s/step at batch 32 with 34 heads, not the 378 ms/step measured with 4 heads), so 150 epochs is ~3.4 days, not the 31 h first estimated. Running-mean total loss over epoch 1: 144.2570 -> 144.2273 -> 144.2152 -> 144.2095 -> **144.1939** vs chance **144.1746** - declining, but still above chance after 2,000 steps (64,000 trace-instances). Per head (chance 5.5452 / 2.7726): `rm` 5.5570 -> 5.5444 (just crossed), `rout` 5.5426 -> 5.5460, `c_0` 5.5405 -> 5.5460, `p_0` 2.7787 -> 2.7730. Epoch 2 had drifted slightly back up by step 163 (144.1793 -> 144.1860).
**This is the opposite of the 512-trace probe**, where lr 3e-3 drove `rm` from 5.55 to 4.29 within 240 steps. The difference is memorising 512 repeated traces vs seeing 640k distinct ones, so the probes established that the pipeline *can* fit, not that GPAM learns generalisable signal here. **Decisive missing test: train on a few thousand raw traces at 3e-3 and evaluate on OTHER raw traces** - that separates "learns signal that transfers" from "memorises only". Needs the GPU, so it means stopping this campaign (~25 h wasted so far rather than 82).

---
**2026-09-29 16:12 — the paper-faithful raw campaign is RUNNING.** `gpam_raw_b0_..._15_20260929_161220`: 34 heads (rm, rout, c[0..15], p[0..15]) on 1,000,000-point ASCADv2 raw traces, patch 400, batch 32, lr 3e-3, 150 epochs x 2,000 steps = 9.6M trace-instances, matching the paper's data exposure. 17,766,824 params. Launched with `setsid` so it survives session exit (the 09-25 run was killed by SIGHUP). Log `GPAM/artifacts/logs_raw/campaign34.log`; per-epoch `[chance-check]` against 144.1746; final Table 11 evaluation on all 80,000 holdout traces. Batch 64 was attempted first and **OOMs** because scaaml's GAU builds a 2500x2500 RoPE Toeplitz matrix per call (30.4/32 GiB at batch 32).


`active` (Phase 1/2 in progress: paper-faithful training running on RTX 5090 across all 8 raw shards)

## Result snapshot

- **2026-09-26: the full raw run's at-chance stall is explained by learning rate, not by the data pipeline.** The run died at epoch 33/150 on 2026-09-25 20:56 (killed, no traceback; no reboot, GPU idle since). Diagnosis:
  - `GPAM/raw_dataset.py` is **exonerated by ground truth** — labels derived from raw shard metadata match the extracted file's *stored* labels on all **40,000** traces joinable by `(key, plaintext, masks)`; the argsort/unsort path reconstructs traces bit-exactly; int8 genuinely fills [-128, 127] so `min=-128, delta=255` is right, and there is no unsigned wrap (signed neighbour steps 13.49 vs 34.31).
  - Memorization probe (`GPAM/artifacts/C-raw-overfit-probe.py`, 512 raw traces, 1,488 steps): at **lr 5e-5** total loss 19.4479 → **19.3071**, `rm` acc → 1.0% — it cannot memorize 512 traces, i.e. a hard stall. At **lr 1e-3** it descends monotonically and is still falling: 19.4429 → **9.5409**, `rm` acc 0 → **29.3%**. Chance CE total is 19.4081.
  - Warmup-then-cosine at the same peak LR was **worse at this budget** (→ 13.13, `rm` 13.3%), because the cosine decay removes LR inside a 1.5k-step run; §5.3.1's schedule is meant for their 150k steps, so this is not evidence against it at scale.
  - The `p` head barely moves in every configuration (2.779 → 2.590 at 1e-3) while `rm` moves 5.563 → 2.291 — consistent with the extracted-crop result where `rm` hit 92.95% and `p` only 8.6%. The shuffle head is the hard one, not merely undertrained.
  - **Why it went unnoticed for three hours:** the live status note recorded "train loss dropped below 19.410, validation stable at 19.4094" as progress. The 4-head chance sum is **19.4081**, so 19.4094 is still *worse than chance*. That band is indistinguishable from slow learning unless chance CE is printed alongside — worth making the trainer print it.
  - Relaunch implication: use our own LR (~1e-3, our optimizer choice; the paper never names one) rather than Table 1's 5e-5, which is a warmup *peak* from an unrepublished schedule. Cost note: 378 ms/step at batch 32 eager ⇒ batch 64 × 1,000 steps × 150 epochs is nearer **25–30 GPU-hours**, not 20.
  - **Full LR curve on the same 512-trace probe** (final total loss / `rm` acc, chance 19.4081): 5e-5 → 19.3071 / 1.0% (stalled); 1e-3 → 9.5409 / 29.3% (monotone); **3e-3 → 5.0860 / 75.4% (monotone, clean)**; 1e-2 → 4.5418 / 76.2% (fastest early but visibly unstable: 5.74→5.96 between epochs 64–71). So Table 1's 5e-5 is ~60x below where this optimizer+architecture begins to learn. Use **1e-3–3e-3**, not 1e-2. At batch 64 their stated rule (double batch ⇒ double target LR) puts the band near 2e-3–6e-3.
  - Caveat: this is memorization on 512 traces, so it proves learnability and bounds the usable LR range — it does **not** establish the best LR for 640k traces, where too-high LR may cost generalization. Validate on the 80k split. Also note `p_0` improves far less than `rm` (2.773 → 2.324 at 3e-3, ~16%, vs `rm` 5.545 → 0.896, ~84%): the shuffle head is slower-moving but it *does* move, so the earlier crop result (`p` 8.6%) was partly undertraining, not only an intrinsic limit.
  - **Pre-flight gate going forward:** any new raw configuration gets the 10-minute `C-raw-overfit-probe.py` run first. It distinguishes stalled from slow in a single digit of the time a full epoch costs.
- **2026-09-28: 34-head raw memorisation probe (`--all-bytes`), 512 traces x 1,488 steps, train=eval.** The paper's 34 heads are rm + rout + c[0..15] + p[0..15]; everything reported up to now was a 4-head byte-0 pilot, so these are the first numbers from the real head set (17,766,824 params).

  | Head | lr 1e-3 acc | lr 3e-3 acc | Paper (80k holdout) | Chance |
  |---|---|---|---|---|
  | `rm` | 32.8% | **94.3%** | 100.00% | 0.39% |
  | `rout` | 38.5% | **96.1%** | 18.25% | 0.39% |
  | `c[i]` mean of 16 | 36.0% | **94.5%** (every byte 92-97%) | 1.18% | 0.39% |
  | `p[i]` mean of 16 | 15.7% | **27.8%** | 95.07% | 6.25% |

  - **lr 3e-3 roughly doubles every head vs 1e-3 at 34 heads**, so the usable LR band rises with model width; the 4-head ranking understated it.
  - **A stated prediction failed:** 16 mutually-informative shuffle heads did NOT make pi learnable. It improved (CE 2.32 -> 2.15) but is the only head far from fitting, while the other three hit 94-96%.
  - **Open interpretation risk, being tested by a label-shuffled control:** with 17.8M params, fitting 512 traces may be rote memorisation and prove nothing about leakage. Note `c`/`rm` have 256 classes yet memorise at 94% while `p` has only 16 classes and does not -- fewer classes should be easier -- so `p`'s gap looks like an optimisation or head-wiring limit, not a statement about the shuffle. Run pending: shuffled-labels at 3e-3, and lr 1e-2 to test whether `p` is LR-limited.
  - **These 34-head numbers are NOT evidence about leakage — the shuffled-label control killed them.** At lr 3e-3 with labels randomised (no trace carrying its own label): `rm` **98.05%**, `rout` 96.29%, `c[i]` 95.69%, `p[i]` 29.43% — i.e. **as good as or better than the real labels** (94.34 / 96.09 / 94.48 / 27.84). A 17.8M-param model fits 512 random labels essentially perfectly, so all four "memorisation" figures measure fitting capacity only. The `p` shortfall is likewise a probe artifact (identical under shuffled labels), **not** a statement about the shuffle; any earlier claim in this file that `p` is intrinsically hard, based on this probe, is withdrawn.
  - **What the probe still validly establishes** (direction is unambiguous, so these survive): at lr **5e-5 the model cannot even memorise 512 traces** (total 19.3071 vs chance 19.4081; `rm` 1.0%) — failing the easiest possible task means it is not learning at all, which fully explains 33 epochs at chance on 640k traces. And the **LR ordering is trustworthy** because both ends are memorisation: 5e-5 (stalled) < 1e-3 (rm 32.8%) < 1e-2 (rm 65.4%) < **3e-3 (rm 94.3%, best)**. So 3e-3 remains the right relaunch LR; the probes are a valid "can it learn at all" gate, never a leakage test.
  - **Consequence: the only reproduction-grade result we hold is the crop run's `rm` 92.95%, and it SURVIVES this — because it was measured on traces never trained on.** A shuffled-label control on a held-out set would score chance by construction and prove nothing, so the meaningful test was contamination, and `GPAM/artifacts/C-crop-holdout-audit.py` ran it: **0 of 2,000** holdout rows share a `(key, plaintext, masks)` fingerprint with the 40,000 training rows; **0** exact trace-vector matches in a 300x300 sample; alpha agrees positionally at 0.35% (independent). And the trace-free prior on alpha is **0.480%** (most frequent of 255 classes), so 92.95% is ~190x any frequency-only strategy. The headline stands.
  - **Standing rule for this task:** a probe where train = eval may only be used to ask *can it learn at all* (it correctly condemned lr 5e-5). Any number meant to say something about leakage must come from held-out data, with a contamination audit like the one above.
- **2026-09-25: Full raw ASCADv2 reproduction actively training on RTX 5090!**
  - Model: GPAM (1M trace len, patch size 400 = 2,500 tokens, 3 GAU blocks, Adafactor lr $5\times 10^{-5}$, `run_eagerly=True`).
  - Live progress (20:30 check, ~3 hours elapsed): Epoch 28 running at step ~591/1000. Epochs 0–27 completed cleanly. Train loss dropped below 19.410 (19.40989 at epoch 27); validation loss stable at 19.4094. Periodic checkpoints saved at epochs 10 and 20 (`gpam_raw_b0_20260925_173231_epoch_0{1,2}0.keras`).
  - Monitoring: Real-time `TelegramMonitorCallback` active; 3rd hourly report scheduled at Epoch 30 (~20:38) along with periodic checkpoint `..._epoch_030.keras`.
- **2026-09-25: Full raw ASCADv2 campaign restored!** All 8 shards (`ascadv2-stm32-conso-raw-traces1.h5` through `...8.h5`, 100 GB each, total 800,000 traces × 1,000,000 samples int8) verified on disk via `h5py`. Phase 0 blocker is cleared.
- **2026-09-24: Four-head local evaluation on extracted ASCADv2** (`40k` train, 8 epochs, patch 150):
  - `rm` ($\alpha$ mask, multiply hide): **92.95% top-1 accuracy on holdout / 92.35% on attack**, mean rank 0.083 (paper is 100%, chance is 0.39%). **Reproduces!**
  - `rout` ($\beta$ mask, add hide): 2.75% acc, mean rank 36.05 (paper is 18.25%, chance is 0.39%).
  - `c` (masked S-box byte 0): **0.65% acc**, mean rank **126.50** (paper is 1.18%, chance is 0.39%, null is 127.5). **Stuck at chance.**
  - `p` (permutation/shuffle index byte 0): **8.60% acc**, mean rank 6.51 (paper is 95.07%, chance is 6.25%).
- **2026-09-24: Shard-8 raw pipeline smoke PASSED** (32 train / 16 val, patch 400, 1 epoch in ~7s). Confirms raw $1,000,000$-point traces ingest and train cleanly with TensorFlow / scaaml in `py10`.
- **2026-09-24: Metric & GE Analysis:**
  - Evaluated key recovery under §7.1 fixed-key simulation ($pt \oplus key$, argmax $r_m$): at 80 traces, local rank is **231 of 255** (paper claims GE < 1). Rank oscillates across 2,000 traces (231 → 133 → 2 → 10 → 20).
  - Permuted-prediction control reached rank **10** (better than real 20), proving our 8-epoch model sits in the noise band.
  - Reason for failure: GPAM key recovery requires all heads to work; our $c$ head (masked S-box) and $p$ head (shuffle) were unlearned at 8 epochs.
  - **Immunity to Scoop's label frequency cheat:** GPAM predicts uniform masked variables ($c[i] = (\alpha \times S) \oplus \beta$). Because $\alpha$ and $\beta$ are uniform random, $c$ is cryptographically flat (no label prior to memorize). GPAM is a worst-case multi-head attack, not a label-prior cheat.
- **2026-09-21 to 2026-09-22:** Verified upstream `scaaml` GPAM model builds and trains on RTX 5090 in `py10`.

---

## Goal

Achieve a paper-faithful reproduction of GPAM (*"GPAM: A General Pre-trained Architecture for Multi-task Side-Channel Analysis"*, Burzstein et al., TCHES 2024) on the full raw ASCADv2 dataset (800,000 traces × 1,000,000 points), targeting Table 11 metrics and §7.1 key recovery.

---

## Context & Artifacts in `GPAM/`

All scripts and evaluation outputs are organized under [`GPAM/`](file:///home/lantian/ASCAD/GPAM/):
- **Core training scripts:**
  - [`GPAM/run_ascadv2_gpam_baseline.py`](file:///home/lantian/ASCAD/GPAM/run_ascadv2_gpam_baseline.py): Raw shard smoke and baseline runner.
  - [`GPAM/C-gpam-eval.py`](file:///home/lantian/ASCAD/GPAM/C-gpam-eval.py): 4-head GPAM trainer and Table 11 evaluator.
  - [`GPAM/C-gpam-ablate.py`](file:///home/lantian/ASCAD/GPAM/C-gpam-ablate.py): Threat model and mask knowledge ablation tester.
  - [`GPAM/train_gpam_ascadv2_budget.py`](file:///home/lantian/ASCAD/GPAM/train_gpam_ascadv2_budget.py): Budget runner (note: local `compute_ge` has known bugs documented below).
- **Artifacts & Reports in [`GPAM/artifacts/`](file:///home/lantian/ASCAD/GPAM/artifacts/):**
  - `C-gpam-eval.json` & `C-gpam-eval.log`: Results of the 40k-trace 4-head evaluation.
  - `C-gpam-eval.keras`: Trained 1.75M-param GPAM checkpoint.
  - `C-gpam-ablation.json`: Mask ablation numbers.
  - `C-gpam-smoke.log` & `C-gpam-smoke-raw2.log`: Raw shard 8 smoke execution logs.
  - `C-talking-points.md`: Full technical talking points from Brief C.
- **Dataset (Full Raw Campaign):**
  - Path: `STM32_AES_v2/ASCAD_data/ASCAD_databases/`
  - 8 shards: `ascadv2-stm32-conso-raw-traces1.h5` through `...8.h5` (100,000 traces × 1,000,000 points int8 each).

---

## Detailed Findings from Brief C (`C-gpam-and-scoop.md`)

### 1. What the GPAM Paper Actually Reports (Table 11 is NOT Guessing Entropy)
Table 11 (TCHES 2024, §7.1, p. 492) reports **machine learning metrics on the 80,000-trace holdout**, not key recovery:

| Head | Meaning | Paper Acc % | Paper Mean Rank | Paper Max Rank | Our Local Acc % (Holdout / Attack) | Our Mean Rank | Status |
|------|---------|-------------|-----------------|----------------|-------------------------------------|---------------|--------|
| `rm` | Multiply mask $\alpha$ (`masks[18]`) | **100.00%** | 0.00 | 0 | **92.95% / 92.35%** | **0.083** | **Reproduced!** |
| `rout` | Additive mask $\beta$ (`masks[17]`) | 18.25% | 4.78 | 65 | 2.75% / 2.75% | 36.05 | Below paper |
| `c[i]` | Masked S-box byte (avg over 16) | **1.18%** | 80.65 | 255 | **0.65% / 0.50%** | **126.50** | **At chance** (null is 127.5) |
| `p[i]` | Shuffle index (avg over 16) | 95.07% | 0.055 | 4.44 | 8.60% / 9.55% | 6.51 | Unlearned (null prior = 7.05%) |

- The paper's masked S-box accuracy is **1.18%** (only ~3× chance). GPAM's success is carried by predicting the **masks ($\alpha, \beta$) and shuffle ($\pi$)**, not by high-confidence S-box predictions.

### 2. The "~80 Traces" Key Recovery Claim
- The 80-trace key recovery is prose in §7.1, not a table.
- **Method:** Because ASCADv2 has per-trace random keys, GPAM simulates a fixed key by replacing $pt$ with $pt \oplus key$. Because $r_m$ is 100% accurate, they set $r_m$ by argmax and accumulate log-likelihoods over traces.
- In our local run, at 80 traces our key rank was **231 of 255** because $c$ and $p$ failed to learn in 8 epochs on the 15k crop.

### 3. Separation from Scoop's Label Frequency Cheat
- **Scoop (ePrint 2026/338, *"Is it Really Broken?"*):** Predicted unmasked $Z = \text{Sbox}(p \oplus k)$. Because $Z$ has a slight frequency bias on ASCADv2, Scoop's MLP memorized the marginal label distribution, achieving false key recovery after 250k–300k traces without reading traces.
- **GPAM is immune to this specific cheat:** GPAM predicts masked $c[i] = (\alpha \times S) \oplus \beta$. Since $\alpha$ and $\beta$ are uniform random, $c[i]$ has 8 full bits of entropy (flat distribution). Furthermore, GPAM reports key recovery in 80 traces, which is orders of magnitude too fast for a 1.1× marginal frequency bias to accumulate.
- **Threat Model:** GPAM operates in the **worst-case / First Threat** scenario (internal random masks and shuffle indices are known during training). It is not a black-box attack.

### 4. Bugs Identified in Local Repo Code
- `GPAM/train_gpam_ascadv2_budget.py` had a flawed `compute_ge`: scored unmasked S-box without dissecting masks, assumed a constant key across traces, and suffered from a tie-break bug where uniform probability vectors produced "rank 0".
- Correct dissection formula is implemented in `GPAM/C-gpam-eval.py` and `ASCAD_test_models.py:687-743`.

---

## Action Plan: Full GPAM Reproduction on Raw ASCADv2

Now that Phase 0 is complete (all 8 raw shards restored), we can proceed with the full raw reproduction.

### Step 1: Multi-Shard Streaming Data Pipeline (`tf.data.Dataset`)
- **Problem:** Full dataset is $800,000 \times 1,000,000$ bytes $\approx 800\text{ GB}$ (int8) or $3.2\text{ TB}$ (float32). It cannot be loaded into RAM.
- **Design:** Build a generator in `GPAM/raw_dataset.py` that streams chunks of traces from `ascadv2-stm32-conso-raw-traces{1..8}.h5`:
  - **Split configuration (matching Table 3):**
    - **Train (640,000 traces):** Shards 1–6 (600,000) + first 40,000 of Shard 7.
    - **Validation (80,000 traces):** Remaining 60,000 of Shard 7 + first 20,000 of Shard 8.
    - **Holdout (80,000 traces):** Remaining 80,000 of Shard 8.
  - **Dynamic label generation:** Vectorized extraction of $r_m = \text{masks}[18]$, $r_{\text{out}} = \text{masks}[17]$, $\pi[i] = \text{perm\_indices}(i, \text{masks}[0..3])$, and $c[i] = r_m \times \text{Sbox}(pt[\pi[i]] \oplus k[\pi[i]]) \oplus r_{\text{out}}$ on the fly.
  - Wrap in `tf.data.Dataset.from_generator` with `.batch(64).prefetch(tf.data.AUTOTUNE)`.

### Step 2: Multi-Shard Pipeline & Throughput Benchmark (~1-2 Hours)
- Train 1 epoch on Shard 1 (1,000 steps, batch size 64) with `run_eagerly=False, jit_compile=False` (graph mode) on the RTX 5090.
- Verify:
  - Throughput: ensure step time is $\le 200\text{ ms}$ (target: 300–500 traces/sec).
  - VRAM usage: monitor 32GB VRAM headroom on RTX 5090 (patch size 400, 2,500 sequence tokens).
  - Multi-task output heads: build with either 4 heads (byte 0 pilot) or full 34 heads (all 16 bytes).

### Step 3: Production Training Campaign (Table 1 Specification)

**Corrected 2026-09-26 against the PDF.** Several lines below credited the authors
with choices that are actually ours. Verified ASCADv2 column of Table 1 (printed
p.480, §5.3) is exactly: batch size **64**, steps per epoch **1,000**, epochs
**150**, **target** learning rate **0.00005**, merge filter 1 **0**, merge filter 2
**0**, trace length **1,000,000**, patch size **400**. Those 8 rows are all the
paper tunes. The CM0–CM3 columns (target LR 0.0006/0.0003, merge 16/8) belong to
the **other** datasets and must not be borrowed; merge filters 0/0 is ASCADv2's
distinctive setting.

- **Architecture:** `scaaml.models.get_gpam_model`
  - `trace_len`: 1,000,000
  - `patch_size`: 400 (2,500 patches = √1,000,000, matching their stated rule)
  - `merge_filter_1`: 0, `merge_filter_2`: 0
  - 3 GAU blocks — per §5.2.2/§5.3, deliberately *not* tuned. **There is no model
    width in the paper**: 768 appears nowhere in it (zero text hits). Figure 1
    labels trunk tensors only symbolically as `(trace_len/patch_size, filters)`;
    heads are `Dense(256, swish)` + `Dropout(0.05)`. Any width we set is our choice.
- **Hyperparameters:**
  - Optimizer: **not named by the authors.** `Adafactor` is our own choice and
    `scaaml 3.0.5` ships no optimizer, schedule, warmup or cosine at all (§3.0.5 has
    no `scaaml.training`). §5.4 only says TensorFlow/Keras + KerasTuner.
  - Learning rate: Table 1's row is literally labelled "**Target** learning rate",
    and §5.3.1 (p.481) defines it: *"we mitigate this issue by using a custom
    learning-rate scheduler. We start from a low initial learning rate value, then
    steadily increase it to reach its target value, and finally decrease it with a
    cosine decay as the model converges."* So 5e-5 is the **warmup peak**, not a
    constant — and §5.3.1 is about training **stalling because of weight
    initialisation**, which is exactly what our full run did. The paper gives no
    initial LR, warmup step count, or cosine period, so their schedule is **not
    reproducible from the paper**; the repo link was anonymised and no config was
    published.
  - Batch size: 64 (the dead run actually used **32**). Their scaling note, p.481:
    doubling batch size ⇒ halve steps per epoch and **double** the target LR.
    Their LR search grid: 0.001 / 0.0001 / 0.00001.
  - Steps per epoch: 1,000; Epochs: 150 (⇒ 64,000 traces/epoch at batch 64 — our
    arithmetic, not a paper statement).
  - Wall-clock: the paper's **20 hours** (Table 2) was measured on an **RTX 4090**,
    not a 5090; hyper-tuning ASCADv2 cost 29 days cumulative, ~7 h per config.
- **Consequence for us:** we cannot be "faithful to the paper's hyperparameters" on
  LR, because the load-bearing part (the schedule) was never published. Running a
  constant 5e-5 was using their *number* while omitting the mechanism that made it
  work. Until the schedule is reproduced or replaced, tune LR/optimizer ourselves.
- Save checkpoints every 10 epochs and evaluate on the 80k validation split.

### Step 4: Holdout Evaluation & Key Recovery
- Evaluate best validation checkpoint on the 80,000 holdout traces from Shard 8.
- Compute Table 11 metrics (`acc_pct`, `mean_rank`, `max_rank`) across all heads.
- Run §7.1 fixed-key simulation ($pt \oplus key$, argmax $r_m$) over $[1, 10, 25, 50, 80, 100, 200, 500, 1000, 2000]$ traces to verify if GE < 1 is achieved at ~80 traces.
- Run permutation and tie-breaking null controls to verify the result is signal-backed.

---

- **2026-09-25 (reproduction implementation & queue launch):**
  - Implemented `GPAM/raw_dataset.py`: Streaming multi-shard HDF5 loader for all 8 shards (640k train, 80k val, 80k holdout) with vectorized labels.
  - Implemented `GPAM/train_gpam_raw_full.py`: Paper-faithful GPAM training on 1,000,000-point traces (patch size 400, 3 GAU blocks, Adafactor lr 5e-5, `run_eagerly=True` for Blackwell RTX 5090) with `PeriodicModelCheckpoint` and `TelegramMonitorCallback` (start alert, hourly updates, final completion & holdout evaluation).
  - Implemented `scripts/queue_and_run_gpam.sh`: Autonomous GPU queue manager waiting for active compute jobs (PID 673001) to release GPU, then launching full training run.
  - Background queue runner launched (`task-428`), initial Telegram alert dispatched, and periodic monitoring scheduled.

- **2026-09-25 (later):** `GPAM/` brought under git (`2ac5807`, `b1d3d75`), `*.keras` checkpoints excluded. The two C scripts now resolve paths from `__file__`, so they run without `docs/2026-09-23/out/` present; the duplicated copies there were deleted and `GPAM/artifacts/` is the single source. Ranking switched to counting ties against the target, so the uniform-head null returns rank 255 instead of the spurious 0 (regimes A–D unchanged, they have no exact ties). `train_gpam_ascadv2_budget.py::compute_ge` deleted and `main()` made to refuse, with all four faults named in `GE_BROKEN`. Caveat on the immunity bullet above: label flatness rules out the *prior* cheat, but the permuted control shows a low local GE is still not evidence on its own.
- **2026-09-24:** Brief C executed: raw shard 8 smoke passed; 4-head GPAM on extracted DB evaluated; Table 11 vs GE analyzed; Scoop prior cheat distinguished; metric bugs documented.
- **2026-09-23:** Background download of raw shards 1–7 initiated with hourly crontab monitor and SMTP alerting.
- **2026-09-21:** Initial scaaml environment validation and CPU/GPU smoke passes.
