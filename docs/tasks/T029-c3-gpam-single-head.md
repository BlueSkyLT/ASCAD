# T-029: Single-head GPAM on c3 for S-box prediction

## Status

`done` — **Stages 0–9 complete. Result: THIRD NULL, quantified — and now reproduced at a second geometry (FIFTH null, README §6f).** One open decision is *not* part of the plan: todo C19 (rerun at 40 epochs for a commensurate exposure statement, or keep 30 and quote both numbers) — it applies to **both** geometries now, and at B a rerun costs ≈51 min against A's ≈68. The plan's null-branch action (second-order scan) is complete **and covered the whole activity region**, where it crossed its own registered bar five times; that crossing is adjudicated in README §6e and turns out to be a **per-trace acquisition-state variable in `auth[15]`**, not leakage — the label nulls survive it, re-tested under normalisation and stratification. Remaining follow-up belongs to T-027 (capturer questions C9/C11, now sharpened to "is `auth[15]`/`auth[10]` a gain code?").

*Run*: `c3gpam_seed0_r1_20261004-170100`, seed 0, 30 epochs = 17,790 steps, 51.0 min, 172.05 ms/step,
7.088 GB peak, 1,225,728 params. Pre-registered stop rule **fired** (best val CE 5.547362 at the
first epoch vs threshold 5.495177; train CE fell to 3.755 while val rose to 7.209).
*Test split, touched once* (`purpose=experiment`, 5 earlier touches all `purpose=control`): CE
**5.547199** — 0.0020 *above* chance; acc 0.44%, top-5 2.15%; **`rank_final` 145 == `prior_only_rank`
145**; `rank_final_percentile` 0.051 vs its own pairing null; GE 145.0; `traces_to_rank0` never;
`degenerate_flags` empty. Against the bar registered 45 min before it existed (`rank_final` < 7):
**fails by a mile**, which is the null.
*Controls* (all gates passed): overfit-512 100.00% / CE 0.0000; shuffled labels `CONTROL PASS`;
injection sweep not blind with **`network_sensitivity_floor_mv` = 1.0 mV**; `controls_pass.json`
`all_pass: true`.
*Sharpest single piece of evidence* (`c3_curve_compare.py`, 13 identical config keys): the permuted
labels fit the training traces better at **30 of 30 epochs**, mean **+0.0462 nats** — the wrong sign
for a leak — while the same tool against the 1 mV injection gives −0.2292 with val below chance.
*Third instrument now persisted* (C22): model-free SNR floor at the same site = **0.3 mV**, so the
network is ~3× less sensitive than the model-free univariate statistic and ~5× less than the matched
t-scan (0.194 mV). Suite **56/56 with the cube, 49/49 without**, `c3_acceptance_check.py` **58/58**
(re-measured 2026-10-06, local).

*Geometry B (todo C29, user-approved 2026-10-05)*: `c0=0, L=51,200, patch=400 → 128 steps`, GPAM's own
patch size, **100.0% activity coverage with a 0-sample gap** (A: 96.729%, 1,662 samples outside). All
controls measured **at B** (`controls_pass_b400.json`, geometry recorded, 4 A runs excluded):
overfit-512 100.00%/CE 1e-05; shuffle-labels at chance (`rank_final` **196**); injection sweep
10/3/1 mV all rank 0, **0.3 mV not found** ⇒ **bracketed floor 1.0 mV**, same as A, though at 1 mV B
reaches rank 0 in **10** traces where A needed **20**. B's real run `c3gpam_seed0_b400_20261006-224158`
returned the **FIFTH null** (best val CE 5.545160 at epoch 1, 1.7e-05 below chance, against a 0.05
bar; stop rule fired at 17,790 steps; 38.5 min, 0.945 GB). **B's test split was never touched** — that
is todo C29, and no `rank_final` exists for B. B's own rank bar was measured first
(`seed0_decision_band_b400.json`: smallest null `rank_final` = **2**, p05 19.55, ≤20 in 28/512) and it
is *looser* than A's `< 7`, which is why thresholds are per-geometry artifacts. See README §6f.

**Stage 7 LAUNCHED 2026-10-04 17:01 local** (go-ahead given, scoped to the pre-registered
decision point): `c3gpam_seed0_r1_20261004-170100`, seed 0, **30 epochs first, not 150** — the user
asked to see the epoch-30 result before committing the remaining ~3.3 h. 17,790 steps at 593/epoch,
~52 min, launched through `queue_and_run_c3.sh` with `C3_GO=1 EPOCHS=30` so every precondition is
exercised for real. The stop rule is armed and will record its verdict either way. See
**Completion audit** below for requirement-by-requirement evidence.

## Goal

Run one GPAM head (`SBOX[rand[5] ^ (K^OPc)[5]]`, 256-way) on a 49,152-sample window of c3 and
return a **quantified** verdict: either multivariate first-order leakage, or a third null with a
measured network sensitivity floor in mV stated next to the t-scan's floor (0.10 mV as re-persisted
    2026-10-05; 0.15 mV as it was first transcribed).

"Done" means: controls passed, seed-0 run finished (or the pre-registered stop rule fired), one
test-split evaluation with rank + GE + both internal nulls + degenerate flags, and the writeup in
`GPAM-c3/README.md` §6c/§7.

## Context

- Code: `GPAM-c3/` — standalone subproject, **gitignored**, does not import ASCAD research modules.
  New for T-029: `c3_data.py`, `c3_ranking.py`, `c3_train_gpam.py`, `c3_eval_gpam.py`,
  `test_c3_gpam.py`, `queue_and_run_c3.sh`.
- Data: `/media/lantian/1.44.1-72806/c3_project_8bit/` — 50,000 traces × 4,500,002 samples as a
  bit-verified uint8 cube (10 shards), split 38,000 train / 10,000 test / 2,000 val.
- Construction (`dpa_aes_opc.jpg`, Fig. 5): the device computes `E_K(RAND ⊕ OPc)`, **unmasked**.
  `K = 00..0f`, `OPc = 00 11 22 .. ff`, so `eff = K ⊕ OPc = 00 10 20 .. f0`. Round 1 can only ever
  recover `K ⊕ OPc`, never `K`.
- Two independent first-order localizations already returned **null**. Their controls are *not*
  equally well documented — corrected 2026-10-04 against the artifacts, because the approved plan
  text claimed a positive control for both and only one is in the record:
  - t-scan (T-027 Gate 1), 94 hypotheses × 100% of 4.5M positions: max|t| 6.30 vs threshold 6.44
    — exactly the noise-only expectation √(2 ln 423M); longest run above |t|>4 is 1–2 samples.
    **Persisted positive control** (README §6 table, `c3_positive_control.py`): |t| 6.52 at
    0.15 mV vs the 6.44 threshold, 4.62 at 0.10 mV (missed), 3.47 at 0 mV against a 5.16 global
    max, plus a negative control that never fired → sensitivity floor as recorded then ≈ 0.15 mV.
    **Corrected 2026-10-05:** the persisted re-run puts 6.52 at **0.10 mV** and 4.62 at 0.05 mV; the
    t-values were right, the amplitude labels were one row high. See the C24 decision entry.
  - model-free 256-class SNR (`utils/snr.py` definition), `snr_scan.log`: real/null **1.20×** on
    the SNR ratio (0.9045 vs 0.7532) and **1.01×** on absolute class separation (**2.745 mV** real
    vs **2.733 mV** null, = 3.86 vs 3.84 LSB). Its `--positive` injection run was **not
    persisted** — `c3_snr_meta.json` records `"positive": false` — so the SNR null rests on the
    permuted-label negative control only. **No SNR sensitivity floor may be quoted**; the
    previously circulated "detects 0.30 mV at 9.1×" has no artifact behind it.
- Activity finding: all elevated activity sits in samples **0–~50,814** (1.13% of the trace) as
  ~76 events on a **667-sample cadence** (ac[667] = +0.901); 98.9% of every trace is inert. Those
  events are *not* label-dependent.
- So there is no POI to crop to and the univariate first-order search space is exhausted. What
  remains untested is exactly what a network adds: **multivariate / learned leakage**. The expected
  outcome is a third null; the controls are what make it a measurement rather than absence of
  evidence.
- Prior art to avoid repeating: `GPAM-c3/train_gpam_c3_single_head.py` (2026-10-02, another
  session) is **superseded** — lr 5e-5 (the value that sat at chance for 33 epochs on ASCADv2),
  default byte 0, no checkpoints, no controls, unbounded generator. Left in place, not run.

## Plan

Approved plan file: `~/.qoder/plans/tough-bluff-thrush.md`. Stages are gated; 4–6 are mandatory and
precede 7; stage 7 onward needs an explicit go-ahead each time (`.grok/rules/04`).

- [x] (I:H U:M S:M) **Stage 0** — artifact dirs, preflight (**27** checks, exit 0 — the plan's count of 28 was never what the script ran; it now prints its own count, write the 6 files.
- [x] (I:H U:M S:M) **Stage 1** — unit tests pass from the repo root, from `/tmp`, and `--with-cube`
      (26/27 + 1 skip without the cube; 30/31 + 1 skip with it). Val cache built and re-verified
      bit-exact (2,000 traces in 5.1 s → the train cache is ~2 min, not the ~30 min estimated).
- [x] (I:H U:H S:M) **Stage 2** — caches built and **measured**. train 1.87 GB / 68 s, test
      0.49 GB / 17 s, val 0.10 GB / 5 s, each re-verified **100% bit-exact** on ≥1% of rows
      (train 380/380, test 100/100, val 40/40). Calibration: **265.9 ms/step cold →
      166 ms/step steady state**, **peak VRAM 6.91–7.11 GB** (the 3.8 GB estimate was low;
      the 12 GB cap still leaves 20 GB for a concurrent job), **1,225,728 params**.
      593 train + 31 val steps/epoch → **~104 s/epoch, 150 epochs ≈ 4.3 h** (not the 5 h
      estimated). Loss after 20 steps 5.5502 vs chance 5.5452, i.e. starts where it should.
- [x] (I:H U:H S:M) **Stage 3** — 2-epoch smoke **PASSED**: chance line prints 5.5452 beside
      every epoch; epoch-1 train CE **5.5493** ∈ [5.49, 5.60], val CE 5.5474, acc 0.39%/0.25%
      (chance 0.39%); `best.keras` (10 MB) + CSV + run JSON all written;
      `steps_actually_run` 1,186 = 2×593. Checkpoint restore verified bit-identical to
      `keras.saving.load_model` (max |Δ| 0.0) and different from a fresh init (max |Δ| 2.2e-3),
      so eval is not scoring random weights.
- [x] (I:H U:H S:M) **Stage 4** — overfit probe **GO**: 512 traces, 3,200 steps, 8.9 min →
      **acc 100.00%, CE 0.0000** (gate ≥95% / ≤0.5). Already 90.2%/1.40 by step 416. The
      trunk can fit c3 traces, so a later null is not an optimization failure. Validation was
      deliberately OFF here (400 epochs × 31 val steps would have cost more wall time than the
      probe itself, and the criterion is train-set memorisation).
- [x] (I:H U:H S:L) **Stage 5** — shuffled-label control, 40 epochs + test eval. **GO.**
      Train CE fell 5.5452 → 4.036 (train acc 13.4%) while **val CE rose to 8.30**: it memorised a
      destroyed mapping and became confidently wrong on held-out traces, and `min(val CE) = 5.5469`
      ≥ 5.5252 across all 40 epochs. Test verdict `CONTROL PASS` — acc **0.28%** ∈ [0.24, 0.54],
      mean value-rank **126.93** ∈ [115, 140] (theory says 127.5), `rank_final` **61** at pairing-null
      percentile **0.789** ≤ 0.95, `prior_only_rank` **145**, no degenerate flags. 23,720 steps, 65.5 min.
- [x] (I:H U:H S:M) **Stage 6** — injection sweep 10 → 3 → 1 → 0.3 mV, 40 epochs + eval each.
      **GO, the network is NOT blind.** `network_sensitivity_floor_mv = 1.0 mV`.

      | injected | test acc | CE below chance | `rank_final` | `traces_to_rank0` | GE final | null pct | found |
      |---|---|---|---|---|---|---|---|
      | 10 mV | 6.17% | 2.520 | **0** | 10 | 0.0 | 0.998 | ✅ |
      | 3 mV | 3.16% | 1.597 | **0** | 10 | 0.0 | 1.000 | ✅ |
      | 1 mV | 1.25% | 0.163 | **0** | 20 | 0.0 | 1.000 | ✅ |
      | 0.3 mV | 0.48% | −0.001 | 156 | none | 156.0 | 0.420 | ❌ |

      `controls_pass.json` → `all_pass: true`, `blind: false`, `control_rank_final: 61`.
- [x] (I:H U:H S:M) **Criterion corrected before any experiment data existed.** My first `found`
      rule required `rank_final == 0` **and** CE >0.5 below chance, which scored 1 mV as NOT FOUND
      despite rank 0 at 20 traces and GE 0. Cross-entropy is a weak detection statistic for a side
      channel — the key is ranked first long before the mean log-loss moves — and the plan's
      registered gate was rank-only, so `injection_detected` is now `rank_final == 0 and
      frac_reached_rank0 > 0`. Two consequences carried into the real run: the verdict accepts a
      full rank recovery as "beats chance", and the **0.05-nat stop-rule margin is weaker evidence
      than a 1 mV injection**, so it must be read with the rank and null percentile, never alone.
      The in-trainer CE heuristic is renamed `learned_by_ce` so it cannot be mistaken for the floor.
- [x] (I:H U:M S:L) **Stage 7** — real run, seed 0. **DONE 2026-10-04 17:52 local: 30 epochs,
      17,790 steps, 51.0 min, 172.05 ms/step, 7.088 GB peak. The pre-registered stop rule fired —
      best val CE 5.547362 at the first epoch vs threshold 5.495177, `stop_rule.triggered: true`,
      `val_loss_at_limit: 7.208673` while train CE fell to 3.755.** **Go-ahead given 2026-10-04, scoped to 30
      epochs first** (`c3gpam_seed0_r1_20261004-170100`, 17,790 steps, ~52 min), because 30 epochs is
      exactly where the pre-registered stop rule makes its decision. Continuing to 150 epochs is a
      second, separate go-ahead. Original plan text: 150 epochs. **Needs explicit go-ahead**:
      `C3_GO=1 ./GPAM-c3/queue_and_run_c3.sh`, which refuses unless `c3_window.json` +
      `controls_pass.json` (all_pass) exist.
- [x] (I:H U:M S:M) **Stage 8** — eval + GE on test, **once**, done 2026-10-04 09:54 UTC.
      `purpose=experiment`, `prior_split_evaluations: 5` (all `purpose=control`),
      `prior_experiment_evaluations: []`. Every metric + both nulls + flags present:
      CE **5.547199** (+0.0020 above chance), acc 0.44% / top5 2.15%, **rank_final 145**,
      `rank_final_percentile` 0.051, `prior_only_rank` **145** (== rank_final), `ge_final` 145.0,
      `traces_to_rank0` None, `ge_traces_to_0` None, `frac_reached_rank0` 0.03, `degenerate_flags`
      []. Verdict **THIRD NULL**, against the bar registered 45 min earlier (`rank_final` < 7).
- [x] (I:M U:M S:M) **Stage 9** — writeup: `GPAM-c3/README.md` §6c/§7 (incl. replacing the stale
      launcher preconditions at README:609-611), this file, `docs/lab-log/todo.md` C8/C15–C21,
      `docs/tasks/INDEX.md`.

## Decisions

- **2026-10-04 (user, before any model ran): byte 5 frozen a priori.** `eff[5] = 0x50`. Byte 5 is
  the byte where using `K` instead of `K ⊕ OPc` silently fails — byte 0 is the one byte where
  `K[0] == OPc[0] == 0` makes the wrong key look right, so byte 0 would have hidden that bug.
  Pinned by `test_label_uses_eff_not_k`.
- **2026-10-04 (user): L = 49,152, patch = 64 → steps = 768.** L is exactly 6 cube chunks
  (1.00× read amplification) and covers **96.729%** of the activity region; the 49,152–50,814 gap
  (1,662 samples, 3.3% of the events) is recorded in every run JSON as
  `activity_gap_samples`. Fallback L = 57,344 if anything appears near sample 49,500.
- **2026-10-04 (user): TF via `scaaml`** (paper-faithful; the explicit opt-in `AGENTS.md` requires
  for a TF path in a PyTorch-first repo), **round 1 only**, **full 4-amplitude injection sweep**,
  **seed 0 first**.
- **Input space is `decode_ordinal` (voltage rank), not the stored code, and not volts.** The stored
  byte is a *first-seen level index*: the level table has 20 inversions, the worst adjacent pair
  being code 253 → 254 at **−178 mV**, and code 207 → 208 is +1 in code but −159 mV (−224 in
  ordinal). The 256 levels form a gapless uniform grid (sorted diffs equal to within 1e-6 V, i.e.
  float32 representation noise at ~5 V), so ordinal is an **exact affine rescale of volts** and
  `inputs={"trace1": {"min": 0, "delta": 255}}` is legitimate. Asserted from the cube in
  `test_cube_level_table_is_a_gapless_grid_but_codes_are_not_numbers`.
- **Adafactor lr 1e-3, never 5e-5.** 5e-5 sat at chance for 33 epochs on ASCADv2 raw.
- **`run_eagerly=True`** — required on the RTX 5090 (Blackwell); the XLA Triton GEMM autotuner is
  broken there.
- **Attention memory is 12·B·steps²·4 bytes, not 6·.** That kills B=256 and patch=16 configs;
  B=64 at steps=768 is ~1.8 GB, capped at 12 GB so a concurrent job keeps the rest of the card.
- **`filters=192` is hardcoded in `scaaml.models.gpam`, not 768.** Capacity arguments must use 192.
- **Head input width == steps** (`GlobalAveragePooling1D(data_format="channels_first")` averages
  over filters, leaving the 768 step axis). At patch=64 the trunk feeds a 768-wide head, which is
  itself a capacity-for-noise mechanism. Any non-null result makes the patch ablation mandatory.
  Consequence: **checkpoints are only comparable at identical (L, patch)**.
- **`drop_remainder=True` is mandatory** — Keras infers a Sequence's output signature from the first
  two batches, so a ragged tail breaks `model.fit`.
- **Per-class val accuracy is never computed** (`per_class_counts` raises on `split="val"`): val has
  2,000 traces over 256 classes, minimum class count ~1. Val gets aggregate CE + top-5 only.
- **Pre-registered stop rule** (registered before the run): if after 30 epochs (17,790 steps at
  593 steps/epoch) val CE is not at least 0.05 below 5.5452 (threshold 5.4952), stop and declare
  the third null. Implemented as the `PreRegisteredStop` callback, which records `triggered` in the
  run JSON. It cannot be moved after seeing the curve.
- **Worst-tie ranking everywhere** (`tie_break="worst"`): a flat score vector must rank **255**, not
  0. The ASCAD `rank()` convention reports 0 for tied vectors — the exact bug that made T-028's
  uniform-ablation "succeed".
- **Shuffled-label control criterion, corrected.** A permuted-label model gives non-flat,
  trace-dependent predictions, so its rank is ~uniform on 0..255 (**mean 127.5**), *not* 255. The
  naive "expect 255" criterion belongs to a separate degenerate-flat guard only.
- **Internal null is `plaintext_pairing_null`, not a row permutation.** The final rank depends on
  column sums of the log-score matrix, and permuting its rows leaves every column sum unchanged —
  that null is identically the observation. The null re-pairs predictions with the wrong
  *plaintexts*, which does change the sums. Percentile uses the mid-rank convention
  `P(null > obs) + 0.5·P(null == obs)`, so a flat model reports 0.5 (indistinguishable) rather than
  a fake 1.0.
- **2026-10-04: the network's floor is compared against a *matched* univariate limit of
  0.194 mV, not the t-scan's own floor.** Those two are different instruments; the t-scan's own was
  with a Hamming-weight template at a quiet site over 4.5M positions under Bonferroni, while the
  network gets a non-HW template at an active site over 200 positions with no multiplicity
  correction. `c3_injection_site_check.py` now inverts the matched configuration by bisection:
  |t| = 6.44 needs **0.194 mV** (0.273 ordinal levels), |t| = 5.0 needs **0.121 mV**, and the
  200-position noise ceiling |t| = 3.26 needs **0.032 mV**; each shrinks by 1.78× at the full
  38,000-trace train size. Two useful consequences: (a) 0.194 mV **independently corroborates**
  0.194/1.78 = **0.109 mV** at full train size, which corroborates the re-persisted 0.10 mV floor
  from a completely different configuration, and (b) even the
  weakest swept amplitude (0.3 mV) is 1.5× above the matched limit, so if the network's floor lands
  at 0.3 mV the honest statement is "~1.6× less sensitive than the univariate scan", and its null
  buys multivariate coverage, not depth. `controls_pass.json` now carries
  `t_scan_matched_floor_mv` and an explicit `floor_comparator` string so the deliverable cannot be
  quoted against the wrong ruler by accident.
- **2026-10-04: the prior-only reference was computed before the seed-0 run, so it is
  pre-registered.** `c3_prior_check.py` (`artifacts/controls/prior_check.txt`) needs no model — only
  the train label histogram and the test plaintexts — so it could be run now: train labels for byte 5
  are uniform (min 117, max 194, mean 148.4, **χ² 257.8 on 255 df**), the prior-only rank of the
  true key `0x50` is **145** (percentile 0.432), and the 256 prior-only ranks are exactly a
  permutation of 0..255 (min 0, median 128, mean 127.5, max 255; 21 hypotheses reach ≤20). The
  consequence is registered in advance: **class imbalance cannot fake a low rank for this key**, so
  any `rank_final` below ~50 must come from the traces. Had this been computed after a low rank
  appeared, it would have been an excuse rather than a control.
- **2026-10-04: deliberate deviation — the internal null is 256 *draws*, not 256 *hypotheses*.**
  The plan specified a "256-hypothesis internal null (`rank_final_percentile`)". Taken literally
  (rank each of the 256 key hypotheses, ask where the true key falls) it is **provably
  uninformative**: under worst-tie ranking the 256 hypothesis ranks are exactly a permutation of
  0..255, so the true key's percentile is `(255 − rank)/255`, a deterministic monotone function of
  `rank_final` itself. It would add a number that looks like corroboration and is not. The
  implemented null keeps the same name, the same 256 count and the same `rank_final_percentile`
  key, but draws 256 random re-pairings of predictions against plaintexts, which does produce an
  empirical distribution. My first attempt permuted the rows of the log-score matrix instead, which
  is *also* worthless — the final rank depends on column sums and a row permutation leaves every
  column sum unchanged; it reported percentile 0.5 for a perfectly informed model. Both dead ends
  are documented in `c3_ranking.plaintext_pairing_null` so nobody re-derives them.
- **2026-10-04, written at 09:16 UTC before the epoch-30 result exists: a 30-epoch null is NOT
  commensurate with the 1.0 mV floor, and must not be worded as one.** Every control ran a frozen
  40-epoch budget = **23,720 steps**, and at that exposure this exact architecture and window *did*
  find 1.0 mV of injected non-HW separation. The scoped Stage 7 run gets 30 epochs = **17,790
  steps = 0.75×** that. So if seed-0 is null at 30 epochs, the licensed claim is only *"nothing
  found in 17,790 steps, a budget 25% smaller than the one that detects 1 mV"* — an upper bound on
  sensitivity **weaker** than 1.0 mV, which is why the floor and the null must be reported as two
  separate numbers and never collapsed into "no leakage above 1.0 mV". Note the conflation the
  scoped run invites: the plan's 30-epoch rule was designed as an *early exit inside a 150-epoch
  run* ("stop while hopeless"), not as the total exposure a null is measured at. Options if it
  comes back null, in the same pre-registered spirit: (a) rerun at **40 epochs = 23,720 steps,
  which matches the control budget exactly** (not 41; 41 overshoots for no reason) so the null is
  commensurate with the 1.0 mV floor being quoted; or (b) keep 30 and report the null's own
  exposure explicitly, never merged with the floor. 150 epochs would be 3.75× the controls.
  **Cost of the scoping, stated plainly:** `c3_train_gpam.py` has no `--resume` and no
  `initial_epoch`, so (a) is a clean relaunch of 67 min that discards the 49 min already spent —
  116 min total for what one 40-epoch launch costs. The scoping only saves time if the answer is
  stop (then it saves ~3 h); if the answer is extend, it costs 49 min. Same gap as T-019's D5.
- **2026-10-04: the seed-0 bar is a measured null band, registered before the result existed.**
  The plan pre-registered a 0.05-nat CE margin, which the sweep proved is a poor statistic. So
  `c3_decision_band.py` took the *shuffled-label control's own cached test predictions* — a model
  already proven to carry no key information — and destroyed its trace↔plaintext link 512 times on
  the real 10,000 test traces (`artifacts/eval/seed0_decision_band.json`):

  | statistic under the null | value |
  |---|---|
  | `rank_final` mean / median | 104.5 / 101.5 |
  | `rank_final` 5th percentile | **29** |
  | `rank_final` minimum over 512 draws | **7** (and **0 never**) |
  | draws with `rank_final` ≤ 10 / ≤ 20 | 2 / 16 of 512 |
  | draws reaching rank 0 *somewhere* in the curve | **7.62%** |
  | earliest such rank-0 crossing | **20 traces** (median 480) |

  Registered bar for calling seed-0 a signal: **`rank_final < 7`**, plus pairing-null percentile
  >0.95 and `prior_only_rank` >100. Two statistics are hereby retired as evidence:
  `frac_reached_rank0` and `traces_to_rank0`. A null model hits rank 0 somewhere in the curve on
  7.62% of draws and can do it as early as **20 traces** — which is exactly the number the 1 mV
  injection printed. So `traces_to_rank0 == 20` is *not* by itself a detection; what carries the
  1 mV result is that `rank_final` stayed 0, an event that occurred **0 times in 512 null draws**.
  Computed 45 minutes before the seed-0 evaluation exists, so it cannot be fitted to the answer.
- **2026-10-04: only the t-scan's sensitivity floor may be quoted.** Checking the artifacts before
  writing the §6c comparison turned up that the SNR scan's positive control was never persisted
  (`c3_snr_meta.json` has `"positive": false`), while the plan text and my own earlier summaries
  claimed "each with working positive and negative controls" and a specific "0.30 mV at 9.1×"
  figure. No artifact supports that number, so it is struck from the README and this file. The
  T-029 writeup compares `network_sensitivity_floor_mv` against the **t-scan's 0.15 mV** only, and
  §6c now says plainly that the SNR null rests on its permuted-label negative control. Re-running
  `c3_snr_scan.py --positive --inject-mv 0.3` (~106 min of GPU) is registered as **C22** — a T-027
  gap, deliberately not inserted ahead of the T-029 controls.
- **2026-10-04: C22's method changed, and it now calibrates the SNR null instead of risking the
  artifact.** Running `c3_snr_scan.py --positive` would have overwritten the recorded
  `c3_snr_meta.json` — the null's only artifact — and wanted ~106 min of GPU during the seed-0 run.
  `c3_snr_sensitivity.py` reads **one** window and sweeps 0/0.3/1/3/10 mV of the *same* non-HW
  template at the *same* site (samples 3000–3200) on byte 5 — exactly the perturbation the network
  controls were shown — using `A_hat = sqrt(mean site between-V²(injected label) − mean over the 16
  permuted nulls)` with a 2σ band measured from those nulls themselves. Two findings before the
  sweep even runs:
  (a) `--selftest`: `window_stats` (lifted out of the scan loop so the two tools cannot drift
  apart) matches a float64 reference at **0.0150%** worst relative error on the between-class
  variance, but **3.02%** on the SNR — at 4.95 V the `E[x²] − E[x]²` form of the within-class
  variance cancels ~5 digits. A numerical caveat on every SNR *ratio* in §6b, not on absolute
  separations. (b) `--numerics` (`artifacts/controls/snr_numerics.json`): leak-free synthetic
  traces with the scan's own shape (38,000 traces, 16 real + 16 permuted label vectors, ~148
  traces/class, 2,000 positions) give a max-over-positions absolute-separation ratio of
  **1.005–1.013×** — so the cube's own ratio (1.004× in mV, recomputed from `c3_snr_meta.json`; the
  scan printed 1.01×, which is the between-variance scale) sits at or below it, which is a stronger
  statement than "no peak appeared" — and float32 equals float64 to three digits, so the
  cancellation does not inflate that max-ratio. The 1.20× SNR-ratio excess therefore remains
  unexplained at 2,000 positions (the scan maxes over 4.5M) and is reported as unexplained rather
  than smoothed over; the null keeps resting on absolute separation.
  **Resolved the same day, and it was neither.** `c3_snr_decompose.py` prices the ratio without any
  noise model: the 16 null vectors are permutations of the 16 real ones, so which group is labelled
  "real" is arbitrary under no-dependence, and re-partitioning them 200,000 times gives median
  0.985x, p95 1.142x, max 1.263x — the observed 1.2008x sits at one-sided p = 0.0215. An ordinary
  draw, not a signal, and it would not survive correcting for 16 bytes. At the real group's own peak
  (sample 44,376) the paired medians are SNR 1.007x, separation 1.007x, within 1.005x, and
  separation/within reproduces the SNR ratio. The first draft of that decomposition was wrong in an
  instructive way — it compared each group's OWN argmax, so 1.036/1.040 could not reproduce 1.20;
  the arithmetic failing to close is what caught it. Until the sweep runs, any
  SNR **mV floor** is still off limits.
- **2026-10-04: the `L=57,344` fallback trigger is literally met, and `L=49,152` stands anyway —
  with the reason recorded rather than buried.** `c3_window_gap_check.py`
  (`artifacts/c3_window_gap_check.txt`) finds **3–5 of the ~76–88 detected 667-cadence events** in
  the excluded 1,662 samples, onsets 49179 / 49606 / 49960 / 50387 / 50689, carrying **3.892%** of
  the region's excess-std energy — so energy coverage is **96.108%**, slightly worse than the
  96.729% sample-count figure the run JSONs quote, because the gap is *more* active per sample than
  the crop. The window is unchanged because the trigger was written for **label-dependent**
  structure: the full-coverage t-scan already searched 100% of 4.5M positions including this gap and
  found nothing above the noise-only expectation, and §6b established the events are not
  label-dependent. `L=57,344` (896 steps, ~1.36× attention memory, inside the 12 GB cap) is
  registered now as the pre-planned follow-up so that adopting it later is a decision, not an
  improvisation after seeing a result.
- **2026-10-04: the injection site was calibrated before the sweep, and the sweep stays frozen.**
  `c3_injection_site_check.py` measures the site on 12,000 real train traces: local per-sample std
  **7.85 levels = 5.58 mV** (window's far end 7.32 levels; documented rail noise ±5.25 mV), so
  samples 3000–3200 are representative rather than unusually loud; noise-only max|t| over those 200
  positions is **2.6** against a √(2 ln 200) = 3.26 ceiling; and the four sweep amplitudes give
  max|t| **8.5 / 22.7 / 64.8 / 213.2** against the full scan's 6.44 Bonferroni threshold. Two
  consequences registered *now*, before the result is known: (a) the 10 mV gate is unambiguous —
  if the network misses |t| = 213 it is blind, full stop; (b) even 0.3 mV is univariate-detectable
  here, so all four amplitudes may come back "found" and the deliverable would then honestly be
  `floor ≤ 0.3 mV` (an upper bound), with a downward extension (0.03/0.1 mV, ~2.4 h) as a
  **follow-up decision, not a mid-run change**.
- **2026-10-04: `c3_window.json` records evidence, not an assertion.** The first version stamped
  `frozen_before_any_model_ran: true` with a timestamp *later* than the first model run, which is
  self-contradicting in the one file whose purpose is to prove the geometry was not chosen after
  seeing results. It now carries `provenance` with `plan_approved_utc` 2026-10-03T16:41:15Z vs
  `first_c3_model_run_utc` 17:11:56Z (~30 min apart, so the claim is true and checkable), the
  writer is **write-once** and refuses a different geometry unless forced, and the launcher refuses
  if `decided_before_any_model_ran` is not true.
- **Controls are evaluated on the test split too**, so the plan's literal control criteria
  (test acc band, mean value-rank band) apply. Every touch is appended to
  `artifacts/eval/test_split_access.log` with `purpose=control|experiment`, and the real-run eval
  JSON reports `prior_split_evaluations` and `prior_experiment_evaluations`, so "test touched once
  *for the experiment*" is auditable. The config was frozen a priori, so control evaluations cannot
  tune the experiment.

- **2026-10-05 (C25): second order is now covered *as a method*, at one slot, and the plan's null
  branch is fully executed.** Nothing in the four nulls so far could have seen a leak that cancels in
  the mean of each sample but survives in a product of two; that family has now been searched with a
  registered bar, a control that is demonstrably first-order-blind, and a measured floor. Scope stays
  explicit — one 667-sample slot is 0.0148% of the trace — so the claim is "no second-order leak in the
  first cadence period down to a 1.0 mV-carrier floor", never "none anywhere". Extending coverage is
  the last untried compute direction on c3 that does not depend on the collaborator, and it is
  expensive (17.1 h), which is why it is a question rather than a task.
- **2026-10-05 (C24): the t-scan's sensitivity floor was wrong in prose for six days, and the cause
  was that the control left no artifact.** README §6 and T-027 both recorded "detected from 0.15 mV
  up (|t| = 6.52)". Re-running `c3_positive_control.py` — deterministic, no RNG — reproduces every
  t-value exactly (3.47 / 4.62 / 6.52 / 14.10 / 40.64) but puts 6.52 at **0.10 mV** and 4.62 at
  0.05 mV: the labels had been transcribed one amplitude row high, and the 0.05 row had been dropped,
  which is exactly the edit that makes a shift invisible. The quoted |r| = 0.029 also matched the
  *theoretical* 0.15 mV line, so the table looked internally consistent. Independent check:
  0.194 mV at n = 12,000 scales by |t| ∝ √n to **0.109 mV** at 38,000. Fixes, not just corrections:
  the script now writes `artifacts/controls/tscan_positive_control.json`, `controls_pass.json`
  **reads** `t_scan_floor_mv` from that file instead of a hardcoded constant, and the drift test
  asserts the README agrees. The lesson generalises to this repo: a printed number is a rumour; the
  network's floor (1.0 mV) is now compared against 10x-more-sensitive univariate tools, which is the
  direction of the conclusion unchanged and its magnitude corrected.
- **2026-10-05: I nearly launched an unapproved 150-epoch run, and the fix is a guard plus a test,
  not an apology.** While verifying that the launcher *refuses* correctly, I ran `C3_GO=1
  queue_and_run_c3.sh` without `EPOCHS`. It passed every gate, silently defaulted to `EPOCHS=150`,
  and only failed to start training because another job held the GPU and it was parked in the
  wait-loop. Three things were wrong: a default that could authorise hours of GPU by typo, a rerun
  path that would have been a second `purpose=experiment` touch of the test split, and refusals that
  wrote to the queue log before deciding. The launcher now (a) has **no** `EPOCHS` default, (b)
  refuses if a real run JSON for that seed already exists unless `C3_ALLOW_RERUN=1`, and (c) checks
  both *before* touching the log. `test_launcher_refusals_never_reach_the_wait_loop` asserts each
  refusal exits 1 **and** that the log byte-for-byte does not grow — it also fails on a hang, so a
  regression that reaches the wait loop cannot pass. The two stray log entries are annotated in
  `artifacts/logs/queue_and_run.log` rather than deleted, and the seed-0 artifacts are provably
  untouched (no new checkpoint dir, CSV or run JSON).
- **2026-10-05: the null is stated as "negative transfer", not "no learning" — and the split that
  proves it is val, not test.** `c3_pred_dispersion.py` puts the trained head next to two baselines on
  held-out val traces: a uniform guess (CE 5.5452) and a constant label-histogram predictor (5.550889).
  The network sits between them at 5.547275 — better than the histogram, worse than nothing — with
  145 distinct argmaxes and mean KL 0.00445 nats to the marginal, so its output is trace-dependent
  rather than a stored vector. Diagnostic choice matters as much as the number: running this on test
  would have been a second experiment-purpose use of the split that the protocol says is touched once,
  so it runs on val, where aggregate CE is exactly what the protocol designates.
- **2026-10-04: C22's first detection rule was rejected by its own negative control and replaced,
  with the rejected run kept on disk.** The rule I registered was `A_hat > sqrt(2·sd)`, with `sd`
  from the dispersion of the 16 permuted null labels' site means. On the full 38,000-trace pool the
  **0 mV control came out DETECTED** (A_hat 0.0711 vs band 0.060), and the reason is structural: all
  32 label vectors are applied to the *same traces*, so they share one noise field and are not 16
  independent replicates — the dispersion underestimates the variability of a single label's site
  mean. The null is now 256 re-pairings of the injected label against the trace rows (the same
  construction as the network's `plaintext_pairing_null`), and an amplitude must clear both that p95
  **and** twice the null level measured on this data at this site. Floor: **0.3 mV**, with amplitude
  recovery accurate to 0.3–1.3%. The site, template, pool and amplitude list never changed — only the
  null did, and the rejected artifact is preserved as
  `artifacts/controls/snr_sensitivity.json.ANTI-CONSERVATIVE-band`. Also recorded rather than hidden:
  the un-injected byte-5 label sits at the **98.8th percentile** of its own permutation null at this
  site (p95 0.0596, p99 0.0714). That is a caution on the floor's precision, not a leak claim — one
  byte, one pre-chosen 200-sample site, and the full-coverage scan corrected for 4.5M positions.
- **2026-10-04: the pairing null is model-dependent, so `rank_final_percentile` is not a cross-model
  yardstick.** 256 re-pairings centre at rank **102.2** for the shuffle control's predictions,
  **139.1** for the 1 mV injection's and **50.7** for the seed-0 run's. Consequence: the seed-0
  percentile of 0.051 means "worse than its own null", which is the same fact as its test CE being
  0.002 above chance — a memorised-noise head is slightly anti-informative on held-out traces. The
  quantity that *is* comparable across models is `prior_only_rank` (145 for all three: it depends
  only on the label frequencies), which is why **`rank_final == prior_only_rank == 145`** is the
  load-bearing statement of this null and the percentile is decoration.
- **2026-10-04: the verdict sentence was fixed in the generator, not retro-patched into the
  artifact.** `verdict_for` used `prior_only_rank <= rank_final`, so it printed "the label prior alone
  reaches rank 145, i.e. better than the model's 145" when the two are equal. The generator now has a
  separate equality branch, and the corrected sentence was regenerated from the cached numbers
  (pure function over `res`/`run`) — **without re-running the evaluator**, because re-running it would
  append a second `purpose=experiment` line and destroy the "test touched once" claim the design buys.
  The JSON on disk keeps the old sentence; README and this file quote the corrected one.

- **2026-10-05 (user): geometry B approved — `c0=0, L=51,200, patch=400 → steps=128`.** Registered in
  `artifacts/c3_window_b400.json` with `decided_before_any_model_ran: false`, because it is not an a
  priori freeze: it exists (a) after A's null, and (b) after the second-order tiling put real
  structure at samples 38,686–44,022, inside A's crop but with its 1,662-sample tail outside. Why
  this geometry rather than another: `51,200 > ACTIVITY_END = 50,814`, so activity coverage is
  **100.0% with gap 0** (A: 96.729%, gap 1,662); the head's input width equals `steps`, so 128
  instead of 768 removes the capacity-for-noise mechanism the plan listed as risk 2; one attention
  matrix is `12·64·128²·4 = 0.05 GB` against A's 1.81 GB. Costs: 51,200 spans 6.25 cube chunks so it
  reads 7 → **1.12× read amplification** (A was exactly 1.00×), and 1,120,000 parameters. Full-length
  was rejected on arithmetic and memory, not taste: `4,500,002 = 2·13·17·10181`, so the only large
  legal patch is 10,181 → 442 steps, which buries a 7–8-sample-wide event in a 10,181-point token,
  while a sane patch over the full trace (667 → 6,746 steps) needs 140 GB for one attention matrix.
- **Registered before B runs, so its controls cannot be tuned afterwards (rules/04).** A null at B is
  only a measurement if B has its own floor: A's `network_sensitivity_floor_mv = 1.0 mV` was measured
  on a 768-step head, and the launcher now **refuses** to licence B with A's controls file
  (`CONTROLS GEOMETRY MISMATCH`, pinned by
  `test_geometry_gate_refuses_to_licence_a_second_window_with_the_first_windows_floor`). Gate order in
  `run_b_campaign.sh`: calibrate → 2-epoch smoke → `overfit512` GO (acc ≥95%, CE ≤0.5) → inject
  **10 mV must give `rank_final == 0`**, then 3 and 1 mV to bracket a floor instead of an upper bound
  → `shuffle-labels` 40 epochs at chance → `--summarize-controls --gate-geometry 0,51200,400` into
  `controls_pass_b400.json` → only then the 30-epoch real run. The stop rule transfers unchanged
  because it is written in nats: **stop if best val CE has not reached 5.495177 (= ln 256 − 0.05) by
  epoch 30**, and steps per epoch are identical (593 train + 31 val), so 30 epochs = 17,790 steps at
  both geometries.
- **Outcome of that registration (2026-10-06): every gate passed at B and the real run fired the rule
  as written** — `controls_pass_b400.json` `all_pass: true`, floor 1.0 mV bracketed, best val CE
  5.545160 vs the 5.495177 threshold, stop rule triggered at 17,790 steps. B's null band was measured
  at B too (`rank_final < 2`, looser than A's `< 7`), so B's thresholds are its own. README §6f.
- **Left deliberately undecided for the user:** whether B's real run touches the 10,000 test traces.
  A's evaluation was the plan's single `purpose=experiment` touch, and `c3_eval_gpam.py` *records*
  rather than blocks a second (`prior_experiment_evaluations`). Default position is **val-only** — CE,
  top-5, and the curve test against B's shuffled control — which supports "the null does not depend on
  the window choice" without spending a second unbiased test evaluation. The campaign script stops
  before that step.

## Working notes / session log
### 2026-10-06 — B's real run came back null: the FIFTH null, its own rank bar, and four harness bugs

- **Result.** `c3gpam_seed0_b400_20261006-224158`: 30 epochs = **17,790 steps** (the same step count
  A ran, so the two nulls are at equal exposure), 129.92 ms/step, **38.5 min**, **0.945 GB** peak,
  1,120,000 params, activity coverage **100.0%** with a **0-sample** gap. Best val CE **5.545160 at
  epoch 1** — 1.7e-05 *below* chance, against a pre-registered bar of 0.05 below, so the rule fired
  and the val curve then rose to **6.560305** while train fell to **4.267731** and train acc climbed
  to **12.37%**. Same curve shape as A: memorising 38,000 traces with nothing in the window to
  memorise about the label. Recorded as **README §6f**.
- **B's test split was not touched.** The chain stops before it by design: the campaign's endpoint is
  the run, and whether B may have a `purpose=experiment` eval of those 10,000 traces is the user's
  call (**todo C29**). The access log still holds exactly **one** `split=test` + `purpose=experiment`
  line (A's); B's 5 test touches are all `purpose=control`. No `rank_final` exists for B's real run.
- **B's rank bar, measured before any B ranking could exist.** `c3_decision_band.py --tag b400` →
  `seed0_decision_band_b400.json`: 512 re-pairings of B's own shuffled-label control cache, same seed.
  Mean 102.43, median 97.0, p05 **19.55**, smallest `rank_final` in 512 draws = **2** (A: 7),
  ≤20 in **28/512** (A: 16), rank 0 mid-curve in **7.03%** of draws at as few as **10** traces. So
  B's registered bar would be `rank_final < 2` — *looser* than A's. B's null looks better by chance,
  which is precisely why a second geometry cannot reuse the first geometry's threshold. Re-running the
  tool on A's cache reproduces `seed0_decision_band.json` number for number, so the two are comparable.
- **Four things in the harness were wrong, all recorded rather than quietly patched.**
  (1) `c3_decision_band.py` picked "the last `*shufflelabels*_probs_test.npz` by filename" — A and B
  sort adjacently and A happened to win, so the band's provenance was luck; it now selects by geometry
  tag, records the geometry it belongs to, and refuses when no cache matches.
  (2) `test_a_two_epoch_smoke_...` read `epochs` from run JSONs, but the trainer writes
  `epochs_requested`/`epochs_run`; with no B real run on disk the loop body never executed and the
  test was **green while checking nothing**. It now uses the same accessor as production and B's run
  is in the loop.
  (3) `purpose_for` classified by *name* only, which is how B's 2-epoch dry run got logged as
  `purpose=experiment` (on `split=val`, so the test-once claim is intact — but the audit trail now
  carries two stale lines from before the fix). It checks the epoch budget too, and a JSON recording
  no budget can no longer claim experiment status.
  (4) The stop-rule callback printed "this is the **third** null" — an ordinal about the project's
  counting that no script can know. Now "null branch". The evaluator's verdict *label* is still
  `THIRD NULL`, because A's published artifact and README §6c quote that exact string; the docstring
  says the ordinal is A's and any other run's count comes from the writeup.
- **Two stale test assertions died when B's run landed**, and the failure was informative rather than
  cosmetic: the geometry-mismatch refusal test was being answered by the *earlier* existing-run guard,
  so it now pins the first refusal's reason and drives a second launch with `C3_ALLOW_RERUN=1` to reach
  the geometry gate (verified live — the launcher prints `CONTROLS GEOMETRY MISMATCH` and exits 1
  without writing to the queue log or the training log).
- **§6f guard, mutation-tested.** New block in `test_writeup_numbers_match_the_result_artifacts` reads
  README §6f *sliced* (not concatenated with A's section) and pins it cell-wise against B's artifacts:
  13 of 13 applied edits caught, including swapping a table's two columns where the neighbouring table
  uses the opposite column order, and replacing the floor cell with 0.3 mV (which a substring test
  would have missed, because "1.0 mV" appears in a sentence elsewhere in the same section).
- Also this session, at the user's request: `docs/notes/c3-usable-trace-region.md` — the standalone
  report on why only the first ~50k samples are usable. Re-derived from the full-coverage profiles
  rather than from this README, which found two figures I had been carrying as remembered (a run-length
  detail, the separation-ratio set) — corrected in place. Sharpest fact: the global first-order maximum
  over all 423,000,189 pairs is **6.303 at sample 1,656,004 — inside the discarded tail** — and equals
  the noise-only expectation √(2 ln M) = 6.303.
- Suite **56/56 with cube** (49 without), `c3_acceptance_check.py` **58/58**, preflight unchanged.

### 2026-10-06 — geometry B characterised: all controls pass, and B is the *more* sensitive instrument
- Summary: B (`c0=0, L=51,200, patch=400 → 128 steps`) has now been measured, not assumed. Every
  gate registered in the Decisions section above passed, and `controls_pass_b400.json` records
  geometry `(0, 51200, 400) / 128 steps` with `all_pass: true`, `blind: false`. The real 30-epoch
  run is queued through `queue_and_run_c3.sh`.
- Gates, measured: caches rebuilt for the new window (train 1.95 GB / 73.5 s, 380/380 rows
  bit-exact; val 0.10 GB, 20/20); calibrate **137–161 ms/step** and **0.943 GB** peak (A: 172.05
  ms, 7.088 GB — B is cheaper on both axes, 6× narrower head, 1,120,000 params); 2-epoch smoke
  wrote its JSON; **overfit-512 GO at 100.00% / CE 0.0000** (identical to A, so the pipeline is
  sound at 128 steps too); **shuffle-labels GO** — min val CE 5.5457 vs chance 5.5452, test acc
  0.5%, mean value-rank 128.17, `rank_final 196` at percentile 0.0625 → `control_rank_final = 196`
  is the reference the real B run must be read against.
- The surprise: **B is more sensitive than A at the amplitude that defines A's floor.**
  10 mV: both rank 0 / 10 traces. 3 mV: both rank 0 / 10 traces (CE 3.547 vs 3.949).
  **1 mV: B rank 0 in 10 traces, A needed 20** (CE 4.607 vs 5.382). A's 0.3 mV was not found
  (rank 156); B's 0.3 mV is running now, and if it is found then B's floor is ≤0.3 mV — at or
  below the model-free SNR's 0.3 mV — and the sentence "the network is the least sensitive of the
  four instruments" becomes a statement about *patching*, not about networks. A's 1.0 mV stands.
- Two guards written because the numbers wanted them, not because they were expected. (1)
  `floor_is_upper_bound`: B's sweep so far found *every* amplitude it measured, so "floor 1.0 mV"
  would read as a bracketed value when it is only an upper bound; the file now says
  `upper bound? True` plus the amplitudes measured at that geometry. (2) geometry-filtered
  run selection: with two geometries on disk, "newest run of each family" mixed them, which made
  the sweep look empty and printed **"network is BLIND, no null may be claimed"** about an
  instrument that had just found 10 mV at rank 0. Cause was mine — `patch_size` sits at the top of
  a run JSON, not inside `crop`, so the filter matched nothing. Foreign runs are now listed as
  excluded rather than silently merged, and an unfiltered `--summarize-controls` refuses outright
  so the published `controls_pass.json` cannot be re-pointed at B.
- Queue note: my own 0.3 mV job is deliberately holding the real run in the launcher's wait loop
  (it refuses to start while any compute process is live, so no job gets OOM-killed). Correct
  ordering, at the cost of ~40 min.
- **0.3 mV at B: NOT FOUND** (rank_final 69, traces_to_rank0 None, test CE 5.545563 vs chance
  5.545177, null percentile 0.68). So B's floor is a **bracketed 1.0 mV**, the same value as A's,
  and the watcher rewrote the roll-up to include it (`upper bound? False`, all four amplitudes
  measured at the geometry). The "B is more sensitive" reading survives only where it is true —
  at 1 mV B reaches rank 0 in 10 traces where A needed 20 — and not as a lower floor.
- **A naming bug of mine stopped the real run for ~20 h, silently.** The chain's smoke ran as
  `--epochs 2 --tag b400`, which produced `c3gpam_seed0_b400_<stamp>.json` — byte-for-byte the
  shape of a real run name. So the launcher's duplicate-run guard fired ("a real seed-0 tag-b400
  run already exists"), the chain died under `set -e`, and the final "campaign complete" line
  never printed, so nothing looked wrong except the absent result. Root cause was a rule that
  depended on someone typing "smoke" into a tag; fixed at the source, in one shared constant
  (`c3_data.SMOKE_EPOCHS_MAX = 10`): the trainer now appends `smoke` to the run name below that
  budget, `_real_run_candidates` reads the epoch budget out of the JSON as well as the name, and
  the launcher's guard counts only real runs. The same bug would have let the evaluator resolve a
  2-epoch checkpoint as "the run", so this was never only an inconvenience.
  `test_a_two_epoch_smoke_cannot_be_resolved_as_the_experiment` covers it, and both halves were
  mutation-checked (removing the self-labelling, or reverting the evaluator to name-only, each
  now fails the suite). Suite **49/49**.
- Two of my own test assertions also turned out to encode a single-geometry world and were wrong
  once B existed — fixed by scoping them, not by deleting them: the control-before-experiment
  ordering is now checked **per geometry** (B's controls necessarily ran after A's experiment;
  requiring otherwise would forbid characterising any second instrument), and the refusal test
  compares training-log *content* rather than asserting the file does not exist.
### 2026-10-05 — the second-order tiling crossed its own bar, and turned out to be the acquisition

- Summary: the 76-slot coverage run finished and **five slots exceeded the bar registered before any
  peak was read** — the first non-null localisation on c3. Two days of adjudication later the label
  nulls are untouched and the finding is a different one: `auth[15]` is a **per-trace acquisition
  state variable**, not a secret. README §6e is the write-up; `c3_2o_{probe,hyp_profile,auth_shape,
  auth_response_shape,auth_causal,phase_hypothesis}.py` and `c3_norm_recheck.py` are the evidence.
- Tiling: M = 76 × 94 × 222,111 = **1.5868e9**, bars noise-only **6.5092** / Bonferroni **6.6393**,
  predicted carrier floor **1.0298 mV**, covering **50,692 of 50,814** active samples. Real mean of
  slot maxima 5.743 vs null 5.497; **5 slots over the bar, null 0** (null max 6.056); largest **8.666**
  at slot 43,355, pair 312/428 = samples 43,667/43,783. All five crossings, and the argmax in 7 of the
  8 slots in 38,686–44,022, are `auth_sboxHW_b15`. 13.1 s per slot per pass — which retires the "17.1 h"
  figure: that was all 6,746 slot *offsets*, while the events are phase-locked so one aligned tiling
  reaches every event.
- Why it is not a numerics or selection artifact: float64 reproduces float32 exactly (8.6663 = 8.6663,
  cancellation ratio 0.0035); disjoint halves give r = 0.0392/0.0499 (1.04 SE apart, same sign);
  200 permutations of this hypothesis never reach the peak (null max-of-draws 5.768, p at the 0.005
  floor); **block** permutations at B = 64/256/1024 do not either, so trace-order drift is excluded;
  first-order |t| at the two samples is 0.42/1.41, a route worth 0.0021 of the 2O statistic; and the
  slot has **223** pairs over 5.5 where the permuted null averages **0.0**. Control slot 20,010 gives
  p = 0.40. It is one column, not a group: runner-up 5.441, nothing else over 5.25.
- What it is: no XOR mask is identified (half A prefers **0x31**, half B independently **0x34**, the
  `eff`-based template ranks 73/256 and 43/256), no AES-shaped template beats a byte that does not
  leak (best 0.2508 vs a `rand` control's 0.2210), yet the 256-class response vector is **0.9559**
  reproducible across halves. Out-of-sample r = **0.4375 (t = 67)** for `auth[15]` against 0.007–0.015
  for every other field, and it survives amplitude residualisation (0.4364) — as it must, since a
  *mean* dependence of a product cannot come from a multiplicative scale. The clincher is in a region
  that computes nothing: in samples 100,000–100,667 a free 256-class fit of the window's per-trace
  **std** on `auth[15]` gives **F = 10.31** (rand 0.977, label 0.959), at 0–667 **F = 18.41** with
  `auth[10]` at 17.55, while the window's **mean** is untouched (**F = 1.23**).
- Two new dataset facts worth more than the peak: `auth[15]` is the most energy-coupled of all 94
  hypotheses (|corr| 0.0225, rank 1), and the traces are **not sample-aligned** — per-trace lag std
  1.4–4.6 samples, only 1–4% at zero lag. Alignment was the leading competing explanation and is
  rejected: corr(lag, auth) = 0.0106, F(lag | auth) = 1.16, and removing the lag leaves **1.0035** of
  the dependence.
- Because a per-trace noise scale could have weakened the *mean*-based nulls, they were re-tested
  (`c3_norm_recheck.py`, 16 label bytes × samples 0–49,152, M = 786,432, bars 5.2106/5.4084, and every
  variant null-calibrated on permuted templates): raw **4.983**, normalised **4.966**, quietest
  quartile **4.644** — all under the bar, calibrations 4.767/4.833/4.655. **NULL SURVIVES
  NORMALISATION.** The claim bar stays the original scan's 6.44, since that family already contains
  these templates; the subfamily's lower bar is not offered as a new claim.
- **My two mistakes, recorded because both looked like results.** (1) The split-half criterion I
  registered — flat |t| > 4.5 per half — is dimensionally wrong: t = r√n, so halves of a true effect
  sit at t_full/√2 (6.1 for the 8.67 peak, 4.7 for the 6.69 one) and a constant cut rejects small real
  effects while accepting large flukes. It failed on exactly the smallest crossing. Replaced by the
  1-df heterogeneity test |r₁ − r₂| < 2 SE; both versions are stored in `c3_2o_probe.json`, and the
  correction is labelled as a correction rather than a quiet edit. (2) Weighting the **raw**
  (DC-carrying) samples and then column-centring adds x̄(w_t − w̄) to every deviation — 2.5 V × 8.5% =
  ~40× the real per-trace noise — which crushed max|t| from 4.98 to **1.955** and would have read as
  "normalisation removed the false positives". The arithmetic selftest could not see it because the
  arithmetic was right; what caught it was checking the *distribution*. Both the per-variant null
  calibration and an assertion that the bad ordering fails it are now in `--selftest`, and the first
  run is kept as `c3_norm_recheck.INVALID-scaling-order.json`.
- Guards added: `test_second_order_adjudication_matches_the_artifacts_and_the_prose` (cell-by-cell
  transcription *and* the relational claims — which hypothesis owns the crossings, whether the two
  halves really disagree on the mask, whether the best template beats a non-leaking byte's best,
  whether the uncalibrated variant is flagged); `c3_acceptance_check.py` now gates the whole
  adjudication chain, **51/51 pass**, and the suite is **54/54 with cube, 47/47 without**.
- Net effect on the plan: the answer to the question the plan asked does not change — no multivariate
  round-1 byte-5 leakage, `rank_final == prior_only_rank == 145`. What changed is that C9(b) ("what is
  `auth`?") became a checkable hypothesis with a predicted answer (a gain/attenuation or trim code),
  and that any future covariance-shaped work on c3 must control for alignment and per-trace noise or it
  will find `auth` again instead of a key.
### 2026-10-05 — swept the sections I inherited, and my own table failed first

- Ran the same decimal-vs-artifact sweep over README §6/§6a/§6b (the T-027 sections I did not
  write): 85 decimals, 14 with no literal artifact match. Most were my checker's fault — it tokenises
  `4.872` and so cannot see a legitimate 2-dp `4.87` in prose; 6.30 / 6.44 / 4.87 / 4.60 are all
  present in `fullscan.log` / `firstorder_scan.log`. Recording that the tool over-reports matters as
  much as recording what it catches, otherwise the next session "fixes" numbers that were fine.
- **What it did catch was mine, written today:** the §6 positive-control table's `expected t` column
  came from re-deriving the formula rather than copying `tscan_positive_control.json`, and was wrong
  by up to 1.6% (18.53 vs **18.476**, 37.05 vs **36.465**). The measured column was right; the
  *predicted* column was the one I had essentially re-calculated by hand. Corrected to the artifact's
  three-decimal values.
- Permanent fix, then mutation-tested: the drift guard now checks **every cell of every row** of that
  table against the artifact (both columns), and reverting the table to my formula numbers makes the
  suite fail — verified by actually editing the README, running the test, and restoring. That is the
  same failure mode as the `6.52`-at-`0.15 mV` transcription error, one layer down: a table that looks
  self-consistent can be internally derived rather than measured.
- Suite **53/53 with cube**, 46/46 without. The 76-slot second-order tiling is at ~28 min of its
  projected ~31 and will report against the registered bars (corrected 2026-10-05 against
  `c3_2o_tiling.json`: noise-only **6.5092**, Bonferroni **6.6393**; the figures first written here
  were a mis-transcription, and the run finished at 914 s + 1,076 s = 13.1 s per slot per pass).

### 2026-10-05 — the plan's null-branch action: second-order scan, run and null

- Executed the one experimental action the plan's decision table attaches to the null outcome
  ("write up; escalate to capturer questions **+ second-order scan**"). `c3_2o_scan.py`, standalone
  numpy, project convention for "2O" = classical second-order CPA
  (`docs/notes/2o-localizer-vs-gpam.md`); `leakage_localizer/exp_2o.py` is ASCADv2/masking-shaped and
  was **not** reused.
- Configuration and bars were frozen in the script's docstring before the first run and the run
  reproduced them: slot 0–667 (one cadence period), J=94, C(667,2)=**222,111** pairs/hypothesis,
  M = 2.088e7 tests, noise-only max|t| **5.806**, Bonferroni **5.968**. Real scan max|t| **5.204** —
  under the noise expectation and under its own permuted-label null (**5.537**) → **SECOND-ORDER
  NULL**, the fourth independent one. (Also corrected: the registered prose said 222,778 pairs; the
  arithmetic is 222,111, and neither printed threshold moves.)
- **A control that cannot be detected proves nothing, and mine was exactly that.** First version
  added `a·h·w` to both samples; the product gains `a²h²w²`, whose correlation with `h` is
  `a²E[h³]E[w²]`, and `E[h³] = 0` for every centred template — symmetric HW, and ±1 with `h³ = h` of
  mean 0. It sat at noise level to 50 mV. Replaced with a **shared carrier modulated by the secret**
  (`Z[i] += a·h·w`, `Z[j] += a·w`): first order stays blind at both samples (measured |t| ≤ 1.13 at
  10 mV) while the product gains `a²h`, linear in `h`. Floor **1.0 mV** carrier; the algebra predicts
  `a = σ√(t/√n) = 0.98 mV`, so the instrument behaves as theory says. Both rejected artifacts are
  preserved (`.INVALID-control.json`, `.INVALID-control.log`).
- Kept the units honest: a 2O "mV" is a *carrier* amplitude, not class-mean separation. The
  correlation-equivalent first-order separation at the same `r_floor = 0.0306` is **0.171 mV**; the
  writeup quotes both and never merges them into one number with the 0.10 / 0.194 / 0.3 / 1.0 chain.
- Costed the remaining coverage instead of guessing it: **9.1 s per slot** at L=667 (7.8 s matmuls,
  1.2 s read) → **17.1 h** for 6,746 slots plus a 225 GB re-read; widening to L=1,334 is *worse*
  (25.2 h) because pairs grow as L². Full-cube 2O is a user decision (rules/04); the cheap targeted
  variant — pairs within each elevated event — is registered as the recommended first purchase if
  coverage is ever bought.

### 2026-10-05 — evidence sweep: every decimal in §6c against the artifacts, and one that had no artifact at all

- Automated cross-check of README §6c: 125 distinct decimal numbers, **8** with no literal occurrence
  in any artifact. Seven were legitimate 1-dp roundings or derived ratios (51.0 min from 3060.7 s,
  0.75 = 17,790/23,720, 102.2/104.5/139.1 null means, 145.0 GE). The eighth, **6.52**, existed only in
  prose — which led to the C24 finding above.
- Re-checked the SNR negative control against `c3_snr_meta.json` itself rather than the log: the
  between-variance ratio is **1.0088** and the mV ratio **1.0044**, while `snr_scan.log` printed
  "1.01×" beside mV numbers that give 1.004×. The scan's own printout mixes the two scales, so the
  writeup now states which is which and quotes the mV figure against the leak-free synthetic
  expectation (1.005–1.013×, `snr_numerics.json`): the cube sits at or *below* the noise value, which
  strengthens the null rather than weakening it.
- Repersisted the t-scan control (CPU, throttled to 8 BLAS threads so the concurrent T-028 job is not
  starved) and regenerated `controls_pass.json` from it: `all_pass` still true, network floor still
  1.0 mV, `t_scan_floor_mv` now 0.1 **with a source path field**, plus
  `t_scan_matched_at_full_train_mv = 0.109`.
- Verified the regeneration touched no data it shouldn't: the access log still has exactly one
  `purpose=experiment` line, and `--summarize-controls` reads cached eval JSONs without predicting.

### 2026-10-05 — full re-audit of the plan against the workspace, and one near-miss fixed

- Re-ran every verification command the plan lists, from evidence rather than memory: preflight
  **exit 0**; `test_c3_gpam.py --with-cube` **52/52** from the repo root **and from `/tmp`**, 45/45
  without the cube; all ten `docs/notes/metrics.md` names present in the eval JSON; `h5py.File(...,"r")`
  read-only; `per_class_counts(y,"val")` raises as specified; `c3_poi_regions.json` survives in the
  README only inside the supersession paragraph; forbidden GE paths appear only in prose explaining
  why they are excluded.
- Found and fixed the launcher near-miss described in the Decisions entry above (an accidental
  `C3_GO=1` with no `EPOCHS` defaulted to 150 and reached the GPU wait loop). No run started; the
  queue log carries the audit note.
- Added `c3_pred_dispersion.py` (val-split, 2 s GPU) and extended the README's val diagnostic section,
  todo **C23**, INDEX, and the drift guard, which now pins the histogram/uniform/network CE ordering.


### 2026-10-05 — one more question the writeup had not answered: is the head just echoing the histogram?

- `rank_final == prior_only_rank == 145` invites a cheap dismissal: *of course* they match, if the head
  outputs the same vector for every trace. `degenerate_flags` was empty, which rules out a literally
  flat output, but a flag is not a magnitude. So `c3_pred_dispersion.py` measures it on the **val**
  split — deliberately not test, which keeps its single `purpose=experiment` touch, and it reuses
  `predict_probs` rather than rebuilding the path. 2 s of GPU on the cached checkpoint.
- Answer: the head is **not** a constant. 145 distinct argmaxes across 2,000 traces, mean KL to the
  training marginal 0.00445 nats per trace. And the variation is worth *negative* information: val CE
  **5.547275** vs **5.5452** for a uniform guess and **5.550889** for a constant histogram predictor —
  it beats the histogram by 0.0036 nats while still losing to doing nothing at all, and on top-1 the
  constant guess wins (0.35% vs 0.25%). That is the exact signature of fitting noise, and it lets the
  null say "learned something that does not transfer" rather than "learned nothing at all" — the
  stronger and more defensible form.
- Guarded against future drift by extending `test_writeup_numbers_match_the_result_artifacts`: the
  artifact must keep showing non-constant output, `ce_histogram > ce_model > chance`, with the model
  within 0.01 nats of chance, and the README must quote each figure it leans on.
- Coincidence flagged in the README so nobody links them: the **145** distinct argmaxes and
  `rank_final` **145** are unrelated measurements.

### 2026-10-04 — Stages 7–9 complete: the THIRD NULL, and three floors beside it

- **Result.** `c3gpam_seed0_r1_20261004-170100`: 30 epochs, 17,790 steps, 51.0 min, 172.05 ms/step
  (calibration said 166, so the 20-step estimate was 3.6% optimistic), 7.088 GB peak. Train CE
  5.549 → 3.755; val CE bottomed at the **first** epoch (5.547362) and rose for 29 straight epochs to
  7.208673. Stop rule `triggered: true`.
- **Test split, once.** CE 5.547199 (0.0020 *above* chance), acc 0.44% / top-5 2.15%,
  **rank_final 145 = prior_only_rank 145**, percentile 0.051 against its own pairing null, GE 145.0,
  `traces_to_rank0` never, `degenerate_flags` empty. Verdict **THIRD NULL**; the registered bar
  (`rank_final < 7`) fails, and it failed against a number written 45 min before the result existed.
- **What makes it a measurement rather than a shrug.** (a) `c3_curve_compare.py`: permuted labels fit
  the training traces better at 30/30 epochs, mean +0.0462 nats — the wrong sign — and the same tool
  flips to −0.2292 with val below chance against the 1 mV injection, so it is an instrument with a
  demonstrated direction. (b) C22 landed a third floor: model-free SNR **0.3 mV** at the very site
  and template the network saw, so the network is the least sensitive of the three instruments
  (0.194 / 0.3 / 1.0 mV), which is the honest form of "the multivariate search is covered".
- **Two of my own defects, recorded because they were the point of running the controls.**
  C22's first noise band (cross-label dispersion) failed its 0 mV control and was replaced by a
  256-permutation null, with the rejected artifact kept. And `verdict_for` printed "better than" for
  two *equal* ranks (`<=`); fixed in the generator and re-rendered from cached numbers, deliberately
  **not** by re-running the evaluator, which would have added a second `purpose=experiment` touch.
- **Left for its author:** `docs/lab-log/todo.md:106` (row **T31**, the retracted resolution-preserving
  front end) renders as a 2-cell row inside an 8-column table — it is missing I/U/S/task/done. It is
  T-028's row and a session is live there, so I did not guess their scores; noted instead. My own
  stray-pipe and trailing-cell defects in the same files are fixed and the checker now ignores
  escaped `\|`.
- **Orchestration lesson worth keeping:** the chain that was supposed to fire Stage 8 the moment the
  trainer exited never fired, because `pgrep -f "<trainer command line>"` also matches any *other*
  shell whose command line contains that string — including the Monitor loop watching it. A
  completion check must match on the process, not on a command-line substring.


### 2026-10-04 — what the seed-0 run already looks like at epoch 23, written BEFORE the eval

Recorded now because it changes how the Stage-8 numbers must be read, and I refuse to decide that
after seeing them (`GPAM-c3/artifacts/logs/c3gpam_seed0_r1_20261004-170100.csv`):

- val CE by epoch: **5.547, 5.548, 5.553, 5.550, 5.556, 5.573, 5.589, 5.658, 5.728, 5.796, 5.812,
  5.931, 5.986, 5.985, 6.001, 6.042, 6.061, 6.414, 6.323, 6.331, 6.477, 6.506, 6.619, 6.650** while
  train CE falls 5.549 → 4.283. Chance is 5.5452; the pre-registered bar was val CE < 5.4952 and it
  has **never** been met, so the stop rule fires at epoch 30 as designed.
- This is the shuffled-label signature — memorise the training traces, pay for it on held-out
  traces — reached ~5× faster than the shuffle control did.
- **Consequence, fixed in advance:** the minimum val CE is at **epoch 0**, so `best.keras` — the
  checkpoint the frozen protocol says to evaluate (`c3_eval_gpam.py` loads `run["checkpoint_best"]`)
  — is the weights as they stood after **one** epoch (593 steps), when train CE was still 5.5493 ≈
  chance. The Stage-8 test numbers are therefore expected to be near-uniform, and `rank_final` will
  be a draw from the null band (mean 104.5, min 7 in 512 pairing draws), not a measurement of
  trained behaviour. **No checkpoint switching**: evaluating the epoch-29 weights instead would be
  a post-hoc change of protocol after seeing the curve, and it is the bar written before the result
  (`rank_final < 7`, percentile > 0.95, `prior_only_rank` > 100) that decides, whichever checkpoint
  the protocol nominates.
- **The honest wording this forces, written now:** the evaluated checkpoint is *not* what carries
  the "17,790 steps of exposure" claim. That claim rests on the val curve — 30 epochs, none of them
  >0.05 below chance — while the ranking measures one epoch of a head that had learned nothing. The
  writeup must keep those two separate and must not imply the tested weights had 30 epochs in them.
- What the eval still buys: `degenerate_flags`, the internal nulls, and the audit log entry that
  the test split was touched exactly once for `purpose=experiment`.

### 2026-10-04 — while the seed-0 run trains: C22 tooling, one env rule, document integrity

- Stage 7 is training (`c3gpam_seed0_r1_20261004-170100`), so everything here deliberately avoids
  the GPU and the cube's write path. The SNR **mV floor** sweep waits for the card.
- **Env rule added to `.grok/rules/01-environment.md`:** in `py10`, `import torch` raises
  `ImportError: .../libc10_cuda.so: undefined symbol: cudaGetDriverEntryPointByVersion` if
  TensorFlow was imported first in the same process; torch-first-then-TF is fine. So the GPAM (TF)
  path and the numpy+torch scan path are separate processes, and a script needing both must import
  torch at the top — and importing `c3_data.py` *counts* as importing TF here, because its guarded
  Keras import succeeds. Found by running `c3_snr_sensitivity.py --selftest`, which is why the new
  test in `test_c3_gpam.py` shells out instead of importing. Suite was **43/43 without the cube**
  at that point (45/45 at handover).
- **New:** `c3_snr_sensitivity.py` (site-scoped sweep + `--selftest` + `--numerics`),
  `artifacts/controls/snr_numerics.json`. **Changed:** `c3_snr_scan.py` — the per-window compute is
  now `window_stats()`, called from the loop with the arithmetic unchanged and verified against a
  float64 reference, so the sweep measures the same statistic the recorded null was produced with
  (and a future full scan can't quietly differ from it).
- **Document integrity.** GFM requires `\|` even *inside code spans* in a table row, and my earlier
  "4 tables inconsistent" report was mostly my own checker counting escaped pipes; with the checker
  corrected, five real defects remained and are fixed: two `max|t|` rows in README §6, a stray
  trailing cell on the T-019 row of `INDEX.md`, and unescaped `(|z| 56.5)` in the T-028 rows of
  `todo.md`/`INDEX.md`. README's stage ledger row 2b corrected **1.00× → 1.01×** to match (itself
  superseded on 2026-10-05: 1.01× is the variance-scale figure; the mV ratio is 1.004×)
  `snr_scan.log` (the prose had been fixed earlier; the ledger row had not), and row 4 no longer
  says "controls pending".
- **A second session is live in this repo right now** — T-028 is its active focus, and
  `docs/tasks/T028-*.md`, `INDEX.md` and `todo.md` all changed between my reads. Consequence for
  any session here: edit the shared hub files with surgical string replacements and re-read before
  editing; a whole-file rewrite would silently discard their rows.

### 2026-10-04 — Stages 5–6 complete: all controls pass, floor = 1.0 mV

- Summary: the detached chain finished ~06:57. **Every training succeeded; every eval failed.**
- Root cause: `drop_remainder=True` is mandatory for `model.fit` (Keras infers the output
  signature from the first two batches), so reusing the flag on the *evaluation* path looked
  right — but under `predict()` it silently discarded the last 16 of the 10,000 test traces,
  desynchronising probability rows from plaintexts. `c3_eval_gpam.py`'s own count guard caught it
  and exited 1, which is the only reason this was not a silent 1.6% row misalignment. Fixed by
  building the eval sequence with `drop_remainder=False`; pinned by
  `test_predict_path_keeps_every_trace`. Note why the earlier smoke dry-run missed it: I had used
  `--max-traces 256`, an exact multiple of the batch size.
- Re-ran all five evals (~2 min total; prediction is 157 batches in 5.6 s). Results above.
- Two findings worth more than the headline numbers:
  1. **Cross-entropy is a weak detection statistic here.** The 1 mV model ranked the key **first
     after 20 traces with GE 0** while its held-out CE improved only 0.163 nats. So the
     pre-registered 0.05-nat stop-rule margin is *weaker evidence than a 1 mV injection* and must
     never be read alone — the verdict now treats a full rank recovery as "beats chance", and
     `learned` was renamed `learned_by_ce` in the trainer so it cannot be mistaken for the floor.
  2. **Both internal nulls earned their keep on real data.** `prior_only_rank` came out **145** —
     exactly the value pre-computed before the run — on all five control evals, and the shuffle
     control's `rank_final` of 61 sits at percentile 0.789 (inside the null). Quoting "61" without
     those two numbers would have looked like a partial recovery.
- Instrument characterised before the experiment: overfit ✅, shuffle ✅ `CONTROL PASS`, floor
  **1.0 mV** against a matched univariate limit of **0.194 mV** (~5× less sensitive), `blind:
  false`, `controls_pass.json` → `all_pass: true`. (Tests passing at that point: the count has only
  grown since — see the 07:43 re-measurement, 52/52 with the cube, 45/45 without.)
- Verified the launcher is now gated *only* on the go-ahead: without `C3_GO` it refuses on exactly
  that one condition, and replaying its precondition block reads all-green.
- **Verified the test-split audit trail rather than asserting it.** All five control evals are
  logged `purpose=control` with `prior_experiment_evaluations: []`, and the `prior_evals` counter
  increments 0,1,2,3,4 across them — so "the test split is touched exactly once *for the
  experiment*" is backed by the log, as §6c and the audit table claim. Finding while checking: the
  smoke dry-run was recorded `purpose=experiment` (its `control` is `None`), which is misleading
  even though it is on `split=val` and cannot pollute the test audit. The existing log line is
  left as-is — rewriting an audit trail is worse than an odd label — but `c3_eval_gpam.py` now
  classifies smoke/calibration checkpoints as `purpose=dry-run` and says so on screen.
- Cross-referenced so nothing is re-run: todo **C13** marked SUPERSEDED with the floor result,
  T-027 opens with a pointer to T-029, and todo **C8** corrected from the stale 31/31 tests and
  265.9 ms/step → 6.6 h to 46/46 and 166 ms/step → **4.3 h**.
- Open: Stage 7 running (`c3gpam_seed0_r1_20261004-170100`, 30 epochs, ETA ~09:53 UTC); then the
  Stage 8 test eval against the measured bar (`rank_final < 7`) and the §6c results table.

### 2026-10-04 — pre-flight verification while Stages 5–6 run

- Summary: verified everything downstream of the controls *before* it is needed, so no result is
  discovered to be uninterpretable five hours from now.
- New file: `GPAM-c3/c3_injection_site_check.py` → `artifacts/controls/injection_site_check.txt`.
- Launcher gates: all three refusal paths tested and they **do** refuse (no `C3_GO`; missing
  `controls_pass.json`; present with `all_pass: false`, printing the full gate report first).
- Eval cost benchmarked at n=10,000: GE 100 attacks **2.3 s**, pairing null 256 draws **5.8 s**,
  rank curve <1 s, `log_scores` 20.5 MB. Each of the five evaluations is therefore ~2 min and is
  dominated by model build, not metrics — the `probs_*.npz` cache means GE never re-reads the cube.
- Fixed `c3_window.json`'s self-contradicting provenance (see Decisions) and made the writer
  write-once; added `test_window_json_is_write_once` and
  `test_frozen_window_file_matches_the_code` → **29/29 tests pass**.
- Confirmed `GPAM-c3/` is still gitignored (`.gitignore:145`) with zero tracked files.
- Verified the eval **overwrite guard**: re-running the same run/split refuses with "the split is
  evaluated once per run; re-running needs `--force` and a written reason", and does so *before*
  building a model or touching the GPU. `artifacts/eval/test_split_access.log` records the one
  smoke eval with `purpose=experiment`, `split=val` — so the real run's `split=test` audit stays
  clean.
- New file: `GPAM-c3/c3_window_gap_check.py` → `artifacts/c3_window_gap_check.txt`. It answers
  "what did the crop leave out" in energy and event terms, not just sample counts, and settles the
  pre-registered `L=57,344` fallback trigger (see Decisions). Region boundary re-confirmed
  independently: max std beyond sample 50,814 is 0.0079 vs 0.0308 inside, a factor of 3.9.
- Regenerated `c3_window.json` with the measured gap note; geometry and `first_written_utc`
  unchanged, so the frozen window was not redefined — only its documentation improved.
- **Metric-name audit against `docs/notes/metrics.md`** (the plan requires these verbatim). All 10
  present in the eval JSON with matching semantics: `rank_final`, `traces_to_rank0`, `ge_final`,
  `ge_at`, `ge_traces_to_0`, `mean_traces_to_rank0`, `median_traces_to_rank0`,
  `frac_reached_rank0`, `success_rate_final`, `ge_n_attacks`; plus all 5 `table11` fields
  (`acc_pct`, `mean_rank`, `max_rank`, `top5_acc_pct`, `chance_acc_pct`). metrics.md:107 requires
  `ge_final == rank_final` when the same full trace set is used — `test_ge_final_equals_single_full_set_rank`
  asserts it and the smoke eval confirms it (both 181). Two c3-namespaced names were missing from
  the per-eval JSON because they are control aggregates, not per-run metrics: added
  **`control_rank_final`** to `controls_pass.json` (the shuffled-label control's `rank_final`,
  which is the only thing the real run's `rank_final` is interpretable against) alongside the
  existing **`network_sensitivity_floor_mv`**, and surfaced both in the summary and in the
  launcher's precondition report.
- **The verdict logic is now a tested contract.** `verdict_for` had zero coverage, and it writes
  the single most consequential line in the whole task. It now returns `(label, text)` with four
  mutually exclusive real-run outcomes matching the plan's decision table — `DEGENERATE` (wins over
  everything, including an otherwise perfect-looking signal), `SIGNAL` (needs *all three*: CE
  >0.05 below chance, pairing-null percentile >0.95, `prior_only_rank` >100), `RANKING ARTIFACT`
  (`rank_final` ≤ 20 or CE below chance **without** both corroborations ⇒ explicitly "no recovery
  claim"), and `THIRD NULL` — plus `CONTROL PASS/FAIL` and `INJECTION FOUND/NOT FOUND`. The
  artifact branch is what stops a rank of 5 produced by class imbalance or luck from being quoted as
  a result; the null branch also states when the label prior alone beat the model. 6 new tests pin
  every branch, including "drop any one leg of SIGNAL and it must not fire" → **36/36 pass**.
- Shuffle control mid-run check (epoch 16/40): train CE 4.856 / train acc 6.07% while **val CE has
  risen to 6.366, i.e. 0.82 above chance**, val acc pinned at 0.20–0.40%. The model is memorising
  a wrong mapping and becoming confidently wrong on held-out traces — the strongest possible form of
  the control, and the gate `min(val CE) ≥ 5.5252` will pass with ~0 margin used.
- **Caught an unevidenced claim before it reached the writeup.** Auditing the artifacts behind the
  numbers I was about to publish showed `c3_snr_meta.json` records `"positive": false` and no log
  anywhere contains the "SNR detects 0.30 mV at 9.1×" figure that the approved plan text and my own
  earlier summaries both asserted. Struck from README §6c and §9 and from this file's Context;
  §6c now states explicitly which controls are persisted for which instrument, and re-running the
  SNR positive control is registered as **C22** rather than being inserted ahead of the T-029
  controls. The t-scan's floor was prose-only until 2026-10-05; it is now persisted in
  `tscan_positive_control.json` (0.10 mV) and read from there by `controls_pass.json`. The
  sensitivity floor this project quotes until the network's is measured.
- README §9 session log backfilled for 2026-09-26 → 10-03 (it had a single 09-25 entry) and a
  2026-10-04 T-029 entry added.
- **Closed a trap that would have produced a confidently wrong headline.** `c3_eval_gpam.py` with no
  `--run-name` selected "the latest `c3gpam_seed*.json`" by plain basename sort — and
  `c3gpam_seed0_smoke_...` sorts **after** `c3gpam_seed0_r1_...`, so Stage 8 would have evaluated
  the 2-epoch smoke checkpoint instead of the 150-epoch campaign and printed `THIRD NULL` for it.
  Selection now goes by the embedded `YYYYMMDD-HHMMSS` timestamp and excludes the `_smoke_` tag;
  with only a smoke run present it errors out rather than guessing (verified). `_latest` and the
  injection-sweep rollup use the same key. Pinned by
  `test_latest_run_selection_cannot_pick_the_smoke_run` → **37/37 tests pass**.
- Wrote the **Completion audit** section below: every plan requirement mapped to concrete evidence
  in the workspace, plus the five deliberate deviations. Nothing result-independent is left open.
- Shuffle control at epoch 7 shows the correct signature: **train CE falling** (5.5441 → 5.4381,
  train acc 0.42% → 1.56%) while **val CE rises above chance** (5.5565 → 5.6229) and val acc sits
  at 0.20–0.35% vs chance 0.39%. This also independently proves the train/val splits do not
  overlap — a leak would have shown up as val improving in step with train.
- Wrote README §6c (design rationale, control gates, injection-site calibration, both internal
  nulls, reproduction commands, results table pending) and de-duplicated §7 down to the stage
  ledger plus a pointer.
- Open questions: none blocking. If all four injection amplitudes are found, the floor is an upper
  bound (≤0.3 mV) and a downward extension becomes a follow-up decision — already registered in
  Decisions so it cannot be improvised later.

### 2026-10-04 — Stages 2–4 (calibrate, smoke, overfit) + Stage 5/6 launched

- Summary: caches built and bit-exact verified, throughput measured, the pipeline proven able to
  fit c3 at all, and the two remaining controls launched detached.
- Commands: `c3_data.py --split {train,test}`; `c3_train_gpam.py --mode calibrate`;
  `--tag smoke --epochs 2`; `--control overfit512`; then `setsid nohup /tmp/c3_controls_chain.sh`
  (progress marker `/tmp/c3_controls_progress.txt`, logs `/tmp/c3_{shuffle,inject_*}*.log`).
- Metrics: see the Stage 2/3/4 checkboxes above. Steady-state **166 ms/step** is the number that
  matters for budgeting; the 265.9 ms/step from `--mode calibrate` includes one-time PTX/kernel
  work and over-estimates a long run by ~60%.
- Early evidence that the internal nulls are earning their keep: on the 2-epoch smoke checkpoint
  (256 val traces, dry run of the eval path) `rank_final` was **181** and `prior_only_rank` was
  **180** — the rank is essentially *entirely* explained by the label frequency prior, and the
  pairing-null percentile was 0.547 (indistinguishable). Without those two numbers, 181 would
  have been quotable as "better than the 127.5 mean".
- Fixes made after Stage 1, each of which would otherwise have corrupted a control:
  1. `EarlyStopping` and the best-val `ModelCheckpoint` were active for `overfit512`, which runs
     400 epochs with a flat val curve; both are now real-run/val-using only, and validation is
     switched off for that probe.
  2. The injection template was standardised **per split**, so "10 mV" would have meant slightly
     different things on train, val and test. Now standardised analytically (uniform nibble std
     √(21.25), Binomial(8,½) std √2) with an assert that the empirical moments match, so
     `inject_mv` is the same ruler everywhere. A skewed label vector now raises instead of being
     silently renormalised.
  3. Keras drops the head-name prefix from metric keys with a single output, so the chance line
     was printing only the losses. It now checks both spellings and prints the chance accuracy
     (0.39%) and chance top-5 (1.95%) beside them.
  4. `C3CropSequence` did not call `super().__init__(**kwargs)`, which Keras 3 warns about and
     which silently ignores `workers`/`max_queue_size`.
  5. A failed GO/NO-GO gate now exits **2**, so the chained control script aborts instead of
     burning GPU hours on a run whose premise is already broken.
  6. `c3_eval_gpam.py --latest-of <prefix>` added so a chained eval can find the run JSON whose
     name contains a timestamp; and `permute_labels` is no longer passed at eval time (it only
     rewrites the y fed to `fit`, while the *injection* does change x and must be replayed).
- Open questions: none blocking.

### 2026-10-04 — Stage 0/1 (modules + tests)

- Summary: wrote all six planned files and got the unit-test gate green from three contexts.
- Files touched (all new): `GPAM-c3/{c3_data,c3_ranking,c3_train_gpam,c3_eval_gpam,test_c3_gpam}.py`,
  `GPAM-c3/queue_and_run_c3.sh`.
- Commands / runs: `test_c3_gpam.py` from the repo root, from `/tmp`, and `--with-cube`;
  `c3_data.py --split val` (via the test) → 0.10 GB in 5.1 s, 2% of rows re-verified bit-exact.
- Metrics / results: **26/27 passed, 1 skipped** (no cube) and **30/31 passed, 1 skipped** (with
  cube). The skip is `test_train_input_stats_are_the_affine_map`, which needs the train cache
  (Stage 2). Level-table fixtures confirmed from the cube: volts 4.8899775–5.0713110, global min at
  code 241 (ordinal 0), code 207 → 5.054955 V / ordinal 232, code 208 → 4.895667 V / ordinal 8,
  20 inversions, worst adjacent step −178.49 mV.
- Bugs found and fixed while writing the tests (each would have silently corrupted a result):
  1. The first internal-null design permuted **score rows**, which leaves every column sum — and
     therefore `rank_final` — unchanged. It reported percentile 0.5 for a perfectly informed model.
     Replaced with `plaintext_pairing_null`.
  2. `P(null >= obs)` reports 1.0 for a flat score vector (all draws tie at 255), i.e. a dead model
     would have looked perfect. Switched to the mid-rank convention.
  3. The overfit probe predicted through the **shuffled** training Sequence, which would have
     compared rows against the wrong labels. It now builds a fresh unshuffled Sequence over the
     same 512 traces.
  4. `EarlyStopping` was active for the controls; since a control's val curve is flat by
     construction, it would have cut the 40-epoch shuffle/injection budgets short. Now real-run only.
  5. `topk_acc(probs, y, 5)` crashes on <5 classes; `table11` now clamps k.
  6. Duplicate plaintexts bias the pairing null low (m² vs m·m′ collision terms); documented in the
     docstring and the test fixture uses distinct plaintexts.
- Open questions: none blocking. PTX JIT on Blackwell may add up to ~30 min to the first TF launch
  if the kernel cache is cold — budget +5 min at Stage 2, investigate past +20.

## Completion audit (plan requirement → evidence)

Checked 2026-10-04 against the workspace, not against memory. **Every row is now ✅** — the whole
plan (Stages 0–9) executed, and the result rows were verified against the artifacts after the fact
rather than written from memory.

| Plan requirement | Evidence | Status |
|---|---|---|
| 6 new files in `GPAM-c3/`, standalone, gitignored | `c3_data.py` 562 L, `c3_ranking.py` 358 L, `c3_train_gpam.py` 587 L, `c3_eval_gpam.py` 551 L, `test_c3_gpam.py` 904 L, `queue_and_run_c3.sh` 95 L; `git check-ignore` → `.gitignore:145`, `git ls-files GPAM-c3` empty | ✅ |
| Ranking vendored, not imported; no `ge_fixed_key_simulation`; no budget-GE | `c3_ranking.py` imports only `os`, `sys`, `numpy`, `c3_crypto.aes_sbox` — no TF, no h5py, no matplotlib; the two forbidden names appear only in prose explaining why they are excluded | ✅ |
| Frozen config: `c0=0 L=49152 patch=64 steps=768 B=64`, head `sbox1_byte_5`, `eff[5]=0x50`, ordinal input `min=0 delta=255` from train only, Adafactor lr 1e-3, `run_eagerly`, 150 ep, EarlyStopping(1e-4, 40, no restore), `drop_remainder`, `keras.utils.Sequence` | every run JSON header prints all of it; `test_steps_of_frozen_window`, `test_head_name_is_not_a_masking_name`, `test_train_input_stats_are_the_affine_map`, `test_sequence_drop_remainder_is_enforced`, `test_model_builds_with_one_head_and_no_plaintext_input` | ✅ |
| Decision-table null branch: "write up; escalate to capturer questions + second-order scan" | Write-up ✅ (README §6c/§6d/**§6e**, this file, todo C15–C28, INDEX). Second-order scan ✅ run **and covered**: one cadence period (bars 5.806/5.968, real max|t| 5.204 under its own permuted null, floor 1.0 mV carrier, first-order-blind control verified) and then the whole activity region (76 aligned slots, M = 1.5868e9, bars 6.5092/6.6393 registered before any peak). **The coverage run crossed its bar five times** and was adjudicated rather than claimed: the crossings are real structure (float64-exact, half-replicated, beyond 200 global and 300 block permutations) but they are a **per-trace acquisition state** recorded in `auth[15]`, not leakage — no mask identified, no AES template distinguishable from a null byte, and the same byte predicts inert-region noise (F = 10.3/18.4 vs 0.96/0.98) while leaving its mean alone (F = 1.2). The label nulls were then re-tested under that correction and survive (4.983/4.966/4.644, all calibrated under H0). Escalation ✅ registered as C9/C11, now sharpened to a checkable hypothesis about `auth`. Full-offset 2O coverage (6,746 offsets, 17.1 h) remains a user decision and is low-value now that the aligned tiling has run |
| Launcher refuses on every unmet precondition, before touching the log | `test_launcher_refusals_never_reach_the_wait_loop`: `C3_GO` missing, `EPOCHS` missing (no default), existing-run guard — each exits 1 and the queue log is unchanged. Added after an accidental `C3_GO=1` reached the GPU wait loop | ✅ |
| Pre-registered 30-epoch stop rule (threshold 5.4952) | `PreRegisteredStop` callback; smoke log shows it registered with the step count printed | ✅ |
| Artifacts under `artifacts/{checkpoints,logs,eval,cache,controls}/` | all five exist and are populated (4/5/3/9/3 entries) | ✅ |
| The ordinal input really is an exact affine rescale of volts | Measured from the cube's level table: 256 levels, 4.8899775–5.0713110 V, span 0.1813335 V; max deviation from the endpoint-anchored ideal grid **4.301e-07 V = 0.0006 LSB** (least-squares fit: 2.394e-07 V max, 1.376e-07 V rms); adjacent sorted diffs span 7.109642e-04 → 7.114410e-04, a spread of **4.768e-07 V = exactly 1.00 × float32 ULP at 5 V**. The grid is uniform to the precision of its own storage format, which is what licenses `min=0/delta=255`. Confirms the plan's "4.3e-7 V" figure and tightens the test to assert it | ✅ |
| Stage 0 preflight exit 0 | 27 checks executed, "ALL 27 CHECKS PASSED" (the plan's stated count of 28 was never true) | ✅ |
| Stage 1 tests pass from repo root, `/tmp`, and `--with-cube` | **37/37** without the cube, **43/43** with it, run from all three contexts (at handover: **45/45** without, **52/52** with). Includes `test_cube_val_split_lacks_a_class`, which measures why per-class val accuracy is refused: the 2,000-trace val split covers only **255 of 256** classes | ✅ |
| Stage 2 cache build + calibrate; 1% re-verify bit-exact; measured ms/step + VRAM replace estimates | train 1.87 GB/68 s (380/380 exact), test 0.49 GB/17 s (100/100), val 0.10 GB/5 s (40/40); **166 ms/step steady, 6.91 GB peak, 1,225,728 params** | ✅ |
| Stage 3 smoke: chance line 5.5452, epoch-1 CE ∈ [5.49,5.60], ckpt + CSV | CE **5.5493**, `best.keras` 10 MB, CSV 2 rows, `steps_actually_run` 1186 | ✅ |
| Stage 4 overfit512 GO (acc ≥95%, CE ≤0.5) | **acc 100.00%, CE 0.0000** in 3,200 steps / 8.9 min | ✅ |
| Stage 5 shuffle control + eval | **GO** — train CE 5.5452→4.036 while val CE rose to 8.30; `min(val CE)` 5.5469 ≥ 5.5252 over all 40 epochs; test `CONTROL PASS` (acc 0.28%, mean value-rank 126.93, `rank_final` 61 at percentile 0.789) | ✅ |
| Stage 6 injection sweep 10/3/1/0.3 mV + evals → `network_sensitivity_floor_mv` | **GO** — `rank_final == 0` with GE 0 at 10, 3 and 1 mV (1 mV reaching rank 0 after only 20 traces); 0.3 mV is a clean null (`rank_final` 156, percentile 0.420). **`network_sensitivity_floor_mv = 1.0`**, ~5× the matched univariate limit of 0.194 mV. **Wiring verified end-to-end on the real val cache** (`test_cube_injection_wiring_end_to_end`): `lsb_volts` 7.109642e-4 comes from the cache meta; 10 mV → **amp_ord 14.065 levels = 5.52% of full scale**, peak \|delta\| 16.27 mV against a local noise std of 5.58 mV; the delta equals `amp_ord × template` to 2e-3; everything outside samples 3000–3200 is **bit-identical**; labels untouched. | ✅ |
| Stage 7 seed-0 run, **explicit go-ahead**, `C3_GO=1 ./queue_and_run_c3.sh` | **DONE 2026-10-04 17:01–17:52 local** — launched through the real launcher with `C3_GO=1 EPOCHS=30` (user-scoped decision point), so every precondition was exercised for real. `c3gpam_seed0_r1_20261004-170100`: 30 epochs, 17,790 steps, 51.0 min, 172.05 ms/step, 7.088 GB peak; `stop_rule.triggered: true` (best val CE 5.547362 vs 5.495177) | ✅ |
| Stage 8 test eval once; refuses overwrite without `--force`; access log | **DONE 2026-10-04 09:54 UTC**: `c3gpam_seed0_r1_20261004-170100_test_eval.json`, access log shows exactly one `purpose=experiment` line (the 5 earlier `split=test` lines are all `purpose=control`), `prior_experiment_evaluations: []` | ✅ done |
| Stage 9 writeup: README §6c/§7, T-029 log, todo C8 + C15–C21, INDEX, README:609-611 | §6c now carries the frozen design, the controls table, the curve test, the three-floor comparison, the results table and the outcome; §7 reduced to the ledger; INDEX row and Active focus set; todo C8/C15–C22 closed out | ✅ done |
| Metric names verbatim from `docs/notes/metrics.md` + `table11` + namespaced c3 names | all 10 metric names and all 5 `table11` fields present with matching semantics; `control_rank_final` and `network_sensitivity_floor_mv` live in `controls_pass.json`; `ge_final == rank_final` on the same full set asserted by test and observed (both 181) | ✅ |
| Final acceptance: run JSON records `activity_coverage_pct`, `steps_actually_run`, `degenerate_flags`, `rank_final_percentile` | verified on the **real** run JSON: 96.729 / 17,790 / `[]` / 0.05078125, plus `rank_final` 145 = `prior_only_rank` 145 and `eval_test_json` pointing at the eval — the eval→run-JSON merge works (it was first proven on the smoke run: 1186 / 0.546875) | ✅ |
| Internal nulls computed on every eval, both from cached probabilities | `plaintext_pairing_null` (256 draws) and `prior_only_rank`; both exercised by the smoke eval (percentile 0.547, prior 180 vs rank 181) and pinned by 6 ranking tests. Prior-only reference pre-computed on real data by `c3_prior_check.py` → true key rank **145**, so imbalance cannot fake a low rank | ✅ |
| Superseded `train_gpam_c3_single_head.py` left in place, never run | not referenced by any new code; only the task file mentions it as superseded | ✅ |

### Known deviations from the plan text, all deliberate and documented

1. **Internal null is 256 pairing *draws*, not 256 *hypotheses*** — the literal reading is
   provably uninformative (see Decisions).
2. **`c3_window.json` provenance** records evidence rather than asserting a boolean, and the writer
   is write-once (see Decisions).
3. **The SNR positive control was not in the record** (fixed by C22: 0.3 mV), and the t-scan's
   floor was prose-only (fixed the same day: 0.10 mV persisted).
   (see Decisions; follow-up **C22**).
4. **The `L=57,344` fallback trigger is literally met** by events at 49,179 / 49,606; `L=49,152`
   stands because the trigger was written for label-dependent structure (see Decisions).
5. **Steady-state throughput is 166 ms/step, not the 265.9 ms/step cold figure**, so 150 epochs is
   ~4.3 h rather than the ~5 h budgeted.

## Next actions

| Action | I | U | S | Notes |
|--------|---|---|---|-------|
| ✅ Stage 2: caches + `--mode calibrate` | H | H | M | Done — 166 ms/step steady, 6.91 GB peak, 1,225,728 params; 150 ep ≈ **4.3 h**, not 5 h |
| ✅ Stage 3: 2-epoch smoke | H | H | M | Done — epoch-1 CE 5.5493, chance line prints, ckpt + CSV written |
| ✅ Stage 4: overfit512 probe | H | H | M | Done — **GO**, acc 100.00% / CE 0.0000 |
| ✅ Stage 5: shuffle-label control + eval | H | H | L | Done — CONTROL PASS on all three criteria |
| ✅ Stage 6: injection sweep 10 → 3 → 1 → 0.3 mV + evals | H | H | M | Done — floor **1.0 mV**, `controls_pass.json` `all_pass: true` |
| ✅ Stage 7: seed-0 run | H | H | L | Done — 30 epochs / 17,790 steps / 51.0 min, stop rule fired, **THIRD NULL** |
| ✅ Stage 8: eval test once | H | M | M | Done — all metrics + both nulls + flags; exactly one `purpose=experiment` touch |
| ✅ Stage 9: results tables + writeup | H | M | H | Done — README §6c/§7/§9, this file, todo C15–C22, INDEX row |
| ✅ **C29: geometry B — full campaign** | H | M | L | Done 2026-10-06 — all controls measured **at** B (overfit 100.00%/1e-05, shuffle `rank_final` 196, floor **1.0 mV** bracketed by a 0.3 mV miss), real 30-epoch run **fired the stop rule** at 17,790 steps: best val 5.545160 at epoch 1, val → 6.5603 while train → 4.2677. **FIFTH null**, README §6f. 38.5 min, 0.945 GB |
| **C29 (open): may B touch the 10,000 test traces?** | M | M | M | B's real run has **no** `rank_final` because the chain stops before it. Default position is **val-only** — CE, top-5, and the curve test against B's shuffled control. If a test ranking is wanted, B's registered bar is `rank_final < 2` from B's own band. **User decision; do not take it in code** |
| **C19: settle the exposure question** | M | M | M | 30 vs 40 epochs — **now for both geometries**. A rerun shares the seed and shuffle order and the val curve rose for 29 straight epochs, so it would in all likelihood nominate the **same first-epoch checkpoint** — buying an exposure *statement*, not a different ranking. ≈68 min GPU at A, **≈51 min at B** (B is the cheaper place to do it). **Ask the user; do not decide it in code** |
| Second byte / patch–L ablation | L | L | L | No longer mandatory: nothing beat chance, so there is no result to protect from the head-width == steps mechanism |
| Patch/L ablation + a second byte | M | L | L | **Mandatory** if anything beats chance (head input width == steps) |
| Escalate capturer questions | H | M | L | What does one trace cover? What is the 667-sample period? Is `OPc = f(K)`? What is `auth`? **Event alignment is no longer on the list** — `c3_event_lock.py` shows the events are phase-locked to ~1 sample (population lag profile peaks at τ=0, r=+0.182, falling to +0.068 at ±1), so trace averaging already is the alignment and jitter is not diluting any signal. That removes the cheapest GPU-free lever the plan listed as "the obvious remaining one"; C12 (second-order) and the capturer questions are what's left. |

### Priority snapshot

- **Do now:** two open user decisions and one offered edit. **C29** — whether geometry B's real run may
  touch the 10,000 test traces (recommended: no, keep it val-only). **C19** — 30 vs 40 epochs, now for
  *both* geometries, and B is the cheaper rerun. Plus the still-unanswered offer to reword the
  evaluator's `CE … (-0.002022 vs chance)` line, which reads as if the model beat chance when it did
  not. Everything else in the plan is executed and written, and a second geometry has since been
  executed and written too.
- **Before next meeting:** the deliverable is a *five*-floor statement — matched univariate
  **0.194 mV** (0.109 rescaled to n=38,000), t-scan in its own configuration **0.10 mV**, model-free
  SNR at a known site **0.3 mV**, multivariate GPAM **1.0 mV at two geometries** (patch 64/768 steps
  and GPAM's own patch 400/128 steps, each measured on its own head), second-order **1.0 mV carrier /
  0.171 mV equivalent** — attached to a null whose key ranking equals a label histogram's. Lead with
  `rank_final == prior_only_rank == 145` (geometry A's; B has no ranking yet, which is exactly what
  C29 decides) and with the curve test (permuted labels fit better at
  30/30 epochs), **not** with the CE margin, which is 0.002 nats and therefore soundbite-able.
- **And lead the second-order result with its interpretation, not its number.** The tiling crossed
  its own registered bar five times and every crossing is `auth_sboxHW_b15`; the adjudication chain
  (README §6e) says that is a **per-trace acquisition-state variable**, not leakage — `auth[15]` sets
  the noise level of a window that computes nothing (F = 10.3 inert, 18.4 active, against 0.96–0.98
  for `rand` and the label) and no XOR mask or AES-shaped template is identified. The label nulls were
  re-tested under that correction and survive (4.983 raw / 4.966 normalised / 4.644 quiet quartile,
  all below a 5.408 bar and calibrated at 4.77–4.83 under H0).
- **Then:** C9(b) is no longer a documentation request but a checkable question — *is `auth[15]` (and
  `auth[10]`) a gain or attenuation code?* — and it is the cheapest remaining lever, because if the
  capture ran at several analog settings, every scan so far pooled heterogeneous traces.
- **Second-order coverage is closed.** The tiling (76 cadence-aligned slots, 50,692 of 50,814 active
  samples, 32 min) replaced the "17.1 h" estimate, which had counted all 6,746 slot *offsets* rather
  than the phase-aligned coverage the events actually need. Full-trace 2O over every offset remains a
  user decision, and is now low-value: the one thing it found came from the aligned tiling.
- **The window fallback is closed.** The plan's contingency was to widen the crop to `L=57,344` if the
  run came back null while the 1,662-sample activity gap looked suspicious. Geometry B (51,200) already
  contains the whole region with a 0-sample gap and is null too, so there is no gap left to blame and
  no wider window to open.

## Related

- Task ids: T-027 (c3 standalone break — the two nulls and the activity finding), T-019 (GPAM
  ASCADv2 reproduction — where lr 5e-5 died and `rank()` ties bit us), T-028 (rank degeneracy).
- `docs/lab-log/todo.md`: C8, C15–C29.
- Plan: `~/.qoder/plans/tough-bluff-thrush.md`; earlier c3 plan `~/.qoder/plans/endless-boulder-moose.md`.
- Notes: `GPAM-c3/README.md` (§6c geometry A, §6e the capture-state finding, **§6f geometry B**),
  `GPAM-c3/NOTES-c3-confusions.md`, `docs/notes/metrics.md`, and
  **`docs/notes/c3-usable-trace-region.md`** — why only samples 0–~50,814 are usable, on what criterion,
  and what the discarded tail is *not* proven free of.
