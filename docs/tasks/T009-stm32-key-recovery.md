# T-009: STM32_AES_v2 key recovery

## Status

`done` — superseded. ASCADv2 attack work continues in T-019 (GPAM), T-024 (MultiResNet all-16 rank 0) and T-028 (non-worst-case). TimesNet multilabel here ended at ~random ranks (2026-10-04 cleanup)

## Goal

Recover AES key bytes on the **ASCADv2 / STM32_AES_v2** dataset (affine masking + shuffling), starting from dataset understanding and moving toward a working multilabel attack path (reference multi-ResNet and/or new models).

## Context

- Related data:
  - Extracted: `STM32_AES_v2/ASCAD_data/ASCAD_databases/ascadv2-extracted.h5` (~7.2 GB, present)
  - Raw (split): `ascadv2-stm32-conso-raw-traces{1..8}.h5` — local: #1 and #8 present; **#2–#7 are 0-byte placeholders** (raw regen incomplete; extracted DB is usable)
- Related code: `ASCAD_generate.py` (`multilabelize`), `ASCAD_train_models.py` (`multi_resnet` / `multi_resnet_without_permind`), `ASCAD_test_models.py` (`multilabel_predict`, recombination)
- Params: `STM32_AES_v2/example_{generate,train,test}_*_params`
- Papers: ASCADv1 eprint 2018/053; ASCADv2 eprint 2021/592; ANSSI secAES-STM32
- Contrast with prior local work: almost all training/benchmarks so far are **ATMEGA fixed-key** (single-label Sbox byte 2)

## Plan

- [x] (I:H U:M S:H) Inventory STM32 vs ATMEGA dataset structure and document differences
- [x] (I:H U:M S:H) Smoke-test eval path: labels/metadata consistency + oracle multilabel recombination
- [x] (I:H U:L S:M) Confirm pre-trained multi-ResNet status: **not local**; ANSSI zips are Keras `.h5` — current loader is `torch.load` only
- [x] (I:H U:M S:H) Draft TimesNet / ASCAD-head multilabel plan for ASCADv2
- [x] (I:H U:M S:M) Implement multi-task TimesNet + train/test wiring + code smoke (Tasks 1–3)
- [x] (I:H U:H S:L) Launch long GPU run (user-approved ≤20h): both multi variants, no-perm, train_len=100k, e=22, B=64
- [x] (I:H U:M S:H) Collect ranks from long run; write fixed-key-style summary JSON
- [x] (I:H U:H S:M) Plan fixes: fixed budget, stability, stride+segment pool, sanity ladder
- [x] (I:H U:H S:M) Implement train stability + improved multi TimesNet (s5 p32)
- [x] (I:H U:H S:H) S1 alpha overfit sanity (CE 5.57→5.21 on 512 traces)
- [x] (I:H U:H S:L) Improved retrain s5p32 fixed e=30 both models + eval (done)
- [ ] (I:H U:L S:L) Optional with-permind / multi-ResNet baseline — **ask first**

## Decisions

- 2026-07-17: New task registered; active focus switched from T-003 (GNN) to T-009 for STM32 key recovery exploration. T-003 remains open as `in_progress`.
- 2026-07-17: Do **not** rely on downloading ANSSI Keras multi-ResNet for the PyTorch path without a conversion step. Prefer train PyTorch `MultiResNet` from `ASCAD_train_models.py`.
- 2026-07-17: Attack eval on v2 must use `simulated_key=1` (variable keys).

## Attack path (verified)

```text
trace (15000) → MultiResNet heads:
  α (256), β (256),
  sbox_i (256) ×16,   [and optionally permind_i (16)×16]

recombine:
  Pr(α·S⊕β)  --dissect β-->  Pr(α·S)  --dissect α-->  Pr(S)
  [optional: unshuffle with Pr(permind)]

→ Pr(Sbox(p_j ⊕ k_j) | t) for j=0..15
→ log-likelihood rank vs key hypotheses (simulated_key=1 → true key byte maps to hyp 0)
```

**Mask fields used by labels** (from `multilabelize`):

| Field | Role |
|-------|------|
| `masks[0:4]` | shuffle seeds → `permIndices` |
| `masks[17]` | β additive mask |
| `masks[18]` | α multiplicative mask |
| rest of 38 | not used by this attack path |

**Oracle smoke (N=64 attack traces, plausibility check only, NOT a statistical result):**

- Label recompute from metadata: **0 mismatches**
- Oracle one-hot → recombine: **100% top-1** on true Sbox for both with-permind and without-permind
- Oracle ranks on smoke subset: bytes 0/2/15 show rank 0 on N=64 traces; **not statistically significant** (no z-score test, no shuffled-trace control, and small N can produce accidental ties in log-likelihood). This is a sanity check only, not a claimed result.
- `MultiResNet(depth=19, L=15000)`: **~137M** params (with permind) / **~78M** (without); **34** / **18** heads

## Working notes / session log

### 2026-08-13 — TimesNet catalog split

- Non-Scoop TimesNet (including this multilabel campaign) is summarized in **T-004**. Scoop-on-TimesNet is **T-011**.
- v2 “success” is **not** this multilabel path; it is black-box `TimesNetPooledClassifier` on **byte 4** (T-004 §C / T-011).

### 2026-07-24 — e60 done; e90 not triggered; meeting note

- e60 train complete (~4.2h); multi train CE **5.28** / val **5.29**; head **5.25** / **5.30**.
- Eval mean rank: multi **124.3**, head **137.2** (both ≥110) → watcher **STOP**, no e90.
- Meeting prep: `docs/lab-log/meetings/20260724-meeting-juncheng.md`

### 2026-07-24 — e60 resume + conditional e90 watcher

- Watcher: `STM32_AES_v2/watch_e60_then_maybe_e90.sh` (threshold mean rank &lt; 110).

### 2026-07-23 — improved retrain completed

- Wall ~**4.2 h** total (`elapsed_sec=14982`); multi ~2.1 h, ASCAD-head multi ~2.1 h; no NaN; full 30 epochs.
- Train CE (mean multilabel): multi **5.551→5.462** train / **5.462** best val; head **5.546→5.453** / **5.458** best val. Random ≈5.545 — real but small drop.
- Ranks @1000 attack traces (mean over 16 bytes, corrected from e60 JSON): multi **124.3** (min 18, max 246; no bytes with ttr0); head **137.2** (min 12, max 253; no bytes with ttr0). Still ≈ random. [Previous prose "129.6" / "126.4" / "ttr0@230" conflicted with stored JSON; JSON is authoritative.]
- Summary: `STM32_AES_v2/ASCAD_data/ASCAD_trained_models/summary_full_fixedkey_gpu.json`

### 2026-07-18 — improved retrain launched

- Code: grad clip, non-finite skip, best+last+sidecar, `train_heads`, `input_stride`, `pool_segments`, fixed-budget default for multi, non-interactive test exit.
- S1: alpha-only overfit 512×30e @ lr1e-3, stride5 pool32 → train CE **5.57→5.21** (learns, not fully memorizes).
- Job: `STM32_AES_v2/run_timesnet_multi_s5p32_retrain.sh` — both models, train_len=100k, e=30, **early_stopping=0**, stride=5, pool_segments=32, grad_clip=1.0.
- Logs: `logs/t009_timesnet_v2/retrain_s5p32_*`
- Ckpts: `.../timesnet_*_noperm_s5p32_t100k_e30.pth`

### 2026-07-18 — eval summary written (fixed-key JSON format)

- Script: `STM32_AES_v2/eval_timesnet_multi_summary.py`
- Outputs:
  - `STM32_AES_v2/ASCAD_data/ASCAD_trained_models/summary_full_fixedkey_gpu.json`
  - `STM32_AES_v2/ASCAD_data/ASCAD_trained_models/aggregate_full_fixedkey_gpu.json`
- Setup: multilabel=2, simulated_key=1, eval_traces=1000, rank_step=10, train_len=100k
- Results (mean over 16 key bytes):
  - `timesnet_multi`: mean rank_final **141.1**, min 18, max 255; no ttr0
  - `timesnet_ascad_head_multi`: mean rank_final **133.2**, min 3, max 245; no ttr0
- Training barely beat random CE (~5.54); ranks ≈ random (~128). Models did not learn a usable attack at this budget.

### 2026-07-17 — long GPU run launched (≤20h approved)

- Implemented `timesnet_multi` / `timesnet_ascad_head_multi` (pool + multi heads; pos-embed max_len ≥ seq_len).
- Smoke train OK: 512 traces × 1 epoch, loss ~5.59 (≈ random CE).
- **Job:** `STM32_AES_v2/run_timesnet_multi_noperm_20h.sh` via nohup.
- Config (each model): `multilabel=2`, `train_len=100000`, `epochs=22`, `batch=64`, `lr=1e-4`, RAdam, early_stopping, val 5%.
- Expected wall ~9h/model worst-case (~0.46h/epoch); both models + eval ≤ ~20h.
- Logs: `logs/t009_timesnet_v2/` · ckpts: `STM32_AES_v2/ASCAD_data/ASCAD_trained_models/timesnet_*_t100k_e22.pth`
- Summary JSON (after eval): `.../summary_timesnet_multi_noperm_t100k_e22.json`

### 2026-07-17 — TimesNet on v2 plan

- Drafted: `docs/superpowers/plans/2026-07-17-timesnet-ascadv2-multilabel.md`
- Core design: multi-task heads + **temporal pool** (no L×d flatten), `timesnet_multi` / `timesnet_ascad_head_multi`, without_permind first, data ladder 2k→20k→100k→420k
- Blockers for naive port: single-label head only today; L=15k flatten → 125M–495M params; full float32 profiling ~30 GB

### 2026-07-17 — smoke-test multilabel path

- Loaded `ascadv2-extracted.h5` attack metadata/labels; verified `multilabelize` math matches stored labels.
- Walked `multilabel_predict` / `multilabel_without_permind_predict` with ground-truth one-hots; recombination correct.
- No `ASCAD_trained_models/` weights locally; Readme ANSSI models are TF/Keras `.h5`, while `load_sca_model` uses `torch.load` → need PyTorch train (or a converter, out of scope for now).
- Train params reference: `example_train_models_params` → 420k train_len, batch 64, 60 epochs, early stopping, multilabel=1.

### 2026-07-17 — dataset comparison

- Compared docs + live HDF5 schemas (v1 fixed, v1 variable, v2 extracted).
- v2 needs multi-task + recombination; not a drop-in for single-label TimesNet/GNN stack.
- Raw #2–#7 empty; extracted DB sufficient.

## Next actions

| Action | I | U | S | Notes |
|--------|---|---|---|-------|
| Interpret ranks; decide next (scale data / arch / multi-ResNet) | H | M | M | ranks ~random at 100k |
| Optional: re-eval with eval_traces=10000 | M | L | H | same script flag |
| Optional with-permind TimesNet | M | L | L | After no-perm results |

### Priority snapshot

- **Do now:** monitor `logs/t009_timesnet_v2/run.log`
- **Before next meeting:** report mean rank_final for both variants
- **Deep work (park):** with-permind / multi-ResNet / 420k scale

## Related

- Plan: `docs/superpowers/plans/2026-07-17-timesnet-ascadv2-multilabel.md`
- Task ids: T-004 (all **non-Scoop** TimesNet, including this multilabel path), T-011 (TimesNet+**Scoop** only), T-001, T-005, T-006
- `docs/lab-log/todo.md` item T13
- External: [eprint 2021/592](https://eprint.iacr.org/2021/592), [ANSSI SecAESSTM32](https://github.com/ANSSI-FR/SecAESSTM32)
