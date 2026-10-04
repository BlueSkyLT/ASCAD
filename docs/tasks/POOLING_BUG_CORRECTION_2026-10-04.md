# Pooling Bug Correction — 2026-10-04

## Executive Summary

Two critical pooling bugs in `LLM/band_scan.py:85` and `LLM/amp_check.py:83` were causing a false claim that "averaging adjacent samples kills α". The bugs have been fixed, and all affected documentation has been corrected to prevent future analysis drawing wrong conclusions.

**The bugs:** `.mean(1)` instead of `.mean(2)` — pooled the group axis instead of the feature axis, returning only b features (2/4/8) per sample instead of n/b (800/400/200).

**Correct measurements:** With identical MLP, data, and seeds, block-mean pooling of 2/4/8 adjacent samples gives **0.893/0.909/0.900 vs 0.860 raw** — no loss, not a collapse.

**Corrected conclusion:** Block-mean pooling is **benign**. α leaks as a smooth, locally redundant waveform. The real training obstacle is optimization protocol (early stopping at epoch 6), not resolution loss.

---

## Files Fixed

### Code Bugs (2 files)

| File | Line | Bug | Fix | Impact |
|------|------|-----|-----|--------|
| `LLM/band_scan.py` | 85 | `.reshape(…).mean(1)` | `.reshape(…).mean(2)` | Pooling now returns n/b features per sample, not b |
| `LLM/amp_check.py` | 83 | `.reshape(…).mean(1)` | `.reshape(…).mean(2)` | Same fix |

### Documentation Corrections (4 files)

#### 1. **`docs/tasks/T028-v2-nonworstcase-pseudolabel-break.md`**
   - **Established Facts row 61** (RESOL entries): Replaced ~~"averaging kills α"~~ with correct pooling behavior
   - **Established Facts row 67** (mean-of-2/4/8): Corrected 0.0083/0.0065/0.0090 → 0.8930/0.9085/0.9000
   - **Phase 1R table A row 2**: Updated with correct measurements and bug reference
   - **Section C3 "Why TimesNet, Time-LLM and PatchTST failed"**: Already comprehensive; added note that code bugs are now fixed

#### 2. **`docs/tasks/T020-llm-transformer-sca-pipeline.md`**
   - **Spot-check entry (line 88)**: Marked the RESOL artifact comparison as retracted due to pooling bug
   - Added cross-reference to T028:C3

#### 3. **`docs/notes/patch-pipeline-survey.md`**
   - **Lines 30–34** ("Explicit averaging is fatal"): Retracted premise and explained correct behavior
   - **Lines 208–214** (table): Corrected all three pooling results with [CORRECTED] markers and measurements
   - **Lines 254–260** (conclusion table): Updated "explicit mean" outcome from "fatal" to "benign" and rewrote conclusion

#### 4. **`docs/lab-log/todo.md`**
   - **T31 line**: Marked entire task as RETRACTED; explains why the resolution-preserving-front-end premise was wrong

---

## What the False Claim Said vs. What's True

| Claim | Source | Now Known |
|-------|--------|-----------|
| Averaging 2 adjacent samples: 86% → 0.008% accuracy | T-020, T028 line 61, patch-pipeline-survey | Correct pooling: 86% → 89.3% (no loss) |
| "Resolution is a binding constraint" | T028 lines 314–317 | Block-mean pooling is **benign**; constraints are different |
| "Early pooling is the binding constraint" for A2/TimesNet failures | patch-pipeline-survey.md, T-020 | Real constraint: **training protocol** (early stopping at 6 epochs) |
| "Patch models' obstacle is pooling, not localization" | patch-pipeline-survey.md:258 | Obstacle is **localization + training length**, not averaging |

---

## Why This Matters for Future Analysis

1. **Don't re-derive the "averaging kills α" number.** It was based on a measurement bug. Use the corrected entry in T028 Established Facts row 67.

2. **"Per-sample resolution is essential" is NOT supported by this data.** The premise collapsed, and the evidence now points to:
   - Smooth, locally redundant leakage waveform (R4a/R4b show adjacent samples correlate +0.99 in class means)
   - Block-mean pooling raises accuracy (mean-of-2 median R² 0.233 vs individual 0.186/0.196)
   - The training protocol (GPAM needs 13 epochs to break through; A2 stopped at epoch 6)

3. **"TimesNet v2's stride-5 subsampling killed the model" is NOT established.** The stride itself is fine (R4a: each offset keeps ~20% of band excess); the v2 failure could be early stopping, fewer epochs, or a trunk issue. See T028:C3 table for what's still unknown.

4. **Labels like "resolution verdict" or "RESOL entries" in task notes now carry wrong conclusions.** Future readers should check T028:C3 before citing them.

---

## Measurements Preserved (Still Valid)

✅ **α peak (sample 867) leaks HW(log₃ α) at R² 0.651** (R1b) — unchanged  
✅ **α's 871-sample band, SNR > 0.1** — unchanged  
✅ **β peak (sample 338) is near-Hamming-weight at R² 0.601** (R1b) — unchanged  
✅ **GPAM achieves 92.95% on crop, 98.34% on raw with grouped attention** — unchanged  
✅ **3-leak product u×β×C replicated on 450k traces, Stouffer z +6.04** (R9b) — unchanged  
✅ **MultiResNet 34-head ranks all 16 bytes 0 at z +3.89σ to +6.86σ** (T-024) — unchanged  

---

## References

- **Full technical account:** T028 section C3 ("Why TimesNet, Time-LLM and PatchTST failed — and a retraction")
- **Replication script:** `experiments/r4c_pooling_bug_replicate.py` (20k/2k traces, 2 repeats)
- **Adjacent structure analysis:** `experiments/r4a_adjacent_structure.py` (50k traces, ORACLE)
- **Commit:** `a42b9cb` — "Fix pooling bugs in band_scan.py and amp_check.py; correct related analysis docs"

---

## Going Forward

All notes in T028 ("Established Facts") and this correction document supersede earlier claims about resolution/pooling. Anyone citing the patch-pipeline-survey, T-020, or T-028 for baseline numbers should verify they are reading the corrected version (git commit a42b9cb or later).

The real unsolved questions for patch-based models remain:
- Band localization without labels (R3a/R8–R10 ruled out variance and simple statistics)
- Training protocol that allows breakthrough (GPAM: 13 epochs at 1e-3 LR; our models: 6-epoch early stop)
- Whether a longer training window (150+ epochs, no early stop) would let patch models work
