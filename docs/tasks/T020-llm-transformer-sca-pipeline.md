# T-020: LLM/Transformer SCA — Patchify → Interpret → Classify

## Status

`paused` — focus moved to T-028. Last state: α readable from 16 SNR bands (oracle band choice); open problem is finding bands without labels (2026-10-04 cleanup)

## Goal

Use the "patchify traces → train to find meanings → map to features → interpretation model → classifier" pipeline (Time-LLM-inspired) for S-box recovery. Final metric: **key rank 0** on ASCADv2's sbox value.

## Context

- Related code / data paths:
  - `LLM/patch_transformer_sca.py` — new multi-strategy implementation (this task)
  - `leakage_localizer/exp_v1_timellm_gpt2.py` — previous Time-LLM attempt (T-016)
  - `LLM/models.py` — deduplicated PatchGPT2SCA
  - `leakage_localizer/data_v1.py`, `leakage_localizer/data_v2.py`
  - `ASCAD_train_models.py` — CNN/MLP/MultiResNet baselines
- Related docs:
  - `docs/LLM-plan.md` — original methodological spec
  - `docs/notes/v1-poi-faq.md` — **critical** leakage properties reference
  - `docs/notes/ascadv2-direct-evidence-explained.md` — v2 leakage analysis
- Related tasks: T-016, T-017
- Budget: 5 hours wall clock, quality metric = ASCADv2 sbox rank 0

## Decisions

- 2026-09-23: Created T-018 as distinct from T-016 (which was Time-LLM GPT-2 smoke runs).
- 2026-09-23: Adopted budget-aware analysis protocol (time: 5h, quality: ASCADv2 sbox recovery).

## Working notes / session log

### 2026-10-02 — visual ASCADv1 and ASCADv2 architecture diagrams created on Whimsical

- **ASCADv2 Board (`S4sZGcUkdCNUPGzTGpcMt1`) Updated & Refined**:
  - Live URL: `https://whimsical.com/S4sZGcUkdCNUPGzTGpcMt1` (89 shapes, 99 connectors).
  - Shape conventions enforced:
    - `cloud` for all items/waveforms/tokens/representations/predictions (true SVG bezier clouds).
    - `pill` for normalization/rescaling steps.
    - `rect` for all methods, models, transformations, and processes.
  - Operation-based block fill color mapping:
    - Rescaling / Normalizing: Green pill (`#e8f5e9` fill, `#2e7d32` stroke).
    - Cropping / Windowing: Blue rect (`#e3f2fd` fill, `#1565c0` stroke).
    - Attention / Transformers / GPT: Purple rect (`#f3e5f5` fill, `#6a1b9a` stroke).
    - Dot Product / Bilinear / Dissection: Red rect (`#ffebee` fill, `#c62828` stroke).
    - Spectral / FFT / Convolutions: Orange rect (`#fff3e0` fill, `#ef6c00` stroke).
    - Pooling / Gating / Selection: Yellow rect (`#fffde7` fill, `#fbc02d` stroke).
    - Linear Projections / Classifiers / Heads: Silver rect (`#f5f5f5` fill, `#424242` stroke).
    - Representations / Tokens / Items: White cloud (`#ffffff` fill, `#424242` stroke).
    - Secret Key Recovery: Yellow cloud (`#fff9c4` fill, `#f57f17` stroke).
  - Method family arrow color mapping:
    - 🟢 Green: GPAM (Grouped Attention, Multi-task, Algebraic dissection).
    - 🟠 Orange: TimesNet (1D FFT, 2D Periodicity, Inception blocks).
    - 🟣 Purple: Patch-Transformers & LLMs (Intra/inter-patch attention, Time-LLM, Autoencoders, Soft-Window).
    - 🔵 Blue: Segment GNN (Peak/valley segmentation, GraphSAGE).
    - 🔴 Red: 2nd-Order Localizer (Sliding window proposer, Centred cross-product).
    - ⚪ Silver: Cross-trace / Extraction linkage.
  - Explicit multi-step sequences:
    - TimesNet FFT: `[Extracted Traces]` (cloud) $\rightarrow$ `[1D FFT]` (rect) $\rightarrow$ `[Dominant Periods]` (cloud) $\rightarrow$ `[Select Top-k]` (rect) $\rightarrow$ `[Top-k Periods]` (cloud) $\rightarrow$ `[Reshape 2D]` (rect) $\rightarrow$ `[2D Period Blocks]` (cloud).
    - GPAM Algebraic Secret Key Recovery:
      `[34 Multi-Task Heads]` $\rightarrow$ `Pr(alpha)`, `Pr(beta)`, `Pr(C_i)`, `Pr(pi_i)` $\rightarrow$ `[proba_dissect_beta]` (un-XOR) $\rightarrow$ `Pr(alpha*S_i)` $\rightarrow$ `[proba_dissect_alpha]` ($GF(256)$ inverse mult) $\rightarrow$ `Pr(S_pi(i))` $\rightarrow$ `[proba_dissect_permind]` (un-shuffle) $\rightarrow$ `Pr(S[P_i^K_i])` $\rightarrow$ `[Key Hypothesis Sum Log-Likelihood Accumulator]` $\rightarrow$ `[GPAM Key Guess (Rank 0 Recovery)]` $\rightarrow$ `[True AES-128 Secret Key]`.
  - Added direct connector between `[Raw AES Traces]` and `[Extracted POI Window]` (`crop POI (15k)`).
  - Auto-layout applied (`left-to-right`), visual preview verified via rendered snapshot.

- **ASCADv1 Board Created (`Do1A6rjjm4SUxtB8LYk9Z`)**:
  - Live URL: `https://whimsical.com/ascadv1-patch-based-dl-sca-architectures-Do1A6rjjm4SUxtB8LYk9Z?ref=mcp` (57 shapes, 67 connectors).
  - Focused specifically on Patch-Based Deep Learning SCA pipelines:
    - Patchify Routes: $p16/s8$ (124 tokens), $p32/s16$ (61 tokens), $p64/s32$ (30 tokens), and 1D Conv Stem with stride 8 $\rightarrow$ LayerNorm.
    - Intra-patch sample attention (local micro-structure) vs. Inter-patch token attention (global macro-structure).
    - Self-Supervised Pretraining: TimeMAE (latent masking), SimMIM (embed masking), VQ-VAE (codebook vocab 32) $\rightarrow$ Reconstruction Decoder $\rightarrow$ Reconstructed Traces.
    - Time-LLM Adaptation: SCA domain prompt + 1,000 vocab prototypes reprogramming $\rightarrow$ Frozen GPT-2 (124M) $\rightarrow$ Contextual Embeddings.
    - Trainable ViT-1D: Learnable pos_embed $\rightarrow$ 4-layer bidirectional TransformerEncoder.
    - The **Pooling Dilemma**:
      - Uniform Mean Pooling failure: collapses 2nd-order leakage to chance (CE 5.545, Acc 0.39%, Rank 245).
      - Proof of token leakage: 9-way Hamming Weight classifier on individual tokens reaches 28% accuracy (vs chance 11%).
      - Resolution-Preserving Readouts: Soft-Window Attention Gating ($h_\psi$) and PairWords Bilinear Cross-Product ($x_i^\top W x_j$).
      - 2O Sliding-Window Localizer: BandAttention + Centred Cross-Product reaching Rank 0 on Byte 2 at 120 traces.
      - Baselines: Standard CNN (Conv1D + MaxPool) and 5-layer MLP for reference.
  - Strict $\le 10$ words per block constraint enforced across all shapes.
  - Auto-layout applied (`left-to-right`), visual preview verified via rendered snapshot.

### 2026-09-29 — full survey of the patch → attention → classify family written up

- Summary: read-only survey of every variant built and every recorded result, now at **`docs/notes/patch-pipeline-survey.md`** (catalogued in `docs/notes/README.md`). ~40 named configurations across 6 families, ~30 recorded runs, **zero learned `rank_final` 0** on either dataset; 46 files in the repo contain `FAIL_CHANCE`.
- Family map: A frozen-GPT-2 + mean pool (`LLM/models.py:42`, `leakage_localizer/exp_v1_*`) — at chance and **rank never computed**; B trainable transformer + attention/bilinear/flatten (`LLM/patch_transformer_sca.py`) — `rank_final` 94–245; C resolution-preserving readouts (`LLM/soft_window.py`, `LLM/pair_words.py`) — best is `SW-beta` atk acc 0.0354 ≈ 9.1× chance with no rank; D sliding-window proposers (2O/3O) — the only rank-0 claim, unpersisted; E GPAM grouped attention — `rm`(α) 92.95 % holdout, worst-case; F TimesNet/MultiResNet/CNN/GNN-segments as contrast.
- **Two informative negatives worth quoting at meetings:** `PW`/`PW2` `rank_final` (199/190) equals their `control_rank_const_trace` (196/190) — the pair head has **zero trace dependence**; and the `SW-alpha` learned gate **collapsed onto padded sample 14992** (entropy −0.0, `eff_positions` 1.0).
- Verified dead code, all three confirmed by grep: `MultiLabelWrapper` (`patch_transformer_sca.py:487`) never instantiated; `--use-lora` builds `lora_modules` (`:296,:302`) that `forward()` never calls; `LLM/config.py` entirely unimported. Also `GPAM/train_gpam_ascadv2_budget.py:183` raises `SystemExit(GE_BROKEN)` — that script cannot run.
- **CORRECTED 2026-10-04:** the §7 collapse table (`BANDCTL16-contiguous.json` 0.8603 → `RESOL-contig16-b2.json` 0.0083) was based on a pooling bug in `LLM/band_scan.py:85`. Correct pooling gives 0.893/0.909/0.900 for blocks 2/4/8, not 0.008. See T028:C3. Other spot-checks stand: `C-gpam-eval.json:53` 92.95% confirmed, `A-verdict.md` retraction confirmed.
- Open doc conflicts recorded in the survey's §9 register rather than smoothed over: `trace-signal-anatomy.md:95` "location/dilution" framing vs `A-followup:68-73` "pooling is the obstacle"; `roadmap:330` "AOD-GE unimplemented" vs AOD-GE artifacts found on disk; `roadmap:189` still attributing the failure to SNR(S) ≈ 0.02 after the retraction; A1 "crashed" prose vs its own complete artifact.
- Live status, **corrected after an over-hasty interim read**: at epoch 5 the 40 k gate probe looked dead (`val rm` oscillating 0.098–0.488 % against a 2 % launch bar, val gap −0.0863, worse than chance) and I recorded "the gate is not passing". **Epoch 6 broke through: `val rm` 17.871 % (45.7× chance), val loss 17.0840 vs chance 19.4081, gap +2.3241** (`GPAM/artifacts/logs_raw/probe40k_20260929_202123.log`). Only `rm` moved — `rout` 0.684 %, `c_0` 0.391 %, `p_0` 6.54 % stay at chance, the same α-only signature as the crop run. Gate was fail-closed and read the final epoch: epochs 7–8 gave 76.270 % then **98.340 %**, val loss 13.4763 vs chance 19.4081, `PROBE GATE VERDICT: PASSES`, and the ~83 h 34-head campaign launched at 21:23:16 (pid 1083690). Full record in `docs/notes/patch-pipeline-survey.md` §10 and `T019`.
- Process lesson for this task: **do not extrapolate a GPAM curve before it breaks through.** `T019:12` already says this; this run proved it with a 37× single-epoch jump after five flat epochs.
- Files touched: `docs/notes/patch-pipeline-survey.md` (new), `docs/notes/README.md` (catalog row), this note. No run launched, no GPU touched.

### 2026-09-25 — α coverage curve + selection controls: the "drop the family" verdict is retracted

Continuation of task-A under a 10 h budget (started 01:17; wall clock ran to ~39 h because the
machine slept between runs). New code: `LLM/band_scan.py`, plus `--repeats`/seeding in
`LLM/amp_check.py` and `--select` controls. Full write-up:
`docs/2026-09-23/out/A-followup-where-alpha-leaks.md`; `out/A-verdict.md` now carries a
correction block retracting its own sentence 1.

- **Coverage curve (α, MLP, 2 reps, chance 1/255):** 1 SNR band (100 samples) 11.8×; 2 bands
  119×; 4 bands 208× (top-1 0.8153); 8 bands 230×; **16 bands (1,600 samples, 10.7 % of the
  crop) 0.9395 = 240×**; 32 bands 0.9653. Median rank of the true α hits 0 at 4 bands.
  So ~11 % of the trace matches the parallel session's full-crop GPAM `rm` (92.95 %).
- **Selection controls, same sample budget — and one refutes my own framing:** contiguous
  samples 0–1600 → 0.8603; random (4 draws) → 0.5622 ± 0.2002 (range 0.347–0.783); evenly
  spread → 0.5240; **lowest-SNR 16 bands → 0.0065 (1.7×) and lowest-SNR 4 → 0.0035 (0.9×)**,
  i.e. guessing. At 4 bands: SNR 0.8153 vs random 0.2081 (range 0.007–0.484).
- **Revised mechanism claim: the obstacle is averaging samples, not looking in the wrong
  place.** α is spread over roughly a third of the crop, so any wide draw catches some of it;
  SNR ranking buys reproducibility (±0.002 vs ±0.20) and small budgets, not "the needle".
  Whole-crop block-pooled (1,500 features) = 1.4×, and `A2`'s attention pooling over 1,874
  tokens = 0.77× — both are weighted averages of what needs per-sample resolution.
- **Methodology, two lessons recorded the hard way.** (1) `amp_check` was unseeded: identical
  settings gave 0.0300 and 0.1640 on one window; it now seeds per repetition and reports the
  spread. (2) My 60-window scan silently used `--block 10`, so those accuracies are a *pooled*
  scale (best 0.0300) and are not comparable to the per-sample headline (same window 0.1544);
  the SNR-vs-accuracy correlation r = 0.939 still holds.
- **Open problem for the LLM line, and it is the real one.** Band choice here used ground-truth
  labels via SNR, so it is an oracle for understanding the data, not an attack. The unsolved
  question is finding the live bands *without* labels — attention, hierarchical/Coarse-to-fine
  gating, or sliding-window compression to a short sequence that keeps per-sample detail. And
  15,000 points × 16,000-sample selection is already the "long trace" problem the plan kept
  hitting on v2.
- Still worst-case throughout: α and β are mask labels; no key, no rank, not a break. The v1
  product probe (rank 0 in 120 traces; same regions additively → rank 233) is the only
  key-rank statement, and it is hand-placed.

### 2026-09-24 — α really does leak, a small readout does see it, and A2's premise was unverified

Resumed brief A as an open research task (10 h budget from 01:17, bar = leave chance by any
significant margin; controls reported but not gating). New code: `LLM/snr_check.py`,
`LLM/amp_check.py`, `LLM/pair_words.py`, `LLM/soft_window.py`.

- **The plan quoted SNR(α) ≈ 2 with no saved artifact — measured it.** Pooled between/within
  per-sample variance ratio on 20k extracted traces: **α 1.986 at sample 867**, β 1.520 at 338,
  `sbox_masked`[byte 0] 0.0678 at 5552, median ≈ 0.016 everywhere. Calibration row: v1
  unmasked S byte 2 → **0.0161**, matching the repo's recorded ≈0.02, so the measure agrees
  with the number the plan trusted. Roadmap gap 3 (SNR(S) computed but never saved) is now
  closed: `docs/2026-09-23/out/SNR-v1-S.json`.
- **A small MLP on samples 800–900 reads α at 12.1× chance**: attack acc **0.0475** on 2000
  held-out traces (95 hits vs 7.8 expected, z ≈ 31); raw 0.0290, global 0.0330, per-position-
  mean-only 0.0165. Controls behave: label-permutation null 0.0015–0.0070, constant-input
  0.0030–0.0045, and an **off-peak window 5000–5100 is at chance** (0.0025–0.0060).
- **A2's NO-GO needs rewording.** α is readable; the attention-pooled patch transformer missed
  a leak confined to a ~60-sample band out of 1874 patch tokens in 6 epochs. The failure is
  localization + training protocol, not "the family cannot see the leak".
- **My own amplitude hypothesis was wrong, in the opposite direction.** Per-trace z-scoring
  (`LLM/data.py:124`, applied by every LLM-path loader) does not destroy α: ztrace 0.0475 >
  global 0.0330 > raw 0.0290. Tested and killed in one run.
- Still worst-case: α is a mask label. No key, no rank, not a break. MLP attack-set CE is
  39–84 (grossly overconfident) — accuracy is the usable metric; do not push these logits
  through a rank/GE accumulator without temperature scaling.
- v1 side finding stands: the hand-placed **centered product of task B's two regions** (mask
  p18–22 × masked output p64–67) reaches **rank 0 in 120 traces**, while both regions given
  additively reach rank 233 — so the obstruction on v1 is multiplicative structure. Four
  learned attempts (attention, flatten, free N×N atoms, separable atoms) all failed to *find*
  that pair, each at ≤2k optimizer steps with train CE barely moving.

### 2026-09-24 — A2 NO-GO: `alpha_mask` stayed at chance. Drop this family for v2

- A2 (`--v2-target alpha_mask --pool attention`, 20k prof / 2k atk, 1.09M params) → `FAIL_CHANCE`. Attack **CE 5.54696** vs chance CE 5.54518, attack **acc 0.0030** vs chance \(1/255=0.00392\). Best epoch 2, val CE 5.54555. Stopped by patience=4 at epoch 6 after **5.3 min** of a 4 h cap; the brief's own early-stop bar was also met (epoch 6 val acc 0.0055 < 0.008, val CE down only 0.0041).
- Train CE fell below chance (5.5530 → 5.5371) while val stayed flat — the head fit the label marginal, not the leak. Same failure shape as `docs/notes/ascadv2-label-frequency.md`.
- A1 (v1 byte-2 `flatten` control) **never ran**: `--pool flatten` dies at `LLM/patch_transformer_sca.py:226` because line 189 sets `self.pool = None` and `forward` has no flatten branch. `TypeError: 'NoneType' object is not callable`, exit 1 after 2 s. So the brief's premise that the class already implemented flatten was wrong; no `rank_final` / `traces_to_rank0` exist for it.
- Files: `docs/2026-09-23/out/A-verdict.md`, `A2-v2-alpha.json`, `A2-v2-alpha.log`, `A1-v1-flatten.log`, `baton.log`. Launcher: `docs/2026-09-23/overnight-baton.sh`.
- Meeting sentence: we stop treating a trainable patch transformer as a route to ASCADv2 — it missed a label with SNR ≈ 2. Next GPU work, if any, is not this family.

### 2026-09-24 — brief A launched; one flag added to `patch_transformer_sca.py`

- Code change (only what brief A allows): `--pool` now accepts `flatten`; added `--v2-target {alpha_mask,beta_mask}` defaulting to `alpha_mask`, and the v2 branch reads that field instead of `sbox_masked`. `sbox_masked` is no longer reachable by default. The 256-way head, training loop and v1 path are untouched; key rank stays v1-only.
- `alpha_mask` verified in `ascadv2-extracted.h5` as `(500000, 1)` uint8, so the label is taken with `[:, 0]`. Class 0 never occurs → compare accuracy to 1/255 (0.00392), not 1/256, even though the script still prints `chance=0.0039`.
- CPU plumbing check at 00:57 (`docs/2026-09-23/out/WIRING.*`, 256 traces, 1 epoch, d=32) reached the loss/CE/json path. Its `VERDICT: ABOVE_CHANCE` is a 256-trace fluke of the 2x-chance test, not a signal — do not cite WIRING anywhere.
- A2 (`alpha_mask`, attention pool, 20k prof / 2k atk, 12 epochs, 4 h cap) took the GPU at 00:56:56 after brief C's GPAM smoke exited early. ~49 s/epoch, epochs 1–2 sitting at CE 5.55 / acc 0.004–0.006 = chance. Baton will then run A1 (v1 byte-2 flatten control) only if the clock is still before 06:00, and hard-stops at 07:50.
- Verdict numbers and the meeting sentence go in `docs/2026-09-23/out/A-verdict.md` when A2 exits. Not written yet.

### 2026-09-24 — why the probe target is v2 `alpha_mask`

- \(\alpha\) exists only on ASCADv2 (affine multiply hide, `alpha_mask` = `masks[18]`). ASCADv1’s leaking shares are Boolean \(r\) and \(C\). The 3O oracle on extracted v2 measured SNR(\(\alpha\)) about 2 and SNR(\(\beta\)) about 1.5, with the hidden S-box share near 0.016. The probe asks whether a patch transformer can predict that label. It is a mask label, so a hit is worst-case.

### 2026-09-23 — Overnight go/no-go briefs (no run yet)

- Summary: Meeting 2026-09-24 14:00 needs a decision, not another chance-level S-box run. Briefs are in `docs/2026-09-23/`. The only GPU job predicts `alpha_mask` on extracted v2 (SNR about 2), not `sbox_masked` and not unmasked \(S\). A v1 flatten-head control runs only if that job exits early. Success is worst-case (mask label on) and is not a strict break.
- Files: `docs/2026-09-23/A-llm-go-nogo.md`, `docs/lab-log/meetings/20260924-meeting.md`

### 2026-09-23 — Diagnosis, experiments, and fundamental findings

#### Phase 1: Research and codebase analysis

Launched two research subagents in parallel:

1. **Codebase explorer** — analyzed all model architectures, data loading, baselines, GPU setup
2. **Literature researcher** — surveyed LLM-for-time-series-classification papers

Key literature findings:
- Frozen LLMs fail for classification (need LoRA or LayerNorm tuning at minimum)
- GPT4TS recipe: freeze attention/FFN but train LayerNorm affine params + positional embeddings
- Attention pooling is better than mean pooling for localized signals
- CNN front-end before LLM recommended as matched filter for micro-transients
- Ablation studies show the LLM backbone may contribute nothing — RevIN + patching + linear does most of the work (NeurIPS 2024 critical study)

#### Phase 2: Initial diagnosis of why PatchGPT2SCA fails

Identified three "fatal flaws" in the original approach:

1. **Mean pooling** → `pooled = patch_states.mean(dim=1)` → `Linear(768, 256)` — this is a weighted sum of independent per-patch classifiers (linear in the hidden states)
2. **Frozen causal GPT-2** — unidirectional attention trained on English, cannot learn cross-patch SCA interactions
3. **Reprogramming to English vocab** — maps voltage patches onto word embeddings, destroying amplitude info

#### Phase 3: Experimental runs (3 strategies × ASCADv1)

Implemented `LLM/patch_transformer_sca.py` with three strategies and multiple pooling modes:

| Run | Strategy | Pooling | Params | Result |
|-----|----------|---------|--------|--------|
| 1 | Trainable patch transformer (4L, d=128) | Attention | 0.86M | **FAIL_CHANCE** (CE=5.548, acc=0.004) |
| 2 | Trainable patch transformer (4L, d=128) | Bilinear interaction | 0.89M | **FAIL_CHANCE** (CE=5.546, rank=94) |
| 3 | CNN-Transformer | Bilinear interaction | 0.93M | **FAIL_CHANCE** (CE=5.548, rank=233) |

Also ran sanity checks:
- Simple MLP (6 layers, 200 hidden, per-trace zscore) → FAIL_CHANCE after 30 epochs
- Simple MLP (6 layers, 200 hidden, global norm) → FAIL_CHANCE after 50 epochs

#### Phase 4: Systematic Re-analysis — ASCADv1 vs ASCADv2 & Model Performance Matrix

##### 1. Physical & Cryptographic Differences: ASCADv1 vs ASCADv2

| Feature | ASCADv1 (ATmega8515) | ASCADv2 (STM32F303) |
|---|---|---|
| **Measurement & Length** | EM, 700 samples (cropped to Byte 2) | Power, 15,000 samples (extracted 5k masks + 10k SubBytes) |
| **Masking Scheme** | **1st-Order Boolean**: $C = S \oplus r$ | **Affine**: $C = (\alpha \times S) \oplus \beta$ in $\text{GF}(256)$ |
| **Execution Order** | Fixed time order (no shuffling) | **Random byte shuffling** ($\pi$ permutes 16 S-boxes) |
| **Leakage Strengths** | • $\text{SNR}(S) \approx 0.02$ (no 1st-order unmasked $S$)<br>• **$\text{SNR}(r) \approx 1.32$** (t=156, 181)<br>• **$\text{SNR}(C) \approx 6.38$** (t=517, 547) | • Direct 3-way correlation $\approx 0.0009$<br>• All individual shares are significantly noisier and higher-order |
| **Target Representation** | Single 256-class label: $S = \text{Sbox}(p_2 \oplus k_2)$ | Multi-task: $\alpha (256), \beta (256), \text{sbox\_masked}[16] (256), \pi[16] (16)$ |
| **Core Learning Difficulty** | Recombining two strong, localized 1st-order leaks ($r$ and $C$) via 2nd-order non-linear interaction | Resolving 3-way affine interaction under random temporal permutation without mask supervision |

##### 2. Master Matrix: What Works vs Fails Across Datasets

| Model Architecture | ASCADv1 (`desync0`) | ASCADv1 (`desync50` / `desync100`) | ASCADv2 (`extracted 15k`) | Why / Mechanism |
|---|---|---|---|---|
| **CNNBest** (5 Conv1D + Flatten + FC 4096) | ✅ **Succeeds** (Rank 0 in ~190 traces) | ✅ **Succeeds** (Rank 0 / low rank) | ❌ Fails as single-head | Conv filters capture local shapes; large flatten head connects all positions. Jitter handled by pooling. |
| **TimesNet (Standard)** | ⚠️ Partial (Rank 22–64) | ❌ Fails (Rank 154–203) | ❌ Fails | Standard linear projector cannot learn the cross-frequency non-linear interaction. |
| **TimesNet (ASCAD-Head)** | ✅ **Succeeds** (Rank 0 in 150 traces) | ✅ **Succeeds** (Rank 0–1 in 150 traces) | ✅ **Succeeds (with multi-label)** | Sequence is **flattened** (22,400) into dense head (1024); with multi-label on v2, recovers byte keys. |
| **MultiResNet** (Depth-19 ResNet + Multi-Head) | N/A (designed for v2) | N/A | ✅ **Succeeds** (Recovers masks & S-boxes) | 9 residual stacks + joint supervision of $\alpha, \beta, \text{sbox}, \pi$ provides direct gradient signals. |
| **GNN (GCN, GAT, GIN, SAGE)** | ❌ Fails (Rank 220–240) | ❌ Fails | ❌ Fails | Standard message passing & neighborhood averaging acts as a linear low-pass filter. |
| **GNN (Plan C Segment-SAGE)** | ✅ **Succeeds** (Rank ~5) | ⚠️ Sensitive to jitter | ❌ Untested on v2 | Specifically computes pairwise products $x_i \odot x_j$ to force 2nd-order combination. |
| **GPAM** (Patch Transformer on Raw Traces) | Untested on v1 | Untested on v1 | ✅ **Succeeds** (Rank 0 in ~80 traces) | Uses multi-task mask/intermediate labels on full ~1M raw traces. |
| **PatchGPT2SCA / Time-LLM** | ❌ **Fails at Chance** (Rank ~230) | ❌ **Fails at Chance** | ❌ **Fails at Chance** | Frozen causal GPT-2 + mean pooling + text vocabulary reprogramming completely destroys SCA leakage. |

##### 3. Re-evaluating Previous Hypotheses & Insights

1. **Mean Pooling vs Flatten / Attention Pooling**:
   - **Status: HELD & REINFORCED.**
   - Mean pooling is mathematically equivalent to an additive sum of per-patch linear classifiers. On v1, because neither $r$ nor $C$ alone reveals $S$, an additive sum yields uniform noise. On v2, it blurs 16 shuffled operations together. Flattening or explicit interaction layers are mandatory.
2. **Frozen Causal Backbone**:
   - **Status: HELD & REINFORCED.**
   - Pretrained causal GPT-2 cannot look forward and backward to link distant leakage points, and frozen text weights have zero inductive bias for cryptographic share recombination.
3. **Reprogramming via Word Embeddings**:
   - **Status: HELD & REINFORCED.**
   - Time-LLM's vocabulary reprogramming maps continuous patches to conceptual word meanings ("trend", "season"), filtering out the exact micro-voltage variations required for SCA.
4. **Single-Head vs Multi-Label Target Formulation**:
   - **Status: CRITICAL CORRECTION.**
   - The failure of PatchGPT2 on v1 is purely an **architectural failure** (inability to compute 2nd-order Boolean combinations of $r$ and $C$).
   - The failure on v2 is **both architectural and methodological**: attacking v2 with a single unmasked S-box head without multi-label guidance ($\alpha, \beta$) is fundamentally intractable for standard DL models (even Scoop's claim of doing this was demonstrated by recent literature to be an artifact of evaluation bias).

##### 4. Resolution of the MLP Baseline on ASCADv1 (Why It Appeared to Fail)
- In the original ASCAD paper (*Benadjila et al.*), `MLP_best` (6 dense layers of 200 units, ReLU) **does work on desync0**, reaching rank 0 in ~250–300 attack traces.
- **Why our earlier run saw rank 183:**
  1. **Epoch count**: Benadjila trained `MLP_best` for **200 to 800 epochs** at RMSprop $10^{-5}$ ($B=100$) on raw unnormalized traces. Our benchmark evaluated only 75 epochs (`ge_final = 183`, but `mean_traces_to_rank0 = 431` on the 6% of subsets that converged). When tested with 200 epochs of RMSprop, CE slowly drops from 5.547 to 5.496.
  2. **Brittleness to jitter**: As shown in Benadjila et al. Figure 33, `MLP_best` completely fails on `desync50` and `desync100` (rank stays ~80–120+), whereas `CNN_best` handles jitter effectively due to its convolutional pooling.

##### 5. Analysis: Can We Replace Mean Pooling with Max Pooling?
- **Core question:** If segment amplitude matters and mean pooling dilutes signal, does global Max Pooling solve the problem?
- **Answer:** **No, global Max Pooling is worse than Mean Pooling for SCA.**
  1. **Extreme Value Noise Dominance (SNR << 1):** In ASCAD, signal-to-noise ratio is typically $< 0.1$ (or even $< 0.01$). Electronic switching noise is Gaussian. In a sequence of 87 patches, taking $\max_i h_{i, j}$ samples from the extreme positive tail of the noise distribution (Gumbel/Fisher-Tippett distribution). Outlier noise spikes in non-leaking regions drown out the subtle cryptographic signal.
  2. **Sign / Polarity Blindness (Both Peaks and Troughs Matter):** Hamming weight leakage is bipolar: $HW=0$ produces a negative deflection (trough), while $HW=8$ produces a positive peak. Max pooling only captures positive extremes, discarding low Hamming weight information and inverted polarity channels.
  3. **Destruction of Multi-Point Temporal Recombination:** On ASCADv1, second-order attack requires information from *both* $t \approx 156$ (patch 18) and $t \approx 517$ (patch 63). Global max pooling picks exactly one winning patch per channel, discarding the relative temporal co-occurrence of the two shares.
  4. **Literature Evidence:** Benadjila et al. systematically benchmarked Average Pooling vs. Max Pooling in CNNs for SCA and proved that **Average Pooling significantly outperforms Max Pooling** because local averaging smooths out high-frequency Gaussian noise.
- **What actually works:**
  - **Attention Pooling (Learned Dynamic Weighting):** $\alpha_i = \text{softmax}(w^T \tanh(W_k h_i))$, allowing the model to act as a soft POI selector (weighting patch 18 and patch 63 while suppressing noise patches).
  - **Flattened Bottleneck Head:** Preserves the spatial/temporal position and amplitude of every segment into a dense head (as in CNNBest and TimesNet ASCAD-Head).

## Next actions

| Action | I | U | S | Notes |
|--------|---|---|---|-------|
#### Phase 5: Average Waveform Subtraction Experiments on ASCADv2

Tested subtracting the average waveform $\bar{X}(t) = \frac{1}{N}\sum_i X_i(t)$ from each trace before feeding to models on ASCADv2 (extracted 15k).

##### 1. Leakage Localizer (3-Order Soft-Window on Byte 4)
- **Raw (without sub-avg)**: Val $L_{3O} = -0.1203$, $\text{corr}(m_3) = +0.0117$.
- **With sub-avg**: Val $L_{3O} = -0.1908$, $\text{corr}(m_3) = +0.0202$ (correlation almost doubled, loss improved).

##### 2. Direct S-box TimesNet (`timesnet_ascad_head` predicting $S = \text{Sbox}(p_0 \oplus k_0)$ directly)
Trained on 10,000 profiling traces for 5 epochs (RTX 5090):
- **Raw (without sub-avg)**:
  - Final train loss: 5.5405, val loss: 5.5485.
  - Attack trace key ranks (byte 0): fluctuates between ~30 and ~70 (Rank 18 at 500 traces, Rank 30 at 1500 traces).
- **With sub-avg**:
  - Final train loss: **5.5322**, val loss: 5.5499.
  - Attack trace key ranks (byte 0): drops into **single digits** (**Rank 9** at 1500 traces, **Rank 10–13** across 700–1700 traces).
- **Conclusion**: Subtracting the average waveform removes ~88% of the uninformative static clock waveform energy, allowing the temporal convolutions and dense head to focus on data-dependent variance.

#### Phase 6: Scaled-up Training of Sub-Avg S-box TimesNet (50k Traces)

Following the promising 10k result (Rank 9), scaling up training to test whether the key rank drops to Rank 0:
- Dataset: `STM32_AES_v2/ASCAD_data/ASCAD_databases/ascadv2-extracted.h5`
- Model: `timesnet_ascad_head` with `input_stride=5`, `pool_segments=32`, `d_model=32`, `e_layers=2`
- Profiling traces: 50,000 (5x larger dataset than Phase 5)
- Preprocessing: `sub_avg = True` (subtracting profiling mean trace $\bar{X}(t)$)
- Training schedule: 15 epochs, batch size 64, AdamW/RAdam lr=1e-4 with AMP and grad clipping
- Checkpointing: saving both `best`, `last`, and all per-epoch checkpoints (`_ep{1..15}.pth`)
- Evaluation: multi-attack Guessing Entropy and single-trace accuracies on 5,000 attack traces

##### Results & Findings (50k Sub-Avg TimesNet):
- **Training Progression**:
  - Epoch 1: loss 5.5468, val_loss 5.5462
  - Epoch 4: loss 5.5443, val_loss **5.5455** (best val checkpoint)
  - Epoch 5: loss 5.5442, val_loss 5.5473
  - Epoch 15: loss 5.5333, val_loss 5.5598 (overfitting on profiling traces)
- **Key Rank & Accuracy Across Checkpoints**:
  - `best` (Epoch 4) on 5,000 attack traces: **Min Key Rank 25**, Final Key Rank 129, Top-1 acc 0.48% (chance 0.39%), Top-10 acc 3.96%.
  - `_ep5` on 2,000 attack traces: **Min Key Rank 16**, Final Key Rank 152.
  - Overfitted epochs (8–15): Key rank degraded to Rank 208–215.
- **Physical Analysis**:
  - Sub-avg traces consistently produce real signal (dips to Rank 9–25 vs chance 127.5), confirming that removing 88% static clock variance is essential.
  - However, direct single-task unmasked S-box prediction hits an overfitting ceiling around ~3,500 steps because $\text{SNR}(S) \approx 0.009$ (no 1st-order unmasked S leakage). Without intermediate mask guidance ($\alpha, \beta$), the 4.7M model eventually memorizes profiling noise rather than solving the 3-way affine decomposition under random byte shuffling.
  - In contrast, the mask shares have massive 1st-order leakage ($\text{SNR}(\alpha) \approx 2.0$, $\text{SNR}(\beta) \approx 1.5$). A patch-based or interpretation model must identify these physical share events ("patch meanings") rather than expecting an unguided black-box classifier to break affine masking end-to-end.

#### Phase 7: Option C Multi-Byte Single-Head Sweep (Bytes 0 to 4)

Trained single-head Sub-Avg TimesNet across Bytes 0 to 4 (20k profiling traces, 5 epochs, evaluated on 3,000 attack traces):
- **Byte 0**: **Min Key Rank = 7**, Final Rank = 21, Top-1 = 0.40%, Top-10 = 3.83%
- **Byte 1**: Min Key Rank = 88, Final Rank = 176, Top-1 = 0.43%, Top-10 = 3.63%
- **Byte 2**: **Min Key Rank = 21**, **Final Rank = 27**, Top-1 = 0.20%, Top-10 = 3.87%
- **Byte 3**: **Min Key Rank = 1** (almost Rank 0!), Final Rank = 82, Top-1 = 0.53%, Top-10 = 3.90%
- **Byte 4**: **Min Key Rank = 15**, Final Rank = 58, Top-1 = 0.33%, Top-10 = 3.77%

Key Observations:
1. **Byte 3 came within 1 candidate of full key recovery (Rank 1)**.
2. Leakage strength is non-uniform across the AES state: Bytes 0, 2, 3, 4 all reach low/single digits (Rank 1, 7, 15, 21), while Byte 1 showed the weakest direct leakage (Rank 88).
3. The multi-byte comparison plot is saved to `STM32_AES_v2/ASCAD_data/ASCAD_trained_models/fig_timesnet_sbox_sweep_b0_b4.png` and results JSON to `results_timesnet_sbox_sweep_b0_b4.json`.

| Action | I | U | S | Notes |
|--------|---|---|---|-------|
| Run MultiResNet reproduction | H | H | M | Test if 137M multi-task model with sub_avg breaks v2 |
| Focus on Byte 3 to reach Rank 0 | H | H | S | Fine-tune or extend traces on Byte 3 (already Min Rank 1) |
| Test Sub-Avg on Mask Heads ($\alpha, \beta$) | H | M | S | Probe whether sub-avg boosts $\alpha$ beyond 12x chance |

### Priority snapshot

- **Do now**: Probe whether Sub-Avg traces allow patch transformer / TimesNet to learn $\alpha$ and $\beta$ with high precision.
- **Next**: Connect learned share representations via algebraic affine layer to achieve Rank 0.
- **Deep work (park)**: whether a frozen LLM adds anything over a scratch transformer.

## Related

- Task ids:
  - T-016 (original Time-LLM smoke runs)
  - T-017 (LLM code organization)
  - T-014 (ASCADv1 leakage localizer — 2O windows)
  - T-006 (CNN/MLP baselines)
- `docs/lab-log/todo.md` items: TBD
- External links:
  - Time-LLM: https://github.com/KimMeen/Time-LLM
  - GPT4TS / One Fits All: https://arxiv.org/abs/2302.11939
  - "Are LLMs Actually Useful for TS Forecasting?": NeurIPS 2024
