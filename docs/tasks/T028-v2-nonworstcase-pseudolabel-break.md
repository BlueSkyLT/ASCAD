# T-028: Non-Worst-Case ASCADv2 Break via Pseudo-Label Bootstrap

## Status

`active` — **current focus (set 2026-10-04; T-029 is the side task).** Phase 0 done (baseline
verified, rank-tie hazard found). Phase 1 (variance-ranked selectors + k-means) closed NO-GO with
a measured cause. **2026-10-04 evidence review reopened the question as Phase 1R**: Phase 1 never
separated *finding α's bands* from *clustering them*, and never clustered the oracle bands, so
"α is not recoverable label-free" is not yet established. Phase 1R is a CPU-only Tier-0 battery
(R1–R7, no GPU) with a hard stop gate at R2. Phase 2 stays blocked until R2 passes; the Step 3.1
fallback is still not executable as written. **Phase 1R approved by the user 2026-10-04 (thresholds frozen); R1 → R2 in progress** — see the 2026-10-04 log entry.

## Goal

Achieve **key rank 0 on ASCADv2** in a non-worst-case setting: trace in → S-box value out, using only `(trace, plaintext, key)` during profiling — **no mask labels** ($\alpha$, $\beta$, $C$, $\pi$) in training. Final metric: all 16 key bytes at rank 0 with z-score separation ≥ +3σ on the shuffled-trace null, validated by AOD-GE.

## Context

- Related code / data paths:
  - `ASCAD_test_models.py:687-743` — `proba_dissect_beta`, `proba_dissect_alpha`, `proba_dissect_permind`, `multilabel_predict` (algebraic dissection pipeline)
  - `ASCAD_test_models.py:423-510` — `log_key_scores_from_predictions`, `ranks_from_log_scores` (key ranking)
  - `ASCAD_train_models.py` — MultiResNet 34-head architecture, `sub_avg` preprocessing
  - `leakage_localizer/v2_variance_localizer.py` — unsupervised active-region detector (finds α at sample 867, β at 341)
  - `leakage_localizer/v2_autoencoder_localizer.py` — unsupervised anomaly localizer (126 active patches, 10× more selective)
  - `LLM/band_scan.py` — band selection infrastructure with `--select {snr,low_snr,contiguous,spread,random}`
  - `GPAM/artifacts/C-snr-profile.py` — SNR measurement script (validates against documented peaks)
  - `GPAM/artifacts/C-snr-v1.py` — v1 SNR measurement
  - Data: `ascadv2-extracted.h5` (15k crop, 500k traces), raw shards (8 × ~12 GB, 1M samples/trace)
- Related docs:
  - `docs/snr_investigation.md` — **CRITICAL**: confirms SNR(C) = 0.072 on both crop and raw (C is invisible everywhere)
  - `docs/notes/ascadv2-label-frequency.md` — label frequency prior artifacts (AOD-GE validation required)
  - `docs/notes/ascadv2-direct-evidence-explained.md` — 3O product failure (corr = 0.0009)
  - `docs/survey-ascadv2-dl-2022-2026.md` — literature: no non-worst-case break exists with shuffle ON
  - `GPAM/artifacts/C-talking-points.md` — campaign results and null controls
- Related tasks: T-019 (GPAM reproduction), T-020 (LLM/Transformer pipeline), T-024 (TimesNet multi-task), T-025 (variance localizer), T-026 (autoencoder localizer)
- Constraints:
  - **C is physically invisible** (SNR 0.072, only 4.6× noise floor, 0 samples > 0.1 in both crop and raw). No approach may assume C is detectable.
  - Non-worst-case: training must NOT use `alpha_mask`, `beta_mask`, `masks`, `desync`, `permind`, or `sbox_masked` fields from HDF5 metadata.
  - MAY use: `traces`, `plaintext`, `key`, `ciphertext` fields (standard profiled attack).
  - GPU budget: ≤ 2 hours total across all phases.
  - Must run AOD-GE and shuffled-trace null controls before claiming any rank 0.

## Established Facts (Do Not Re-Derive)

These are confirmed results. An executing agent should use them as ground truth, not re-investigate:

| Fact | Source | Number |
|------|--------|--------|
| SNR(α) = 2.0 at sample 867 (crop), 205867 (raw) | `docs/snr_investigation.md` | 871 samples > 0.1 in crop, 14,576 in raw |
| SNR(β) = 1.4 at sample 338 (crop), 205338 (raw) | `docs/snr_investigation.md` | 55 samples > 0.1 in crop, 267 in raw |
| SNR(π) = 0.13 at sample 5552 (crop), 0.21 at 282070 (raw) | `docs/snr_investigation.md` | 1 sample > 0.1 in crop, 27 in raw |
| SNR(C) = 0.072 everywhere, 0 samples > 0.1 | `docs/snr_investigation.md` | Identical in crop and raw |
| MultiResNet 34h + sub_avg: all 16 bytes rank 0 at 5k traces | T-024 | z-scores +3.89σ to +6.86σ |
| MultiResNet head accs, 5k attack traces, **recomputed 2026-10-04** from `experiments/out/head_preds_5k.npz` vs `Attack_traces/labels`: **α 100.00%**, β 5.84%, C 0.69%, π 67.03%. Log-lik gain over uniform: α +5.48 nats (max 5.545), β +1.51, π +1.80 (max 2.77), **C −0.09** | this session | Corrects T-024's "α 0.49%, β 0.41%" (wrong label column, source not located). Same error appears in `T024:108-111`, `docs/notes/patch-pipeline-survey.md:196`, `docs/roadmap-break-ascad.md:211` and **has been corrected in all three locations** (2026-10-04). α is `masks[:,18]`, β is `masks[:,17]` |
| 3O centered product: corr = 0.0009 (dead) | `docs/notes/ascadv2-direct-evidence-explained.md` | All 16 bytes within noise |
| Averaging 2 adjacent samples: 86% → 0.83% accuracy | T-020 / `patch-pipeline-survey.md` | Resolution is binding constraint |
| Variance localizer flags α's patch 108 and β's patch 42 — but it also flags **90% of all patches** active (1686/1874) | T-025 | Without labels, but not a band selector: Phase 1 measured SNR-band recall 0.000 for label-free variance selectors. The old wording "detects α" overstated it |
| `desync` metadata is exactly 0 on all 20k profiling and 10k attack traces checked | this session | No artificial jitter was applied. Says nothing about natural clock/sub-sample jitter (untested — see D5) |
| `sub_avg` = subtract the **global profiling mean trace** (one value per sample position), no labels, nothing per-trace | `ASCAD_train_models.py:1157-1160`, `ASCAD_test_models.py:948-953` | Provable no-op for the band probe (SNR-16 0.9395 → 0.9395). **No 34-head MultiResNet run without it exists**, so T-024's 16/16 cannot be credited to `sub_avg` |
| Mean-of-2 / 4 / 8 adjacent samples: α top-1 0.8603 → 0.0083 / 0.0065 / 0.0090 (same 1,600 samples, 2-layer MLP) | `LLM/band_scan.py`, `RESOL-SUBAVG.log` | Sum of adjacent samples destroys α; the *difference* has never been tested (D3) |
| SNR(π) was computed on **`perm_index[:,0]` only**, plain between/within variance, K=16. Peak 0.2085 = 227× its null floor (0.0009); β 1.42 = 90× its floor (0.0158) | `C-snr-profile.py:34-41,64-79`, fork recomputation | Across all 16 π bytes, Spearman(accuracy, peak SNR) = 0.10, so first-order peak SNR does not predict learnability of π |
| AE localizer: 126/1874 patches flagged as active (6.7%) | T-026 | 10× more selective than variance |
| WPP23 collision: ~70k traces shuffle OFF, ~8.4M shuffle ON | `docs/survey-ascadv2-dl-2022-2026.md` | 120× blowup from shuffle |

## Plan

### Phase 0: Diagnostic — Which Heads Are Essential? (I:H U:H S:H)

**Purpose:** Before building the unsupervised pipeline, determine which mask heads are essential for key recovery by ablating the working MultiResNet checkpoint.

- [x] (I:H U:H S:H) **Step 0.0:** Locate the trained MultiResNet 34-head checkpoint.
  - Expected path: `multiresnet_34h_50k_subavg.pth` or similar in `out/` or `GPAM/artifacts/`.
  - If not found, retrain: `sub_avg=True`, 50k traces, 15 epochs, batch 64, RAdam + AMP. Takes ~8.6 min.

- [x] (I:H U:H S:H) **Step 0.1:** Create `experiments/ablation_heads.py` — DONE 2026-10-03,
  see Phase 0 results in the session log. Required adding `experiments/rank_degeneracy.py`
  because uniform-head ablation reports rank 0 on ties; read the "genuine rank-0" column, not
  the reported one.
  - Load the trained checkpoint and run inference on 5k attack traces to get all 34 head predictions.
  - For each ablation run, replace specified heads' softmax outputs with uniform $1/K$ distributions.
  - Run `multilabel_predict` (or `multilabel_without_permind_predict` when π is ablated) on the modified predictions.
  - Compute 16-byte key ranks via `log_key_scores_from_predictions` → `ranks_from_log_scores`.

  **Ablation matrix:**

  | Run | α | β | C (16 heads) | π (16 heads) | What it tests |
  |-----|---|---|-------------|-------------|---------------|
  | A0 | ✅ | ✅ | ✅ | ✅ | Reproduces T-024 baseline (must be all-16 rank 0) |
  | A1 | uniform | uniform | ✅ | ✅ | Do we need α/β discovery at all? |
  | A2 | ✅ | ✅ | uniform | ✅ | Is the C-head decorative? **KEY QUESTION** |
  | A3 | uniform | uniform | uniform | ✅ | Does π alone carry any signal? |
  | A4 | ✅ | ✅ | ✅ | uniform | Is π routing essential? |
  | A5 | ✅ | uniform | ✅ | ✅ | Is β essential? |

  ```python
  # Core logic
  def ablate(predictions, head_indices_to_uniform):
      """Replace specified head outputs with uniform 1/K."""
      preds = [p.copy() for p in predictions]
      for idx in head_indices_to_uniform:
          K = preds[idx].shape[1]
          preds[idx] = np.full_like(preds[idx], 1.0 / K)
      return preds

  # Head index mapping:
  # 0 = alpha (256-way), 1 = beta (256-way)
  # 2..17 = C_0..C_15 (256-way each)
  # 18..33 = pi_0..pi_15 (16-way each)
  ABLATIONS = {
      "A0_baseline": [],
      "A1_no_alpha_beta": [0, 1],
      "A2_no_C": list(range(2, 18)),
      "A3_pi_only": [0, 1] + list(range(2, 18)),
      "A4_no_pi": list(range(18, 34)),
      "A5_no_beta": [1],
  }
  ```

  **Output:** JSON file `experiments/out/ablation_results.json` with:
  ```json
  {
    "A0_baseline": {"rank_final": [0,0,...], "traces_to_rank0": [...]},
    "A1_no_alpha_beta": {"rank_final": [...]},
    ...
  }
  ```

  **Go/no-go decisions based on results:**
  - If A2 (no C) still rank 0 → **C is decorative** → skip C pseudo-labels entirely (HUGE simplification)
  - If A4 (no π) fails → **π is essential** → must solve unsupervised π discovery
  - If A1 (no α/β) still works → α/β discovery unnecessary
  - If A3 (π only) still works → only need π, simplest possible path

  **Cost:** ~15 min (inference + dissection only, no training).

---

### Phase 1: Unsupervised Mask & Permutation Discovery (I:H U:M S:M)

**Purpose:** Discover pseudo-labels for α, β, and π from traces alone, without any HDF5 mask metadata.

- [ ] (I:H U:M S:H) **Step 1.1:** Create `experiments/unsup_alpha_discovery.py`

  **Inputs:**
  - Extracted traces from `ascadv2-extracted.h5`, field `Profiling_traces/traces` (shape: `(N, 15000)`)
  - SNR curve: `GPAM/artifacts/snr_crop_rm.npy` (or recompute via `C-snr-profile.py`)
  - True α labels (for validation ONLY, not used in discovery): `Profiling_traces/metadata` field `masks[:,18]`

  **Method 1: SNR-guided feature extraction + k-Means**
  ```python
  import numpy as np
  from sklearn.cluster import MiniBatchKMeans
  from scipy.optimize import linear_sum_assignment

  # 1. Load SNR curve and select high-SNR sample indices
  snr = np.load("GPAM/artifacts/snr_crop_rm.npy")
  high_snr_idx = np.where(snr > 0.1)[0]  # ~871 indices

  # 2. Extract features from those samples
  traces = h5f["Profiling_traces/traces"][:50000, :]
  X = traces[:, high_snr_idx].astype(np.float32)  # (50000, 871)

  # 3. Optional: z-score normalize each sample position
  X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

  # 4. k-Means with 255 clusters (α ∈ {1..255}, class 0 never occurs)
  km = MiniBatchKMeans(n_clusters=255, batch_size=4096, n_init=5, random_state=42)
  pseudo_alpha = km.fit_predict(X)

  # 5. Evaluate via Hungarian matching against true labels
  true_alpha = metadata[:50000, 18]  # masks[:,18] = alpha_mask
  confusion = np.zeros((255, 255), dtype=int)
  for p, t in zip(pseudo_alpha, true_alpha - 1):  # shift α from {1..255} to {0..254}
      confusion[p, t] += 1
  row_ind, col_ind = linear_sum_assignment(-confusion)
  matched_acc = confusion[row_ind, col_ind].sum() / len(pseudo_alpha)
  print(f"k-Means α pseudo-label accuracy: {matched_acc:.4f} (chance: {1/255:.4f})")
  ```

  **Method 2: VQ-VAE on high-SNR samples**
  ```python
  # Encoder: Linear(871 → 256) → GELU → Linear(256 → 64)
  # Codebook: nn.Embedding(255, 64), straight-through estimator
  # Decoder: Linear(64 → 256) → GELU → Linear(256 → 871)
  # Loss: MSE(X, X_hat) + 0.25 * VQ_commitment
  # Train: 50k traces, AdamW lr=1e-3, 30 epochs, batch 512
  # Pseudo-label: nearest codebook index per trace
  ```

  **Method 3: Supervised oracle (validation ceiling)**
  ```python
  from sklearn.neural_network import MLPClassifier
  clf = MLPClassifier(hidden_layer_sizes=(512, 256), max_iter=100, random_state=42)
  clf.fit(X[:40000], true_alpha[:40000])
  oracle_acc = clf.score(X[40000:50000], true_alpha[40000:50000])
  print(f"Oracle α accuracy ceiling: {oracle_acc:.4f}")
  ```

  **BUT WAIT: SNR-guided feature selection uses SNR which requires labels.**
  
  **Non-oracle variant:** Replace SNR-guided selection with the **variance localizer** output (T-025), which identifies high-variance patches without any labels:
  ```python
  # Use variance localizer to find α-informative samples
  residual = traces - traces.mean(axis=0)  # mean subtraction (no labels)
  sample_var = residual.var(axis=0)
  # Threshold: median + 3*IQR (same as T-025)
  q1, q3 = np.percentile(sample_var, [25, 75])
  threshold = np.median(sample_var) + 3 * (q3 - q1)
  high_var_idx = np.where(sample_var > threshold)[0]
  # These should overlap heavily with the SNR>0.1 indices
  X_unsup = traces[:, high_var_idx].astype(np.float32)
  ```

  **Output:** `experiments/out/pseudo_alpha.npy` (shape: `(N,)`, values 0–254), plus accuracy report.

  **Expected outcome:** k-Means: 20–60%; VQ-VAE: 40–70%; Oracle: 80–95%.
  **Go/no-go:** If best method < 10% accuracy, α discovery is infeasible → skip to collision fallback.

- [ ] (I:H U:M S:H) **Step 1.2:** Create `experiments/unsup_beta_discovery.py`
  - Same structure as Step 1.1 but for β (55 samples with SNR > 0.1 in crop).
  - Use variance localizer to find β-informative samples (patch 42, near sample 341).
  - Expect lower accuracy due to fewer informative features and lower SNR (1.4 vs 2.0).
  - **Output:** `experiments/out/pseudo_beta.npy`

- [ ] (I:H U:M S:M) **Step 1.3:** Create `experiments/unsup_pi_discovery.py`
  - π leaks through the **loop counter** (control flow), not through data values. SNR 0.13–0.21.
  - **Method:** The 16 S-box slots in window W2 (samples 5000–15000 of extracted trace) execute in shuffled order. Each slot's loop iteration counter value (0–15) leaks differently.
  - Extract features from the π-informative region (around sample 5552 in crop).
  - k-Means with k=16 on extracted features per execution slot.
  - 16-way classification is much easier than 255-way.
  - **Output:** `experiments/out/pseudo_pi.npy` (shape: `(N, 16)`, per-trace permutation)
  - **Expected:** 20–50% per-position accuracy (vs 6.25% chance).

---

### Phase 1R: Phase 1 re-opened — evidence review + CPU-only Tier-0 battery (I:H U:H S:M)

Added 2026-10-04. Steps 1.1–1.3 above stay as the record of what was tried. Run everything in
`py10`. Tier-0 uses **no GPU**. Labels appear only as an *oracle for scoring or understanding*,
and every such number is tagged ORACLE — none of it is a non-worst-case step.

Evidence tags: **M** = measured this session (re-verified by me), **F** = read or computed by a
research fork from code/logs and not independently re-verified, **I** = inferred, untested.

#### A. What the worst-case classifier actually does (what any label-free route must reproduce)

| # | Component | Evidence | Tag |
|---|-----------|----------|-----|
| 1 | α is read **at full per-sample resolution** from a window of ~871 crop samples (14,576 raw). Supervised MLP on 16 SNR bands: 93.95% | `LLM/band_scan.py`, T-020 | F |
| 2 | **Averaging** adjacent samples kills α (0.8603 → 0.0083 / 0.0065 / 0.0090 for mean-of-2/4/8); **patching does not**: GPAM uses 150-sample (crop) and 400-sample (raw) tokens and gets 92.95% / 99.996%, because a learned `Dense(192)` per patch keeps a separate weight per sample | `RESOL-SUBAVG.log`, `scaaml/models/gpam.py` | F |
| 3 | Every token attends to every token (3 gated-attention blocks, learned Toeplitz/RoPE bias). α, β, π sit in different patches (~513, ~513, ~705 at patch 400), so the model must mix across them. Heads are 34 independent dense stacks on one shared pooled vector; `output_relations=[]`, so **the model is never told** C = α⊗S⊕β | `train_gpam_raw_full.py`, `gpam.py` | F |
| 4 | **Phase transition, not a slope**: epochs 0–11 at chance; α jumps 18.6% → 91% at epochs 12→13; every π head leaves chance in epochs 12–16; β stays <5% until epoch 38 and passes 10% only at epoch 61; C never leaves chance (~0.6%) | `…212317_metrics.csv` | F |
| 5 | The α head is **essentially solved** (100.00% on the 5k attack traces); β is weak (5.84%, +1.51 nats); π is moderate (67.0%, +1.80 nats of 2.77); the 16 C heads score **below uniform** on their own labels (−0.09 nats) | recomputed from `head_preds_5k.npz` | **M** |
| 6 | α and β are **each near-essential** for the key: zeroing α alone leaves 2/16 bytes, zeroing β alone 1/16 (Phase 0). π is not (12/16 without it) | Phase 0 table | **M** |
| 7 | The algebra (C = α⊗S⊕β, α and β shared by all 16 slots) is used **only as a decoder** after the heads are trained. Joint (α,β) decoding across slots is not implemented; the code marginalises per slot | `ASCAD_test_models.py:745-773` | F |
| 8 | No exploitable prior: α, β and each `perm_index` column are uniform (χ² below critical), MI(α,β) = 0.0661 vs 0.0671 shuffled null | 500k profiling traces | F |

**Why π is learned better than β although SNR(π) = 0.13 < SNR(β) = 1.4** — three measured facts
dissolve most of the paradox:

1. *The SNR yardstick is not comparable.* SNR(π) was computed on `perm_index[:,0]` only, K=16;
   its raw peak 0.2085 is **227× its own null floor** (0.0009), while β's 1.42 is 90× its floor
   (0.0158). Relative to chance, π's peak is the stronger first-order signal. (F)
2. *Chance level.* β: 8 bits, π: 4 bits. By information the two heads are close: MultiResNet β
   +1.51 nats vs π +1.80 nats. π's 67% looks large only because 1/16 is a high chance level. (M)
3. *Peak SNR does not predict learnability of π.* Across the 16 π heads, Spearman(accuracy,
   peak SNR) = 0.10 and (accuracy, summed SNR) = 0.09; byte 15 has the most spread-out SNR and the
   best accuracy (99.7%). π is a **control-flow / which-slot-when** label, so it is probably
   carried by where a pattern occurs and by many weak samples, which a single-sample amplitude
   SNR cannot see. (F for the correlations, I for the mechanism)

What this tells us: **do not use first-order single-sample SNR as the gate for a label-free
method**, and do not read "SNR(π) tiny" as "π hidden". It does *not* make π the way in: π is not
essential for the key (Phase 0), and α and β are.

**"Can we exploit the α/β/C/π relationship or the shared 256-symbol space?"** — assessed, and
the answer is *not for producing labels*:

- α and β each **leak directly** (SNR 2.0 and 1.4) and are statistically independent of one
  another and of (p,k). So each can in principle be recovered on its own band; the C relation is
  not needed to create pseudo-labels. (M for independence, I for the conclusion)
- The only relation that would help label-free is the pairwise one,
  C_a ⊕ C_b = α⊗(S_a ⊕ S_b) (β cancels). It requires **C to be readable**, and it is not:
  SNR(C) = 0.072 (0 samples > 0.1) and the C heads score below uniform (M). Tied-weight / shared
  256-codebook ideas give no extra signal, since a uniform output satisfies every consistency
  constraint.
- EM over the joint latent (α,β) is the Phase 1 floor again (65,280 components, ~1.3× variance
  excess). (I)
- So the relations are **decoders and validators** (joint (α,β) marginalisation, Hungarian over
  the 16×16 π matrix), not label sources. They raise the worst-case baseline (R6); they do not
  make a non-worst-case run.

**How to reduce the "background waveform", and is resolution a must?**

- `sub_avg` = subtract the *global* mean trace (`ASCAD_train_models.py:1157-1160`). It removes
  only the static, fixed-position, data-independent template. It is a **provable no-op** for the
  band probe (0.9395 → 0.9395) and there is **no 34-head run without it**, so its role in T-024
  is unmeasured. Its reputation ("removes 88% of variance") has no source artifact. (F)
- The 3-tier anatomy puts the opcode wave at ~94.5% of *amplitude* (not variance) and data
  leakage at ~5.5%. The remaining ~98% of variance is not decomposed anywhere (shuffle-induced
  data variance, per-trace π structure, noise). `desync` metadata is 0 everywhere (M), which
  rules out *injected* jitter only. (F/M)
- TimesNet's FFT-period front end *folds* clock periodicity rather than suppressing it, and the
  v2 runs used `input_stride=5` plus a 32-segment pool — subsampling, which the resolution
  result predicts is fatal for α. The TimesNet v2 nulls therefore do not test these hypotheses,
  and **TimesNet should not be revived for this task**. (F/I)
- **Resolution verdict:** keep per-sample information *inside the first learned layer* (patchify,
  then learn a linear map over raw samples, as GPAM does). Coarse tokens are fine; averaging or
  strided subsampling is not. Nothing varies patch size inside a working model (gap, G2).

#### B. Why Phase 1 is not yet a verdict (three gaps)

1. **Selection and clustering were never separated.** Phase 1 used variance, normalised variance
   and random selectors only; SNR was a post-hoc overlap diagnostic. No `unsup_alpha*.py` run
   clustered the SNR-selected bands. If that clustering also sits at the null, the problem is
   clustering. If it works, the bottleneck is band *finding*. We currently cannot say which.
2. **Variance is the wrong band statistic.** The variance scanner flags 90% of patches, and
   SNR-band recall was 0.000. Higher-order statistics (excess kurtosis, bimodality, per-sample
   mixture BIC) have never been tried as label-free band finders.
3. **The target may be mis-specified.** A 255-way Hungarian match has a 2.0% inflation floor and
   treats α as an unordered label. If α leaks bit-wise or by Hamming weight, the right latent is
   8 binary variables, and a per-bit agreement metric is orders of magnitude more sensitive.

#### C. Tier-0 battery (CPU only, ~hours). Thresholds below are **proposed**; approved and frozen 2026-10-04 by the user; they cannot be moved after any R-number is seen.

| ID | Test | Purpose / gate |
|----|------|----------------|
| R1 | **Leakage model of α and β** (ORACLE, understanding only): on the SNR>0.1 band, between-class R² of the samples against (a) the 8 label bits, (b) Hamming weight, (c) 255-way one-hot. Also β | Decides whether the label-free target is 255 classes or 8 binary latents, which fixes R2's design. No gate |
| R2 | **Oracle-band clustering gate (decisive).** Cluster the SNR>0.1 band of α (871 samples, 50k traces): (a) 8-bit factored latent (sign of leading PCs / ICA / binary mixture), (b) 255-way GMM/k-means. Score by **per-bit agreement** (chance 0.5, shuffled-label null) and NMI vs null, never Hungarian-255 alone | **PASS** = pseudo-α exact-match ≥ 10% (≈ 0.75 per-bit if the 8 bit errors are independent, since 0.75⁸ ≈ 0.10; 10% is ~25× chance and about GPAM's holdout β head, 10.48%, the weakest mask head the worst-case pipeline still uses — a judgement, not a derived minimum). **STOP** if per-bit < 0.55. 0.55–0.75 is a grey zone: continue only if R3 shows a route |
| R3 | **Label-free band finders beyond variance**: per-sample excess kurtosis, bimodality / mixture-BIC, pairwise-sample dependence. Score by enrichment of SNR>0.1 in the top-871 (random base rate 871/15000 = 5.8%) | Run only if R2 passes. **PASS** = precision ≥ 29% (≥ 5× random) **and** R2's clustering on the *selected* bands recovers ≥ half of R2's oracle-band per-bit agreement |
| R4 | **Background/resolution ablation inside the supervised probe** (same 2-layer MLP, same 1,600 samples, same α label): raw / global-mean subtraction (reproduce the no-op) / per-trace mean + scale / top-k PCA removal / **adjacent difference** x[t]−x[t−1] / adjacent mean (reproduce the 0.008 kill) | Tests "α lives in fine structure": the difference should survive while the sum dies. Picks the label-free front end that is also resolution-preserving. Falsifier: if the difference also collapses, α needs absolute level |
| R5 | **Variance budget by source** (ORACLE, understanding only): per-sample fraction explained by α, β, π, the 16 known S, top-k label-free PCs, residual. Plus a clock-phase lag test (cross-correlation of each trace to the mean in a quiet region) to check for *natural* jitter | Decomposes the ~98% nobody has measured; says whether re-alignment or PCA removal is worth building |
| R6 | **Decoder upside on cached heads** (zero GPU, `head_preds_5k.npz`): Hungarian assignment over the 16×16 π matrix vs the 67.0% argmax; joint (α,β) marginalisation over the 16 slots vs per-slot dissection on ~200 traces | Raises the **worst-case** baseline bar. Not a non-worst-case step; say so in any report |
| R7 | **Re-verify two fork-reported anomalies**: (a) substituting the *true* β for the β head lowers key z (−0.02 to −0.11); (b) C heads are below uniform (−0.09, M) yet are the structural carrier (Phase 0) | If (a) holds, Phase 0's "C is the carrier" is incomplete and Phase 2's pseudo-C training target needs rethinking before any GPU is spent |

#### D. Decision tree after Tier-0

- **R2 fails** → close the pseudo-label bootstrap. State it as: *α is not recoverable
  label-free by clustering even from its oracle band.* The remaining routes are the Tier-1
  runs below, with an explicit expectation of null.
- **R2 passes, R3 fails** → the task becomes label-free **band finding**. Candidates, each
  needing an R3-style enrichment gate: higher-order statistics, SimMIM-embedding saliency
  (T-017 never made it a key signal, so lowest priority), bootstrapping bands from the
  attributions of a weakly trained network.
- **R2 and R3 pass** → Phase 2 unblocked for α. **β gets its own R2/R3 pair** (55-sample band,
  SNR 1.4, and supervised β is only 5.8–14%), so expect it to be the harder gate.

#### E. Tier-1 (GPU; ask first; total budget ≤ 2 h; only after the Tier-0 gate)

- **G1 label-efficiency curve:** α-head accuracy vs number of mask-labelled traces (1k / 5k /
  20k / full). Quantifies how close a "few labels" regime is. Not itself non-worst-case.
- **G2 patch-size sweep at fixed label inside a working model** (α-only, patch 25/50/100/200/400):
  the missing experiment behind "do we need token-level resolution".
- **G3 S-only end-to-end** (no mask labels) with a GPAM-style backbone, hard epoch budget, and
  the pre-registered chance-line stop rule from T-029. Expect null: with shuffle on and no
  first-order foothold, GPAM itself sat at chance for 12 epochs *with* α supervision.

**Do not:** cite the 93.95% band result as a non-worst-case number (it uses α labels to pick
bands); report any rank without the tie/flatness check; revive TimesNet for v2.

---

### Phase 2: Pseudo-Label Multi-Task Training & Key Recovery (I:H U:M S:M)

**Purpose:** Train the same MultiResNet 34-head model using pseudo-labels from Phase 1, then attempt key recovery.

- [ ] (I:H U:M S:M) **Step 2.1:** Create `experiments/pseudo_multitask_train.py`

  **Critical insight: C labels are COMPUTABLE from pseudo-masks + known (plaintext, key):**
  ```python
  from ASCAD_test_models import multGF256
  # During profiling, we know plaintext and key
  for b in range(16):
      S_b = AES_SBOX[plaintext[:, b] ^ key[:, b]]
      # C = α × S ⊕ β (all in GF(256))
      C_b = np.array([multGF256(a, s) ^ be
                       for a, s, be in zip(pseudo_alpha, S_b, pseudo_beta)])
      pseudo_c_labels[:, b] = C_b
  ```

  **Training configuration:**
  - Architecture: MultiResNet with 34 heads, identical to T-024
  - Preprocessing: `sub_avg=True` (mean trace subtraction)
  - Labels: pseudo-α, pseudo-β, computed pseudo-C, pseudo-π (all from Phase 1, NO HDF5 mask metadata)
  - Data: 50,000 profiling traces, batch 64, 15 epochs, RAdam + native AMP
  - Save best checkpoint by validation loss

  **Noise-aware training (optional, if pseudo-label quality is low):**
  - Add label smoothing proportional to estimated pseudo-label noise
  - Or: use mixup / soft targets from the clustering posterior
  - Or: bootstrap — train model, use model predictions as refined pseudo-labels, retrain

- [ ] (I:H U:H S:H) **Step 2.2:** Evaluate key recovery

  **On 5,000 attack traces:**
  1. Run model inference → 34 head softmax outputs
  2. Run `multilabel_predict` → 16 per-byte posterior distributions
  3. Run `log_key_scores_from_predictions` → `ranks_from_log_scores`
  4. Report: per-byte rank_final, traces_to_rank0

  **Mandatory null controls:**
  1. **Shuffled-trace control:** Permute trace-to-metadata assignments → must return rank ~128
  2. **AOD-GE control:** Divide posterior by training label frequency prior → rank must still be 0
     (See `docs/notes/ascadv2-label-frequency.md` — byte 4 reaches rank 3 from frequency alone!)
  3. **Constant-trace control:** Replace all traces with the mean trace → must return rank ~128

  **Output:** `experiments/out/pseudo_multitask_results.json` with per-byte ranks, z-scores, null control ranks.

  **Success criterion:** All 16 bytes rank 0 with z-scores ≥ +3σ on shuffled null, AND AOD-GE confirmed.

---

### Fallback: WPP23 Collision with π Assistance (I:M U:L S:L)

**Purpose:** If Phase 2 fails, implement a collision-based attack as an alternative non-profiled path.

- [ ] (I:M U:L S:L) **Step 3.1:** Implement WPP23 collision on extracted traces
  - Segment the W2 window (samples 5000–15000) into 16 execution slots
  - For each pair of traces, compare corresponding slots
  - Accumulate collision statistics for key-delta recovery
  - First validate with shuffle OFF (expect ~70k traces for all key-deltas)
  
- [ ] (I:M U:L S:L) **Step 3.2:** Add π-assisted slot routing
  - Use predicted π (from Phase 1 or from the trained model) to align slots between traces
  - Measure trace count reduction vs the 120× blowup from shuffle

## Decisions

- 2026-10-04: Phase 1 re-opened as **Phase 1R**. The NO-GO stands for variance-ranked selection + k-means only. Order of work: R1 → R2 (hard stop gate) → R3 → R4–R7. Thresholds in the plan were approved and frozen by the user on 2026-10-04. The task goal is unchanged (no mask labels in training); every ORACLE number is for understanding only.
- 2026-10-04: The α/β/C/π algebra is treated as a **decoder and validator**, not a label source: α and β leak directly and independently, and the pairwise C relation needs C readable (SNR 0.072). Do not spend effort on EM over the joint (α,β) latent or on tied/shared-codebook heads.
- 2026-10-03: Created as distinct from T-020 (which was the LLM/Transformer pipeline) and T-024 (which was worst-case multi-task). This task is specifically about the non-worst-case / pseudo-label bootstrap approach.
- 2026-10-03: Grounded in `docs/snr_investigation.md` — C is invisible everywhere (SNR 0.072, 4.6× noise floor). No approach in this task may assume C is detectable from the trace.
- 2026-10-03: C pseudo-labels are COMPUTED from pseudo-α + pseudo-β + known (plaintext, key), not observed from traces. This is the central technical insight.
- 2026-10-03: Phase 0 ablation runs BEFORE any unsupervised work, to determine which heads are essential and which can be skipped.

## Working notes / session log

### 2026-10-04 — Phase 1R: evidence review of what makes ASCADv2 classification work

Three read-only forks (GPAM mechanism, background/resolution, α/β/C/π algebra), then I re-verified
the numbers that change the plan. Nothing was trained; no GPU; no files edited outside docs.

- **A wrong number in the Established Facts table was corrected.** `T024:108-111` and
  `patch-pipeline-survey.md:196` said α 0.49%, β 0.41%. Recomputed
  from `head_preds_5k.npz` against `Attack_traces/labels`: **α 100.00%, β 5.84%, C 0.69%, π 67.03%**.
  The 0.49% / 0.41% figures were never reproduced; a wrong label column is the likely cause (not
  located). `MultiResNet/evaluate.py` states in its docstring that it reports head accuracies but
  contains no such code. **Both files were corrected the same day** on the user's instruction;
  `roadmap-break-ascad.md:211` never had the α/β figures.
- **Consequence of the correction:** "rank 0 while α is at chance" was never a puzzle, because α is
  solved and β is only weakly read. The real puzzle moved to C: its heads are *below* uniform
  (−0.09 nats) while remaining the structural carrier of the key signal (R7).
- **Phase 1's NO-GO is narrower than it was written.** It excludes variance-ranked selection plus
  clustering; it does not test clustering the oracle bands, nor higher-order band finders, nor a
  bit-wise target. Phase 1R is the cheapest way to settle that, and R2 is a hard stop gate.
- **Not verified by me (fork-reported):** the GPAM learning-order numbers, the SNR(π) provenance and
  floors, the 94.5% amplitude split, the true-β substitution result, the TimesNet front-end reading.
  They are tagged F in the plan.
- **GPU:** none used or requested. Nothing was run on the R-battery yet; it needs a go-ahead.

### 2026-10-03 — `rank()` tie convention fixed repo-wide: ties now count AGAINST us

Changed in `ASCAD_test_models.py`, because Phase 0 showed uniform-replacing all 34 heads
reported **rank 0 on 16/16 bytes** — total ignorance scored as a perfect break.

- `rank()` now takes `tie_break` and **defaults to `"worst"`**: the rank is
  `#{scores > true} + (#{scores == true} - 1)`, i.e. the last position among equals. A
  completely flat vector returns **255** instead of 0.
- `tie_break="best"` preserves the previous behaviour (and the published-ASCAD convention,
  which is `#{scores > true}`), for apples-to-apples comparison with the literature.
- `full_ranks()` accepts and forwards `tie_break`, defaulting to `"worst"`. Its only call sites
  pass 8 positional args, so nothing breaks; the added parameter is 9th.
- `ranks_from_log_scores()` — the vectorized twin used by `LLM/` and these experiments — had the
  same optimistic rule and an explicit comment endorsing it. It now matches `rank()` with the
  same parameter, so the repo has one convention rather than two.
- Verified on synthetic cases: flat → 255; strictly ordered → 0 under both; exactly one rival
  tied with the true key → best 0 / worst 1; uniform predictions through `full_ranks` → 255;
  informative predictions under `simulated_key=1` → 0 under both. One apparent FAIL in the first
  test run was my fixture, not the code: with `simulated_key=1` the reference key byte is forced
  to 0, so the probability mass has to sit at `Sbox[p ^ 0]`.

**End-to-end confirmation on the real model** (cached head predictions, 5k attack traces, 16 bytes):

| case | rank-0 under `best` | rank-0 under `worst` | median rank under `worst` |
|---|---|---|---|
| `A0_baseline` | 16/16 | **16/16** | 0 |
| `A7_all_uniform` | 16/16 | **0/16** | 255 |

So the fix removes the artifact without touching the genuine result — the baseline's rank 0 is
not a tie, and the zero-information model's rank 0 was entirely a tie.

**Consequences for anything already stored.** Every rank/GE JSON in this repo written before this
change used the optimistic rule — `multiresnet_34h_50k_subavg_eval.json`, the Scoop `ge_last`
values, `experiments/out/ablation_results.json`, and the AOD/alt-metrics JSONs. For a genuinely
informative model best == worst, so those numbers do not move; where they do move, the old number
was a tie artifact. Re-reading a stored rank-0 without also reading the score spread is still
unsafe — `experiments/rank_degeneracy.py` reports spread, tie count and z in one pass.

**Heads-up for the other active sessions (T-019 GPAM, T-024, T-027 C3):** any evaluation you run
from now on reports the conservative rank. If you are comparing to a published table that uses
the optimistic convention, pass `tie_break="best"` explicitly rather than quietly mixing them.

### 2026-10-03 — Phase 1 NO-GO: unsupervised α is not reachable by variance-ranked selectors, measured

Executed on CPU only (GPU never used). New: `experiments/unsup_alpha.py`,
`experiments/unsup_alpha_residual.py`, `experiments/alpha_variance_budget.py`. Outputs in
`experiments/out/unsup_alpha{,_residual,_variance_budget}.json`. Telegram report sent.

- **Clustering found nothing.** Across 3 label-free selectors (per-sample variance,
  per-trace-normalised variance, seeded random) × 3 band sizes (200/871/2000), k-means(255)
  and equal-frequency binning of the leading component all returned **2.00–2.06 %**
  Hungarian-matched accuracy — *including when coordinates were chosen at random*. A random
  labelling with the same cluster sizes scores identically, so 2.0 % is the inflation floor of
  the matching metric, not signal. Excess over null: ±0.0004.
- **Self-correction:** the 5k-trace smoke read 7.5 % (19× chance). That was upward-biased
  because clusters held ~20 traces each; at 50k it collapses to the floor. The `null_matched`
  control now exists precisely because I quoted 19× before adding it.
- **The supervised ceiling on the same features is also near chance**: ORACLE logistic on the
  top 16 principal components reached only 0.5–0.6 % (chance 0.39 %). α is simply not in the
  high-variance directions, so no clusterer on those directions can recover it. (First ORACLE
  attempt printed 0.36 %, *below* chance — that was `max_iter=200` failing to converge, a
  broken measurement, not a finding.)
- **Measured cause of the whole failure** (`alpha_variance_budget.py`, 20k traces):
  α owns **1.68 %** of total crop trace variance, but a **random** 255-way grouping of the same
  traces already owns **1.29 %** — the (K−1)/(N−1) sampling floor. α's excess over chance is
  **0.39 %**, ratio **1.3×**. Any selector ranking positions by variance is choosing on a
  statistic whose noise is ~75 % of its signal.
- **Why removing known-S variance did not help**: at each position only 1 of the 16 known
  bytes applies and which one is governed by the shuffle, so regressing on all 16 Hamming
  weights dilutes the fit ~16×; SNR-band recall stayed 0.000. Recovering π is required to
  clean the residual, and π is one of the things Phase 1 was supposed to produce — a genuine
  chicken-and-egg, not a tuning problem.
- **Consequence for the task.** Steps 1.1/1.2/1.3 fail the plan's own <10 % gate, so Phase 2
  cannot run as specified: pseudo-C labels are computed from pseudo-α ⊕ pseudo-β, and those are
  not obtainable label-free by this family. Step 1.3 (π) was not attempted — Phase 0 showed π
  is not essential (12/16 genuine with π zeroed) and SNR(π) = 0.13 is weaker still, so it could
  not rescue the pipeline.
- **The oracle boundary, stated plainly.** SNR band selection works (93.9 % with 16 bands)
  *because it uses α labels* to compute between-class variance. Legitimate for profiling-side
  analysis and for understanding the data; it is not a non-worst-case attack step, and must not
  be cited as one.
- **Fallback defect.** Step 3.1 says "first validate with shuffle OFF (~70k traces)". ASCADv2 is
  shuffled in both the 15k crop and the raw shards — there is no shuffle-OFF ASCADv2 database
  here, so that instruction is not executable as written and needs redesigning (e.g. validate on
  a self-generated shuffled-ON synthetic with known key, or on ASCADv1 where a fixed-key
  unshuffled database does exist) before any collision budget is spent.

### 2026-10-03 — Phase 0 DONE: baseline is real, but `rank 0` is unsafe in this repo

`experiments/ablation_heads.py` (new) + `experiments/rank_degeneracy.py` (new). Results in
`experiments/out/ablation_results.json`, `rank_degeneracy.json`, `rank_degeneracy_pi.json`,
`rank_degeneracy_alphaonly.json`. Head predictions cached at
`experiments/out/head_preds_5k.npz` (inference 1.4 s), so re-ablation costs no GPU.
Reported to Telegram (delivery confirmed True).

**The plan's Step 0.1 go/no-go was unanswerable as written, and the reason is a metric defect.**
Replacing all 34 heads with uniform — literally zero information — returns `rank 0` on 16/16
bytes. `ASCAD_test_models.rank():362` scores ties as
`np.where(sorted_proba == key_bytes_proba[real_key])[0][0]`, i.e. the *best* position among
equals, so a flat score vector reports a perfect break. Four of nine ablations had score
spread **exactly 0.000**. Any rank reported here must therefore carry a flatness/z check;
`rank_degeneracy.py` provides one, plus `rank_worst_tie` (ties broken against us).

| case | heads zeroed | rank-0 reported | **genuine** rank-0 | median z | score spread |
|---|---|---|---|---|---|
| A0 baseline | 0 | 16/16 | **16/16** | **+9.50** | 180–302 |
| A4 no π | 16 | 12/16 | **12/16** | +3.29 | non-flat |
| A8 no α only | 1 | 2/16 | 2/16 | +1.84 | non-flat |
| A5 no β only | 1 | 1/16 | 1/16 | +0.61 | 7e-6–1.2e-5 |
| A1 no α,β | 2 | 0/16 | 0/16 | +0.24 | ~5e-7 (noise) |
| A2 no C | 16 | 16/16 | **0/16** | 0.00 | **exactly 0** |
| A3 only π | 18 | 16/16 | 0/16 | 0.00 | exactly 0 |
| A6 no C,α | 17 | 16/16 | 0/16 | −0.50 | exactly 0 |
| A7 all uniform | 34 | 16/16 | 0/16 | ~0 | exactly 0 |

- **T-024's baseline survives scrutiny**: 16/16 at z +9.50, no flat vectors. The worst-case
  34-head model genuinely recovers all 16 key bytes on 5k attack traces. That is the bar
  Phase 2 must meet without mask labels.
- **α and β are each individually near-essential** (2/16 and 1/16 when removed alone), and
  removing both leaves pure noise. Both are on the critical path for Phase 1.
- **π is NOT essential for most bytes** (12/16 genuine with π uniform). Step 1.3 is
  de-prioritized — welcome, since SNR(π) = 0.13 is the hardest label-free target in the file.
- **The plan's question "is the C-head decorative?" cannot be answered by uniform ablation**:
  uniform C makes the dissected posterior identically constant. Structurally, the C heads are
  the *only* carriers of data-dependent information about S, and α/β are algebraic unmasking
  transforms applied after them. So C must remain a trained head — which strengthens the
  plan's central insight (compute C labels from pseudo-α ⊕ pseudo-β + known (p,k), never
  observe them).
- **Citation fix:** `docs/snr_investigation.md`, listed as CRITICAL above, does not exist, and
  `0.072` appears nowhere in `docs/roadmap-break-ascad.md`. The facts are real but live in
  `GPAM/artifacts/snr_crop15k_{rm,c,p}.npy`: rm max 1.996 @867 (871 samples > 0.1), c max
  0.072 @5552 (**0** samples > 0.1), raw rm @205867 (14,576 of 1M). These agree with my
  independently written curve (`docs/2026-09-23/out/SNR-alpha-curve.npy`, max 1.986 @867), so
  the Established Facts table stands — cite the artifacts, not the missing doc.
- **Phase 1 input warning:** the variance localizer flags 1686/1874 patches active
  (`active_ratio 0.90`, `docs/lab-log/v2_variance_localizer.json`) — far too blunt to act as
  the label-free band selector Step 1.1 assumes. The autoencoder's 123/1874 (6.6%) is usable.
  Step 1.1 must measure its band overlap against SNR>0.1 explicitly rather than assume it.

### 2026-10-03 — task created from approved plan

- Summary: Created T-028 from the approved implementation plan in the Antigravity session. The plan has 3 phases: (0) diagnostic ablation, (1) unsupervised mask/permutation discovery, (2) pseudo-label multi-task training. Fallback is WPP23 collision.
- Key insight: C pseudo-labels are computable from pseudo-α + pseudo-β + known (p, k). This sidesteps the C-invisibility problem entirely.
- Files touched: this file, `docs/tasks/INDEX.md`
- Whimsical board: workflow diagram created on board (see Related section)

## Next actions

| Action | I | U | S | Notes |
|--------|---|---|---|-------|
| ~~Step 0.1: Head ablation script~~ | H | H | H | **DONE 2026-10-03** — see Phase 0 log. Needed `rank_degeneracy.py`: uniform ablation reports rank 0 on ties |
| ~~Step 1.1: Unsup. α discovery~~ | H | M | H | **DONE — NO-GO.** k-means/quantisation sit on the matching-null at 2.0%; α owns only 1.3× the variance of a random 255-way split |
| ~~Step 1.2: Unsup. β discovery~~ | H | M | H | **NOT RUN — overtaken.** Same selector family, SNR(β)=1.4 and only 55 samples >0.1, so it inherits Step 1.1's measured blocker; running it would only confirm the null |
| ~~Step 1.3: Unsup. π discovery~~ | H | M | M | **NOT RUN — deprioritised by Phase 0.** π is not essential (12/16 bytes genuine with π uniform) and SNR(π)=0.13 |
| Step 2.1: Pseudo multi-task train | H | M | M | **BLOCKED**: needs pseudo-α/β that Phase 1 showed are not obtainable label-free by this method |
| Step 2.2: Key recovery eval | H | H | H | Blocked behind 2.1. Keep the AOD-GE + shuffled-trace + tie/flatness controls — Phase 0 proved rank 0 is claimable with zero information |
| Step 3.1: WPP23 collision (fallback) | M | L | L | **NOT EXECUTABLE AS WRITTEN**: "validate with shuffle OFF" has no ASCADv2 dataset (crop and raw are both shuffled). Redesign the validation target first |
| Decide: is a non-worst-case v2 break still the goal? | H | H | M | **Superseded by Phase 1R:** the cheap paths are not yet closed; R2 decides in hours of CPU. Collision stays a large build with a 120× shuffle penalty |
| **R1** α/β leakage model (ORACLE): bits vs HW vs one-hot | H | H | H | CPU, minutes. Fixes the design of R2 |
| **R2** Oracle-band clustering gate (per-bit agreement + NMI vs null) | H | H | M | **Decisive stop gate.** PASS ≥ 10% exact pseudo-α (~0.75/bit); STOP < 0.55/bit |
| R3 Label-free band finders beyond variance (kurtosis, bimodality) | H | M | M | Only if R2 passes; PASS = ≥ 5× enrichment and ≥ half the oracle-band clustering |
| R4 Adjacent-difference / per-trace / PCA-removal ablation in the supervised probe | H | M | H | CPU; tests "α lives in fine structure"; falsifier is the difference also collapsing |
| R5 Variance budget by source + natural clock-jitter test | M | M | M | Decomposes the unmeasured ~98% of variance |
| R6 Decoders on cached heads (π Hungarian, joint α,β) | M | L | H | Raises the **worst-case** bar only; zero GPU |
| R7 Re-verify true-β-swap anomaly; reconcile C below uniform vs "C is the carrier" | H | M | H | Before any pseudo-C training is designed |
| ~~Fix the wrong α 0.49% / β 0.41% in `T024:108-111` and `patch-pipeline-survey.md:196`~~ | M | M | H | **DONE 2026-10-04.** `roadmap-break-ascad.md:211` had no α/β figure |

### Priority snapshot

- **Do now:** R1 → R2 (CPU only, no GPU). R2 is the go/stop gate for the whole pseudo-label route.
- **Before next meeting:** report the tie-artifact hazard in `rank()` to whoever quotes v2 rank 0;
  (wrong α/β head accuracies already corrected in T024 and the patch-pipeline survey).
- **Deep work (park):** Tier-1 G1–G3 (GPU, ask first, ≤ 2 h); Steps 3.1–3.2 (collision) — Step
  3.1 still needs a redesigned validation target.

## Related

- Task ids: T-019, T-020, T-024, T-025, T-026
- `docs/lab-log/todo.md` items: T30 (unsup band selection), T31 (resolution-preserving front end)
- External links:
  - WPP23: ePrint 2023/724
  - DDPM denoise: ePrint 2024/966
  - "Is it Really Broken?": ePrint 2026/338
  - Whimsical workflow board: (see board ID below)
