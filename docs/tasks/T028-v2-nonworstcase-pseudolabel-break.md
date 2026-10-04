# T-028: Non-Worst-Case ASCADv2 Break via Pseudo-Label Bootstrap

## Status

`active` — **current focus (set 2026-10-04; T-029 is the side task).** Phase 0 done (baseline
verified, rank-tie hazard found). Phase 1 (variance-ranked selectors + k-means) closed NO-GO with
a measured cause. **2026-10-04 evidence review reopened the question as Phase 1R**: Phase 1 never
separated *finding α's bands* from *clustering them*, and never clustered the oracle bands, so
"α is not recoverable label-free" is not yet established. Phase 1R is a CPU-only Tier-0 battery
(R1–R10 run so far, no GPU). **R1–R10 changed the picture — see "Revision after R1–R10" below.**
C is first-order visible once labelled slot-aligned (|z| 56.5); the u×β×C three-leak product
replicated on 450k fresh traces (Stouffer z +6.04); but every label-free band finder tried
(variance, kurtosis) fails, and S-information per trace is only ~5e-5 nats (measured lower bound, R10) even with oracle
π and positions. The raw-bit R2 STOP rule is superseded and R2 was **not** run. Phase 2
(pseudo-label multi-task) is demoted. The Step 3.1 fallback is still not executable as written.

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
  - **C is weak but NOT invisible** (corrected 2026-10-04). Its 256-class SNR of 0.072 is only 4.6× the 256-class noise floor at 16k traces, but a 1-dof Hamming-weight test on the slot-aligned label gives z = 56.5 at 50k traces (R², 0.064; R8). The earlier "invisible" reading was a detector-power and threshold artifact (256-class SNR is a poor detector for HW leakage; the ">0.1" cut-off is arbitrary), plus a byte-indexed label in the 3O test. C is only readable slot-aligned (needs π) and as HW(C).
  - Non-worst-case: training must NOT use `alpha_mask`, `beta_mask`, `masks`, `desync`, `permind`, or `sbox_masked` fields from HDF5 metadata.
  - MAY use: `traces`, `plaintext`, `key`, `ciphertext` fields (standard profiled attack).
  - GPU budget: ≤ 2 hours total across all phases.
  - **Trace budget** (baseline config, confirmed by user 2026-10-04): 50,000 profiling traces, 5,000 attack traces (the crop's attack split holds 10,000). Any run that exceeds it is *beyond-budget*: fine for measuring where information lives, never a budget-compliant claim.
  - Must run AOD-GE and shuffled-trace null controls before claiming any rank 0.

## Established Facts (Do Not Re-Derive)

These are confirmed results. An executing agent should use them as ground truth, not re-investigate:

| Fact | Source | Number |
|------|--------|--------|
| SNR(α) = 2.0 at sample 867 (crop), 205867 (raw) | `docs/snr_investigation.md` | 871 samples > 0.1 in crop, 14,576 in raw |
| SNR(β) = 1.4 at sample 338 (crop), 205338 (raw) | `docs/snr_investigation.md` | 55 samples > 0.1 in crop, 267 in raw |
| SNR(π) = 0.13 at sample 5552 (crop), 0.21 at 282070 (raw) | `docs/snr_investigation.md` | 1 sample > 0.1 in crop, 27 in raw |
| SNR(C) = 0.072 on a 256-class SNR, 0 samples > 0.1 — **but NOT invisible: slot-aligned HW(C) 1-dof z = 56.5 at 50k traces, 530 samples |z| > 6 (R8, 2026-10-04)** | `docs/snr_investigation.md` | Identical in crop and raw |
| MultiResNet 34h + sub_avg: all 16 bytes rank 0 at 5k traces | T-024 | z-scores +3.89σ to +6.86σ |
| MultiResNet head accs, 5k attack traces, **recomputed 2026-10-04** from `experiments/out/head_preds_5k.npz` vs `Attack_traces/labels`: **α 100.00%**, β 5.84%, C 0.69%, π 67.03%. Log-lik gain over uniform: α +5.48 nats (max 5.545), β +1.51, π +1.80 (max 2.77), **C −0.09** | this session | Corrects T-024's "α 0.49%, β 0.41%" (wrong label column, source not located). Same error appears in `T024:108-111`, `docs/notes/patch-pipeline-survey.md:196`, `docs/roadmap-break-ascad.md:211` and **has been corrected in all three locations** (2026-10-04). α is `masks[:,18]`, β is `masks[:,17]` |
| 3O centered product: corr = 0.0009 (dead) — **INVALID TEST, superseded by R9/R9b** (C was labelled by byte, not by time slot; α was modelled as HW(α), not HW(log₃ α)) | `docs/notes/ascadv2-direct-evidence-explained.md` | All 16 bytes within noise |
| **CORRECTED 2026-10-04:** Block-mean pooling of 2/4/8 samples gives 0.893/0.909/0.900 vs 0.860 raw (no loss); the 0.0083 came from pooling bug in `LLM/band_scan.py:85` which averaged over the *group* axis and returned only b features | `LLM/band_scan.py:85` / `r4c_pooling_bug_replicate.py` | Coarse patches are benign, not fatal |
| Variance localizer flags α's patch 108 and β's patch 42 — but it also flags **90% of all patches** active (1686/1874) | T-025 | Without labels, but not a band selector: Phase 1 measured SNR-band recall 0.000 for label-free variance selectors. The old wording "detects α" overstated it |
| α's band samples drift over acquisition order: holdout−train mean 0.134 sd vs 0.011 expected, trend corr +0.81 over 10 blocks | R1b (`experiments/out/r1b_alpha_function.log`) | Measured on α's 871 band samples, first 50k profiling traces. Contiguous splits are contaminated; use interleaved |
| α's peak sample (867) leaks HW(log₃ α): R² 0.651 vs one-hot 0.652; HW(α) itself 0.0004. β's peak (338) is near-HW: hw 0.601, bits8 0.637, one-hot 0.634 | R1/R1b | Only log₃ of 8 pre-listed candidates fits, and only at the peak: over the whole band it explains 17–28% of one-hot excess. Firmware source not checked |
| `desync` metadata is exactly 0 on all 20k profiling and 10k attack traces checked | this session | No artificial jitter was applied. Says nothing about natural clock/sub-sample jitter (untested — see D5) |
| `sub_avg` = subtract the **global profiling mean trace** (one value per sample position), no labels, nothing per-trace | `ASCAD_train_models.py:1157-1160`, `ASCAD_test_models.py:948-953` | Provable no-op for the band probe (SNR-16 0.9395 → 0.9395). **No 34-head MultiResNet run without it exists**, so T-024's 16/16 cannot be credited to `sub_avg` |
| **CORRECTED 2026-10-04:** Mean-of-2 / 4 / 8 adjacent samples: α top-1 0.8603 → 0.893 / 0.909 / 0.900 (correctly pooled, identical MLP/data/seeds) vs 0.0083 / 0.0065 / 0.0090 from pooling bug | `LLM/band_scan.py:85` → fixed to `.mean(2)`, verified in `r4c_pooling_bug_replicate.py` | M |
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
| 2 | **CORRECTED 2026-10-04:** Block-mean pooling of 2/4/8 adjacent samples *raises* α accuracy to 0.893/0.909/0.900 (vs 0.860 raw). The earlier "0.008 collapse" came from pooling bug in `LLM/band_scan.py:85` (`.mean(1)` should be `.mean(2)`), which returned only b features per sample instead of n/b. Coarse patches are benign. | `r4c_pooling_bug_replicate.py`; bug fixed in `band_scan.py:85` and `amp_check.py:83` | **M** |
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
- **[PARTLY RETRACTED, see C3: stride-5 subsampling keeps ~20% of α's one-hot excess per offset, and R4b shows no loss under stride or block-mean]**  *folds* clock periodicity rather than suppressing it, and the
  v2 runs used `input_stride=5` plus a 32-segment pool — subsampling, which the resolution
  result predicts is fatal for α. The TimesNet v2 nulls therefore do not test these hypotheses,
  and **TimesNet should not be revived for this task**. (F/I)
- **[RETRACTED 2026-10-04: pooling bug in `LLM/band_scan.py:85`; see C3]** ~~Resolution verdict~~  keep per-sample information *inside the first learned layer* (patchify,
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

#### C. Tier-0 battery (CPU only, ~hours). Thresholds below were approved and frozen by the user on 2026-10-04 and cannot be moved after any R-number is seen. **R1 has run; its results put R2's STOP rule in question — see the R2 design note below the table. R2 is on hold until you decide.**

| ID | Test | Purpose / gate |
|----|------|----------------|
| R1 ✅ **DONE 2026-10-04** (+R1b; numbers in the session log) | **Leakage model of α and β** (ORACLE, understanding only): on the SNR>0.1 band, between-class R² of the samples against (a) the 8 label bits, (b) Hamming weight, (c) 255-way one-hot. Also β | Decides whether the label-free target is 255 classes or 8 binary latents, which fixes R2's design. No gate |
| R2 | **Oracle-band clustering gate (decisive).** Cluster the SNR>0.1 band of α (871 samples, 50k traces): (a) 8-bit factored latent (sign of leading PCs / ICA / binary mixture), (b) 255-way GMM/k-means. Score by **per-bit agreement** (chance 0.5, shuffled-label null) and NMI vs null, never Hungarian-255 alone | **PASS** = pseudo-α exact-match ≥ 10% (≈ 0.75 per-bit if the 8 bit errors are independent, since 0.75⁸ ≈ 0.10; 10% is ~25× chance and about GPAM's holdout β head, 10.48%, the weakest mask head the worst-case pipeline still uses — a judgement, not a derived minimum). **STOP** if per-bit < 0.55. 0.55–0.75 is a grey zone: continue only if R3 shows a route |
| R3 | **Label-free band finders beyond variance**: per-sample excess kurtosis, bimodality / mixture-BIC, pairwise-sample dependence. Score by enrichment of SNR>0.1 in the top-871 (random base rate 871/15000 = 5.8%) | Run only if R2 passes. **PASS** = precision ≥ 29% (≥ 5× random) **and** R2's clustering on the *selected* bands recovers ≥ half of R2's oracle-band per-bit agreement |
| R4 ✅ **DONE as R4a/R4b/R4c 2026-10-04 (see C3)** | **Background/resolution ablation inside the supervised probe** (same 2-layer MLP, same 1,600 samples, same α label): raw / global-mean subtraction (reproduce the no-op) / per-trace mean + scale / top-k PCA removal / **adjacent difference** x[t]−x[t−1] / adjacent mean (reproduce the 0.008 kill) | Tests "α lives in fine structure": the difference should survive while the sum dies. Picks the label-free front end that is also resolution-preserving. Falsifier: if the difference also collapses, α needs absolute level |
| R5 | **Variance budget by source** (ORACLE, understanding only): per-sample fraction explained by α, β, π, the 16 known S, top-k label-free PCs, residual. Plus a clock-phase lag test (cross-correlation of each trace to the mean in a quiet region) to check for *natural* jitter | Decomposes the ~98% nobody has measured; says whether re-alignment or PCA removal is worth building |
| R6 | **Decoder upside on cached heads** (zero GPU, `head_preds_5k.npz`): Hungarian assignment over the 16×16 π matrix vs the 67.0% argmax; joint (α,β) marginalisation over the 16 slots vs per-slot dissection on ~200 traces | Raises the **worst-case** baseline bar. Not a non-worst-case step; say so in any report |
| R7 | **Re-verify two fork-reported anomalies**: (a) substituting the *true* β for the β head lowers key z (−0.02 to −0.11); (b) C heads are below uniform (−0.09, M) yet are the structural carrier (Phase 0) | If (a) holds, Phase 0's "C is the carrier" is incomplete and Phase 2's pseudo-C training target needs rethinking before any GPU is spent |

**R2 status: superseded, not run.** R1 showed α's raw bits are the wrong coordinates, so the
frozen raw-bit STOP rule (per-bit < 0.55) could fire on a *correct* clustering. R3a/R8–R10 (below)
then answered the question R2 was meant to gate, so R2 is dropped rather than amended. The
frozen PASS rule (≥ 10% exact pseudo-α) is kept as the bar any future label-free α finder must clear.

#### C2. Revision after R1–R10 (2026-10-04) — what works, what does not, what to do

All numbers ORACLE/diagnostic unless stated; scripts in `experiments/r*.py`, outputs in `experiments/out/r*`.

**What works (measured):**

| Finding | Number | Source |
|---|---|---|
| Slot-aligned C leaks first-order as HW(C) | max \|z\| 56.5 @50k, R² 0.064, 530 samples \|z\|>6; the per-slot C samples sit at 5552, 5925, … (≈373 apart) | R8 |
| u = HW(log₃ α) at sample 867, β ≈ HW at 338 | R² 0.66, 0.60 | R1b, R8 |
| S, log₃S, log-sum w, and the pre-β product P are **not** first-order visible (the leak is only in C) | max \|z\| 4.0–4.4 vs null max 4.85–5.4 | R8 |
| u×β×C three-sample product tracks the corrected moment m3p(S) | Stouffer z **+6.04** on 450k never-used traces, frozen choices; old model z −0.83 | R9b |
| Model-free: product separates S classes | F_z +28.98 on the same 450k | R9b |

**What does not work (measured):**

| Finding | Number | Source |
|---|---|---|
| Label-free kurtosis finds α's band | 5/871 hits = 0.1× random (precision 0.6%); sample 867 ranks #8156/15000 | R3a |
| Variance / k-means finds α's band | SNR-band recall 0.000; 2.0% = matching-null floor | Phase 1 |
| Blind Bayes read of S from the three frozen samples is far below the budget bar | I = +5.31e-05 nats/trace at 200k evaluation traces (16/16 slots > 0); bar is 1.1e-3 at 5k attack traces (21× short) and 5.6e-4 at 10k (10× short). Oracle-mask reference 2.8e-02 | R10 |
| 3O null of the earlier plan | invalid test, not evidence | R9 |

**Why the old story was wrong, and what replaces it.** (1) "C is invisible" came from a 256-class SNR with
an arbitrary 0.1 cut-off and a byte-indexed label. (2) "3O is dead" used that label and HW(α). (3) The
product leakage is real but tiny: a Gaussian Bayes read of three samples gives ≈ 5e-5 nats/trace (R10, 200k evaluation traces) *even
with π and positions known*, which means N_min ≳ ln(256)/I ≈ 5.5e4–1.8e5+ traces (a necessary condition
only). **Trace-budget caveat (corrected):** the budget is 50k profiling / 5k attack (crop attack split: 10,000 traces, each with its own key). A key byte needs I ≥ ln 256 / N_attack: **1.1e-3 nats/trace at 5k, 5.5e-4 at 10k**. R10's blind estimate is ~3e-5 (20k evaluation traces; 200k rerun pending), so the three-sample second-order read is roughly 20–35× short of the budget. Reading many more samples per leak (R12) is the only route I can see to close that gap. Within budget: R1/R1b/R3a/R8/R9 selection and the R10 emission fits (all 50k). **Beyond budget, measurement only:** R9b (450k labelled traces) and the R10 200k evaluation.

**Why GPAM-style classification works where this fails.** It is mask-supervised: α (100%), π (86.5%) and
β (10–14%) are read at their own bands with labels, then C at ~0 nats per head is combined algebraically.
The label-free path would have to find α's band without labels, and R3a/Phase 1 say simple statistics cannot.

**Revised direction (ranked by what R1–R10 support):**

1. **Label-free π first.** π is the one latent that controls *where* C (and, per R8, only C) leaks;
   without it every slot-aligned statistic is blind. π needs ≥ the 4-way loop-index structure and is
   learnable with 86.5% accuracy under supervision, so this is the load-bearing unknown. Unlike α, it
   has never been attacked label-free.
2. **A second-order attack that uses multi-sample leak vectors**, not one sample per leak (R9/R10 used
   one). Fit/score on the SNR>0.1 α band (871 samples) and the 55-sample β band using a *linear* HW(log₃)
   readout, then product with the C slot sample. This is the cheapest test of whether I rises above
   1e-4 nats; R10 is the harness.
3. **Fix the R2 gate permanently.** The PASS bar (≥10% exact pseudo-α) stands; the raw-bit STOP is
   retired.
4. **Demote Phase 2** (pseudo-label multi-task training): it needs pseudo-α/β that no label-free finder
   has produced. Keep only as a downstream consumer once (1) or (2) yields labels.
5. **Tier-1 G3 (S-only GPAM on GPU)** stays as the explicit expected-null control.

**Not claimed:** none of R8–R10 is a non-worst-case attack — every sample choice, sign and emission
parameter used oracle labels. They show where information lives, not that it can be found without labels.

#### C3. Why TimesNet, Time-LLM and PatchTST-style models failed — and a retraction (2026-10-04)

**Retraction first.** The premise "averaging adjacent samples destroys α, so resolution must be preserved" was wrong.
`LLM/band_scan.py:85` (and `LLM/amp_check.py:83`) pool with `.reshape(N,-1,b).mean(1)`, which averages the *group* axis and
returns **b features** (stored `n_features` 2/4/8 in `RESOL-*.json`). Replicated with identical data, seeds and MLP, only the pooling changed
(`experiments/r4c_pooling_bug_replicate.py`, 20k/2k traces, window 0–1599, 2 repeats):

| block | as written (b features) | correct (n/b features) |
|---|---|---|
| 1 (raw) | 0.8595 | — |
| 2 | 0.0068 (2 features) | **0.8930** (800) |
| 4 | 0.0048 (4 features) | **0.9085** (400) |
| 8 | 0.0065 (8 features) | **0.9000** (200) |

Further measurements (50k traces, interleaved split, ORACLE): adjacent band samples carry nearly the same class-mean profile
(median corr +0.99; 0% negative), the mean of a pair has *higher* one-hot R² than either sample (median 0.233 vs 0.186/0.196), and the
difference carries none (0.001) (`r4a_adjacent_structure.py`). Each stride-5 offset keeps ~20% of the band's one-hot excess and its best sample
(R² 0.46–0.66) (R4a). My earlier "α lives in fine structure" hypothesis was the *opposite* of the data: α leaks as a smooth, locally redundant waveform.
Retracted in this file: Established-Facts rows (resolution), Phase 1R table A row 2, the "resolution verdict" and the TimesNet-subsampling bullets.
**Code bugs fixed (2026-10-04):** `LLM/band_scan.py:85` and `LLM/amp_check.py:83` (`.mean(1)` → `.mean(2)`).
**Still carrying the false claim as of 2026-10-04, now corrected (see below):** `T020` (RESOL entries), `docs/notes/patch-pipeline-survey.md` lines 30–31, 210, 254, and the todo.md T31 line.

**What each failure does and does not show.**

| Family | What was run (code/log) | Result | Measured cause | Not diagnosed |
|---|---|---|---|---|
| Time-LLM (frozen GPT-2) | `leakage_localizer/exp_v1_timellm_gpt2.py`: frozen causal GPT-2, mean pool over patches, reprogramming onto word embeddings; ASCADv1 desync0, 40 epochs | chance; rank never computed (T-016) | none: the three suspected flaws (mean pool, frozen causal backbone, word-prototype reprogramming) were never ablated one at a time; never run on v2 | which flaw matters. "Mean pooling is mathematically incompatible with second-order attacks" (`llm-sca-diagnosis-and-plan.md`) is an overstatement: attention layers mix tokens before the pool, so it holds only if the backbone does not compute the cross-position product |
| PatchTST-style | **No PatchTST exists in the repo.** Proxy = A2 (`LLM/patch_transformer_sca.py`): patch 16 / stride 8, d_model 128, 4 layers, attention pool over 1,874 tokens, 1.09M params, α target, 20k traces, lr 1e-3 cosine, batch 32, patience 4 | `FAIL_CHANCE`, **stopped at epoch 6 after 5.3 min** of a 4 h cap; train CE dipped below chance while val stayed flat | α is readable by a plain MLP on a 100-sample window (12× chance) and by ~870 band samples (0.86), so the leak is there; resolution is exonerated (above) | whether A2 would learn given more epochs: GPAM's own curve sat at chance for epochs 0–11 (lr 1e-3, 40k traces) before α jumped 18.6% → 91% at epochs 12→13, so a patience-4 / 6-epoch stop **would have killed GPAM too** (inference from the logs, not run) |
| TimesNet v2 | multilabel `timesnet_multi` / `_ascad_head_multi`: stride 5, 32-segment pool, 30–60 epochs, 100k traces; α one of 18 heads | train CE 5.551 → 5.462 (chance 5.545), mean rank 124–137 ≈ random; α-only 512-trace overfit 5.57 → 5.21, "learns, not fully memorizes" | stride/pool are *not* the cause (R4a/R4b: no loss under stride 5 or block-mean) | the α-head accuracy was **never measured**; so whether the trunk learned α at all is unknown. The one v2 success (pooled, byte 4, GE 2.9) sits at the label-frequency prior's GE 3 (`ascadv2-label-frequency.md`) |

**What differs in GPAM (read from code by a fork, tag F):** 400-sample tokens (patch 400 on raw) through `Dense(192)`, three full-attention GAU
blocks, a head that averages over *filters* and keeps the token axis (so each head reads a 2,500-wide position-aware vector rather than a mean over
tokens), lr 1e-3, and a 150-epoch budget with no early stop. That head is flatten-like, which matches the v1 pattern (flatten MLP/CNN/TimesNet rank 0,
mean-pool GPT-2 chance).

**Most plausible common cause (inference, not tested):** optimisation, not input resolution or absence of leakage. Plateau-then-transition dynamics
need a protocol that waits (GPAM: lr 5e-5 never moved, lr 1e-3 broke after 12 epochs); A2 stopped at 6 and TimesNet's α-only probe did not memorise.
**Cheap discriminating test (GPU, ask first, ≤ 1 h):** α-only, 20k traces, no early stop, 30 epochs, three architectures (A2-style patch transformer with a
position-aware head, TimesNet trunk, GPAM-style) on (i) the 100-sample band 800–900, then (ii) the full 15k crop. If (i) learns and (ii) does not, localisation
is the obstacle; if neither learns, it is the protocol/architecture. This replaces G2 (patch-size sweep), which the retraction makes low-value.

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

- 2026-10-04 (later): **R2 superseded, not run.** Raw-bit STOP retired; ≥10% exact pseudo-α PASS bar kept. Reason: R1 showed the coordinates were wrong, and R3a/R8–R10 answered the underlying question first (label-free band finding fails; C and the 3-leak product are real but near the information floor).
- 2026-10-04 (later): "C physically invisible" and "3O dead" removed as constraints; both rested on a byte-indexed C label and HW(α). **Do not re-derive the old numbers; cite R8/R9b.**
- 2026-10-04 (later): New priority order for the non-worst-case goal: label-free π → multi-sample second-order read → (only then) Phase 2. Phase 2 demoted.
- 2026-10-04: Phase 1 re-opened as **Phase 1R**. The NO-GO stands for variance-ranked selection + k-means only. Order of work: R1 → R2 (hard stop gate) → R3 → R4–R7. Thresholds in the plan were approved and frozen by the user on 2026-10-04. The task goal is unchanged (no mask labels in training); every ORACLE number is for understanding only.
- 2026-10-04: The α/β/C/π algebra is treated as a **decoder and validator**, not a label source: α and β leak directly and independently, and the pairwise C relation needs C readable (SNR 0.072). Do not spend effort on EM over the joint (α,β) latent or on tied/shared-codebook heads.
- 2026-10-03: Created as distinct from T-020 (which was the LLM/Transformer pipeline) and T-024 (which was worst-case multi-task). This task is specifically about the non-worst-case / pseudo-label bootstrap approach.
- 2026-10-03: Grounded in `docs/snr_investigation.md` — C is invisible everywhere (SNR 0.072, 4.6× noise floor). No approach in this task may assume C is detectable from the trace.
- 2026-10-03: C pseudo-labels are COMPUTED from pseudo-α + pseudo-β + known (plaintext, key), not observed from traces. This is the central technical insight.
- 2026-10-03: Phase 0 ablation runs BEFORE any unsupervised work, to determine which heads are essential and which can be skipped.

## Working notes / session log

### 2026-10-04 (latest) — R4a/R4b/R4c: the resolution premise was a pooling bug

Asked why TimesNet / Time-LLM / PatchTST fail. Reading the T-020 probe code before explaining, I found `.mean(1)` where `.mean(2)` was meant (`LLM/band_scan.py:85`, `LLM/amp_check.py:83`). Replicated: correct pooling gives 0.893 / 0.909 / 0.900 for block 2 / 4 / 8 vs 0.860 raw. I had relayed the "averaging kills α" number from a research fork and wrote it into this file as established; I had re-verified other numbers but not that one. R4b (nearest-class-mean readout, weak: 0.034 raw) agrees on ordering: block-mean 0.032–0.048, stride-5 0.033, adjacent difference 0.005 (chance 0.004). Analysis and retractions in C3. No GPU used.

### 2026-10-04 (later) — R3a, R8, R9, R9b, R10: C is visible, the 3-leak product is real, band-finding is the wall

New: `experiments/{r3a_kurtosis_probe,r8_log_domain_scan,r9_corrected_3o,r9b_replicate,r10_bayes_bound}.py`
(+ logs/JSON in `experiments/out/`). CPU only. All ORACLE/diagnostic. Numbers are in section C2.

- **R3a (label-free kurtosis, drift-removed):** sample 867 has excess kurtosis −0.147 (SE 0.017), almost
  the global median −0.232; top-871 platykurtic samples contain 5 band samples (0.1× random). My
  pre-run back-of-envelope (−0.1 at 867) was right in size, but a −0.1 signal under a robust sd of 0.87
  cannot be found by ranking. Kurtosis is not a band finder here.
- **R8:** the earlier "C invisible" claim does not survive a slot-aligned HW(C) test (|z| 56.5). Verified
  first that `sbox_masked[:,i] == C[perm[i]]` on 100% of traces; the old 3O oracle labelled byte 4 with
  byte-indexed C, correct for only 6.5% of traces.
- **R9 / R9b:** first run (selection on even blocks, test on odd blocks of the first 50k) gave Stouffer
  z +3.21 vs synthetic expectation +1.51 and a per-slot spread consistent with noise. I did not trust it:
  I froze every choice and ran a replication on traces 50,000–499,999 (450k): **+6.04, replicated**, old
  model −0.83. A bug fixed on the way: R9's printed "selection corr" was scaled wrongly (displayed −0.005
  where the real value is −0.81); the fit was unaffected, the display was.
- **R10:** with positions frozen and Gaussian emissions fitted by OLS, the blind Bayes read of S carries
  +3.1e-05 ± 2.1e-05 nats/trace at 20k evaluation traces (within noise at that size) — versus +2.7e-02
  with masks known. The shuffled-S control sits at −6.6e-05. A 200k-trace rerun was started because
  20k is inside the standard error (result: next bullet).
- **R10 at 200k evaluation traces (beyond budget, measurement only):** blind I = +5.31e-05 nats/trace, all 16 slots positive (6.6e-06 to 8.8e-05). Slot SE averages 2.9e-05, so the mean-of-16 SE is 7.3e-06 if slots were independent and 2.9e-05 if fully correlated (they share α, β and the evaluation traces); the truth is in between, so the estimate sits between ~1.8σ and ~7σ. It is consistent in sign with R9b's independent z +6.04. The shuffled-S control is **negative on 16/16 slots** (−7.7e-05): expected for a plug-in score, since a posterior that is not uniform assigns below-average probability to an unrelated S; its size is close to the blind value, which fits a weak-signal model (inference, not proven). The score is a proper-scoring lower bound on the information in those three samples under the fitted Gaussian-HW model, so a better emission model or more samples can only raise it. Against the budget bar: **21× short at 5k attack traces, 10× at 10k.**
- **Limits I have not removed:** one C sample per slot and one sample each for u and β; a Gaussian
  emission model on HW; no check of the firmware source for log-table multiplication; the Stouffer
  combination treats slots as independent although they share α and β per trace.
- **Does not change:** no non-worst-case break exists in this repo; every R8–R10 selection used labels.

### 2026-10-04 — R1 / R1b run (CPU, ORACLE, understanding only)

New: `experiments/{_r_common,r1_leakage_model,r1b_alpha_function}.py`; outputs
`experiments/out/r1_leakage_model{,_contiguous}.{json,log}`, `r1b_alpha_function{,_contiguous}.{json,log}`.
50,000 profiling traces, 80/20 split, shuffled-label null for every model. All numbers below are
the final **interleaved** split unless marked.

- **Acquisition-order drift is large (measured, first thing found).** In α's 871-sample band the
  holdout-minus-train mean differs by a median **0.134 sd** against **0.011** expected for a
  stationary process; 95% of samples are beyond 5× that; the median trend correlation across 10
  consecutive trace blocks is **+0.81**. My first, contiguous 80/20 split therefore gave every
  model a *negative* null R² (about −0.02 per sample), which is consistent with that drift and
  not with over-fitting nine parameters. I re-ran with a seeded interleaved split and kept the
  contiguous outputs under `*_contiguous.*`. **The conclusions below did not change between the
  two splits** (α bits8 ratio 0.104 → 0.119; β hw ratio 0.956 → 0.958). Drift is a real part of the
  "background waveform" and was not previously measured (feeds R5; any label-free clustering run
  in trace order is exposed to it).
- **α: not bit-linear, not Hamming-weight in α itself.** Ratios to the one-hot ceiling (sum of
  held-out R² excess over the null): hw 0.075, bits8 0.119, bits+pair 0.278. 829 of the 871
  significant samples are "arbitrary" and carry 96% of the one-hot excess.
- **α's strongest sample leaks HW(log₃ α).** Sample 867: one-hot R² 0.652; hw(log₃ α) 0.651;
  bits(log₃ α) 0.654; HW(α) 0.0004; bits8(α) 0.028. One parameter on log₃ α reproduces the whole
  one-hot ceiling there. Candidates were fixed in advance (id, inverse, log₃, exp₃, S-box, ×2, ×3,
  rotate-left-1); only log₃ fits. **Limits:** log₃ explains just 17% (hw) to 28% (bits) of the
  one-hot excess over the whole band, so most band samples leak other functions that I have not
  identified; and I did not check the ASCADv2 firmware source, so "log-table multiplication" is
  an inference from the fit.
- **β: near-Hamming-weight at its peak.** Sample 338: one-hot R² 0.634, hw 0.601, bits8 0.637.
  Ratios hw 0.958, bits8 0.988 — the bit weights differ by about 2.5× (−7.7 … −3.1), so it is
  HW-like with unequal bit weights, not pure HW. A pure-HW read caps exact β match at 9/256 =
  3.5%; GPAM's 10.48% on `rout` shows other samples carry more.
- **A smoke run at 4k traces called β "bit-linear" (0.845); at 50k it is near-HW (0.958).** Treat
  small-N R1 verdicts as unreliable.
- **Consequence:** the R2 STOP rule needs a decision (design note under the Tier-0 table). The
  label-free band finder in R3 now has a concrete target: a sample whose histogram is a
  binomially weighted 9-level mixture (weights 1:8:28:56:70:56:28:8:1) — if that is how α's best
  sample looks without labels, a per-sample mixture test should find it. Not yet tested.
- Nothing trained, no GPU.

### 2026-10-04 — Phase 1R: evidence review of what makes ASCADv2 classification work

Three read-only forks (GPAM mechanism, background/resolution, α/β/C/π algebra), then I re-verified
the numbers that change the plan. Nothing was trained; no GPU; no files edited outside docs.

- **A wrong number in the Established Facts table was corrected.** `T024:108-111` and
  `patch-pipeline-survey.md:196` (and the roadmap's π line) said α 0.49%, β 0.41%. Recomputed
  from `head_preds_5k.npz` against `Attack_traces/labels`: **α 100.00%, β 5.84%, C 0.69%, π 67.03%**.
  The 0.49% / 0.41% figures were never reproduced; a wrong label column is the likely cause (not
  located). `MultiResNet/evaluate.py` states in its docstring that it reports head accuracies but
  contains no such code. **All three locations now carry the corrected figures** (T024, the survey row, and
  `roadmap-break-ascad.md:211`, which previously quoted π only and now also lists α/β). I verified the
  on-disk text; the edits landed via a concurrent write, not my own tool calls.
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
| ~~**R1** α/β leakage model (ORACLE): bits vs HW vs one-hot~~ | H | H | H | **DONE 2026-10-04** (+R1b). α: HW(log₃ α) at sample 867, arbitrary elsewhere; β: near-HW; large acquisition-order drift |
| ~~**R2** Oracle-band clustering gate~~ | H | H | M | **SUPERSEDED 2026-10-04, not run.** Raw-bit STOP retired; ≥10% exact pseudo-α PASS bar kept for any future label-free α finder |
| ~~R3 Label-free band finders beyond variance~~ | H | M | M | **R3a DONE, FAIL:** kurtosis enrichment 0.1× random. Mixture-BIC / pairwise dependence untried, low expectation |
| ~~R4 adjacent-difference ablation~~ | M | M | H | **DONE as R4a/b/c.** Overturned the resolution premise (C3). Per-trace/PCA-removal arms still untried, low value |
| R5 Variance budget by source + natural clock-jitter test | M | M | M | Drift already measured (0.134 sd); still wanted for π-localisation |
| R6 Decoders on cached heads (π Hungarian, joint α,β) | M | L | H | Raises the **worst-case** bar only; zero GPU |
| R7 Re-verify true-β-swap anomaly | M | L | H | Demoted: C heads below uniform is now explained by per-slot HW leakage being weak and shuffled |
| ~~R8 slot-aligned log-domain scan~~ | H | H | H | **DONE.** C visible (|z| 56.5); S/log S/w/P not first-order visible |
| ~~R9/R9b corrected 3-leak product~~ | H | H | H | **DONE.** Replicated on 450k fresh traces, Stouffer +6.04 |
| ~~R10 Bayes read of S from 3 samples~~ | H | H | H | **DONE.** 200k: 5.3e-5 nats/trace (lower bound), 16/16 slots > 0; 21× / 10× short of the 5k / 10k-trace bar |
| **R11 Label-free π** (the load-bearing unknown): find, without π labels, which slot/time each of the 16 C-samples belongs to; first test = can a clustering of the 5552…11147 C-sample window recover `perm_index` above a shuffle null | H | H | M | PASS bar to pre-register before running. π supervised = 86.5% (GPAM), so the information exists |
| **R12 Multi-sample second-order read**: replace one sample per leak by a linear HW(log₃ α)/HW(β) readout over their SNR bands, product with the slot C sample, re-run R10's information estimate | H | M | M | Harness exists (`r10_bayes_bound.py`); question is whether I rises above ~1e-4 nats |
| ~~Fix the wrong α 0.49% / β 0.41% in `T024:108-111`, `patch-pipeline-survey.md:196`, `roadmap-break-ascad.md:211`~~ | M | M | H | **DONE 2026-10-04** (T024, survey and roadmap line 211 all verified on disk) |

### Priority snapshot

- **Do now:** R11 (label-free π) and R12 (multi-sample second-order read) — both CPU, both build on
  the R8–R10 harness. Pre-register each PASS bar *before* running. Nothing needs a GPU.
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
