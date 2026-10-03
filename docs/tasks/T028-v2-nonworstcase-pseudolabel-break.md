# T-028: Non-Worst-Case ASCADv2 Break via Pseudo-Label Bootstrap

## Status

`active` — **current focus (set 2026-10-04).** Open item is the go/stop decision below; Phase 0 done (baseline verified, rank-tie hazard found); Phase 1 executed and
closed NO-GO with a measured cause (α owns only 1.3× the trace variance of a random 255-way
split); Phase 2 depends on pseudo-labels Phase 1 showed are unobtainable by variance/PCA
methods; the Step 3.1 fallback is not executable as written. Awaiting a decision — see the
2026-10-03 Phase 1 log entry.

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
| MultiResNet head accs: π 67.42%, C 0.72%, α 0.49%, β 0.41% | T-024 | On 5k attack traces |
| 3O centered product: corr = 0.0009 (dead) | `docs/notes/ascadv2-direct-evidence-explained.md` | All 16 bytes within noise |
| Averaging 2 adjacent samples: 86% → 0.83% accuracy | T-020 / `patch-pipeline-survey.md` | Resolution is binding constraint |
| Variance localizer: detects α at patch 108, β at patch 42 | T-025 | Without any labels |
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

- 2026-10-03: Created as distinct from T-020 (which was the LLM/Transformer pipeline) and T-024 (which was worst-case multi-task). This task is specifically about the non-worst-case / pseudo-label bootstrap approach.
- 2026-10-03: Grounded in `docs/snr_investigation.md` — C is invisible everywhere (SNR 0.072, 4.6× noise floor). No approach in this task may assume C is detectable from the trace.
- 2026-10-03: C pseudo-labels are COMPUTED from pseudo-α + pseudo-β + known (plaintext, key), not observed from traces. This is the central technical insight.
- 2026-10-03: Phase 0 ablation runs BEFORE any unsupervised work, to determine which heads are essential and which can be skipped.

## Working notes / session log

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
| Decide: is a non-worst-case v2 break still the goal? | H | H | M | Phase 1's measured blocker says the cheap paths are closed; collision is a large build with a 120× shuffle penalty |

### Priority snapshot

- **Do now:** nothing on GPU — Phase 1 closed as NO-GO and Phase 2 is blocked by it. Decide
  whether to redesign the collision fallback or stop.
- **Before next meeting:** report the tie-artifact hazard in `rank()` to whoever quotes v2 rank 0.
- **Deep work (park):** Steps 3.1–3.2 (collision); α/β discovery by a criterion other than spread.

## Related

- Task ids: T-019, T-020, T-024, T-025, T-026
- `docs/lab-log/todo.md` items: T30 (unsup band selection), T31 (resolution-preserving front end)
- External links:
  - WPP23: ePrint 2023/724
  - DDPM denoise: ePrint 2024/966
  - "Is it Really Broken?": ePrint 2026/338
  - Whimsical workflow board: (see board ID below)
