# T-041: Permutation-aware π head (use "π is a permutation", not the 2^16 dictionary)

## Status

`in_progress` — registered 2026-10-17. **P1-400k CRF done 2026-10-17: G1 FAIL on both seeds (π worse: exact permutation 65–66% vs 71%), G2 PASS (Δz +0.04 / +0.08), G3 not met. Keep CE training + exact attack routing. P2-D (Σ CE + λ·D, 50k pilot) DONE: G1 PASS by the letter (π +1–3 pp), G2 FAIL (λ=1 Δz −2.4/−0.6; λ=3 attack fails; the decoder takes off later); closed.** P0 done the same day (CPU, ~20 min): pre-registered gate FAIL, but exact permutation marginals raise z on 7/7 models (+0.08 to +0.55 z, 21–43% of oracle-π headroom).** P1/P2 re-estimated below the 40% bar; nothing further runs without a go-ahead.

## Goal

Decide whether the π head can be redesigned to respect the fact that `perm_index` is a **permutation of 16 bytes** (16 positions, each byte exactly once), instead of 16 independent 16-way softmaxes. The redesign must **not** use the empirical finding that only 2^16 permutations occur (T-039, 2026-10-17). That finding comes from π labels and may be a firmware artefact; this task uses only the structural knowledge that a shuffling countermeasure produces a permutation, which is public by design.

"Done" means one of two outcomes:
- (a) a permutation-aware routing gives a measured attack gain beyond run-to-run noise on at least one model that has π headroom;
- (b) a documented negative result: π structure improves π metrics but not the attack, or the improvement is lost in noise.

## Context

- **Current head:** `permind_{s}_output` for s = 0..15, each a softmax over the 16 bytes. This gives the routing matrix R[s, b] = P(slot s processes byte b). Rows sum to 1 (measured max deviation 2.4e-7); columns do not.
  - Training uses 16 independent cross-entropies (`MultiResNet/train.py`, inherited from the MS23-style 34-head design).
  - The decoder (`MultiResNet/decoder_loss.py`; attack: `experiments/w2_decoders.py`) uses column b of R, mixes it over slots, and renormalises per byte via `z_b`. That patches the missing column normalisation numerically, but never imposes "each byte occurs exactly once".
- **Why the slot-wise view stays:** the leakage is per slot. At time slot s the device loads or indexes byte π(s), so a per-slot reading of "which index is processed here" is the natural observation model (T-039 occlusion: π is read in the trace tail). The redesign should keep the slot-wise **unary scores** L[s, b] and change how they are **normalised and coupled**, not what is read.
- **Information budget:** 16 independent heads represent 16^16 = 2^64 configurations; permutations are 16! = 2^44.25. The decoder already has to assign probability to 2^64 − 16! impossible routings.
- Data: `STM32_AES_v2/ASCAD_data/ASCAD_databases/ascadv2-extracted.h5` (read-only). Cached heads: `experiments/out/t039/*_heads.npz`, `experiments/out/t039_e6/*_heads.npz` (softmax probabilities, float64, first 5,000 attack traces).

## Measured diagnostic (2026-10-17, CPU, cached heads, attack traces 0–5k)

Columns:
- **elem:** fraction of slots whose argmax byte is correct.
- **argmax is a perm:** fraction of traces whose 16 argmaxes form a valid permutation.
- **exact (argmax / Hungarian):** fraction of traces with the whole permutation correct, using independent argmax, or using the maximum-likelihood permutation argmax_σ Σ_s log R[s, σ(s)] (Hungarian, `scipy.optimize.linear_sum_assignment`).
- **|col sum − 1|:** mean column deviation.
- **Σ −log at truth:** Σ_s −log R[s, π(s)] before and after Sinkhorn normalisation. After Sinkhorn this is a sum of log-marginals, not a joint likelihood.

| model | elem | argmax is a perm | exact (argmax) | exact (Hungarian) | elem (Hungarian) | \|col sum − 1\| | Σ −log at truth: raw → Sinkhorn |
|---|---|---|---|---|---|---|---|
| E1 300k s2 (all labels) | 0.870 | 0.200 | 0.182 | **0.625** | 0.924 | 0.185 | 6.44 → **3.78** |
| E3 50k s1 (all labels) | 0.725 | 0.023 | 0.019 | **0.280** | 0.797 | 0.305 | 13.98 → 10.70 |
| E6 R0 50k s1 (all labels + decoder) | 0.730 | 0.024 | 0.018 | **0.273** | 0.803 | 0.307 | 13.61 → 10.21 |
| E6 R1 50k s1 (no π labels) | 0.072 | 0.000 | 0.000 | 0.000 | 0.070 | 0.174 | 51.23 → 50.78 |
| E6 R1 300k s2 (no π labels) | 0.524 | 0.001 | 0.000 | **0.099** | 0.603 | 0.370 | 41.14 → 37.42 |

Reading:
- The heads already contain the information to get the whole permutation right far more often than independent argmax shows: 3.4× at 300k and about 15× at 50k. Doing nothing but enforcing the constraint at decode time moves element accuracy by 5–8 points.
- R1 50k is at chance (π not learned there; E6), so no post-hoc structure can help it.
- **Caveat that bounds everything below:** better π metrics are not automatically a better attack. Oracle-π headroom measured so far:
  - E1 300k s2: 45 → 40 traces (true π);
  - 137M baseline: +1.67 z (T-028 C4);
  - R1@300k s2: learned π 105 traces vs oracle 55 vs uniform 640.

  So structure can only matter where that headroom exists. The 50k models' oracle-π headroom has not been measured yet; it is the first thing P0 measures.

## Design space

All options below use only "π ∈ S₁₆". The 2^16 that appears in A2/B3 is the lattice of **column subsets** of any 16×16 assignment problem, not the dataset's dictionary.

### A. Inference-time routing from the existing logits (no retraining, CPU)

- **A1 Sinkhorn normalisation.** Alternate row and column normalisation of R in log space (≈ 20–50 iterations), giving a doubly-stochastic routing matrix that replaces R in the decoder. This is cheap and approximates the marginals of the distribution in A2 (Sinkhorn/Bethe-type approximation).
- **A2 Exact permutation marginals.** Treat the heads as unary potentials of a distribution over permutations,
  P(σ | x) ∝ Π_s R[s, σ(s)],
  and compute its exact marginals μ[s, b] = P(σ(s) = b | x). The normaliser is the permanent of R, which is #P-hard in general (Valiant 1979) but trivial at n = 16 with a subset DP:
  - f[M] = Σ_{b ∈ M} f[M∖b] · R[|M|−1, b] (slots 0..|M|−1 assigned to the byte set M);
  - a backward pass g[M] covers the remaining slots;
  - μ[s, b] = Σ_{|M|=s, b∉M} f[M] · R[s, b] · g[M ∪ b] / f[full].

  This is 2^16 × 16 ≈ 1M multiply-adds per trace, all terms positive (stable in log space, unlike Ryser's signed formula). Note that per-byte key ranking only needs the column marginals μ[·, b], so A2 is the exact routing for the current per-byte decoder under the product-of-heads model.
- **A3 MAP / k-best permutations.** Hungarian gives the single most likely permutation; Murty's algorithm gives the k best. A k-best mixture is a cheap alternative to A2 when the posterior is peaked (E1 300k). A hard MAP alone discards uncertainty and should lose to A1/A2 for the attack.

Modelling caveat: each head was trained to estimate the per-slot posterior p(π(s) = b | x). Multiplying them treats the 16 slot readings as conditionally independent evidence; the per-slot prior is uniform 1/16, so no prior needs dividing out. This is a product-of-experts heuristic, which is exactly why B3 trains the joint model directly.

### B. Training losses on the same 16 × 16 logit matrix L[s, b]

- **B1 Row + column cross-entropy.** Keep the 16 row CEs and add 16 column CEs, each a softmax over slots for byte b, scored at the true slot π⁻¹(b). It costs one extra softmax, gives no exact likelihood, and pushes the logits toward column consistency.
- **B2 Sinkhorn layer + CE** (Adams & Zemel 2011, "Ranking via Sinkhorn Propagation"; Mena et al., ICLR 2018, "Gumbel-Sinkhorn"). Run Sinkhorn on exp(L), then apply CE on the doubly-stochastic matrix. Differentiable and standard for latent permutations; Gumbel noise gives samples when π is unlabelled.
- **B3 Exact permutation CRF.**
  NLL = −Σ_s L[s, π(s)] + log perm(exp L),
  with the log-permanent from A2's DP. The gradient is μ − one-hot, i.e. exact marginals minus truth. A batch of 64 costs ≈ 64 × 2^16 × 16 ≈ 67M operations per step, small next to the trunk; 17 popcount layers vectorise on GPU. This is the principled version of A2: the joint distribution is trained, not assembled from independently trained parts.
- **B4 Autoregressive / Plackett–Luce-type likelihood.**
  P(π) = Π_{s=0..15} softmax over unused bytes of L[s, ·].
  This is the Pointer-Network decoding of Vinyals et al. 2015 and the ListMLE / Plackett–Luce likelihood of Xia et al. 2008, with position-dependent scores. Exact and normalised, trivial to implement as a masked softmax with teacher forcing. Slot order equals time order, which is the device's own order. If L does not condition on earlier choices, its marginals also come from the used-set DP. Main weakness: early slots are scored without knowing which bytes later slots claim; B3 has no such order bias.
- **B5 Differentiable sorting / ranking** (NeuralSort, Grover et al. ICLR 2019; SoftSort, Prillo & Eisenschlos ICML 2020; Blondel et al. ICML 2020; differentiable sorting networks, Petersen et al. ICML 2021 / ICLR 2022; ListNet, Cao et al. 2007). These need one scalar per **byte** ("when is byte b processed?") and sort it. That fits poorly here: the trace exposes "which index is processed at slot s", not a per-byte timing feature. It becomes natural only with byte queries attending over time, which is T-038's cross-attention design. Kept as a pointer, not a first candidate.

### C. Head architecture

- **C1 Bilinear slot × byte scoring:** L[s, b] = q_s · k_b, with slot queries from the trunk (or slot-local windows) and 16 learned byte keys. This adds symmetry across bytes and fewer parameters than 16 separate 16-way heads.
- **C2 Slot interaction before scoring:** a 1–2 layer Transformer over the 16 slot tokens, so a slot can "see" what other slots claim and resolve conflicts (if slot 3 strongly claims byte 7, slot 5 should not), followed by B2 or B3. Under full labels this mostly reproduces what B3 enforces analytically; it matters more when π is unlabelled.

### D. Decoder integration and the unlabelled-π case (E6 R1)

- Swap R for μ (A2/B3) or the Sinkhorn matrix (A1/B2) in both `decoder_nll` and the attack decoder. No other change is needed: the per-byte formula uses column marginals.
- **For R1 (π learned only through the decoder) structure may matter most:**
  - Independent heads admit degenerate solutions (several slots claiming the same byte) that cost nothing in the row-wise objective. A permutation-normalised routing removes them.
  - It also bounds the routing. The R1@300k runs needed `--heads-fp32` because unanchored π logits overflowed fp16, and α collapsed on a drifted data region. A bounded, doubly-stochastic routing might mitigate both. This is a hypothesis, not tested.
- Exact joint inference over σ together with the C evidence couples the 16 bytes and would need joint key enumeration; it is out of scope. Per-byte marginals are the right object for per-byte key ranking.

## Plan

Pre-registration rule: freeze gates before any number. Use works_v2 (20 pooled permutations + constant-trace nulls) for the attack, seeds 1 and 2, and judge gains against the measured run-to-run noise (about 2.6× in trace counts at 50k; E6 R2L).

- [x] (I:M U:L S:H) **P0 — inference-only, CPU, ~minutes.** DONE 2026-10-17 (user go-ahead "可以使用它看看"), see Results P0.
  - Measure oracle-π headroom on the 50k models (E3 s1/s2, R0 s1/s2), which is not yet recorded.
  - On every model with headroom (≥ 1 z or ≥ 1.5× traces between A0 and oracle π), compare R vs A1 vs A2 vs A3-k-best inside the existing attack decoder: z, median traces, works_v2.
  - Prior estimate: π metrics improve ≈ certain (table above); an attack gain beyond noise on at least one model ≈ **55%**. The most likely place is R1@300k s2 (105 vs oracle 55); the least likely is E1 300k (headroom 5 traces).
  - Needs a go-ahead; above the 40% bar.
- [ ] (I:M U:L S:M) **P1 — train with B3 (exact CRF) or B1 at the 50k R0 recipe, seeds 1, 2** (~2 × 8 min GPU). Run only if P0 shows the 50k models have π headroom. Estimate ≈ 35% for an attack gain beyond noise, below the bar as things stand; re-estimate after P0.
- [ ] (I:M U:L S:M) **P2 — R1 at 50k with a B2/B3-normalised routing in the decoder** (π unlabelled). Question: does the permutation constraint make π learnable at 50k, where independent heads stayed at chance (7%)? Estimate ≈ 30%: the E6 diagnosis was that each π slot gets too little signal before C becomes informative, which a constraint does not fix by itself. Below the bar; do not run unless P0/P1 change the picture.
- [ ] (I:L U:L S:M) C2 slot-interaction head: only after P1, and only if B3 helps.

## Pre-registration P0 (frozen 2026-10-17, before any P0 number; user: "可以使用它看看，用了以后是否能直接提升我们的破解速率")

- **Models (cached heads, attack traces 0–5k):** E1 300k s1/s2, E3 50k s1/s2, E6 R0 50k s1/s2, E6 R1@300k s2 (π unlabelled). R1@300k s1 is excluded because its failure is α collapse and oracle π does not rescue it.
- **Routings**, each written as a modified heads npz with only `permind_*` replaced, then scored with the unchanged `experiments/c4_oracle_swaps.py` arm A0:
  - `raw` (as trained);
  - `hung` (Hungarian MAP, one-hot);
  - `sink` (Sinkhorn, 50 iterations, log space);
  - `exact` (exact marginals of P(σ) ∝ Π R[s, σ(s)] via subset DP).

  B4 (oracle π) and F2 (uniform π) on `raw` give the headroom.
- **Headroom set H:** models with B4 − A0 ≥ +1.0 z or median traces(A0) / traces(B4) ≥ 1.5.
- **PASS:** on at least one model in H, `exact` or `sink` gives Δz ≥ +1.0 or median traces ≤ A0 / 1.5, with no fewer non-flat rank-0 bytes than A0, and no model anywhere loses rank-0 bytes under the routing that passes. Also report the fraction of headroom closed, (z_X − z_A0) / (z_B4 − z_A0).
- **Expectation stated in advance:** `hung` ≤ `exact` on uncertain models (a hard MAP discards routing uncertainty); E1 300k cannot show a 1.5× gain (headroom 45 → 40).
- **DP check before use:** exact marginals must match brute-force enumeration on random 6×6 matrices to 1e-10, and rows/columns of μ must sum to 1.

## Results P0 (2026-10-17 04:11–04:31, CPU only, `scripts/run_t041_p0.sh` → `experiments/out/t041/`)

DP self-test passed: brute force 6×6 max error 4.4e-16; 16×16 row/column deviation 5e-15. Every arm has 16/16 non-flat rank-0 bytes; GE cross-check 0.000e+00 everywhere.

**Attack: median z and median traces to rank 0 (attack set, first 5k traces)**

| model | raw (A0) | hung | sink | **exact** | oracle π (B4) | uniform π (F2) | B4 − A0 | exact Δz | headroom closed |
|---|---|---|---|---|---|---|---|---|---|
| R1@300k s2 (π unlabelled) | +12.57 / 105 | +12.53 / 100 | +12.81 / 90 | **+13.11 / 90** | +14.30 / 55 | +6.57 / 640 | +1.73 | +0.54 | 31% |
| E3 50k s1 | +8.98 / 175 | +8.74 / 175 | +9.11 / 210 | **+9.18 / 165** | +9.96 / 180 | +2.39 / 2090 (7/16) | +0.98 | +0.20 | 21% |
| E3 50k s2 | +6.78 / 510 | +7.05 / 445 | +7.06 / 420 | **+7.34 / 425** | +8.33 / 380 | +2.01 / 2040 (4/16) | +1.55 | +0.55 | 36% |
| R0 50k s1 | +11.14 / 135 | +11.05 / 135 | +11.36 / 130 | **+11.36 / 125** | +11.98 / 100 | +3.77 / 1540 (14/16) | +0.84 | +0.22 | 27% |
| R0 50k s2 | +10.35 / 160 | +10.41 / 160 | +10.57 / 160 | **+10.72 / 145** | +11.41 / 110 | +3.15 / 3110 (10/16) | +1.06 | +0.37 | 35% |
| E1 300k s1 | +14.04 / 50 | +14.06 / 50 | +14.13 / 50 | +14.12 / 50 | +14.25 / 45 | +7.48 / 480 | +0.21 | +0.08 | 39% |
| E1 300k s2 | +14.04 / 45 | +14.00 / 60 | +14.13 / 60 | +14.11 / 55 | +14.22 / 40 | +7.31 / 445 | +0.18 | +0.08 | 43% |

**π metrics, raw → exact:**

| model | elem | mean p(true byte) |
|---|---|---|
| R1@300k s2 | 0.524 → 0.610 | 0.412 → 0.515 |
| E3 50k | 0.72 → 0.80 | 0.65 → 0.76 |
| R0 50k | 0.72 → 0.80 | 0.65 → 0.76 |
| E1 300k | 0.88 → 0.93 | 0.85 → 0.92 |

Sinkhorn reaches the same elem as exact but barely moves p(true) (e.g. 0.412 → 0.415).

**Verdict against the frozen gate: FAIL.**
- Headroom set H = {R1@300k s2, E3 s2, R0 s2}.
- The best gain inside H is +0.55 z / 1.2× traces (E3 s2), under the +1.0 z / 1.5× bar.
- No arm lost rank-0 bytes.

**What did hold:**
- `exact` raises z on **7/7 models**; a sign test gives p = 1/128 ≈ 0.008. The gain is +0.08 to +0.55 z and closes 21–43% of the oracle-π headroom.
- The ordering matches the prediction: exact ≥ sink > raw ≈ hung.
- Hungarian helps π accuracy as much as exact but not the attack. It is worst or tied on 5/7, because a hard MAP sends C mass confidently to the wrong byte whenever the MAP is wrong.
- Trace counts move in 5-trace steps and are noisier than z: on E1 300k s2, z rises while traces go 45 → 55.

**Reading:** the permutation constraint is real, free information. A no-retraining exact decode recovers about a third of what knowing π would give. But oracle-π headroom is itself small (≤ +1.7 z) on every model, so no attack-time π change can be a large lever.

## Pre-registration P1-400k: train with the permutation CRF (frozen 2026-10-17, before any P1 number; user: "我同意你训练B3: CRF（400k profiling trace）")

**Run.**
- Identical to the 400k reference (`scripts/run_t039_400k.sh`: Compact B h64, `--train-len 400000`, 15 epochs, RAdam 1e-4, batch 64, `--sub-avg`, C weight 0.5, decoder NLL weight 1.0, seeds 1 and 2), plus `--pi-loss crf`.
- `--pi-loss crf` replaces the 16 per-slot π CEs with one permutation-CRF NLL (`MultiResNet/perm_crf.py`) and routes the decoder loss through the exact, differentiable permutation marginals.
- Checkpoints `t041_crf_t400k_seed{1,2}[_last].pth`; unit `t041-p1`; script `scripts/run_t041_p1.sh`.
- Cost: +38 ms per batch, so ≈ 480 s/epoch vs 254 s, ≈ 2 h per seed.

**Comparator.** The reference 400k checkpoints, scored with the same (exact-routing) decoder:
- s1 best: z +14.4457, 30 traces;
- s2 best: z +14.4448, 40 traces.

Measured before the run: oracle-π headroom at 400k is only +0.14 / +0.11 z (no trace gain). So **π cannot improve the 400k attack through routing**; any attack change would come from the shared trunk.

**Gates** (best checkpoints, attack traces 0–5k, exact routing, same seed vs same seed):
- **G1 π quality (primary, prior ≈ 60%).** Both seeds: mean μ(true byte) **and** exact-permutation recovery (Hungarian MAP) higher than the reference.
- **G2 attack non-inferiority.** Both seeds: 16/16 non-flat rank-0, works_v2 PASS, median z ≥ reference − 0.3.
- **G3 attack gain (secondary, prior ≤ 10%).** Median z ≥ reference + 0.5 on both seeds. Given the π headroom, a pass would be attributed to the trunk and needs a follow-up.
- **Reported, not gated:**
  - α / β / C head quality vs the reference (interference check);
  - α top-1 on the attack set ≥ 0.99 (the R1@300k collapse check);
  - val π CRF NLL per epoch;
  - wall time.

## Results P1-400k (DONE 2026-10-17 09:27; both seeds)

**Seed 1** (best = epoch 8; 15 epochs at 457 s/epoch, 1 h 55 min). Attack traces 0–5k, exact routing for both models.

| | ref 400k s1 best | **CRF 400k s1 best** | CRF s1 last |
|---|---|---|---|
| median z / traces | +14.4457 / 30 | **+14.4866 / 35** | +14.2721 / 40 |
| works_v2 | PASS | PASS | PASS |
| oracle-π z (B4) | +14.5844 | +14.5307 | +14.4317 |
| uniform-π z / traces (F2) | +8.29 / 465 | +8.34 / 365 | +7.80 / 495 |
| **mean μ(true byte)** | **0.9475** | 0.9214 | — |
| **exact permutation (MAP)** | **0.7144** | 0.6554 | — |
| element (MAP) / row argmax | 0.954 / 0.911 | 0.941 / 0.812 | — / 0.805 |
| α top-1 | 0.9994 | 0.9988 | 0.9978 |
| β top-1 / gain | 0.2328 / 2.985 | 0.2352 / 2.950 | 0.2412 / 2.700 |
| C mean gain | 0.390 | 0.406 | 0.315 |

Validation curves vs the reference at the same epoch: β top-1 and decoder NLL lead at every epoch (decoder 86.24 at ep8 vs 86.57; best 86.23 vs 86.43). Row-argmax π is lower throughout (0.83 vs 0.92), as expected for CRF-trained rows.

**Gate status after seed 1:**
- **G1 FAIL**, already decided because G1 needs both seeds. CRF π is *worse* on the attack set under exact routing: μ(true) 0.921 vs 0.948, exact permutation 65.5% vs 71.4%.
- **G2 PASS on s1:** 16/16, works_v2, Δz +0.04.
- **G3 not met on s1:** Δz +0.04 < +0.5.
- α shows no collapse (0.9988).

**Seed 2** (pipeline finished 09:27). Same scoring.

| | ref 400k s2 best | **CRF 400k s2 best** | CRF s2 last |
|---|---|---|---|
| median z / traces | +14.4448 / 40 | **+14.5206 / 40** | +14.3462 / 40 |
| works_v2 | PASS | PASS | PASS |
| oracle-π z (B4) | +14.5580 | +14.6175 | +14.4638 |
| uniform-π z / traces (F2) | +8.03 / 400 | +8.45 / 355 | +7.77 / 390 |
| **mean μ(true byte)** | **0.945** | 0.922 | — |
| **exact permutation (MAP)** | **0.7066** | 0.6644 | — |
| element (MAP) / row argmax | 0.953 / 0.906 | 0.943 / 0.820 | — / 0.811 |
| α top-1 | 0.998 | 0.9992 | 0.9984 |
| β top-1 / gain | 0.2344 / 3.031 | 0.2546 / 3.068 | 0.2472 / 2.769 |
| C mean gain | 0.383 | 0.405 | 0.299 |

**Final gates (both seeds):**
- **G1 FAIL**, replicated: μ(true) 0.921 / 0.922 vs 0.948 / 0.945; exact permutation 65.5% / 66.4% vs 71.4% / 70.7%.
- **G2 PASS:** 16/16, works_v2 on both; Δz +0.04 / +0.08.
- **G3 not met:** Δz +0.04 / +0.08 < +0.5.
- Side observation, within noise and not gated: C gain +0.016 / +0.022 and uniform-π z +0.05 / +0.42 above the reference on both seeds; β mixed (−0.04 / +0.04 nats). A slightly better trunk, a worse π head.

**Reading (provisional, both seeds agree):** the reference recipe (independent CE training + exact decoding at attack time) beats training the joint permutation model. A plausible mechanism, not tested: the CRF gradient μ − one-hot vanishes for a slot once the other slots pin the permutation ("explaining away"). Each π row therefore gets less training signal than under CE, which keeps pushing every row to be sharp, and that redundancy is what survives the attack-set shift. The CRF gains on validation β / decoder NLL do not carry to the attack set (β gain 2.95 vs 2.99).

**Identity behind P1 (checked numerically to 1e-14):** Σ_s CE_π[s] = L_CRF + D, with D = Σ_s logsumexp_b L[s,b] − log perm(exp L) = −log P_indep(the 16 independent rows draw a valid permutation) ≥ 0. dD/dL = softmax_row − μ. The reference therefore already contains L_CRF; P1 removed only D, the term that makes each row agree on its own with the joint marginal μ. P1 says: D weight 1 (reference) gives better π than D weight 0 (CRF). A mixed CE + λ·CRF = (1+λ)L_CRF + D lies between the two (D relative weight 1/(1+λ)), so it is expected to land between them, not above the reference (≈15%; below the bar, not run).

## Pre-registration P2-D: CE + λ·D, 50k pilot (frozen 2026-10-17, before any P2 number; user: "CE + λ·D（λ > 1）更meaningful。可以试试吗？")

- **Change:** π objective = Σ CE_π + λ·D (= L_CRF + (1+λ)·D), `train.py --pi-d-extra λ`. Everything else = the E6 R0 50k recipe (34 heads, all labels, C weight 0.5, decoder weight 1.0 with row-softmax routing, 15 epochs). Smoke test: D = 13.68 at init ≈ 16 ln16 − ln 16! = 13.7.
- **Arms:** λ ∈ {1, 3} (D weight 2, 4) × seeds {1, 2}; references R0 50k s1/s2, same seed. Exact routing at attack time for all. `scripts/run_t041_p2.sh` → `experiments/out/t041_p2/`.
- **Why 50k first:** after exact routing, 50k still has +0.62 / +0.69 z oracle-π headroom (R0 s1/s2); at 400k it is ≤ +0.14 z. Runs take ~8 min each, versus ~2 h at 400k.
- **G1 π quality (primary, prior ≈ 45%).** For at least one λ, on both seeds: mean μ(true byte) **and** exact-permutation recovery (Hungarian MAP) higher than R0 of the same seed.
- **G2 attack (secondary, prior ≈ 20%).** For the same λ, on both seeds: median z ≥ R0 + 0.3, 16/16 non-flat rank-0, and works_v2 PASS.
- **Guard:** α top-1 must not drop more than 0.01 below R0.
- **Follow-up:** only if G1 passes, propose the best λ at 400k (≈ 4 h; separate go-ahead).

## Results P2-D (DONE 2026-10-17 17:50; `experiments/out/t041_p2/`)

Attack traces 0–5k, exact routing, best checkpoints; R0 = E6 R0 50k same seed.

| | R0 s1 | λ=1 s1 | λ=3 s1 | R0 s2 | λ=1 s2 | λ=3 s2 |
|---|---|---|---|---|---|---|
| median z / traces | **+11.36 / 125** | +8.94 / 275 | −0.01 / — (0/16) | **+10.72 / 145** | +10.09 / 125 | +0.69 / 650 (1/16) |
| works_v2 | PASS | PASS | **FAIL** | PASS | PASS | **FAIL** |
| oracle-π z (B4) | +11.98 | +9.92 | +0.35 | +11.41 | +11.04 | +0.48 |
| mean μ(true byte) | 0.771 | 0.785 | 0.780 | 0.755 | 0.761 | 0.767 |
| exact permutation (MAP) | 0.273 | 0.294 | 0.301 | 0.261 | 0.270 | 0.290 |
| α top-1 | 0.9994 | 1.0000 | 1.0000 | 0.9992 | 0.9988 | 0.9996 |
| β top-1 / gain | 0.081 / 1.73 | 0.068 / 1.62 | 0.041 / 1.18 | 0.063 / 1.60 | 0.060 / 1.62 | 0.034 / 1.18 |
| C mean gain | −0.021 | −0.092 | −0.106 | −0.036 | −0.033 | −0.116 |

**Gates:**
- **G1 PASS by the letter (both λ, both seeds)**, but small: μ +0.006 to +0.014, exact permutation +0.9 to +2.9 pp.
- **G2 FAIL:** Δz −2.42 / −0.63 (λ=1), −11.4 / −10.0 (λ=3).
- Guard α: OK.

**Diagnosis.** The loss is in the trunk, not in π. Oracle π does not rescue λ=3 (z +0.35 / +0.48), so π routing is not the cause. At 50k, R0's decoder NLL leaves its uniform value 88.72 only at epoch 11. With λ=1 it leaves at epoch 14; with λ=3 it never leaves within 15 epochs (88.75 at the end). β and C learn later too. D (≈13.7 at init, ×λ) competes for the shared trunk early on. A likely mechanism (untested) is the global grad-norm clip at 1.0: a large D gradient takes the clip budget and slows every other head. 50k sits on a phase transition near the end of training, so any delay costs a lot.

**Decision.** P2-D is closed and not followed at 400k, although the pre-registration allowed it after G1:
- the π gain is ≤ +3 pp;
- the 400k oracle-π headroom is ≤ +0.14 z;
- the attack-cost risk was just shown.

Estimated P(400k is useful) < 20%, below the bar. The result for the paper is that π structure helps at attack time (exact marginals) and not as a training loss: neither CRF (D weight 0) nor extra D (weight 2, 4) beats plain CE (weight 1) on the attack.

## Decisions

- **2026-10-17 (user: "我同意你训练B3: CRF（400k profiling trace）"): P1 runs at 400k, not 50k.** The user chose the paper's operating point. Consequence, stated before launch: the 400k oracle-π headroom is +0.11–0.14 z, so the primary gate is π quality (G1), with attack non-inferiority (G2). Code: `MultiResNet/perm_crf.py`, which passes a brute-force check (3.6e-15), log 16! at uniform, a double-backward gradcheck, and matches the attack-time DP in float32 to 2.9e-6. `train.py --pi-loss crf` and `decoder_loss.head_probs(pi_routing="crf")` were added; the default `ce` path is unchanged (smoke: CE − CRF initial loss gap = 16 ln16 − ln 16! as expected). Launched 2026-10-17 in unit `t041-p1`.

- **2026-10-17 (user: "把精确边缘概率作为默认的攻击路由"): exact permutation marginals are the default attack routing.**
  - Implemented in `perm_routing.py` (torch, CUDA if available: 5,000 traces in 0.6 s; equals brute force to 4e-16 and the P0 numpy version to 3e-14) and wired into the shared decoders:
    - `ASCAD_test_models.multilabel_predict`: MultiResNet/evaluate.py, TimesNet summary, ablation scripts;
    - `experiments/w2_decoders.route_marginal` and `run_variant`: w6_nochead_eval, x1_occlusion, train_multihead_cv_ts (Spectrogram2D / InceptionTime1D multihead), c4_oracle_swaps A0;
    - `experiments/w3_s3_eval.decode_and_measure`: t039_controls, TimesNet S3 eval;
    - `compact_verify.route_marginal_local`.
  - `ASCAD_PI_ROUTING=raw` restores the old routing. Verified: R0 50k s1 gives z +11.3639 (exact) and +11.1401 (raw, = recorded); multilabel_predict ≡ route_marginal to 4e-17 in both modes. w2 / w7 / w6 test suites and `tests/test_compact_multiresnet.py` pass.
  - **Consequence:** c4's R0 reproduction gate for the built-in historical models only passes with `ASCAD_PI_ROUTING=raw`.
  - Not changed: GPAM scoring (`GPAM/artifacts/C-ge71-score.py`, its own per-byte formula and a reproduction of the published model); the training-time `MultiResNet/decoder_loss.py` (training, not attack; would be P1/P2); explicit-matrix routings (oracle / Hungarian / Sinkhorn / `route_with_matrix`).

- 2026-10-17: Scope fixed by the user. Use only the permutation structure, not the 2^16 dictionary (that is T-039's T39-PIDICT, a separate question). Keep slot-wise unary scores; redesign normalisation and coupling.
- 2026-10-17: The first experiment must be inference-only (P0), because the diagnostic shows the information is already in the heads and the attack headroom is model-dependent and partly unmeasured.

## Working notes / session log

### 2026-10-17 — registration

- Summary: document written at the user's request; diagnostic table computed from cached heads (CPU, ~1 min, `.h5` read-only). Hungarian MAP lifts exact-permutation recovery 0.18 → 0.63 (E1 300k) and 0.02 → 0.28 (50k), with no retraining.
- Files touched: this file, `docs/tasks/INDEX.md`, `docs/lab-log/todo.md`.
- Open questions: oracle-π headroom of the 50k models; whether DP marginals (A2) beat Sinkhorn (A1) by anything measurable on peaked heads.

## Next actions

| Action | I | U | S | Notes |
|--------|---|---|---|-------|
| ~~P0 inference-only comparison~~ | M | L | H | **DONE 2026-10-17: gate FAIL; exact marginals +z on 7/7 models, ≤ +0.55 z** |
| ~~Adopt exact π marginals as the default attack routing~~ | M | M | H | **DONE 2026-10-17** (`perm_routing.py`; `ASCAD_PI_ROUTING=raw` for old numbers) |
| P1 B3 CRF loss at 50k | M | L | M | GPU ~16 min; re-estimated ≈20% for the +1 z gate (P0 already takes ~1/3 of the oracle headroom, so P1 must close nearly all of the remaining +0.5–1.2 z) — below the bar |
| ~~P2-D Σ CE + λ·D at 50k~~ | M | L | M | **DONE 2026-10-17: G1 PASS (small), G2 FAIL; closed, no 400k follow-up** |
| CE + λ·CRF mix at 400k | L | L | M | ≈15%: interpolates between the measured D=1 (ref) and D=0 (CRF) points, and 400k oracle-π headroom is ≤ +0.14 z anyway — below bar |
| P2 R1 50k with permutation-normalised routing | M | L | M | ≈30%; below bar |

### Priority snapshot

- **Do now:** nothing without a go-ahead.
- **Before next meeting:** the diagnostic table is a ready-made figure ("independent π heads waste the permutation constraint").
- **Deep work (park):** P1, P2, C2.

## Related

- Task ids: T-039 (E6 R1/R1@300k, T39-PIDICT dictionary decoder), T-038 (cross-attention byte queries, relevant to B5), T-028 C4 (oracle-π headroom), T-031 N2 (exact-permutation 1.3% on the 137M model).
- `docs/lab-log/todo.md` items: T41-P0.
- Literature pointers (from memory, **verify before citing**): Adams & Zemel 2011 (arXiv:1106.1925); Mena, Belanger, Linderman, Snoek, ICLR 2018; Linderman et al., AISTATS 2018 (Birkhoff polytope reparameterisation); Vinyals, Fortunato, Jaitly, NeurIPS 2015; Xia et al., ICML 2008 (ListMLE); Cao et al., ICML 2007 (ListNet); Grover et al., ICLR 2019; Prillo & Eisenschlos, ICML 2020; Blondel et al., ICML 2020; Petersen et al., ICML 2021 / ICLR 2022; Cuturi, NeurIPS 2013 (Sinkhorn); Valiant 1979 (permanent); Ryser 1963. SCA side: Veyrat-Charvillon, Medwed, Kerckhof, Standaert, ASIACRYPT 2012 (shuffling study).
