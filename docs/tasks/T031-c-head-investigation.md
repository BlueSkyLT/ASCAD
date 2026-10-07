# T-031: Investigate and improve C-head learning in worst-case multi-task ASCADv2

## Status

`in_progress` — registered 2026-10-05 from T-030 findings. **T31-N4b DONE 2026-10-07→08 — the frozen rule fired on its first branch: adopt C loss weight 0.5 — and it INVERTS T31-N4.** Four seeded 15-epoch runs (3296.7 s GPU, best-val epoch 7 for all four): at **w=0.5** as-trained z is **+10.2582 / +9.7092** (mean 9.984, spread **0.549**) with median traces-to-rank-0 **145 / 150** (spread 5); at **w=0.25** it is **+5.3737 / +9.5211** (mean 7.447, spread **4.147**) with **955 / 180** traces (spread 775). mean z(0.5) − mean z(0.25) = **+2.5363** against the registered 0.5 threshold. **The gate-passing arm of N4 is the unstable one and the gate-failing arm is the best operating point found in this task.** N4's w=0.25 head-quality claim does not survive either: the bar was β ≥ 7.5% and seed 1 gives **7.22%** (seed 2 8.02%), so `w025_head_quality_claim.stands: false`. Carrier robustness holds everywhere: **16/16 non-flat rank-0 in all six C-including runs** (three seeds, two weights) while N2 (w=0) is 0/16 — the cliff stays between 0.25 and 0, and inside [0.25, 1] the weight mostly buys variance. On **validation** β the two weights are indistinguishable (means 0.0802 vs 0.0742; within-weight seed spread 0.76–1.20 pp ≥ the 0.60 pp between-weight gap) and both clear w=1 (0.0593): head quality separates {0.25, 0.5} from 1, the attack separates 0.5 from 0.25. **Boundary kept honest:** w=0.5 also beats the baseline's attack numbers on both metrics (9.984 vs +9.5047; 147.5 vs 170 traces) but the baseline is a single unseeded run, so "better than the baseline" is **suggested, not established** — the missing measurement is one seeded w=1 replicate (~16 min). Methodological finding for the programme: **single-run gates on this task are unreliable; `--seed` should be mandatory in every future comparison.** N5/N6 remain untriggered and the plan says stop after the first pass. Earlier: **T31-N4 DONE 2026-10-07 — PASS at C loss weight 0.25, FAIL at 0.5: the middle exists** (both arms single-run, since superseded by N4b). At w=0.25 the attack still recovers 16/16 non-flat (no oracle) while β top-1 goes 5.84% → **7.74%** (gain +1.5112 → +1.8316, mean p(true) 0.0340 → 0.0525) and π 67.03% → **71.53%** with exact-permutation recovery 0.60% → **1.22%**. Two things the gate did not settle: its **median traces-to-rank-0 is 335 vs the baseline's 170** (per-byte distributions overlap, so worse median rather than half as efficient), and **w=0.5 — which failed the β bar — has the best attack of all five models** at z **+9.7556**. On validation the β response is monotone in w (0.0593 → 0.0724 → 0.0770 → 0.1002, cw025 above control in 13/15 paired epochs, π in 15/15); on the attack split it is not, and each weight is one run against a 1.44 pp two-seed band. **Per the cascade's own rule, N5 and N6 are not triggered.** Earlier: **T31-N2 DONE 2026-10-07 — the first POSITIVE result in this task: deleting the 16 C heads from the loss genuinely improves the other heads.** β top-1 5.84% → **9.30%** (gain +1.5112 → **+2.0002** nats, mean p(true) ×2.03), π element 67.03% → **72.13%**, exact-permutation recovery ×2.2, α unchanged. Verified against a **matched 34-head control** (added because the trainer sets no RNG seed; two 34-head runs differ by **1.44 pp** on β), with paired per-epoch validation putting N2 above control in 13/15 epochs for β and 14/15 for π. Its own attack is dead exactly as W2 predicted (as-trained z +0.4741, 0/16, GE 85.0), so the trade is real and sharp. **The clip-budget alternative was measured and rejected:** dropping C raises the post-clip step 1.78× (pre-clip ‖g‖ 122.9 → 69.4, clip factor 0.0081 → 0.0144), but a global clip is a *uniform* rescale and RAdam barely responds to one (displacement ratio 0.959, direction cosine 0.99960 at the measured factor) ⇒ the mechanism is the gradient **sink**, not an accidental learning rate. One middle remains untested: **C present but down-weighted** (β = 4.4–5.8% at weight 1, 9.30% at weight 0) → proposed **T31-N4**, weights 0.25/0.5, ~28 min GPU, gate pre-registered in `T031-reports/T31-N2-c-deletion-retrain.md` §6, awaiting go-ahead. Earlier: **T31-N1 DONE 2026-10-07 (gate FAIL; the gate's fallback branch met): the one-(α,β)-per-trace constraint cannot be exploited by a decoder.** It is *provably* vacuous while each byte's own key is marginalised (the hypothesis family tiles all 256 values, so the other 15 slots say nothing about the mask — identity deviation 8.66e-15, and the correctly-normalised joint decoder moves z by +4.2e-05 and nothing else); plugging key estimates makes it informative but self-confirming (pass-1 top-1 key accuracy 0.50% ⇒ EM +5.72/460 and +5.48/420 vs the reference +9.50/170); and the C heads do not know the mask even given the key (slot-only α 0.96% / β 0.64% at chance 0.39%; oracle-key EM lifts β only to 11.34%). β **correctness** (+3.4 z, W2), β **confidence** (~0.3 z) and β **posterior shape** (~0: τ=20 still +9.2032) separate cleanly, so only the *head* can close the gap — a training-side decision, not a decoding one. Earlier: **W2 DONE 2026-10-06 (gate FAIL for the C-head deletion — a LUT is not an observation — but it located the lever on the mask side: oracle masks on the same weak C head give z +12.93 and 95 traces vs +9.50/170). W3/W4 from the original spec are superseded; the next candidate is a joint (α,β)-consistency decoder plus β-head calibration, and the C-deletion retrain is only worth doing as an interference measurement with a new gate. Earlier: **W1 DONE 2026-10-05 (gate PASS)**, report `T031-reports/W1-c-diagnosis.md`: label imbalance refuted (the masked target is statistically uniform), C is near-invisible per sample (SNR excess 0.045–0.061, 21–25 significant samples vs α's 1.83/2,872) and unbeatable by a linear probe, yet it eats 52.6% of the loss and 38.3% of the trunk gradient — and the unmasked targets are *worse*, so W4 as specced is predicted to fail and W3's specced direction is contrary to the evidence. No compute is committed past W1 until the user rules on that direction. **Split of roles:** an implementing agent runs the steps below and writes reports to `docs/tasks/T031-reports/`; the requesting agent (and the user) read those reports and decide the next step.

## Goal

Understand why the C head (masked S-box output) sits at chance (0.39% top-1, −0.05 nats) across all tested models, and test interventions to move it. **All results are worst-case (mask labels in training).** Success means either (a) C headroom gain ≥ +1 nats in key recovery on 5k attack traces, or (b) a diagnosis that explains why C cannot be learned from ASCADv2 labels.

## Context

- **Why this task exists:** T-030's three-pronged investigation (S1, S2, S3) converged on a single blocker. S2 (oracle decoder analysis) ranked headroom: C +6.28 z ≫ β +3.42 > π +1.67 > α 0.00. S3 (fair TimesNet re-test) proved α can be learned (81.54%) while C remains at chance (0.39%) and key recovery fails (0/16 rank 0, median z −0.377). The C problem is not an encoder limitation (both 8.4M and 137M models suffer it) and not a label-mapping error (verified in S3 §4). It is a target/loss problem.

- **C in both models:**
  - MultiResNet: 16 C heads (slot-indexed `sbox_masked`), 0.69% top-1, gain −0.090 nats
  - Fair TimesNet (S3): 16 C heads (slot-indexed `sbox_masked`), 0.39% top-1, gain −0.053 nats
  - Old TimesNet (S1): 18 C heads (byte-indexed `sbox_masked_with_perm`), 0.36–0.42% top-1, gain −0.000…−0.045 nats

- **Why C scores below uniform — premise CORRECTED by W1 on 2026-10-05** (the original text said "the 16
  classes (slot values 0–15) are not balanced in the training data... a constant predictor that guesses
  the mode reaches ~0.7% top-1 and −0.002 nats (checked in S1)"). Three errors, all measured in
  `experiments/out/w4_c_diagnosis.json`: the C target has **256 classes**, not 16 (16 is the number of
  slots/heads); on 50,000 profiling traces the mode is **0.45–0.49%** against 0.39% uniform with
  chi² p = 0.16–0.62, i.e. **statistically uniform** — the ~0.6–0.76% figures are mode frequencies
  estimated on the 5,000-trace attack split (≈19 traces/class, ±4 sampling noise), and scoring a
  "constant predictor" fitted on the same traces it is scored on is not a fair bar; and no "−0.002
  nats" for a constant predictor exists in S1's output, while a prior-only head is *above* uniform by
  construction (+0.0022…+0.0028 nats here), so imbalance cannot produce −0.09 at all.
  What survives from the original framing: **C's −0.09 nats does mean the head actively avoids its true
  class (anti-informative as a standalone classifier), yet T-028 Phase 0 and T-030 S2 both show that
  replacing C with uniform destroys key recovery** (S2: uniform C → median z 0.00, all 16 bytes flat;
  oracle C → +15.78). That paradox is still the starting point; its imbalance explanation is not.

- **Related tasks:** T-030 (trunk swap — closed), T-028 (label verification, non-worst-case route), T-024 (baseline), T-019 (GPAM).

- **Code / data:**
  - Baseline: `STM32_AES_v2/ASCAD_data/ASCAD_trained_models/multiresnet_34h_50k_subavg.pth` (137M, 16/16 rank 0, z +9.50)
  - Cached baseline predictions: `experiments/out/head_preds_5k.npz`
  - Fair TimesNet (S3): `STM32_AES_v2/ASCAD_data/ASCAD_trained_models/timesnet_ascad_head_multi_ml1_subavg_s5p32_t100k_e15.pth` (8.4M, 0/16 rank 0, z −0.377)
  - Training entry: `ASCAD_train_models.py` (head loss at lines ~687, unified loss per head; multilabel layouts at lines ~748–751, ~1170–1184)
  - C label: `build_multilabel_targets(…) → sbox_masked[:,i]` = masked S-box at slot `i` = `multGF256(α, Sbox[p_i ⊕ k_i]) ⊕ β`
  - Head accuracy measurement: `experiments/ablation_heads.py` (~line 76) for softmax collection; `ASCAD_test_models.py` for ranking/dissection

- **Environment:** conda env `py10`, RTX 5090. GPU budget is open; ask before runs over 2 hours.

## Plan

Gates are **pre-registered: they may not move after any number is seen.** Tags are I/U/S per `.grok/rules/03-priority-triage.md`.

- [x] (I:H U:H S:H) **W1 — diagnose why C is anti-informative (−0.09 nats).**
  - Hypothesis 1: C label class imbalance. Check the distribution of `sbox_masked` values across 50k profiling traces. Is there a dominant class? Does the baseline's 0.69% top-1 accuracy match the prior?
  - Hypothesis 2: C is genuinely unlearnable at this SNR. Run a toy test: train a 1-layer MLP on 5k traces to predict C from the raw traces (no masking, no leakage model), measure top-1 and nats. If it also scores below uniform, C may be noise.
  - Hypothesis 3: The loss function treats C the same as α/β/π (unified cross-entropy, §2). Per-head loss weighting or per-head LR might help. Measure the gradient magnitudes and loss contribution of each head during a baseline training checkpoint.
  - Report: `docs/tasks/T031-reports/W1-c-diagnosis.md`; data: `experiments/out/w4_c_diagnosis.json`.
  - Gate: identify one or more candidates for intervention.
  - **DONE 2026-10-05, gate PASS** (`experiments/w4_c_diagnosis.py`, 410 s, `experiments/out/w4_c_diagnosis.json`; no network trained, `.h5` read-only). **H1 refuted:** 256 classes all present, mode 0.45–0.49% vs 0.39% uniform, chi² p 0.16–0.62, prior-head ceiling only +0.003 nats and *positive*, so imbalance cannot give −0.090. **H2 supported (with ceiling caveat):** SNR excess over the (K−1)/N floor — α 1.8325 (2,872 sig samples), β 1.7288 (245), π 0.069–0.124 (220–802), **C 0.045–0.061 (21–25)**, unmasked `Sbox[p^k]` and `p^k` **0.0014–0.0021 (0)**; logistic probe on the same 50k traces reads α 40.32%, π 18.18%, C 0.52%/0.34% vs its permuted null 0.48%/0.50%. **H3 confirmed:** C = 52.6% of the averaged head loss (CE 5.635 nats > log 256) and 38.3% of trunk gradient power, α = 0.6% and 0.005%, π 59.7%. Vectorised SNR validated against `utils/snr.snr_vector` to 2.0e-14. Candidates: (1) rebalance C's loss/LR — **evidence says down-weight, the pre-registered W3 says up-weight**; (2) per-head LR (W2, unaffected by that objection); (3) a *mask-conditional* target, which is not among W4's three alternatives and would need registering.

- [x] (I:H U:H S:H) **W2 — replace the C-head with algebraic lookup-table (LUT) computation.**
  - **Rationale:** W1 measures C as the least visible target (SNR 0.045–0.061, only 21–25 significant samples) while consuming 52.6% of loss and 38.3% of trunk gradient. The C head is asking the network to learn a deterministic algebraic function (masked S-box computation in GF(2⁸)) that the network cannot learn from the marginal leakage. T-028 Phase 0 ablation showed uniform-C → 0/16 rank with flat posterior (A2), proving C carries signal but the head cannot extract it. Replace training with truth: compute C algebraically from α/β predictions using a pre-built LUT.
  - **Implementation:**
    1. Modify `ASCAD_test_models.py` dissection path to replace the C-head predictions with deterministic computation: for each (α, β) prediction pair and each candidate key, compute the true masked value via LUT (`LUT_affine[alpha_idx, beta_idx]` using the known formula `sbox_masked[slot_i] = (α · Sbox[p_i ⊕ k]) ⊕ β`).
    2. Build the pre-computed 256×256 LUT once at module init and register it as a buffer (no gradients).
    3. Train **a new baseline with C-head deleted** on 50k profiling traces, 15 epochs, all other hyperparams unchanged. This removes the 38.3% gradient sink and lets α/β/π training converge cleanly.
    4. Evaluate both (old baseline with trained C vs new baseline with LUT-C) on 5k attack traces using the same tie-safe rank-0 scoring.
  - **Gate:** the LUT model reproduces the baseline's 16/16 rank-0, or improves it. Sanity gate: baseline reproduces (16/16, z +9.50).
  - **DONE 2026-10-06 (steps 1, 2, 4 measured; step 3 deliberately not run). Gate FAIL for the intervention; sanity PASS.** `experiments/w5_c_lut.py` → `experiments/out/w5_c_lut.json` (268 s, numpy, no GPU, no training). Literal replacement is **flat by construction**: spread exactly 0.0, `flat_bytes: 16`, 0/16 rank 0, median z 0.0000; the generous no-V reading gives z **−0.3124**; a C-less decode (uniform C) is z 0.0000 and flat. Reason: V is the only quantity in the path that is both key-dependent *and* measured from the trace, and a LUT cannot supply a measurement. **Where the LUT did move the number: keeping the weak C head and feeding the LUT the TRUE masks → z +9.5047 → +12.9332, traces-to-rank-0 170 → 95** (gate-passing, but oracle ⇒ a bound); plugging in mask argmaxes instead of marginalising **loses** (z +5.1152, 1,065 traces). And the dissection **already is** the LUT marginalised (parity 4.16e-17; D1b/D1c/D2/D3 reproduce S2's V14/V6/V9/V7 to the last digit). Report: `T031-reports/W2-c-lut-replacement.md`.
  - **Report:** `docs/tasks/T031-reports/W2-c-lut-replacement.md`; checkpoint and eval data: `experiments/out/w5_*.npz` / `w5_*.json`.
  - **Why this supersedes the original W2–W4:** This is the direct test of the core hypothesis. No parameter tuning (W2/W3) without first testing whether the architecture itself is the problem. W4's alternatives are predicted to fail by W1's measurements (unmasked C and p^k have zero significant samples). LUT tests the irreducible minimum: can α/β/π alone, with perfect C, recover the key?

- [x] (I:H U:H S:H) **T31-N1 — joint (α,β)-consistency decoding + β-head calibration (CPU).**
  - **Rationale:** W2 measured the cost of β-posterior uncertainty: oracle β → z +12.93, true argmax β → z +5.11 (both on the same weak C head). β̂ is only 5.84% accurate (21–25 significant samples per W1); the current per-slot marginalisation ignores that one mask pair (α_slot, β_slot) serves all 16 slots per trace. A joint (α,β) decoder would treat this one-to-16 constraint and measure how much a β-consistency check and calibration unlock.
  - **Implementation (CPU-only, on cached `experiments/out/head_preds_5k.npz`):**
    1. Decode jointly across the 16 slots: for each trace, the single (α, β) pair implies a unique per-slot C value via the LUT. Marginalise the posterior as p(k | all 16 slot pairs) rather than per-slot.
    2. Measure: (a) posterior consistency — how much enforcing the constraint reduces β uncertainty; (b) β-head calibration — measure the confidence-error curve and interpolate; (c) what z and traces-to-rank-0 this yields.
  - **Gate:** median z ≥ +10.0 and traces-to-rank-0 ≤ 150, or a clear measurement of why the constraint does not help.
  - **Report:** `docs/tasks/T031-reports/T31-N1-joint-alpha-beta-decode.md`.
  - **DONE 2026-10-07. Gate FAIL; the gate's fallback branch met — three measured reasons why the constraint does not help. Sanity gate PASS** (`experiments/w7_joint_ab_decode.py` → `experiments/out/w7_joint_ab_decode.json`, 301.6 s; `experiments/w7_joint_ab_decode_tests.py` 33/33; supplementary `experiments/w7_dissect_normalisation.py` → `w7_dissect_normalisation.json`, 160.1 s. No GPU, no training, no checkpoint; `.h5` read-only). Reference decode reproduces the baseline at z **+9.504667781758279**, 16/16 non-flat, 170 traces, parity vs `multilabel_predict` **4.16e-17**.
    1. **The constraint is provably vacuous under marginalisation.** For a byte whose own key is unknown the other 15 slots enter only through `Σ_v P[a·S[p^v] ⊕ c]`, which is **exactly 1 for every a ≠ 0** (deviation 8.66e-15; tested over all 256 α). The correctly normalised joint decoder (E1) changes median z by **+4.2e-05** and changes **nothing else** — identical traces-to-rank-0 and final rank on all 16 bytes. The premise "one mask pair serves 16 slots, so cross-slot consistency sharpens β" is false *as a decoding statement*: marginalising a byte's key makes that byte's slot information-free about (α,β), because the hypothesis family tiles the whole 256-value space.
    2. **Plugging key estimates makes the constraint informative, and it backfires.** Pass-1 per-byte top-1 key accuracy is **0.50%** (chance 0.39%), so the EM evidence is fabricated; soft EM **+5.7201 / 460 traces**, hard EM **+5.4785 / 420**, both ≫ worse than +9.50 / 170. The mask is redrawn every trace, so nothing can be borrowed across traces either.
    3. **The C heads do not know the mask even given the key.** Slot-only argmax under the TRUE mask-free prior picks α **0.96%** / β **0.64%** (chance 0.39%); the oracle-key EM lifts β only from 5.84% to **11.34%**. This answers W1's open mask-conditional-visibility question in the negative.
  - **The three β quantities now separate**, and this is the decision-relevant result: β **correctness** is worth **+3.4 z** (W2 D1c, +12.9332 / 95 traces), β **confidence** is worth **~0.3 z**, and β **posterior shape** is worth almost nothing (tempering to τ = 20 still gives **+9.2032**). No decoder, calibration or consistency scheme converts the second or third into the first — only the head does, which is a training-side decision.
  - **Calibration measured:** β ECE **0.0395** with NLL-optimal **τ\* = 1** on both halves (so β is *not* miscalibrated in the NLL sense); α ECE 0.0549 with τ\* 0.05 (confident and right). Selecting τ on the *attack* metric instead picks β τ = 3 in both directions and gains **+0.232 / +0.371 z** on the held halves while traces-to-rank-0 goes 205→200 on one half and 170→175 on the other: inconsistent, sub-gate, not claimed as a result.
  - **Two audit findings.** (a) `proba_dissect_permind`'s per-byte mixture mass `R = Σ_s PI[t,s,b]` is **not** normalised per trace (median 1.015, range **0.008–3.587**, 68,491/80,000 cells outside ±5%), and `proba_dissect_alpha` leaves its α = 0 branch unnormalised — both enter as a constant over the 256 hypotheses, so they are **provably and measurably inert for ranking** (E5 normalises R and returns byte-for-byte identical traces). The verified identity is `Σ_j Pr(j) = 256·A[0]·Q[0] + (1−A[0])·R` (max dev 8.91e-07); my first closed form omitted R and was caught by its own control (disagreement up to 2.587 was impossible for the α term, bounded 0.65). (b) **ASCADv2's five α=0 profiling rows are record-boundary artifacts**, not mask draws: rows 99999/199999/299999/399999/499999, each with all 38 mask columns zero; the attack split has none. So α ∈ 1–255 as `docs/notes/datasets.md:170` states, and the α=0 branch is unreachable from labels.
  - **Consequence for the plan:** the mask-side *decoding* programme closes negative. T31-N2 survives only as a question about **β head accuracy** (not about C), and W4 (π) stays untriggered.

- [x] (I:M U:H S:M) **T31-N2 — C-deletion retrain, reframed as an interference measurement (GPU, conditional).**
  - **Rationale:** W2's D2 (C-less decode on the existing trunk) is flat, but that does not mean retraining without C heads will fail. The original interference question (do α/β/π heads improve when C's 38.3% gradient sink is removed?) requires GPU time. Gate cannot be "recovers rank 0" (as D2 shows, that is unreachable without a measured quantity), but instead "head accuracies improve relative to baseline" (α ≥ 100.00%, β ≥ 10%?, π ≥ 70%?).
  - **Implementation (if T31-N1 completes and shows no other path forward):**
    1. Time the run with a 1-epoch probe (5–10 min on 50k / RTX 5090).
    2. Train a new MultiResNet (same 34-head layout, no C heads) on 50k profiling, 15 epochs, other hyperparams unchanged.
    3. Measure per-head accuracies vs the baseline. Measure likelihoods and GE-vs-traces on 5k attack (uniform C in the decoder).
    4. A new gate: do α/β/π improve?
  - **Report:** `docs/tasks/T031-reports/T31-N2-c-deletion-retrain.md`; checkpoint: `experiments/out/w6_multiresnet_nochead_*.pth`.
  - **DONE 2026-10-07, run on the user's instruction "Run T31-N2 as planned". PASS on the primary criterion for β and π; α unchanged; the bracketed provisional numbers met 1/3.** Probe first (plan step 1): **56 s/epoch**, so 15 epochs ≈ 15 min — the 2-hour rule never engaged. Implementation: `--no-c-heads` added to `MultiResNet/train.py`, removing the 16 `sbox_*` terms from the **loss** only — same architecture (137,396,064 params), same optimiser/batch/lr/clip/sub_avg/train_len/epochs/best-val rule as the baseline's own sidecar. **A matched 34-head control was also run, which was not in the plan but is not optional**: the trainer sets no RNG seed, and two 34-head runs differ by **1.44 pp** on β top-1.
    - **Head quality (5k attack traces, identical metric code for all three):** β top-1 **9.30%** vs baseline 5.84% / control 4.40%; β gain **+1.5112 → +2.0002** nats; β mean p(true) ×2.03; π element **72.13%** vs 67.03% / 68.06%; exact permutation **1.30%** vs 0.60% / 0.58%; α unchanged (1.0000 / 0.9994 / 0.9996). Paired per-epoch validation: N2 above control in **13/15** epochs for β (mean +0.0218) and **14/15** for π (mean +0.0348).
    - **Its own attack is dead, as W2 predicted:** as-trained z **+0.4741**, 0/16 rank-0, 1,390 traces, GE 85.0. The only cross-model key-recovery readout is the oracle-C decode, and it is **saturated** (ceiling √255 = 15.97; all three models reach rank-0 at the grid floor of 10 traces; N2 +15.8520 vs +15.7846 / +15.7792).
    - **Mechanism measured, not assumed.** The alternative explanation was that dropping C loosens the global clip (clip is at 1.0, so 18 heads instead of 34 means a 1.78× bigger post-clip step). Pre-clip ‖g‖: **122.9 (34 heads) → 69.4 (18)**, clip factor **0.0081 → 0.0144** — the clip binds hard and always. But a clip is a *uniform* rescale and RAdam is near-invariant to one: at the measured factor the net displacement ratio is **0.959** with direction cosine **0.99960** (at 10×: 0.917 / 0.9981). 4–9% of step size cannot explain a ×2.11 relative β change ⇒ **the gain is gradient direction, i.e. the sink.**
    - **Controls:** the evaluator reproduced every baseline number to the digit before use (α +5.4827, C −0.0900, z +9.5047/170, oracle-C +15.7846/10 = W2's D0/D3); GE vs the tie-safe rank agrees to **0.000e+00** on every non-flat decoder; **8/8 patch guard tests** — the load-bearing one shows that iterating the filtered key list positionally (the obvious implementation) would have read **16 of 18** surviving heads from the wrong label column and silently trained on scrambled targets. Two traps recorded: val_loss is a *sum* over heads so N2's 11.68 and the control's 102.92 are different currencies (and the control reproducing ~103 is the proof the patch kept the baseline's convention); and GE on a flat posterior ranks float summation order (249.11–249.81, disagreeing with the rank by 31.9–67.5).
    - **Consequence — the first positive result in T-031, and it prices the tension exactly:** C is useless as a classifier yet indispensable as the carrier, and removing it helps every other head. So the untested middle is **C present but down-weighted**: β is 4.4–5.8% at weight 1 and 9.30% at weight 0. Registered as **T31-N4** (weights 0.25/0.5, ~2×14 min GPU, gate proposed in the report §6 before any number is seen), pending the user's go-ahead.

- [x] (I:H U:H S:M) **T31-N4 — C loss-weight sweep: the one untested middle (GPU; ran 2026-10-07).**
  - **Rationale:** N2 bracketed the trade on both axes and nothing has measured between them. β top-1 is **4.4–5.8% with C at full weight** (baseline 5.84%, matched control 4.40%) and **9.30% with C at zero weight**, while the attack works only in the former and exists at all only in the former (z +9.5047/170 vs +0.4741/0-16). If C's harm is a gradient-direction effect, some intermediate weight should keep the carrier and recover part of the mask-head quality; if it does not exist, that is a result in its own right — the two objectives are not tradeable and C's contribution to the trunk is simply bad.
  - **Implementation (only on the user's go-ahead):**
    1. Add a C loss-weight flag to `MultiResNet/train.py` (additive, default 1.0 = today's behaviour). Scale **only** the 16 `sbox_*` CE terms; leave α/β/π at weight 1, and keep the name→column target map so nothing can shift.
    2. Reuse the existing null machinery, no new eval code: `experiments/w6_nochead_eval.py` (already validated against every baseline number), the control checkpoint, and `<ckpt>.history.json` per-epoch curves.
    3. Train weights **w = 0.25 and 0.5**, 50k traces, 15 epochs, all other hyperparams as in N2/control. Probe first (already known: ~56–65 s/epoch, so ~28 min for both).
    4. Evaluate both on the 5k attack split; report per-head top-1/gain, as-trained z + traces-to-rank-0 + GE, and the paired per-epoch val curves against the control.
  - **Gate, frozen here before any number exists:** a weight **passes** if it holds ≥ 16/16 non-flat rank-0 on the as-trained decode **and** lifts β top-1 ≥ **7.5%** — at least half the way from 5.84% to the no-C 9.30%, and beyond the measured two-seed spread of 1.44 pp. Pass at either weight ⇒ the trade is real and worth a follow-up sweep (and a re-look at the baseline recipe). **No weight passing ⇒ report the negative and stop the training line**; do not add weights post hoc.
  - **Secondary readouts, not gates:** π element/exact, α (expected flat, it saturates), C's own top-1/gain, traces-to-rank-0 monotonicity in w.
  - **Report:** `docs/tasks/T031-reports/T31-N4-c-loss-weight-sweep.md`; checkpoints: `experiments/out/w9_multiresnet_cw025*.pth`, `w9_multiresnet_cw05*.pth`.
  - **DONE 2026-10-07. PASS at w=0.25.** The middle exists: at **w=0.25** β top-1 **5.84% → 7.74%** (gain +1.5112 → **+1.8316**, mean p(true) 0.0340 → **0.0525**), π **67.03% → 71.53%** with exact-permutation recovery **0.60% → 1.22%**, α unchanged at 1.0, and the as-trained decode still gives **16/16 non-flat rank-0**. 
  - **Three different "best" models depending on metric**, revealed after the gate but not covered by it: 
    - Head quality (β): **w=0.25** (7.74%, gate passes)
    - Attack score (z): **w=0.5** (+9.7556, best of all five, but fails β gate at 5.74%)
    - Speed (traces-to-rank-0): baseline w=1.0 (170 vs cw025's 335; per-byte ranges 50–720 vs 20–850 overlap)
  - **Single-seed results sit inside the noise band:** Two-seed spread is 1.44 pp on β. The 1.1 z gap between cw05 (+9.7556) and cw025 (+8.6892) is not separable from run-to-run variance with one seed each. Validation is monotone in w (0.0593 → 0.0724 → 0.0770 → 0.1002; cw025 and π both above control in 13–15/15 epochs); attack split is not monotone.
  - **The carrier cliff is between w=0.25 and w=0** (where attack dies at z +0.4741), not inside [0.25, 1]. **C's presence carries the key; its weight modulates competition with α/β/π.**
  - **Implementation verified:** `--c-loss-weight` extracted as a testable free function (20/20 guard checks including exact parity to w=1 and w=0 objectives). Probe caught a dropped `.to(device)` before GPU run. Verdict from `experiments/w9_gate_check.py` → `w9_n4_gate.json`.
  - **By the plan's own rule:** gate passes ⇒ cascade stops. **N5 and N6 are NOT triggered.** But the incomplete metric coverage exposes a real trade: adopt w=0.25 now (clean gate pass, best β), or run **T31-N4b** (seed replication on 0.25 and 0.5, ~30 min, to break the tie on which weight actually improves attack). Report: `T031-reports/T31-N4-c-loss-weight-sweep.md`. **N4b RAN and overturned this row: the w=0.25 PASS did not replicate (z 5.37 / 9.52, β 7.22% at seed 1 — under the bar) and w=0.5 won (z 10.26 / 9.71, 145 / 150 traces). See the N4b block below.**
  - **Design alternative if GPU time is to be saved:** replace C's CE target with a *uniform* target instead of reweighting — it keeps the head count and the clipping regime fixed, isolating the sink from the loss-scale change. Cheaper to reason about, but it is a different intervention and would need its own gate.

- [x] (I:H U:H S:M) **T31-N4b — C loss-weight replication (seed variance resolution, conditional on N4).**
  - **Rationale:** N4 found one weight (0.25) passes its gate, but the attack split is non-monotone and single-seed runs sit inside the 1.44 pp two-seed spread. w=0.5 fails the β gate yet has the best z (+9.7556 vs baseline +9.5047). To decide between 0.25 (head quality) and 0.5 (attack score), run with `--seed` for stable variance estimates.
  - **Status: DONE 2026-10-07→08 — the rule fired on its FIRST branch: adopt w=0.5.** (Go-ahead GIVEN 2026-10-07: "run T31-N4b (seed replication, ~30 min GPU) to break the tie".) Cost corrected before launching: the rule needs **4** runs (`--seed {1,2}` × 2 weights) at ~16 min each ≈ **64 min**, not the ~30 min the request assumed — order put both seed-1 arms first so the n=2-per-weight answer existed at the halfway mark; actual training wall **3296.7 s (55.0 min)** plus four 5k-trace evaluations. `--seed` added to `MultiResNet/train.py` (seeds python/numpy/torch before init and loader order; recorded in sidecar + history). **Same-seed reproducibility itself was NOT measured** — the user declined the two extra probe runs — so the claim is only "same init and data order", with AMP/cuDNN atomics still free to add jitter. Launcher `logs/t031_n4b_train.sh`, log `experiments/out/w10_run.log`.
  - **RESULT (inverts N4):** w=0.5 z **+10.2582 / +9.7092** (mean 9.984, spread 0.549), median traces→rank-0 **145 / 150** (spread 5); w=0.25 z **+5.3737 / +9.5211** (mean 7.447, spread **4.147**), traces **955 / 180** (spread 775). mean z(0.5) − mean z(0.25) = **+2.5363** vs the 0.5 threshold ⇒ **use w=0.5**. **The gate-passing arm was the unstable one.** The w=0.25 head-quality claim also falls: β 7.22% at seed 1 is below the 7.5% bar (8.02% at seed 2) — N4's single-run PASS was a draw. 16/16 non-flat rank-0 in **all six** C-including runs; N2 (w=0) still 0/16, so the cliff is unchanged. Validation β cannot separate the weights (0.0802 vs 0.0742, within-weight seed spread ≥ between-weight gap) but both clear w=1 (0.0593).
  - **Pre-registered decision rule (chosen before seeing numbers):**
    - If w=0.5 replicate > w=0.25 replicate by >0.5 z (mean of two seeds): **use w=0.5** (attack wins). ← **THIS BRANCH FIRED**
    - If w=0.25 replicate > w=0.5 by >0.5 z: **use w=0.25** (gate-passing model, head quality wins).
    - If they bracket (overlap): **report both and defer to user** (no clear winner).
  - **Implementation (GPU, ~30 min → 4 runs, 55 min actual):**
    1. Train both w=0.25 and w=0.5 on 50k profiling, 15 epochs, with `--seed {1, 2}` (two seeds each).
    2. Evaluate both seeds for each weight on the same 5k attack split.
    3. Report per-weight mean/std of: β top-1, z, traces-to-rank-0, π exact.
  - **Report:** `docs/tasks/T031-reports/T31-N4b-c-loss-weight-replication.md`; checkpoints: `experiments/out/w10_multiresnet_cweight_{0p25,0p5}_seed{1,2}_*.pth`.
  - **If N4b clarifies the choice:** Adopt the winning weight as the new baseline. **Then stop — N5 and N6 remain not triggered.** ← **DONE: w=0.5 adopted by the frozen rule; stopped; N5/N6 untouched.**
  - **If N4b remains ambiguous:** Report both and proceed to user judgment call. *(Not this branch, but see report §6: "w=0.5 beats the baseline" is **suggested, not established** — no seeded w=1 replicate exists. One run, ~16 min, and it needs the user's go-ahead.)*

- [N/A] (I:H U:— S:H) **T31-CL — Contrastive Learning investigation (FALSIFIED BY CL-0 CPU PREMISE CHECK).**
  - **Registered 2026-10-08 after C4 oracle swaps.** **Status: CL-0 premise check (CPU only, ~1 min, 2026-10-08) falsified the plan; CL-α and CL-C not triggered; CL-Bridge remains as optional diagnostic.**
  - **Original hypothesis:** C4 showed true C lifts TimesNet from z −0.1066/0/16 to z +14.2108/16/16, suggesting C learning is the gap. Contrastive learning could exploit C's relational structure.
  - **CL-0 findings (CPU check before any GPU):**
    1. **CL-α premise FALSE:** The dataset has one key per trace (50,000 distinct keys / 50,000 traces). α is redrawn per trace. P(αᵢ=αⱼ | same key byte) = 0.0039256 vs exact floor 0.0039439 (−1.2σ on 16M pairs). **No positive equivalence class exists for contrastive clustering.** Random pairs give the same signal.
    2. **CL-C premise FALSE:** The relation C = α ⊗ S(p⊕k) ⊕ β holds in byte space (P(same unmasked S) = 1.0) but the head's masked label is uncorrelated with it (0.00327 vs floor 0.00391). Additionally, slots ≠ bytes across traces: 17,255 distinct permutations in 20k traces; byte 0 in slot 0 only 6.22% (chance 6.25%). **The relation CL-C depends on doesn't exist in the label the model sees.**
    3. **CL-0's deeper finding:** Neither MultiResNet nor TimesNet *learned* C. C-head top-1 0.69% / −0.0900 nats (MultiResNet) and 0.40% / −0.0729 nats (TimesNet) are both at chance. **C4's z difference (−0.1066 → +14.21 with oracle C) is carriage (routing), not learning.** Different question entirely from what the plan posed.
  - **Report:** `docs/tasks/T031-reports/CL-0-premise-check.md` (measurements, MI(α;k₀) trap avoided, permutation nulls)
  - **Surviving option:** **CL-Bridge (optional diagnostic only)** — test whether trunk representation alignment can make TimesNet learn to *use* C in the decode, even if C stays at chance. Gate (corrected): oracle α+β+π (C from model) z ≥ +3.0, ≥8/16 rank-0. Cost: ~45 min GPU. **Only if you want a representation-alignment diagnostic; not required.**
  - **Consequence:** Label-free C discovery (non-worst-case) is not achievable on ASCADv2 because **C is not learnable as a classifier on either model** — it's slot-indexed masked S-box, a function of the masks, not an observation. The non-worst-case bottleneck remains α (label-free band finding fails at 2% matched accuracy; necessary condition ≳100k traces per R10).

- [ ] (I:M U:M S:H) **T31-N4c — Seeded C-loss weight 1.0 baseline replicate (validation of N4b adoption).**
  - **Rationale:** N4b adopted w=0.5 (two-seed mean z +9.984) based on beating the baseline (z +9.5047), but the baseline is one unseeded run. A seeded w=1 run with the same `--seed` protocol as N4b validates whether w=0.5's advantage is stable or within the w=1 noise band. Cost: GPU ~16 min.
  - **Status: PENDING USER GO-AHEAD** (registered 2026-10-08 after N4b delivered).
  - **Implementation (if approved):**
    1. Train one replicate at w=1.0, 50k profiling, 15 epochs, with `--seed 1` (same seed protocol as N4b to match conditions, not for reproducibility check).
    2. Evaluate on the same 5k attack split.
    3. Compare one-seed z to N4b's baseline (z +9.5047) and to N4b's two-seed w=0.5 mean (z +9.984).
  - **Report:** `docs/tasks/T031-reports/T31-N4c-baseline-replication.md`; checkpoint: `experiments/out/w11_multiresnet_cweight1p0_seed1_*.pth`.
  - **Decision:** If w=1 (seeded) sits within N4b's w=0.5 spread (0.549), the advantage is suggested but not separated from seed noise. If w=1 stays below w=0.5 two-seed mean by >0.5 nats, w=0.5 adoption is validated and T-031 closes.

- [N/A] (I:H U:— S:M) **T31-N5 — Gradient projection (SOTA #2: conflict resolution, NOT TRIGGERED).**
  - **STATUS (2026-10-07): NOT TRIGGERED.** N4 passed its gate at w=0.25. By the plan's cascade rule (gate pass ⇒ use that weight; gate fail ⇒ proceed to next step), N5 is not started. **Reaffirmed 2026-10-08 by T31-N4b:** replication moved *which* weight wins (0.5, not 0.25) but not the cascade state — the first SOTA step produced an operating point, so N5 is still not triggered. **FLAGGED: the implementation code block has errors and should not be run as written.** It filters parameters that don't exist in MultiResNet (`alpha_heads`, `beta_heads`, `pi_heads`, `c_heads` do not exist; the real structure is `alpha`, `beta`, `sbox.i.permind.i`), projects head gradients instead of the shared trunk (where N2 located the interference), and doesn't implement PCGrad correctly. If N5 is ever needed, the code must be rewritten. Do not start this step without the user explicitly reopening it.
  - **Original rationale (archived):** If N4 finds no weight that balances the objectives, the gradients from C and α/β/π are fundamentally in conflict (not just different magnitudes). Gradient projection (PCGrad, Gradient Surgery) would eliminate this conflict by making C's gradient orthogonal to α/β/π's direction, removing interference while preserving C's signal.
  - **Pre-registered gate:** β top-1 ≥ 8.0% (between fixed-weight 0.5 and no-C 9.30%) AND attack ≥ 16/16 rank-0 AND z ≥ +9.0. If gradients are truly orthogonal, both head quality and key recovery should improve together.
  - **Implementation (GPU, ~25 min):**
    1. Add a `--gradient-projection` flag to `MultiResNet/train.py`. After `loss.backward()` but before `optimizer.step()`, implement the projection:
       ```python
       # Collect gradients from α/β/π heads and C heads separately
       grads_ab_pi = [p.grad.clone() for name, p in model.named_parameters() if 'alpha_heads' in name or 'beta_heads' in name or 'pi_heads' in name]
       grad_c = [p.grad.clone() for name, p in model.named_parameters() if 'c_heads' in name]
       
       # Compute dot product and norm of α/β/π gradients
       dot_product = sum(torch.dot(g_c.flatten(), g_ab.flatten()) for g_c in grad_c for g_ab in grads_ab_pi)
       norm_sq = sum(torch.dot(g.flatten(), g.flatten()) for g in grads_ab_pi)
       
       # Project C's gradient away if it points the same direction (conflict)
       if dot_product > 0 and norm_sq > 0:
           scale = dot_product / (norm_sq + 1e-8)
           for p in model.parameters():
               if any('c_heads' in name for name, _ in model.named_parameters() if _ is p):
                   if p.grad is not None:
                       # Subtract the projection of C's gradient onto α/β/π's direction
                       p.grad -= scale * sum(grads_ab_pi) / len(grads_ab_pi)
       ```
    2. Train with `--gradient-projection` on 50k profiling, 15 epochs, all other hyperparams as N2/control.
    3. Measure per-head accuracies (β/π/α), as-trained z and traces-to-rank-0 vs baseline.
    4. Evaluate on 5k attack; compare per-epoch val curves against control (same infrastructure as N4).
  - **Report:** `docs/tasks/T031-reports/T31-N5-gradient-projection.md`; checkpoint: `experiments/out/w10_multiresnet_gradproj_*.pth`.
  - **Why this SOTA approach:** Gradient projection is the standard solution in multi-task learning when fixed weighting fails. It removes the direction of conflict while leaving magnitude and other task-specific information intact. Cost: minimal (one backward pass hook), no architectural change.
  - **If N5 succeeds:** Deploy gradient projection as the production trainer. The problem is solved.
  - **If N5 fails:** Proceed to N6 (two-trunk modular).

- [N/A] (I:M U:— S:L) **T31-N6 — Two-trunk modular architecture (SOTA #3: structural separation, NOT TRIGGERED).**
  - **STATUS (2026-10-07): NOT TRIGGERED.** N4 passed its gate at w=0.25. By the plan's cascade rule (gate pass ⇒ use that weight; gate fail ⇒ proceed), N6 is not started. (N5's cascade rule also applies: even if N5 were triggered, N6 would not fire until N5 fails.)
  - **Original rationale (archived):** If gradient projection failed, α/β/π and C might need separate representation capacity. A modular architecture with independent trunks would eliminate interference structurally, allowing each task to optimize its own feature hierarchy.
  - **Pre-registered gate:** β top-1 ≥ 9.0% (close to or matching no-C) AND attack ≥ 16/16 rank-0 AND z ≥ +9.3. If modular works, it should combine the head quality of no-C with the key-recovery of the baseline.
  - **Implementation (GPU, ~45 min architecture + 30 min training):**
    1. Create `MultiResNetModular(nn.Module)` in `MultiResNet/__init__.py`:
       ```python
       class MultiResNetModular(nn.Module):
           def __init__(self, input_len, n_classes=16):
               super().__init__()
               # Trunk 1: α/β/π (main, focused on mask recovery)
               self.trunk_ab_pi = ResNetTrunk(input_len, hidden=128, depth=6, name='ab_pi')
               self.alpha_heads = nn.ModuleList([LinearHead(128, 256) for _ in range(16)])
               self.beta_heads = nn.ModuleList([LinearHead(128, 256) for _ in range(16)])
               self.pi_heads = nn.ModuleList([LinearHead(128, 16) for _ in range(16)])
               
               # Trunk 2 or side-branch: C (isolated, focused on key signal)
               self.trunk_c = ResNetTrunk(input_len, hidden=64, depth=4, name='c')
               self.c_heads = nn.ModuleList([LinearHead(64, 256) for _ in range(16)])
               
           def forward(self, x):
               # Independent feature extraction
               feat_ab_pi = self.trunk_ab_pi(x)
               feat_c = self.trunk_c(x)
               
               # Independent head outputs
               alpha = torch.stack([h(feat_ab_pi) for h in self.alpha_heads], dim=1)
               beta = torch.stack([h(feat_ab_pi) for h in self.beta_heads], dim=1)
               pi = torch.stack([h(feat_ab_pi) for h in self.pi_heads], dim=1)
               c = torch.stack([h(feat_c) for h in self.c_heads], dim=1)
               
               return alpha, beta, c, pi
       ```
    2. Train on 50k profiling, 15 epochs, same hyperparams (RAdam lr 1e-4, batch 64, AMP, sub_avg, best-val checkpointing).
    3. Parameter count: trunk_ab_pi ~100M + trunk_c ~37M + heads ~43M = **~180M** (vs baseline 137M).
    4. Measure per-head accuracies and attack outcome on 5k attack.
    5. Per-epoch validation curves vs control (modular with standard trainer, no gradient projection).
  - **Report:** `docs/tasks/T031-reports/T31-N6-two-trunk-modular.md`; checkpoint: `experiments/out/w11_multiresnet_modular_*.pth`.
  - **Cost/benefit:** +43M params, +20 min training, but if it works it is the recommendation for any worst-case multi-task setup with conflicting gradient directions. This is SOTA practice in modern vision and RL (e.g., Transformers with separate head trunks).
  - **If N6 succeeds:** Close T-031 with modular as the solution. Recommend updating the baseline architecture.
  - **If N6 fails:** See decision section below — the problem may be deeper than training.

- [ ] (I:L U:L S:M) **W4 — conditional: recover π via post-hoc search (only if future results show π as a bottleneck).**
  - **Rationale:** T-028 Phase 0 showed A4 (removing π, uniform π predictions) still recovers 12/16 at z +3.31. π's information is marginalizable, though the correct permutation routing buys z gain (+9.50 → +9.67, S2). If W2 (with perfect algebraic C and trained α/β/π) recovers 16/16, π is no issue. If it leaves 2–4 bytes at median rank > 0, π might be the bottleneck — and unlike C, π is *deterministic*, not a leakage model.
  - **Implementation (only if W2 leaves bytes unsolved):**
    1. Take the W2 model's α/β/C predictions and find the best π *post-hoc* per-evaluation-set, rather than per-sample training:
       - (a) **Brute-force on a pruned space:** If π has known structure (e.g. rotation, affine), enumerate the likely permutations and score each against the data.
       - (b) **Partial key search:** For each candidate key guess, compute which permutation π best explains the predicted α/β/C scores, then score that key.
       - (c) **Adversarial π recovery:** Formulate as "given α/β/C, find π that maximizes key score" and solve off-trace.
    2. Re-evaluate the W2 checkpoint with π recovered via the chosen method.
    3. Measure: does it recover the remaining bytes?
  - **Gate:** the bytes that fail W2 now pass (improve from rank > 0 to rank 0), or all three variants fail (π recovery does not unlock them).
  - **Report:** `docs/tasks/T031-reports/W4-pi-recovery.md`; evaluation data: `experiments/out/w5_pi_*.json`.
  - **Why W4 is conditional:** W2's outcome will show whether π is actually the blocker. This is a hypothesis test with a clear yes/no after W2 completes.

## Decision Points for the User (What Triggers Each SOTA Step)

**N4b Result 2026-10-08: the frozen rule fired on its first branch — adopt w=0.5 — and N4's ranking did not replicate.**

Four seeded runs (2 weights × seeds {1,2}), judged by `experiments/w10_replication_check.py` against the rule frozen in the row below before any number existed:

| | mean z (spread) | median traces→rank-0 (spread) | β top-1 both seeds | 16/16 non-flat |
|---|---|---|---|---|
| w=0.25 | 7.447 (**4.147**) | 567.5 (**775**) | 7.22% / 8.02% — **bar missed at seed 1** | ✓ both |
| w=0.5 | **9.984 (0.549)** | **147.5 (5)** | 7.02% / 6.04% | ✓ both |

mean z(0.5) − mean z(0.25) = **+2.5363** > 0.5 ⇒ **use w=0.5 (attack wins)**. The gate-passing N4 arm is the one that cannot be relied on; N4's head-quality claim also falls (β ≥ 7.5% on both seeds was never met). By the plan's instruction the cascade **stops here**: **N5 and N6 stay untriggered, and nothing further has been started.**

**The one open question N4b could not close** (report §6): w=0.5 beats the baseline's attack numbers on both metrics (9.984 vs +9.5047 z; 147.5 vs 170 traces), but the baseline is a single **unseeded** run, so "better than the current baseline" is suggested, not established. A **seeded w=1 replicate (one run ~16 min GPU; two for a variance)** is the missing measurement and needs the user's go-ahead. Adopting w=0.5 as specced requires no further compute. Recorded as a programme-level lesson: **single-run gates on this task are unreliable — `--seed` should be mandatory in every comparison from here on.**

---

**N4 Result 2026-10-07: w=0.25 passes the gate. By the plan's rule, cascade stops.** *(Superseded on 2026-10-08 by N4b: this table is single-run evidence, and the row that looked like a win did not replicate.)*

Gate criterion met at w=0.25 (16/16 non-flat rank-0, β **7.74%** ≥ 7.5%), failed at w=0.5 (β **5.74%**). N5 and N6 are **not triggered**. But three different "best" models exist depending on metric:

| Model | β | z | Traces | Status |
|---|---|---|---|---|
| w=0.25 | **7.74%** | +8.6892 | 335 | **Gate passes**: head quality wins |
| w=0.5 | 5.74% | **+9.7556** | 200 | **Gate fails, best attack**: attack wins |
| w=1.0 (baseline) | 5.84% | +9.5047 | 170 | **Fastest**, reference |

**Single-seed variance issue:** Each weight is one run; two w=1 references differ by 1.44 pp on β. The 1.1 z gap (9.7556 vs 8.6892) sits inside the noise band.

**Recommended next step: T31-N4b (seed replication) — RUN 2026-10-07→08, ANSWER ABOVE.** Train both 0.25 and 0.5 with `--seed {1, 2}` (corrected before launch: 4 runs ≈ **55 min**, not ~30; actual 3296.7 s + 4 evals) to resolve which weight's attack advantage is stable. Decision rule (frozen before seeing numbers):
- If w=0.5 mean z > w=0.25 by >0.5 nats: **use w=0.5** (attack wins).
- If w=0.25 mean z > w=0.5 by >0.5 nats: **use w=0.25** (gate-passing, head quality).
- If they overlap: **report both**, defer to user.

**After N5 (gradient projection) completes (only if N4 failed):**
- **If gradient projection passes the gate** (β ≥ 8.0% AND ≥16/16 rank-0 AND z ≥ +9.0): Deploy gradient projection in the trainer. Task is complete. This is a minimal architectural change that solves the conflict.
- **If gradient projection fails:** Proceed to N6.

**After N6 (two-trunk modular) completes (only if N5 failed):**
- **If modular passes the gate** (β ≥ 9.0% AND ≥16/16 rank-0 AND z ≥ +9.3): Adopt the modular architecture as the new baseline. Task is complete. This is SOTA for multi-task SCA with conflicting gradients.
- **If modular fails:** T-031 is diagnostically complete. C is a pathological carrier that cannot be resolved by training-side methods. Report this as a finding and stop. The next step (if any) would be architectural research: a fundamentally different C target, or a redesigned C head as a carrier-specific head rather than a classifier.

## Hard rules for the implementing agent

1. Read `docs/tasks/INDEX.md` and this file first; use `py10`.
2. Do not edit T-028/T-029/T-030 files. Do not delete or overwrite existing checkpoints or `.h5`; write new outputs under new names (e.g., `w4_*.pth`, `w4_*.npz`).
3. Verify every number from a source file before writing it in a report; state what was not verified.
4. After each step: write its report, append a line to `docs/tasks/T031-reports/LEDGER.md`, update this file's Working notes, then **stop and wait**. Do not start the next gated step.

## Decisions

- 2026-10-05 (user): register as a new task, T-031, to investigate C-head learning; implemented by another agent; reports to `docs/tasks/T031-reports/`.
- 2026-10-05: Active focus stays with T-028; T-031 is listed as also live (or user moves it). (Default; the user can move it.)
- 2026-10-05: **W1 delivered (PASS).** It refutes the imbalance premise this task was registered on, measures C as the least visible target in the dataset (while α, which S3 proved learnable, is the most), and shows the loss spends half its mass and 38% of the trunk gradient there. Consequence for the plan: **W4's alternatives are now predicted to fail by measurement** (unmasked C and p^k have zero significant samples), and **W3's specced direction contradicts the evidence**. W2 stands unchanged. No GPU run was started, and none will be until the user rules.
- **W1 go-ahead: GIVEN.** Diagnostic work (no GPU commitment yet).
- **2026-10-06: W2 delivered (gate FAIL for the intervention as originally specced, but the measurement discovered the real lever).** A lookup table is a function, not a measurement: literal LUT replacement yields a flat posterior (z 0.0000, spread 0.0, all 16 bytes flat). Deleting the C head entirely also yields flat posteriors on the cached trunk (z 0.0000, D2). But the measurement found the binding constraint: keeping the weak C head and feeding the LUT the **true masks** (oracle) gains z **+12.93** vs baseline +9.50, cuts traces-to-rank-0 from 170 to **95** — a **3.4 nats gain and 44% fewer traces**. Conversely, plugging in mask argmaxes instead of marginalising **loses** (z **+5.11**, 1,065 traces), demonstrating that β-posterior uncertainty (5.84% accuracy, 21–25 significant samples per W1) is the cost. Three consequences: (1) **W4 (π post-hoc recovery) not triggered** — W2's cases either recover all 16 or are structurally flat; (2) **W3 (C-head retraining) superseded** — the C-head deletion is only worth doing as an *interference* measurement with a different gate (do head accuracies improve?), requiring GPU time and a new go-ahead; (3) **the next concrete step is T31-N1: joint (α,β)-consistency decoding across the 16 slots plus β calibration** — CPU-only on cached predictions, high leverage (the 3.4-nats swing).
- **User decision requested on three candidates:** (N1) joint (α,β)-consistency decoder + β calibration (CPU); (N2) C-deletion retrain reframed as an interference experiment (GPU, requires timed probe and new go-ahead per rule 04); (W4 conditional) stay unstarted unless future results show π as a bottleneck.
- **2026-10-06 (user): N1 go-ahead GIVEN by writing it into the plan** (I:H U:H S:H, CPU-only) — the user re-specced T-031's next step as joint (α,β)-consistency decoding + β-head calibration, superseding W3/W4 of the original spec. N2 kept as conditional GPU with a non-rank-0 gate.
- **2026-10-07: T31-N1 delivered (gate FAIL, fallback branch met).** Consequences recorded in the plan: (1) the mask-side *decoding* programme is closed negative — three independent reasons, one of them a proved identity; (2) the +3.4 z that W2 located is a **head-accuracy** prize, not a decoder prize (β correctness +3.4 z, β confidence ~0.3 z, β shape ~0), so any further GPU spend should be aimed at β's accuracy rather than at C's gradient sink — T31-N2 as currently worded is therefore partly mis-aimed and should be re-scoped or judged as a weaker test; (3) two apparent bugs in `multilabel_predict` (unnormalised per-byte mixture mass, unnormalised α=0 branch) are provably inert for ranking and must not be treated as levers; (4) `experiments/w2_decoders.py` has triplicate dead definitions from S2 — correct code binds last, so S2's numbers stand; left untouched under hard rule 2, pending the user's cleanup ruling.
- **2026-10-07 (user): T31-N2 go-ahead GIVEN explicitly** — "Run T31-N2 as planned". The probe (plan step 1) came back at **56 s/epoch**, so the 15-epoch run is ~15 min and the 2-hour rule never engaged. **Two deviations from the letter of the plan, both to keep the result honest:** (a) a **matched 34-head control run** was added, because `MultiResNet/train.py` sets no RNG seed and "improve relative to baseline" has no null otherwise — the two 34-head runs differ by 1.44 pp on β, which is the scale every claim is now stated against; (b) `--save-last` and per-head validation metrics were added to the trainer, since best-val checkpointing alone hides which epoch produced the number. Verdict: **PASS on the primary criterion for β and π, α unchanged; the bracketed provisional numbers met 1/3** (π 72.13% ≥ 70%; β 9.30% just under its provisional 10%; α 99.96% vs ≥100%). The clip-budget explanation was measured and rejected, so the finding stands as a multi-task interference result. **Next candidate registered as T31-N4** (C loss-weight 0.25/0.5, ~28 min GPU, gate pre-registered in the report before any number is seen) — **needs a go-ahead; none assumed.
- **2026-10-07 (plan verification, flagging not deciding): the Goal's success criterion (a) is now unsatisfiable by anything this task has tested, and (b) is already met several times over.** The Goal reads "success means either (a) C headroom gain ≥ +1 nats **in key recovery** on 5k attack traces, or (b) a diagnosis that explains why C cannot be learned from ASCADv2 labels." W1, W2 and T31-N1 each satisfy (b) from a different angle (uniform target, algebra cannot substitute for an observation, the shared-mask constraint is provably unexploitable), and N2 satisfies neither literally: it gained **+0.49 nats in β's head likelihood** (not key recovery — key recovery *died*, 16/16 → 0/16). So (a) can only be reached by a step like T31-N4, or by changing what counts. I have not edited the Goal — it is the user's framing — but the next session should read this task as **complete under (b)** and treat (a) as an open, optional bar rather than the definition of success.
- **2026-10-07 (SOTA strategy results): T31-N4 passed its gate at w=0.25, stopping the cascade per plan rule.** (1) **T31-N4 (loss-weight sweep, DONE)** found a middle weight (0.25) that keeps 16/16 attack while lifting β from 5.84% to 7.74%. (2) **T31-N5 (gradient projection, NOT TRIGGERED)** — cascade stops on N4 success. Code has errors; if needed in future work, must be rewritten. (3) **T31-N6 (two-trunk modular, NOT TRIGGERED)** for the same reason. **Remaining decision:** w=0.25 is the gate-passing choice with best head quality; w=0.5 fails the gate but has the best attack (z +9.7556). Single-seed runs sit inside the 1.44 pp noise band. **Recommended T31-N4b:** seed replication on both weights (~30 min) to resolve the attack variance before final adoption. The decision rule is frozen before seeing results.

- **2026-10-07 (user): T31-N4 go-ahead GIVEN — "go head with the new Task plan".** Launched at 19:44 (`logs/t031_n4_train.sh`, log `experiments/out/w9_run.log`), arms w=0.25 then w=0.5, 15 epochs each. Pre-flight per the plan's step 1: a 1-epoch probe at w=0.5 gave **train_loss 100.9965**, which is the interpolated prediction (N2's 18-term epoch-1 loss 56.12 + half the C mass ≈ 100.96) — so the weighting is arithmetically right on the live GPU path, not just in unit tests. Probe also caught a defect the unit tests could not reach: the first draft of the refactor dropped `.to(device)` on the targets, which only a real CUDA batch would surface; fixed and re-probed. **Timing: 63.5 s/epoch ⇒ ~16 min/arm, ~32 min total**, well inside the 2-hour rule. The frozen gate (16/16 non-flat rank-0 **and** β top-1 ≥ 7.5%) is unchanged from its registration, and no weight will be added after seeing numbers.
- **Flag raised while implementing N4, not acted on: T31-N5's code block cannot run as written.** Its projection filter selects on parameter names `'alpha_heads' / 'beta_heads' / 'pi_heads' / 'c_heads'`, but this model's parameters are `alpha.* / beta.* / sbox.<i>.* / permind.<i>.*` (verified from the checkpoint: 455 tensors, head groups `['alpha','beta','permind','sbox']`). Four separate problems: (1) no name matches, so both gradient lists would be empty and the branch would be a no-op; (2) it touches only *head* parameters, whereas the interference N2 measured is in the **shared trunk** — projecting a C head's own gradient changes nothing about what C sends into the trunk; (3) the math is not PCGrad — `dot = Σ_c Σ_o g_c·g_o` double-counts across parameter tensors and `scale * mean(grads_ab_pi)` is not the projection of one vector onto another; (4) `if any('c_heads' in name for name,_ in model.named_parameters() if _ is p)` is O(n²) and would not compile semantically. **N5 is only triggered if N4 fails**, so nothing is blocked now — but it must be rewritten as trunk-space PCGrad (per-task trunk gradients, project C's trunk gradient onto the orthogonal complement of the α/β/π trunk gradient) before anyone runs it. Registering that as a plan-correction task rather than silently "fixing" a step the user authored.

- **2026-10-07: T31-N4 delivered — PASS at w=0.25, FAIL at w=0.5.** The frozen gate was not moved after the numbers: cw05's β 5.74% fails the 7.5% bar even though its attack (z +9.7556) is the best measured, and cw025's pass is reported together with the trace-efficiency cost (median 170 vs 335 traces-to-rank-0). By the cascade rule (gate pass ⇒ use that weight), **N5 and N6 are NOT triggered**. Registered **T31-N4b** (seeded replication of both weights) as the recommended next step before final adoption, because single-seed results sit inside the 1.44 pp two-seed band and the attack split is non-monotone (cw05's z advantage is not stable against w=0.25 on a second run). Pre-frozen decision rule: if w=0.5 > w=0.25 by >0.5 z → adopt 0.5 (attack wins); if w=0.25 > w=0.5 by >0.5 z → adopt 0.25 (head quality wins); if overlap → report both. **User go-ahead given 2026-10-07.**

- **2026-10-08: T31-N4b delivered — the frozen rule fired on its FIRST BRANCH: adopt w=0.5.** Two-seed runs (costs 3296.7 s + evals): w=0.5 z **+10.2582 / +9.7092** (mean 9.984, spread 0.549), median traces 145/150; w=0.25 z **+5.3737 / +9.5211** (mean 7.447, spread 4.147), traces 955/180. mean(w=0.5) − mean(w=0.25) = **+2.5363 nats**, past the 0.5 threshold. **The gate-passing arm was unstable.** Its head-quality claim also falls: β 7.22% at seed 1 ≤ 7.5% bar (N4's single-run 7.74% PASS was a draw). **N5 and N6 reaffirmed NOT triggered** by the cascade rule — first SOTA step produced an operating point. All six C-including runs keep 16/16 non-flat rank-0; w=0 still 0/16; cliff between 0.25 and 0 unchanged. One measurement remains: **T31-N4c** — seeded w=1 replicate (~16 min, pending go-ahead) to validate that w=0.5 beats baseline or sits within w=1 noise. Report: `docs/tasks/T031-reports/T31-N4b-c-loss-weight-replication.md` (detailed result table, seeds 1–2 both weights, validation curves, attack splits, per-byte spread analysis).

- **2026-10-07 (user): T31-N4b go-ahead GIVEN — "run T31-N4b (seed replication, ~30 min GPU) to break the tie between head quality (w=0.25) and attack score (w=0.5)".** Corrected the cost before launching (the registered design is 4 runs, ~64 min, not 30) and ordered both seed-1 arms first so a paired answer existed at the halfway mark. Added `--seed` to `MultiResNet/train.py` (it set none; default `None` preserves earlier semantics). The user **declined** the two extra determinism probes, so same-seed bit-reproducibility is recorded as *not measured*.
- **2026-10-08: T31-N4b delivered — the frozen rule fired on its first branch: adopt w=0.5 — and the result inverts T31-N4.** mean z(0.5) − mean z(0.25) = **+2.5363** (threshold 0.5): w=0.5 gives **+10.2582 / +9.7092** (spread 0.549) with median traces-to-rank-0 **145 / 150** (spread 5), while w=0.25 gives **+5.3737 / +9.5211** (spread **4.147**) with **955 / 180** (spread 775). **The gate-passing arm is the unstable one.** N4's head-quality story also falls: β at seed 1 is **7.22%**, under the 7.5% bar the single run cleared. All six C-including runs keep 16/16 non-flat rank-0; N2 (w=0) remains 0/16, so the carrier cliff is where N4 put it. Consequences accepted as written: **(1) w=0.5 is the adopted C loss weight; (2) N5 and N6 remain NOT triggered and nothing further was started (hard rule 4); (3) "w=0.5 beats the baseline" is NOT claimed as established** — the baseline is one unseeded run, so a seeded w=1 replicate (~16 min) is named as the single missing measurement rather than run on inference. Methodological ruling recorded for the programme: **single-run gates on this task are unreliable; `--seed` should be mandatory in every comparison from here on.**

## Working notes / session log

### 2026-10-08 — CL-0: the contrastive plan was checked before it was run, and two of its three approaches died on the labels

- **What arrived:** a new sibling plan, `docs/tasks/T031-contrastive-learning-plan.md` (three contrastive approaches,
  ~95 min GPU, staged gates), handed over with "try it". The task-memory discipline says a plan's premises get
  checked before its compute is spent, so the CPU pre-run came first: `experiments/cl_premise_check.py` →
  `experiments/out/cl_premise_check.json` (+ `.log`), ~1 min, **no GPU, nothing trained, no checkpoint written**.
- **The plan's own title premise is false, and it matters more than any stage.** "MultiResNet learned C; TimesNet
  did not" — neither learned C: top-1 **0.69%** vs chance 0.39% at gain **−0.0900 nats** on MultiResNet, and
  **0.40% / −0.0729** on TimesNet. C4's finding is a *carriage* difference (+14.0303 z / 16-of-16 vs +0.0766 z /
  0-of-16 with true α/β/π fed to both). Framing the question as "why is C learnable in one model" would have sent
  three GPU stages at a question the data does not pose.
- **CL-α's positive class does not exist.** ASCADv2 here carries **one key per trace** (50,000 distinct full keys
  in 50,000 profiling traces; 10,000 in 10,000 attack traces), and α is redrawn per trace, so "traces with the same
  key byte share α" is not merely weak but exactly at chance: P(αᵢ=αⱼ | same key byte 0) = **0.0039256** against the
  empirical floor **Σp² = 0.0039439** (−1.2σ over 16M sampled pairs); β the same. No temperature, batch rule or
  encoder repairs an equivalence class that isn't there.
- **The MI number was a trap and I nearly reported it.** Plug-in MI(α; k₀) = **0.7208 nats** looks like real
  information, but at n≈195 per group with K=256 the Miller bias alone is ≈(K−1)/2n ≈ 0.65 nats. The identical
  statistic on permuted keys gives **0.7205 ± 0.0022**, so the excess is **+0.0003 nats (0.15σ)**. Every MI in the
  report is quoted against its own permutation null: β −0.0014, π-slot0 −0.0004, attack-split α −0.0088 nats.
- **CL-C's relation is real but lives in the wrong space.** P(same unmasked `S[p⊕k]` | same p₀,k₀) = **1.000** over
  37,974 ordered pairs — the plan is right that it is deterministic and needs no mask labels. But the C head's label
  is the *masked* value, and P(same masked value in that byte's slot) = **0.00327** (floor 0.00391), P(same α) =
  0.00421, P(same β) = 0.00390. Slots are not bytes across traces: **17,255 distinct permutations in 20,000 traces**,
  byte 0 in slot 0 for **6.22%** of traces (chance 6.25%). And the identifiable target is the one W1 measured as
  *least* visible (0 significant samples). So the visible leak needs the labels and the label-free relation has no
  visibility — the same wall from a fourth direction (after W1's SNR, W2's algebra, N1's vacuity).
- **CL-C2 as written breaks the task's own constraint** ("compute true α/β from the cache") — that is mask-label
  training, i.e. worst-case, which T-028 forbids for any non-worst-case claim. Flagged rather than silently rerouted.
- **What survived, with the gate fixed:** **CL-Bridge** is a representation hypothesis and nothing here touches it.
  But its gate ("TimesNet z ≥ +0.4") grades a decode that is dead because its C head carries nothing. Gate on
  carriage instead: re-run C4's E3 arm and require **+0.0766 / 0-of-16 → z ≥ +3.0 with ≥8/16 non-flat rank-0**; the
  harness already costs ~1 min CPU per checkpoint. A **repaired CL-C** is conceivable as a bag-level,
  permutation-invariant match against the 16 computable unmasked values per trace (no slot assignment needed) — the
  only contrastive idea in the plan that maps onto an available signal — but it should earn its own CPU visibility
  check before any GPU, given W1.
- **Not run, on purpose:** Stage 1 (~30 min), Stage 2 as written (~20 min), Stage 3 (~45 min), the
  `cl_alpha_dataset.py` loader, and the `ASCAD_train_models.py` `--contrastive-loss` flags. Budget intact: T-028's
  ≤ 2 h untouched, T31-N4c's ~16 min still queued. Stopped for a ruling — the plan's own mitigation clause ("if
  CL-α fails, close the line immediately") is satisfied here at ~1 min CPU instead of 30 min GPU.
- **Files touched:** the plan file (Status header, premise table, checklist annotated),
  `T031-reports/CL-0-premise-check.md` (new), `T031-reports/LEDGER.md` (row 8 of 8), this file (Working notes, Next
  actions ×2, Related), `docs/tasks/INDEX.md`, `docs/lab-log/todo.md`. New code:
  `experiments/cl_premise_check.py`. T-028/T-029/T-030 files untouched by this step.

### 2026-10-08 — synthesis note, including the N4b correction

- [`docs/grok-deep-research-2026-10-08.md`](../grok-deep-research-2026-10-08.md) records the three-route cross-check. The workflow draft had stopped at single-run N4 (w=0.25 looked like the pass). This note follows N4b: **adopt w=0.5**. The hard GF lookup remains a flat key score; the only LUT gain is true masks on top of the measured C posterior.

### 2026-10-08 — T31-N4b: the replication ran, and it reversed the sweep's verdict

- **Ran** the registered seeded design — 34-head MultiResNet, 50k profiling traces, 15 epochs, batch 64,
  RAdam lr 1e-4, sub_avg, global clip 1.0, identical to the N4 arms and the matched control except `--seed`:
  `w=0.25 seed 1` (917.9 s), `w=0.5 seed 1` (794.7 s), `w=0.25 seed 2` (791.7 s), `w=0.5 seed 2` (792.4 s)
  = **3296.7 s of training (~55 min)**, best-val epoch **7** for all four, then four 5k-trace evaluations
  (three logged at 64.0/64.5/65.5 s; the fourth bounded under 2 min, its timing printed to the console only). `experiments/out/w10_multiresnet_cweight_{0p25,0p5}_seed{1,2}(_last).pth` + sidecars +
  histories + mean waveforms; launcher `logs/t031_n4b_train.sh`, log `experiments/out/w10_run.log`.
- **Judgment:** `mean z(0.5) − mean z(0.25) = +2.5363` against the 0.5 threshold registered before any number
  existed ⇒ **adopt w=0.5**. Per-arm: w=0.5 z **+10.2582 / +9.7092** (spread 0.549), median traces→rank-0
  **145 / 150** (spread 5); w=0.25 z **+5.3737 / +9.5211** (spread **4.147**), traces **955 / 180** (spread 775).
  **So N4's gate-passing arm is the unstable one and its gate-failing arm is the best operating point in the
  task** — the kind of reversal a single run cannot see.
- **N4's head-quality claim also does not survive.** The bar was β ≥ 7.5% *and* 16/16; seed 1 gives **7.22%**
  (seed 2 8.02%), so the single-run PASS at 7.74% was a draw, not a result. Recorded in
  `w10_replication.json` as `w025_head_quality_claim.stands: false`.
- **What did not move:** all six C-including runs keep **16/16 non-flat rank-0** (three seeds, two weights) and
  N2 at w=0 is still 0/16, so the carrier cliff stays between 0.25 and 0. Inside [0.25, 1] the weight mostly
  buys **variance**, not capability.
- **Two scales disagree, and that is the finding.** On validation β the weights are indistinguishable
  (means 0.0802 vs 0.0742; within-weight seed spread 0.76–1.20 pp ≥ the 0.60 pp between-weight gap) yet both
  clear w=1 (0.0593) ⇒ head quality separates {0.25, 0.5} from 1; the attack metric separates 0.5 from 0.25
  decisively. The cheap validation scale is not a proxy for the expensive attack scale here.
- **Boundary held deliberately:** w=0.5 beats the baseline on both attack metrics (9.984 vs +9.5047 z;
  147.5 vs 170 traces) **but the baseline is a single unseeded run**, so "better than the baseline" is written
  as *suggested, not established* and a **seeded w=1 replicate (~16 min)** is named as the one missing
  measurement — not run, because hard rule 4 says stop and the user has not asked for it.
- **Methodological takeaway registered for the programme:** single-run gates on this task are unreliable, so
  `--seed` should be mandatory in every future comparison. This is the fifth result of T-031 and the only one
  about how we measure rather than what the model does.
- **Honest caveats:** n = 2 per weight is a coarse instrument and the rule has no confidence interval; the
  threshold was applied **symmetrically** (the plan states it only for the w=0.5 direction, so a 0.01 gap
  would otherwise have "decided" for w=0.25) with the raw difference printed alongside; **same-seed
  determinism was not measured** (the two extra probes were declined), so part of what is attributed to "seed"
  could be ordinary AMP/cuDNN run jitter — which would not change the ordering but would change how the
  variance is described; `_last` checkpoints exist for all four arms and were **not** evaluated.
- **Artifacts:** report `docs/tasks/T031-reports/T31-N4b-c-loss-weight-replication.md`; LEDGER row 7;
  `experiments/w10_replication_check.py` → `w10_replication.json` (script written and run while all four
  artifacts were missing; it printed `incomplete` rather than interpolating).
- Files touched: this file (Status, N4b plan block ticked with RESULT, N4 row annotated, N5 status reaffirmed,
  Decisions ×2, this working note, Next actions, Priority snapshot, Related), `docs/tasks/INDEX.md`,
  `docs/lab-log/todo.md`, `MultiResNet/train.py` (`--seed`, additive, default None). Nothing pre-existing was
  overwritten; the HDF5 was read-only; another session's GPU process was left running untouched.

### 2026-10-07 — T31-N4: the interpolation exists, and it has a price the gate didn't measure

- **Ran** `--c-loss-weight` (new flag, additive) on 34-head MultiResNet, 50k traces, 15 epochs, all
  other hyperparams as N2 and its control: **w=0.25 → `experiments/out/w9_multiresnet_cw025.pth`**
  (904.9 s, best epoch 8) and **w=0.5 → `w9_multiresnet_cw05.pth`** (958.3 s, best epoch 6);
  evaluated both on the 5k attack split with the N2 evaluator (77.8 s / 97.8 s), and judged the gate
  with `experiments/w9_gate_check.py` → `w9_n4_gate.json`. Guard tests 20/20. No checkpoint or `.h5`
  overwritten; shared the GPU with another session's 8.6 GB process, untouched.
- **The result:** **w=0.25 passes** (β top-1 **7.74%**, π **71.53%**, exact permutation **1.22%**, still
  16/16 non-flat rank-0 with no oracle) and **w=0.5 fails** the β bar at 5.74%. So C is not all-or-nothing:
  its *presence* in the objective carries the key and its *weight* modulates how hard it competes.
- **The uncomfortable part, reported rather than smoothed:** the gate never tested traces-to-rank-0, and
  cw025's median is **335 vs 170/205** — same recovery, later. Per-byte spread (50–720 vs 20–850) means
  the honest statement is "worse median, overlapping distributions". And **w=0.5 has the best attack of
  the five** (z +9.7556 > baseline +9.5047) with a better β likelihood, while failing the bar. Both say
  the gate's two criteria under-describe what "better" means here.
- **Validation says monotone, attack split says otherwise.** Late-epoch val β: 0.0593 (w=1) → 0.0724 →
  0.0770 → 0.1002 (w=0), with the arms above the control in 14/15 and 13/15 paired epochs (π 15/15 at
  w=0.25). The attack-split points are 5.84/4.40 → 5.74 → 7.74 → 9.30 — non-monotone at 0.5 — and since
  two identical w=1 runs differ 1.44 pp, that non-monotonicity is within noise. One run per weight is the
  real limit of this step; `w9_n4_gate.json` records `beta_rises_monotonically_as_c_weight_falls: false`
  rather than my prose.
- **Engineering notes.** Extracting the weighting as `weighted_multilabel_loss()` made the arithmetic
  testable without a GPU, and the probe then caught the one thing CPU tests cannot: the refactor had
  dropped `.to(device)` on targets. Its measured losses were also a numerical check on the whole design —
  100.9965 (w=0.5) and 78.5962 (w=0.25) against predictions 100.96 and 78.54 interpolated from N2's and
  the control's epoch-1 losses. `--no-c-heads` is now the w=0 special case of the same flag.
- **Cascade consequence:** by the plan's own trigger (success ⇒ use that weight, failure ⇒ N5), **N5 and
  N6 are not triggered**. That is fortunate: N5's code block as written cannot run (parameter names that
  don't exist in this model, head-space rather than trunk-space projection, and not PCGrad math) — flagged
  in Decisions and `todo.md`, not silently rewritten.
- **Not done deliberately:** no extra weights (0.1/0.15/0.35) after seeing numbers — the registered rule
  forbids it; `_last` (epoch-15) checkpoints exist but were not evaluated; W4 (π) still untriggered.
- Files touched: `MultiResNet/train.py` (the `--c-loss-weight` addition), new
  `experiments/w9_gate_check.py`, `experiments/w6_trainer_patch_tests.py` (extended),
  `logs/t031_n4_train.sh`, `experiments/out/w9_*` (2 checkpoints + `_last` + sidecars + histories +
  npz caches + 2 eval JSONs + gate JSON + logs), this file, LEDGER, `INDEX.md`, `todo.md`,
  report `T031-reports/T31-N4-c-loss-weight-sweep.md`. No `.h5` write, no GPU training outside these arms.


### 2026-10-07 — T31-N2: taking C out of the loss helps every other head, and kills the attack

- **Ran** the probe, then two 15-epoch trainings and three evaluations, all on GPU where inference was
  needed (`py10`, RTX 5090): `logs/t031_n2_train.sh` → `experiments/out/w6_multiresnet_nochead.pth`
  (839.8 s) and `w6_multiresnet_34h_control.pth` (980.9 s); `experiments/w6_nochead_eval.py` →
  `w6_eval_{baseline,control,nochead}.json` (61.7/71.8/80.1 s); `experiments/w6_clip_binding.py` →
  `w6_clip_binding.json`; `experiments/w6_trainer_patch_tests.py` (8 checks, 0 failures). Logs in
  `w6_run.log`, `w6_eval_*.log`. No existing checkpoint or `.h5` touched; the probe's 550 MB weight
  file was deleted once it had served its timing purpose, its log line kept.
- **The result T-031 has been circling, stated plainly:** with C out of the loss, β top-1 goes
  5.84% → **9.30%** (gain +1.51 → **+2.00** nats of a 5.545 ceiling, mean p(true) ×2.03) and π
  67.03% → **72.13%** with exact-permutation recovery doubled; α is unchanged. And the same model
  cannot attack at all (z **+0.4741**, 0/16, GE 85.0) — the numbers W2 predicted for a C-less decode.
- **Why the control run was not optional.** The trainer sets no seed; on two 34-head runs β differs by
  **1.44 pp** (5.84% vs 4.40%). Without that, "+3.5 pp on β" would have been an uncalibrated number.
  The paired per-epoch validation curves do the heavy lifting: N2 above control in 13/15 epochs for β
  (mean +0.0218) and 14/15 for π (mean +0.0348) — same split, same metric, 15 points.
- **The mechanism was contested and settled by measurement.** Dropping C shrinks the pre-clip gradient
  (122.9 → 69.4) so the fixed 1.0 clip rescales it 1.78× *more* — which looks exactly like a 1.78×
  learning-rate increase and would have made the whole result an optimiser artifact. It is not: a
  global clip is a uniform rescale, and an RAdam update is near-0-homogeneous in gradient scale
  (measured displacement ratio **0.959**, direction cosine **0.99960** at that factor; even 10× buys
  only ~8%). So the effect is gradient *direction* — the trunk no longer fitting the least visible
  target in the dataset.
- **A silent-corruption hazard found and guarded in my own patch.** The trainer reads targets
  positionally (`batch[k_idx + 1]`). Filtering the key list and keeping that indexing — the obvious
  implementation — would have pointed **16 of 18** surviving heads at the wrong label columns and
  trained on scrambled targets with no error raised. The patch uses a name→column dict instead, and a
  guard test asserts both that the dict equals the original positional read on the full set and that
  the naive version would have broken.
- **Two currency traps worth remembering.** val_loss is a *sum* over heads, so N2's 11.68 and the
  control's 102.92 are different quantities (the control landing at ~103 is also the proof the patch
  preserved the baseline's convention). And GE on a flat posterior ranks float summation order: 249.11
  here vs a tie-safe rank of 255, disagreeing by 31.9–67.5 — while agreeing to 0.000e+00 on every
  non-flat decode. The `flat`/spread fields are the real readout for degenerate cases (T-028's lesson
  again).
- **What this changes.** T-031 now has one positive and three negative results, and the positive one
  prices the tension: C is useless as a classifier yet indispensable as the carrier, so the live
  question is no longer "can we replace or ignore C" but "**at what loss weight does C stop hurting
  the mask heads without being removed**" — β is 4.4–5.8% at weight 1 and 9.30% at weight 0, and the
  interpolation has never been measured. Registered as **T31-N4** with a gate proposed in the report
  before any number is seen. Needs a go-ahead; none assumed.
- Files touched: new `experiments/w6_nochead_eval.py`, `w6_trainer_patch_tests.py`,
  `w6_clip_binding.py`, `logs/t031_n2_train.sh`, checkpoints + sidecars + npz caches under
  `experiments/out/`, report `T031-reports/T31-N2-c-deletion-retrain.md`, LEDGER row, this file,
  `docs/tasks/INDEX.md`, `docs/lab-log/todo.md`. **Modified repo training code:** `MultiResNet/train.py`
  (additive flags + per-head val metrics + column map). No `.h5` write; no T-028/T-029/T-030 file.

### 2026-10-07 — T31-N1: the shared-mask constraint is provably exploitable by nothing

- **Ran** `experiments/w7_joint_ab_decode.py` (301.6 s, `gpu_used: False`, git `e355c1a`) →
  `experiments/out/w7_joint_ab_decode.json` + `w7_run.log`; `experiments/w7_joint_ab_decode_tests.py`
  (33 checks, 0 failures); supplementary audit `experiments/w7_dissect_normalisation.py` (160.1 s) →
  `w7_dissect_normalisation.json`. Smoke-tested at `--limit 200` to a scratch path first, which caught
  two shape bugs before any number was recorded. No training, no checkpoint, `.h5` read-only, N2/W4
  not started.
- **The step's headline is a proof, not a measurement.** Marginalise a byte's own key and that byte's
  slot contributes `Σ_v P[a·S[p^v] ⊕ c]`, which is exactly **1** for every a ≠ 0 — the hypothesis
  family is a bijection onto the whole value space, so 15 other slots tell you nothing about the mask
  pair. The task's premise (one (α,β) serves 16 slots ⇒ cross-slot consistency sharpens β) is false
  *as a decoding statement*, and E1 confirms it empirically: Δz **+4.2e-05**, identical
  traces-to-rank-0 on all 16 bytes. The identity is tested over all 256 α, not asserted.
- **The only way to make the constraint speak is to plug in key estimates, and the data refuses:**
  pass-1 per-byte top-1 accuracy is **0.50%** (chance 0.39%). The EM then eats its own mistake —
  **+5.72/460** (soft) and **+5.48/420** (hard) against **+9.50/170**. With the true key supplied
  (oracle bound, correctly labelled and excluded from the gate) the same estimator lifts β from
  **5.84% to 11.34%**, and with the mask heads removed the 16 slots alone reach **α 0.96% / β
  0.64%** — i.e. the C heads' joint structure carries almost no mask information. That closes W1's
  open "mask-conditional visibility" question in the negative, which is the most useful thing this
  step produced.
- **The number that decides where the money is:** β **correctness** +3.4 z (W2 D1c) vs β
  **confidence** ≈0.3 z vs β **shape** ≈0 (τ=20 still +9.2032). Tempering β to τ≈3 is the only
  legitimate change that moves anything (+0.232/+0.371 z cross-fitted) and it *worsens*
  traces-to-rank-0 (175 vs 170), so it is not a pass and is not presented as one. NLL says τ\*=1:
  β is not miscalibrated, it is merely weak. Conclusion recorded in the plan: further effort belongs
  on the β head, not on the decoder — which partly mis-aims T31-N2 as worded.
- **A false identity of my own, caught by its own control.** The first diagnostic claimed
  `Σ_j Pr(j) = 1 + A[0](256Q[0] − 1)`; measurement disagreed by up to **2.587**, which the α term
  cannot produce (its ceiling is 0.65), so the algebra — not the code — was wrong. The missing factor
  is `proba_dissect_permind`'s per-byte mixture mass `R = Σ_s PI[t,s,b]`, which is *not* normalised
  per trace (median 1.015, range **0.008–3.587**, 86% of cells outside ±5%). Corrected identity
  `256·A[0]·Q[0] + (1−A[0])·R` matches to **8.91e-07**. Both R and the α=0 branch are constant over
  hypotheses within a (trace, byte), hence inert for `rank()` — E5 verifies this by returning
  byte-for-byte identical traces. Two production "bugs" retired before anyone spends a step on them.
- **Dataset fact pinned** (measured on the HDF5, consistent with `docs/notes/datasets.md:170`):
  α is drawn from 1–255 (profiling counts average 1960.8 = 500000/255; attack labels show 255 distinct
  values), β from 0–255 (1953.1 = 500000/256). The 5 profiling rows with `masks[18]=0` are rows
  99999/199999/299999/399999/499999 with **all 38 mask columns zero** — record-boundary artifacts, not
  α draws; 0 of 10,000 attack rows. So the α=0 dissection branch is unreachable from labels.
- **Process note:** the gate now classifies every decoder by role (`candidate` / `oracle_bound` /
  `control`) before judging, because at n=200 the oracle EM *did* report a gate pass. That was a real
  trap and the classification exists to keep a later session from repeating it.
- **Neighbour defect found, not touched:** `experiments/w2_decoders.py` defines
  `c_head_target_table` ×3, `run_on_indices` ×3 and `split_half_stability` ×4 (S2 editing accident).
  Python binds the last, which is the correct one (the first would raise), so S2's recorded numbers
  stand; this step imports only singly-defined helpers. Left alone under hard rule 2 and registered
  for the user's cleanup ruling.
- Files touched: new `experiments/w7_joint_ab_decode.py`, `experiments/w7_joint_ab_decode_tests.py`,
  `experiments/w7_dissect_normalisation.py`, `experiments/out/{w7_joint_ab_decode.json, w7_run.log,
  w7_dissect_normalisation.json, w7_dissect_norm.log}`, `T031-reports/T31-N1-joint-alpha-beta-decode.md`,
  LEDGER row, this file (Status, N1 checkbox + DONE block, Decisions, notes, next actions),
  `docs/tasks/INDEX.md` T-031 row, `docs/lab-log/todo.md`. T-028/T-029/T-030 files untouched; no
  checkpoint, no `.h5` write, no GPU.

### 2026-10-06 — W2: the LUT cannot replace the C head, and that locates the lever

- **Ran** `experiments/w5_c_lut.py` (new) → `experiments/out/w5_c_lut.json` + `w5_run.log`, 268 s, pure numpy on the cached 34-head softmax. 8 decoders, one tie convention, one zero floor.
- **The spec's step 1 is flat by construction, and the flatness check is what says so**: a point mass at `LUT[α̂, Sbox[p⊕k]] ⊕ β̂` evaluated at that same value gives every hypothesis the same score → spread exactly 0.0, `flat_bytes: 16`. The generous reading (probability of the implied value from the mask posteriors alone) reaches z −0.31 and never ranks 0. **A lookup table is a function, not a measurement** — V is the only key-dependent *observed* quantity in the decode path.
- **D2 is the measured decode of a C-less model** on the existing trunk: z 0.0000, spread 0.0, 0/16. That is why step 3 (retrain without C heads) was not run: its gate is unreachable. It remains worth doing for a *different* question (does deleting the 38.3% gradient sink improve α/β/π quality?) which needs a new, non-rank-0 gate.
- **The finding worth acting on is on the mask side.** Same weak C head, true masks fed to the LUT: median z **+9.5047 → +12.9332**, traces-to-rank-0 **170 → 95**. Same but with mask *argmaxes* instead of marginalising: **z +5.1152, 1,065 traces** (α̂ is 100% accurate, β̂ only 5.84%, so the plug-in is wrong on 94% of traces). Both bracket the cost of β-posterior uncertainty → next: joint (α,β)-consistency decoding across the 16 slots (one mask pair serves all 16) and β calibration. CPU-only on cached preds.
- **Algebra and decoder controls:** `V_slot = LUT[α, Sbox[p⊕k]] ⊕ β` reconstructed from metadata reproduces the stored `sbox_masked` on **100.0000%** of 80,000 values; D0 equals the S2 fast path to **4.16e-17**; sanity gate D0 z **+9.50466778175834** vs documented +9.50 (and vs S1's independent rank_degeneracy value), 16/16 non-flat. Cross-task replication: D1b/D1c/D2/D3 = S2's V14/V6/V9/V7 to the last digit.
- **A hazard this exposed, recorded in the report:** with hard mask-plugging the wrong-hypothesis scores collapse onto `rank()`'s zero floor (spreads 290–11,210 nats), and the D1b shuffled null touched rank 0 on 2 bytes at 10/200 traces before ending at 139/99. So decoder comparisons here should be read on traces-to-rank-0, not z. z saturating at √255 (T-030 S2) and this floor behaviour are now two documented reasons not to rank decoders by z alone.
- **W4 not triggered:** its condition was 'W2 leaves 2–4 bytes unsolved'; W2's cases either solve all 16 or are structurally flat, so the π hypothesis is untouched.
- Files touched: new `experiments/w5_c_lut.py`, `experiments/out/w5_c_lut.json`, `w5_run.log`, `T031-reports/W2-c-lut-replacement.md`, LEDGER row, this file, INDEX T-031 row, `docs/lab-log/todo.md`. No checkpoint, no `.h5`, no GPU, no T-028/T-029/T-030 file.


### 2026-10-05 — W1: the C target is uniform and nearly invisible; the loss is aimed at it

- **Ran** `experiments/w4_c_diagnosis.py --parts h1,h2,h3` (410 s, exit 0) →
  `experiments/out/w4_c_diagnosis.json` + `w4_run.log`. Profiling traces only: 50k for label
  stats/SNR/probe-training (same count the baseline trained on), a disjoint 5k as probe test set;
  baseline head numbers recomputed from the cache rather than quoted. No network trained, no
  checkpoint written, `.h5` read-only.
- **The task's framing premise was wrong and is corrected above** (Context §"Why C scores below
  uniform"): 256 classes not 16; mode 0.46% at 50k not 0.7%; and the 0.6–0.76% "constant predictor"
  bars S1 reported were modes fitted on the same 5k attack traces the model is scored on. That last
  one matters methodologically — a prior estimated on the evaluation set is not a null.
- **What is actually true:** C's masked target is statistically uniform and barely visible per sample
  (21–25 of 15,000 samples past p<1e-6, versus 2,872 for α), and a linear model on the whole trace
  cannot beat its own permuted-label null on it, while reading α at 40% and π at 18%. So the head that
  carries the key (S2: uniform-C → z 0.00) is trained on the least visible target in the dataset.
- **The measurement that redirects W4:** unmasked `Sbox[p^k]` and `p^k` are *worse* — excess SNR
  0.0014–0.0021 with **zero** significant samples. Marginal visibility orders
  α ≫ β ≫ π > V(masked) ≫ C(unmasked) ≈ p^k ≈ 0. Swapping to an unmasked target is therefore not a
  lever, and byte-indexing the same masked value is a relabelling. The remaining idea with any
  support is mask-**conditional** visibility, which the SNR used here cannot measure.
- **The optimisation finding:** at the baseline checkpoint the 16 C heads are 52.6% of the averaged
  loss (CE 5.635 nats, i.e. above log 256 = 5.545 — the head is worse than a uniform guess) and
  38.3% of the trunk's gradient power, while α — solved — is 0.6% and 0.005%. π takes 59.7%. So more
  than a third of the shared trunk's learning signal goes to the target this step showed to be nearly
  invisible. Attribution scaled as the trainer scales it, and cross-checked against the full 34-head
  gradient (quadrature 0.00126 vs measured 0.00132).
- **Open question put to the user (W1 report §6.1):** pre-registered W3 tests *up*-weighting C to 2.0;
  W1's evidence points the other way. Running W3 as specced is a legitimate test of a contrary
  hypothesis, but it must be labelled that way — and gates stay frozen either way.
- **Tooling notes:** the vectorised SNR is validated against `utils/snr.snr_vector` (2.0e-14) before
  use; floors are analytic (K−1)/N with the empirical permuted-label null reported alongside (they
  agree to ≤3e-4); probe likelihoods are deliberately not reported as nats (unregularised softmax on
  15,000 features gives −55 nats for real *and* null — miscalibration, not information).
- Files touched: new `experiments/w4_c_diagnosis.py`, `experiments/out/w4_c_diagnosis.json`,
  `experiments/out/w4_run.log`, `T031-reports/W1-c-diagnosis.md`, LEDGER row, this file (Status,
  Context correction, W1 checked, notes, next actions), `docs/tasks/INDEX.md` T-031 row, and
  `docs/lab-log/todo.md`. T-028/T-029/T-030 files untouched (hard rule 2).

### 2026-10-06 — plan revision after W1: LUT replaces training, π as conditional post-hoc

- **W1 refutes the original task premise.** The three hypotheses were: (1) label imbalance, (2) unlearnable target, (3) training loss allocation. W1 refutes (1) and confirms (3) is the real problem: C takes 52.6% of averaged loss and 38.3% of trunk gradient while being nearly invisible (SNR 0.045–0.061, 21–25 significant samples vs α's 1.8325 / 2,872). This means (2) "unlearnable" is **not true** — it is a *learnable* target that the current setup makes the network try to learn when it should use algebra instead.
- **New strategy:** W2 and W4 from the original spec (per-head LR, alternative targets) are no longer the tests to run. Instead: **W2-LUT tests the core hypothesis** — replace the C-head with deterministic computation and retrain α/β/π, freeing their 62% of gradient. This is the minimum viable test: does perfect C, without training, recover 16/16?
- **Why W4 becomes conditional on π-recovery:** T-028 Phase 0 showed A4 (no π, uniform π) still recovers 12/16 at z +3.31. If W2-LUT recovers 16/16, π training is working fine. If W2-LUT leaves 2–4 bytes unsolved, π becomes the suspect, and π is *deterministic* (not a leakage model) — so W4 tests π-recovery via post-hoc search (brute-force on permutation space, partial key search, or adversarial π finding).
- **W3 direction resolved:** down-weight C (per W1 evidence) rather than up-weight (per pre-registration). But this is now moot: W2-LUT deletes C entirely, removing the allocation problem at its source.
- **W1 artifacts:** `experiments/w4_c_diagnosis.py`, `experiments/out/w4_c_diagnosis.json`, `w4_run.log`, `T031-reports/W1-c-diagnosis.md`, LEDGER row, task file updates, INDEX row, todo.md updates. No checkpoints or `.h5` touched.
- Files touched: this file (Status, Plan sections, next actions, working notes), `docs/tasks/T031-reports/` (LEDGER W1 row added), `docs/tasks/INDEX.md` (T-031 status), `docs/lab-log/todo.md` (W1 DONE). No checkpoint or `.h5` touched.

## Next actions

| Action | I | U | S | Notes |
|--------|---|---|---|-------|
| ~~W1 diagnose C anti-informativeness~~ | H | H | H | **DONE 2026-10-05, gate PASS** — imbalance refuted; target near-invisible per sample; C takes 52.6% of loss / 38.3% of trunk gradient. Report `T031-reports/W1-c-diagnosis.md` |
| ~~W2 replace C-head with algebraic LUT~~ | H | H | H | **DONE 2026-10-06, gate FAIL** (steps 1/2/4 measured; step 3 not run — its gate is unreachable). A LUT is a function, not an observation: the point-mass replacement is exactly flat (spread 0.0, 16 flat bytes). Lever located on the mask side: oracle masks → z +12.93 / 95 traces vs +9.50 / 170. Report `T031-reports/W2-c-lut-replacement.md` |
| ~~T31-N1 joint (α,β)-consistency decoder + β calibration~~ | H | H | H | **DONE 2026-10-07, gate FAIL, fallback branch met.** The constraint is **provably vacuous** under marginalisation (bijection identity, deviation 8.66e-15; E1 moves z by +4.2e-05 and nothing else); plug-in-key EM is self-confirming and loses (pass-1 key accuracy **0.50%** → +5.72/460, +5.48/420); the C heads do not know the mask even given the key (slot-only α 0.96% / β 0.64% at chance 0.39%). β correctness +3.4 z vs β confidence ~0.3 z vs β shape ~0 ⇒ the prize is head accuracy, not decoding. Report `T031-reports/T31-N1-joint-alpha-beta-decode.md` |
| ~~Rule on where the β-accuracy question goes~~ | H | H | M | **RESOLVED 2026-10-07 by the user: "Run T31-N2 as planned"** — executed, and it answered the question affirmatively (see the row below) |
| ~~T31-N2 C-deletion interference retrain (GPU)~~ | M | H | M | **DONE 2026-10-07, PASS on the primary criterion for β and π.** β top-1 5.84% → **9.30%** (gain +1.51 → +2.00 nats), π 67.03% → **72.13%**, exact permutation ×2.2, α unchanged; matched 34-head control supplies the null (two 34-head runs differ 1.44 pp on β), paired per-epochs favour N2 in 13/15 (β) and 14/15 (π). Its own attack is dead (z +0.4741, 0/16, GE 85.0) exactly as W2 predicted. **Clip-budget explanation measured and rejected** (1.78× post-clip rescale buys ~4–9% of update magnitude, cosine 0.9996) ⇒ the mechanism is the sink. Report `T031-reports/T31-N2-c-deletion-retrain.md` |
| ~~T31-N4 — C loss-weight sweep (GPU)~~ | H | H | M | **DONE 2026-10-07: PASS at w=0.25, FAIL at w=0.5.** The middle exists — β top-1 5.84% → **7.74%**, π 67.03% → **71.53%**, exact-permutation recovery 0.60% → **1.22%**, still 16/16 non-flat rank-0 — but median traces-to-rank-0 goes **170 → 335**, which the frozen gate never tested, and w=0.5 has the best attack of all five models (z **+9.7556**) despite failing the β bar. Gate verdict computed from artifacts by `experiments/w9_gate_check.py`. **N5/N6 therefore not triggered.** Report `T031-reports/T31-N4-c-loss-weight-sweep.md` |
| ~~T31-N4b — replicate the two weights before adopting either~~ (GPU) | H | H | M | **DONE 2026-10-08 — the frozen rule fired on its first branch: adopt w=0.5.** mean z(0.5) − mean z(0.25) = **+2.5363** (threshold 0.5): w=0.5 **+10.2582 / +9.7092** (spread 0.549) at 145/150 median traces→rank-0 (spread 5) vs w=0.25 **+5.3737 / +9.5211** (spread **4.147**) at 955/180 (spread 775). **N4's gate-passing arm was the unstable one**, and its head-quality claim falls too (β 7.22% at seed 1 < the 7.5% bar). 16/16 non-flat rank-0 in all six C-including runs; N2 (w=0) still 0/16. 4 runs, 3296.7 s + 4 evals. **N5/N6 still not triggered; stopped per rule 4.** Report `T031-reports/T31-N4b-c-loss-weight-replication.md` |
| **Seeded w=1 replicate — the one measurement that would justify "w=0.5 beats the baseline"** (GPU, ~16 min/run; needs go-ahead) | H | M | M | Everything in N4b is compared **between** the two down-weighted arms, paired by seed. The baseline's +9.5047 z / 170 traces is one **unseeded** run, so w=0.5's mean (9.984 / 147.5) is *suggested* better, not shown. One seeded w=1 run makes the comparison like-for-like; a second gives the baseline its own variance, which is also what every future step here will be measured against. Not started — hard rule 4, and the stronger claim is deliberately not made in the report |
| **Adopt C loss weight 0.5 as the production setting** (user ruling; no compute) | H | H | L | By the frozen rule this is already decided operationally: `--c-loss-weight 0.5` in `MultiResNet/train.py`, 16/16 non-flat rank-0 on both seeds, mean z +9.984, median 147.5 traces. Needs only the user's confirmation that the cascade ends here (N5/N6 stay untriggered) rather than continuing to optimise inside the weight |
| **CL-0: contrastive-plan premises checked — Stage 1 not run** | H | H | H | **DONE 2026-10-08 (CPU ~1 min, no GPU).** `docs/tasks/T031-contrastive-learning-plan.md` proposes three contrastive approaches; two are measured false before any training. **(i)** Its title premise ("MultiResNet learned C") is wrong — C head top-1 **0.69%** at **−0.0900 nats** vs TimesNet 0.40% / **−0.0729**; what differs is *carriage* (C4: +14.0303/16-of-16 vs +0.0766/0-of-16). **(ii) CL-α's positive class does not exist:** each profiling trace carries its own key (50,000 distinct), and α is redrawn per trace — P(αᵢ=αⱼ \| same key byte) **0.0039256** vs exact floor **Σp² = 0.0039439**; MI 0.7208 nats is estimator bias (own permutation null 0.7205 ± 0.0022). **(iii) CL-C's relation is real only in byte space** (P(same unmasked c \| same p,k) = 1.000 over 37,974 pairs) while the head's masked label is uncorrelated with it (P = 0.00327 vs floor 0.00391) and slots are not bytes across traces (byte 0 in slot 0: 6.22% vs chance 6.25%). **(iv) CL-C2 as written uses oracle α/β**, which violates T-028's non-worst-case constraint. **Survives: CL-Bridge** (representation hypothesis) with its gate replaced by the C4 carriage arm E3 (+0.0766/0-of-16 → z ≥ +3.0, ≥8/16), and a possible **bag-level permutation-invariant CL-C** that needs its own CPU visibility check. Report `T031-reports/CL-0-premise-check.md`; code `experiments/cl_premise_check.py` |
| **Ruling requested: how to spend the remaining GPU** (each arm timed; none started) | H | H | L | Three candidates, all needing a go-ahead because every one is a training run: **(a) T31-N4c** — seeded w=1 MultiResNet reference, ~16 min, the only thing that converts "w=0.5 is the best C-including weight tested" into "w=0.5 beats the baseline"; **(b) CL-Bridge** — ~45 min, the one contrastive arm whose premise survived, gated on carriage rather than z; **(c) TimesNet `--c-loss-weight 0.5`** — ~16 min × 2 seeds, licensed by the letter of T-028 §C4's branch but argued against by measurement (an oracle swap tests the other heads; down-weighting C moves β/π, never makes C readable). Budget note: T-028's ceiling is ≤ 2 h total across all phases and it is currently untouched; (a)+(b) = ~61 min |
| W4 conditional: π-recovery post-hoc search (not triggered) | M | L | M | **CPU-only.** T-028 A4 shows 12/16 without π (marginalizable). Neither N1 nor N2 triggers it: every legitimate decoder recovers 16/16 or is inert, and N2's better π heads (72.13%) still ride on a C head the model cannot train. Gate unchanged: does post-hoc π recovery unlock bytes that fail otherwise? **Reinforced 2026-10-08 by T-028 C4 item 1: oracle π is worth +1.67 z on the baseline and +0.18 z on TimesNet, so π is not the binding constraint in either model** |

### Priority snapshot

- **Do now:** nothing this task can decide alone, and per hard rule 4 the implementing agent has stopped. **T31-N4b came back and reversed T31-N4: adopt C loss weight 0.5** (mean z +9.984 vs 7.447, spread 0.55 vs 4.15; 147.5 vs 567.5 median traces; 16/16 non-flat rank-0 at both weights and both seeds). The one open ruling is whether to spend **~16 min on a seeded w=1 run** to convert "w=0.5 is the best of the C-including weights tested" into "w=0.5 is better than the current baseline" — without it the stronger sentence is not claimed. **N5/N6 remain untriggered** (the cascade stopped at the first pass; N5's block also cannot run as written). If the user prefers the original N4 reading, the counter-evidence is now on file: cw025's single-run β 7.74% did not replicate (7.22% / 8.02%) and its attack collapsed at seed 1 (z +5.37, 955 traces). Five results stand: three closing replacement routes (W1 diagnosis, W2 algebra, N1 decoding), one opening the trade (N2 interference), one settling it and one methodological lesson (N4/N4b: **single-run gates here are unreliable — `--seed` from now on**).
- **Before next meeting:** six step reports delivered (W1, W2, N1, N2, N4, N4b). **The headline changed between N4 and N4b, and that is the meeting-worthy part:** the arm that passed a pre-registered gate did not survive replication, and the arm that failed it became the recommended model — C at loss weight **0.5** gives the best attack measured in this task (z **+10.2582 / +9.7092**, median traces-to-rank-0 **145 / 150**, 16/16 non-flat at both seeds) while β still improves over w=1 on validation (0.0742 vs 0.0593). Reference numbers unchanged: baseline **+9.5047 / 170**, N2 without C **9.30% β and a dead attack (0/16, z +0.4741)**, carrier cliff between 0.25 and 0. Also for the record: two production "bugs" retired as provably inert; one dataset clarification (α ∈ 1–255; the five α=0 profiling rows are all-zero record boundaries).
- **Not triggered:** W4 (π post-hoc).
- **Obsolete (pre-W1 spec):** original W2 (per-head LR), W3 (loss weight as *up*-weighting — N2 now shows the down direction is the productive one), W4 (alternative targets).
- **Housekeeping pending the user:** triplicate dead definitions in `experiments/w2_decoders.py` (S2 artifact, left untouched under hard rule 2; correct body binds last so S2's numbers stand). Also uncommitted: `MultiResNet/train.py` (patched today) and `ASCAD_test_models.py` / `ASCAD_train_models.py` (earlier sessions).

## Related

- Task ids: T-030 (closed), T-028, T-024, T-019
- Sibling plan: `docs/tasks/T031-contrastive-learning-plan.md` (2026-10-08, three contrastive approaches) — **checked, not executed**, see `T031-reports/CL-0-premise-check.md`; **CL-Bridge (Stage 3) remains live** with a corrected carriage gate
- `docs/lab-log/todo.md` items: **T31-N4b** (DONE 2026-10-08 — adopt w=0.5), **seeded w=1 replicate** (the one missing measurement, needs go-ahead), **T31-N5/N6** (SOTA #2/#3, **not triggered** — the cascade stopped at the first pass; N5's plan block also needs rewriting before it could run), **T31-W4** (π, not triggered), **T31-C1** (housekeeping: dead triplicate definitions in `experiments/w2_decoders.py`, a T-030 artifact)
- Reports: `docs/tasks/T031-reports/` — `W1-c-diagnosis.md`, `W2-c-lut-replacement.md`, `T31-N1-joint-alpha-beta-decode.md`, `T31-N2-c-deletion-retrain.md`, `T31-N4-c-loss-weight-sweep.md`, `T31-N4b-c-loss-weight-replication.md`, `CL-0-premise-check.md`; running index in `LEDGER.md` (8 rows); template in its `README.md`
- Code: CL-0 — `experiments/cl_premise_check.py` → `experiments/out/cl_premise_check.json` (+ `.log`); dataset-premise test with permutation-null MI, coincidence-floor comparison and slot/byte identifiability, ~1 min CPU. N4b — `experiments/w10_replication_check.py` → `w10_replication.json` (the frozen rule applied to artifacts, paired by seed, written before any seeded arm existed), `logs/t031_n4b_train.sh`, and the `--seed` addition to `MultiResNet/train.py` (default `None` = previous behaviour; recorded in the sidecar and `<ckpt>.history.json`). N4 — `experiments/w9_gate_check.py` (judges the frozen gate from the eval JSONs), `w6_trainer_patch_tests.py` (20 checks), `logs/t031_n4_train.sh`, and the `--c-loss-weight` addition to `MultiResNet/train.py`. N1 — `experiments/w7_joint_ab_decode.py` (8 decoders + 2 sweeps), `w7_joint_ab_decode_tests.py` (33 checks), `w7_dissect_normalisation.py` (production-decoder audit). N2 — `experiments/w6_nochead_eval.py`, `w6_trainer_patch_tests.py` (8 checks), `w6_clip_binding.py`, `logs/t031_n2_train.sh`, and the additive patch to `MultiResNet/train.py` (`--no-c-heads`, `--save-last`, name→column target map, per-head validation history)
