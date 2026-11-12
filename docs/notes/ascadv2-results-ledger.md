# ASCADv2 results ledger — current valid numbers, scale and metric tags

Created 2026-10-19. **Purpose:** one place to look a number up *before* quoting it. Task files
(`docs/tasks/T0xx-*.md`) are history plus plans and keep superseded text for the record; this note holds what
is currently believed. If a number is not here, read its source file; do not quote it from memory or from a
summary.

## 0. Two tags every number must carry

**Regime (scale).** 50k profiling traces is a **smoke run**: it checks that code and recipe work and says
nothing about what a method can do at the real data size. The real regime is **300k–500k**.

| Tag | Meaning | Use |
|---|---|---|
| `[SMOKE 50k]` | trained on profiling rows 0–49,999 (or less) | never as evidence for or against a method's capability; never as a comparison row against a 300k+ result |
| `[VALID 300k]` | profiling rows 0–299,999 (validation = last 5 % of that range) | primary evidence |
| `[VALID 400k]` | profiling rows 0–399,999 (MS23's volume) | primary evidence; the only scale with the MS23 comparison |
| `[LABEL-BUDGET]` | E3's N-axis (N mask-labelled traces, N ≤ 50k) | the *variable* of that study, not a smoke run; but N and the number of traces change together there (unlabelled traces were discarded) |

The rows matter, not just the count: the profiling traces drift over the 500k-trace run and the last ones look
most like the attack traces (`ascadv2-profiling-drift.md`). A 300k model never saw the rows closest to the attack
set; a 400k model saw some of them.

**Metric.** These are different quantities and must not be compared with each other:

| Tag | Definition | Typical value (Compact, 400k) |
|---|---|---|
| `MTD1` | median over the 16 bytes of traces to rank 0, **one** attack ordering ("median traces-to-rank-0" in the T-039 tables) | 35 / 40 (400k), 45 / 50 (300k) |
| `GE<1-500` | smallest n with guessing entropy < 1 on **all 16 bytes**, averaged over **500 random attack orderings** (MS23's metric) | 60 (400k) |
| `MTDw` | mean over random attack orderings of the traces until the **worst** byte reaches rank 0 (T-042 `compare_400k.json`; the review table calls it "MTD worst byte"; ordering count differs by source file — 100 in T-042's definition, 500 in the review's GE columns — check the file) | 35.8 / 37.0 (400k) |
| `z` | median per-byte z-score of the true key against the wrong keys | +14.4 (400k) |

"37.5 traces" is `MTD1` (35 and 40 for the two seeds); MS23's "≈ 60" is `GE<1-500`. The old "1.6× faster than MS23" compared the two and is withdrawn.

## 1. Valid results (300k–500k)

All with 5,000 attack traces (per-trace keys), worst-tie ranking, π routing noted where it matters.

| Result | Regime | Metric | Value | Seeds | Source |
|---|---|---|---|---|---|
| Compact B 15.65M, fully labelled | VALID 300k | MTD1 / z | 16/16; 50 and 45 traces; z +14.04 (both) | 2 | T-039 E1 (`experiments/out/t039/e1_*`) |
| Compact B + w_C 0.5 + decoder loss | VALID 400k | GE<1-500 | **60 / 60** (GE = 0 at 165 / 170); z +14.44 / +14.41; MTDw 35.8 / 37.0 | 2 | `T039-reports/tches-claim-review.md` §1, `GPAM/artifacts/crop34/compare_400k.json` |
| same models, T-039 400k benchmark | VALID 400k | MTD1 | 16/16; 35 / 40 traces (max 60); z +14.44 | 2 | T-039 "400k Benchmark" |
| Compact, last checkpoint | VALID 400k | GE<1-500 | 70 / 75 | 2 | same |
| MultiResNet 137.4M (our MS23 reproduction), best / last | VALID 400k | GE<1-500 | 75 / 160 (z +14.21 / +13.25) | 1 | same |
| MS23 paper (MultiResNetSCA-1) | VALID 400k | GE<1-500 | ≈ 60 (printed, Fig. 12a) | — | MS23, ePrint 2021/592 |
| GPAM 6.11M on the 15k extract, lr 1e-3 | VALID 400k | GE<1-500 | 1145 (z +8.65) | 1 | same |
| GPAM, paper config (LR 5e-5) | VALID 400k | z | −0.24, 0/16 (does not learn) | 1 | same |
| Compact plain @400k vs Compact + w_C + decoder | VALID 400k | GE<1-500 | 60 vs 60: the two tricks add nothing at 400k | 1 | review Claim 6 |
| Oracle head swaps, Compact 300k s2 | VALID 300k | MTD1 | as trained 45; true α 45; true β 20; true π 40; true C 10; all true 10; uniform π 445 (still 16/16); uniform C, uniform α+β: 0/16 | 1 | T-039 "Attack-time turn off a mask" |
| E6b R1 (drop π loss; C, α, β labelled) | VALID 300k | MTD1 / head | π learnable via decoder, both seeds (val π top-1 64.6 % / 66.9 %); attack **split**: s2 16/16, 105 traces (learned routing worth +6.0 z; uniform π 640, oracle π 55); s1 1/16, α collapses on late-profiling and early-attack rows | 2 | T-039 E6b |
| E6-A RCπ: α, β labelled; C and π CE dropped; decoder loss only | VALID 300k | MTD1 / z | `best` (epoch 6): 16/16, MTD1 **1060**, **GE<1-500 = 2250** (500 orderings, `Attack_traces[:5000]`; curve `experiments/out/t039_e6/e6_RCpiT300k_s1_best_ge_curve.png`), z +5.40, works_v2 True; heads α 99.98 %, β 17.1 %, C top-1 0.45 % (chance), π 6.29 % (chance); uniform π 630 traces, oracle π 2270 (worse); `last` (epoch 15) fails 5/16, z +2.45. **π not learned; passes only via early stopping; one seed** | 1 | T-039 "Results RCπ seed 1" |
| E6-A RCπ seed 2 | VALID 300k | MTD1 / z | `best` (epoch 6): 16/16, MTD1 **605** (max 3320), z +5.47 (min +4.53), works_v2 True; α 99.76 %, β 17.9 %, C top-1 0.45 %, π 6.43 % (chance); `last` fails 4/16. `GE<1-500` not computed | 1 | T-039 "RCπ seed 2 result" |
| RCπ C heads: what they encode (both seeds, `best`, 5,000 attack traces) | VALID 300k | mean log(256·P) per head | at the 16 true C values of the trace −0.090 / −0.093; at a random value −0.120 / −0.123; at the head's *own* slot's value −0.092 vs another slot's −0.090 / −0.097. Fully labelled E1 s2: own +0.31 vs other slot −0.75. So RCπ's C heads know the unordered set, not the slot; uniform routing is what works (own-slot oracle π is worse: 2270) | 2 | scratch analysis 2026-10-20 on `e6_RCpiT300k_s{1,2}_best_heads.npz` |
| E6-A RCπ-U: RCπ with the decoder routing fixed at 1/16 (`--uniform-pi`), scored with uniform π | VALID 300k | MTD1 / z | `best`: s1 16/16, `MTD1` 820, z +6.02; s2 16/16, `MTD1` 790, z +5.97; works_v2 True both; `last`: 14/16 and 15/16 (fail). Not faster than RCπ with uniform-π decode (630 / 430); `last` degrades less than RCπ (5/16, 4/16) | 2 | T-039 "RCπ-U result" |
| 3headResNet (α, β, ONE C head, no π heads; 8,264,096 params; α, β labelled, decoder loss, uniform routing) | VALID 300k | MTD1 / z | `best`: s1 16/16, `MTD1` 950, z +5.72; s2 16/16, `MTD1` 705, z +6.00; works_v2 True both. `last`: s1 14/16, z +3.23 (fail); s2 16/16, `MTD1` 1250, z +4.28, works_v2 True (the first `last` of the C-unlabelled runs that passes). Inside the spread of RCπ-U (820 / 790) and RCπ | 2 | T-039 "3headResNet seed 1 result", "3headResNet seed 2 result" |
| E6 R2 at 300k: β CE dropped; C (w 0.5), α, π labelled; decoder loss (loss-only cell, the C label fixes β) | VALID 300k | MTD1 / z | `best` (epoch 10): 16/16, `MTD1` 40, z +14.07, works_v2 True; `last`: 16/16, `MTD1` 40, z +14.06, works_v2 True. β head top-1 only 0.067 (`best`) / 0.121 (`last`) on the attack traces, yet the attack is as fast as fully labelled (50 / 55). One seed. The 50k R2 pass (§2) survives at 300k | 1 | T-039 "R2 at 300k result" |
| 3headA: 3headResNet with the β CE on in epochs 1–3 only (α labelled throughout; C, π never) | VALID 300k | MTD1 / z | `best` (selection restarted at the switch): s1 16/16, `MTD1` 890, z +5.53; s2 16/16, `MTD1` 885, z +5.07; works_v2 True both; `last`: 9/16 (z +3.06) and 11/16 (z +3.12), works_v2 False. Val β top-1 0.17 (epoch 3, CE on) → 0.11–0.12 (epoch 4) → 0.10 (epoch 15): partly kept, not held, both seeds | 2 | T-039 "3headA result" |
| 3headAB: 3headResNet with the α AND β CE on in epochs 1–3 only (afterwards only the decoder loss on S) | VALID 300k | MTD1 / z | `best` = `last`: 0/16, z +0.67, works_v2 False (fail). α val top-1 0.9999 → 0.005 in the first epoch without CE, β 0.173 → 0.008; val decoder NLL back at the uniform value 88.723. α's CE is a continuing anchor | 1 | T-039 "3headAB result" |
| 3headW: 3headResNet with the single C head widened 64 → 1024 (11.95M params; β CE all epochs) | VALID 300k | MTD1 / z | `best` (epoch 7): works_v2 True, 16/16, z +5.58, MTD1 1075 (3head controls 950 / 705); `last` 9/16, fail. Val decoder NLL min 88.662 (controls 88.635 / 88.627), final 89.817 (89.19 / 89.03): no gain, faster overfitting; C information unchanged (0.032) | 1 | T-039 "3headW seed 1 result" |
| Post-hoc bit-linear / HW projection of the trained C posteriors (offline decode variant, no retraining; α, β, π unchanged; nulls not run) | VALID 300k | MTD1 / z | `best`: 3head s1 950 → 725 / 620, s2 705 → 595 / 585, 3headW 1075 → 560 / 580; `last`: 3head s1 1470 → 1045 / 960, s2 1250 → 1430 / 810, 3headW 2040 → 845 / 1155; all `last` reach 16/16 except s2 HW 15/16; z up in all 12 | 2 (+3headW) | T-039 "Offline estimate for a bit-linear / HW C head" |
| 3headBit: 3headResNet with a bit-linear C head (logit(v) = Σ bit_i(v)·w_i, trained this way; β CE all epochs) | VALID 300k | MTD1 / works_v2 | `best` (epoch 7): 16/16, z +6.16, MTD1 910, works_v2 True; `last`: 16/16, z +4.54, MTD1 1485, works_v2 True. Val decoder NLL min 88.624, final 88.967 (3head 89.19 / 89.03). Slower overfitting, no `best` gain (3head 950 / 705; post-hoc bit projection of 3head 725 / 595) | 1 | T-039 "3headBit seed 1 result" |
| Same projection on all cached 300k checkpoints, with works_v2 nulls (projected constant-trace null) + structure-free control (C log-probs × one factor matched to the bit projection's spread) | VALID 300k | MTD1 / works_v2 | Labelled C (E1, R2): unchanged (E1 50 / 45 → bit 45 / 60). Unlabelled C `best` (bit / HW): RCπ 1060 / 605 → 490 / 480, 575 / 435; RCπ-U 820 / 790 → 850 / 515, 765 / 480; 3headA 890 / 885 → 810 / 835, 865 / 940; all pass works_v2. Failing `last`: bit repairs 7 of 9, matched temperature 0 of 9; bit beats matched temperature on 18 / 20 paired MTD1. R1 s1 (wrong π routing): 1/16 → temperature 15/16 (880) → bit 16/16 (105). 3headAB stays 0/16. Projection chosen after the structure diagnostic on the same attack traces. **For citing, use the held-out row below** | 1–2 per model | T-039 "(b) result", "Temperature control result" |
| **Held-out check of the bit projection** (pre-registered 2026-11-01; attack traces **5000–9999**, unused for any projection decision; bit primary) | VALID 300k | MTD1 / works_v2 | Unlabelled C (18 checkpoints, `best` + `last`): bit better than original **15 / 2 / 1** (better / worse / tie), better than matched temperature 15 / 2 / 1; original fails 8 (all `last`), bit repairs **7**, temperature 0. `best` only: 7 / 9 better, median `MTD1` 540 → 515 (original `best` on this half: RCπ 1040 / 670, RCπ-U 425 / 540, 3head 445 / 485, 3headA 405 / 685, 3headW 795); `last`: 8 / 9 better, 1 tie (RCπ s1 stays 15/16). Labelled C (E1, R1 s2, R2; 8): median bit / original ratio 1.05, all pass. R1 s1 (wrong routing): 6/16 → temperature 16/16 (790) → bit 16/16 (100). 3headAB 0/16 under every transform; 69 projected copies pass plain decoding, none fails a null. H1–H5 all pass. HW (secondary) 12 / 4 / 2 vs original. Original `best` `MTD1` moved up to 2× between the two attack halves, so single-half `best` gains are within noise; the robust effect is on overfitted `last` checkpoints | 1–2 per model | T-039 "Hold-out result"; `experiments/out/t039_e6/holdout/summary.txt` |
| Oracle α, β with the model's C heads, π routing varied (E1 s1/s2, RCπ s1/s2, RCπ-U s1; `best`) | VALID 300k | MTD1 | E1: as trained 50 / 55, α,β true 20 / 20, α,β true + uniform π 185 / 155; RCπ: α,β true + uniform π 365 / 380, α,β,π true 1950 / 1200; RCπ-U s1: α,β true 335, α,β,π true 700. π is worth ≈ 8–9× with slot-specific C, negative with set-only C | 1–2 | `experiments/out/t039_e6/ab_oracle_c4_e1_rcpi_rcpiu.json`; T-039 |
| T-034 direct black-box S classifier (shuffle ON and known-π arms) | VALID 300k | rank / z | chance, after the S-box-index fix | 1 | T-034 |

**Head accuracy by scale (what a head *can* reach — never quote one without its scale):**

| Head | `[SMOKE 50k]` Compact | `[VALID 300k]` Compact | `[VALID 400k]` Compact best | MS23 Table 2 (400k) |
|---|---|---|---|---|
| α top-1 | 99.9 % | ≥ 0.992 (E1 s2, every row range tested; E6b note) | 99.8–99.9 % | 99.2 % |
| β top-1 | **5.5–8.1 %** | **20.1–20.6 %** | **23.3–23.4 %** | 21.1 % |
| C top-1 | not checked | not checked | 1.47–1.53 % | 1.6 % |
| π (slot-wise, π element) | 70.6–72.5 % | not tabulated for E1; the labelled π heads of the E6b comparison reach ≈ 80 % val (E6b note) | 90.6–91.1 % | 88.9 % |

Sources: T-039 E3 table (50k column), E1 table (300k β), E6b notes, review §1 (400k). The β row is the one that was misquoted: **"β ≈ 6 %" is the 50k figure.** "Not checked" means I have not found the number in a source file — fill in from the head-accuracy JSONs before quoting.

## 2. `[SMOKE 50k]` and `[LABEL-BUDGET]` results (do not carry over to 300k+)

| Result | Value | Source | What it can and cannot say |
|---|---|---|---|
| Compact vs MultiResNet, 50k | 177.5 vs 170 traces (MTD1) | T-033 | recipe sanity only; the valid equivalence is the 400k row above |
| w_C 0.5 vs 1.0, MultiResNet, 50k | 147.5 vs 155 (MTD1) | T-031 N4b/N4c | small effect, no CI; the "optimal w_C" claim is demoted |
| E6 R0 (all labels + decoder), 50k | 135 / 160 traces vs 175 / 510 without decoder | T-039 E6 | decoder loss does not hurt at 50k; not a 400k result |
| E6 R2 (β loss dropped), 50k | 16/16 both seeds, 130 / 415 traces; same-seed re-run 335 | T-039 E6 | the β *loss* can be replaced by the decoder at 50k; labels C, α, π still loaded (see §3) |
| E6 R1 (π loss dropped), 50k | 1/16, 3/16 (fails) | T-039 E6 | smoke figure; do not cite as "π is hard". At 300k π is learnable (§1) |
| E3 label-efficiency curve | 0/16 at N = 1k, 5k, 20k; 16/16 at N = 50k; α top-1 0.34–1.2 % at 20k, 99.9 % at 50k | T-039 E3 | `[LABEL-BUDGET]`; unlabelled traces were discarded, so labelled count and trace count are confounded |
| MultiResNet 137.4M, 34 heads + sub_avg, 50k | 16/16, z +9.50, 170 traces (MTD1); seeded w = 1 replicate z +9.48 | T-024, T-031 N4c (T-039 F1 table) | recipe sanity; scale of T-024's own per-byte z range (+3.9…+6.9) not re-checked |

## 3. Withdrawn or corrected — do not repeat these

| Statement | Status | Correct statement | Where it is fixed |
|---|---|---|---|
| "37.5 traces, 1.6× faster than MS23 (≈ 60)", "fastest ever" | **Withdrawn 2026-10-18** | Compact@400k `GE<1-500` = 60 = MS23's ≈ 60 (parity); 1.25× better than our 75; 8.8× fewer parameters | review Claim 2; T-039 header |
| "Decoder loss: 47.5 → 37.5 traces at 400k" | **Refuted 2026-10-18** | 47.5 was Compact plain at *300k*, 37.5 Compact + tricks at *400k* (scale and tricks changed together); plain vs tricks at 400k: 60 vs 60 | review Claim 6 |
| "β labels are unnecessary" / "mask labels dropped entirely" | **Rewritten 2026-10-19** | R2 drops only the β cross-entropy; C, α, π labels stay loaded and the C label alone fixes (α, β, π) (`experiments/t039_c_determines_masks.py`), so every cell that keeps C removes a loss, not information. 50k only | Claim 7 |
| "R3 (drop α) is an informationally clean cell" | **Retracted 2026-10-19** | clean = C dropped **and** ≥ 1 of α, β, π dropped | T-039 E6-A |
| "R4/R5's low prior is stale because R1 passed at 300k" | **Retracted 2026-10-19** | π is not one of N1's three shares; R1 had α, β, C supervised. R1 says nothing about two uniform shares | T-039 E6-A |
| "β is weak, top-1 ≈ 6 %" applied to 300k | **Corrected 2026-10-19** | 6 % is 50k; β is 20.5 % at 300k, 23.4 % at 400k | §1 above |
| "α-only loaded is the most promising clean cell" | **Retracted 2026-10-19** | α loaded with β, C uniform has zero decoder gradient (`experiments/t039_n1_gradient_map.py`) | T-039 E6-A |
| "E3: the α cliff is a label-count effect" | **Not established** | labelled count, trace count (and, below 50k, epochs) change together; `[LABEL-BUDGET]` | Claim 7 |
| "R10's 5.3e-5 nats/trace shows non-worst-case is walled off" | **Corrected 2026-10-12** | a lower bound for a one-sample-per-leak Gaussian reader, not a bound on the data | T-039 F3 |
| "T-034 / T-011 key-rank claims" | **Withdrawn 2026-10-09** | S-box index typo (167 → 0x5C), trace-0 key on a variable-key set; corrected rerun is chance | T-034 banner |
| "MultiResNet already absorbs the information in the windows" (general) | **Rescoped** | shown for β only (T-043); not for α, C, π | T-043 |
| "Anti-bottleneck principle, unshared heads strictly required" | **Softened** | a shared 512-d dense bottleneck collapsed the C carrier in our setting; GPAM shares one token representation and still carries C at the right LR | review Claim 3 |
| "w_C = 0.5 is optimal" | **Demoted** | only {0, 0.25, 0.5, 1} tested, 50k, two seeds | review Claim 5 |
| "OvC reaches TGE0 = 47" as a comparison row | **Unverified definition** | do not tabulate until the PDF is read by hand | review §4 |
| "Post-hoc bit projection: `best` `MTD1` ≈ 0.5–0.9×" (attack traces 0–4999) | **Magnitude not confirmed 2026-11-02** | held-out (5000–9999): `best` 7 / 9 better, median 540 → 515, within the half-to-half noise of `best` `MTD1` (up to 2×); the robust effect is on overfitted `last` checkpoints (7 of 8 failures repaired). Cite the held-out row only | §1 held-out row; T-039 "Hold-out result" |

## 4. Citing rule

1. Look the number up here, or in its source file, before writing it.
2. Write it with its regime tag and its metric tag, in the same sentence.
3. Do not put a `[SMOKE 50k]` number in a table next to a `[VALID …]` number.
4. When a statement is withdrawn, add it to §3 *and* fix the sentence in every file that still contains it
   (`grep` the old wording).
