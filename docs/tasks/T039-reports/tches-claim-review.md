# TCHES-oriented review of the CompactMultiResNet (T-039) claims

2026-10-18, from T-042 (user request: "等结果出来后帮我按TCHES梳理claim"). Stage C baselines
(MRN@400k + Compact tricks, Compact@400k plain, MRN@400k seed 2, GPAM + sub_avg) are still running;
sections marked **[C]** will be updated when they land.

## 1. The evidence base, on one protocol

Same 5,000 attack traces, same decoders (`compact_verify.py`), `ASCAD_PI_ROUTING=raw` (independent
π heads, MS23's own decoding), all 400k profiling traces, 15k extracted input.

| Model | Params | Train recipe | z (median) | **GE < 1, full key** (500 attacks, MS23's metric) | GE = 0 (all 500) | MTD worst byte |
|---|---|---|---|---|---|---|
| *MS23 MultiResNetSCA-1 (paper, Fig. 12a)* | 137.4M | 400k, 15 ep, batch 64, no preprocessing | — | *≈ 60* | — | — |
| MultiResNet, our reproduction, best | 137.4M | repo recipe: sub_avg, RAdam 1e-4, 15 ep | +14.21 | 75 | 235 | 47.4 |
| MultiResNet, our reproduction, last | 137.4M | same | +13.25 | 160 | 450 | 84.0 |
| **Compact s1 best** | 15.65M | repo recipe + w_C 0.5 + decoder loss | +14.44 | **60** | 165 | 35.8 |
| **Compact s2 best** | 15.65M | same | +14.41 | **60** | 170 | 37.0 |
| Compact s1 / s2 last | 15.65M | same | +14.37 / +14.28 | 70 / 75 | 190 / 175 | 40.9 / 41.7 |
| GPAM, lr 1e-3, Compact's exposure | 6.11M | GPAM paper except LR/epochs; no sub_avg | +8.65 | 1145 | 2475 | 599 |
| GPAM, paper config (target LR 5e-5) | 6.11M | GPAM paper, input 1M→15k | −0.24, 0/16 | — | — | — |
| *OvC (ePrint 2023/1615)* | — | randomness labels | — | *"TGE0 47" — definition not yet verified* | — | — |
| *GPAM (TCHES 2024, raw 1M)* | — | paper | — | *≈ 80 (paper); our raw reproduction 600* | — | — |

Head accuracies (MS23 Table 2 vs ours): α 99.2 / 99.94, β 21.1 / 19.42, c 1.6 / 1.51, π 88.9 / 88.77
→ our 137.4M run is a faithful MS23 reproduction. Compact best: α 99.8–99.9, β 23.3–23.4,
C 1.47–1.53, π 90.6–91.1. All decodes: shuffled-heads null |z| < 0.6, uniform C flattens 16/16,
oracle C → z ≈ +15.9 in 10 traces (C is the shared ceiling).
Sources: `GPAM/artifacts/crop34/{compare_400k.json, ge_ms23_style.json, *_verify_raw.json}`.

## 2. Claim-by-claim verdict

| # | Current claim (T-039 F1b) | Verdict | What to write instead / what is missing |
|---|---|---|---|
| 1 | 94.5% of MS23's weights sit in redundant heads; −88.6% params with zero degradation | **Keep — strongest claim** | Parameter audit is exact (7.53M trunk, 129.87M heads). Equivalence: 50k (177.5 vs 170 traces) and now 400k (GE<1: 60 vs 75). **Confound at 400k:** Compact also has w_C 0.5 + decoder loss; **[C]** MRN + same tricks and Compact plain settle it. MRN has 1 seed **[C]** |
| 2 | "Fastest full-key recovery ever reported, 37.5 traces, 1.6× faster than MS23 (~60)" | **Withdraw as stated** | (a) Under MS23's own metric Compact is **60 = MS23's ≈ 60**: no speed-up over the paper. (b) "37.5" is a single-ordering median at rank_step 10, not comparable to a 500-attack GE. (c) OvC reports 47 — unverified definition; if it is full-key GE, OvC is *faster*. (d) "Ever reported" needs a completeness argument we cannot give (DBLP/Scholar not fully searched). **Rewrite:** "matches MS23's reported ≈60 traces (GE < 1, 500 attacks) with 8.8× fewer parameters, and beats a same-pipeline 137.4M reproduction (75)". Fix also: "official 400k scale" — the paper used 400k; the released code uses 420k |
| 3 | Anti-bottleneck "principle": shared bottleneck kills low-SNR heads; unshared heads "strictly required" | **Keep as empirical finding, soften** | One architecture, 50k. Oracle-swap evidence is good. "Principle" / "strictly required" overclaim: GPAM shares one token representation and still carries C (gain +0.14, 16/16) once its LR is right. Say "a shared dense bottleneck collapsed the C carrier in our setting" |
| 4 | C is a probabilistic carrier; top-1 is misleading | **Keep — mechanistic core** | Now 5 models: Compact (C 1.5%, 16/16), MRN (1.5%, 16/16), GPAM-lr (0.93%, 16/16 but ≈19× slower), GPAM-paper and Variant A (≈ chance, 0/16). Caveat: gain on the C label is *not* monotone in attack strength (MRN last: C gain −0.48 yet z +13.25; T-024 MRN 50k: −0.09 yet 16/16) → do not use "gain over uniform" as the carrier measure; use a key-discrimination quantity (e.g. per-trace log-likelihood ratio of the true key after mask marginalisation) |
| 5 | Gradient sink; w_C = 0.5 optimal | **Demote to ablation** | Evidence: 2 seeds at 50k on 137.4M, 147.5 vs 155 traces — small effect, no CI. "Optimal" untested (only 0, 0.25, 0.5, 1). Keep the gradient-budget audit as motivation |
| 6 | Decoder loss regularises; "47.5 → 37.5 traces at 400k" | **400k part refuted (2026-10-18)** — Compact plain @400k: GE<1 **60**, GE=0 150 = Compact + w_C + L_dec (60 / 165); MRN + both losses 80 vs 75. Keep only the 50k ablation. Earlier note: | 47.5 is Compact plain at **300k**, 37.5 is Compact + w_C + decoder at **400k**: data size and two tricks change at once. 50k ablation (177.5 → 147.5, 2 seeds) is the valid one. **[C]** Compact plain @400k gives the 400k ablation |
| 7 | β labels unnecessary; α learning cliff | **Rewrite (2026-10-19) — as stated it is wrong in two ways** | (a) R2 drops the β **CE only**; C, α, π labels are still loaded and β = C ⊕ α⊗S[π] is computable from them, so no information was removed — it shows the decoder can replace the β *loss*, not the β *label*. A reviewer will say this in one line. Stronger: the C label alone fixes (α, β, π) (CPU check 2026-10-19: unique (α, β) in 3000/3000 synthetic instances), so **every ladder cell that keeps C — R1, R2, R3 — is loss-only**. (b) R2 exists only at 50k (2 seeds, 130 / 415 traces; same-seed re-run 335 → ~2.6× spread; fully labelled 175 / 510). **Write:** "a decoder loss on S replaces the β cross-entropy at 50k". The informationally clean cells (C dropped plus ≥ 1 of α/β/π: R4 β+C, α+C, π+C, R5 all) were unrun at any scale → E6-A. **[2026-10-20: first clean cell RCπ (α, β labelled; C and π CE dropped; decoder loss) at `[VALID 300k]`, 2 seeds: `best` (epoch 6) 16/16, works_v2, z +5.40 / +5.47, `MTD1` 1060 / 605 vs 45–50 fully labelled, `GE<1-500` 2250 (seed 1 only; fully labelled 80); nulls at chance. Write it as "C and π labels are not needed when α, β are labelled", not as a non-worst-case result. Limits: α, β still labelled; C and π heads at chance top-1, but C heads encode the unordered 16-value set (mass at the 16 true values −0.09 vs random −0.12 nats; own slot = other slot), so the attack marginalises slots uniformly; `last` (epoch 15) fails 5/16 and 4/16; early stopping required.]** (c) E3 discards unlabelled traces, so the α cliff confounds label count with trace count → E6-B. Still a good TCHES angle, but as an honest ladder with open cells, not as a bridge already built |
| 8 | Immune to evaluation pitfalls | **Keep, extend** | Add: worst-case labels C are uniform by construction (masked), so the label-prior false positive of ePrint 2026/338 cannot arise; the shuffled/uniform-C/constant-trace nulls are the evidence. Report all metrics side by side (§1) — this is itself a pitfall the paper can document (single-ordering vs 100- vs 500-attack GE differ ×3–4) |

## 3. New claims the T-042 data supports

- **N1. Robustness to checkpoint choice.** MRN degrades from best to last (GE<1 75 → 160, GE=0
  235 → 450, C gain +0.34 → −0.48; val loss rises from epoch 4); Compact barely moves (60 → 70/75).
  Clean, cheap, and directly tied to Claim 1 (excess head capacity memorises). Needs **[C]** MRN s2.
- **N2. Same-budget comparison against the TCHES-2024 transformer.** On identical data, exposure,
  attack set and decoder, GPAM needs ≈19× more traces (GE<1 1145 vs 60); at its published target
  LR it does not learn C at all (0/16). Must be stated with the LR caveat and **[C]** the sub_avg
  variant, and as "on the 15k extract" (GPAM was designed for raw 1M traces).

## 4. Missing before submission (priority order)

| Item | Cost | Status |
|---|---|---|
| Verify OvC's TGE0 definition (full key? how many attacks?) — ePrint blocks scripted download; fetch PDF by hand | 10 min, human | **open** |
| 400k trick confound: MRN + w_C 0.5 + decoder; Compact plain | GPU ~2.5 h | **[C] running** |
| MRN@400k second seed | GPU ~75 min | **[C] running** |
| GPAM with the ResNets' preprocessing (sub_avg) | GPU ~2 h | **[C] running** |
| Confidence intervals: bootstrap over attack orderings + seeds for GE<1 / MTD | CPU | open |
| Manual Google Scholar pass for 2024–2026 worst-case ASCADv2 papers (DBLP/S2 were blocked) | 30 min, human | open |
| Exact-marginal (T-041) numbers as a secondary "post-processing" column, clearly labelled | CPU, have it for best/last | done in `compare_400k.json` |

## 5. Suggested framing for TCHES

Lead with **where the parameters go and why it matters** (Claim 1 + N1), the **C-carrier mechanism**
(Claim 4, with the anti-bottleneck evidence of Claim 3 as its corollary), and the **label-ablation
ladder** (Claim 7, in its 2026-10-19 scope-limited form: which *losses* the decoder can replace, with the
informationally clean cells reported as open or measured) as the bridge to the non-worst-case problem
the community now cares about.
Report trace counts only under declared metrics (GE < 1 over 500 attacks to match MS23, plus GE = 0
and MTD), and present the "fewer traces" result as parity with MS23's paper and a 1.25× gain over a
same-pipeline reproduction — not as a new speed record.
