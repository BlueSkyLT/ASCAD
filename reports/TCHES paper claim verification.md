# Primary sources confirm the numbers, not the framing

The draft's prior-work numbers survive a check against the primary sources. MultiResNetSCA-1 has 137,396,064 trainable parameters, MS23 recover the full key in about 60 traces, and MS23 Table 2 reports head accuracies of 99.2 / 21.1 / 1.6 / 88.9. GPAM needs roughly 80 traces on raw 1M-sample traces, and its Table 1 hyperparameters are what the draft says they are. The decoder equation, the fixed-key simulation and the dataset geometry are all correct. What fails is the framing around those numbers, in five places. (1) The 500k/10k extracted split comes from the ANSSI release, not from MS23, whose paper uses 700k/20k/50k. (2) MS23 never calls its setting "worst-case" or "first threat", and its "First Threat Scenario" is a different, partial attack. (3) MS23 never states "GE<1 on all bytes" for its 60 traces. (4) "The worst-case reference model has not been revisited" is partly contradicted by UNIDLE/HierNet. (5) GPAM's "published learning rate" was paired in the draft with an optimizer that the released code runs at 10× higher rates. The open OvC question is settled in the draft's favour: OvC's TGE₀ = 47 is for **the first key byte only**, at GE = 0 over 100 attacks, so it is not a full-key record and does not beat 60. The internal review suspected that `DDPM24` was a mis-citation. **It is not.** "Diffuse Some Noise" does run Wu et al.'s non-profiled collision attack on ASCADv2 with shuffling disabled; the sentence only needs to credit the denoising rather than the attack, and the entry should move to its SAC 2025 version. The bibliography needs nine fixes to existing entries and about a dozen must-add works, led by Bronchain and Standaert's 2020 dissection of the same ANSSI implementation. Without these, a TCHES reviewer will say the paper's general points (small models suffice, accuracy misleads, multi-task heads compete) are already known.

## MS23's numbers hold; three attributions to MS23 do not

Every quantitative MS23 fact in the draft checks out against the ePrint version. The campaign has 800,000 traces of 1M samples on an STM32F303RCT7 (Cortex-M4) ([MS23 ePrint §2.1, pp. 8–9](https://eprint.iacr.org/2021/592)). The masked label is MS23's c[i] = rm × Sbox[pt[p[i]] ⊕ k[p[i]]] ⊕ rout (Eq. 6, p. 16; §4.3, p. 19), so the draft's α is MS23's rm and β is rout. The ANSSI labelling code builds the same slot-indexed `sbox_masked` field ([ANSSI `ASCAD_generate.py`](https://github.com/ANSSI-FR/ASCAD/blob/master/ASCAD_generate.py)). MS23 state the **137,396,064 trainable parameters** verbatim (§3.2, p. 15). The draft's split into a 7,525,472 trunk, 129,870,592 heads and 124,815,360 first dense layers is consistent with that total: the same per-head arithmetic reproduces MS23's own 11,460,960 (single-branch) and 78,364,256 (18-head) figures exactly. The architecture matches both MS23's Figs. 16–17 and `resnet_v1` in the official code: kernel 11, nine stacks, stride 2 from stack 2, 16→256 channels, AvgPool(4), a 3,584-wide flatten, and Dense(1024)→ReLU→BN→Dense(K) heads ([ANSSI `ASCAD_train_models.py`](https://github.com/ANSSI-FR/ASCAD/blob/master/ASCAD_train_models.py)). Further confirmed in the ePrint ([MS23 ePrint](https://eprint.iacr.org/2021/592)):

| Draft fact | Where MS23 states it |
|---|---|
| About 60 traces for the full key | §4.3, p. 20 |
| Head accuracies 99.2 / 21.1 / 1.6 / 88.9 | Table 2, p. 20 |
| GE averaged over 500 attacks | §1.2, p. 5 |
| Decoder with no slot renormalisation | Eq. 12, p. 20; App. G, Eq. 23 |
| Fixed-key simulation via p ⊕ k with correct key 0 | Eq. 7, p. 16; §4.3, p. 19 |

**Attribution 1: the 500k/10k split is not MS23's.** The draft's background presents the 500k profiling / 10k attack split under a `\cite{MS23}`. MS23's paper instead says its "final extracted dataset is made of up to 700,000 training traces, 20,000 validation traces and 50,000 test traces" (§2.2, p. 11). Its GE estimates sample attack sets "among a test set of 50,000 traces" (App. A, p. 27). The 500k/10k split, with the windows [205k, 210k) and [455k, 465k), comes from the released `example_generate_params` and the HDF5 on data.gouv ([ANSSI generate params](https://github.com/ANSSI-FR/ASCAD/blob/master/STM32_AES_v2/example_generate_params); [data.gouv ASCADv2](https://www.data.gouv.fr/en/datasets/ascadv2/)). The draft should cite the release for it. This has a substantive consequence: **MS23's 60 was computed on a different, larger test set than the 5,000 attack traces the draft uses**. "Same number of traces as the original paper" is therefore a protocol match, not a data match.

**Attribution 2: MS23 do not call their setting "worst-case" or "first threat".** MS23 call the full-key setting the "First Scenario" (§4.3, p. 19; Fig. 12a caption) and describe it as a "trade-off" (Contributions, p. 4). Their paragraph titled "First Threat Scenario" (§4.1, p. 16) is the *partial* attack in which masks and permutation are known even during the attack. "Worst-case" appears in MS23 only as a general description of open-sample profiling (Intro, p. 3) ([MS23 ePrint](https://eprint.iacr.org/2021/592)). GPAM is where the "First Threat Scenario" label for the full-key attack comes from: GPAM writes that it replicates MS23's "First Threat Scenario" ([GPAM §7.1, p. 492](https://tches.iacr.org/index.php/TCHES/article/view/11685)). The draft should drop "first threat" and attribute "worst-case" to the follow-up literature.

**Attribution 3: MS23 never state "GE<1 on all bytes" for their 60.** MS23 say that "the attack requires approximately 60 attack traces to retrieve the correct full key". That number is read from 16 per-byte mean-rank curves on a linear 0–140 axis. τ = 1 appears only in the general definition of N_a(τ) (App. A, p. 26) ([MS23 ePrint](https://eprint.iacr.org/2021/592)). The background's "is read off such curves" is accurate. The abstract's parenthesis and results.tex's "the metric of [MS23] (GE<1 on all bytes, 500 attacks)" overstate what MS23 make explicit.

**Recipe and rank-convention gaps the draft should disclose.** MS23 train with **Adam at the Keras default learning rate (10⁻³) on raw int8 traces**. The released script trains on the first 420k traces with Keras `validation_split=0.05`. Keras takes the validation split from the *last* 5%, so in practice it trains on the first 399k and validates on the next 21k. It uses early stopping (patience 10, restore best weights, at most 60 epochs) ([ANSSI train params](https://github.com/ANSSI-FR/ASCAD/blob/master/STM32_AES_v2/example_train_models_params)). The draft's RAdam at 10⁻⁴ with mean-trace subtraction is therefore a deviation and should be named as one. MS23's "stops after 15 epochs" is the number of epochs *run*. With patience 10, the restored model would come from about epoch 5. This is an inference, but it fits the draft's own finding that the 137.4M network's validation loss bottoms out at epoch 4. It also means MS23's 60 is comparable to the draft's "best" checkpoint, not its "last". MS23's rank definition (Eq. 15, strict ">") and their code (first matching index) both count **ties in favour of the correct key**, the opposite of the draft's convention ([ANSSI `ASCAD_test_models.py`](https://github.com/ANSSI-FR/ASCAD/blob/master/ASCAD_test_models.py)). The ANSSI README reports a third figure, "approximatively 120 test traces" for the released checkpoint on a single ordering ([ANSSI README](https://github.com/ANSSI-FR/ASCAD/blob/master/STM32_AES_v2/Readme.md)). That is a ready-made illustration for the pitfalls section that the same authors' numbers differ by about 2× between protocols.

One locator risk remains. The JCE version (13(2):129–147) has a different title and could not be read ([Crossref](https://api.crossref.org/works/10.1007/s13389-023-00311-7)). Every "Eq. 12", "Table 2" and "Fig. 12a" in the draft is verified only against **ePrint 2021/592**. Either cite the locators as ePrint locators or check the JCE PDF before submission.

## GPAM's 80 and OvC's 47 come with fine print that changes the comparison

GPAM's headline claims are verified. It uses "all 1,000,000 points of each trace without preprocessing" and replicates MS23's worst-case setting. It predicts c[i], r_m, r_out and p[i], which makes 34 heads, although the paper never prints the number 34. It needs "roughly 80 traces to have GE of all key bytes under 1" ([GPAM §7.1, p. 492](https://tches.iacr.org/index.php/TCHES/article/view/11685)). Three details flatter that 80, and the draft should state them:

- **Each c[i] head is the best of seven training runs**, chosen by validation accuracy.
- **r_m is used as a hard value** rather than as a distribution.
- **GE is estimated from 10,000 samplings** of an 80,000-trace holdout, after training on 640,000 traces (Table 3, pp. 484–485).

The 80 appears only in the text; there is no GE figure for ASCADv2. "Without preprocessing" is mostly true, but GPAM's Fig. 1 begins with "Normalize, reshape", and the code applies a global min–max rescale to [−1, 1] ([scaaml `gpam.py`](https://github.com/google/scaaml/blob/main/scaaml/models/gpam.py)). The draft's mean-trace subtraction is therefore genuinely extra relative to GPAM.

The setup paragraph's reading of GPAM Table 1 is verified: batch 64, 1,000 steps per epoch, 150 epochs, target learning rate 5·10⁻⁵, merge filters 0/0, and an unspecified warm-up followed by cosine decay (Table 1, p. 480; §5.3.1, p. 481). It is also true that the paper names neither the optimizer nor the warm-up length. Two statements need correcting:

- **The √L patch rule.** The draft says it follows the paper's √L rule. GPAM does state that rule (§5.3, p. 481), but **its own ASCADv2 patch size is 400, not √10⁶ = 1000**, so the paper does not follow its rule on this dataset.
- **"The published learning rate."** The intro and results claim that GPAM "at its published learning rate … does not learn the masked S-box". The released scaaml code compiles with **Adafactor at a constant learning rate** and says the warm-up/cosine schedule "is not necessary". Every released ECC config uses **10× the Table 1 rate** (0.006 vs 0.0006; 0.003 vs 0.0003), and there is no ASCADv2 config ([scaaml `gpam_ecc_cm1.py`](https://github.com/google/scaaml/blob/main/papers/2024/GPAM/gpam_ecc_cm1.py); [scaaml configurations](https://github.com/google/scaaml/tree/main/papers/2024/GPAM/configurations)).

By analogy with those configs, Table 1's 5·10⁻⁵ under Adafactor may correspond to an original optimizer that was never disclosed. This is an inference, but it is strong enough that the claim must be qualified rather than stated as a property of GPAM. GPAM publishes no parameter count, so the draft's 6.11M cannot be compared with a published figure.

OvC resolves cleanly. Its TGE₀ is the smallest number of traces at which GE = 0, averaged over **100 experiments** ([OvC §3.4.2, p. 18](https://web.archive.org/web/2023id_/https://eprint.iacr.org/2023/1615.pdf)). On ASCADv2, "our analysis targets the first key byte". It uses the 15k extract, a 90/10 split of the 500k profiling traces and 10k attack traces (§4.1.1, p. 19), with additive mask, multiplicative mask and permutation labels during profiling (§4.2, p. 21). OvC's Table 1 sets its single-byte 47 next to "60 [MS23]", which is copied from MS23 and is a full-key figure (p. 21). **The 47 therefore neither beats nor matches 60. The two are not comparable.** OvC is still an unpublished preprint, last revised 2024-01-16 ([ePrint 2023/1615](https://eprint.iacr.org/2023/1615)). The 2022 ACNS-W paper "Order vs. Chaos: Multi-trunk Classifier" is a different paper ([Springer](https://link.springer.com/chapter/10.1007/978-3-031-16815-4_13)). The like-for-like comparison would be the draft's own byte-0 traces to GE = 0 over 100 orderings, which has not been computed.

## Related work overstates novelty in two sentences and misses five ASCADv2 papers

The non-worst-case thread is verified, with one precision to add. Scoop claims "the first non-worst-case attack on the ASCADv2 dataset", but it recovers **one key byte**, reaching GE 0 after about 250k–300k attack traces carved from the profiling file ([Scoop, TCHES 2025(3)](https://doi.org/10.46586/tches.v2025.i3.56-80)). "Is it Really Broken?" (ePrint 2026/338, preprint, five authors) shows that the result is a false positive. With the MLP's first layer zeroed, GE still converges. The targeted 4th byte fails a χ² uniformity test, and its Lemma 1 shows that the posterior distinguisher equals maximum likelihood only under a uniform label prior ([ePrint 2026/338](https://eprint.iacr.org/2026/338)). Karayalcin, Krček and Picek's SoK independently notes that the Scoop attacks "were not valid due to an inconsistency in the dataset" ([ePrint 2025/1309](https://eprint.iacr.org/2025/1309)). **ReallyBroken never says that masked, uniform labels are immune.** The draft's pitfalls sentence ("cannot arise here") is the draft's own corollary of that Lemma 1 and must be written in the draft's voice. The draft's shuffled-posterior null and uniform-C control correspond to ReallyBroken's "Null Benchmark" and input-ablation checks, and citing that correspondence strengthens the section.

**DDPM24 is correctly cited.** The internal review's suspicion was reasonable, because "Diffuse Some Noise" is a denoising paper. Its §5, however, runs "the collision attacks as described by Wu et al." on ASCADv2 "with shuffling disabled". It reaches GE<1 on all key deltas with 60k diffused traces, while three deltas remain unrecovered at 100k raw traces: the abstract's 40% reduction ([SAC 2025 pre-proceedings](https://www.sacworkshop.org/SAC25/preproceedings/sac2025-2-paper1.pdf); [ePrint 2024/966](https://eprint.iacr.org/2024/966)). No other Karayalcin paper attacks ASCADv2. The sentence's only flaw is that "Karayalcin et al. mount non-profiled collision attacks" credits them with the attack instead of the denoising. The entry is out of date: the paper is now **SAC 2025, LNCS 16207, pp. 724–752, DOI 10.1007/978-3-032-10536-3_27** ([Crossref](https://api.crossref.org/works/10.1007/978-3-032-10536-3_27)). Wu et al. are verified as a non-profiled collision attack with shuffling disabled that recovers all key deltas at about 70k traces (SAC 2023, LNCS 14201, pp. 82–96) ([ePrint 2023/724](https://eprint.iacr.org/2023/724); [Crossref](https://api.crossref.org/works/10.1007/978-3-031-53368-6_5)). Marquet and Oswald (COSADE 2024) are verified as mask-label-free with shuffling disabled. They did, however, pick mask points of interest by SNR on the masks, and they call it a "simplified scenario". Their best model recovers the full key in 16 traces (mean 17.6) on half the seeds ([ePrint 2023/006](https://eprint.iacr.org/2023/006)). Gammell and Roy are verified: arXiv:2608.30105, OPTIMIST '26 at CHES 2026, on uncropped ASCADv1f/v1r and CHES-CTF-2018, with no ASCADv2 ([arXiv](https://arxiv.org/abs/2608.30105)). Maghrebi 2020/436 is the right source, and MS23 say they "revisit and leverage" it ([ePrint 2020/436](https://eprint.iacr.org/2020/436)).

Two sentences overreach:

- **"No newer worst-case full-key result" is essentially true, with two exceptions to name.** Marquet and Oswald's superseded ePrint 2023/006 v1 claimed about 24 traces with "full knowledge about the randomness during training" ([Wayback 2023-03 copy](http://web.archive.org/web/20230326015100/https://eprint.iacr.org/2023/006.pdf)). Their AIHWS 2023 paper reports about 21 traces with rm and π known in *both* profiling and attack ([ePrint 2023/611](https://eprint.iacr.org/2023/611)).
- **"The worst-case reference model itself has not been revisited" is partly contradicted.** UNIDLE/HierNet discusses the "capacity bottleneck" of multi-task trunks whose parameters scale as O(d²). It beats MultiResNetSCA-2 "with model sizes nearly one-tenth", but on the authors' own Arduino dataset, not on ASCADv2 ([ePrint 2024/1437](https://eprint.iacr.org/2024/1437)).

No paper re-audits MultiResNetSCA-1 on ASCADv2, so a narrower claim stands.

The related-work section is also missing five ASCADv2 papers:

- **Bronchain and Standaert (TCHES 2020(2))** dissect exactly this ANSSI affine-masked, shuffled AES and recover the full key with fewer than 2,000 measurements. They remark that some steps are "hard to automate without implementation knowledge (even with machine learning tools)" ([doi:10.46586/tches.v2020.i2.1-25](https://doi.org/10.46586/tches.v2020.i2.1-25)).
- **Masure et al. (TCHES 2023(1)), scheme-aware modelling.** Their non-worst-case model failed on ASCADv2, while "the same model trained in a worst-case scenario could get a positive PI" ([ePrint 2022/493](https://eprint.iacr.org/2022/493)).
- **Marquet and Oswald (AIHWS 2023)**, the ~21-trace result above.
- **DS-NSCA (TCHES 2026(2)).** Non-profiled DL fails within 500k traces ([doi:10.46586/tches.v2026.i2.659-685](https://doi.org/10.46586/tches.v2026.i2.659-685)).
- **UNIDLE**, above.

## The general claims are anticipated, so frame them as consistent with prior work

Four of the draft's general messages already have TCHES precedents. A reviewer will name them if the draft does not.

- **"Big SCA models are over-parameterised."** Zaid et al. built CNNs "up to 31810 times" less complex and concluded that SCA CNNs "do not need to be very complex" ([TCHES 2020(1)](https://doi.org/10.46586/tches.v2020.i1.1-36)). Wouters et al. cut a further 52% of parameters ([TCHES 2020(3)](https://doi.org/10.46586/tches.v2020.i3.147-168)). Rijsdijk et al. and Perin et al. found small optimal models by search ([TCHES 2021(3)](https://doi.org/10.46586/tches.v2021.i3.677-707); [TCHES 2022(4)](https://doi.org/10.46586/tches.v2022.i4.828-861)). The counterweight is Karayalcin et al., who argue that deep ResNets help on long inputs ([Mathematics 2023](https://doi.org/10.3390/math11153265)). That fits the draft's "keep the trunk, slim the heads". The draft's novelty is narrower and real: no source audits trunk-versus-head parameters in a *multi-task* SCA network or ties that excess to checkpoint fragility.
- **"Top-1 accuracy is the wrong measure."** This is long established. Picek et al. found accuracy "can be extremely deceptive" ([TCHES 2019(1)](https://doi.org/10.46586/tches.v2019.i1.209-237)). Masure et al. linked NLL to perceived information ([TCHES 2020(1)](https://doi.org/10.46586/tches.v2020.i1.348-375)). Zhang et al. proposed CER ([TCHES 2020(3)](https://doi.org/10.46586/tches.v2020.i3.73-96)). Ito et al. showed that CE/PI is "non-calibrated" with respect to success rate ([TCHES 2022(4)](https://doi.org/10.46586/tches.v2022.i4.228-254)). Ito et al. directly support the draft's observation that a negative C-head log-likelihood gain can coexist with key recovery. MS23 itself reports a 1.56% accuracy that still attacks. What is new is the *causal* oracle/uniform head-swap attribution inside a multi-task decoder.
- **"A shared bottleneck suppresses the low-SNR heads."** This is a concrete instance of task competition under hard sharing (Standley et al., [ICML 2020](https://proceedings.mlr.press/v119/standley20a.html)). Marquet and Oswald warned that "losses can compete as much as they can collaborate". Their positive result for *low-level* sharing does not contradict the draft's negative result for a *late* dense bottleneck, because the settings differ. Xin et al. and Kurin et al. (NeurIPS 2022) show that tuned scalarisation is rarely beaten, which justifies a plain w_C sweep over PCGrad or GradNorm.
- **Walsh–Hadamard XOR convolution is standard SASCA tooling.** SCALib's belief-propagation XOR factor runs in the Walsh–Hadamard domain, which was verified in its source code ([SCALib `belief_propagation.rs`](https://github.com/simple-crypto/SCALib/blob/main/src/scalib_ext/scalib/src/sasca/belief_propagation.rs)). Ueno et al. frame higher-order SCA as XOR-convolution ([TCHES 2025(3)](https://doi.org/10.46586/tches.v2025.i3.317-360)). MS23 contain no WHT, so the WHT evaluation is the draft's implementation choice and belongs to the SASCA tradition.

The scheme-aware model of Masure et al. anticipates the general idea of building the masking algebra into training. The ladder section must distinguish itself: it uses an explicit affine-plus-routing decoder loss, gives a gradient analysis, and keeps α/β labelled. For metric reporting, Wu, Perin and Picek recommend reporting dispersion and the median rather than the mean GE ([COSADE 2022](https://doi.org/10.1007/978-3-030-99766-3_3)). That makes the draft's 5× spread a quantified instance of a known concern rather than a discovery. The tie-handling pitfall has **no published source**: it is the draft's own caution and should stay uncited. Checkpoint fragility sits next to SCA early-stopping work: Robissout et al. ([COSADE 2020](https://doi.org/10.1007/978-3-030-68773-1_10)), Perin, Buhan and Picek ([COSADE 2021](https://doi.org/10.1007/978-3-030-89915-8_3)), and ensembles from Perin et al. ([TCHES 2020(4)](https://doi.org/10.46586/tches.v2020.i4.337-364)). The draft's contribution is architecture-level and complements a better stopping rule rather than replacing one.

## Section-by-section verdicts with replacement text

Verdicts: **V** = verified; **R** = verified in substance but needs rewording or a qualifier; **C** = contradicted by the cited source; **Own** = the draft's own result or inference, so it must not be attributed to prior work. "ePrint" means MS23 ePrint 2021/592 printed pages. Line numbers refer to `paper/tches/sections/*.tex` as of 2026-11-02.

### Abstract (abstract.tex)

| Line | Claim | Verdict | Source location | Change |
|---|---|---|---|---|
| 1–3 | 137.4M-parameter MultiResNetSCA-1 recovers the full key in about 60 traces with mask and permutation labels in profiling | V | ePrint §3.2 p. 15 (137,396,064 trainable); §4.3 p. 20; Fig. 12a; Table 2 | None |
| 6–7 | "the same number of traces as the original paper (60, guessing entropy below 1 on all 16 bytes over 500 attacks)" | R | ePrint §4.3 p. 20 ("approximately 60"); App. A p. 27 (500 attacks from a 50k test set); τ=1 not tied to the 60 | → "reaches the trace count reported for the original network (60; in our evaluation, guessing entropy below 1 on all 16 bytes over 500 attack orderings)" |
| 10–12 | GPAM needs 9× more traces at equal budget | Own; R | GPAM Table 1 p. 480; scaaml Adafactor configs | Keep, but the setup must carry the optimizer/LR caveat (see setup 20–27) |

### Introduction (intro.tex)

| Line | Claim | Verdict | Source location | Change |
|---|---|---|---|---|
| 4–5 | DL-SCA converged on public benchmarks; ASCADv2 [MS23] | V; missing cite | — | → "a small set of public benchmarks~\cite{ASCAD,MS23}" (`ASCAD` is in refs.bib but never cited); optionally add the survey `\cite{SoKCSUR23}` |
| 6–8 | ANSSI implementation on STM32 Cortex-M4; affine masking α∈GF*, β; shuffling of 16 bytes | V; missing cite | ePrint §1.3.1–1.3.2 pp. 6–7; §2.1 p. 8 | Append "first dissected by Bronchain and Standaert~\cite{BS20}" |
| 8–9 | "In the worst-case (or 'first threat') setting" | C (wording) | ePrint §4.3 p. 19 ("First Scenario"); §4.1 p. 16 ("First Threat Scenario" = partial attack); Intro p. 3 | → "In the setting usually called worst-case -- MS23's first full-key scenario (§4.3), where masks and permutation indices are known during profiling only --" |
| 11 | Full key in about 60 traces [MS23] | V | ePrint §4.3 p. 20 | None |
| 11–12 | GPAM on raw 1M-sample traces | V | GPAM §7.1 p. 492 | None |
| 12 | OvC changed the training objective | V | OvC §3 (order-vs-chaos classifier) | None |
| 12–14 | Non-worst-case setting [Scoop]; false positive [ReallyBroken] | R | Scoop §6.3, Table 5; ePrint 2026/338 §3.2, §5.1 | → "…where a recent single-byte claim of success~\cite{Scoop} was shown to be a false positive~\cite{ReallyBroken}" |
| 14 | "The worst-case reference model itself has not been revisited." | C (partly) | UNIDLE ePrint 2024/1437 (capacity bottleneck; vs MultiResNetSCA-2 on its own Arduino dataset) | → "Its architecture has not been re-audited on ASCADv2; Hajra et al.~\cite{UNIDLE} criticise the capacity bottleneck of such multi-task networks, but on a different target." |
| 21–23 | Parameter audit (94.5%, 90.8%) | Own; R (positioning) | ZBHV20, WAGP20 abstracts | Add after the bullet: "Small single-task SCA networks are known to suffice~\cite{ZBHV20,WAGP20,RWPP21}; we localise the excess inside a multi-task reference model." |
| 24–27 | Head accuracies match MS23 within two points; "Under the paper's own metric" Compact needs 60 | V (accuracies); R (metric) | ePrint Table 2 p. 20 (on a 50k test set) | → "Under GE$<$1 on all bytes over 500 attacks, the protocol behind the per-byte curves of~\cite{MS23}, CompactMultiResNet needs 60 traces…" |
| 30–32 | GPAM "at its published learning rate it does not learn the masked S-box" | R | GPAM Table 1 p. 480; §5.3.1 p. 481; scaaml `gpam_ecc_cm1.py`, configurations/ | → "with Table~1's target learning rate under Adafactor and our warm-up (the paper names no optimizer; the released code's configurations use 10$\times$ Table~1's rates), it does not learn the masked S-box" |
| 46–47 | TODO: no record over OvC until its TGE₀ is checked | Resolved | OvC §3.4.2 p. 18; §4.1.1 p. 19; Table 1 p. 21 | Replace the TODO with: "We claim no new trace record: our 60 traces equal the full-key figure of~\cite{MS23}. OvC's 47~\cite{OvC} is for the first key byte only (GE$=0$, 100 attacks) and is not comparable." |

### Background (background.tex)

| Line | Claim | Verdict | Source location | Change |
|---|---|---|---|---|
| 5–6 | 800k traces × 1M samples, STM32F303, random key and plaintext [MS23] | V (key/plaintext from the README, not the paper text) | ePrint §2.1 pp. 8–9; ANSSI README | → "…per encryption~\cite{MS23,ANSSIASCADv2}" |
| 6–7 | Official extracted set: two windows → 15,000 samples, 500k profiling + 10k attack | V for the release; C as an MS23 fact (ePrint §2.2 p. 11 says 700k/20k/50k) | `example_generate_params`; HDF5 shapes; ePrint §2.2 pp. 9–11 | → "We use the official extracted set released with the ANSSI code~\cite{ANSSIASCADv2}, which concatenates the raw windows $[205\mathrm{k},210\mathrm{k})$ and $[455\mathrm{k},465\mathrm{k})$ into 15,000 samples, with 500k profiling and 10k attack traces (MS23's experiments used a 700k/20k/50k split of the raw campaign~\cite[§2.2]{MS23})." |
| 9–13 | Eq. (masked) C_j = α⊗Sbox[p_π(j)⊕k_π(j)]⊕β; π random permutation | V | ePrint Eq. 6 p. 16; §4.3 p. 19; `multilabelize` | Optional: "π is a pseudo-random permutation generated from four seed bytes~\cite[Eq.~3]{MS23}" |
| 16 | "the worst-case setting of [MS23]" | C (wording) | ePrint §4.3 p. 19 | → "the worst-case setting, i.e.\ the first full-key scenario of~\cite[§4.3]{MS23}" |
| 18–20 | Variable attack keys; fixed all-zero key simulated via p⊕k as in MS23 | V | HDF5 (10,000 distinct keys); ePrint Eq. 7 p. 16; §4.3 p. 19; `simulated_key=1` | None |
| 24–30 | Decoder Eq. (decode) = [MS23, Eq. 12], no slot normalisation | V (ePrint numbering only) | ePrint Eq. 12 p. 20; App. G Eq. 23 pp. 33–34 | → `\cite[ePrint 2021/592, Eq.~12]{MS23}`, or check the JCE numbering. Apply the same to "Table 2" (results 5) and "Fig. 12a" |
| 37–38 | Ties counted against the correct key | Own; MS23 differ | ePrint Eq. 15 p. 26 (strict >); `ASCAD_test_models.py` (first index) | Add: "(MS23~\cite[Eq.~15]{MS23} count ties in favour of the correct key; ours is the conservative choice)" |
| 39 | GE definition [SMY09] | V | SMY09 | Optional: add `\cite{Massey94}` |
| 41–43 | A = 500 as in MS23, "about 60" read off such curves | V; R | ePrint §1.2 p. 5; App. A p. 27 | → "($A=500$, as in~\cite{MS23}, whose attacks are drawn from a 50k-trace test set and whose ``about 60 traces'' is read off per-byte GE curves)" |

### Related work (related.tex)

| Line | Claim | Verdict | Source location | Change |
|---|---|---|---|---|
| 4–5 | MS23 introduced the dataset and attack, "following the multi-label approach of [Maghrebi20]" | V | MS23 abstract ("revisit and leverage … Maghrebi in 2020"); ePrint 2020/436 | → "…revisiting the multi-label approach of~\cite{Maghrebi20}" |
| 6 | "OvC reports TGE₀ 47 with randomness labels" | R (incomplete) | OvC §3.4.2 p. 18; §4.1.1 p. 19; Table 1 p. 21 | → "OvC~\cite{OvC} reports 47 traces to reach GE$=0$ on the first key byte only (100 attacks, 15k extract, mask and permutation labels during profiling) and gives no full-key result." |
| 6–8 | GPAM: patch transformer with gated attention on raw traces; full-key GE<1 in about 80 | V; R | GPAM §5.2 pp. 479–480; §7.1 p. 492; Table 3 | Append: "(GE over 10,000 samplings of an 80k holdout after 640k profiling traces, with each masked-S-box head taken from the best of seven runs)" |
| 8 | "We found no newer worst-case full-key result" + Scholar TODO | R | ePrint 2023/006 v1; ePrint 2023/611; ePrint search for ASCADv2 | → "Apart from GPAM we found no peer-reviewed full-key result in the standard worst-case setting. Marquet and Oswald report about 21 traces with $\alpha$ and $\pi$ known in profiling and attack~\cite{MarquetOswald23AIHWS}." Keep the Scholar pass as an internal task, not a PDF TODO |
| new | Missing worst-case precedents | — | BS20; MCLS23 | Add: "Bronchain and Standaert~\cite{BS20} dissected the same ANSSI implementation with implementation knowledge and recovered the key with fewer than 2,000 traces. Scheme-aware models~\cite{MCLS23} inject the masking scheme instead of labelling shares; on ASCADv2 they succeeded only when trained in a worst-case setting." |
| 10–11 | Marquet and Oswald: MTL without mask labels, shuffling disabled | V; R | ePrint 2023/006 §2.2, §5.3 | Append ", with mask points of interest selected by SNR on the masks" |
| 11–12 | "Wu et al. [WPP23] and Karayalcin et al. [DDPM24] mount non-profiled collision attacks, again without shuffling" | V (**not a mis-citation**); R | WPP23 §1, §5.1; DDPM24 §5 | → "Wu et al.~\cite{WPP23} mount a non-profiled collision attack with shuffling disabled, and Karayalcin et al.~\cite{DDPM24} reduce its trace count by about 40\% by denoising the traces with diffusion models. Non-profiled deep learning has not broken ASCADv2 within 500k traces~\cite{DSNSCA26}." |
| 12–13 | Scoop claimed the first non-worst-case attack; ReallyBroken showed a false positive | V; R | Scoop abstract, §6.3; ePrint 2026/338 §3–§5 | → "Scoop~\cite{Scoop} claimed the first non-worst-case attack, on one key byte, which~\cite{ReallyBroken} showed to exploit a non-uniform prior of the targeted byte rather than leakage (see also~\cite{SoKKKP25})." |
| 13–15 | Gammell and Roy: plain transformer, uncropped ASCADv1 and CHES-CTF | V | arXiv:2608.30105 abstract | → "…uncropped ASCADv1 and CHES-CTF-2018 traces" |
| new | Model-size and multi-task positioning paragraph | — | ZBHV20, WAGP20, KPP23resnet, UNIDLE, Standley20, MO24 | Add a "Model size and multi-task learning" paragraph using the sentences given in the bibliography section below |

### Experimental setup (setup.tex)

| Line | Claim | Verdict | Source location | Change |
|---|---|---|---|---|
| 4–5 | "first 400,000 profiling traces … as in [MS23]" | V (paper); R (code) | ePrint §4.3 p. 19; `example_train_models_params` (`train_len` 420000, `validation_split` 0.05) | → "as in~\cite{MS23} (§4.3; the released script trains on the first 420k with the last 5\% held out, i.e.\ about the first 399k)" |
| 5–6 | Four all-zero placeholders at 99,999 … 399,999 | V | HDF5 (a fifth at 499,999 lies outside the 400k subset) | None |
| 10–13 | Recipe: mean subtraction, RAdam 10⁻⁴, batch 64, 15 epochs, best = lowest val loss | Own; deviation from MS23 | ePrint §3.1.3 p. 13 (Adam); §4.1 p. 16 (patience 10); `load_ascad` (raw int8); train params (max 60 epochs) | Add: "This differs from~\cite{MS23}, who train with Adam at its default learning rate ($10^{-3}$) on raw traces with early stopping (patience 10; training stopped after 15 epochs). Selecting by validation loss follows the NLL--PI equivalence of~\cite{MDP20}." |
| 16 | Ldec "computed exactly with a Walsh–Hadamard transform" | V (standard); missing cite | SCALib `bp_compute.rs`/`belief_propagation.rs`; MS23 has no WHT | → "…with a Walsh--Hadamard transform, as is standard for XOR factors in soft analytical attacks~\cite{VGS14,GGSB20,SCALib}" |
| 20–21 | "the 34 heads of the paper and its Table 1 settings" | V | GPAM §7.1 p. 492 (targets listed, 34 never printed); Table 1 p. 480 | → "the 34 heads of the paper ($c[i]$, $p[i]$, $r_m$, $r_{out}$)" |
| 21–23 | batch 64, 1,000 steps, 150 epochs, merge 0/0, LR 5·10⁻⁵, warm-up + cosine | V | GPAM Table 1 p. 480; §5.3.1 p. 481 | None |
| 23–24 | Patch 125 "following the paper's √L rule" | R | GPAM §5.3 p. 481 (rule); Table 1 (ASCADv2 patch 400 ≠ 1000) | → "following the $\sqrt{L}$ rule stated in~\cite[§5.3]{GPAM} (their own ASCADv2 setting uses 400 rather than $\sqrt{10^6}=1000$)" |
| 24–25 | "The optimizer and the warm-up length are not given in [GPAM]" | V; R | GPAM §5.3.1; scaaml code and configs | Append: "The released code uses Adafactor at a constant learning rate, states that the warm-up schedule is unnecessary, and its released configurations (none for ASCADv2) use 10$\times$ the Table~1 rates." |
| 31–32 | Ties counted against the correct key | Own | No literature on tie handling found | Keep, uncited; add the MS23 contrast once (background 37–38) |

### Where the parameters go (parameters.tex)

| Line | Claim | Verdict | Source location | Change |
|---|---|---|---|---|
| 5–11 | Architecture: 1-D ResNet [ResNet], stem + 9 stacks, k=11, stride 2 from stack 2, 16→256, AvgPool 4 → 3584, heads Lin(3584,1024)→ReLU→BN→Lin(1024,K) | V | ePrint §3.1.3 p. 13; Figs. 16–17 pp. 31–32; `resnet_v1` | → "a one-dimensional residual network~\cite{ResNet}, in the line of SCA ResNets~\cite{ZS20}," |
| Table | 137,396,064; trunk 7,525,472; heads 129,870,592; first dense 124,815,360 | V / consistent | ePrint §3.2 p. 15 (also reproduces 11,460,960 and 78,364,256) | Caption → "trainable parameters (BN running statistics excluded, as in~\cite{MS23})" |
| 35–39 | Heads hold the capacity to fit noise | Own; supported | NiN (GAP vs FC overfitting); KPP23resnet | Optional `\cite{NiN14}` |

### Results (results.tex)

| Line | Claim | Verdict | Source location | Change |
|---|---|---|---|---|
| 5–8 | Head accuracies vs MS23 Table 2 (99.2 / 21.1 / 1.6 / 88.9) | V | ePrint Table 2 p. 20 | Add: "(theirs on a 50k-trace test set, ours on 5,000 attack traces)" |
| 20–24 | "Under the metric of [MS23] (GE<1 on all bytes, 500 attacks)… the figure reported" | R | ePrint §4.3 p. 20; App. A pp. 26–27 | → "At GE$<$1 on all bytes over 500 attacks, the threshold at which the per-byte curves of~\cite{MS23} give their ``about 60'', CompactMultiResNet needs 60 traces…" Add: "MS23 sample their attacks from a different 50k test set, so the match is in protocol, not data." |
| 41–46 | 137.4M overfits after epoch 4; last checkpoints 135–160 | Own; consistent | Inference from MS23 patience 10 + "stops after 15 epochs" (best ≈ epoch 5) | Add: "MS23's early stopping (patience 10) implies their published model is likewise an early checkpoint." Cite `RZCBH20,PBP21` for SCA early stopping |
| 50 | GPAM designed for raw 1M-sample traces | V | GPAM §7.1 p. 492 | None |
| 51–52 | "With the published target learning rate it stays near chance" | R | as setup 24–25 | → "With Table~1's target learning rate under Adafactor…" |
| 58–59 | GPAM reports about 80 on raw traces; our reproduction needed 600 | V (80); Own (600) | GPAM §7.1 p. 492 | Append: "GPAM's figure uses 640k profiling traces, the best of seven runs per masked-S-box head and a hard $r_m$; we ran one configuration." Replace the TODO with an appendix reference |

### Carrier (carrier.tex)

| Line | Claim | Verdict | Source location | Change |
|---|---|---|---|---|
| 4 (heading) | "Top-1 accuracy is the wrong measure" | Anticipated | PHJBR19; MDP20; ZZNHY20; IUH22; MS23 §4.1 (1.56% accuracy still attacks) | Heading → "Top-1 accuracy does not locate the key information"; add after line 9: "That accuracy is a poor proxy for SCA success is well known~\cite{PHJBR19,MDP20,ZZNHY20}; what the swaps add is a causal attribution of all key information to the $C$ heads." |
| 10–12 | Per-head log-likelihood gain not a reliable proxy | Own; supported | IUH22 (CE/PI "non-calibrated") | Append: "an instance of the calibration gap of cross-entropy and perceived information~\cite{IUH22}" |
| 15–21 | Shared 512-d bottleneck extinguishes C | Own; consistent | Standley20; MarquetOswald24 conclusion | Append: "This is consistent with task competition under hard parameter sharing~\cite{Standley20} and with Marquet and Oswald's warning that losses can compete~\cite{MarquetOswald24}; their positive result concerns low-level sharing without shuffling." |
| 26–29 | Loss weighting TODO | — | GradNorm18; KGC18; XGGGF22; KDKWK22 | When filled: "We use a fixed scalarisation, which tuned studies find hard to beat~\cite{XGGGF22,KDKWK22}, rather than adaptive weighting~\cite{GradNorm18,KGC18}." |

### Label ladder (ladder.tex)

| Line | Claim | Verdict | Source location | Change |
|---|---|---|---|---|
| 4–11 | Ldec uses only the unmasked S_i; rungs that keep C remove a loss, not information | Own; positioning needed | MCLS23 (scheme-aware model) | Add: "Scheme-aware modelling~\cite{MCLS23} also builds the masking algebra into training, but without share labels; our rungs keep $\alpha$/$\beta$ labels and study which head losses the decoder can replace." |
| 44–45 | XOR convolution by WHT | V (standard) | SCALib; UIT+25 | Add `\cite{SCALib,UIT+25}` |
| 113–115 | ReallyBroken's negative result for Scoop not contradicted | V | ePrint 2026/338 | None |

### Reporting trace counts (pitfalls.tex)

| Line | Claim | Verdict | Source location | Change |
|---|---|---|---|---|
| 4–6 | Single-ordering / MTD / GE<1 / GE=0 differ almost 5× | Own; anticipated in spirit | WPP22eval (median, dispersion); WWKLPBP20 (simple vs generalised GE) | Append: "consistent with earlier warnings that single attacks and mean GE mislead~\cite{WPP22eval,WWKLPBP20}. The authors of~\cite{MS23} themselves report about 120 traces for their released checkpoint on one ordering~\cite{ANSSIASCADv2}." |
| 7 | "we compare with MS23 at GE<1 because that is what its figure shows" | R | ePrint Fig. 12a (per-byte mean rank, linear axis) | → "because its Fig.~12a plots per-byte guessing entropy" |
| 7–9 | TODO: OvC's TGE₀ definition | Resolved | OvC §3.4.2 p. 18; §4.1.1 p. 19 | Replace with: "OvC's TGE$_0$~\cite{OvC} is the trace count at which GE$=0$ over 100 attacks, for the first key byte only, so its 47 is not comparable with any full-key figure here." |
| 11–14 | Tie convention can create false positives | Own; no prior source | None found | Keep uncited; mention that MS23 count ties in favour |
| 14–17 | Uniform masked labels, so "the label-prior false positive identified by [ReallyBroken] cannot arise here" | R (own corollary, not stated by ReallyBroken) | ePrint 2026/338 Lemma 1 / Corollary 1 §4.1; Null Benchmark §5 | → "Label-prior artefacts are documented for non-uniform labels~\cite{PHJBR19,ISUH21,ReallyBroken}. By Lemma~1 of~\cite{ReallyBroken}, the posterior and maximum-likelihood distinguishers coincide when the label is uniform, which the masked targets $C_j$ are by construction; the mask labels' priors do not depend on the key. Our shuffled-posterior null (median $|z|<0.6$) and uniform-$C$ control play the role of their null benchmark and input ablation." |

### Conclusion (conclusion.tex)

| Line | Claim | Verdict | Source location | Change |
|---|---|---|---|---|
| 4–9 | Reference attack spends parameters on heads; masked S-box heads carry the key | Own | — | None. In the limitations TODO, add "MS23's results are on a different 50k test split" and "GPAM's original optimizer is undisclosed" |

## Bibliography fixes and additions, ready for refs.bib

Every existing entry exists, and every author list is correct. Nine entries need edits:

- **Remove TODO notes only:** `OvC` and `Maghrebi20`.
- **Fill in or add fields:** `MarquetOswald24`, `ASCAD`, `SMY09`, `ResNet` and `GammellRoy26`.
- **Switch to the published version:** `WPP23` (SAC 2023) and `DDPM24` (SAC 2025).

`MS23`, `GPAM`, `Scoop` and `ReallyBroken` are verified as they stand ([Crossref MS23](https://api.crossref.org/works/10.1007/s13389-023-00311-7); [Crossref GPAM](https://api.crossref.org/works/10.46586/tches.v2024.i3.472-499); [Crossref Scoop](https://api.crossref.org/works/10.46586/tches.v2025.i3.56-80); [ePrint 2026/338](https://eprint.iacr.org/2026/338)). Keep `ReallyBroken` and `OvC` as `@misc`: neither has a venue. Fields still unverified are marked `% UNVERIFIED` in the BibTeX below. Two traps:

- Two different papers share the title "SoK: Deep Learning-based Physical Side-channel Analysis": Picek et al. in ACM CSUR 2023, and Karayalcin, Krček and Picek in ePrint 2025/1309 (title, authors and preprint status confirmed on the [ePrint page](https://eprint.iacr.org/2025/1309)). The research notes gave both the key `SoKDLSCA`, so they get separate keys below.
- Semantic Scholar mislabels the SAC venue of WPP23 as "ACM Symposium on Applied Computing", so do not auto-import that BibTeX.

Where to cite the new entries:

| Key | Priority | Where to cite |
|---|---|---|
| BS20 | Must | intro.tex 6–8; related.tex new paragraph |
| MCLS23 | Must | related.tex new paragraph; ladder.tex 4–11 |
| UNIDLE | Must | intro.tex 14; related.tex model-size paragraph |
| MarquetOswald23AIHWS | Must | related.tex 8 |
| ANSSIASCADv2 | Must | background.tex 5–7; setup.tex 4; pitfalls.tex 4–6 |
| ZBHV20, WAGP20 | Must | intro.tex 21–23; related.tex model-size paragraph |
| PHJBR19, MDP20, IUH22 | Must | carrier.tex 4–12; setup.tex 10–13 (MDP20) |
| WPP22eval | Must | pitfalls.tex 4–6 |
| Standley20 | Must | carrier.tex 15–21 |
| VGS14, GGSB20, SCALib | Must | setup.tex 16; ladder.tex 44 |
| DSNSCA26, SoKKKP25 | Should | related.tex 11–13 |
| ZZNHY20, ISUH21, WWKLPBP20 | Should | carrier.tex 4; pitfalls.tex 4–6, 14–17 |
| RWPP21, KPP23resnet, ZS20 | Should | related.tex model-size paragraph; parameters.tex 5 |
| UIT+25 | Should | ladder.tex 44 |
| RZCBH20, PBP21, PCP20 | Should | results.tex 41–46 |
| XGGGF22, KDKWK22, GradNorm18, KGC18 | Should (once the w_C section is filled) | carrier.tex 26–29 |
| SoKCSUR23, Massey94, NiN14 | Optional | intro.tex 4; background.tex 39; parameters.tex 35 |

Sentences for the new "Model size and multi-task learning" paragraph in related.tex: "Small single-task networks are known to suffice for many SCA targets~\cite{ZBHV20,WAGP20,RWPP21}, while deep residual trunks help on long inputs~\cite{ZS20,KPP23resnet}. Hajra et al.~\cite{UNIDLE} point to the capacity bottleneck of multi-task trunks. Marquet and Oswald~\cite{MarquetOswald24} find that low-level sharing helps but that task losses can compete, in line with the multi-task literature~\cite{Standley20}." If a reviewer asks for a true full-key rank rather than "GE<1 on all bytes", the standard rank-estimation references are Glowacz et al. FSE 2015 ([doi:10.1007/978-3-662-48116-5_6](https://doi.org/10.1007/978-3-662-48116-5_6)) and Poussier et al. CHES 2016 ([doi:10.1007/978-3-662-53140-2_4](https://doi.org/10.1007/978-3-662-53140-2_4)). Cristiani et al.'s non-profiled attack on the same implementation, which needs a separate 100M-trace acquisition, is an optional related-work mention ([doi:10.1109/ACCESS.2022.3222760](https://doi.org/10.1109/ACCESS.2022.3222760)); the 100M figure is from Wu et al.'s summary and was not read in the original.

```bibtex
% ===================== FIXES TO EXISTING ENTRIES =====================

% MS23: verified (Crossref). Equation/table/figure locators in the draft are verified
% only against ePrint 2021/592; JCE numbering UNVERIFIED.
@article{MS23,
  author  = {Lo{\"\i}c Masure and R{\'e}mi Strullu},
  title   = {Side-channel analysis against {ANSSI}'s protected {AES} implementation on {ARM}:
             end-to-end attacks with multi-task learning},
  journal = {Journal of Cryptographic Engineering},
  volume  = {13}, number = {2}, pages = {129--147}, year = {2023},
  doi     = {10.1007/s13389-023-00311-7},
  note    = {Preprint: IACR ePrint 2021/592},
}

% OvC: verified; still a preprint (received 2023-10-18, revised 2024-01-16). Remove the TODO.
@misc{OvC,
  author       = {Praveen Kulkarni and Vincent Verneuil and Stjepan Picek and Lejla Batina},
  title        = {Order vs. Chaos: A Language Model Approach for Side-channel Attacks},
  howpublished = {Cryptology {ePrint} Archive, Paper 2023/1615},
  year         = {2023},
  url          = {https://eprint.iacr.org/2023/1615},
  note         = {Version of 2024-01-16},
}

% MarquetOswald24: verified (Crossref DOI/pages/editors); LNCS 14595 via Scoop's reference list + K10plus.
@inproceedings{MarquetOswald24,
  author    = {Thomas Marquet and Elisabeth Oswald},
  title     = {Exploring Multi-task Learning in the Context of Masked {AES} Implementations},
  booktitle = {Constructive Side-Channel Analysis and Secure Design -- {COSADE} 2024},
  editor    = {Romain Wacquez and Naofumi Homma},
  series    = {Lecture Notes in Computer Science}, volume = {14595},
  pages     = {93--112}, publisher = {Springer}, year = {2024},
  doi       = {10.1007/978-3-031-57543-3_6},
}

% WPP23: verified (Crossref); LNCS 14201 via Scoop's reference list + K10plus. Published 2024.
@inproceedings{WPP23,
  author    = {Lichao Wu and Guilherme Perin and Stjepan Picek},
  title     = {Not so Difficult in the End: Breaking the Lookup Table-Based Affine Masking Scheme},
  booktitle = {Selected Areas in Cryptography -- {SAC} 2023},
  editor    = {Claude Carlet and Kalikinkar Mandal and Vincent Rijmen},
  series    = {Lecture Notes in Computer Science}, volume = {14201},
  pages     = {82--96}, publisher = {Springer}, year = {2024},
  doi       = {10.1007/978-3-031-53368-6_5},
}

% DDPM24: NOT a mis-citation (SAC 2025 version, Sec. 5: Wu et al.'s collision attack on ASCADv2,
% shuffling disabled, about 40% fewer traces). Verified via Crossref (DOI, pages, editors, year).
@inproceedings{DDPM24,
  author    = {Sengim Karayal{\c{c}}in and Guilherme Perin and Stjepan Picek},
  title     = {Diffuse Some Noise: Diffusion Models for Measurement Noise Removal in Side-Channel Analysis},
  booktitle = {Selected Areas in Cryptography -- {SAC} 2025},
  editor    = {Christina Boura and Atefeh Mashatan and Ali Miri},
  series    = {Lecture Notes in Computer Science},
  volume    = {16207},  % UNVERIFIED: from a Springer search snippet only; confirm on link.springer.com
  pages     = {724--752}, publisher = {Springer}, year = {2026},
  doi       = {10.1007/978-3-032-10536-3_27},
  note      = {Preprint: IACR ePrint 2024/966},
}

% Maghrebi20: title and number verified; still a preprint. Remove the TODO.
@misc{Maghrebi20,
  author       = {Houssem Maghrebi},
  title        = {Deep Learning based Side-Channel Attack: a New Profiling Methodology based on
                  Multi-Label Classification},
  howpublished = {Cryptology {ePrint} Archive, Paper 2020/436},
  year         = {2020},
}

% ASCAD: verified (Crossref; online 2019-11-30, issue 2020). Currently never cited: cite in intro.tex line 5.
@article{ASCAD,
  author  = {Ryad Benadjila and Emmanuel Prouff and R{\'e}mi Strullu and Eleonora Cagli and C{\'e}cile Dumas},
  title   = {Deep learning for side-channel analysis and introduction to {ASCAD} database},
  journal = {Journal of Cryptographic Engineering},
  volume  = {10}, number = {2}, pages = {163--188}, year = {2020},
  doi     = {10.1007/s13389-019-00220-8},
}

% SMY09: verified (Crossref pages/DOI); LNCS 5479 via K10plus/OpenLibrary + OvC's reference list.
@inproceedings{SMY09,
  author    = {Fran{\c{c}}ois-Xavier Standaert and Tal Malkin and Moti Yung},
  title     = {A Unified Framework for the Analysis of Side-Channel Key Recovery Attacks},
  booktitle = {Advances in Cryptology -- {EUROCRYPT} 2009},
  editor    = {Antoine Joux},
  series    = {Lecture Notes in Computer Science}, volume = {5479},
  pages     = {443--461}, publisher = {Springer}, year = {2009},
  doi       = {10.1007/978-3-642-01001-9_26},
}

% ResNet: verified (Crossref + CVF open-access BibTeX).
@inproceedings{ResNet,
  author    = {Kaiming He and Xiangyu Zhang and Shaoqing Ren and Jian Sun},
  title     = {Deep Residual Learning for Image Recognition},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition ({CVPR})},
  pages     = {770--778}, year = {2016},
  doi       = {10.1109/CVPR.2016.90},
}

% GammellRoy26: verified (arXiv API record; v1 2026-08-31). No proceedings DOI exists.
@misc{GammellRoy26,
  author       = {Jimmy Gammell and Kaushik Roy},
  title        = {A Simple Transformer Pipeline for Full-Key Side-Channel Attacks on Uncropped Datasets},
  howpublished = {arXiv:2608.30105},
  note         = {Accepted to the OPTIMIST Workshop '26 at CHES 2026},
  year         = {2026},
}

% ===================== NEW: ASCADv2 / ANSSI implementation =====================

% ANSSIASCADv2: repository URL and commit verified (local git history, upstream 58c527f, 2021-05-01).
% Author/title are descriptive. UNVERIFIED: whether ANSSI prescribes a citation form for the repo.
@misc{ANSSIASCADv2,
  author       = {{ANSSI}},
  title        = {{ASCAD} repository, {STM32\_AES\_v2}: code, parameters and extracted {ASCADv2} database},
  howpublished = {\url{https://github.com/ANSSI-FR/ASCAD/tree/master/STM32_AES_v2}},
  year         = {2021},
  note         = {Commit 58c527f; data at \url{https://www.data.gouv.fr/en/datasets/ascadv2/}},
}

% BS20: verified (Crossref). Must-cite: dissection of the same ANSSI affine+shuffled AES.
@article{BS20,
  author  = {Olivier Bronchain and Fran{\c{c}}ois-Xavier Standaert},
  title   = {Side-Channel Countermeasures' Dissection and the Limits of Closed Source Security Evaluations},
  journal = {IACR Transactions on Cryptographic Hardware and Embedded Systems},
  volume  = {2020}, number = {2}, pages = {1--25}, year = {2020},
  doi     = {10.46586/tches.v2020.i2.1-25},
}

% MCLS23: verified (Crossref; full title from the TCHES page).
@article{MCLS23,
  author  = {Lo{\"\i}c Masure and Valence Cristiani and Maxime Lecomte and Fran{\c{c}}ois-Xavier Standaert},
  title   = {Don't Learn What You Already Know: Scheme-Aware Modeling for Profiling Side-Channel Analysis against Masking},
  journal = {IACR Transactions on Cryptographic Hardware and Embedded Systems},
  volume  = {2023}, number = {1}, pages = {32--59}, year = {2023},
  doi     = {10.46586/tches.v2023.i1.32-59},
}

% MarquetOswald23AIHWS: verified (Crossref); LNCS 13907 via K10plus + catalogue snippet.
@inproceedings{MarquetOswald23AIHWS,
  author    = {Thomas Marquet and Elisabeth Oswald},
  title     = {A Comparison of Multi-task Learning and Single-Task Learning Approaches},
  booktitle = {Applied Cryptography and Network Security Workshops -- {ACNS} 2023 Satellite Workshops},
  series    = {Lecture Notes in Computer Science}, volume = {13907},
  pages     = {121--138}, publisher = {Springer}, year = {2023},
  doi       = {10.1007/978-3-031-41181-6_7},
  note      = {Preprint: IACR ePrint 2023/611},
}

% UNIDLE: verified (ePrint page: title, author order, preprint, last revised 2025-07-07).
@misc{UNIDLE,
  author       = {Suvadeep Hajra and Debdeep Mukhopadhyay and Soumi Chatterjee},
  title        = {{UNIDLE}: A Unified Framework for Deep Learning-based Side-channel Analysis},
  howpublished = {Cryptology {ePrint} Archive, Paper 2024/1437},
  year         = {2024},
}

% DSNSCA26: verified (Crossref).
@article{DSNSCA26,
  author  = {Di Li and Zheng Gong and Yufeng Tang and Chun Li},
  title   = {{DS-NSCA}: Differentiable Search for Non-Profiling Side-channel Analysis},
  journal = {IACR Transactions on Cryptographic Hardware and Embedded Systems},
  volume  = {2026}, number = {2}, pages = {659--685}, year = {2026},
  doi     = {10.46586/tches.v2026.i2.659-685},
}

% SoKKKP25: verified (ePrint page: title, authors, preprint, revised 2026-06-01).
% Same title as SoKCSUR23 below; these are different papers.
@misc{SoKKKP25,
  author       = {Sengim Karayalcin and Marina Kr{\v{c}}ek and Stjepan Picek},
  title        = {{SoK}: Deep Learning-based Physical Side-channel Analysis},
  howpublished = {Cryptology {ePrint} Archive, Paper 2025/1309},
  year         = {2025},
}

% ===================== NEW: model size / architectures =====================

% verified (Crossref)
@article{ZBHV20,
  author  = {Gabriel Zaid and Lilian Bossuet and Amaury Habrard and Alexandre Venelli},
  title   = {Methodology for Efficient {CNN} Architectures in Profiling Attacks},
  journal = {IACR Transactions on Cryptographic Hardware and Embedded Systems},
  volume  = {2020}, number = {1}, pages = {1--36}, year = {2020},
  doi     = {10.46586/tches.v2020.i1.1-36},
}
% verified (Crossref)
@article{WAGP20,
  author  = {Lennert Wouters and Victor Arribas and Benedikt Gierlichs and Bart Preneel},
  title   = {Revisiting a Methodology for Efficient {CNN} Architectures in Profiling Attacks},
  journal = {IACR Transactions on Cryptographic Hardware and Embedded Systems},
  volume  = {2020}, number = {3}, pages = {147--168}, year = {2020},
  doi     = {10.46586/tches.v2020.i3.147-168},
}
% verified (Crossref)
@article{RWPP21,
  author  = {Jorai Rijsdijk and Lichao Wu and Guilherme Perin and Stjepan Picek},
  title   = {Reinforcement Learning for Hyperparameter Tuning in Deep Learning-based Side-channel Analysis},
  journal = {IACR Transactions on Cryptographic Hardware and Embedded Systems},
  volume  = {2021}, number = {3}, pages = {677--707}, year = {2021},
  doi     = {10.46586/tches.v2021.i3.677-707},
}
% verified (Crossref)
@article{KPP23resnet,
  author  = {Sengim Karayalcin and Guilherme Perin and Stjepan Picek},
  title   = {Resolving the Doubts: On the Construction and Use of {ResNets} for Side-Channel Analysis},
  journal = {Mathematics},
  volume  = {11}, number = {15}, pages = {3265}, year = {2023},
  doi     = {10.3390/math11153265},
}
% verified (Crossref; online 2019, issue 2020)
@article{ZS20,
  author  = {Yuanyuan Zhou and Fran{\c{c}}ois-Xavier Standaert},
  title   = {Deep learning mitigates but does not annihilate the need of aligned traces and a generalized {ResNet} model for side-channel attacks},
  journal = {Journal of Cryptographic Engineering},
  volume  = {10}, number = {1}, pages = {85--95}, year = {2020},
  doi     = {10.1007/s13389-019-00209-3},
}
% verified (arXiv; comment "for iclr2014"). UNVERIFIED: abstract tail on GAP overfitting; ICLR has no pages.
@inproceedings{NiN14,
  author    = {Min Lin and Qiang Chen and Shuicheng Yan},
  title     = {Network In Network},
  booktitle = {International Conference on Learning Representations ({ICLR})},
  year      = {2014}, note = {arXiv:1312.4400},
}
% verified (Crossref pages 1--35). UNVERIFIED: article number (227) -- omitted.
@article{SoKCSUR23,
  author  = {Stjepan Picek and Guilherme Perin and Luca Mariot and Lichao Wu and Lejla Batina},
  title   = {{SoK}: Deep Learning-based Physical Side-channel Analysis},
  journal = {ACM Computing Surveys},
  volume  = {55}, number = {11}, pages = {1--35}, year = {2023},
  doi     = {10.1145/3569577},
}

% ===================== NEW: metrics / accuracy vs SCA =====================

% verified (Crossref)
@article{PHJBR19,
  author  = {Stjepan Picek and Annelie Heuser and Alan Jovic and Shivam Bhasin and Francesco Regazzoni},
  title   = {The Curse of Class Imbalance and Conflicting Metrics with Machine Learning for Side-channel Evaluations},
  journal = {IACR Transactions on Cryptographic Hardware and Embedded Systems},
  volume  = {2019}, number = {1}, pages = {209--237}, year = {2019},
  doi     = {10.46586/tches.v2019.i1.209-237},
}
% verified (Crossref)
@article{MDP20,
  author  = {Lo{\"\i}c Masure and C{\'e}cile Dumas and Emmanuel Prouff},
  title   = {A Comprehensive Study of Deep Learning for Side-Channel Analysis},
  journal = {IACR Transactions on Cryptographic Hardware and Embedded Systems},
  volume  = {2020}, number = {1}, pages = {348--375}, year = {2020},
  doi     = {10.46586/tches.v2020.i1.348-375},
}
% verified (Crossref)
@article{ZZNHY20,
  author  = {Jiajia Zhang and Mengce Zheng and Jiehui Nan and Honggang Hu and Nenghai Yu},
  title   = {A Novel Evaluation Metric for Deep Learning-Based Side Channel Analysis and Its Extended Application to Imbalanced Data},
  journal = {IACR Transactions on Cryptographic Hardware and Embedded Systems},
  volume  = {2020}, number = {3}, pages = {73--96}, year = {2020},
  doi     = {10.46586/tches.v2020.i3.73-96},
}
% verified (Crossref)
@article{IUH22,
  author  = {Akira Ito and Rei Ueno and Naofumi Homma},
  title   = {Perceived Information Revisited: New Metrics to Evaluate Success Rate of Side-Channel Attacks},
  journal = {IACR Transactions on Cryptographic Hardware and Embedded Systems},
  volume  = {2022}, number = {4}, pages = {228--254}, year = {2022},
  doi     = {10.46586/tches.v2022.i4.228-254},
}
% verified (Crossref metadata only; content not read)
@article{ISUH21,
  author  = {Akira Ito and Kotaro Saito and Rei Ueno and Naofumi Homma},
  title   = {Imbalanced Data Problems in Deep Learning-Based Side-Channel Attacks: Analysis and Solution},
  journal = {IEEE Transactions on Information Forensics and Security},
  volume  = {16}, pages = {3790--3802}, year = {2021},
  doi     = {10.1109/TIFS.2021.3092050},
}
% verified (Crossref; LNCS 13211 via K10plus + TU Delft record)
@inproceedings{WPP22eval,
  author    = {Lichao Wu and Guilherme Perin and Stjepan Picek},
  title     = {On the Evaluation of Deep Learning-Based Side-Channel Analysis},
  booktitle = {Constructive Side-Channel Analysis and Secure Design -- {COSADE} 2022},
  editor    = {Josep Balasch and Colin O'Flynn},
  series    = {Lecture Notes in Computer Science}, volume = {13211},
  pages     = {49--71}, publisher = {Springer}, year = {2022},
  doi       = {10.1007/978-3-030-99766-3_3},
}
% verified (ePrint page; still a preprint)
@misc{WWKLPBP20,
  author       = {Lichao Wu and L{\'e}o Weissbart and Marina Kr{\v{c}}ek and Huimin Li and Guilherme Perin and Lejla Batina and Stjepan Picek},
  title        = {On the Attack Evaluation and the Generalization Ability in Profiling Side-channel Analysis},
  howpublished = {Cryptology {ePrint} Archive, Paper 2020/899},
  year         = {2020},
}
% verified (Crossref)
@inproceedings{Massey94,
  author    = {James L. Massey},
  title     = {Guessing and Entropy},
  booktitle = {Proceedings of the 1994 IEEE International Symposium on Information Theory ({ISIT})},
  pages     = {204}, year = {1994},
  doi       = {10.1109/ISIT.1994.394764},
}

% ===================== NEW: decoding / XOR convolution =====================

% verified (Crossref); LNCS 8873 = Part I via K10plus (do not confuse with 8874)
@inproceedings{VGS14,
  author    = {Nicolas Veyrat-Charvillon and Beno{\^\i}t G{\'e}rard and Fran{\c{c}}ois-Xavier Standaert},
  title     = {Soft Analytical Side-Channel Attacks},
  booktitle = {Advances in Cryptology -- {ASIACRYPT} 2014, Part {I}},
  series    = {Lecture Notes in Computer Science}, volume = {8873},
  pages     = {282--296}, publisher = {Springer}, year = {2014},
  doi       = {10.1007/978-3-662-45611-8_15},
}
% verified (Crossref)
@article{GGSB20,
  author  = {Qian Guo and Vincent Grosso and Fran{\c{c}}ois-Xavier Standaert and Olivier Bronchain},
  title   = {Modeling Soft Analytical Side-Channel Attacks from a Coding Theory Viewpoint},
  journal = {IACR Transactions on Cryptographic Hardware and Embedded Systems},
  volume  = {2020}, number = {4}, pages = {209--238}, year = {2020},
  doi     = {10.46586/tches.v2020.i4.209-238},
}
% verified (Crossref); WHT-domain XOR factor verified in source (simple-crypto/SCALib)
@article{SCALib,
  author  = {Ga{\"e}tan Cassiers and Olivier Bronchain},
  title   = {{SCALib}: A Side-Channel Analysis Library},
  journal = {Journal of Open Source Software},
  volume  = {8}, number = {86}, pages = {5196}, year = {2023},
  doi     = {10.21105/joss.05196},
}
% verified (Crossref). UNVERIFIED: that the paper itself evaluates via WHT (abstract only read).
@article{UIT+25,
  author  = {Rei Ueno and Akira Ito and Yosuke Todo and Akiko Inoue and Kazuhiko Minematsu and Hibiki Ishikawa and Naofumi Homma},
  title   = {All You Need is {XOR}-Convolution: A Generalized Higher-Order Side-Channel Attack with Application to {XEX/XE}-based Encryptions},
  journal = {IACR Transactions on Cryptographic Hardware and Embedded Systems},
  volume  = {2025}, number = {3}, pages = {317--360}, year = {2025},
  doi     = {10.46586/tches.v2025.i3.317-360},
}

% ===================== NEW: overfitting / early stopping =====================

% verified (Crossref; LNCS 12244 via K10plus). Use the LNCS title (ePrint title lacks "Profiled").
@inproceedings{RZCBH20,
  author    = {Damien Robissout and Gabriel Zaid and Brice Colombier and Lilian Bossuet and Amaury Habrard},
  title     = {Online Performance Evaluation of Deep Learning Networks for Profiled Side-Channel Analysis},
  booktitle = {Constructive Side-Channel Analysis and Secure Design -- {COSADE} 2020},
  editor    = {Guido Marco Bertoni and Francesco Regazzoni},
  series    = {Lecture Notes in Computer Science}, volume = {12244},
  pages     = {200--218}, publisher = {Springer}, year = {2021},
  doi       = {10.1007/978-3-030-68773-1_10},
}
% verified (Crossref; LNCS 12910 via K10plus)
@inproceedings{PBP21,
  author    = {Guilherme Perin and Ileana Buhan and Stjepan Picek},
  title     = {Learning When to Stop: A Mutual Information Approach to Prevent Overfitting in Profiled Side-Channel Analysis},
  booktitle = {Constructive Side-Channel Analysis and Secure Design -- {COSADE} 2021},
  editor    = {Shivam Bhasin and Fabrizio De Santis},
  series    = {Lecture Notes in Computer Science}, volume = {12910},
  pages     = {53--81}, publisher = {Springer}, year = {2021},
  doi       = {10.1007/978-3-030-89915-8_3},
}
% verified (Crossref)
@article{PCP20,
  author  = {Guilherme Perin and {\L}ukasz Chmielewski and Stjepan Picek},
  title   = {Strength in Numbers: Improving Generalization with Ensembles in Machine Learning-based Profiled Side-channel Analysis},
  journal = {IACR Transactions on Cryptographic Hardware and Embedded Systems},
  volume  = {2020}, number = {4}, pages = {337--364}, year = {2020},
  doi     = {10.46586/tches.v2020.i4.337-364},
}

% ===================== NEW: multi-task learning (ML) =====================

% verified (PMLR page)
@inproceedings{Standley20,
  author    = {Trevor Standley and Amir Zamir and Dawn Chen and Leonidas Guibas and Jitendra Malik and Silvio Savarese},
  title     = {Which Tasks Should Be Learned Together in Multi-task Learning?},
  booktitle = {Proceedings of the 37th International Conference on Machine Learning ({ICML})},
  series    = {Proceedings of Machine Learning Research}, volume = {119},
  pages     = {9120--9132}, year = {2020},
}
% verified (PMLR page)
@inproceedings{GradNorm18,
  author    = {Zhao Chen and Vijay Badrinarayanan and Chen-Yu Lee and Andrew Rabinovich},
  title     = {{GradNorm}: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks},
  booktitle = {Proceedings of the 35th International Conference on Machine Learning ({ICML})},
  series    = {Proceedings of Machine Learning Research}, volume = {80},
  pages     = {794--803}, year = {2018},
}
% verified (Crossref pages; author order from CVF open access -- Crossref reverses it)
@inproceedings{KGC18,
  author    = {Alex Kendall and Yarin Gal and Roberto Cipolla},
  title     = {Multi-task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition ({CVPR})},
  pages     = {7482--7491}, year = {2018},
  doi       = {10.1109/CVPR.2018.00781},
}
% verified (Crossref metadata; body not read)
@inproceedings{XGGGF22,
  author    = {Derrick Xin and Behrooz Ghorbani and Justin Gilmer and Ankush Garg and Orhan Firat},
  title     = {Do Current Multi-Task Optimization Methods in Deep Learning Even Help?},
  booktitle = {Advances in Neural Information Processing Systems ({NeurIPS})},
  volume    = {35}, pages = {13597--13609}, year = {2022},
}
% verified (Crossref metadata). UNVERIFIED: last-author name form (Crossref "Pawan K Mudigonda").
@inproceedings{KDKWK22,
  author    = {Vitaly Kurin and Alessandro De Palma and Ilya Kostrikov and Shimon Whiteson and M. Pawan Kumar},
  title     = {In Defense of the Unitary Scalarization for Deep Multi-Task Learning},
  booktitle = {Advances in Neural Information Processing Systems ({NeurIPS})},
  volume    = {35}, pages = {12169--12183}, year = {2022},
}
```

Some verification remains open. The LNCS volume numbers could not be read from Springer pages because of bot blocking. Each is corroborated by a reference list or a library catalogue, but 16207 (SAC 2025) rests on a single search snippet, so check it in a browser. Three claims rest on secondary sources or abstracts only: Cristiani et al.'s 100M-trace figure, UIT+25's use of the WHT, and DLWYAK's recombination layer. A final Google Scholar pass for 2025–2026 worst-case ASCADv2 results was not possible by script. The ePrint, arXiv and Semantic Scholar sweeps found none after GPAM.

## Conclusion

The audit inverts the internal review's risk ranking. The citation it suspected, DDPM24, is sound. The real exposure is three attribution slips around MS23, where the draft credits MS23 with things that come from the ANSSI release (the 500k/10k split), from GPAM ("first threat"), or from the draft's own evaluation protocol ("GE<1 on all bytes"). Each is a one-line fix, but together they would let a reviewer who knows MS23 question the "same number of traces as the original paper" headline. The honest version of that headline is "protocol-matched parity at 8.8× fewer parameters on a different test split", and it is still a clean result. The GPAM learning-rate claim deserves more caution than the draft gives it. Because the released code's configurations use 10× the Table 1 learning rates, "does not learn at the published learning rate" can be read as a mismatch between optimizer and learning rate rather than a property of the model. Results.tex should present it as Table 1's rate under Adafactor, and a small learning-rate sweep would remove the issue.

The positioning work has a clear benefit. Once the draft cites Zaid, Wouters, Picek, Masure, Ito and Standley for the general points, what remains distinctive is easier to see: no prior work audits where a multi-task SCA network's parameters sit, attributes key information causally through oracle head swaps, or analyses the gradient of an affine decoder used as a training loss. No prior work names the tie-convention false positive either, and it is worth stating plainly as the draft's own finding. Resolving OvC also gives the intro a defensible "not claimed" sentence. The draft holds no record over anyone, and no one holds a comparable full-key record over MS23 in the standard worst-case setting.
