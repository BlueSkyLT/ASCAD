# ASCADv2 related work and bibliography: verification for the TCHES draft (paper/tches)

Scope: claims in `sections/intro.tex`, `sections/related.tex`, the last paragraph of `sections/ladder.tex` and `sections/pitfalls.tex`, plus the matching `refs.bib` entries. Checked 2026-11-02.

How the texts were read:
- **Full text, local PDFs in `papers/`:** Scoop (TCHES 2025), "Is it Really Broken?", Marquet & Oswald (the COSADE/ePrint 2023/006 final version), Diffuse Some Noise (SAC 2025 pre-proceedings copy), GPAM, DS-NSCA, MS23 (ePrint 2021/592).
- **Full text, Wayback Machine copies of ePrint PDFs:** 2023/724, 2023/1615 (OvC), 2023/006 v1 (snapshot of 2023-03-26), 2023/611, 2024/1437, 2025/1309, 2022/493. eprint.iacr.org serves PDFs behind a Cloudflare challenge, but the HTML abstract pages could be fetched directly.
- **Full text, arXiv:** 2608.30105.
- **Metadata:** Crossref API for DOIs and pages. LNCS volume numbers come from secondary sources: web-search snippets of Springer and library catalogues, plus reference lists of the papers above. link.springer.com and dblp blocked scripted access, so no LNCS volume was read off a Springer page.

## Q1. Scoop (Rousselot, Heydemann, Masure, Migairou; TCHES 2025(3):56-80): what does it claim on ASCADv2?

### Takeaway
Scoop explicitly claims "the first non-worst-case attack on the ASCADv2 dataset", with no mask labels, shuffling left on, and no shared-mask assumption. The attack recovers **one key byte**: GE reaches 0 after about 250k attack traces (single-hidden-layer MLP, about 231M parameters) or 300k (fine-tuned MLP). The attack set was carved from the *profiling* file. The bib entry is correct.

### Cited Findings
- Abstract: "Scoop also performs the first non-worst-case attack on the ASCADv2 dataset." Introduction: "on the ASCADv2 dataset (Section 6.3), using Scoop, we perform a successful non-worst-case attack, which, to the best of our knowledge, has not been reached before". Footnote 4: "A gray box attack has been achieved [WPP23], although in a non-comparable setting". — [Scoop, TCHES 2025](https://doi.org/10.46586/tches.v2025.i3.56-80) (local `papers/Scoop_TCHES2025_3_03.pdf`, p. 58-59)
- Section 6.3 says: "To our knowledge, no non-worst-case attack have been successful on ASCADv2. Only one attack has been successful so far [WPP23] ... the multiplicative mask is shared among all the elements in the AES state. Moreover, Wu et al. deactivated the loop shuffling mechanism ... Hereafter, we do not rely on any of these assumptions." — [Scoop §6.3](https://doi.org/10.46586/tches.v2025.i3.56-80)
- Setup. Profiling uses 200k traces (the first 200k of the profiling file), validation uses the official 10k, and attack uses 300k (the *last* 300k of the profiling file). Five all-zero traces were removed (indices 99,999; 199,999; ...). The models are MLPs only ("we discard costly architectures such as CNNs and Transformers"). — [Scoop §6.3.1, fn. 16](https://doi.org/10.46586/tches.v2025.i3.56-80)
- Table 5 gives the results. The fine-tuned MLP has Na = 300k and validation loss 7.9998. The single-hidden-layer MLP has Na = 250k and validation loss 7.9997. The conclusion repeats "the first successful non-worst-case attack on ASCADv2". — [Scoop §6.3.2, §7](https://doi.org/10.46586/tches.v2025.i3.56-80)
- The paper does not name the targeted byte. "Is it Really Broken?" says it was "the 4-th byte". — [ePrint 2026/338 §4.3](https://eprint.iacr.org/2026/338)
- Acknowledgements: "thank Sengim Karayalcin and Stjepan Picek, for helping in detecting the bug in the ASCADv2 dataset." — [Scoop](https://doi.org/10.46586/tches.v2025.i3.56-80)
- The ePrint version has a different title: "Scoop: An Optimizer for Profiling Attacks against Higher-Order Masking" (2025/498, received 2025-03-16, listed as Preprint). — [ePrint 2025/498](https://eprint.iacr.org/2025/498)
- Crossref confirms the metadata: Nathan Rousselot, Karine Heydemann, Loïc Masure, Vincent Migairou; TCHES vol. 2025, no. 3, pp. 56-80; DOI 10.46586/tches.v2025.i3.56-80; published 2025-06-05. — [Crossref](https://api.crossref.org/works/10.46586/tches.v2025.i3.56-80)

### Inferences
- The draft's intro ("moved to the harder non-worst-case setting [Scoop], where a recent claim of success was shown to be a false positive") and related work ("Scoop claimed the first non-worst-case attack") are **VERIFIED**. For precision, add "on one key byte".

### Gaps
- Exactly which intermediate was labelled (unmasked Sbox output of byte 4, by plaintext index or by processing slot) is not stated in Scoop. The "bug in the ASCADv2 dataset" is not described.

## Q2. "Is it Really Broken? The Failure of DL-SCA Scoring Metrics under Non-Uniform Priors" (ePrint 2026/338)

### Takeaway
The paper exists. It is ePrint 2026/338 by Rousselot, Heydemann, Masure, Migairou and Strullu, received 2026-02-20 and still listed as a Preprint. It shows that Scoop's ASCADv2 result is a false positive. With the first MLP layer zeroed, GE still converges, because the targeted byte (the 4th) has a non-uniform value distribution that the model learned as a prior. The posterior distinguisher equals the maximum-likelihood distinguisher only under a uniform label prior. The paper proposes a prior-normalised "Asymptotically Optimal Distinguisher" (AOD). It says nothing specific about worst-case masked (uniform) labels. The draft's uniform-label argument is the draft's own inference, and it is consistent with the paper's Lemma 1.

### Cited Findings
- The ePrint page lists the title "Is it Really Broken? The Failure of DL-SCA Scoring Metrics under Non-Uniform Priors". Authors: Nathan Rousselot, Karine Heydemann, Loïc Masure, Vincent Migairou, Rémi Strullu. "Publication info: Preprint." History: "2026-02-20: received; 2026-02-23: approved". — [ePrint 2026/338](https://eprint.iacr.org/2026/338)
- The abstract says the paper "investigates a recent, claimed state-of-the-art attack on the ASCADv2 dataset ... which we demonstrate to be a false positive ... the model learned no actual side-channel leakage. Instead, it exploited a statistical bias in the intermediate value distribution." — [ePrint 2026/338](https://eprint.iacr.org/2026/338)
- The target is Scoop. Section 3.1 is titled "Scoop Attack on ASCADv2" and covers the single-hidden-layer MLP with 231M parameters. Reference [12] is "Rousselot ... Scoop: An optimizer for profiling attacks against higher-order masking. Cryptology ePrint Archive (2025)". — [ePrint 2026/338 §3](https://eprint.iacr.org/2026/338) (local `papers/Is it really broken.pdf`)
- The mechanism, part 1 (input ablation): "by setting all weights of the first layer of the multi-layer perceptron (MLP) to zero (effectively removing any influence from the input traces), the guessing entropy keeps converging towards 1 ... the model built a shortcut, such as learning the prior distribution of the intermediate values bytes." — [ePrint 2026/338 §3.2](https://eprint.iacr.org/2026/338)
- The mechanism, part 2 (theory). Lemma 1 says the posterior distinguisher is proportional to the ML distinguisher when Z is uniform. Corollary 1 says: "Z ≁ U ... the prior distribution of Z will skew the score on the bias on intermediate values, potentially making the estimated guessing entropy converge despite no actual leakage being learned." The paper adds that the problem is "exacerbated in highly-secured implementations, where the relevant information is scarce and lies in higher-order statistics". — [ePrint 2026/338 §4.1](https://eprint.iacr.org/2026/338)
- The mechanism, part 3 (data). A χ² uniformity test on ASCADv2: "the 4-th byte, used as the target intermediate value in the attack from [12], shows a significant bias (p-value < 0.05) ... the 6-th byte is uniform". — [ePrint 2026/338 §5.1, Fig. 3](https://eprint.iacr.org/2026/338)
- Remedies proposed. Pre-emptive: the χ² test and a "Null Benchmark". Post-mortem: input-layer ablation, prediction-entropy monitoring, gradient visualisation. The AOD is d(key) = Σ_i log( y_i[z_{i,key}] / Pr(Z = z_{i,key}) ), using the empirical prior of the attack set. The conclusion says evaluations "should systematically report bias metrics and utilize AOD or any prior agnostic distinguisher". — [ePrint 2026/338 §5-§6, §9](https://eprint.iacr.org/2026/338)
- On worst-case settings, the only statement is that checking gradient visualisation with an expert (ANOVA) analysis "requires the evaluator to be in a worst-case setting (practitioner has access to shares during the profiling)". No statement says masked or worst-case labels are immune. — [ePrint 2026/338 §5.2](https://eprint.iacr.org/2026/338)
- The paper describes the extracted ASCADv2 file as "500,000 profiling traces and 15,000 attack traces". OvC and MS23 users report 10,000 attack traces, so this inconsistency is in the source. — [ePrint 2026/338 §3](https://eprint.iacr.org/2026/338); [OvC §4.1](https://eprint.iacr.org/2023/1615)
- An independent confirmation comes from the SoK by Karayalcin, Krček and Picek, which reproduced Scoop as a case study. Its footnote 10 says: "we were informed that the attacks were not valid due to an inconsistency in the dataset which authors will release a note on". — [ePrint 2025/1309](https://eprint.iacr.org/2025/1309) (Wayback copy, version updated 2026-06-01)

### Inferences
- The draft's "shown to be a false positive [ReallyBroken]" is **VERIFIED**. The mechanism should be named precisely: a non-uniform prior of the targeted unmasked intermediate (byte 4), exploited through the posterior/log-likelihood distinguisher.
- pitfalls.tex says: "because the worst-case targets C_j are masked, their label distribution is uniform, so the label-prior false positive identified by [ReallyBroken] cannot arise here". This is **not stated in ReallyBroken**. It is the draft's own argument, and it follows from Lemma 1 / Corollary 1 when the masks are uniform. The draft should say it in its own voice ("by their Lemma 1 ...") and not imply that ReallyBroken says it.
- Caveat for that argument (inference). Mask priors (α, β, π) do not depend on the key, so even if those labels were slightly non-uniform they cannot produce a key-dependent bias. Marquet & Oswald (2023/611 §2.6) note that "a few zeros are present" in the multiplicative mask. The draft's shuffled-posterior null and uniform-C control match ReallyBroken's "null benchmark" and ablation tests. Citing that correspondence, and optionally reporting AOD-GE, would strengthen the section.
- The last paragraph of ladder.tex ("the negative results of [ReallyBroken] for the attack claimed in [Scoop] are not contradicted") is consistent with the source.

### Gaps
- No peer-reviewed venue was found for 2026/338. Keep it as @misc ePrint.
- The SoK footnote calls the cause "an inconsistency in the dataset", while ReallyBroken calls it label-prior bias. The two descriptions are compatible but not identical.

## Q3. Marquet & Oswald, "Exploring Multi-Task Learning in the Context of Masked AES Implementations", COSADE 2024

### Takeaway
The bibliographic metadata is **VERIFIED**: LNCS 14595, pp. 93-112, DOI 10.1007/978-3-031-57543-3_6, editors Romain Wacquez and Naofumi Homma. The setting is also **VERIFIED**, with one caveat. The labels are only the unmasked S-box inputs and outputs (t_j, s_j), and shuffling is disabled. However, the mask points of interest were selected by SNR on the masks, so mask knowledge was used for feature selection. Best full key: 16 traces (mean 17.6; 0.5 of seeds succeed).

### Cited Findings
- Crossref metadata: Thomas Marquet; Elisabeth Oswald; "Exploring Multi-task Learning in the Context of Masked AES Implementations"; LNCS, Constructive Side-Channel Analysis and Secure Design; pp. 93-112; 2024; book editors Romain Wacquez, Naofumi Homma. — [Crossref chapter](https://api.crossref.org/works/10.1007/978-3-031-57543-3_6); [Crossref book](https://api.crossref.org/works/10.1007/978-3-031-57543-3)
- LNCS volume 14595. Scoop's reference list: "COSADE 2024, volume 14595 of LNCS, pages 93-112". The Fraunhofer catalogue and dblp agree (via search snippet). — [Scoop references](https://doi.org/10.46586/tches.v2025.i3.56-80); [Fraunhofer publica](https://publica.fraunhofer.de/handle/publica/485486)
- The ePrint page shows "Published elsewhere. COSADE 2024"; last of 4 revisions 2024-02-06. — [ePrint 2023/006](https://eprint.iacr.org/2023/006)
- Abstract: "under the assumption that the attacker cannot access masks during training ... we propose models that defeat all masks of the affine masking on ASCAD-v2 for the first time." — [ePrint 2023/006](https://eprint.iacr.org/2023/006)
- Dataset: "the shuffling is disabled in this work ... We extract from the traces the points of interest of the masks using SnR analysis ... training (Nt = 450k traces), validation (Nv = 45k traces), and attack (Na = 5k traces)". Traces are subsampled from the raw 1M-sample traces with a moving average. — [Marquet & Oswald §2.2](https://eprint.iacr.org/2023/006) (local `papers/Twin_Unmasked_MTL_Marquet_Oswald_COSADE2024.pdf`)
- Result (§5.3, Table 3). The md model reaches n_win/n_seeds = 0.5, mean T_win 17.6, best 16. Quote: "Our best model recovers the full key in only 16 traces and is ... the best attack on ASCAD-v2, even in this simplified scenario, where PoIs from the masks are assumed and permutations are disabled." The model learns t_j through (rm⊗t_j⊕rin, rm, rin) and s_j jointly, with GF(2^8) recombination layers and low-level weight sharing. — [Marquet & Oswald §5.3](https://eprint.iacr.org/2023/006)
- An earlier ASCAD-v2 experiment in the same paper (§5.1) is not mask-free: "Knowledge of rm is given to the network during profiling and attack". — [Marquet & Oswald §5.1](https://eprint.iacr.org/2023/006)

### Inferences
- The draft's sentence "Marquet and Oswald use multi-task learning without mask labels but with shuffling disabled" is **VERIFIED**. Consider adding "and with mask points of interest chosen by SNR on the masks", since Scoop does not count this work as non-worst-case and the authors themselves call it a "simplified scenario".

### Gaps
- None for the metadata. The LNCS volume was not read from a Springer page, but it is corroborated by two independent secondary sources.

## Q4. Wu, Perin, Picek, "Not so Difficult in the End: Breaking the Lookup Table-based Affine Masking Scheme" (ePrint 2023/724; SAC 2023)

### Takeaway
The metadata is **VERIFIED**: SAC 2023, LNCS 14201, pp. 82-96, DOI 10.1007/978-3-031-53368-6_5, editors Carlet, Mandal and Rijmen, published 2024. The content is also **VERIFIED**. It is a non-profiled, non-deep-learning side-channel collision attack on the original ANSSI ASCADv2 traces, with shuffling disabled. It exploits the α/β shared by all 16 bytes. About 70k traces recover all 15 key differences, leaving 2^8 brute force. A second attack, a correlation attack on GF(0), is only partly successful.

### Cited Findings
- Crossref metadata: Lichao Wu; Guilherme Perin; Stjepan Picek; "Not so Difficult in the End: Breaking the Lookup Table-Based Affine Masking Scheme"; LNCS, Selected Areas in Cryptography – SAC 2023; pp. 82-96; 2024; editors Claude Carlet, Kalikinkar Mandal, Vincent Rijmen. — [Crossref](https://api.crossref.org/works/10.1007/978-3-031-53368-6_5); [Crossref book](https://api.crossref.org/works/10.1007/978-3-031-53368-6)
- LNCS volume 14201. Scoop's reference: "SAC 2023, volume 14201 of LNCS, pages 82-96". The AUT library catalogue agrees (search snippet). Diffuse Some Noise's reference also gives pp. 82-96 (Springer, 2024). — [Scoop references](https://doi.org/10.46586/tches.v2025.i3.56-80); [AUT catalogue](https://librarysearch.aut.ac.nz/vufind/Record/1891499)
- The ePrint page shows "Published elsewhere. Minor revision. Selected Areas in Cryptography (SAC)"; revised 2023-09-27. An earlier ePrint title was "Not so difficult in the end: Breaking the ASCADv2 dataset", and the COSADE paper and ePrint 2023/1109 cite it under that title. — [ePrint 2023/724](https://eprint.iacr.org/2023/724); [Marquet & Oswald ref. 21](https://eprint.iacr.org/2023/006)
- "Note that we disable the shuffling countermeasure and concentrate solely on masking schemes ... one would need approximately 120 times more traces to overcome a true random permutation." — [ePrint 2023/724 §1](https://eprint.iacr.org/2023/724)
- The paper classes its attack as non-profiled: "side-channel collision attack ... [is] also considered a non-profiling SCA". The attack uses the original ANSSI traces. — [ePrint 2023/724 §2, §5](https://eprint.iacr.org/2023/724)
- Result: "When attacking with 30 000 traces, only k1⊕k2 and k13⊕k14 cannot be successfully recovered ... With around 70 000 attack traces, all δ* ... can be recovered ... the entropy of the key is reduced to 256". — [ePrint 2023/724 §5.1](https://eprint.iacr.org/2023/724)
- Semantic Scholar lists the venue of the SAC chapter as "ACM Symposium on Applied Computing", which is wrong; SAC here is Selected Areas in Cryptography. Avoid auto-importing this BibTeX. — [Semantic Scholar API](https://api.semanticscholar.org/graph/v1/paper/DOI:10.1007/978-3-031-53368-6_5)

### Inferences
- The draft's "Wu et al. mount non-profiled collision attacks ... without shuffling" is **VERIFIED**. The bib entry should cite the SAC 2023 chapter (see the BibTeX below), not the ePrint.

### Gaps
- How shuffling was "disabled" on already-shuffled captures is not explained in 2023/724. Diffuse Some Noise, which follows Wu et al., says it disabled shuffling "by manipulating plaintexts a posteriori".

## Q5. "Karayalcin et al. mount non-profiled collision attacks, again without shuffling" with cite DDPM24 (Diffuse Some Noise): mis-citation?

### Takeaway
This is **not a mis-citation**. Diffuse Some Noise (Karayalçin, Perin, Picek) runs Wu et al.'s non-profiled collision attack on ASCADv2, with shuffling disabled, on traces denoised by a diffusion model. It reports about 40% fewer traces: all key-delta GE < 1 at 60k traces, versus three deltas still unrecovered at 100k without denoising. The authors in the bib entry are correct. The bib entry is out of date: the paper is now published at SAC 2025 (LNCS 16207, pp. 724-752, DOI 10.1007/978-3-032-10536-3_27, 2026). The sentence slightly overstates the contribution: Karayalcin et al. *reuse* Wu et al.'s collision attack and contribute the denoising. No other Karayalcin et al. paper attacks ASCADv2.

### Cited Findings
- Abstract: "for a collision attack against the ASCADv2 dataset, we reduced the number of traces required to retrieve the key by 40%". — [ePrint 2024/966](https://eprint.iacr.org/2024/966)
- §5: "We focus on the collision attacks as described by Wu et al. [46] ... We include this attack as it is the only attack that can break the ASCADv2 dataset in a non-profiled setting [footnote: With shuffling disabled] ... for these attacks, the shuffling countermeasure is disabled, and we simulate a fixed attack key for the profiling set ... using diffused traces can successfully reduce GE for all key-delta candidates below 1 using 60 000 traces, while three of the deltas are not fully recovered using 100 000 traces for the original traces." For the SNR analysis the authors also state "we disable shuffling by manipulating plaintexts a posteriori". — [SAC 2025 pre-proceedings PDF](https://www.sacworkshop.org/SAC25/preproceedings/sac2025-2-paper1.pdf) (local `papers/ddpm_sac2025-2-paper1.pdf`)
- The same paper also cites Cristiani et al. [8], who "showcased successful attacks against the same implementation, but they require a different acquisition campaign with significantly more traces". — [SAC 2025 pre-proceedings](https://www.sacworkshop.org/SAC25/preproceedings/sac2025-2-paper1.pdf)
- Crossref metadata: Sengim Karayalçin; Guilherme Perin; Stjepan Picek; "Diffuse Some Noise: Diffusion Models for Measurement Noise Removal in Side-Channel Analysis"; LNCS, Selected Areas in Cryptography – SAC 2025; pp. 724-752; 2026; DOI 10.1007/978-3-032-10536-3_27; book editors Christina Boura, Atefeh Mashatan, Ali Miri. — [Crossref](https://api.crossref.org/works/10.1007/978-3-032-10536-3_27)
- SAC 2025 is LNCS volume 16207 (32nd SAC, Toronto, 13-15 Aug 2025), according to a search snippet citing the Springer book page. — [Springer book page (via search)](https://link.springer.com/10.1007/978-3-032-10536-3)
- The ePrint 2024/966 page lists the same three authors and "Preprint", received 2024-06-15. — [ePrint 2024/966](https://eprint.iacr.org/2024/966)
- Other Karayalcin papers checked:
  - The conditional-GAN paper "It's a Kind of Magic" (ePrint 2023/1108) has no ASCADv2 mention in its full text.
  - The SoK (ePrint 2025/1309) only reproduces Scoop's ASCADv2 plot and does not mount its own attack.
  - "An End-to-end Plaintext-based Side-channel Collision Attack without Trace Segmentation" (Wu, Tiran, Perin, Picek; ePrint 2023/1109) mentions ASCADv2 only in its reference list.
  - Sources: [ePrint 2023/1108](https://eprint.iacr.org/2023/1108); [ePrint 2025/1309](https://eprint.iacr.org/2025/1309); [ePrint 2023/1109](https://eprint.iacr.org/2023/1109)

### Inferences
- Suggested rewording: "Wu et al. [WPP23] mount a non-profiled collision attack with shuffling disabled, and Karayalcin et al. [DDPM] cut its trace count by about 40% by denoising with diffusion models."
- Update the bib key or entry to the SAC 2025 chapter. Diacritic: Crossref and the PDF use "Karayalçin"; ePrint uses "Karayalcin".

### Gaps
- The LNCS volume 16207 comes from a search snippet of the Springer page, not a direct page read. The DOI and pages are verified via Crossref.

## Q6. Gammell & Roy, arXiv:2608.30105, OPTIMIST at CHES 2026

### Takeaway
**VERIFIED.** The paper exists: arXiv:2608.30105v1, submitted 2026-08-31, authors Jimmy Gammell and Kaushik Roy, comment "Accepted to the OPTIMIST Workshop '26 at CHES 2026". It attacks uncropped ASCADv1f, ASCADv1r and CHES-CTF-2018, not ASCADv2.

### Cited Findings
- Title "A Simple Transformer Pipeline for Full-Key Side-Channel Attacks on Uncropped Datasets"; authors Jimmy Gammell, Kaushik Roy; published 2026-08-31; categories cs.CR, cs.LG. Comment: "Accepted to the OPTIMIST Workshop '26 at CHES 2026. 6 pages, 1 figure. Code can be found at https://github.com/jimgammell/simple-transformer-pipeline-for-sca". — [arXiv API record](https://export.arxiv.org/api/query?id_list=2608.30105); [arXiv abs](https://arxiv.org/abs/2608.30105)
- Abstract: "We release our implementation, training recipes, and pretrained weights for uncropped ASCADv1f, ASCADv1r, and CHES-CTF-2018 which achieve performance competitive with previously-reported results, while using less than 10GB of VRAM and requiring at most 3.34 hours of training on a single NVIDIA A6000." A full-text search finds no "ASCADv2". — [arXiv:2608.30105](https://arxiv.org/abs/2608.30105)

### Inferences
- The draft's sentence "a plain transformer for full-key attacks on uncropped ASCADv1 and CHES-CTF traces" is accurate. Consider "CHES-CTF-2018".

### Gaps
- No DOI or proceedings page was found for the OPTIMIST workshop. Cite as arXiv.

## Q7. Maghrebi, ePrint 2020/436, and whether MS23 builds on it

### Takeaway
**VERIFIED.** The title and number are exact: "Deep Learning based Side-Channel Attack: a New Profiling Methodology based on Multi-Label Classification", single author Houssem Maghrebi, received 2020-04-19, still a Preprint. MS23 explicitly says it "revisit[s] and leverage[s] the multi-task learning approach, introduced by Maghrebi in 2020". It changes Maghrebi's sigmoid multi-label outputs to one softmax per target.

### Cited Findings
- The ePrint page lists the title above, author Houssem Maghrebi, "Publication info: Preprint", "2020-04-19: received". The abstract proposes "a new profiling methodology for DL-SCA based on the so-called multi-label classification" and says the strategy "can be applied as well to train several intermediate operations at once". — [ePrint 2020/436](https://eprint.iacr.org/2020/436)
- MS23 abstract: "Our attacks revisit and leverage the multi-task learning approach, introduced by Maghrebi in 2020". Its §3 cites "[Mag20] Houssem Maghrebi ... IACR Cryptol. ePrint Arch., 2020:436". On the change MS23 makes: "we replace the 2^{nφ} × N output branches ... made of sigmoids by N Softmax". MS23 also says the methodology "is not practically limited to the joint learning of two variables, as previously claimed by Maghrebi". — [ePrint 2021/592](https://eprint.iacr.org/2021/592) (local `papers/MS23_Masure_Strullu_ePrint2021_592.pdf`)

### Inferences
- related.tex says "following the multi-label approach of [Maghrebi20]". That is acceptable. "Revisiting the multi-label (multi-task) approach of [Maghrebi20]" matches MS23's own wording more closely. Remove the "TODO: verify title/number" note.

### Gaps
- No peer-reviewed version of 2020/436 was found.

## Q8. Systematic 2023-2026 search: who attacks ASCADv2, in which setting? Are "no newer worst-case full-key result" and "the reference model has not been revisited" true?

### Takeaway
Among *peer-reviewed* worst-case full-key results with all randomness labelled during profiling only, GPAM is the only one after MS23. OvC (preprint) reports a **single-byte** TGE0, not a full key. The claim "no newer worst-case full-key result" is therefore **essentially true but needs two qualifications**:
1. Marquet & Oswald's ePrint 2023/006 **v1** (Jan 2023, superseded) claimed a new best of about 24-25 traces for the full key with "full knowledge about the randomness during training", from raw traces with their own PoIs.
2. Their AIHWS 2023 paper (LNCS 13907) reports about 21 traces with rm and π known in *both* profiling and attack.

The claim "the worst-case reference model itself has not been revisited" is **partly contradicted**. UNIDLE/HierNet (ePrint 2024/1437) explicitly critiques the "capacity bottleneck" and model size of MS23's multi-task MultiResNetSCA design, and benchmarks MultiResNetSCA-2 with a model one tenth the size. It does this on its own Arduino shuffled dataset, not on ASCADv2. No paper was found that re-audits MultiResNetSCA-1's parameters on ASCADv2.

### Cited Findings
Inventory of ASCADv2 / ANSSI-AES attack papers, 2020-2026. WC = mask labels in profiling; NWC = no mask labels.

| Paper | Setting | Shuffling | Input | Reported result |
|---|---|---|---|---|
| MS23, JCE 13(2) 2023 / ePrint 2021/592 | WC (α, β, C, π) | on | 15k extracted | ~60 traces, full key |
| Marquet & Oswald, ePrint 2023/006 **v1** (Jan 2023, superseded) | WC: "For all efforts we assumed full knowledge about the randomness during training" | on (π learned/labelled) | own SNR PoIs from raw traces (rm, rin, rout, masked t and s, π) | "recovering the key in around 24 traces, with a single model for all key bytes"; hierarchical model "uniquely reveals the correct secret key with 25 traces" |
| Marquet & Oswald, AIHWS 2023 (ePrint 2023/611) | rm and π given in profiling **and** attack; rin unlabelled | effectively off (π known) | PoIs of t_j and rin | full key ~21 traces (as summarised in their COSADE paper); 1000 attacks of 1000 traces each |
| OvC (Kulkarni, Verneuil, Picek, Batina), ePrint 2023/1615, preprint | WC (additive and multiplicative masks, π in profiling) | on | 15k extracted | TGE0 = 47 for **the first key byte**, GE over 100 experiments, 10k attack traces |
| GPAM, TCHES 2024(3) | WC (c[i], rm, rout, p[i]) | on | raw 1M samples | "roughly 80 traces to have GE of all key bytes under 1" |
| Wu, Perin, Picek, SAC 2023 | non-profiled, no masks | off | ANSSI original traces, per-byte segments | collision: all key deltas at ~70k |
| Karayalçin, Perin, Picek, SAC 2025 | non-profiled collision (after Wu et al.) + diffusion denoising | off | 1,600-sample concatenated segments | all key-delta GE<1 at 60k (vs. >100k undenoised) |
| Marquet & Oswald, COSADE 2024 | no mask labels, but mask PoIs by SNR | off | subsampled raw | full key 16 (mean 17.6), half the seeds |
| Masure, Cristiani, Lecomte, Standaert, TCHES 2023(1):32-59 (ePrint 2022/493) | NWC scheme-aware MLP | artificially off (relabelled with permutation seeds) | 15k extracted | NWC failed; "the same model trained in a worst-case scenario could get a positive PI" |
| Scoop, TCHES 2025(3) | claimed NWC | on | 15k extracted | 1 byte, ~250k-300k traces; **false positive** per ePrint 2026/338 |
| DS-NSCA (Li, Gong, Tang, Li), TCHES 2026(2):659-685 | non-profiled DL | not addressed | 500 points | "None of these methods ... succeeded within 500,000 traces" |
| Cristiani, Lecomte, Hiscock, Maurine, IEEE Access 10 (2022) | non-profiled joint-moments regression | on (per Wu et al.) | own acquisition | broke the implementation with 100M traces (as summarised by Wu et al.) |

- The quotations from 2023/006 v1 ("around 24 traces ... previous state of the art was set by [MS21] with 60 traces in a similar setting"; §4.3.6 "we assumed full knowledge about the randomness during training") are from the Wayback snapshot of 2023-03-26. — [Wayback ePrint 2023/006 (2023-03 copy)](http://web.archive.org/web/20230326015100/https://eprint.iacr.org/2023/006.pdf)
- The sources for the 2023/611 row disagree about the randomness assumptions behind the 24-trace result:
  - 2023/611 describes [6] (2023/006 v1) as setting "a new best attack on the ASCADv2 dataset in the scenario where randomness is fully assumed ... the full key at 24 traces". Its own experiments "give to both approaches the knowledge of the multiplicative mask rm and the permutations. Therefore the only randomness not assumed during profiling and attack is the additive mask rin." — [ePrint 2023/611 §2, §5.1](https://eprint.iacr.org/2023/611)
  - OvC instead says that Marquet et al.'s 24 traces used "partial randomness knowledge about multiplicative masks and permutation indices but omitted additive mask information", on a different dataset of 7,181 PoIs (Zenodo 7885814). It also says the results cannot be compared directly. — [OvC §1](https://eprint.iacr.org/2023/1615)
- AIHWS metadata (Crossref): Thomas Marquet, Elisabeth Oswald, "A Comparison of Multi-task Learning and Single-Task Learning Approaches", LNCS, Applied Cryptography and Network Security Workshops, pp. 121-138, 2023, DOI 10.1007/978-3-031-41181-6_7. The volume is LNCS 13907 (ACNS 2023 Satellite Workshops, AIHWS), per a search snippet of the Springer and Hacettepe catalogue. — [Crossref](https://api.crossref.org/works/10.1007/978-3-031-41181-6_7); [ePrint 2023/611](https://eprint.iacr.org/2023/611); [Springer (via search)](https://link.springer.com/978-3-031-41180-9)
- OvC TGE0 definition: "we conducted 100 separate experiments using unique sets of Na measurements ... TGE0 can be conceptualized as any minimum value for i (≤ Na) for which ... GE = 0". For ASCADv2, "Our analysis targets the first key byte ... in the attack phase, we use 10,000 traces." Table 1 lists ASCAD-V2: MS23 = 60, OvC = 47. — [OvC §3-§4](https://eprint.iacr.org/2023/1615) (Wayback copy; ePrint lists "Preprint", revised 2024-01-16)
- GPAM: "need roughly 80 traces to have GE of all key bytes under 1, comparable with the 60 traces" of MS23. Targets are c[i], rm, rout, p[i], on all 1,000,000 points. — [GPAM §7.1](https://doi.org/10.46586/tches.v2024.i3.472-499)
- UNIDLE/HierNet (Hajra, Mukhopadhyay, Chatterjee; ePrint 2024/1437; preprint, last revised 2025-07-07). It discusses "Overcoming the Capacity Bottleneck: In multi-task DL models, the output of the shared trunk network is typically reduced to a single vector ... parameters often scaling as O(d²) ... more susceptible to overfitting". Against shuffling plus random delay it reports HierNet TGE1 = 3-5 versus 23-158 for MultiResNetSCA-2, "with model sizes nearly one-tenth". The dataset is the authors' own "software implementation of AES with a shuffling countermeasure on an Arduino Uno R3", not ASCADv2. ASCADv2 appears only in Fig. 1, as an illustration of the 16 SubBytes segments. — [ePrint 2024/1437](https://eprint.iacr.org/2024/1437)
- MCLS23: "we could not get effective attacks using scheme-aware models with simple MLPs as branch models, whereas the same model trained in a worst-case scenario could get a positive PI ... no successful attack has ever been reported using non-worst-case models". — [ePrint 2022/493](https://eprint.iacr.org/2022/493); metadata [Crossref](https://api.crossref.org/works/10.46586/tches.v2023.i1.32-59)
- DS-NSCA: "To the best of our knowledge, there are currently no public results showing whether DL-NPA methods can break ASCADv2". It cites Scoop [RHMM25] as having broken ASCADv2, so it predates or ignores 2026/338. — [DS-NSCA](https://doi.org/10.46586/tches.v2026.i2.659-685) (local PDF)
- Cristiani et al. metadata: Valence Cristiani, Maxime Lecomte, Thomas Hiscock, Philippe Maurine, "Fit the Joint Moments: How to Attack Any Masking Scheme", IEEE Access 10:127412-127427, 2022, DOI 10.1109/ACCESS.2022.3222760. The 100M-trace result is from Wu et al.'s summary and was not read in the original. — [Crossref](https://api.crossref.org/works/10.1109/ACCESS.2022.3222760); [ePrint 2023/724 §3](https://eprint.iacr.org/2023/724)
- Search coverage:
  - IACR ePrint full-site search for "ASCADv2", "ASCAD-v2", "ASCAD v2" (titles and abstracts) returns only 2021/592, 2023/006, 2023/611, 2023/724, 2023/1615, 2024/966, 2025/498, 2026/338. — [ePrint search](https://eprint.iacr.org/search?q=ASCADv2)
  - Citations of MS23 (29 in Semantic Scholar), and citers of WPP23, Scoop and GPAM, were scanned. Candidates with full text checked turned out to use ASCADv2 only in illustrations or reference lists, or not at all: 2024/1437, 2023/1109, 2023/1108, 2024/124, EstraNet (TCHES 2024), the Mamba-CNN paper (Entropy 2025), and 2026 ePrints 2026/2224, 2026/2358, 2026/2544, 2026/1468. — [Semantic Scholar citations of MS23](https://api.semanticscholar.org/graph/v1/paper/DOI:10.1007/s13389-023-00311-7/citations)
  - The arXiv API search for "ASCADv2" found no attack paper after GPAM. — [arXiv API](https://export.arxiv.org/api/query?search_query=all:ASCADv2&sortBy=submittedDate)
- The SoK (Karayalcin, Krček, Picek; ePrint 2025/1309; preprint) calls ASCADv2 a dataset where "due to the low leakage of some shares, it remains difficult to say whether it is possible to learn the mask recombination in practice", and notes "successful attacks against ASCADv2 in [22]" when share knowledge is assumed. — [ePrint 2025/1309](https://eprint.iacr.org/2025/1309)

### Inferences
- The draft's related-work section is **missing**:
  1. Marquet & Oswald AIHWS 2023 (rm and π known; ~21 traces), plus a note on the superseded 24-trace worst-case claim in 2023/006 v1.
  2. MCLS23 (non-worst-case scheme-aware attempt failed; worst-case positive PI).
  3. DS-NSCA (non-profiled DL fails within 500k).
  4. UNIDLE/HierNet, directly relevant to the parameter-audit and "shared dense bottleneck" narrative, and the closest thing to a critique of the MultiResNet design.
  5. Optionally Bronchain & Standaert TCHES 2020 (the original analysis of the ANSSI code) and Cristiani et al. 2022 (non-profiled, separate acquisition).
  6. The SoK footnote, as independent confirmation of the Scoop invalidity.
- Suggested rewording of "We found no newer worst-case full-key result on ASCADv2": "Apart from GPAM, we found no peer-reviewed full-key result on ASCADv2 in the standard worst-case setting (all randomness labelled in profiling only). Marquet and Oswald report about 21 traces with α and π known in profiling and attack [AIHWS23], and OvC reports a single-byte TGE0 of 47 [OvC]."
- Suggested rewording of "The worst-case reference model itself has not been revisited": "has not been re-audited on ASCADv2". Hajra et al. [UNIDLE] point to the capacity bottleneck of such multi-task trunks and compare against MultiResNetSCA-2 on a different target.
- OvC numbers: "OvC reports TGE0 47 with randomness labels" is **VERIFIED**, but the figure is for one key byte (the first), with GE over 100 attacks on 10k attack traces. MS23's 60 is a full-key figure. This resolves the TODO in pitfalls.tex. OvC's "47 vs 60" is not a like-for-like full-key comparison, and the draft should say "single-byte TGE0 47". OvC is still only an ePrint preprint, last revised 2024-01-16, with no published version found.
- The "TCHES/CHES/CARDIS/COSADE" sweep turned up no 2025-2026 conference paper with a new worst-case ASCADv2 attack. Coverage is limited by blocked dblp, Springer and Google Scholar (see Gaps).

### Gaps
- Google Scholar could not be queried by script, and dblp and Springer were bot-blocked. The final "manual Google Scholar pass" TODO therefore remains open. Semantic Scholar citation lists are incomplete: for example, they do not list OvC as citing WPP23.
- The exact trace count for 2023/611's ASCADv2 attack is shown only in figures (Figs. 6-8), which pdftotext could not extract. "About 21" comes from the authors' own COSADE summary.
- Bronchain & Standaert (TCHES 2020, DOI 10.46586/tches.v2020.i2.1-25) was verified only by title and DOI via Crossref search. Its attack numbers were not read.
- CHES 2026 / TCHES 2026 issue 4 contents were not enumerated page by page.

## Q9. Entry-by-entry verdicts on refs.bib and corrected BibTeX

### Takeaway
All entries exist and all author lists are correct. Changes needed:
- **WPP23**: switch to the SAC 2023 chapter.
- **DDPM24**: switch to the SAC 2025 chapter.
- **MarquetOswald24**: add LNCS 14595, pp. 93-112.
- **ASCAD**: pages 163-188 verified; add the DOI.
- **SMY09**: add LNCS 5479, pp. 443-461, DOI.
- **ResNet**: add pp. 770-778 and DOI.
- **OvC**: still a preprint, so keep it as @misc.
- **Maghrebi20**: title and number verified; remove the TODO.
- **Add** Marquet & Oswald AIHWS 2023, MCLS23, DS-NSCA and UNIDLE (and optionally the SoK and Bronchain & Standaert).

### Cited Findings
| Key | Verdict | Source |
|---|---|---|
| MS23 | VERIFIED (JCE 13(2):129-147, 2023, DOI 10.1007/s13389-023-00311-7; online 2023-03-10) | [Crossref](https://api.crossref.org/works/10.1007/s13389-023-00311-7) |
| GPAM | VERIFIED (TCHES 2024(3):472-499) | [Crossref](https://api.crossref.org/works/10.46586/tches.v2024.i3.472-499) |
| Scoop | VERIFIED | [Crossref](https://api.crossref.org/works/10.46586/tches.v2025.i3.56-80) |
| ReallyBroken | VERIFIED (ePrint 2026/338, preprint, five authors) | [ePrint](https://eprint.iacr.org/2026/338) |
| OvC | VERIFIED as ePrint 2023/1615, preprint (no published version found) | [ePrint](https://eprint.iacr.org/2023/1615) |
| MarquetOswald24 | VERIFIED; fill in LNCS 14595, pp. 93-112 | [Crossref](https://api.crossref.org/works/10.1007/978-3-031-57543-3_6) |
| WPP23 | VERIFIED; cite LNCS 14201, pp. 82-96, 2024 | [Crossref](https://api.crossref.org/works/10.1007/978-3-031-53368-6_5) |
| DDPM24 | VERIFIED authors; now SAC 2025, LNCS 16207, pp. 724-752, 2026 | [Crossref](https://api.crossref.org/works/10.1007/978-3-032-10536-3_27) |
| Maghrebi20 | VERIFIED title and number | [ePrint](https://eprint.iacr.org/2020/436) |
| ASCAD | VERIFIED JCE 10(2):163-188; DOI 10.1007/s13389-019-00220-8 (online 2019-11-30, issue 2020) | [Crossref](https://api.crossref.org/works/10.1007/s13389-019-00220-8) |
| SMY09 | VERIFIED pp. 443-461, DOI 10.1007/978-3-642-01001-9_26, editor Antoine Joux; LNCS 5479 per OvC's reference list | [Crossref](https://api.crossref.org/works/10.1007/978-3-642-01001-9_26); [OvC refs](https://eprint.iacr.org/2023/1615) |
| ResNet | VERIFIED CVPR 2016, pp. 770-778, DOI 10.1109/CVPR.2016.90 | [Crossref](https://api.crossref.org/works/10.1109/CVPR.2016.90) |
| GammellRoy26 | VERIFIED arXiv:2608.30105; OPTIMIST '26 at CHES 2026 | [arXiv](https://arxiv.org/abs/2608.30105) |

BibTeX-ready entries for the changed and new items. Every field below was verified, with the source of the LNCS volume noted in each comment.

```bibtex
@inproceedings{MarquetOswald24,
  author    = {Thomas Marquet and Elisabeth Oswald},
  title     = {Exploring Multi-task Learning in the Context of Masked {AES} Implementations},
  booktitle = {Constructive Side-Channel Analysis and Secure Design -- {COSADE} 2024},
  editor    = {Romain Wacquez and Naofumi Homma},
  series    = {Lecture Notes in Computer Science},
  volume    = {14595},   % volume: Scoop ref list + Fraunhofer/dblp snippet
  pages     = {93--112},
  publisher = {Springer},
  year      = {2024},
  doi       = {10.1007/978-3-031-57543-3_6},
}

@inproceedings{WPP23,
  author    = {Lichao Wu and Guilherme Perin and Stjepan Picek},
  title     = {Not so Difficult in the End: Breaking the Lookup Table-Based Affine Masking Scheme},
  booktitle = {Selected Areas in Cryptography -- {SAC} 2023},
  editor    = {Claude Carlet and Kalikinkar Mandal and Vincent Rijmen},
  series    = {Lecture Notes in Computer Science},
  volume    = {14201},   % volume: Scoop ref list + AUT catalogue snippet
  pages     = {82--96},
  publisher = {Springer},
  year      = {2024},
  doi       = {10.1007/978-3-031-53368-6_5},
}

@inproceedings{DDPM24,
  author    = {Sengim Karayal{\c{c}}in and Guilherme Perin and Stjepan Picek},
  title     = {Diffuse Some Noise: Diffusion Models for Measurement Noise Removal in Side-Channel Analysis},
  booktitle = {Selected Areas in Cryptography -- {SAC} 2025},
  editor    = {Christina Boura and Atefeh Mashatan and Ali Miri},
  series    = {Lecture Notes in Computer Science},
  volume    = {16207},   % volume: Springer book page via search snippet
  pages     = {724--752},
  publisher = {Springer},
  year      = {2026},
  doi       = {10.1007/978-3-032-10536-3_27},
  note      = {Preprint: IACR ePrint 2024/966},
}

@inproceedings{MarquetOswald23AIHWS,
  author    = {Thomas Marquet and Elisabeth Oswald},
  title     = {A Comparison of Multi-task Learning and Single-Task Learning Approaches},
  booktitle = {Applied Cryptography and Network Security Workshops -- {ACNS} 2023 Satellite Workshops ({AIHWS})},
  series    = {Lecture Notes in Computer Science},
  volume    = {13907},   % volume: Springer/Hacettepe catalogue via search snippet
  pages     = {121--138},
  publisher = {Springer},
  year      = {2023},
  doi       = {10.1007/978-3-031-41181-6_7},
  note      = {Preprint: IACR ePrint 2023/611},
}

@article{MCLS23,
  author  = {Lo{\"\i}c Masure and Valence Cristiani and Maxime Lecomte and Fran{\c{c}}ois-Xavier Standaert},
  title   = {Don't Learn What You Already Know: Scheme-Aware Modeling for Profiling Side-Channel Analysis against Masking},
  journal = {IACR Transactions on Cryptographic Hardware and Embedded Systems},
  volume  = {2023},
  number  = {1},
  pages   = {32--59},
  year    = {2023},
  doi     = {10.46586/tches.v2023.i1.32-59},
}

@article{DSNSCA26,
  author  = {Di Li and Zheng Gong and Yufeng Tang and Chun Li},
  title   = {{DS-NSCA}: Differentiable Search for Non-Profiling Side-channel Analysis},
  journal = {IACR Transactions on Cryptographic Hardware and Embedded Systems},
  volume  = {2026},
  number  = {2},
  pages   = {659--685},
  year    = {2026},
  doi     = {10.46586/tches.v2026.i2.659-685},
}

@misc{UNIDLE,
  author       = {Suvadeep Hajra and Debdeep Mukhopadhyay and Soumi Chatterjee},
  title        = {{UNIDLE}: A Unified Framework for Deep Learning-based Side-channel Analysis},
  howpublished = {Cryptology ePrint Archive, Paper 2024/1437},
  year         = {2024},
}

@misc{SoKDLSCA,
  author       = {Sengim Karayalcin and Marina Krcek and Stjepan Picek},
  title        = {{SoK}: Deep Learning-based Physical Side-channel Analysis},
  howpublished = {Cryptology ePrint Archive, Paper 2025/1309},
  year         = {2025},
}

@article{ASCAD,
  author  = {Ryad Benadjila and Emmanuel Prouff and R{\'e}mi Strullu and Eleonora Cagli and C{\'e}cile Dumas},
  title   = {Deep learning for side-channel analysis and introduction to {ASCAD} database},
  journal = {Journal of Cryptographic Engineering},
  volume  = {10}, number = {2}, pages = {163--188}, year = {2020},
  doi     = {10.1007/s13389-019-00220-8},
}

@inproceedings{SMY09,
  author    = {Fran{\c{c}}ois-Xavier Standaert and Tal Malkin and Moti Yung},
  title     = {A Unified Framework for the Analysis of Side-Channel Key Recovery Attacks},
  booktitle = {Advances in Cryptology -- {EUROCRYPT} 2009},
  editor    = {Antoine Joux},
  series    = {Lecture Notes in Computer Science},
  volume    = {5479},   % volume: OvC reference list
  pages     = {443--461},
  publisher = {Springer},
  year      = {2009},
  doi       = {10.1007/978-3-642-01001-9_26},
}

@inproceedings{ResNet,
  author    = {Kaiming He and Xiangyu Zhang and Shaoqing Ren and Jian Sun},
  title     = {Deep Residual Learning for Image Recognition},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition ({CVPR})},
  pages     = {770--778},
  year      = {2016},
  doi       = {10.1109/CVPR.2016.90},
}

@misc{GammellRoy26,
  author       = {Jimmy Gammell and Kaushik Roy},
  title        = {A Simple Transformer Pipeline for Full-Key Side-Channel Attacks on Uncropped Datasets},
  howpublished = {arXiv:2608.30105},
  note         = {Accepted to the OPTIMIST Workshop '26 at CHES 2026},
  year         = {2026},
}
```

Sources for the new entries:
- [Crossref 10.1007/978-3-031-41181-6_7](https://api.crossref.org/works/10.1007/978-3-031-41181-6_7)
- [Crossref 10.46586/tches.v2023.i1.32-59](https://api.crossref.org/works/10.46586/tches.v2023.i1.32-59)
- [Crossref 10.46586/tches.v2026.i2.659-685](https://api.crossref.org/works/10.46586/tches.v2026.i2.659-685)
- [ePrint 2024/1437](https://eprint.iacr.org/2024/1437)
- [ePrint 2025/1309](https://eprint.iacr.org/2025/1309)

### Inferences
- Claim-level verdicts for the draft text:
  - Intro: "Later work either changed the input (GPAM on raw 1M-sample traces), the training objective (OvC) ...": **VERIFIED**. GPAM uses all 1,000,000 points. OvC replaces the multi-task classifier with a language-model, multimodal and metric-learning classifier; "objective" is a fair shorthand.
  - Intro: "The worst-case reference model itself has not been revisited": **PARTLY CONTRADICTED** (see Q8: UNIDLE, Marquet & Oswald v1). Soften it.
  - Related: "OvC reports TGE0 47 with randomness labels": **VERIFIED, single byte**.
  - Related: "GPAM ... full-key GE<1 in about 80 traces": **VERIFIED**.
  - Related: "We found no newer worst-case full-key result": **essentially VERIFIED with qualifications** (Q8).
  - Related: Marquet & Oswald sentence: **VERIFIED** (add the mask-PoI caveat).
  - Related: Wu et al.: **VERIFIED**.
  - Related: Karayalcin et al. [DDPM24]: **VERIFIED, not a mis-citation**. Reword (Q5).
  - Related: Scoop / ReallyBroken: **VERIFIED**.
  - Related: Gammell & Roy: **VERIFIED**.
  - Ladder, last paragraph: **consistent with the sources**.
  - Pitfalls: the uniform-label argument is the draft's inference, not a statement in ReallyBroken; the OvC TODO is resolved (single-byte TGE0, 100 experiments).
- One BibTeX style point: the MS23 entry title in the draft uses "ANSSI's", while Crossref has "ANSSI’s". This is cosmetic.

### Gaps
- None of the LNCS volume numbers (14595, 14201, 16207, 13907, 5479) could be read off Springer directly because of bot blocking. Each is corroborated by at least one reference list or catalogue snippet. Check them on link.springer.com in a browser before submission.
