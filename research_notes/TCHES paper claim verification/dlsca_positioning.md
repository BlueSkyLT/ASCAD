# DL-SCA literature positioning for the draft TCHES paper (general, non-ASCADv2-specific claims)

Scope: prior work that a TCHES reviewer would expect us to cite for claims (1)–(7) of the draft at
`paper/tches/` (sections/*.tex and refs.bib read on 2026-11-02). For each theme: what the prior work says,
whether it **supports**, **anticipates** (already established the general point) or **contradicts** our
claim, and how our contribution differs. The last section is a consolidated BibTeX list.

How verification was done:
- Metadata (authors, title, venue, pages, DOI) came from the Crossref REST API (`api.crossref.org/works/<DOI>`).
  Every entry tagged "verified" resolves at `https://doi.org/<DOI>`.
- LNCS volume numbers came from the K10plus union catalogue (SRU query by the volume's e-ISBN, MARC field 490).
- NeurIPS/ICML/CVPR page ranges come from the official proceedings pages and BibTeX.
- Abstracts were read from Crossref (TCHES abstracts are deposited there) or the arXiv API.
- Local PDFs: `docs/papers/MS23_Masure_Strullu_ePrint2021_592.pdf` and
  `docs/papers/Twin_Unmasked_MTL_Marquet_Oswald_COSADE2024.pdf`.
- **Limitation:** tches.iacr.org, eprint.iacr.org, dblp.org, hal.science and link.springer.com all served bot
  challenges (Anubis or a Springer login redirect) to both curl and WebFetch. Full texts of most TCHES papers
  could not be read, so any claim below that goes beyond an abstract says so.

---

## Q1. Small/efficient CNNs, model size, and over-parameterisation (claim 1: 94.5% of parameters in dense heads; slim heads give 8.8× smaller model at equal attack performance)

### Takeaway
Earlier work already established that SCA networks "do not need to be very complex". Zaid et al. report models up to 31,810× less complex, Wouters et al. cut a further 52% of parameters, and RL or hyperparameter search finds small models. The general claim "big SCA models are overparameterised" is therefore **anticipated**. Ours differs in three ways: (a) the setting is a 34-head *multi-task* ResNet on a masked and shuffled target; (b) we show *where* the excess sits (the first dense layer of each head, 90.8%), keeping the deep trunk; (c) we tie the excess to checkpoint fragility. Karayalcin et al. argue that minimal architectures may be *insufficient* for harder countermeasures and that deep ResNets help on large inputs. That partly cuts against "small is enough", but it fits our "keep the trunk, slim the heads" message.

### Cited Findings
- **Zaid et al. (TCHES 2020(1), pp. 1–36)**:
  - They propose a methodology for "building efficient CNN architectures in terms of attack efficiency and network complexity". They report networks "up to 25 times more efficient than previous state-of-the-art while their complexity is up to 31810 times smaller".
  - They conclude "CNN networks do not need to be very complex to perform well in the side-channel context".
  - Source: [doi:10.46586/tches.v2020.i1.1-36](https://doi.org/10.46586/tches.v2020.i1.1-36) (abstract via Crossref).
  - Role: **anticipates** the general "smaller is enough" point.
- **Wouters et al. (TCHES 2020(3), pp. 147–168)**:
  - A critical review of Zaid et al. "We can trivially reduce their number of parameters on average by 52%, while maintaining a similar performance."
  - The filter size is "not strictly related to the amount of misalignment".
  - Increasing filter size and the number of convolutions *improves* performance.
  - Source: [doi:10.46586/tches.v2020.i3.147-168](https://doi.org/10.46586/tches.v2020.i3.147-168).
  - Role: **supports** our point that parameter counts can be cut without loss. It is also a precedent for a paper whose main contribution is revisiting a reference architecture, which is the genre of our paper.
- **Rijsdijk, Wu, Perin, Picek (TCHES 2021(3), pp. 677–707)**:
  - Q-learning hyperparameter tuning with SCA-metric rewards finds CNNs "exhibiting top performance while having small numbers of trainable parameters".
  - Source: [doi:10.46586/tches.v2021.i3.677-707](https://doi.org/10.46586/tches.v2021.i3.677-707).
  - Role: **anticipates or supports**.
- **Perin, Wu, Picek (TCHES 2022(4), pp. 828–861)**:
  - Attacking raw measurements "with small deep neural networks also provides optimal models".
  - "The hyperparameter search always indicates a successful model with up to eight hidden layers … suggesting that complex models are not required".
  - The worst-case feature-selection scenario is explicitly compared with knowing the mask shares during profiling.
  - Source: [doi:10.46586/tches.v2022.i4.828-861](https://doi.org/10.46586/tches.v2022.i4.828-861).
  - Role: **supports**.
- **Wu, Perin, Picek, "I Choose You" (IEEE TETC 12(2):546–557, 2024)**:
  - Automated hyperparameter tuning for DL-SCA.
  - Source: [doi:10.1109/TETC.2022.3218372](https://doi.org/10.1109/TETC.2022.3218372).
  - Only metadata was verified; the abstract was not read.
- **Perin, Wu, Picek, "Gambling for Success: The Lottery Ticket Hypothesis in Deep Learning-Based Side-Channel Analysis"**:
  - Chapter in *Artificial Intelligence for Cybersecurity*, Springer Advances in Information Security, 2022, pp. 217–241.
  - Source: [doi:10.1007/978-3-030-97087-1_9](https://doi.org/10.1007/978-3-030-97087-1_9).
  - Role: pruning/sparsity precedent. Only metadata was verified.
- **Karayalcin, Perin, Picek, "Resolving the Doubts: On the Construction and Use of ResNets for Side-Channel Analysis" (Mathematics 11(15):3265, 2023, MDPI)**:
  - The paper exists, but the venue is an MDPI journal, not a crypto venue.
  - Their framing: the community has focused on "(if possible) minimal" MLP/CNN architectures, but "as targets with more complex countermeasures become available, these minimal architectures may be insufficient".
  - Their ResNets are "significantly deeper than commonly seen … require minimal hyperparameter tuning" and "work especially well when the number of profiling traces and features in a trace is large".
  - Source: [doi:10.3390/math11153265](https://doi.org/10.3390/math11153265).
  - Role: **partially contradicts** an extreme "small is enough" reading. It **supports** keeping a deep trunk on 15k-sample inputs with 400k profiling traces.
- **MS23 (local PDF, ePrint 2021/592)**:
  - MS23 replaced the one-layer branches of Maghrebi's multi-label network with a per-branch MLP ("Leaf MLP") on top of ResNetSCA. "Dense layers are added to the bottom of the network for classification".
  - MS23's footnote 20 notes that for 1-D data the trainable parameters of a convolutional stack "roughly equals the size of the receptive field, regardless the number of layers". This is consistent with the trunk being cheap.
  - Source: [doi:10.1007/s13389-023-00311-7](https://doi.org/10.1007/s13389-023-00311-7).
- **Lin, Chen, Yan, "Network In Network" (arXiv:1312.4400, ICLR 2014)**:
  - The classic ML precedent for replacing fully connected classifiers with global average pooling. The abstract states GAP is "easier to interpret and less prone to overfitting than traditional fully connected layers".
  - The arXiv comment reads "for iclr2014".
  - Source: [arXiv:1312.4400](https://arxiv.org/abs/1312.4400). The first part of the abstract was read via the arXiv API; the overfitting phrase is in the abstract tail and should be re-checked on the page.
  - Role: **supports** "large dense heads are where the overfitting capacity lives".

### Inferences
- A reviewer will likely say "small SCA models are known to suffice (Zaid, Wouters, Rijsdijk, Perin 2022)". The intro and related work should state the difference explicitly. Those works size *single-task* MLP/CNNs on POI-cropped or first-order targets, mostly with search. We audit a *published multi-task reference model* on an affine-masked and shuffled target. We localise the excess in the 34 per-head first dense layers, change no trunk hyperparameter, and match the published trace count.
- Citing Karayalcin et al. as the counterpoint ("deep ResNets are needed for long inputs") is useful. Our result says depth in the *trunk* is cheap (7.53M) and the size problem is the flatten→dense interface, which reconciles the two lines of work.
- NiN or GAP is the natural ML citation for "dense classifier heads dominate parameters and overfit". No SCA paper was found that makes the "dense heads hold most of the parameters" observation for a multi-task SCA network.

### Gaps
- Could not read the full text of Zaid et al. to quote whether they explicitly attribute parameter count to the dense layers (the HAL PDF was bot-blocked). Their abstract only gives aggregate complexity.
- No SCA paper was found that audits parameter distribution between trunk and heads in a multi-task network. This appears novel, but the search was limited to Crossref and abstracts.
- The "Gambling for Success" and "I Choose You" claims were not read beyond metadata.

---

## Q2. Accuracy vs. SCA metrics (claim 3: masked S-box heads at ~1.5% top-1 carry all the key information; "top-1 accuracy is the wrong measure"; even per-head log-likelihood gain is not a reliable proxy)

### Takeaway
That accuracy is a poor or deceptive proxy for SCA success is **well established**: Picek et al. 2019, Masure et al. 2020, Zhang et al. 2020 and Ito et al. 2022. So "top-1 accuracy is the wrong measure" must not be presented as a new finding. Ito et al. 2022 also show that cross-entropy/PI itself is "non-calibrated" with respect to success rate, which **supports** our observation that a negative per-head log-likelihood gain can coexist with key recovery. Our new contribution is *causal attribution inside a multi-task decoder*: oracle and uniform head swaps that show the C heads are the sole key carrier.

### Cited Findings
- **Picek, Heuser, Jovic, Bhasin, Regazzoni (TCHES 2019(1), pp. 209–237)**:
  - "Machine learning metrics (and especially accuracy as the most often used one) can be extremely deceptive."
  - The context is imbalanced Hamming-weight / Hamming-distance labels.
  - Source: [doi:10.46586/tches.v2019.i1.209-237](https://doi.org/10.46586/tches.v2019.i1.209-237).
  - Role: **anticipates** the general point.
- **Masure, Dumas, Prouff (TCHES 2020(1), pp. 348–375)**:
  - Notes that the relevance of ML-based SCA "has been questioned … based on the lack of relation between the accuracy … and common SCA metrics like the Guessing entropy or the key-discrimination success rate".
  - Proves that minimising NLL "is actually asymptotically equivalent to maximizing the Perceived Information introduced by Renauld et al. at EUROCRYPT 2011".
  - Source: [doi:10.46586/tches.v2020.i1.348-375](https://doi.org/10.46586/tches.v2020.i1.348-375).
  - Role: **supports** using NLL/validation loss (not accuracy) for checkpoint selection. This is the theoretical basis for our "best = lowest validation loss" choice.
- **Zhang, Zheng, Nan, Hu, Yu (TCHES 2020(3), pp. 73–96)**:
  - "Deep learning metrics are not consistent with side channel metrics … deceptive in foreseeing the feasibility and complexity of mounting a successful attack, especially for imbalanced data".
  - Proposes the Cross Entropy Ratio (CER) metric and loss.
  - Source: [doi:10.46586/tches.v2020.i3.73-96](https://doi.org/10.46586/tches.v2020.i3.73-96).
  - Role: **anticipates**.
- **Ito, Ueno, Homma, "Perceived Information Revisited" (TCHES 2022(4), pp. 228–254)**:
  - "The conventional CE/PI is non-calibrated and insufficient for evaluating the SCA performance … an infinite number of probability distributions with different CE/PI can achieve an identical SR."
  - Proposes effective CE/PI (ECE/EPI).
  - Source: [doi:10.46586/tches.v2022.i4.228-254](https://doi.org/10.46586/tches.v2022.i4.228-254).
  - Role: **supports** our claim that the C-head mean log-likelihood gain is not a reliable proxy (the overfitted 137.4M model has −0.48 nats and still recovers the key). It also supplies a candidate "key-discrimination measure" for the carrier.tex TODO.
  - A follow-up, "Perceived Information Revisited II" (TCHES 2025(1), pp. 450–474), exists: [doi:10.46586/tches.v2025.i1.450-474](https://doi.org/10.46586/tches.v2025.i1.450-474). Only metadata was verified.
- **Renauld, Standaert, Veyrat-Charvillon, Kamel, Flandre (EUROCRYPT 2011, LNCS 6632, pp. 109–128)**:
  - Introduced the perceived information (PI).
  - Source: [doi:10.1007/978-3-642-20465-4_8](https://doi.org/10.1007/978-3-642-20465-4_8).
- **Bronchain, Hendrickx, Massart, Olshevsky, Standaert, "Leakage Certification Revisited" (CRYPTO 2019, LNCS 11692, pp. 713–737)**:
  - Bounds model errors in evaluations (PI vs hypothetical information).
  - Source: [doi:10.1007/978-3-030-26948-7_25](https://doi.org/10.1007/978-3-030-26948-7_25).
  - Only metadata was verified; the content is described as in Masure et al. 2023's abstract.
- **Masure, Cassiers, Hendrickx, Standaert (TCHES 2023(3), pp. 522–569)**:
  - The hypothetical information "does not offer such a general bound". They introduce the "training information" and study convergence rates of profiled distinguishers, including the effect of "granting them access to the countermeasures' randomness or not".
  - Source: [doi:10.46586/tches.v2023.i3.522-569](https://doi.org/10.46586/tches.v2023.i3.522-569).
  - Role: relevant to worst-case vs. non-worst-case profiling cost (our label ladder).
- **MS23 itself (local PDF)**:
  - For the single-task first-threat CNN: "The accuracy obtained on the test set equals 1.56 %, which is clearly above the 2^−8 = 0.39 % threshold", followed by a successful attack.
  - MS23's own C-head accuracy (Table 2) is about 1.6%.
  - Role: **anticipates** "near-chance accuracy still attacks" on this very dataset.
- **Zaid et al., "Ranking Loss" (TCHES 2021(1), pp. 25–55)**:
  - Argues that cross-entropy induces approximation and estimation errors relative to SR, and proposes a learning-to-rank loss that "maximizes the success rate".
  - Source: [doi:10.46586/tches.v2021.i1.25-55](https://doi.org/10.46586/tches.v2021.i1.25-55).
  - Role: related loss-design precedent for our decoder loss.
- **Kerkhof, Wu, Perin, Picek, "Focus is Key to Success: A Focal Loss Function …" (COSADE 2022, LNCS 13211, pp. 29–48)**:
  - Source: [doi:10.1007/978-3-030-99766-3_2](https://doi.org/10.1007/978-3-030-99766-3_2).
  - Metadata only.

### Inferences
- Recommended rephrasing of carrier.tex §"Top-1 accuracy is the wrong measure": "Consistent with [Picek19, MDP20, ZZN+20, IUH22], top-1 accuracy is uninformative here; what is new is that oracle/uniform swaps of a single head family *causally* locate all key information in heads at 1.4–1.5% accuracy, and that even the per-head NLL gain (the PI estimate of [MDP20]) can be negative for a model that recovers the key, an instance of the calibration gap of [IUH22]."
- Masure et al. 2020's NLL ↔ PI equivalence is the justification for selecting checkpoints by validation loss. It also explains why the C-head NLL can mislead in our case: PI of a *single head on its own label* is not the PI of the *decoded key distribution* after mask marginalisation. That is a clean way to phrase the TODO about a key-discrimination measure.

### Gaps
- None of these papers looks at multi-task decoders. No prior work was found that uses oracle head substitution as an attribution tool in DL-SCA. This appears novel, but the search was limited.
- The full text of Ito et al. 2022 (which ECE/EPI estimator would apply to a marginalised multi-head output) was not read.

---

## Q3. Guessing entropy / success rate definitions and evaluation pitfalls (claim 6: GE, SR, traces-to-rank-0, tie handling, single vs many orderings differ by ~5×)

### Takeaway
SMY09 is confirmed as LNCS 5479, pp. 443–461. Massey 1994 is the origin of "guessing entropy". Several DL-SCA works already warn that a single attack or the arithmetic-mean GE can mislead:
- Wu, Perin, Picek (COSADE 2022) recommend reporting the standard deviation and the *median* over attacks.
- Wu et al. (ePrint 2020/899) separate "simple" from "generalized" GE.
- Perin, Wu, Picek (Algorithms 2023) study cheap GE for early stopping.

Our "~5× spread between metrics on the same posteriors" is therefore **supported and partly anticipated** in spirit. The specific quantification and the tie-handling false positive appear new. **No source was found that discusses tie-breaking conventions in rank computation.**

### Cited Findings
- **Standaert, Malkin, Yung, "A Unified Framework for the Analysis of Side-Channel Key Recovery Attacks"**:
  - EUROCRYPT 2009, LNCS 5479, pp. 443–461, Springer.
  - Sources: [doi:10.1007/978-3-642-01001-9_26](https://doi.org/10.1007/978-3-642-01001-9_26) (Crossref pages 443–461); volume 5479 confirmed in K10plus and OpenLibrary records for ISBN 978-3-642-01000-2 ([OpenLibrary](https://openlibrary.org/isbn/9783642010002.json)).
- **Massey, "Guessing and entropy", Proc. IEEE ISIT 1994, p. 204**:
  - Source: [doi:10.1109/ISIT.1994.394764](https://doi.org/10.1109/ISIT.1994.394764).
  - Role: origin of the guessing-entropy notion.
- **Wu, Perin, Picek, "On the Evaluation of Deep Learning-Based Side-Channel Analysis"**:
  - COSADE 2022, LNCS 13211, pp. 49–71.
  - Besides GE, results should include the standard deviation. "The arithmetic mean is not the best summary for guessing entropy"; they recommend the median.
  - Sources: [doi:10.1007/978-3-030-99766-3_3](https://doi.org/10.1007/978-3-030-99766-3_3); [TU Delft record](https://research.tudelft.nl/en/publications/on-theevaluation-ofdeep-learning-based-side-channel-analysis).
  - Role: **supports or anticipates** our multi-metric reporting.
- **Wu, Weissbart, Krček, Li, Perin, Batina, Picek, "On the Attack Evaluation and the Generalization Ability in Profiling Side-channel Analysis" (Cryptology ePrint 2020/899)**:
  - Defines "simple" GE (few attack traces, can mislead) vs "generalized" GE.
  - Proposes a "profiling model fitting" metric.
  - Source: [ePrint 2020/899](https://eprint.iacr.org/2020/899).
  - Status: still listed as a preprint on the ePrint page. A published version was not found. The same authors' TIFS 2023 paper "Label Correlation in Deep Learning-Based Side-Channel Analysis" ([doi:10.1109/TIFS.2023.3287728](https://doi.org/10.1109/TIFS.2023.3287728)) is a different paper.
- **Perin, Wu, Picek, "The Need for Speed: A Fast Guessing Entropy Calculation for Deep Learning-Based SCA" (Algorithms 16(3):127, 2023)**:
  - GE "can be efficiently computed during training by reducing the number of validation traces without affecting the efficiency of early stopping decisions", up to 16× faster.
  - Source: [doi:10.3390/a16030127](https://doi.org/10.3390/a16030127).
  - **Correction to the task brief:** this paper is by Perin, Wu and Picek, not "Rioja et al." No Rioja GE-estimation paper was found.
- **MS23 (local PDF)**: GE "is estimated over 500 independently repeated attacks". This confirms our A = 500 matches the reference.
- **Full-key rank estimation, for any full-key rank or GE claim**:
  - Veyrat-Charvillon, Gérard, Standaert, "Security Evaluations beyond Computing Power", EUROCRYPT 2013, LNCS 7881, pp. 126–141: [doi:10.1007/978-3-642-38348-9_8](https://doi.org/10.1007/978-3-642-38348-9_8).
  - Glowacz, Grosso, Poussier, Schüth, Standaert, "Simpler and More Efficient Rank Estimation for Side-Channel Security Assessment", FSE 2015, LNCS 9054, pp. 117–129: [doi:10.1007/978-3-662-48116-5_6](https://doi.org/10.1007/978-3-662-48116-5_6).
  - Poussier, Standaert, Grosso, "Simple Key Enumeration (and Rank Estimation) Using Histograms: An Integrated Approach", CHES 2016, LNCS 9813, pp. 61–81: [doi:10.1007/978-3-662-53140-2_4](https://doi.org/10.1007/978-3-662-53140-2_4).
  - Martin, O'Connell, Oswald, Stam, "Counting Keys in Parallel After a Side Channel Attack", ASIACRYPT 2015 (Part II), LNCS 9453, pp. 313–337: [doi:10.1007/978-3-662-48800-3_13](https://doi.org/10.1007/978-3-662-48800-3_13).
  - All metadata only.
- **Zhu & Lu, "Leading Degree: A Metric for Model Performance Evaluation and Hyperparameter Tuning in Deep Learning-Based Side-Channel Analysis" (TCHES 2025(2), pp. 333–361)**: [doi:10.46586/tches.v2025.i2.333-361](https://doi.org/10.46586/tches.v2025.i2.333-361). Metadata only; a recent metric paper that a reviewer may expect.

### Inferences
- Our metric set (GE<1, GE<0.5, GE=0, mean MTD, z) and the 35 → 165 trace spread on identical posteriors give a concrete number for the concerns of Wu–Perin–Picek 2022 and Wu et al. 2020/899. Cite both as "consistent with", not "we discovered".
- Our "per-byte GE below threshold for every byte" is not a full-key rank. If a reviewer asks for full-key GE or rank, the rank-estimation papers (Glowacz 2015, Poussier 2016, Martin 2015) are the standard references.
- On ties: the observation that a stable sort or "ties in favour" convention ranks a constant score vector first when the simulated key is 0 is a genuine, cheap-to-state pitfall. It should be presented as our own caution with no citation, since no source was found.

### Gaps
- **No published work found on tie handling in key ranking.** This was searched via web and Crossref; none surfaced.
- No paper was found that quantifies the ratio between single-ordering, worst-byte MTD and all-attack GE=0 on the same posteriors.
- No formal treatment was found of how many attacks A are needed for a given confidence on GE<1. Ito et al. bound SR/GE analytically instead (IEICE 2021 per search snippet; not verified).

---

## Q4. Multi-task / multi-label SCA and ML multi-task literature (claim 4: shared dense bottleneck suppresses low-SNR heads; loss weighting w_C)

### Takeaway
DL-SCA multi-task precedents are Maghrebi 2020 (multi-label), MS23, Marquet & Oswald (ACNS-W 2023 and COSADE 2024) and the scheme-aware DLWYAK (TCHES 2023). Marquet & Oswald explicitly note that "losses can compete as much as they can collaborate" and favour hard parameter sharing. Their positive result for *low-level* sharing (without mask labels, shuffling disabled) **partly contrasts** with our negative result for a *late dense* shared bottleneck, but the settings differ.

In ML, negative transfer and task competition under shared capacity are textbook (Caruana 1997; Ruder 2017; Standley et al. ICML 2020; Wang et al. CVPR 2019), as are gradient conflict (PCGrad) and loss balancing (GradNorm, Kendall et al.). Xin et al. and Kurin et al. (NeurIPS 2022) show that specialised MTO methods rarely beat tuned scalarisation. That **supports** reporting a simple w_C sweep rather than adopting a gradient-surgery method.

### Cited Findings
- **Maghrebi, "Deep Learning based Side-Channel Attack: a New Profiling Methodology based on Multi-Label Classification"**:
  - Cryptology ePrint Archive, Paper 2020/436. The title is verified exactly. The page lists it as "Preprint. MINOR revision" with no venue.
  - Multi-label profiling targets ~16 key bits at once and "can also train several intermediate operations together".
  - Source: [ePrint 2020/436](https://eprint.iacr.org/2020/436).
  - Our refs.bib entry is correct; the TODO can be removed.
- **Marquet & Oswald, "Exploring Multi-task Learning in the Context of Masked AES Implementations"**:
  - COSADE 2024, LNCS 14595, pp. 93–112, eds. Wacquez & Homma. Source: [doi:10.1007/978-3-031-57543-3_6](https://doi.org/10.1007/978-3-031-57543-3_6).
  - Local PDF: on ASCAD-v2 "the shuffling is disabled in this work". Masks are not available during training.
  - Their conclusion: "hard-parameter sharing allows to focus the propagation of losses towards fewer weights … However, one has to be assured of the collaboration between the targets of the network, as losses can compete as much as they can collaborate."
  - Another takeaway: "Low-level parameter sharing allows multi-task learning to benefit from the learning of multiple bytes".
  - Role: **supports** loss competition and **partly contrasts** on sharing. Their sharing is near the input, with no mask labels and no shuffling. Ours is a late shared dense bottleneck under full labels and shuffling.
- **Marquet & Oswald, "A Comparison of Multi-task Learning and Single-Task Learning Approaches"**:
  - ACNS 2023 Satellite Workshops, LNCS 13907, pp. 121–138. Source: [doi:10.1007/978-3-031-41181-6_7](https://doi.org/10.1007/978-3-031-41181-6_7).
  - Metadata only; this is their earlier work, cited as [7] in the COSADE 2024 paper.
- **Masure, Cristiani, Lecomte, Standaert, "Don't Learn What You Already Know: Scheme-Aware Modeling for Profiling Side-Channel Analysis against Masking"**:
  - TCHES 2023(1), pp. 32–59. Source: [doi:10.46586/tches.v2023.i1.32-59](https://doi.org/10.46586/tches.v2023.i1.32-59).
  - Proposes a "scheme-aware" threat model that "does not need to have access to the random nonces used by masking during the profiling phase like in a worst-case model, while it does not need to learn the masking scheme". It combines DL with that prior and can "reduce by an order of magnitude the profiling complexity".
  - Role: **anticipates** the idea behind our decoder-as-loss ladder (see Q5).
- **MS23 (local PDF)**: introduces MultiResNetSCA-1 with 34 outputs, "revisit[ing] Mahgrebi's idea", and a "Leaf MLP" per branch.
- **Caruana, "Multitask Learning", Machine Learning 28(1):41–75, 1997**: [doi:10.1023/A:1007379606734](https://doi.org/10.1023/A:1007379606734).
- **Ruder, "An Overview of Multi-Task Learning in Deep Neural Networks", arXiv:1706.05098, 2017**:
  - Introduces "the two most common methods for MTL in Deep Learning", i.e. hard vs soft parameter sharing.
  - Source: [arXiv:1706.05098](https://arxiv.org/abs/1706.05098) (arXiv API).
- **Standley, Zamir, Chen, Guibas, Malik, Savarese, "Which Tasks Should Be Learned Together in Multi-task Learning?"**:
  - ICML 2020, PMLR 119:9120–9132.
  - Conflicting objectives "often reduce overall accuracy"; competing tasks should be separated into different networks.
  - Source: [PMLR](https://proceedings.mlr.press/v119/standley20a.html).
  - Role: **supports** "unshared heads (variant B) beat a shared bottleneck (A)".
- **Wang, Dai, Póczos, Carbonell, "Characterizing and Avoiding Negative Transfer"**: CVPR 2019, pp. 11285–11294. Source: [doi:10.1109/CVPR.2019.01155](https://doi.org/10.1109/CVPR.2019.01155). Metadata only.
- **Yu, Kumar, Gupta, Levine, Hausman, Finn, "Gradient Surgery for Multi-Task Learning" (PCGrad)**:
  - NeurIPS 33 (2020), pp. 5824–5836.
  - Identifies "three conditions of the multi-task optimization landscape that cause detrimental gradient interference".
  - Sources: [NeurIPS proceedings BibTeX](https://proceedings.neurips.cc/paper_files/paper/2020/file/3fe78a8acf5fda99de95303940a2420c-Bibtex.bib); [arXiv:2001.06782](https://arxiv.org/abs/2001.06782).
- **Chen, Badrinarayanan, Lee, Rabinovich, "GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks"**:
  - ICML 2018, PMLR 80:794–803.
  - Balances per-task gradient magnitudes; the authors report it beats fixed weights and matches grid search.
  - Source: [PMLR](https://proceedings.mlr.press/v80/chen18a.html).
  - Role: directly relevant to our "C heads get 52.6% of the loss but 38.3% of the trunk gradient" audit.
- **Kendall, Gal, Cipolla, "Multi-task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics"**:
  - CVPR 2018, pp. 7482–7491.
  - Sources: [doi:10.1109/CVPR.2018.00781](https://doi.org/10.1109/CVPR.2018.00781); author order from the [CVF open-access page](https://openaccess.thecvf.com/content_cvpr_2018/html/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.html). Crossref lists the authors in reverse order, so use the CVF order.
- **Xin, Ghorbani, Gilmer, Garg, Firat, "Do Current Multi-Task Optimization Methods in Deep Learning Even Help?"**: NeurIPS 35 (2022), pp. 13597–13609. Source: [doi:10.52202/068431-0988](https://doi.org/10.52202/068431-0988). Metadata only; the claim is as the title states.
- **Kurin, De Palma, Kostrikov, Whiteson, Kumar, "In Defense of the Unitary Scalarization for Deep Multi-Task Learning"**:
  - NeurIPS 35 (2022), pp. 12169–12183. Source: [doi:10.52202/068431-0884](https://doi.org/10.52202/068431-0884). Metadata only.
  - The last author is listed as "Pawan K Mudigonda" in Crossref; he is commonly cited as M. Pawan Kumar. Check the NeurIPS page before finalising.
- **Vandenhende et al., "Multi-Task Learning for Dense Prediction Tasks: A Survey" (IEEE TPAMI, 2021)**: [doi:10.1109/TPAMI.2021.3054719](https://doi.org/10.1109/TPAMI.2021.3054719). Crossref gives early-access pages "1–1"; the final volume and pages were not verified.

### Inferences
- The carrier finding (low-SNR C heads lose the shared subspace) is a concrete SCA instance of task competition and negative transfer under hard sharing. The right framing is "consistent with Standley et al. and Marquet & Oswald's warning that losses can compete", not a new principle. The draft's TODO already says "phrase as empirical finding".
- Marquet & Oswald 2024 is the closest SCA precedent. It is not contradicted, because (i) their sharing is early or low-level while ours is a late 512-d dense bottleneck after the trunk, (ii) they disable shuffling, and (iii) they train without mask labels. The related-work section should say this in one sentence.
- Given Xin et al. and Kurin et al., a reviewer is unlikely to demand PCGrad or GradNorm baselines. Citing them together with the w_C sweep pre-empts "did you try MTO methods?".

### Gaps
- The bodies of Wang et al. 2019, Xin et al. 2022 and Kurin et al. 2022 were not read (metadata only).
- No DL-SCA paper was found that measures per-task gradient share in the trunk.

---

## Q5. Mask marginalisation / decoding, and Walsh–Hadamard XOR convolution (claim 5)

### Takeaway
Combining per-share probability distributions through the masking algebra is standard in the SASCA / belief-propagation line:
- Veyrat-Charvillon et al. ASIACRYPT 2014; Guo et al. TCHES 2020(4).
- Bronchain & Standaert TCHES 2020(2) and 2021(3).
- SCALib (JOSS 2023).

The WHT is the standard way to propagate through XOR at O(b·2^b) cost. I **verified in SCALib's source code** that its BP XOR factor runs in the Walsh–Hadamard domain. Ueno et al. (TCHES 2025(3)) recently framed profiled higher-order SCA on masked implementations as XOR-convolution distinguishers.

The prompt's description of Bronchain & Standaert TCHES 2021(3) is **wrong**: that paper targets Boolean-masked bitsliced AES and Clyde on Cortex-M0/M3. The ANSSI affine-masked and shuffled AES (the ASCADv2 implementation) was attacked in **Bronchain & Standaert TCHES 2020(2)**, "Side-Channel Countermeasures' Dissection …", with full key recovery in under 2,000 measurements. That paper is a **must-cite**: it is the earliest worst-case-style dissection of exactly our target.

Using the marginalisation as a *differentiable training loss* is close to DLWYAK's scheme-aware modelling, which **anticipates** the general idea. Our difference: an explicit affine (GF(2^8) multiplicative + additive) plus shuffling-routing decoder, its gradient analysis (soft-label/EM interpretation, zero-gradient saddle), and the label ladder.

### Cited Findings
- **Bronchain & Standaert, "Side-Channel Countermeasures' Dissection and the Limits of Closed Source Security Evaluations"**:
  - TCHES 2020(2), pp. 1–25. Source: [doi:10.46586/tches.v2020.i2.1-25](https://doi.org/10.46586/tches.v2020.i2.1-25).
  - The target is the ANSSI AES, which "combines additive and multiplicative secret sharings into an affine masking scheme that is additionally mixed with a shuffled execution".
  - A "countermeasures' dissection exploiting a mix of dimensionality reduction, multivariate information extraction and key enumeration can recover the full key with less than 2,000 measurements".
  - Certain steps "are hard to automate without implementation knowledge (even with machine learning tools)".
  - Role: **anticipates** per-share modelling plus recombination on ASCADv2's implementation. **Currently missing from refs.bib.**
- **Bronchain & Standaert, "Breaking Masked Implementations with Many Shares on 32-bit Software Platforms"**:
  - TCHES 2021(3), pp. 202–234. Source: [doi:10.46586/tches.v2021.i3.202-234](https://doi.org/10.46586/tches.v2021.i3.202-234).
  - Targets "higher-order masked software implementations of the AES and … Clyde, in ARM Cortex-M0 and M3" with "multivariate, horizontal and analytical attacks". Reports "full 128-bit key recovery in less than 10 traces for a 6-share masked AES" on the M0.
  - Not affine masking and not ANSSI.
  - A search-engine extraction of the PDF reported that it notes generic two-input factor evaluation is quadratic in the alphabet size, "reduced to roughly b·2^b for Boolean addition using the WHT" ([search result pointing to the TCHES PDF](https://tches.iacr.org/index.php/TCHES/article/download/8973/8551)). I could not open the PDF myself (bot wall), so treat this as **unverified in full text**.
- **Veyrat-Charvillon, Gérard, Standaert, "Soft Analytical Side-Channel Attacks"**:
  - ASIACRYPT 2014 (Part I), LNCS 8873, pp. 282–296. Source: [doi:10.1007/978-3-662-45611-8_15](https://doi.org/10.1007/978-3-662-45611-8_15).
  - LNCS 8873 = Part I, confirmed via K10plus for e-ISBN 978-3-662-45611-8. The print ISBN 978-3-662-45607-1 maps to Part II (8874); do not confuse them.
  - Whether this paper itself uses the WHT for XOR nodes could **not** be verified, because the full text was blocked.
- **Guo, Grosso, Standaert, Bronchain, "Modeling Soft Analytical Side-Channel Attacks from a Coding Theory Viewpoint"**:
  - TCHES 2020(4), pp. 209–238. Source: [doi:10.46586/tches.v2020.i4.209-238](https://doi.org/10.46586/tches.v2020.i4.209-238).
  - SASCA "represent the target implementation and its leakages as a code … decoded thanks to belief propagation". Introduces the LRPM. Notes it "can considerably speed up the evaluation of other countermeasures such as shuffling".
  - A search snippet of ePrint 2018/498 states that propagating through XOR on a b-bit bus costs O(b·2^b) operations ([ePrint 2018/498](https://eprint.iacr.org/2018/498.pdf)). Only the snippet was seen.
- **Cassiers & Bronchain, "SCALib: A Side-Channel Analysis Library"**:
  - Journal of Open Source Software 8(86):5196, 2023. Source: [doi:10.21105/joss.05196](https://doi.org/10.21105/joss.05196).
  - **Verified in code:** `src/scalib_ext/scalib/src/sasca/bp_compute.rs` defines `/// Walsh-Hadamard transform (non-normalized). fn slice_wht`. In `belief_propagation.rs`, `factor_xor` calls `.wht()` and states "Simply make the product if Walsh-Hadamard domain" ([bp_compute.rs](https://github.com/simple-crypto/SCALib/blob/main/src/scalib_ext/scalib/src/sasca/bp_compute.rs), [belief_propagation.rs](https://github.com/simple-crypto/SCALib/blob/main/src/scalib_ext/scalib/src/sasca/belief_propagation.rs)).
  - Role: the cleanest verifiable citation for "XOR convolution of share distributions via WHT" as standard SCA tooling.
- **Ueno, Ito, Todo, Inoue, Minematsu, Ishikawa, Homma, "All You Need is XOR-Convolution: A Generalized Higher-Order Side-Channel Attack with Application to XEX/XE-based Encryptions"**:
  - TCHES 2025(3), pp. 317–360. Source: [doi:10.46586/tches.v2025.i3.317-360](https://doi.org/10.46586/tches.v2025.i3.317-360).
  - XCCA "estimates the sum of keys from two leakages by XOR-convoluting probability distributions". "The proposed SCA distinguisher … is a generalization of higher-order SCAs, including non-collision SCAs on masked implementations."
  - Its use of the WHT is reported by a search summary only ([TCHES article page](https://tches.iacr.org/index.php/TCHES/article/view/12219)).
  - Role: recent, directly relevant citation for "decoding = XOR convolution".
- **Grosso & Standaert, "Masking Proofs Are Tight and How to Exploit it in Security Evaluations"**:
  - EUROCRYPT 2018 (Part II), LNCS 10821, pp. 385–412. Source: [doi:10.1007/978-3-319-78375-8_13](https://doi.org/10.1007/978-3-319-78375-8_13). The volume was verified in K10plus as "EUROCRYPT 2018 Part 2" (10821, not 10820).
  - ePrint 2017/116 ([link](https://eprint.iacr.org/2017/116)): evaluating a masked implementation reduces "essentially to that of an unprotected one", and SASCA can approach optimal attacks efficiently.
  - WHT use is unverified.
- **Bronchain, Durvaux, Masure, Standaert, "Efficient Profiled Side-Channel Analysis of Masked Implementations, Extended"**: IEEE TIFS 17:574–584, 2022. Source: [doi:10.1109/TIFS.2022.3144871](https://doi.org/10.1109/TIFS.2022.3144871). Metadata only; likely relevant to per-share profiling and recombination, but the content was not verified.
- **DLWYAK (TCHES 2023(1))**: see Q4. Scheme-aware DL that injects the known masking scheme instead of learning it.
  - I **could not verify** whether its recombination layer uses the WHT (the HAL PDF was bot-blocked).
  - Role: the closest precedent to "decoder used as a training loss". Must be cited and distinguished in the ladder section.
- **Scoop (TCHES 2025(3), pp. 56–80)**: already in refs.bib and verified via Crossref ([doi:10.46586/tches.v2025.i3.56-80](https://doi.org/10.46586/tches.v2025.i3.56-80)). A grep of the local PDF found no Walsh–Hadamard mention; its only "Hadamard" mention is the Hutchinson Hessian estimator.
- **MS23 (local PDF)**: a grep for "Walsh", "Hadamard", "Fourier" and "convolution" in the ePrint version found **no** WHT mention. MS23's Eq. 12 is a direct sum. Our WHT evaluation is therefore an implementation choice relative to MS23, and should be attributed to the SASCA/SCALib tradition.
- **Wu, Perin, Picek, "Not so Difficult in the End: Breaking the Lookup Table-Based Affine Masking Scheme"**:
  - SAC 2023, LNCS 14201, pp. 82–96. Source: [doi:10.1007/978-3-031-53368-6_5](https://doi.org/10.1007/978-3-031-53368-6_5).
  - This resolves the refs.bib TODO for WPP23.

### Inferences
- Suggested wording for setup.tex / ladder.tex: "computed exactly with a Walsh–Hadamard transform, as is standard for XOR factors in SASCA [VGS14, GGSB20] and implemented e.g. in SCALib [CB23]; see also [UIT+25] for XOR-convolution as a general higher-order distinguisher."
- Our affine-specific step (re-indexing by a⊗v for a ≠ 0, then summing over P_α) is not a WHT. It is a GF(2^8) multiplicative permutation, so only the β step is an XOR convolution. The text already says this correctly.
- Bronchain & Standaert 2020 should go in the related work and intro next to MS23. It is the first worst-case-style attack on the same ANSSI implementation, and its "hard to automate even with ML" remark is exactly what MS23 and our decoder address.
- Our label ladder has to be positioned against DLWYAK's scheme-aware model. In DLWYAK the masking scheme is known and the shares are not labelled. Our RCπ rung still labels α and β but drops C and π. The two are complementary; our paper should not claim novelty for "using the masking algebra inside training" in general.

### Gaps
- **Which original paper first used the WHT for XOR convolutions of share distributions in SCA?** This could not be verified from full text. Verified: SCALib's code, Guo et al.'s complexity statement (snippet only), and Ueno et al.'s framing (abstract). Unverified: VGS14, Grosso–Standaert 2018 and Bronchain–Standaert 2021 full text.
- Whether DLWYAK's recombination is WHT-based is unverified.
- The content of Bronchain et al. TIFS 2022 is unverified.

---

## Q6. Overfitting, checkpoint selection and early stopping (claim 2: 137.4M model overfits after epoch 4, last checkpoint 135–160 traces vs 70–75 for the compact model)

### Takeaway
The DL-SCA literature treats epoch selection and overfitting as a known problem and proposes SCA-specific stopping signals:
- online train-vs-validation metric (Robissout et al., COSADE 2020);
- mutual-information-based stopping (Perin, Buhan, Picek, COSADE 2021);
- fast GE for early stopping (Perin, Wu, Picek 2023);
- ensembles to reduce variance across models and seeds (Perin, Chmielewski, Picek, TCHES 2020(4)).

Our claim, "robustness to checkpoint choice comes from removing head capacity", is **supported** by this framing and is a different kind of contribution: architecture-level rather than a stopping rule. Masure et al. 2020 justify selecting checkpoints by validation NLL.

### Cited Findings
- **Robissout, Zaid, Colombier, Bossuet, Habrard, "Online Performance Evaluation of Deep Learning Networks for Profiled Side-Channel Analysis"**:
  - COSADE 2020, LNCS 12244, pp. 200–218 (published 2021, eds. Bertoni & Regazzoni). Source: [doi:10.1007/978-3-030-68773-1_10](https://doi.org/10.1007/978-3-030-68773-1_10).
  - From the [ePrint 2020/039 page](https://eprint.iacr.org/2020/039): introduces an online metric that compares training and validation performance to detect under- and overfitting. Used for early stopping, it reduces the number of traces needed by about 30% for most networks considered.
  - **Title note:** the ePrint title is "Online Performance Evaluation of Deep Learning Networks for Side-Channel Analysis"; the published LNCS title adds "Profiled". Use the LNCS title.
- **Perin, Buhan, Picek, "Learning When to Stop: A Mutual Information Approach to Prevent Overfitting in Profiled Side-Channel Analysis"**:
  - COSADE 2021, LNCS 12910, pp. 53–81, eds. Bhasin & De Santis. Source: [doi:10.1007/978-3-030-89915-8_3](https://doi.org/10.1007/978-3-030-89915-8_3).
  - From the ePrint 2020/058 summary: if training is too long the network "overfits and cannot generalize". They use the information transferred to the output layer to choose the epoch ([ePrint 2020/058](https://eprint.iacr.org/2020/058)).
- **Perin, Chmielewski, Picek, "Strength in Numbers: Improving Generalization with Ensembles in Machine Learning-based Profiled Side-channel Analysis"**:
  - TCHES 2020(4), pp. 337–364. Source: [doi:10.46586/tches.v2020.i4.337-364](https://doi.org/10.46586/tches.v2020.i4.337-364).
  - Output probabilities "are sensitive to small changes, like selecting specific test traces or weight initialization". "Ensembles increase a profiled side-channel attack's performance and reduce the variance of results".
  - Role: **supports** reporting two seeds and being cautious about single-model conclusions.
- **Perin, Wu, Picek, "The Need for Speed" (Algorithms 2023)**: "deep learning models usually suffer from overfitting due to their high fitting capacity … early stopping is employed as an efficient regularization method that requires a consistent validation metric" ([doi:10.3390/a16030127](https://doi.org/10.3390/a16030127)).
- **MS23 (local PDF)**: uses "a early-stopping strategy [GBC16], using a patience of 10 epochs", which selected epoch 9 for the first-threat CNN. This confirms early stopping is part of the reference pipeline.
- **Masure, Dumas, Prouff 2020** (see Q2): NLL ↔ PI. Validation NLL is a principled selection criterion.

### Inferences
- Our robustness result (last vs best checkpoint) is a useful complement. These works fix the *stopping rule*; we show that removing excess head capacity makes the stopping rule matter less. Reviewers may ask whether early stopping on GE (Perin 2023) or on the Robissout metric would rescue the 137.4M model's last checkpoint. "Best" (lowest validation loss) already is early stopping, so the comparison is fair.
- The negative C-head gain at the overfitted checkpoint combined with key recovery is a concrete instance of why Perin–Buhan–Picek and Robissout et al. argue for SCA-specific rather than generic validation signals.

### Gaps
- The full texts of Robissout et al. and Perin–Buhan–Picek were not read; their exact metric definitions were not checked.

---

## Q7. ResNets in SCA, and the core references to fix in refs.bib (He et al.; SMY09; ASCAD)

### Takeaway
ResNet-in-SCA precedents are Zhou & Standaert (JCEN 2020), MS23 (ResNetSCA / MultiResNetSCA-1) and Karayalcin et al. (Mathematics 2023). Long-range alternatives are Lu et al. TCHES 2021(3), EstraNet TCHES 2024(1) and GPAM TCHES 2024(3). He et al. CVPR 2016 is pp. 770–778, DOI 10.1109/CVPR.2016.90.

### Cited Findings
- **He, Zhang, Ren, Sun, "Deep Residual Learning for Image Recognition"**: CVPR 2016, pp. 770–778. Sources: [doi:10.1109/CVPR.2016.90](https://doi.org/10.1109/CVPR.2016.90); author list matches the [CVF open-access BibTeX](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html).
- **Zhou & Standaert, "Deep learning mitigates but does not annihilate the need of aligned traces and a generalized ResNet model for side-channel attacks"**: Journal of Cryptographic Engineering 10(1):85–95, 2020. Crossref "published" is 2019 (online first). Source: [doi:10.1007/s13389-019-00209-3](https://doi.org/10.1007/s13389-019-00209-3).
- **Karayalcin, Perin, Picek 2023**: see Q1 ([doi:10.3390/math11153265](https://doi.org/10.3390/math11153265)).
- **Lu, Zhang, Cao, Gu, Lu, "Pay Attention to Raw Traces: A Deep Learning Architecture for End-to-End Profiling Attacks"**: TCHES 2021(3), pp. 235–274. Source: [doi:10.46586/tches.v2021.i3.235-274](https://doi.org/10.46586/tches.v2021.i3.235-274).
- **Hajra, Chowdhury, Mukhopadhyay, "EstraNet: An Efficient Shift-Invariant Transformer Network for Side-Channel Analysis"**: TCHES 2024(1), pp. 336–374. Source: [doi:10.46586/tches.v2024.i1.336-374](https://doi.org/10.46586/tches.v2024.i1.336-374).
- **Kim, Picek, Heuser, Bhasin, Hanjalic, "Make Some Noise. Unleashing the Power of Convolutional Neural Networks for Profiled Side-channel Analysis"**: TCHES 2019(3), pp. 148–179. Source: [doi:10.46586/tches.v2019.i3.148-179](https://doi.org/10.46586/tches.v2019.i3.148-179).
- **Benadjila, Prouff, Strullu, Cagli, Dumas, "Deep learning for side-channel analysis and introduction to ASCAD database"**: JCE 10(2):163–188, 2020. Source: [doi:10.1007/s13389-019-00220-8](https://doi.org/10.1007/s13389-019-00220-8). Pages verified; add the DOI and remove the TODO.
- **Cagli, Dumas, Prouff, "Convolutional Neural Networks with Data Augmentation Against Jitter-Based Countermeasures"**: CHES 2017, LNCS 10529, pp. 45–68. Source: [doi:10.1007/978-3-319-66787-4_3](https://doi.org/10.1007/978-3-319-66787-4_3).
- **Maghrebi, Portigliatti, Prouff, "Breaking Cryptographic Implementations Using Deep Learning Techniques"**: SPACE 2016, LNCS 10076, pp. 3–26. Source: [doi:10.1007/978-3-319-49445-6_1](https://doi.org/10.1007/978-3-319-49445-6_1).
- **Picek, Perin, Mariot, Wu, Batina, "SoK: Deep Learning-based Physical Side-channel Analysis"**: ACM Computing Surveys 55(11), article 227, pp. 1–35, 2023. Source: [doi:10.1145/3569577](https://doi.org/10.1145/3569577). Crossref gives pages 1–35; the article number 227 is from memory and unverified. Role: the standard survey to cite once in the intro.
- **Chari, Rao, Rohatgi, "Template Attacks"**: CHES 2002, LNCS 2523, pp. 13–28. Source: [doi:10.1007/3-540-36400-5_3](https://doi.org/10.1007/3-540-36400-5_3).

### Inferences
- The parameters.tex sentence "a one-dimensional residual network [ResNet]" should also cite MS23's ResNetSCA lineage and Zhou & Standaert 2020 as the first SCA ResNet.

### Gaps
- Gohr et al. ResNet-for-SCA (mentioned in the brief) was not searched or verified, so it is not included.

---

## Q8. Label priors / class imbalance (claim 7: the label-prior false positive cannot arise with uniform masked labels)

### Takeaway
The imbalance literature concerns non-uniform leakage-model labels (HW/HD): Picek et al. 2019, Zhang et al. 2020 and Ito et al. TIFS 2021. "Is it really broken?" (ePrint 2026/338, in refs.bib) shows the scoring-metric failure under non-uniform priors. Masked S-box outputs C_j = α⊗S⊕β with β uniform are uniformly distributed, so the mechanism does not apply. The claim is **supported**: it follows from these works' own premise, and our shuffled-posterior null is the empirical check.

### Cited Findings
- **Picek et al. 2019**: imbalanced data arises "in the Hamming weight or Hamming distance leakage models"; they use balancing such as SMOTE ([doi:10.46586/tches.v2019.i1.209-237](https://doi.org/10.46586/tches.v2019.i1.209-237)).
- **Ito, Saito, Ueno, Homma, "Imbalanced Data Problems in Deep Learning-Based Side-Channel Attacks: Analysis and Solution"**: IEEE TIFS 16:3790–3802, 2021. Source: [doi:10.1109/TIFS.2021.3092050](https://doi.org/10.1109/TIFS.2021.3092050). Metadata only.
- **Zhang et al. 2020**: CER "works stably while … accuracy becomes rather unreliable when the training data tends to be imbalanced" ([doi:10.46586/tches.v2020.i3.73-96](https://doi.org/10.46586/tches.v2020.i3.73-96)).

### Inferences
- Add one sentence in pitfalls.tex citing Picek19 and ITUH21 alongside ReallyBroken. "Label-prior artefacts are documented for non-uniform (HW/HD) labels [Picek19, ITUH21, ReallyBroken]; worst-case masked targets are uniform by construction."

### Gaps
- ReallyBroken's precise mechanism was not re-verified here; it is in local docs/papers and already cited.

---

## Q9. Consolidated BibTeX (verified unless marked)

### Takeaway
The entries below are ready to paste. The "verified at" comments name the source. Fixes to existing refs.bib entries are marked **[FIX]**.

### Cited Findings

```bibtex
% ---------- fixes to existing refs.bib entries ----------
% [FIX] SMY09 — verified at https://doi.org/10.1007/978-3-642-01001-9_26 (pages) + K10plus/OpenLibrary (LNCS 5479)
@inproceedings{SMY09,
  author    = {Fran{\c{c}}ois-Xavier Standaert and Tal Malkin and Moti Yung},
  title     = {A Unified Framework for the Analysis of Side-Channel Key Recovery Attacks},
  booktitle = {Advances in Cryptology -- {EUROCRYPT} 2009},
  series    = {Lecture Notes in Computer Science},
  volume    = {5479},
  pages     = {443--461},
  publisher = {Springer},
  year      = {2009},
  doi       = {10.1007/978-3-642-01001-9_26},
}
% [FIX] ResNet — verified at https://doi.org/10.1109/CVPR.2016.90
@inproceedings{ResNet,
  author    = {Kaiming He and Xiangyu Zhang and Shaoqing Ren and Jian Sun},
  title     = {Deep Residual Learning for Image Recognition},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition ({CVPR})},
  pages     = {770--778},
  year      = {2016},
  doi       = {10.1109/CVPR.2016.90},
}
% [FIX] ASCAD — verified at https://doi.org/10.1007/s13389-019-00220-8 (vol 10, issue 2, pp. 163-188)
@article{ASCAD,
  author  = {Ryad Benadjila and Emmanuel Prouff and R{\'e}mi Strullu and Eleonora Cagli and C{\'e}cile Dumas},
  title   = {Deep learning for side-channel analysis and introduction to {ASCAD} database},
  journal = {Journal of Cryptographic Engineering},
  volume  = {10}, number = {2}, pages = {163--188}, year = {2020},
  doi     = {10.1007/s13389-019-00220-8},
}
% [FIX] MarquetOswald24 — verified at https://doi.org/10.1007/978-3-031-57543-3_6 ; LNCS 14595 via K10plus
@inproceedings{MarquetOswald24,
  author    = {Thomas Marquet and Elisabeth Oswald},
  title     = {Exploring Multi-task Learning in the Context of Masked {AES} Implementations},
  booktitle = {Constructive Side-Channel Analysis and Secure Design ({COSADE} 2024)},
  editor    = {Romain Wacquez and Naofumi Homma},
  series    = {Lecture Notes in Computer Science}, volume = {14595},
  pages     = {93--112}, publisher = {Springer}, year = {2024},
  doi       = {10.1007/978-3-031-57543-3_6},
}
% [FIX] WPP23 — verified at https://doi.org/10.1007/978-3-031-53368-6_5 ; LNCS 14201 via K10plus
@inproceedings{WPP23,
  author    = {Lichao Wu and Guilherme Perin and Stjepan Picek},
  title     = {Not so Difficult in the End: Breaking the Lookup Table-Based Affine Masking Scheme},
  booktitle = {Selected Areas in Cryptography -- {SAC} 2023},
  series    = {Lecture Notes in Computer Science}, volume = {14201},
  pages     = {82--96}, publisher = {Springer}, year = {2024},
  doi       = {10.1007/978-3-031-53368-6_5},
}
% [FIX] Maghrebi20 — title/number verified at https://eprint.iacr.org/2020/436 (still a preprint; remove TODO)

% ---------- Q1 model size ----------
% verified at https://doi.org/10.46586/tches.v2020.i1.1-36
@article{ZBHV20,
  author  = {Gabriel Zaid and Lilian Bossuet and Amaury Habrard and Alexandre Venelli},
  title   = {Methodology for Efficient {CNN} Architectures in Profiling Attacks},
  journal = {IACR Transactions on Cryptographic Hardware and Embedded Systems},
  volume  = {2020}, number = {1}, pages = {1--36}, year = {2020},
  doi     = {10.46586/tches.v2020.i1.1-36},
}
% verified at https://doi.org/10.46586/tches.v2020.i3.147-168
@article{WAGP20,
  author  = {Lennert Wouters and Victor Arribas and Benedikt Gierlichs and Bart Preneel},
  title   = {Revisiting a Methodology for Efficient {CNN} Architectures in Profiling Attacks},
  journal = {IACR Transactions on Cryptographic Hardware and Embedded Systems},
  volume  = {2020}, number = {3}, pages = {147--168}, year = {2020},
  doi     = {10.46586/tches.v2020.i3.147-168},
}
% verified at https://doi.org/10.46586/tches.v2021.i3.677-707
@article{RWPP21,
  author  = {Jorai Rijsdijk and Lichao Wu and Guilherme Perin and Stjepan Picek},
  title   = {Reinforcement Learning for Hyperparameter Tuning in Deep Learning-based Side-channel Analysis},
  journal = {IACR Transactions on Cryptographic Hardware and Embedded Systems},
  volume  = {2021}, number = {3}, pages = {677--707}, year = {2021},
  doi     = {10.46586/tches.v2021.i3.677-707},
}
% verified at https://doi.org/10.46586/tches.v2022.i4.828-861
@article{PWP22,
  author  = {Guilherme Perin and Lichao Wu and Stjepan Picek},
  title   = {Exploring Feature Selection Scenarios for Deep Learning-based Side-channel Analysis},
  journal = {IACR Transactions on Cryptographic Hardware and Embedded Systems},
  volume  = {2022}, number = {4}, pages = {828--861}, year = {2022},
  doi     = {10.46586/tches.v2022.i4.828-861},
}
% verified at https://doi.org/10.1109/TETC.2022.3218372 (metadata only)
@article{WPP24choose,
  author  = {Lichao Wu and Guilherme Perin and Stjepan Picek},
  title   = {I Choose You: Automated Hyperparameter Tuning for Deep Learning-Based Side-Channel Analysis},
  journal = {IEEE Transactions on Emerging Topics in Computing},
  volume  = {12}, number = {2}, pages = {546--557}, year = {2024},
  doi     = {10.1109/TETC.2022.3218372},
}
% verified at https://doi.org/10.1007/978-3-030-97087-1_9 (metadata only; book editors not checked)
@incollection{PWP22lottery,
  author    = {Guilherme Perin and Lichao Wu and Stjepan Picek},
  title     = {Gambling for Success: The Lottery Ticket Hypothesis in Deep Learning-Based Side-Channel Analysis},
  booktitle = {Artificial Intelligence for Cybersecurity},
  series    = {Advances in Information Security},
  pages     = {217--241}, publisher = {Springer}, year = {2022},
  doi       = {10.1007/978-3-030-97087-1_9},
}
% verified at https://doi.org/10.3390/math11153265
@article{KPP23resnet,
  author  = {Sengim Karayalcin and Guilherme Perin and Stjepan Picek},
  title   = {Resolving the Doubts: On the Construction and Use of {ResNets} for Side-Channel Analysis},
  journal = {Mathematics},
  volume  = {11}, number = {15}, pages = {3265}, year = {2023},
  doi     = {10.3390/math11153265},
}
% verified at https://arxiv.org/abs/1312.4400 (arXiv comment: "for iclr2014")
@inproceedings{NiN14,
  author    = {Min Lin and Qiang Chen and Shuicheng Yan},
  title     = {Network In Network},
  booktitle = {International Conference on Learning Representations ({ICLR})},
  year      = {2014},
  note      = {arXiv:1312.4400},
}

% ---------- Q2 accuracy vs SCA metrics ----------
% verified at https://doi.org/10.46586/tches.v2019.i1.209-237
@article{PHJBR19,
  author  = {Stjepan Picek and Annelie Heuser and Alan Jovic and Shivam Bhasin and Francesco Regazzoni},
  title   = {The Curse of Class Imbalance and Conflicting Metrics with Machine Learning for Side-channel Evaluations},
  journal = {IACR Transactions on Cryptographic Hardware and Embedded Systems},
  volume  = {2019}, number = {1}, pages = {209--237}, year = {2019},
  doi     = {10.46586/tches.v2019.i1.209-237},
}
% verified at https://doi.org/10.46586/tches.v2020.i1.348-375
@article{MDP20,
  author  = {Lo{\"\i}c Masure and C{\'e}cile Dumas and Emmanuel Prouff},
  title   = {A Comprehensive Study of Deep Learning for Side-Channel Analysis},
  journal = {IACR Transactions on Cryptographic Hardware and Embedded Systems},
  volume  = {2020}, number = {1}, pages = {348--375}, year = {2020},
  doi     = {10.46586/tches.v2020.i1.348-375},
}
% verified at https://doi.org/10.46586/tches.v2020.i3.73-96
@article{ZZNHY20,
  author  = {Jiajia Zhang and Mengce Zheng and Jiehui Nan and Honggang Hu and Nenghai Yu},
  title   = {A Novel Evaluation Metric for Deep Learning-Based Side Channel Analysis and Its Extended Application to Imbalanced Data},
  journal = {IACR Transactions on Cryptographic Hardware and Embedded Systems},
  volume  = {2020}, number = {3}, pages = {73--96}, year = {2020},
  doi     = {10.46586/tches.v2020.i3.73-96},
}
% verified at https://doi.org/10.46586/tches.v2022.i4.228-254
@article{IUH22,
  author  = {Akira Ito and Rei Ueno and Naofumi Homma},
  title   = {Perceived Information Revisited: New Metrics to Evaluate Success Rate of Side-Channel Attacks},
  journal = {IACR Transactions on Cryptographic Hardware and Embedded Systems},
  volume  = {2022}, number = {4}, pages = {228--254}, year = {2022},
  doi     = {10.46586/tches.v2022.i4.228-254},
}
% verified at https://doi.org/10.1007/978-3-642-20465-4_8 ; LNCS 6632 via K10plus/OpenLibrary
@inproceedings{RSVKF11,
  author    = {Mathieu Renauld and Fran{\c{c}}ois-Xavier Standaert and Nicolas Veyrat-Charvillon and Dina Kamel and Denis Flandre},
  title     = {A Formal Study of Power Variability Issues and Side-Channel Attacks for Nanoscale Devices},
  booktitle = {Advances in Cryptology -- {EUROCRYPT} 2011},
  series    = {Lecture Notes in Computer Science}, volume = {6632},
  pages     = {109--128}, publisher = {Springer}, year = {2011},
  doi       = {10.1007/978-3-642-20465-4_8},
}
% verified at https://doi.org/10.1007/978-3-030-26948-7_25 ; LNCS 11692 via K10plus
@inproceedings{BHMOS19,
  author    = {Olivier Bronchain and Julien M. Hendrickx and Cl{\'e}ment Massart and Alex Olshevsky and Fran{\c{c}}ois-Xavier Standaert},
  title     = {Leakage Certification Revisited: Bounding Model Errors in Side-Channel Security Evaluations},
  booktitle = {Advances in Cryptology -- {CRYPTO} 2019},
  series    = {Lecture Notes in Computer Science}, volume = {11692},
  pages     = {713--737}, publisher = {Springer}, year = {2019},
  doi       = {10.1007/978-3-030-26948-7_25},
}
% verified at https://doi.org/10.46586/tches.v2023.i3.522-569
@article{MCHS23,
  author  = {Lo{\"\i}c Masure and Ga{\"e}tan Cassiers and Julien Hendrickx and Fran{\c{c}}ois-Xavier Standaert},
  title   = {Information Bounds and Convergence Rates for Side-Channel Security Evaluators},
  journal = {IACR Transactions on Cryptographic Hardware and Embedded Systems},
  volume  = {2023}, number = {3}, pages = {522--569}, year = {2023},
  doi     = {10.46586/tches.v2023.i3.522-569},
}
% verified at https://doi.org/10.46586/tches.v2021.i1.25-55
@article{ZBDHV21,
  author  = {Gabriel Zaid and Lilian Bossuet and Fran{\c{c}}ois Dassance and Amaury Habrard and Alexandre Venelli},
  title   = {Ranking Loss: Maximizing the Success Rate in Deep Learning Side-Channel Analysis},
  journal = {IACR Transactions on Cryptographic Hardware and Embedded Systems},
  volume  = {2021}, number = {1}, pages = {25--55}, year = {2021},
  doi     = {10.46586/tches.v2021.i1.25-55},
}

% ---------- Q3 GE / SR / evaluation ----------
% verified at https://doi.org/10.1109/ISIT.1994.394764
@inproceedings{Massey94,
  author    = {James L. Massey},
  title     = {Guessing and Entropy},
  booktitle = {Proceedings of the 1994 IEEE International Symposium on Information Theory ({ISIT})},
  pages     = {204}, year = {1994},
  doi       = {10.1109/ISIT.1994.394764},
}
% verified at https://doi.org/10.1007/978-3-030-99766-3_3 ; LNCS 13211 via K10plus + TU Delft record
@inproceedings{WPP22eval,
  author    = {Lichao Wu and Guilherme Perin and Stjepan Picek},
  title     = {On the Evaluation of Deep Learning-Based Side-Channel Analysis},
  booktitle = {Constructive Side-Channel Analysis and Secure Design ({COSADE} 2022)},
  editor    = {Josep Balasch and Colin O'Flynn},
  series    = {Lecture Notes in Computer Science}, volume = {13211},
  pages     = {49--71}, publisher = {Springer}, year = {2022},
  doi       = {10.1007/978-3-030-99766-3_3},
}
% verified at https://eprint.iacr.org/2020/899 (preprint)
@misc{WWKLPBP20,
  author       = {Lichao Wu and L{\'e}o Weissbart and Marina Kr{\v{c}}ek and Huimin Li and Guilherme Perin and Lejla Batina and Stjepan Picek},
  title        = {On the Attack Evaluation and the Generalization Ability in Profiling Side-channel Analysis},
  howpublished = {Cryptology ePrint Archive, Paper 2020/899},
  year         = {2020},
}
% verified at https://doi.org/10.3390/a16030127
@article{PWP23speed,
  author  = {Guilherme Perin and Lichao Wu and Stjepan Picek},
  title   = {The Need for Speed: A Fast Guessing Entropy Calculation for Deep Learning-Based {SCA}},
  journal = {Algorithms},
  volume  = {16}, number = {3}, pages = {127}, year = {2023},
  doi     = {10.3390/a16030127},
}
% verified at https://doi.org/10.1007/978-3-662-48116-5_6 ; LNCS 9054 via K10plus
@inproceedings{GGPSS15,
  author    = {Cezary Glowacz and Vincent Grosso and Romain Poussier and Joachim Sch{\"u}th and Fran{\c{c}}ois-Xavier Standaert},
  title     = {Simpler and More Efficient Rank Estimation for Side-Channel Security Assessment},
  booktitle = {Fast Software Encryption ({FSE} 2015)},
  series    = {Lecture Notes in Computer Science}, volume = {9054},
  pages     = {117--129}, publisher = {Springer}, year = {2015},
  doi       = {10.1007/978-3-662-48116-5_6},
}
% verified at https://doi.org/10.1007/978-3-662-53140-2_4 ; LNCS 9813 via K10plus
@inproceedings{PSG16,
  author    = {Romain Poussier and Fran{\c{c}}ois-Xavier Standaert and Vincent Grosso},
  title     = {Simple Key Enumeration (and Rank Estimation) Using Histograms: An Integrated Approach},
  booktitle = {Cryptographic Hardware and Embedded Systems -- {CHES} 2016},
  series    = {Lecture Notes in Computer Science}, volume = {9813},
  pages     = {61--81}, publisher = {Springer}, year = {2016},
  doi       = {10.1007/978-3-662-53140-2_4},
}
% verified at https://doi.org/10.1007/978-3-662-48800-3_13 ; LNCS 9453 (Part II) via K10plus
@inproceedings{MOOS15,
  author    = {Daniel P. Martin and Jonathan F. O'Connell and Elisabeth Oswald and Martijn Stam},
  title     = {Counting Keys in Parallel After a Side Channel Attack},
  booktitle = {Advances in Cryptology -- {ASIACRYPT} 2015, Part {II}},
  series    = {Lecture Notes in Computer Science}, volume = {9453},
  pages     = {313--337}, publisher = {Springer}, year = {2015},
  doi       = {10.1007/978-3-662-48800-3_13},
}
% verified at https://doi.org/10.1007/978-3-642-38348-9_8 ; LNCS 7881 via K10plus
@inproceedings{VGS13,
  author    = {Nicolas Veyrat-Charvillon and Beno{\^\i}t G{\'e}rard and Fran{\c{c}}ois-Xavier Standaert},
  title     = {Security Evaluations beyond Computing Power},
  booktitle = {Advances in Cryptology -- {EUROCRYPT} 2013},
  series    = {Lecture Notes in Computer Science}, volume = {7881},
  pages     = {126--141}, publisher = {Springer}, year = {2013},
  doi       = {10.1007/978-3-642-38348-9_8},
}

% ---------- Q4 multi-task ----------
% verified at https://doi.org/10.1007/978-3-031-41181-6_7 ; LNCS 13907 via K10plus (metadata only)
@inproceedings{MO23,
  author    = {Thomas Marquet and Elisabeth Oswald},
  title     = {A Comparison of Multi-task Learning and Single-Task Learning Approaches},
  booktitle = {Applied Cryptography and Network Security Workshops ({ACNS} 2023 Satellite Workshops)},
  series    = {Lecture Notes in Computer Science}, volume = {13907},
  pages     = {121--138}, publisher = {Springer}, year = {2023},
  doi       = {10.1007/978-3-031-41181-6_7},
}
% verified at https://doi.org/10.46586/tches.v2023.i1.32-59 (Crossref title is the short form; full title from TCHES page)
@article{MCLS23,
  author  = {Lo{\"\i}c Masure and Valence Cristiani and Maxime Lecomte and Fran{\c{c}}ois-Xavier Standaert},
  title   = {Don't Learn What You Already Know: Scheme-Aware Modeling for Profiling Side-Channel Analysis against Masking},
  journal = {IACR Transactions on Cryptographic Hardware and Embedded Systems},
  volume  = {2023}, number = {1}, pages = {32--59}, year = {2023},
  doi     = {10.46586/tches.v2023.i1.32-59},
}
% verified at https://doi.org/10.1023/A:1007379606734
@article{Caruana97,
  author  = {Rich Caruana},
  title   = {Multitask Learning},
  journal = {Machine Learning},
  volume  = {28}, number = {1}, pages = {41--75}, year = {1997},
  doi     = {10.1023/A:1007379606734},
}
% verified at https://arxiv.org/abs/1706.05098
@misc{Ruder17,
  author       = {Sebastian Ruder},
  title        = {An Overview of Multi-Task Learning in Deep Neural Networks},
  howpublished = {arXiv:1706.05098},
  year         = {2017},
}
% verified at https://proceedings.mlr.press/v119/standley20a.html
@inproceedings{Standley20,
  author    = {Trevor Standley and Amir Zamir and Dawn Chen and Leonidas Guibas and Jitendra Malik and Silvio Savarese},
  title     = {Which Tasks Should Be Learned Together in Multi-task Learning?},
  booktitle = {Proceedings of the 37th International Conference on Machine Learning ({ICML})},
  series    = {Proceedings of Machine Learning Research}, volume = {119},
  pages     = {9120--9132}, year = {2020},
}
% verified at https://proceedings.neurips.cc/paper_files/paper/2020/file/3fe78a8acf5fda99de95303940a2420c-Bibtex.bib
@inproceedings{PCGrad20,
  author    = {Tianhe Yu and Saurabh Kumar and Abhishek Gupta and Sergey Levine and Karol Hausman and Chelsea Finn},
  title     = {Gradient Surgery for Multi-Task Learning},
  booktitle = {Advances in Neural Information Processing Systems ({NeurIPS})},
  volume    = {33}, pages = {5824--5836}, year = {2020},
}
% verified at https://proceedings.mlr.press/v80/chen18a.html
@inproceedings{GradNorm18,
  author    = {Zhao Chen and Vijay Badrinarayanan and Chen-Yu Lee and Andrew Rabinovich},
  title     = {{GradNorm}: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks},
  booktitle = {Proceedings of the 35th International Conference on Machine Learning ({ICML})},
  series    = {Proceedings of Machine Learning Research}, volume = {80},
  pages     = {794--803}, year = {2018},
}
% verified at https://doi.org/10.1109/CVPR.2018.00781 (pages) + CVF open access (author order)
@inproceedings{KGC18,
  author    = {Alex Kendall and Yarin Gal and Roberto Cipolla},
  title     = {Multi-task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition ({CVPR})},
  pages     = {7482--7491}, year = {2018},
  doi       = {10.1109/CVPR.2018.00781},
}
% verified at https://doi.org/10.1109/CVPR.2019.01155
@inproceedings{WDPC19,
  author    = {Zirui Wang and Zihang Dai and Barnab{\'a}s P{\'o}czos and Jaime Carbonell},
  title     = {Characterizing and Avoiding Negative Transfer},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition ({CVPR})},
  pages     = {11285--11294}, year = {2019},
  doi       = {10.1109/CVPR.2019.01155},
}
% verified at https://doi.org/10.52202/068431-0988 (metadata only)
@inproceedings{XGGGF22,
  author    = {Derrick Xin and Behrooz Ghorbani and Justin Gilmer and Ankush Garg and Orhan Firat},
  title     = {Do Current Multi-Task Optimization Methods in Deep Learning Even Help?},
  booktitle = {Advances in Neural Information Processing Systems ({NeurIPS})},
  volume    = {35}, pages = {13597--13609}, year = {2022},
}
% verified at https://doi.org/10.52202/068431-0884 (metadata only; check last-author name form on NeurIPS page)
@inproceedings{KDKWK22,
  author    = {Vitaly Kurin and Alessandro De Palma and Ilya Kostrikov and Shimon Whiteson and M. Pawan Kumar},
  title     = {In Defense of the Unitary Scalarization for Deep Multi-Task Learning},
  booktitle = {Advances in Neural Information Processing Systems ({NeurIPS})},
  volume    = {35}, pages = {12169--12183}, year = {2022},
}

% ---------- Q5 decoding / WHT ----------
% verified at https://doi.org/10.46586/tches.v2020.i2.1-25   ** must-cite: ANSSI affine+shuffle AES dissection **
@article{BS20,
  author  = {Olivier Bronchain and Fran{\c{c}}ois-Xavier Standaert},
  title   = {Side-Channel Countermeasures' Dissection and the Limits of Closed Source Security Evaluations},
  journal = {IACR Transactions on Cryptographic Hardware and Embedded Systems},
  volume  = {2020}, number = {2}, pages = {1--25}, year = {2020},
  doi     = {10.46586/tches.v2020.i2.1-25},
}
% verified at https://doi.org/10.46586/tches.v2021.i3.202-234
@article{BS21,
  author  = {Olivier Bronchain and Fran{\c{c}}ois-Xavier Standaert},
  title   = {Breaking Masked Implementations with Many Shares on 32-bit Software Platforms},
  journal = {IACR Transactions on Cryptographic Hardware and Embedded Systems},
  volume  = {2021}, number = {3}, pages = {202--234}, year = {2021},
  doi     = {10.46586/tches.v2021.i3.202-234},
  note    = {Full title on the TCHES page adds "or: When the Security Order Does Not Matter"},
}
% verified at https://doi.org/10.1007/978-3-662-45611-8_15 ; LNCS 8873 (Part I) via K10plus (e-ISBN 978-3-662-45611-8)
@inproceedings{VGS14,
  author    = {Nicolas Veyrat-Charvillon and Beno{\^\i}t G{\'e}rard and Fran{\c{c}}ois-Xavier Standaert},
  title     = {Soft Analytical Side-Channel Attacks},
  booktitle = {Advances in Cryptology -- {ASIACRYPT} 2014, Part {I}},
  series    = {Lecture Notes in Computer Science}, volume = {8873},
  pages     = {282--296}, publisher = {Springer}, year = {2014},
  doi       = {10.1007/978-3-662-45611-8_15},
}
% verified at https://doi.org/10.46586/tches.v2020.i4.209-238
@article{GGSB20,
  author  = {Qian Guo and Vincent Grosso and Fran{\c{c}}ois-Xavier Standaert and Olivier Bronchain},
  title   = {Modeling Soft Analytical Side-Channel Attacks from a Coding Theory Viewpoint},
  journal = {IACR Transactions on Cryptographic Hardware and Embedded Systems},
  volume  = {2020}, number = {4}, pages = {209--238}, year = {2020},
  doi     = {10.46586/tches.v2020.i4.209-238},
}
% verified at https://doi.org/10.21105/joss.05196 ; WHT-based XOR factor verified in source code (GitHub simple-crypto/SCALib)
@article{SCALib,
  author  = {Ga{\"e}tan Cassiers and Olivier Bronchain},
  title   = {{SCALib}: A Side-Channel Analysis Library},
  journal = {Journal of Open Source Software},
  volume  = {8}, number = {86}, pages = {5196}, year = {2023},
  doi     = {10.21105/joss.05196},
}
% verified at https://doi.org/10.46586/tches.v2025.i3.317-360
@article{UIT+25,
  author  = {Rei Ueno and Akira Ito and Yosuke Todo and Akiko Inoue and Kazuhiko Minematsu and Hibiki Ishikawa and Naofumi Homma},
  title   = {All You Need is {XOR}-Convolution: A Generalized Higher-Order Side-Channel Attack with Application to {XEX/XE}-based Encryptions},
  journal = {IACR Transactions on Cryptographic Hardware and Embedded Systems},
  volume  = {2025}, number = {3}, pages = {317--360}, year = {2025},
  doi     = {10.46586/tches.v2025.i3.317-360},
}
% verified at https://doi.org/10.1007/978-3-319-78375-8_13 ; LNCS 10821 (Part II) via K10plus
@inproceedings{GS18,
  author    = {Vincent Grosso and Fran{\c{c}}ois-Xavier Standaert},
  title     = {Masking Proofs Are Tight and How to Exploit it in Security Evaluations},
  booktitle = {Advances in Cryptology -- {EUROCRYPT} 2018, Part {II}},
  series    = {Lecture Notes in Computer Science}, volume = {10821},
  pages     = {385--412}, publisher = {Springer}, year = {2018},
  doi       = {10.1007/978-3-319-78375-8_13},
}
% verified at https://doi.org/10.1109/TIFS.2022.3144871 (metadata only)
@article{BDMS22,
  author  = {Olivier Bronchain and Fran{\c{c}}ois Durvaux and Lo{\"\i}c Masure and Fran{\c{c}}ois-Xavier Standaert},
  title   = {Efficient Profiled Side-Channel Analysis of Masked Implementations, Extended},
  journal = {IEEE Transactions on Information Forensics and Security},
  volume  = {17}, pages = {574--584}, year = {2022},
  doi     = {10.1109/TIFS.2022.3144871},
}

% ---------- Q6 overfitting / early stopping ----------
% verified at https://doi.org/10.1007/978-3-030-68773-1_10 ; LNCS 12244 via K10plus (COSADE 2020, proceedings published 2021)
@inproceedings{RZCBH20,
  author    = {Damien Robissout and Gabriel Zaid and Brice Colombier and Lilian Bossuet and Amaury Habrard},
  title     = {Online Performance Evaluation of Deep Learning Networks for Profiled Side-Channel Analysis},
  booktitle = {Constructive Side-Channel Analysis and Secure Design ({COSADE} 2020)},
  editor    = {Guido Marco Bertoni and Francesco Regazzoni},
  series    = {Lecture Notes in Computer Science}, volume = {12244},
  pages     = {200--218}, publisher = {Springer}, year = {2021},
  doi       = {10.1007/978-3-030-68773-1_10},
}
% verified at https://doi.org/10.1007/978-3-030-89915-8_3 ; LNCS 12910 via K10plus
@inproceedings{PBP21,
  author    = {Guilherme Perin and Ileana Buhan and Stjepan Picek},
  title     = {Learning When to Stop: A Mutual Information Approach to Prevent Overfitting in Profiled Side-Channel Analysis},
  booktitle = {Constructive Side-Channel Analysis and Secure Design ({COSADE} 2021)},
  editor    = {Shivam Bhasin and Fabrizio De Santis},
  series    = {Lecture Notes in Computer Science}, volume = {12910},
  pages     = {53--81}, publisher = {Springer}, year = {2021},
  doi       = {10.1007/978-3-030-89915-8_3},
}
% verified at https://doi.org/10.46586/tches.v2020.i4.337-364
@article{PCP20,
  author  = {Guilherme Perin and {\L}ukasz Chmielewski and Stjepan Picek},
  title   = {Strength in Numbers: Improving Generalization with Ensembles in Machine Learning-based Profiled Side-channel Analysis},
  journal = {IACR Transactions on Cryptographic Hardware and Embedded Systems},
  volume  = {2020}, number = {4}, pages = {337--364}, year = {2020},
  doi     = {10.46586/tches.v2020.i4.337-364},
}

% ---------- Q7 ResNets / architectures ----------
% verified at https://doi.org/10.1007/s13389-019-00209-3
@article{ZS20,
  author  = {Yuanyuan Zhou and Fran{\c{c}}ois-Xavier Standaert},
  title   = {Deep learning mitigates but does not annihilate the need of aligned traces and a generalized {ResNet} model for side-channel attacks},
  journal = {Journal of Cryptographic Engineering},
  volume  = {10}, number = {1}, pages = {85--95}, year = {2020},
  doi     = {10.1007/s13389-019-00209-3},
}
% verified at https://doi.org/10.46586/tches.v2021.i3.235-274
@article{LZCGL21,
  author  = {Xiangjun Lu and Chi Zhang and Pei Cao and Dawu Gu and Haining Lu},
  title   = {Pay Attention to Raw Traces: A Deep Learning Architecture for End-to-End Profiling Attacks},
  journal = {IACR Transactions on Cryptographic Hardware and Embedded Systems},
  volume  = {2021}, number = {3}, pages = {235--274}, year = {2021},
  doi     = {10.46586/tches.v2021.i3.235-274},
}
% verified at https://doi.org/10.46586/tches.v2024.i1.336-374
@article{EstraNet,
  author  = {Suvadeep Hajra and Siddhartha Chowdhury and Debdeep Mukhopadhyay},
  title   = {{EstraNet}: An Efficient Shift-Invariant Transformer Network for Side-Channel Analysis},
  journal = {IACR Transactions on Cryptographic Hardware and Embedded Systems},
  volume  = {2024}, number = {1}, pages = {336--374}, year = {2024},
  doi     = {10.46586/tches.v2024.i1.336-374},
}
% verified at https://doi.org/10.1145/3569577 (article number not verified)
@article{SoKDLSCA,
  author  = {Stjepan Picek and Guilherme Perin and Luca Mariot and Lichao Wu and Lejla Batina},
  title   = {{SoK}: Deep Learning-based Physical Side-channel Analysis},
  journal = {ACM Computing Surveys},
  volume  = {55}, number = {11}, pages = {1--35}, year = {2023},
  doi     = {10.1145/3569577},
}

% ---------- Q8 imbalance ----------
% verified at https://doi.org/10.1109/TIFS.2021.3092050 (metadata only)
@article{ISUH21,
  author  = {Akira Ito and Kotaro Saito and Rei Ueno and Naofumi Homma},
  title   = {Imbalanced Data Problems in Deep Learning-Based Side-Channel Attacks: Analysis and Solution},
  journal = {IEEE Transactions on Information Forensics and Security},
  volume  = {16}, pages = {3790--3802}, year = {2021},
  doi     = {10.1109/TIFS.2021.3092050},
}
```

### Inferences
- The highest-priority additions for a TCHES reviewer are:
  1. **BS20**, the ANSSI-implementation dissection, currently absent.
  2. **MCLS23 / DLWYAK**, the scheme-aware precedent for the decoder loss.
  3. **ZBHV20 + WAGP20**, the small-CNN precedent.
  4. **PHJBR19 + MDP20 + IUH22**, the accuracy/PI precedent.
  5. **WPP22eval**, the GE reporting precedent.
  6. **Standley20 + MarquetOswald24**, the shared-capacity competition precedent.
  7. **VGS14 / GGSB20 / SCALib**, WHT-based XOR convolution.
- Year convention: Crossref's "published" year for TCHES 2020(1) papers is 2019 (online date). The entries above use the issue year (2020), matching IACR's DOI scheme `tches.v2020.i1`.

### Gaps
- Unverified items: the SoK article number; Kurin et al.'s last-author name form; the NiN abstract tail; the editors of "Artificial Intelligence for Cybersecurity"; the final volume and pages of the Vandenhende TPAMI survey (it was not given a BibTeX entry).
- Not searched or verified: Gohr et al. ResNet SCA; Bronchain's PhD thesis; any "Hajra et al." multi-task SCA paper (none found by the Crossref title queries used).
- The OvC (ePrint 2023/1615) published version was not checked here; it is out of scope (ASCADv2-specific).
