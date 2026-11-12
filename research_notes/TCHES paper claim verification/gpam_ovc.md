# GPAM (TCHES 2024) and Order-vs-Chaos (ePrint 2023/1615): verification of the draft's claims

Sources actually read (primary):
- GPAM, TCHES version, full text: local copy `docs/papers/gpam_TCHES2024_3_16.pdf` (28 pp.; PDF metadata Subject = "IACR Transactions on Cryptographic Hardware and Embedded Systems, DOI:10.46586/tches.v2024.i3.472-499"). Page numbers below are journal pages (PDF page p = journal page 471+p). Online: [TCHES article page](https://tches.iacr.org/index.php/TCHES/article/view/11685) (the link given by scaaml's site), [arXiv 2306.07249](https://arxiv.org/abs/2306.07249).
- scaaml GPAM code and configs: [papers/2024/GPAM/](https://github.com/google/scaaml/tree/main/papers/2024/GPAM), [gpam_ecc_cm1.py](https://github.com/google/scaaml/blob/main/papers/2024/GPAM/gpam_ecc_cm1.py), [train_with_config.py](https://github.com/google/scaaml/blob/main/papers/2024/GPAM/train_with_config.py), [configurations/](https://github.com/google/scaaml/tree/main/papers/2024/GPAM/configurations), [scaaml/models/gpam.py](https://github.com/google/scaaml/blob/main/scaaml/models/gpam.py) (also installed locally as scaaml 3.0.5 in the `py10` env), [website page gpam_ches_2024.md](https://github.com/google/scaaml/blob/main/website/src/content/docs/papers/gpam_ches_2024.md).
- OvC full text: ePrint direct download is blocked for scripts (HTTP 403 / Anubis bot page), but the **revised (2024-01-16) PDF was retrieved via the Wayback Machine**: [web.archive.org/web/2023id_/https://eprint.iacr.org/2023/1615.pdf](https://web.archive.org/web/2023id_/https://eprint.iacr.org/2023/1615.pdf) (36 pp.; PDF CreationDate 2024-01-16, which matches the ePrint "Revised: 2024-01-16" date, i.e. the latest version). Page numbers below are the printed page numbers (equal to PDF page numbers). Abstract page: [eprint.iacr.org/2023/1615](https://eprint.iacr.org/2023/1615).

Draft text checked: `paper/tches/sections/intro.tex`, `related.tex`, `setup.tex` (para "GPAM"), `results.tex` (subsection "A long-range transformer at equal budget"), `pitfalls.tex`, `abstract.tex` (one line), `refs.bib` (entries GPAM, OvC).

---

## Q1. GPAM bibliographic metadata (authors, volume/issue/pages/DOI)

### Takeaway
VERIFIED. The `refs.bib` entry `GPAM` is correct in every field (authors, title, journal, 2024(3), pp. 472–499, DOI). Only typo risk is in the paper's own running header ("Burzstein"), which the bib correctly does not copy.

### Cited Findings
- First-page header: "IACR Transactions on Cryptographic Hardware and Embedded Systems ISSN 2569-2925, Vol. 2024, No. 3, pp. 472–499. DOI:10.46586/tches.v2024.i3.472-499" — [GPAM TCHES PDF, p. 472](https://tches.iacr.org/index.php/TCHES/article/view/11685)
- Title "Generalized Power Attacks against Crypto Hardware using Long-Range Deep Learning"; authors "Elie Bursztein¹, Luca Invernizzi², Karel Král², Daniel Moghimi¹, Jean-Michel Picod² and Marina Zhang¹" (¹Google Sunnyvale, ²Google Zurich) — [GPAM TCHES PDF, p. 472](https://tches.iacr.org/index.php/TCHES/article/view/11685)
- scaaml's own BibTeX for the paper: journal TCHES, volume 2024, number 3, pages 472--499, year 2024 (it omits the DOI); arXiv preprint is 2306.07249; presented at CHES 2024 — [scaaml website page](https://github.com/google/scaaml/blob/main/website/src/content/docs/papers/gpam_ches_2024.md)
- The odd-page running header and PDF "Author" metadata misspell the first author as "Elie Burzstein" — [GPAM TCHES PDF, e.g. p. 481](https://tches.iacr.org/index.php/TCHES/article/view/11685)

### Inferences
- Recommended BibTeX (all fields verified from the PDF): `author = {Elie Bursztein and Luca Invernizzi and Karel Kr{\'a}l and Daniel Moghimi and Jean-Michel Picod and Marina Zhang}`, `title = {Generalized Power Attacks against Crypto Hardware using Long-Range Deep Learning}`, `journal = {IACR Transactions on Cryptographic Hardware and Embedded Systems}`, `volume = {2024}`, `number = {3}`, `pages = {472--499}`, `year = {2024}`, `doi = {10.46586/tches.v2024.i3.472-499}`. Optional: `url = {https://tches.iacr.org/index.php/TCHES/article/view/11685}`. This is identical to the current entry.

### Gaps
- Exact online publication date of the issue not read from the article page (PDF creation date is 2024-07-11, an inference only).

---

## Q2. Does GPAM attack ASCADv2 on raw 1M-sample traces, in which threat model, with how many heads?

### Takeaway
VERIFIED on all three points: GPAM uses all 1,000,000 points per ASCADv2 trace with no SNR/POI selection; it replicates MS23's "First Threat Scenario" (randomness known during profiling = worst-case); and it predicts c[i], r_m, r_out and p[i] for i=0..15, which is 16+1+1+16 = **34 heads** (the paper lists the targets but never writes the number 34). Caveat for the draft's wording "without preprocessing": the architecture itself normalises/rescales the input.

### Cited Findings
- "Critically, unlike previous work [MS23], we perform an attack by using all 1, 000, 000 points of each trace without preprocessing, instead of using an SNR (signal-to-noise ratio analysis) analysis to use only 15, 000 points (1.5%) out of the total trace." — [GPAM §7.1 "Attack Scenario", p. 492](https://tches.iacr.org/index.php/TCHES/article/view/11685)
- "We replicate the "First Threat Scenario" described in [MS23]." Target: c[i] = r_m × Sbox[pt[p[i]] ⊕ k[p[i]]] ⊕ r_out — [GPAM §7.1, p. 492](https://tches.iacr.org/index.php/TCHES/article/view/11685)
- "Our model targets the intermediate value c[i], mask bytes r_m, r_out, and the permutation p[i] (for i = 0, . . . , 15)." — [GPAM §7.1 "AES attack evaluation", p. 492](https://tches.iacr.org/index.php/TCHES/article/view/11685)
- Threat model: "White-box threat model: ... the attacker has full knowledge of the countermeasures used and they control all the protection parameters during the training phase. This is the main threat model used in this paper"; during the attack phase the model sees raw traces only — [GPAM §3, p. 476](https://tches.iacr.org/index.php/TCHES/article/view/11685)
- Dataset split used by GPAM for ASCADv2: trace length 1M, train 640,000, test 80,000, holdout 80,000 (0.9 TB); "we took a portion of consecutive samples for train, a portion for test, and a portion for holdout" — [GPAM Table 3 and §6.2, p. 484–485](https://tches.iacr.org/index.php/TCHES/article/view/11685). (The paper says ASCADv2 "comprises 800,000 power traces" in §7.1, p. 492, and mis-attributes the dataset to [ERR+18] on p. 485.)
- Architecture: three GAU (Gated Attention Unit [HDLL22]) transformer encoder blocks, a temporal patchification stem, a convolutional combiner, and a multi-head DAG of relational outputs — [GPAM §5.2.1–5.2.3, pp. 479–480](https://tches.iacr.org/index.php/TCHES/article/view/11685). This supports the draft's "patch transformer with gated attention".
- Preprocessing/normalisation: the abstract claims GPAM works "without the need for manual tuning or trace preprocessing" ([p. 472](https://tches.iacr.org/index.php/TCHES/article/view/11685)), but Figure 1's first block is "Full trace (trace_len,) → Normalize, reshape → (trace_len/patch_size, patch_size)" ([Fig. 1, p. 479](https://tches.iacr.org/index.php/TCHES/article/view/11685)). In the code this is a `Rescale` layer mapping the input to [-1, 1] with dataset-global min and max: `x = 2 * ((x - self.trace_min) / self.trace_delta) - 1` — [scaaml/models/gpam.py `Rescale`](https://github.com/google/scaaml/blob/main/scaaml/models/gpam.py). The paper also says "adding normalization layers, residual connections, or more dropout layers did not seem to improve" the heads — [§5.2.4, p. 480](https://tches.iacr.org/index.php/TCHES/article/view/11685).

### Inferences
- The draft's "34 heads of the paper" (setup.tex) is VERIFIED by counting the listed targets. The paper never prints the number, so the draft could say "the 34 heads (c[i], p[i], r_m, r_out)" to show the count.
- GPAM's ASCADv2 setting is the same worst-case threat model as MS23 and the draft, so "worst-case" comparisons with GPAM are valid on threat model; they differ in input (1M raw vs 15k extract) and profiling size (640k vs the draft's 400k).
- GPAM's own preprocessing is only a global min–max rescale to [-1, 1], with no mean-trace subtraction. That is consistent with the draft's claim that mean-trace subtraction is an extra step the draft adds to put GPAM on equal footing.

### Gaps
- None for this question.

---

## Q3. What does GPAM report for ASCADv2 full-key recovery ("about 80 traces", GE<1)?

### Takeaway
VERIFIED, with caveats the draft should mention. The paper says "roughly 80 traces to have GE of all key bytes under 1". The number appears **only in the text of §7.1 (p. 492)**, not in a table or figure (Table 11 gives only head accuracies and ranks). GE is estimated from **10,000 samplings** on the 80,000-trace holdout. Two details flatter the result: c[i] is the **best of 7 runs per attack point** (chosen by validation accuracy), and r_m is used as a hard value ("directly") instead of its distribution.

### Cited Findings
- "Following [MS23] we simulate a fixed key (all zero) by replacing pt by its XOR with the corresponding random key. Since r_m has very high accuracy we use it directly instead of the whole probability distribution. We sample 10,000 times to estimate the GE and need roughly 80 traces to have GE of all key bytes under 1, comparable with the 60 traces needed by [MS23] (but use heavy preprocessing)." — [GPAM §7.1, p. 492](https://tches.iacr.org/index.php/TCHES/article/view/11685)
- Success criterion defined just before: "When the GE of all key bytes is less than one we call the attack successful." GE is defined as the mean rank of the correct value — [GPAM §7 "AES attack evaluation", pp. 491–492](https://tches.iacr.org/index.php/TCHES/article/view/11685)
- "For c[i], we report the best of 7 runs ... we train a model seven times, and for each attack point, we pick the model with the highest accuracy on the validation set. We then use that to evaluate the performance on the holdout split." — [GPAM §7.1, p. 492](https://tches.iacr.org/index.php/TCHES/article/view/11685)
- Table 11 (holdout, 80,000 examples), GPAM vs MS23 accuracy: r_m 100.00% (MS23 99.2), r_out 18.25% (21.1), c[i] average 1.18% (1.6; MeanRank 80.65 vs 80), p[i] average 95.07% (88.9) — [GPAM Table 11, p. 492](https://tches.iacr.org/index.php/TCHES/article/view/11685)

### Inferences
- The draft's related.tex wording "reports full-key GE<1 in about 80 traces" is accurate. More precise wording: "GE below 1 on every key byte after roughly 80 traces (10,000 samplings, holdout set), with each c[i] head taken from the best of seven runs".
- The draft's 500 attack orderings and GPAM's 10,000 samplings both estimate GE; the trace counts are comparable in definition (GE<1 on every byte) but not in the number of resamplings.
- Best-of-7 per head plus a hard r_m make GPAM's 80 an optimistic single-configuration figure. This helps explain why the draft's single-run reproduction on raw traces (600; internal result, not verifiable here) is much worse. The draft could say so in results.tex.

### Gaps
- There is no GE curve or figure for ASCADv2 in the paper, so 80 cannot be checked against a plot.

---

## Q4. GPAM Table 1 hyperparameters for ASCADv2, patch-size rule, optimizer, warm-up

### Takeaway
Table 1 values are VERIFIED: batch 64, 1,000 steps/epoch, 150 epochs, target LR 0.00005, merge filters 0/0, trace length 1,000,000. Three corrections are needed:
1. **Patch size.** The paper states a √L rule, but its own ASCADv2 patch size is **400, not √10⁶ = 1000**. "Following the paper's √L rule" is therefore true of the stated rule and false of the paper's ASCADv2 setting.
2. **Optimizer and warm-up.** VERIFIED that the paper names neither the optimizer nor the warm-up length or initial LR.
3. **Released code.** The public scaaml code uses **Adafactor with a constant LR** and says the warm-up/cosine schedule "is not necessary". Its released ECC configs use target LRs **10× larger than Table 1** (0.006 vs 0.0006; 0.003 vs 0.0003), and there is no ASCADv2 config.

### Cited Findings
- Table 1 ("Model hyper-parameters for each targeted implementation used in this paper"), ASCADv2 column: Batch size 64; Steps per epoch 1,000; Epochs 150; Target learning rate 0.00005; Merge filter 1 = 0; Merge filter 2 = 0; Trace length 1,000,000; Patch size 400 — [GPAM Table 1, p. 480](https://tches.iacr.org/index.php/TCHES/article/view/11685)
- Other columns, for comparison: CM0 128/200/25/0.0006/16/8/1,620,000/1,200; CM1 64/200/500/0.0006/16/8/4,194,304/2,048; CM2 64/200/500/0.0006/16/8/8,388,608/4,096; CM3 32/400/500/0.0003/16/8/16,777,216/4,096 — [GPAM Table 1, p. 480](https://tches.iacr.org/index.php/TCHES/article/view/11685)
- Rule: "Trace length and patch size depend on the size of the dataset, and the capture sample rate. The Patch size is the square root of trace length (rounded)." Also: "Learning rate is to be searched for in logarithmic scale (good values are 0.001, 0.0001, 0.00001)" — [GPAM §5.3, p. 481](https://tches.iacr.org/index.php/TCHES/article/view/11685)
- Schedule, described only qualitatively: "We start from a low initial learning rate value, then steadily increase it to reach its target value, and finally decrease it with a cosine decay as the model converges [LH16]." No warm-up length, initial LR or optimizer name is given — [GPAM §5.3.1, p. 481](https://tches.iacr.org/index.php/TCHES/article/view/11685). A full-text search of the PDF for "Adam", "Adafactor", "optimizer" and "warm" finds only this passage and the phrase "difficulty for the optimizer to learn" (p. 481).
- Implementation: TensorFlow/Keras with Keras Tuner. ASCADv2 hyper-tuning took 29 GPU-days on RTX 4090s; ASCADv2 training time is about 20 h (Table 2) — [GPAM §5.3–5.4 and Table 2, pp. 481–482](https://tches.iacr.org/index.php/TCHES/article/view/11685)
- scaaml archived code docstring: "We found that CosineDecayWithWarmupSchedule is not necessary and one can use Adafactor instead to get the same results with much fewer parameters to set." It compiles with `optimizer = keras.optimizers.Adafactor(target_lr)` (constant LR) and defaults to `target_lr: float = 0.006`. This was already true in the first commit (4457ca30dc, 2024-09-06) — [gpam_ecc_cm1.py](https://github.com/google/scaaml/blob/main/papers/2024/GPAM/gpam_ecc_cm1.py)
- `train_with_config.py` also uses `keras.optimizers.Adafactor(config["target_lr"])` — [train_with_config.py](https://github.com/google/scaaml/blob/main/papers/2024/GPAM/train_with_config.py)
- Released configs: ECC_CM0 `target_lr` 0.006 (Table 1: 0.0006), ECC_CM1_k0 0.006 (0.0006), ECC_CM2_k0 0.006 (0.0006), ECC_CM3_k0 0.003 (0.0003). Batch, steps, epochs, merge filters and patch sizes match Table 1. There are **no ASCADv2 configs** in the directory — [configurations/](https://github.com/google/scaaml/tree/main/papers/2024/GPAM/configurations)
- The README's model-loading snippet must stub a custom object `"CosineDecayWithWarmupSchedule"`, i.e. the originally trained models were saved with that schedule; it does not say which optimizer they used — [papers/2024/GPAM/README.md](https://github.com/google/scaaml/tree/main/papers/2024/GPAM)

### Inferences
- Draft setup.tex: "batch 64, 1,000 steps per epoch, 150 epochs, merge filters 0/0, target learning rate 5·10⁻⁵ reached by warm-up and followed by cosine decay" — VERIFIED.
- "The optimizer and the warm-up length are not given in [GPAM]" — VERIFIED. Suggested addition: "the released scaaml code uses Adafactor at a constant learning rate and states the warm-up/cosine schedule is unnecessary".
- "patch size (125 samples, i.e. 120 tokens, following the paper's √L rule)" — the rule is stated by the paper, but the paper's own ASCADv2 entry breaks it (400 vs √10⁶ = 1000), and so does CM2 (4,096 vs √8,388,608 ≈ 2,896). √15,000 ≈ 122, so 125 is the nearest divisor. Suggested wording: "following the √L rule stated in §5.3 of [GPAM] (their own ASCADv2 setting uses 400 rather than 1000)". If "the paper's setting" is the goal instead, 400/10⁶ scaled to 15k does not give a usable patch, so √L is a defensible choice; just say so.
- **Risk for the claim "at its published learning rate it does not learn the masked S-box at all"** (intro.tex, results.tex). The draft combines Table 1's 5e-5 with Adafactor plus its own 10-epoch warm-up/cosine schedule. Table 1's LRs may belong to the original, undisclosed optimizer/schedule. In the released Adafactor code, every ECC config uses 10× the Table 1 LR. By analogy, the Adafactor counterpart of 5e-5 might be about 5e-4. That would make "published LR" partly a mismatch of optimizer and LR, not purely a GPAM property. **This is an inference from the ECC configs; no ASCADv2 config exists to confirm it.** Suggest qualifying it, e.g. "with Table 1's target learning rate (5·10⁻⁵) under Adafactor, the optimizer of the released code", and noting that the released ECC configs use 10× Table 1's rates.

### Gaps
- The original optimizer and the warm-up length/initial LR of the CosineDecayWithWarmupSchedule are not recoverable from the paper or the public repo.
- I did not check the sedpack GPAM tutorial (google.github.io/sedpack/tutorials/sca/gpam/) for an ASCADv2 recipe.

---

## Q5. GPAM parameter count

### Takeaway
NOT FOUND. The TCHES paper gives no parameter count for any GPAM model, ASCADv2 included. The draft's 6.11M for its 15k-input variant is the draft's own measurement and cannot be compared with a published figure.

### Cited Findings
- A full-text search of the TCHES PDF for "million", "parameters", "trainable" and "M param" finds only "up to 16 million samples" (trace lengths, p. 474) and the "8 parameters to be tuned" hyper-parameters (p. 480–481). There is no model-size figure — [GPAM TCHES PDF](https://tches.iacr.org/index.php/TCHES/article/view/11685)
- The ASCADv2 configuration has merge filters 0/0, so the trunk output size (and the head input size) differs from the ECC configs, and patch size and trace length set the token count — [GPAM Table 1, p. 480](https://tches.iacr.org/index.php/TCHES/article/view/11685); [scaaml/models/gpam.py](https://github.com/google/scaaml/blob/main/scaaml/models/gpam.py)

### Inferences
- If the draft wants a reference size, it could instantiate `scaaml.models.get_gpam_model` with the Table 1 ASCADv2 settings (L = 10⁶, patch 400, merge 0/0, 34 heads) and report `model.count_params()`. This is a local computation, not a paper claim, and was not done here.

### Gaps
- The arXiv versions (2306.07249) were not checked for a parameter count; the TCHES version has none.

---

## Q6. OvC publication status and bibliographic metadata

### Takeaway
VERIFIED as an **unpublished preprint**. ePrint 2023/1615 is labelled "Preprint" (received 2023-10-18, revised 2024-01-16), and no journal or conference version was found in DBLP, Crossref or web search. The ACNS-Workshops 2022 paper "Order vs. Chaos: Multi-trunk Classifier for Side-Channel Attack" (Kulkarni & Verneuil) is a **different, earlier** paper. The current `@misc` entry is right; drop its TODO note and add the URL.

### Cited Findings
- ePrint page: "Order vs. Chaos: A Language Model Approach for Side-channel Attacks"; Praveen Kulkarni (NXP Germany; Radboud University Nijmegen), Vincent Verneuil (NXP Germany), Stjepan Picek and Lejla Batina (Radboud University Nijmegen); Publication info: "Preprint"; Category: Attacks and cryptanalysis; History: received 2023-10-18, revised 2024-01-16 — [ePrint 2023/1615](https://eprint.iacr.org/2023/1615)
- Web search finds DBLP and researchr listing it only as "IACR Cryptol. ePrint Arch. 2023: 1615" (DBLP key journals/iacr/KulkarniVPB23); no other venue — [DBLP record](https://dblp.uni-trier.de/rec/journals/iacr/KulkarniVPB23.html), [researchr](https://researchr.org/publication/KulkarniVPB23). The DBLP site itself returned a bot-check page to direct fetches, so this rests on search-result snippets.
- A Crossref bibliographic query for the title and author returns only the 2022 ACNS-W paper (DOI 10.1007/978-3-031-16815-4_13), not the 2023 language-model paper — [Crossref API query](https://api.crossref.org/works?query.bibliographic=Order+vs+Chaos+language+model+side-channel+Kulkarni&rows=5); [Springer: ACNS-W 2022 chapter](https://link.springer.com/chapter/10.1007/978-3-031-16815-4_13)
- The PDF title page carries no venue header (unlike TCHES papers); its affiliations are "NXP Semiconductors Germany GmbH" and "Radboud University, Nijmegen, Netherlands" — [OvC PDF (Wayback), p. 1](https://web.archive.org/web/2023id_/https://eprint.iacr.org/2023/1615.pdf)

### Inferences
- Recommended BibTeX: `@misc{OvC, author = {Praveen Kulkarni and Vincent Verneuil and Stjepan Picek and Lejla Batina}, title = {Order vs. Chaos: A Language Model Approach for Side-channel Attacks}, howpublished = {Cryptology {ePrint} Archive, Paper 2023/1615}, year = {2023}, url = {https://eprint.iacr.org/2023/1615}, note = {Version of 2024-01-16}}`. Remove `note = {TODO: check for a published version}`.

### Gaps
- Google Scholar and Kulkarni's PhD thesis (Radboud) were not checked. If the paper later appeared as a thesis chapter or under a new title, this search would not catch it.

---

## Q7. OvC's TGE0 definition and its ASCADv2 value; does it beat the draft's 60 traces?

### Takeaway
RESOLVED. The definition and the value are verified, and the draft's framing is **misleading by omission**. OvC's **TGE₀ = 47 on ASCAD-V2** is **single-byte** ("Our analysis targets the first key byte"), uses **GE = 0** (not GE<1), averages **100 experiments** drawn from a 10,000-trace attack set, uses the **15k extract**, and is **worst-case** (additive mask, multiplicative mask and permutation indices labelled during profiling). It is **not** a full-key result, so it does **not** beat MS23's 60 or the draft's 60 (GE<1 on all 16 bytes, 500 attacks). OvC's Table 1 sets its single-byte 47 directly beside MS23's full-key "60", copied from MS23 as a TGE₀ value without being re-run.

### Cited Findings
- Verbatim definition: "Metric TGE0: TGE0 is a commonly used metric in side-channel attacks, indicating the necessary number of side-channel measurements to achieve a guessing entropy GE of zero. GE denotes the average ranking of the secret key (k*) across a multitude of experiments. In this study, we conducted 100 separate experiments using unique sets of Na measurements. ... We further average the rank of the secret key (k*) over 100 times where we use a different set of attack measurements per experiment, which gives us GE. Consequently, TGE0 can be conceptualized as any minimum value for i (≤ Na) for which the probability of k* is highest, meaning GE = 0. If, after all Na attack measurements, k* fails to attain the highest rank, we consider the attack as failed." — [OvC §3.4.2 "Attack Phase", p. 18](https://web.archive.org/web/2023id_/https://eprint.iacr.org/2023/1615.pdf)
- Scope of the ASCAD-V2 experiment: "ASCAD-V2: While the initial raw dataset comprises 1,000,000 samples, we adhere to the reduced sample set recommended in [MS23], bringing it down to 15,000 samples for each trace. Our analysis targets the first key byte. As previously, we split the 500,000 traces available for profiling 90%–10% between training and validation; in the attack phase, we use 10,000 traces." — [OvC §4.1.1, p. 19](https://web.archive.org/web/2023id_/https://eprint.iacr.org/2023/1615.pdf)
- Raw traces were tried and abandoned: "Our subsampling efforts for these were not successful, leading us to rely solely on the limited 15,000 samples, as mentioned in [MS23]." — [OvC §4.1.1, p. 19](https://web.archive.org/web/2023id_/https://eprint.iacr.org/2023/1615.pdf)
- Table 1 ("Results for protected real datasets", column header TGE0), row ASCAD-V2: M_Stn "✗", M_MTask "60 [MS23]", M_OvC "47" — [OvC Table 1, p. 21](https://web.archive.org/web/2023id_/https://eprint.iacr.org/2023/1615.pdf)
- Threat model: "in presenting results for the OvC classifier on the ASCAD-V2 dataset, we incorporate knowledge about randomness during the profiling phase, meaning that besides plaintext, key, and the AES S-box output, we also include information on additive masks, multiplicative masks, and permutation indices." — [OvC §4.2, p. 21](https://web.archive.org/web/2023id_/https://eprint.iacr.org/2023/1615.pdf)
- Comparison with MS23: "we focus on the scenario where knowledge of randomness is assumed during the profiling phase but not during the attack. With the knowledge of randomness limited to the profiling phase, they reportedly succeeded in attacking the ASCAD-V2 dataset using just 60 traces [MS23]." OvC benchmarks explicitly against MS23 because of the same dataset, randomness assumptions and target (S-box output) — [OvC §1, p. 2](https://web.archive.org/web/2023id_/https://eprint.iacr.org/2023/1615.pdf)
- OvC also cites Marquet & Oswald's full-key recovery "using only 24 traces" on a different 7,181-POI dataset with partial randomness knowledge, and declares it not directly comparable — [OvC §1, p. 2](https://web.archive.org/web/2023id_/https://eprint.iacr.org/2023/1615.pdf)
- "Note that we have not performed hyperparameter search, and most of the parameters are heuristically selected." — [OvC Appendix A "Network Design Choices", p. 27](https://web.archive.org/web/2023id_/https://eprint.iacr.org/2023/1615.pdf)

### Inferences
- **The answer to "does OvC beat our 60?" is no, the two numbers are not comparable.** OvC's 47 is one key byte (byte 0), GE = 0, 100 attacks. The draft's 60 is all 16 bytes, GE < 1, 500 attacks. A full-key figure from the same model would be at least as large as its byte-0 figure and probably larger, because the worst byte dominates.
- The like-for-like comparison is the draft's own **byte-0 traces to GE = 0** over its attack orderings. pitfalls.tex already reports GE = 0 on all bytes = 165 for CompactMultiResNet seed 1, but not a byte-0-only value. Computing byte-0 GE = 0 (ideally also over 100 orderings) would settle the comparison directly. Not computed here.
- OvC's "60 [MS23]" in a TGE0 column (GE = 0) beside its own single-byte 47 compares unlike metrics. MS23's 60 is a full-key figure; the draft says it is read off GE<1 curves. The draft can note this neutrally.
- Draft wording to fix:
  - related.tex "OvC reports TGE₀ 47 with randomness labels" → e.g. "OvC reports 47 traces to reach GE = 0 on the first key byte only (100 attacks, 15k extract, mask and permutation labels during profiling); it has no full-key result."
  - pitfalls.tex: resolve the TODO with the definition above.
  - intro.tex: the TODO "no new attack-trace record over OvC until its TGE₀ definition is checked" can now say the results are not comparable (single byte vs full key) rather than that OvC holds a record.
  - intro.tex "changed ... the training objective (OvC)" is a fair one-line summary (OvC is a binary "order vs chaos" language-model/metric-learning classifier).
- OvC profiles on 450k/50k of the 500k profiling traces; the draft uses 400k. Worth one clause if profiling budget is compared.

### Gaps
- OvC reports no per-byte curve and no full-key result for ASCAD-V2, so its full-key performance is unknown.

---

## Summary table of draft claims

| Draft location | Claim | Verdict | Source |
|---|---|---|---|
| refs.bib GPAM | Authors, TCHES 2024(3):472–499, DOI 10.46586/tches.v2024.i3.472-499 | VERIFIED | GPAM p. 472 header |
| intro/results | GPAM designed for / run on raw 1M-sample traces | VERIFIED | GPAM §7.1, p. 492 |
| setup | 34 heads | VERIFIED (by count: c[i], p[i] ×16, r_m, r_out) | GPAM §7.1, p. 492 |
| related | "patch transformer with gated attention" | VERIFIED (GAU, patchification stem) | GPAM §5.2, pp. 479–480 |
| related, results | full-key GE<1 in about 80 traces | VERIFIED ("roughly 80 traces to have GE of all key bytes under 1", 10,000 samplings); text only, no table/figure; best of 7 runs per c[i]; r_m used directly | GPAM §7.1, p. 492 |
| setup | batch 64, 1,000 steps/epoch, 150 epochs, merge 0/0, target LR 5e-5, warm-up + cosine | VERIFIED | GPAM Table 1, p. 480; §5.3.1, p. 481 |
| setup | patch size via paper's √L rule | PARTLY: rule stated (§5.3, p. 481), but paper's ASCADv2 patch = 400 ≠ √10⁶ = 1000 | GPAM Table 1, p. 480 |
| setup | optimizer and warm-up length not given in paper | VERIFIED. Released code uses Adafactor at constant LR; its ECC configs use 10× Table 1 LRs; no ASCADv2 config | GPAM §5.3.1; scaaml gpam_ecc_cm1.py, configurations/ |
| intro/results | "at its published LR it does not learn" | Caution: LR is Table 1's, but paired with Adafactor plus our schedule; the original optimizer is unknown | as above (inference) |
| — | GPAM parameter count | NOT FOUND in paper | — |
| GPAM abstract | "without trace preprocessing" | Mostly true; GPAM still min–max rescales the input to [-1, 1] ("Normalize, reshape", Fig. 1) | GPAM Fig. 1, p. 479; scaaml `Rescale` |
| refs.bib OvC | ePrint 2023/1615, 2023 | VERIFIED; still a preprint (revised 2024-01-16); no venue | ePrint page; DBLP/Crossref |
| related | "OvC reports TGE₀ 47 with randomness labels" | VERIFIED but incomplete: first key byte only, GE = 0, 100 attacks, 15k extract, masks + permutation labels | OvC pp. 18, 19, 21 |
| pitfalls/intro TODO | Does OvC beat 60 (full key, GE<1, 500 attacks)? | No; not comparable (single byte) | OvC p. 19 |
| — | Did OvC compare with MS23's ~60? | Yes: Table 1 lists "60 [MS23]" (copied, not re-run) beside its 47; intro p. 2 cites MS23's 60 | OvC p. 2, p. 21 |
