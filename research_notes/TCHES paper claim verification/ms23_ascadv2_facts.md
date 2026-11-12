# MS23 / ASCADv2 primary-source facts: checking the TCHES draft's claims

**Sources and access (please read first).**
- **[MS23-ePrint]** Masure & Strullu, IACR ePrint 2021/592, 34 pp. Local copy: `docs/papers/MS23_Masure_Strullu_ePrint2021_592.pdf`, PDF dated 2021-05-05. The ePrint page (https://eprint.iacr.org/2021/592) shows a single entry, "2021-05-10: received", and no revisions, so the local PDF is the only ePrint version. Every section, equation, figure, table and page locator below refers to **this ePrint version**. Page numbers are the printed page numbers.
- **[MS23-JCE]** The JCE 13(2):129–147 version could not be reached. Springer redirects to a login/bot wall, and Semantic Scholar reports `isOpenAccess: false` (https://api.semanticscholar.org/graph/v1/paper/DOI:10.1007/s13389-023-00311-7). Its title differs ("…: end-to-end attacks with multi-task learning"). **Equation, figure and table numbers in the JCE version are therefore UNVERIFIED.** The draft cites `[MS23]` as the JCE article but quotes locators ("Eq. 12", "Fig. 12a", "Table 2") that are verified only in the ePrint.
- **[ANSSI-code]** The official ANSSI-FR/ASCAD repo at upstream commit `58c527f` (Ryad Benadjila, 2021-05-01, "Major updates for ASCAD"). It was read from the local git history (`git show 58c527f:<file>`). The `STM32_AES_v2/` param files in the working tree are unchanged from upstream. Files: `ASCAD_train_models.py`, `ASCAD_generate.py`, `ASCAD_test_models.py` (TensorFlow/Keras originals), and `STM32_AES_v2/{Readme.md, example_*_params}`. Public URLs: https://github.com/ANSSI-FR/ASCAD/tree/master/STM32_AES_v2 and https://github.com/ANSSI-FR/ASCAD/blob/master/ASCAD_train_models.py. The live README was also fetched from https://raw.githubusercontent.com/ANSSI-FR/ASCAD/master/STM32_AES_v2/Readme.md.
- **[HDF5]** The released `ascadv2-extracted.h5` (data.gouv.fr, https://www.data.gouv.fr/en/datasets/ascadv2/). It was inspected read-only on the local copy, `STM32_AES_v2/ASCAD_data/ASCAD_databases/ascadv2-extracted.h5`.

---

# ASCADv2 dataset and MS23 worst-case multi-task attack: verification of TCHES draft claims

## Q1. Dataset facts (800k × 1M traces, STM32F303 Cortex-M4, random key/plaintext, extracted 500k + 10k × 15,000 samples, variable-key attack set)

### Takeaway
The raw-campaign facts are **VERIFIED**. The 500k/10k split is **VERIFIED for the released file** but **CONTRADICTED as a citation to MS23**. The MS23 paper (ePrint) describes a different split: up to 700k training, 20k validation and 50k test traces. Cite the ANSSI repo or data.gouv for 500k/10k. The extracted attack set **does** have variable keys: 10,000 distinct keys in 10,000 traces.

### Cited Findings
- **800k traces, 1M samples: VERIFIED.**
  - ePrint §2.1, pp. 8–9: "The whole execution … covers a 1 M time samples window … and 800,000 traces have been acquired in the whole ASCADv2 database, which weights in total around 800 GB." Data is distributed as "eight h5py files of 100 GB". — [MS23-ePrint](https://eprint.iacr.org/2021/592)
  - The ANSSI README says "800,000 AES encryptions" and "1,000,000 samples points, encompassing the whole AES encryption". — [ANSSI README](https://github.com/ANSSI-FR/ASCAD/blob/master/STM32_AES_v2/Readme.md)
- **Device: VERIFIED.**
  - ePrint §2.1, p. 8: "compiled and embedded on an STM32-F303-RCT7 chip using a Cortex-M4 architecture … ChipWhisperer board".
  - The README says "STM32 Cortex M4 microcrontroller (STM32F303RCT7) … ChipWhisperer board CW308T-STM32F". — [MS23-ePrint](https://eprint.iacr.org/2021/592); [ANSSI README](https://github.com/ANSSI-FR/ASCAD/blob/master/STM32_AES_v2/Readme.md)
  - **Minor conflict, not used by the draft.** The ePrint says the device clock is 4 MHz and the oscilloscope clock is "100 MHz, which allows for 25 time samples per cycle". The README says "50,000,000 samples per second". — same sources
- **Random key and plaintext per trace: VERIFIED, from the README and the data rather than the paper text.**
  - README: "800,000 AES encryptions with random keys and random plaintexts". — [ANSSI README](https://github.com/ANSSI-FR/ASCAD/blob/master/STM32_AES_v2/Readme.md)
  - I found no explicit sentence on key randomness in the ePrint body. It is implied by the fixed-key simulation in §4.1/§4.3.
  - [HDF5] has 499,996 distinct keys in the 500,000 profiling rows. The duplicates are the all-zero placeholder rows.
- **Extracted set, as released: VERIFIED.**
  - The upstream `example_generate_params` has `"profiling_index": range(0, 500000)`, `"attack_index": range(500000, 500000+10000)` and `"target_points": range(205000, 210000) + range(455000, 465000)`. That is two half-open windows of 5,000 + 10,000 = 15,000 samples. — [ANSSI-code example_generate_params](https://github.com/ANSSI-FR/ASCAD/blob/master/STM32_AES_v2/example_generate_params)
  - [HDF5] shapes are `Profiling_traces/traces (500000, 15000) int8` and `Attack_traces/traces (10000, 15000) int8`.
  - The README calls the extracted file "7 GB" and says it holds "15,000 points of interest".
- **Two windows: VERIFIED in the paper.**
  - ePrint §2.2, pp. 9–11: "we make the choice to split the extracted dataset into two contiguous regions". The first is "the range ⟦205,000; 210,000⟧" (rm / GTab precomputation, Fig. 2). The second is "the range ⟦455,000; 465,000⟧" (round-1 SubBytes, Fig. 4).
  - ePrint §3.1.3, p. 13: "15,000 input points formed by the concatenation of the two windows". — [MS23-ePrint](https://eprint.iacr.org/2021/592)
- **CONTRADICTION: the paper's split is not 500k/10k.**
  - ePrint §2.2, p. 11: "Our final extracted dataset is made of up to 700,000 training traces, 20,000 validation traces and 50,000 test traces."
  - App. A, p. 27: "to estimate GE(Na), we sample 500 attack sets by randomly choosing Na attack traces among a test set of 50,000 traces".
  - §4.2, p. 19, also mentions "a separated test dataset of 50,000 traces". — [MS23-ePrint](https://eprint.iacr.org/2021/592)
- **The attack set has variable keys: VERIFIED.**
  - [HDF5] `Attack_traces/metadata['key']` has 10,000 distinct 16-byte keys in 10,000 rows.
  - The official test script simulates a fixed key with `simulated_key=1` (see Q7). — [ANSSI-code ASCAD_test_models.py](https://github.com/ANSSI-FR/ASCAD/blob/master/ASCAD_test_models.py)
- **Placeholder rows: VERIFIED, and slightly extended.**
  - [HDF5] Profiling rows 99,999, 199,999, 299,999, 399,999 **and 499,999** have all-zero traces, an all-zero key and `alpha_mask = 0`. These are the only rows with α = 0, which is invalid because α ∈ GF(2^8)*. The attack set has no α = 0 rows.
  - The draft lists only the first four. That is correct for a 400k-trace profiling subset. Mention the fifth (499,999) if the full 500k set is ever used.

### Inferences
- The release split (500k/10k, indices 0–509,999 from the eight raw files) was very likely produced after the experiments in the ePrint, which used a 700k/20k/50k split. Therefore MS23's published GE curves (Fig. 12a) were computed on a **50k-trace test set that is not the released 10k attack set**. This matters for the draft's "same number of traces as the original paper" framing: the comparison is protocol-matched (500 attacks, GE), not data-identical.
- Suggested wording for the background: "The official extracted set (released with the ANSSI code) concatenates the raw windows [205k, 210k) and [455k, 465k) into 15,000 samples, with 500k profiling and 10k attack traces." Cite the ANSSI repo or data.gouv for 500k/10k, not MS23.

### Gaps
- The JCE version may have updated the split description to match the release. This is unverified.

## Q2. Countermeasures and masked S-box label (α ⊗ Sbox[p⊕k] ⊕ β; HDF5 field names)

### Takeaway
**VERIFIED.** The draft's Eq. (masked) matches MS23 Eq. 6 / §4.3 and the official labelling code exactly. In MS23's notation α is `rm` and β is `rout`. The HDF5 label fields are `alpha_mask`, `beta_mask`, `sbox_masked`, `sbox_masked_with_perm` and `perm_index`.

### Cited Findings
- **Affine masking.** ePrint §1.3.1, p. 6: "three independent shares: a non-zero random multiplicative share rm, a random additive share β and the masked state maskedState[i] = (rm × state[i]) ⊕ β, where × denotes the field multiplication over GF(2^8)". β may be stateM[i], rin or rout. — [MS23-ePrint](https://eprint.iacr.org/2021/592)
- **Shuffling.** ePrint §1.3.2, pp. 6–7: "the order in which each byte is processed is defined by a pre-computed permutation". Permutations from S16 (permIndices) are "generated from four seeds seed1…seed4" via Eq. 3, `permIndices[i] = pGF(pGF(pGF(pGF(i⊕seed1)⊕seed2)⊕seed3)⊕seed4)`. Permutations from S4 are used in MixColumns. — [MS23-ePrint](https://eprint.iacr.org/2021/592)
- **Label used by MultiResNetSCA-1.** §4.3, p. 19: "c[i] = rm × Sbox[pt[p[i]] ⊕ k[p[i]]] ⊕ rout for i in ⟦0,15⟧ and p[i]". This is identical to Eq. 6 on p. 16, `c[i] = maskedState[p[i]] = rm × Sbox[pt[p[i]] ⊕ k[p[i]]] ⊕ rout`. — [MS23-ePrint](https://eprint.iacr.org/2021/592)
- **Official labelling code.** `ASCAD_generate.py::multilabelize`, upstream 58c527f:
  - `alpha = masks[18]` (comment: "alpha which correpond to r_m")
  - `beta = masks[17]` (comment: "beta wich correspnd to rout")
  - `sbox_masked[i] = multGF256(alpha, Sbox[p[ind]^k[ind]]) ^ beta`, with `ind = permIndices(i, masks[0..3])`. This is **slot-indexed**, as in the draft's C_j.
  - `sbox_masked_with_perm[i] = multGF256(alpha, Sbox[p[i]^k[i]]) ^ beta`. This is **byte-indexed**; despite its name it is the *unshuffled* label, used by MultiResNetSCA-2.
  - `perm_index[i] = permIndices(i, …)`, 16 classes.
  - The dtype is `alpha_mask (u1,1), beta_mask (u1,1), sbox_masked (u1,16), sbox_masked_with_perm (u1,16), perm_index (u1,16)`. — [ANSSI-code ASCAD_generate.py](https://github.com/ANSSI-FR/ASCAD/blob/master/ASCAD_generate.py)
- **[HDF5] confirms the structure.**
  - `labels` has exactly those five fields.
  - `metadata` has `plaintext (16)`, `key (16)`, `masks (38 bytes)` and `desync`. Of the 38 mask bytes, 0–3 are the permutation seeds, 17 is rout/β and 18 is rm/α.

### Inferences
- The draft's notation (α, β, π, C_j = α ⊗ Sbox[p_{π(j)} ⊕ k_{π(j)}] ⊕ β) is correct.
- Optional precision: the S16 permutation is a seed-derived pseudo-random permutation (MS23 Eq. 3, four seed bytes). It is not drawn uniformly from all 16! permutations. "Random permutation" is acceptable shorthand.

### Gaps
- None for the ePrint and the code.

## Q3. Threat-model naming ("worst-case" / "first threat")

### Takeaway
**PARTLY CONTRADICTED.** MS23 (ePrint) calls the full-key setting the **"first scenario"** (§4.3; Fig. 12a caption "First scenario with MultiResNetSCA-1"). It does **not** call it "worst-case" or "first threat model". The phrase "First Threat Scenario" in MS23 refers to a **different** setting in §4.1. The draft's "worst-case (or 'first threat') setting" should be reworded.

### Cited Findings
- **The full-key setting.** §4.3, p. 19: "In the first one, the attacker has access to all the affine masks and the shuffling permutation indices during the profiling step, but no access to these values during the attack step. In the second one, the attacker has only access to the affine mask." Paragraph headings: "First Scenario." (p. 19) and "Second Threat Scenario." (p. 20). Fig. 12 captions: "(a) First scenario with MultiResNetSCA-1. (b) Second scenario with MultiResNetSCA-2." — [MS23-ePrint](https://eprint.iacr.org/2021/592)
- **"First Threat Scenario" means something else.** In §4.1 (pp. 15–16), the partial attacks are "1. The attacker knows the affine shares rm and rout during the profiling **and attack** steps"; the permutation indices are known "even during the attack step". The paragraph "First Threat Scenario" (p. 16) is this first-order setting (CNN200poi, ~20 traces). It is not the full-key attack. — [MS23-ePrint](https://eprint.iacr.org/2021/592)
- **"Worst-case" in MS23** describes Bronchain et al.'s open-sample profiling assumption in general, not its own attack. Intro, p. 3: "the attacker may access the values of random shares during the profiling. Such hypothesis enables to easily assess the worst-case scenario [SMY09, ABB+20]". App. A, p. 26, calls profiling attacks "the worst-case scenario for the attacked device". MS23 calls its own full-key setting a "trade-off" (Contributions, p. 4): "we propose a trade-off in the evaluation setting … Our final attack still requires the knowledge of the affine masks during the profiling step." — [MS23-ePrint](https://eprint.iacr.org/2021/592)

### Inferences
- Suggested wording: "the setting of MS23's first full-key scenario (§4.3), in which masks and permutation indices are known during profiling only, often called the worst-case setting". "Worst-case" is follow-up literature usage and should be attributed to that literature, not to MS23. Avoid "first threat", which collides with MS23 §4.1.

### Gaps
- I did not verify which later papers (GPAM, Scoop, etc.) coined "worst-case" for this setting. That is outside this scope.

## Q4. MultiResNetSCA-1 architecture and parameter count (137,396,064)

### Takeaway
**VERIFIED.** The 137,396,064 count is stated verbatim. The draft's architecture description matches the official Keras code: stem plus nine stacks of one residual block each, kernel 11, stride 2 from the second stack, channels 16 → 256 by doubling, AvgPool(4), Flatten to 3584, and 34 heads Dense(1024, ReLU) → BN → Dense(K, softmax). The draft's breakdown (trunk 7,525,472; heads 129,870,592; first dense layers 124,815,360) is arithmetically consistent with MS23's own numbers. MS23 calls these "trainable parameters", so BN moving statistics are excluded.

### Cited Findings
- **Parameter counts.** §3.2, p. 15: "We named MultiResNetSCA-1 an architecture with 34 outputs corresponding to rm, rout, maskedState[p[i]] and p[i] … The numbers of trainable parameters are equal to 137,396,064 and 78,364,256 respectively. For comparison, a single branch ResNetSCA model with the same input points has 11,460,960 parameters … 34 … different models … 389,672,640". — [MS23-ePrint](https://eprint.iacr.org/2021/592)
- **ResNetSCA description.** §3.1.3, p. 13: "the window size of the convolutional layers is set to 11, dense layers are added … trained with Adam optimizer. The total number of stacked convolutional layers is equal to 19."
- **Fig. 16 (App. E, p. 31).** Input 15000×1. Stem "Conv Layer: 16 filters of size 11". Res Blocks 1–9 with outputs 15000×16, 7500×32, 3750×64, 1875×128, 938×256, "459"×256, 235×256, 118×256, 59×256. Then "Avg Pooling 14×256", "Flatten 3584". Each residual block is conv(t, size 11, stride 2) → BN → ReLU → conv(t, size 11, stride 1) → BN, plus a shortcut "t filters of size 1, stride 2", followed by addition and ReLU. "The number of filters t … min(16l, 256)" (as extracted; the printed sizes show doubling). "459" is probably a typo: 'same' padding gives 469.
- **Fig. 17 (p. 32).** Each branch is "Dense 1024 nodes, ReLU, BN, Dense 256 nodes, Softmax". The figure draws every branch with 256 outputs, but the code uses 16 for p[i]. — [MS23-ePrint](https://eprint.iacr.org/2021/592)
- **Code.** `ASCAD_train_models.py::resnet_v1(input_shape=(15000,1), depth=19)`, upstream 58c527f:
  - `num_res_blocks = (19-1)/18 = 1`; `for stack in range(9)`.
  - Stride 2 `if stack > 0 and res_block == 0`. The shortcut is a 1×1 conv with stride 2 and no BN.
  - `kernel_size=11`, `padding='same'`, `he_normal`.
  - `if num_filters < 256: num_filters *= 2`.
  - `AveragePooling1D(pool_size=4)`, then `Flatten`.
  - Branches are `Dense(1024, relu)` → `BatchNormalization()` → `Dense(256 or 16, softmax)`, with 16 outputs for `permind_i`.
  - `optimizer = Adam()`, loss `categorical_crossentropy` on every output. — [ANSSI-code ASCAD_train_models.py](https://github.com/ANSSI-FR/ASCAD/blob/master/ASCAD_train_models.py)
- **Arithmetic check (my computation from the code; trainable parameters only).**
  - K = 256 head: 3584·1024 + 1024 + 2·1024 (BN γ, β) + 1024·256 + 256 = 3,935,488.
  - K = 16 head: 3,689,488.
  - 18 × 3,935,488 + 16 × 3,689,488 = **129,870,592** (matches the draft).
  - 137,396,064 − 129,870,592 = **7,525,472** trunk (matches the draft).
  - Trunk + one 256-head = 11,460,960, and trunk + 18 heads = 78,364,256. Both match MS23 §3.2 exactly, which independently confirms the draft's breakdown.
  - First dense layers: 34 × 3,671,040 = 124,815,360 (matches).

### Inferences
- The draft's description is accurate. One nuance: MS23 counts *trainable* parameters. Counting the BN running mean and variance would add 34 × 2,048 = 69,632 non-trainable parameters to the heads, plus more in the trunk. The draft could say "trainable parameters" once.

### Gaps
- None.

## Q5. Training recipe (profiling traces, epochs, batch, optimizer/LR, preprocessing, validation split)

### Takeaway
The figures of **400,000 traces, batch 64 and "15 epochs" are VERIFIED as paper statements**.
- The released code uses `train_len = 420000` with `validation_split = 0.05`, Keras EarlyStopping (patience 10, restore best weights) and at most 60 epochs.
- The optimizer is **Adam with the Keras default LR (10⁻³)**, not RAdam 10⁻⁴.
- There is **no preprocessing**: raw int8 traces.

The draft does not attribute RAdam to MS23, but the line "Profiling uses the first 400,000 … as in [MS23]" needs the nuance below.

### Cited Findings
- **Paper.** §4.3, p. 19: "The training set is here composed of 400,000 traces, gathered by batches of 64. We use the same training strategy as described in the previous experiments. We monitor the average cross-entropy of the targeted labels. The training stops after 15 epochs with the maximal validation cross-entropy." (sic; presumably "minimal").
  - The "same training strategy" is early stopping. §4.1, p. 16, says "a early-stopping strategy [GBC16], using a patience of 10 epochs".
  - App. H, p. 34: "the profiling traces are randomly partitioned into a training dataset and a validation dataset … we select the trained backup-model with the highest validation metric". — [MS23-ePrint](https://eprint.iacr.org/2021/592)
- **Optimizer.** §3.1.3, p. 13: ResNetSCA is "trained with Adam optimizer". The code uses `optimizer = Adam()`. I found no learning rate in the paper; the Keras default for `Adam()` is 1e-3. For the VGG models, §3.1.2 (p. 13) says "we finally choose Adam optimizer with the default learning rate (10⁻³)". — [MS23-ePrint](https://eprint.iacr.org/2021/592); [ANSSI-code](https://github.com/ANSSI-FR/ASCAD/blob/master/ASCAD_train_models.py)
- **Released params.** `STM32_AES_v2/example_train_models_params` sets `"multilabel": 1, "validation_split": 0.05, "network_type": "multi_resnet", "epochs": 60, "batch_size": 64, "early_stopping": 1, "train_len": 420000`.
  - `train_model` adds `EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)` and calls `model.fit(..., validation_split=validation_split, ...)` on `X_profiling[:train_len]`. — [ANSSI-code](https://github.com/ANSSI-FR/ASCAD/blob/master/STM32_AES_v2/example_train_models_params)
- **Preprocessing: none in the code.** `load_ascad` does `np.array(in_file['Profiling_traces/traces'], dtype=np.int8)`, and the data is passed to `fit` after a reshape only. There is no mean subtraction, standardisation or scaling. The paper does not mention preprocessing for the ResNet (NOT FOUND, i.e. absent). — [ANSSI-code ASCAD_train_models.py](https://github.com/ANSSI-FR/ASCAD/blob/master/ASCAD_train_models.py)

### Inferences
- **Keras `validation_split` takes the *last* fraction of the arrays, before any shuffling** (documented Keras behaviour). So the released recipe trains on profiling indices 0–398,999 (399,000 traces) and validates on 399,000–419,999 (21,000 traces).
  - That is effectively "the first ~400k for training", which reconciles the paper's "400,000" with the code's 420k.
  - Suggested draft wording: "MS23 train on 400k profiling traces (the released script: first 420k, last 5% for validation)".
  - The paper's App. H says the partition is "random". The code's partition is a contiguous tail.
  - The upstream run also trains on placeholders 99,999, 199,999 and 299,999, and validates on 399,999. The draft removes them.
- **"Stops after 15 epochs" with patience 10 and `restore_best_weights` implies the best validation loss was at about epoch 5.** This assumes the paper ran the released settings. It is consistent with the draft's own finding that the 137.4M network reaches its lowest validation loss at epoch 4 and then overfits, so the draft's 15-epoch "best" checkpoint is comparable to MS23's restored early-stopped model.
  - It also means the draft's "15 epochs" is MS23's *total* epochs run, not the epoch of the selected model.
  - MS23's published 60 traces come from an early-stopped (best-val) model. Comparing it with the draft's "best" checkpoint is apt; comparing it with "last" is not.
- The draft's recipe (RAdam 1e-4, mean-trace subtraction, mixed precision) differs from MS23 (Adam 1e-3 default, raw int8). The draft's statement that "our reproduction subtracts the mean trace" while "MS23 used none" is supported by the code. The setup paragraph should say explicitly that the optimizer, LR and preprocessing differ from MS23's.

### Gaps
- The exact epoch of MS23's selected checkpoint is not stated. The "~epoch 5" figure above is an inference.

## Q6. Results: "about 60 traces", Fig. 12a, metric, 500 attacks; Table 2 head accuracies

### Takeaway
- **"About 60 traces" in Fig. 12a: VERIFIED.**
- **Table 2 accuracies of 99.2 / 21.1 / 1.6 / 88.9: VERIFIED.**
- **500 attacks: VERIFIED**, but they are sampled from a 50k test set, not from the released 10k attack set.
- **The "GE < 1 on all bytes" criterion is NOT stated by MS23 for the 60.** The figure is the authors' reading of per-byte mean-rank curves, which are plotted on a linear 0–140 axis. The draft's phrasing "is read off such curves" is appropriate; "the metric of [MS23] (GE<1 on all bytes, 500 attacks)" overstates what is explicit.
- The official README reports a **different** number for the released checkpoint: about 120 traces, single ordering.

### Cited Findings
- **The 60-trace claim.** §4.3, p. 20: "For each key byte, an estimation of the GE is displayed in Figure 12a. We observe that the attack requires approximately 60 attack traces to retrieve the correct full key."
- **Fig. 12 (p. 19)** is captioned "GE for full key recovery scenarios". Panel (a) is "First scenario with MultiResNetSCA-1". It shows 16 per-byte curves ("Mean rank", linear y-axis 0–140; log x-axis 10⁰–10²⁺) and a shaded quartile band. I inspected the rendered page: all curves reach ≈0 between roughly 50 and 100 traces. — [MS23-ePrint](https://eprint.iacr.org/2021/592)
- **Table 2 (p. 20)**, "Overall results of attacks on ANSSI's implementation". The row "MultiResNetSCA-1 | rm ✓ β ✓ p[i] ✓ | 60 | 99.2 21.1 1.6 88.9" has columns "Test accuracies (%) rm, β, c[i], p[i]". The MultiResNetSCA-2 row is 220 traces, 99.6 / 20.5 / 1.0. The table does not say whether c[i] and p[i] are averaged over the 16 heads. — [MS23-ePrint](https://eprint.iacr.org/2021/592)
- **Number of attacks.** §1.2, p. 5: "we will rather assess the Guessing Entropy (GE) [SMY09], i.e., the average rank of the correct key hypothesis … In our experiments, the average is estimated over 500 independently repeated attacks." App. A, p. 27: "we sample 500 attack sets by randomly choosing Na attack traces among a test set of 50,000 traces that are not used during the training". — [MS23-ePrint](https://eprint.iacr.org/2021/592)
- **Threshold.** App. A, p. 26: "the minimal number Na(τ) of attack traces required … to get a guessing entropy below the threshold τ – typically, τ = 1". This general definition is not tied explicitly to the 60. — [MS23-ePrint](https://eprint.iacr.org/2021/592)
- **README for the released model.** "all the correct key byte values are ranked first after approximatively 120 test traces for the first trained model, and 250 test traces for the second model".
  - The test script ranks the **first** `num_traces` (250) attack traces in file order, every 10 traces, with no averaging. That is a single ordering. — [ANSSI README](https://github.com/ANSSI-FR/ASCAD/blob/master/STM32_AES_v2/Readme.md); [ANSSI-code ASCAD_test_models.py](https://github.com/ANSSI-FR/ASCAD/blob/master/ASCAD_test_models.py)

### Inferences
- The draft can keep "60, as reported in MS23 (Fig. 12a, Table 2)". It should describe the 60 as "MS23's reading of its per-byte GE curves over 500 attacks (sampled from a 50k test set)" rather than as an explicit GE < 1 all-bytes figure.
- The README's 120 traces (single ordering, released checkpoint, released 10k attack set) is a useful data point for the draft's pitfalls section. The same authors report 60 (GE over 500 attacks) and about 120 (one ordering).
- The draft's head-accuracy comparison (α 99.94 vs 99.2, β 19.42 vs 21.1, C 1.51 vs 1.6, π 88.77 vs 88.9) uses the correct MS23 numbers. MS23's accuracies are on its 50k test set; the draft's are on 5k attack traces.

### Gaps
- I cannot confirm whether the JCE version kept "Fig. 12a" and "Table 2", or re-ran the experiments on the released 500k/10k split.

## Q7. Decoder equation (Eq. 12), slot normalisation, fixed-key simulation, rank ties

### Takeaway
- **Eq. 12 and its form: VERIFIED** in the ePrint.
- **No normalisation over slots**, in either the paper or the code: Σ_j f_{p[j]}(i|x) is not renormalised.
- **Fixed-key simulation with p ⊕ k and correct key 0: VERIFIED** in the paper and the code.
- **Rank convention: MS23 counts ties in favour of the correct key.** Eq. 15 uses a strict ">", and the code takes the first matching index. This is the opposite of the draft's convention, which counts ties against the correct key. The draft does not claim MS23 does the same, but the difference should be stated.

### Cited Findings
- **Eq. 12, §4.3, p. 20.** "f_{s[i]}(ṡ | x) = Σ_j Σ_α̇ Σ_β̇ f_{c[j]}(α̇ × ṡ ⊕ β̇ | x) · f_{rm}(α̇ | x) · f_{rout}(β̇ | x) · f_{p[j]}(i | x), (12) where s[i] = Sbox[pt[i] ⊕ k]."
  - Complexity drops from O(256·|j|·|α̇|·|β̇|) to O(256·(|j|+|α̇|+|β̇|)).
  - The proof is in App. G (Eqs. 19–23, pp. 33–34). The final step is Eq. 23, f_{s[i]}(ṡ|x) = Σ_j f_{c''[j]}(ṡ|x) · f_{p[j]}(i|x) "by using the total probability formula". There is no normalisation.
  - MultiResNetSCA-2 uses Eq. 13, which has no slot sum. — [MS23-ePrint](https://eprint.iacr.org/2021/592)
- **Code.** `ASCAD_test_models.py::multilabel_predict`:
  - `proba_dissect_beta` gives proba[:, j] = Σ_β P_C[β ⊕ j] · P_β[β].
  - `proba_dissect_alpha` gives proba[:, j] = Σ_α P[α ⊗ j] · P_α[α]. It sums over all 256 α values; the paper's Eq. 12 also does not restrict α̇ ≠ 0.
  - `proba_dissect_permind(proba_x, proba_permind, j)` gives proba[:, s] = Σ_slot proba_x[:, slot, s] · proba_permind[:, slot, j]. There is no renormalisation.
  - Log-scores are summed over traces; a zero probability is replaced by `log(min_nonzero**2)`. — [ANSSI-code ASCAD_test_models.py](https://github.com/ANSSI-FR/ASCAD/blob/master/ASCAD_test_models.py)
- **Fixed-key simulation.**
  - Partial attacks, §4.1, p. 16, Eq. 7: "we simulate a fixed key on the test dataset … by replacing the value of the plaintext byte pt[i] with pt'[i] = pt[p[i]] ⊕ k[p[i]] … where k is a fixed key byte equal to 0". This version includes the permutation, because p is known during the attack there.
  - Full key, §4.3, p. 19: "We simulate a fixed key on the dataset by replacing the test plaintexts and we target the sensitive value s[i] = Sbox[pt'[i] ⊕ k[i]]."
  - Code: `simulated_key=1` gives `proba = predictions[p][AES_Sbox[plaintext ^ key ^ i]]` and `real_key = 0`. — [MS23-ePrint](https://eprint.iacr.org/2021/592); [ANSSI-code](https://github.com/ANSSI-FR/ASCAD/blob/master/ASCAD_test_models.py)
- **Rank definition.**
  - App. A, p. 26, Eq. 15: "g(Sa)[k*] ≜ Σ_{k∈K} 1_{d_Sa[k] > d_Sa[k*]}", with success when g = 0. This is 0-based, and ties do not count against k*.
  - Code: `real_key_rank = np.where(sorted_proba == key_bytes_proba[real_key])[0][0]`, the first position among equal scores. This is optimistic under ties.
  - GE is defined as the expected rank (Eq. 16, p. 26). §1.2 adds "we may also plot the first and third quartile". — [MS23-ePrint](https://eprint.iacr.org/2021/592); [ANSSI-code](https://github.com/ANSSI-FR/ASCAD/blob/master/ASCAD_test_models.py)

### Inferences
- The draft's Eq. (decode) and its "[MS23, Eq. 12]" citation are correct **for the ePrint numbering**.
- The draft's "ties counted against the correct key" is stricter than MS23's. For the reported curves this only matters when scores tie, for example in flat or degenerate decodes such as the draft's uniform-C control. A one-line note would avoid a reviewer flagging it: "MS23 (Eq. 15) counts ties in favour; ours is the conservative variant."

### Gaps
- Equation numbering in the JCE version is unverified. Either cite the ePrint for locators ("[MS23, ePrint 2021/592, Eq. 12]") or check the JCE PDF before submission.

---

## Summary table (draft claim → status)

| Draft claim | Status | Source location (ePrint 2021/592 unless noted) |
|---|---|---|
| 800k traces × 1M samples | VERIFIED | §2.1 pp. 8–9; ANSSI README |
| STM32F303 (Cortex-M4) | VERIFIED (STM32F303RCT7) | §2.1 p. 8; README |
| Random key & plaintext per trace | VERIFIED (README, HDF5); not explicit in paper | README |
| Extracted = 2 windows → 15,000 samples | VERIFIED: [205k,210k) + [455k,465k) | §2.2 pp. 9–11; §3.1.3 p. 13; example_generate_params |
| 500k profiling + 10k attack "[MS23]" | VERIFIED for released file; **paper says 700k/20k/50k** | §2.2 p. 11; App. A p. 27; generate params; HDF5 |
| Attack set has variable keys | VERIFIED (10,000 distinct keys) | HDF5; test script `simulated_key` |
| α ∈ GF(2⁸)*, β additive, shuffle of 16 bytes | VERIFIED (α = rm, β = rout; S16 perm from 4 seeds) | §1.3.1–1.3.2 pp. 6–7 |
| C_j = α⊗Sbox[p_π(j)⊕k_π(j)]⊕β | VERIFIED | Eq. 6 p. 16; §4.3 p. 19; `multilabelize` |
| HDF5 fields | `alpha_mask, beta_mask, sbox_masked, sbox_masked_with_perm, perm_index`; masks[18] = α, masks[17] = β, masks[0..3] = seeds | ASCAD_generate.py; HDF5 |
| "worst-case (or 'first threat') setting" of MS23 | **CONTRADICTED wording**: MS23 says "first scenario" (§4.3, Fig. 12a); "First Threat Scenario" is the §4.1 partial setting; "worst-case" is not used for this attack | §4.1 pp. 15–16; §4.3 p. 19; Intro p. 3 |
| 137,396,064 params (137.4M) | VERIFIED (trainable) | §3.2 p. 15 |
| Architecture (stem, 9 stacks, k = 11, stride 2 from stack 2, 16→256, AvgPool 4 → 3584, 34 × [Lin(3584,1024)→ReLU→BN→Lin(1024,K)]) | VERIFIED | §3.1.3 p. 13; Figs. 16–17 pp. 31–32; `resnet_v1` |
| Trunk 7,525,472 / heads 129,870,592 / first dense 124,815,360 | CONSISTENT (my arithmetic reproduces MS23's 11,460,960 and 78,364,256) | §3.2 p. 15 + code |
| First 400,000 profiling traces "as in MS23" | VERIFIED as "400,000" (paper); code uses first 420k with last 5% as validation ≈ first 399k trained | §4.3 p. 19; train params |
| 15 epochs, batch 64 | VERIFIED as paper statements (training "stops after 15 epochs", early stopping patience 10; code max 60 epochs) | §4.3 p. 19; §4.1 p. 16; train code |
| Optimizer | MS23: **Adam, default LR (1e-3)**; not RAdam 1e-4 | §3.1.3 p. 13; `Adam()` |
| MS23 used no preprocessing | VERIFIED in code (raw int8); not mentioned in paper | `load_ascad` |
| Full key in ≈60 traces, Fig. 12a | VERIFIED | §4.3 p. 20; Fig. 12a p. 19; Table 2 p. 20 |
| Metric GE<1 all bytes, 500 attacks | 500 attacks VERIFIED (from a **50k test set**); "GE<1 all bytes" **NOT stated** for the 60 (read off per-byte curves); τ = 1 only as a general definition | §1.2 p. 5; App. A pp. 26–27 |
| Table 2: 99.2 / 21.1 / 1.6 / 88.9 | VERIFIED | Table 2 p. 20 |
| Eq. 12 decoder, no slot normalisation | VERIFIED (ePrint numbering) | Eq. 12 p. 20; App. G Eq. 23 p. 33; `proba_dissect_permind` |
| Fixed key via p⊕k, correct key 0 | VERIFIED | Eq. 7 p. 16; §4.3 p. 19; test script |
| Rank ties | MS23 counts ties **in favour** (strict >, first index); the draft counts ties against | Eq. 15 p. 26; `rank()` |
| (extra) Released-checkpoint README result | ≈120 traces (single ordering) for model 1, ≈250 for model 2 | ANSSI README |

Search also surfaced: Marquet & Oswald (ePrint 2023/611) discuss MS23's branch architecture. It was not used for verification. — [ePrint 2023/611](https://eprint.iacr.org/2023/611.pdf)
