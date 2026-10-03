# T-014: ASCADv1 leakage localizer (2O windows)

## Status

`in_progress`

**Snapshot (2O / order / desync100):** Claimed model is **no-band + order** \(\mu_r<\mu_C\) (not r/C bands). Key byte 2 abs rank 0 on desync0 and desync100 (3/3 seeds). \(C\) windows stay **wide**. Loss walk-through (batch Pearson, regularizers): plan §8. Experiment table: plan §11.

## Goal

Learn two per-trace windows on ASCADv1 Boolean-masked traces and recover key byte 2 by **profiled 2O CPA** (non-worst-case: profiling \(\mathrm{HW}(S^\star)\) only).

## Context

- **Current spec:** [`docs/superpowers/plans/2026-08-20-ascadv1-leakage-localizer.md`](../superpowers/plans/2026-08-20-ascadv1-leakage-localizer.md) (renamed from “POI searcher”)
- Old CE-searcher stub: `docs/superpowers/plans/2026-08-18-ascadv1-poi-searcher.md`
- v2 affine idea: `docs/superpowers/plans/2026-08-17-poi-searcher.md` (T-013)
- Data: `ATMEGA_AES_v1/ATM_AES_v1_fixed_key/ASCAD_data/ASCAD_databases/ASCAD.h5` (\(T=700\), already `[45400,46100)`)
- Algebra: \(S=r\oplus C\) with \(r=\texttt{masks}[:,0]\), \(C=S\oplus r\), \(S=\) official `labels` (Boolean; **not** v2 \(\alpha=r_m\)) — [`docs/notes/notation.md`](../notes/notation.md)
- Measured SNR (8k profiling, 2026-08-18): \(r\) max 1.33 at \(t=156\); \(S\oplus r\) max 6.59 at \(t=517\); unmasked \(S\) max 0.042
- Current eval: 2O CPA key rank of byte 2 (`exp_2o.two_order_cpa_rank`). CNN-style `guessing_entropy` is a *different* claim (CE-searcher appendix).
- Related: T-013 (v2, do not mix code paths), T-001 / `visualize/snr_poi_xor.ipynb` (share SNR), T-005 (GE)

## Plan

- [x] (I:H U:L S:H) Lock D1–D13 in the v1 plan (2 windows, per-trace \(h_\psi\), XOR combine, GE on \(P(S)\), `MASK_IDX=0`)
- [x] (I:H U:L S:H) Package `leakage_localizer/` (was `poi_searcher_v1/`) + train/test hooks
- [x] (I:H U:L S:H) Unit tests (15/15) + 3-epoch synthetic smoke (CE_S at chance)
- [ ] (I:H U:L S:M) S0 synthetic: planted \(\mu\) must recover before any HDF5 train
- [x] (I:H U:L S:L) S1 8k×15e fixed-key desync0 CUDA — windows **did not** find peaks (hit band floors 40 / 400)
- [ ] (I:H U:L S:L) S2 full GE table vs existing v1 methods; \(\lambda_{\mathrm{share}}\) + Variant A/0 ablations
- [x] (I:H U:L S:M) 2O on desync0: key_rank=0 (not 256-way \(S^\star\))
- [x] (I:H U:L S:M) 2O on desync100: shifted-oracle and learned windows both key_rank=0
- [ ] (I:H U:M S:M) Hygiene pass: persist `exp2o_*_rank.json` across 3 seeds on desync0 and desync100; reconcile SNR(S) calculation
- [ ] (I:M U:M S:M) Expanded ASCADv1 survey: search for additional learned POI/window selection and 2O DL-SCA papers
- [ ] (I:M U:M S:M) Benchmark comparison: compare published ASCADv1 results with 2O Localizer to identify needed experiments
- [ ] (I:M U:L S:L) S3 official CE-searcher on desync50/100 only if S2 is non-random

## Decisions

- 2026-08-18: Do **not** port \(\alpha,\beta,C\). v1 uses Boolean shares \((r,C=S\oplus r)\). Attack metric is GE of unmasked S-box (byte 2), same API as other methods.
- 2026-08-18: CE(\(P(S),S^\star\)) **is** the consistency term when \(P(S)\) is the XOR convolution. Optional share CE only to break \(r\leftrightarrow C\) swap.
- 2026-08-18: Use the v2-shaped loss, not a fourth CE. \(\mathcal{L}_{\mathrm{id}}=\mathrm{CE}_r+\mathrm{CE}_C+\lambda(-\log P(r\oplus C=S^\star))\) with default \(\lambda=1\). That last term **is** \(\mathrm{CE}(p_S,S^\star)\). Ablate \(\lambda\in\{0,1,3\}\). Do not add an independent \(S\)-head CE on top.
- 2026-08-18: User wants \((r-\bar r)(C-\bar C)=0\) as a loss. Locked as \(\mathcal{L}_{\mathrm{ind}}=(\widehat{\mathrm{Cov}}(\tilde r,\tilde C))^2\) on **predicted** bytes (v2 §2.4.1 analogue). Default \(\lambda_{\mathrm{ind}}=1\). **Not** the sole loss (constants zero it). **Not** XOR / GE. Keep \(\mathrm{CE}_r+\mathrm{CE}_C+\lambda\mathrm{CE}(p_S,S^\star)\).
- 2026-08-18: `MASK_IDX=0` locked from local SNR scan (idx 0 dominates; idx 15 is a weak second).
- 2026-08-18: Network = two tiny MLPs on 24-d window features, **not** TimesNet / `MLPBest` / `CNNBest`. Those stay GE baselines. Upgrade to a small 1-D CNN on \(w\odot X\) only if S1 localizes but CE(\(S\)) fails.
- 2026-08-18: Not an LLM task. A full-trace Transformer is a *different* backbone (GPAM-style), not the POI-searcher. Soft attention as \(\mathcal{W}\) is optional later; two Gaussians first.
- 2026-08-18: Same as v2: **per-trace** \(h_\psi(X)\), not four global `nn.Parameter`s. Global \(\theta\) is Variant G only. (CE-searcher draft; current model is the 2O localizer.)
- 2026-08-18: Do **not** pin \(\mu\). Searcher starts at band **front** (wide \(\sigma\)), slides right; SNR 156/517 is the success check. Pinning would be Variant G (rejected).
- 2026-08-18: \(\sigma_{\min}=1\), \(\sigma_{\mathrm{init}}=15\). \(\sigma_{\max}\) anneals linearly \(15\to 1\) over epochs so the window is forced to shrink. Physical SNR FWHM is ~10 samples (\(\sigma\sim 4\)).
- 2026-08-18: User: do **not** put SNR (or HW-correlation, which is the same first-order idea) in the **training loss**. \(\lambda_{\mathrm{hw}}=0\) default. SNR remains an *evaluation* check only (did \(\mu\) land on 156/517?). The HW term was a debug probe, not the agreed objective.
- 2026-08-20: **“Legal” = fair profiled-SCA split**, not the CE-searcher design preference. Training on \(\mathrm{HW}(S^\star)\) / SNR on the **profiling** set, then attacking with 256 key **guesses**, is standard and fair (weaker labels than ASCAD’s 256-way \(S^\star\)). Unfair = using the attack key/traces to train or pick POIs. “No HW in the CE-searcher loss” is a **method design** (avoid rediscovering 2O CPA; match CNN GE). Notes: `docs/notes/metrics.md` + `v1-poi-faq.md` §0.
- 2026-08-20: The **claimed current model** is the 2O **leakage localizer** (`TwoOrderWindows` + `loss_2o`), not the CE-searcher. Plan: `2026-08-20-ascadv1-leakage-localizer.md`.
- 2026-08-21: Claimed 2O = **full-trace softmax** + order \(\mu_r<\mu_C\) only (no §6.1 bands). Loss is **batch** \(\mathcal{L}_{2\mathrm{O}}=-\lvert\mathrm{corr}(\mathrm{prod},\mathrm{HW}(S^\star))\rvert\). Regularizers in the loss: order hinge \(\lambda=1,m=200\) and edge mass \(\lambda=0.3\). `time_bias=0.08` is a score tilt, not a loss term. \(\lambda_{\mathrm{rep}}=0\). Spec §8; FAQ §16–§19.
- 2026-08-21: Order **direction** is a Boolean-masking / time-order assumption (not 156/517). **Margin 200, edge 12, time-bias** are still this-crop knobs (weaker than bands). Temperature is an optimizer schedule. Spec §6.2; FAQ §20. Possible ablation: \(m=0\), no tilt.

## Working notes / session log

### 2026-09-24 (post-brief update) — Task D roadmap: hygiene pass, expanded survey, and benchmark comparison

- **Context & Status:** Preliminary survey results are in hand (`docs/2026-09-23/out/D-paper-gap.md`). While our mechanism (learned ordered windows + 2O CPA without mask labels) appears un-scooped, publication readiness requires an empirical **hygiene pass**, an **expanded literature survey**, and a **benchmark comparison**.
- **What "Hygiene Pass" Means:**
  1. *Artifact Persistence (Gap 1):* Update `exp_2o.py` with `json.dump` to persist `exp2o_*_rank.json` across seeds 0, 1, 2 on desync0 and desync100. Currently, `rank_final = 0` is only printed to stdout; the only JSON on disk says `FAIL`.
  2. *Metric Reconciliation (Gap 3):* Reconcile printed "$\text{SNR}(S) \approx 0.02$" with `snr-peak-stability-desync0.json` ($\text{snr\_max} = 0.00613$) and save the validated SNR curve to disk.
  3. *Hyperparameter Ablation:* Ablate crop-shaped inductive biases (order margin $m=200$, edge 12, time bias 0.08) to demonstrate whether the localizer relies on smuggled domain knowledge.
  4. *Citation & Nomenclature Hygiene:* Enforce verified DOIs and official titles (e.g. avoid internal shorthands like "AdvMask", correct MS23 as offline/ASCADv2).
- **Expanded ASCADv1 Literature Survey:**
  - Broaden the search for related ASCADv1 papers featuring learned POIs, attention mechanisms, contrastive learning (e.g. CL-SCA), and higher-order deep-learning attacks without mask labels.
- **Benchmark Comparison & Experiment Gap Identification:**
  - Structure a side-by-side comparison matrix of surveyed papers vs. our 2O Localizer (attack traces to rank 0/GE $\le 1$, profiling trace requirements, desynchronization tolerance, model complexity, supervision level).
  - Use the comparison to pinpoint missing experimental evidence (e.g., side-by-side baseline runs against AutoPOI/B-CNN, multi-byte evaluation, or reduced-profiling budget runs).

### 2026-09-24 — brief D: is T-014 a small paper? Verdict **2 — not yet** (prior art re-verified)

- Deliverable: merged into `docs/2026-09-23/out/D-paper-gap.md`. A parallel session had written that path at 01:02 with the same recommendation but two citation errors, so my pass was folded into it rather than left beside it — one card for the meeting. Nothing below contradicts its disk findings; its blocker list is correct and is reused.
- **MS23 took three readings tonight; the settled one is: the ASCADv2 paper, with two windows that are *frozen offline*.** Masure & Strullu, J. Cryptographic Engineering **13(2):129–147**, 2023 (preprint ePrint **2021/592**), DOI 10.1007/s13389-023-00311-7. It **creates** ASCADv2 and defines the official extracted set as raw **W1 `[205k,210k)` ‖ W2 `[455k,465k)` = 15,000** samples, chosen by **code analysis + SNR** (200–800-point POI nets too); its networks are **multi-task over several intermediate computations with mask labels included** (the "First Threat" regime GPAM inherits), all-byte GE < 1 in ~60 traces. The verified abstract contains no region / localization / POI / second-order wording. **Wrong readings to avoid repeating:** "MS23 is a phantom" (01:02 pass) and "MS23 has a learned leakage-region head" (this session's first merge, from trusting an OpenAlex abstract *reconstruction*). The W1‖W2 ranges are from our own survey §I.2 = **secondary reading**, so spot-check the PDF. **Our sell is fixed-vs-learned, not present-vs-absent.**
- **AutoPOI is our closest localization neighbor and was missing from all our notes**: Remmerswaal, Wu, Tiran, Mentens, J. Cryptographic Engineering 2023, ePrint 2023/008, DOI 10.1007/s13389-023-00328-y. PPO selects/scales **regions of raw features**, reward = template-attack performance; proxy-read datasets ASCAD_F / ASCAD_R / AES_HD / CHES_CTF; **no second-order sentence found**; how many regions and whether their size is fixed = unverified.
- **Measured, not argued** (`docs/2026-09-23/out/D-v1-playback-control.py` + `.json`): trace-blind playback of the training-frequency prior on fixed-key v1 puts byte 2 at rank **216** (8k train) / **240** (50k train) on the full 10k attack set, vs chance 127.5 ⇒ the ePrint 2026/338 prior artifact **cannot** explain our byte-2 rank 0. Label χ² ≡ plaintext-byte χ² (293.5, p=0.049 on 50k) — on v1 the "prior" is sampling noise. **Caution:** playback ranks spread over 0–255 across the 16 bytes and **byte 7 reaches rank 0 at 5k/10k with no leakage at all**, so any single-byte rank claim must print that byte's playback rank, and a 16-byte sweep guarantees ~1 free rank-0. Desync-invariant (identical on `ASCAD_desync100.h5`).
- Verified further: Cao et al., *Improving DL Based Second-Order SCA With Bilinear CNN*, IEEE TIFS 2022, DOI 10.1109/TIFS.2022.3216959 (was "unverified" in the 01:02 draft); Xiao & Tang ACNN, IJIS 2024, DOI 10.1007/s10207-024-00874-4; Perin/Wu/Picek feature-selection scenarios, **TCHES 2022(4)** (this, not Masure–Strullu, is the RPOI/OPOI/NOPOI row); ASCAD crop baseline = ePrint 2018/053 / **Benadjila, Prouff, Strullu, Cagli, Dumas, JCEng 2020 10(2):163–188**, DOI 10.1007/s13389-019-00220-8 — cite this, not "Maghrebi et al.".
- Gammell et al. arXiv:2410.22425 is **synthetic-data only** (authors state real-dataset results are weak and hyperparameter-sensitive); "AdvMask" is our shorthand, not a paper name. ePrint **2026/338** *Is it Really Broken? …* (Rousselot, Heydemann, Masure, Migairou, Strullu) is the AOD metric paper.
- Evidence check on disk: rank 0 still has **no result file** (`exp_2o.py:74-85,471-474,550-565` prints and saves `.pth` only); six `exp2o_*_order_rst3_b2_s{0,1,2}.pth` confirmed (56,482/56,512 B, 2026-08-20 06:19/06:41) among **34** `exp2o_*.pth`; `exp-v1-a2-two-order-baseline.json` = `corr 0.0211 / FAIL` and carries **no** `rank`/`restarts`/`batch`/`seed` key, so gap 2's knob-dependency is prose inference. `snr-peak-stability-desync0.json` has `snr_max 0.00613`, which disagrees with the unsaved "SNR(S)≈0.02".
- Remaining novelty delta, stated against **AutoPOI / ACNN / B-CNN / Perin-Wu-Picek**: two windows + order hinge + explicit product + non-worst-case labels, with the attack outside the net. **Not scooped on that combination by anything we could read — so "not yet" now rests on evidence, not novelty.** Never frame the claim as "learned POIs" (published twice over); frame it as an *order-constrained window pair whose product goes to classical 2O CPA*.

### 2026-08-21 — order / edge / tilt / temp: which assume this crop?

- Direction \(\mu_r<\mu_C\): AES Boolean mask-before-S-box on an unshuffled time-ordered round. Not 156/517.
- \(m=200\), edge 12, `time_bias=0.08`: **this** 700-crop. Temperature is generic.
- Written as spec §6.2 and FAQ §20.

### 2026-08-21 — spec: claimed model is order, not bands; write the loss

- User: drop §6.1 two bands from the spec; the model is the best 2O that only forces \(\mu_r<\mu_C\).
- Expanded plan §8: HW, \(\ell\approx a+b\mathrm{HW}+\varepsilon\), batch centering, why \(4\), \(\mathbb{E}[\mathrm{prod}\mid S]\), Pearson, no per-trace 2O loss, regularizer table.
- FAQ §16–§19. Example launch now `--no-bands --lambda-order 1 --lambda-edge 0.3 --time-bias 0.08 --restarts 3`.

### 2026-08-20 — train order recipe ON desync100 (it has a profiling set)

- User: desync100 **does** have `Profiling_traces` (50k, `desync` 0–99) and `Attack_traces` (10k, same). I had only trained the new recipe on desync0 while debugging seeds. Fair protocol = train on that file’s profiling split.
- Trained no-band+order (`--restarts 3`, seeds 0/1/2) on `ASCAD_desync100.h5`. Key **abs rank 0** all 3 seeds, but location tracking is weaker than the old **banded** desync100 model.
- Seed 1 (best locations): corr(\(\mu_r,156-d\))=+0.55, corr(\(\mu_C,517-d\))=+0.22. Five traces: \(r\) errors 1.4, 13.5, 4.0, 4.8, 4.8; \(C\) errors 75, 91, 45, 44, 18.

### 2026-08-20 — desync100 test-time locations (5 traces)

- **Trained on desync100** (banded `exp2o_desync100_abs_s1.pth`), attack split: \(\mathrm{corr}(\mu_r,156-d)=+0.83\); 89% of \(\mu_r\) within 20 of \(156-d\) or \(181-d\). \(\mu_C\) only loosely follows \(517-d\) (corr +0.14, MAE 44).
- Five attack traces (\(d\approx 6,23,49,75,96\)): \(r\) errors 0.7, 9.5, 5.0, 29.0, 4.3 vs nearest r-peak; \(C\) errors 21, 39, 19, 86, 32.
- Current no-band+order ckpt (trained desync0) **zero-shot on desync100 does not** recover \(C\) (often \(\mu_C\approx 0\)). Not trained on jitter.

### 2026-08-20 — no-band + order \(\mu_r<\mu_C\): 3/3 rank 0

- Recipe (desync0, byte 2, 8k/2k, 15e, `abs`): `--no-bands --lambda-order 1 --order-margin 200 --lambda-edge 0.3 --time-bias 0.08 --restarts 3`. Keep **best val \(\mathcal{L}\)**. Temp 1.0→0.6 (0.40 crashed a good \(C\) window).
- Seeds 0/1/2: **abs rank 0 all three.** argmax \(\approx 157\times516\), \(157\times507\), \(156\times509\). \(\lvert\mathrm{corr}\rvert=0.20/0.17/0.22\).
- Fair scoreboard is **abs** (train mode). pos/neg just follow the sign the net picked.
- Hinge \(\mu_r+\mathrm{margin}\le\mu_C\) beats repulsion (which parked on edges). Restarts fix the leftover seed-1 valley.

### 2026-08-20 — no-band + repulsion, 3 seeds (byte 2)

- Recipe: desync0, 8k/2k, 15e, `corr_mode=abs`, `--no-bands`, \(\lambda_{\mathrm{rep}}=0.25\), seeds 0/1/2.
- **Not consistent.** Best ranks: 12, 88, 25. **None hit rank 0.** corr \(\lvert c\rvert \le 0.035\).
- Seed 0: \(\mu_r\approx 161\) (near 156), \(\mu_C\) on the **right edge** (699). Seed 1: one window on the **left edge**. Seed 2: both windows in the \(r\) region (168/147).
- Repulsion keeps the two \(\mu\) apart but they park on **trace ends** or on **two \(r\) leaks**, not the 156×517 pair.

### 2026-08-20 — no-band prior + other key bytes

- Previous success used a **hand prior** \(r\in[40,280]\), \(C\in[400,640]\) on the **byte-2** crop only.
- Added `--no-bands` (both windows see all 700 samples), `--target-byte`, `--bytes all`. \(S^\star_i=\mathrm{Sbox}(p_i\oplus k_i)\).
- Oracle \(156\times517\) on desync0: **only byte 2** has corr\(\approx-0.20\) and rank 0. Other bytes: first-order SNR(\(S_i\))\(\sim 0.09\) (noise), frozen 156×517 ranks 6–120.
- Sweep all 16 bytes, no-band + repel, 12e: **no byte reached a convincing rank 0**. Byte 11 rank 3 / byte 0 rank 5 have \(\lvert\mathrm{corr}\rvert\sim 0\) and windows on the **trace edges** — not leaks.
- Conclusion: the r/C bands are doing real work on this crop. Other AES bytes are not in ANSSI’s 700-sample window; a localizer cannot invent a 2O pair that is not in \(X\).

### 2026-08-20 — 2O localizer vs GPAM

- User asked to compare the latest 2O/poi_searcher run with GPAM.
- Latest run **is** the leakage localizer (`exp2o_desync100_abs_s1.pth`, rank 0), not the failed CE-searcher.
- Write-up: `docs/notes/2o-localizer-vs-gpam.md`. Shared idea: look across time on a trace. Not the same method: dataset, mask labels, two-window product vs Transformer+multi-task GE.

### 2026-08-20 — high-school spec + failure log

- Spec rewritten with 1-bit then 8-bit proof of \(\mathrm{Cov}(\mathrm{HW}(r),\mathrm{HW}(S\oplus r))=(4-\mathrm{HW}(S))/2\): `2026-08-20-ascadv1-leakage-localizer.md`.
- `2026-08-18-ascadv1-poi-searcher.md` is only tried/failed experiments (not a stub).
- Package folder `leakage_localizer/`; train/test imports updated.

### 2026-08-20 — plan renamed to leakage localizer

- Current spec is `docs/superpowers/plans/2026-08-20-ascadv1-leakage-localizer.md`.

### 2026-08-20 — worst-case vs v2 transfer

- Trained 2O (HW(\(S^\star\)) only) = **non-worst-case** (Scoop sense). Oracle 156/517 POIs via SNR(\(r\),\(C\)) = worst-case *characterization*. CE\(_r\)+CE\(_C\) searcher = worst-case profiling.
- v2 drop-in of 2-point Boolean 2O is unlikely: affine needs \(\alpha,\beta,C\) (not \(r\oplus C\)); shuffle moves logical \(C\). T-013 (3 windows + mask labels) is worst-case and the plausible first port. Non-worst-case v2 remains the open Scoop/AOD problem.

### 2026-08-20 — “legal” = SCA-fair split, not the HW-loss preference

- User: do not call 2O “illegal” because *they* asked to avoid HW in a loss. Field bar = profiling vs attack. 2O-train on profiling HW(\(S^\star\)) + attack CPA on guesses **is fair**. CE-searcher without HW is a different *claim*, still unsolved.

### 2026-08-20 — FAQ notes + 2O desync100

- Wrote recurring T-014 Qs to `docs/notes/v1-poi-faq.md` (catalog + `.grok/rules/05-notes.md`).
- Latest **desync0 2O** was a **success as key recovery**, not as 256-way \(S^\star\): oracle \(156\times517\) \(\mathrm{corr}\approx-0.20\), rank 0 by \(-\mathrm{corr}\) or \(|\mathrm{corr}|\); trained \(h_\psi\) \(\mu_r\approx175\), \(\mu_C\approx518\), attack \(+\mathrm{corr}\) rank 0.
- Official CE searcher is still a **failure** (flat CE, windows miss peaks unless HW/SNR is put in the loss).
- Local `ASCAD_desync100.h5`: profiling **and** attack `desync` \(\in[0,99]\) (not the generate.py “attack-only” default).
- Frozen \(156\times517\) on desync100: corr≈0, rank random.
- Shifted oracle \(t_0-d\): corr≈−0.20, `rank_neg=rank_abs=0`. Formula confirmed.
- Train 8k×15e CUDA, `corr_mode=abs`, no extra aug: **learned `rank_pos=rank_abs=0`**, attack corr=+0.20. \(\mathrm{corr}(\mu_r,156-d)=+0.83\) (mean \(\mu_r=131\approx 181-\bar d\)); \(\mu_C\) weakly tracks (+0.14) but C SNR is large. Ckpt: `…/poi_searcher_runs/exp2o_desync100_abs_s1.pth`.

### 2026-08-18 — L_2O experiment (windows + HW(S*), no r/C heads)

- Oracle samples 156×517: \(\mathrm{corr}(\mathrm{prod},\mathrm{HW}(S^\star))=-0.20\). 9-way HW(\(S^\star\)) acc from the product **0.225** (not ID of \(S^\star\)). 2O CPA: **key_rank=0** if ranked by *most negative* corr (true \(k_2\) is the strongest anti-corr); rank 255 if ranked by \(+\mathrm{corr}\).
- Train attention \(h_\psi\) with only \(-\mathrm{corr}(\mathrm{prod},\mathrm{HW}(S^\star))\): \(\mu_r\approx 175\) (argmax 162; target 156), \(\mu_C\approx 518\) (target 517). Attack \(+\mathrm{corr}\) **key_rank=0** on 2k traces.
- Conclusion: known two POIs \(\Rightarrow\) **key byte via 2O CPA**, not a 256-way \(S^\star\) predictor.

### 2026-08-18 — debug loop: oracle CE flat → attention \(h_\psi\) + HW-corr

- Oracle μ=156/517, σ=5, freeze windows: \(\mathrm{CE}_r\) 5.55→5.51 in 8e. 256-way CE is too flat to steer \(h\).
- Linear probe on oracle features / crop / t=156 also ~chance in 30e. First-order **HW correlation** is the usable localization loss.
- Pool-MLP \(h\) (8k params) + front Gaussian never left \(t\sim 64\).
- **Band-attention \(h\)** (conv scores, softmax in band, 12k params) + \(\lambda_{\mathrm{hw}}(-\mathrm{corr}(\sum wX, \mathrm{HW}))\):
  - after 15e: \(\mu_r=166\pm 4\) (target 156), \(\mu_C=513\pm 6\) (target 517); \(\mathrm{hw}\to -1.32\); \(\mathrm{CE}_c\) 5.55→4.87
  - + temp anneal: \(\mu_r\approx 153\), \(\mu_C\approx 512\), \(\sigma_r\) 46→21; \(\sigma_C\) still ~74 (C mass not fully peaked)

### 2026-08-18 — S1c σ_init=15, anneal → 1 (8k×15e CUDA)

- \(\sigma\) followed the schedule: 14.3 → **1.00**. \(\mu\) did **not**: stayed at front \(64/424\) (targets 156/517). \(\mathrm{CE}_S=5.545\).
- From \(\mu_r=64\), \(\sigma=15\), peak 156 is \(\sim 6\sigma\) away — the window never covers the leak, then shrinks further. Shrink without moving \(\mu\) cannot find the POI.

### 2026-08-18 — S1b front-init (no pin)

- User: do not pin μ; start at band front and slide right. Agreed. `snr_init` off; wide σ=90; `lr_psi=1e-3`; joint train.
- Result: start \(\mu\approx 57/417\); end \(52\pm 0.7\) / \(413\pm 0.4\). Did **not** walk to 156/517. \(\mathrm{CE}_r\) 5.55→5.19; \(\mathrm{CE}_S\) still ~5.54.
- Sliding right needs an explicit drive (schedule / \(\lambda_{\mathrm{slide}}\)); CE alone parked at the front.

### 2026-08-18 — S1 fixed-key desync0 (8k × 15e, CUDA)

- SNR-init: 153 / 517. After train: \(\mu_r=40\pm 0\), \(\mu_c=400\pm 0\) (band **floors**). \(\mathrm{CE}_S\) stuck at 5.545.
- Init is fine (forward at step 0 is μ≈154/520). Failure is **training**: warmup moved \(h_\psi\) with **frozen random heads** + `lr_psi=1e-2`; sigmoid band then traps at `lo` (vanishing \(\partial\mu/\partial\mathrm{raw}\)).
- Proposer collapsed; per-trace std 0. Next: freeze \(h_\psi\) at SNR-init, train heads, then unfreeze (or lower `lr_psi`).
- Ckpt: `.../poi_searcher_runs/poi_searcher_desync0_s1.pth`

### 2026-08-18 — implemented `poi_searcher_v1/`

- Package + param files for fixed/variable desync100. Train/test via `ASCAD_train_models.py` / `ASCAD_test_models.py` with `network_type: poi_searcher`.
- 15 unit tests pass. No full HDF5 GPU train (ask first).
- Feature moments signed k-th root + LayerNorm; \(\mathcal{L}_{\mathrm{ind}}\) scaled by \(1/255^2\).
- S1 launched; peak-finding failed (band floors).

### 2026-08-18 — flowchart in the plan

- Added mermaid §2.0 to `2026-08-18-ascadv1-poi-searcher.md`: trained \(\psi,\phi\) vs computed Gaussians / XOR; labels only in \(\mathcal{L}\); attack is \(p_S\) → GE.

### 2026-08-18 — \((r-\bar r)(C-\bar C)=0\)

- Checked new v2 §2.4.1 (Juncheng \(\alpha,\beta\) cov). v1 analogue is batch Cov of predicted \(r,C\), not window product, not per-trace product, not \(S=r\oplus C\).
- Plan §2.3 rewritten: \(\lambda_{\mathrm{ind}}=1\) default + XOR/GE term still on.

### 2026-08-18 — per-trace proposer (aligned with v2)

- User rejected frozen windows. Default model is `WindowProposer` (`X → 4` scalars → 2 Gaussians). S0 must follow moving \(\mu^\star(X)\).

### 2026-08-18 — network choice

- User asked TimesNet vs MLP. Locked: tiny MLP heads on pooled window features. TimesNet/CNN/full-trace MLP are controls, not the searcher.

### 2026-08-18 — loss shape

- User asked whether to keep \(\mathrm{CE}_\alpha+\mathrm{CE}_\beta+\mathrm{CE}_C+\lambda(-\log P(\rho=0\mid S))\).
- Answer: same shape, two share CEs; \(\lambda\) term = CE of XOR-combined \(P(S)\). Default \(\lambda=1\).

### 2026-08-18 — v1 adaptation plan

- Summary: User asked what changes if the POI-searcher plan is applied to ASCADv1. Drafted a full model/pipeline plan: two windows, XOR combine, GE on \(P(S)\).
- Files touched: `docs/superpowers/plans/2026-08-18-ascadv1-poi-searcher.md`; this task; INDEX; `docs/lab-log/todo.md`
- Commands / runs: SNR scan on 8k profiling traces of `ASCAD.h5` (`utils.snr.snr_vector`)
- Metrics / results: \(r\) SNR 1.33 @ 156; \(S\oplus r\) 6.59 @ 517; \(S\) 0.042
- Open questions: none blocking S0; Variant A vs B comparison is an ablation, not a pre-S0 fork

## Next actions

| Action | I | U | S | Notes |
|--------|---|---|---|-------|
| Sharpen \(C\): entropy / \(\sigma_C\) penalty + report argmax not only CoM | H | L | H | 2O loss does not require a tight \(C\) window |
| Other bytes on **raw** 100k / new crops | H | L | L | This 700-window is byte 2 only |
| Official CE-searcher without HW in \(\mathcal{L}\) | H | L | L | Still failed |
| S0 synthetic planted \(\mu\) | M | L | M | CPU |
| 3-seed **banded** 2O (ablation only) | L | L | H | Not the claimed model |
| Paper delta must be argued per paper against **AutoPOI (JCEng 2023) + ACNN (IJIS 2024) + Cao B-CNN (TIFS 2022) + Perin/Wu/Picek (TCHES 2022)** | H | M | M | Brief D verdict **2 — not yet**; `out/D-paper-gap.md` (merged card). **MS23 is not a *learned*-localization competitor** — its W1‖W2 are frozen offline by code + SNR and it trains with mask labels, so it is our worst-case mirror rather than prior art for per-trace windows. Fixes the citation errors in `docs/notes/ASCADv1-leakage-localization-prior-art.md` §7 (= todo T23). The note's RPOI/OPOI/NOPOI acronyms are unconfirmed against the Perin PDF |
| Print the playback rank for **every** byte we ever claim | H | L | H | Control built and run (`out/D-v1-playback-control.py`): byte 2 = 216/240, so the AOD cheat cannot fake it. But byte 7 hits **rank 0 with no leakage**, so a bare rank-0 is not proof. Extend per claimed byte; it is CPU and takes seconds |
| **T28** Scoop sweep: 2024–2026 search for a network emitting **two ordered windows for a second-order attack with no mask label**; then fold AutoPOI/ACNN/Cao-B-CNN/Perin/MS23 into the v2 survey as a localization section, one delta line each | H | M | M | Turns "not scooped" from absence of evidence into a result. Needs Springer access for JCEN 13(2) (confirm W1/W2 are offline-only) and AutoPOI §5 (region count/size). If positive, the v1 paper option dies |
| **T29** Crop-prior ablation in the same run as the rank JSON: baseline `--no-bands --lambda-order 1 --order-margin 200` (must reproduce 0/0/0) vs `--order-margin 0 --time-bias 0`, seeds 0,1,2 × both desyncs, then `json.dump` ranks + T24's playback rank per claimed byte | H | L | M | Decides whether "learns windows without dataset knowledge" survives review. **`batch=256` is not a knob** — it is `train_2o`'s default (`exp_2o.py:144`) and no caller overrides it, so it was 256 in the FAILing a2 run as well. Ask before running; never before 14:00 |
| Re-run `--seeds 0,1,2` and `json.dump` the rank table | H | M | M | Roadmap gap 1, todo T19. After 14:00; not tonight's GPU |

### Priority snapshot

- **Do now:** sharpen \(C\) on the **order** model (I:H S:H)
- **Before next meeting:** claimed recipe is no-band + \(\mu_r<\mu_C\); key rank 0 is reproducible; \(C\) location is still weak; **paper = not yet on evidence, not on novelty** — cite AutoPOI/ACNN/B-CNN/Perin, and stop calling MS23 a localizer (brief D, 2026-09-24)
- **Deep work (park):** CE \(P(S)\); other bytes on raw traces; v2; banded ablation

## Related

- Task ids: T-013 (v2 affine, keep separate), T-005 (GE), T-001 (SNR notebook)
- `docs/lab-log/todo.md` item T16
- Plan: `docs/superpowers/plans/2026-08-20-ascadv1-leakage-localizer.md`
