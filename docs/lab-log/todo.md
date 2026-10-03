# Lab todo (I / U / S)

Cross-cutting action list for day-to-day prioritization and **meeting prep**.  
Technical depth: `docs/tasks/T00x-*.md`. Progress narrative: `progress.md`. Stable context: `notes.md`.

**Triage rule:** `.grok/rules/03-priority-triage.md` (Importance / Urgency / Simpleness).

| Axis | Meaning here |
|------|----------------|
| **I** Importance | Research / thesis / method value |
| **U** Urgency | Needed for the **next meeting** (present, answer, unblock) |
| **S** Simpleness | Quick & easy (wall-clock + low risk) |

Scores: **H** / **M** / **L**. Re-score **U** when the meeting changes.

### Default sort

1. U=H & S=H → 2. U=H → 3. I=H & S=H → 4. I=H deep work → 5. low I/U backlog  

---

## Priority snapshot

_Update when the open table changes or before a meeting._

| Bucket | Items (ids or short text) |
|--------|---------------------------|
| **Do now** | **T-028 (active focus):** decide go/stop on the non-worst-case v2 break — Phase 1 closed NO-GO (label-free α unobtainable by variance/PCA), Phase 2 blocked, the Step 3.1 collision fallback needs a redesigned validation target. T32: tell any session quoting v2 ranks that `rank()` now defaults to worst-among-ties |
| **T-019 GPAM — CLOSED 2026-10-04** | **Table 11 delivered on raw 1M traces, 80,000 never-seen holdout traces:** `rm` **99.996%** (paper 100.00), `p[i]` **86.51%** mean, all 16 heads ≥11.8× chance (paper 95.07), `rout` **10.48%** / MeanRank 10.52 (paper 18.25 / 4.78), `c[i]` **0.679%** / MeanRank 108.6 vs a 127.5 null (paper 1.18 / 80.65 — near chance in the paper too). Campaign ran **76/150 epochs**, killed externally 2026-10-01 ~21:57 (no traceback, no reboot — it did not crash); best val 96.928 at epoch 64, overfit gap ~3.9 nats by ep75. `epoch_070` within ±2 pp, so it is not checkpoint luck. Report: `GPAM/artifacts/C-table11-report.md`. Only GPAM item left: **D4** below |
| **Before next meeting** | D4 (T-019 §7.1 GE + null controls — now unblocked by `rm` at 99.996%); T19 persist 2O rank JSON; T28 scoop sweep |
| **Deep work (park)** | T19 persist 2O JSON; T16–T17; T22 patch-name probe; T29 crop-prior ablation; T30–T31 (T-020 paused) |
| **T-027 c3 (standalone)** | **Blocked on the collaborator.** Gate 1 full first-order scan DONE 2026-09-29 (Outcome E, no first-order leakage). Ask C9/C11 first: what one 4.5M-sample trace covers, OPc=f(K)?, what `auth` is. Only then C12 (second-order scan) |

**Next meeting focus:** T-019 GPAM Table 11 reproduced on raw 1M traces (`rm` 99.996%, `p` 86.5%, `rout` 10.5%, `c` 0.68% — paper 100/95.07/18.25/1.18); T-028 non-worst-case v2 go/stop
**Last triage:** 2026-10-04 (T-019 **closed** for Table 11 — see the new D4/D5 rows and `GPAM/artifacts/C-table11-report.md`; the campaign was found to have been killed at epoch 76/150, not crashed, best epoch 64). Earlier same day: task-registry cleanup — T-028 active; T-009/010/011/016/018/024/025 done; T-013 cancelled; 10 tasks paused. Previous triage: 2026-09-26 22:20 (Phase 0 download complete, 8/8 shards at 93.14 GiB, monitor retired).

---

## Open

| ID | Action | I | U | S | Task | Meeting | Notes |
|----|--------|---|---|---|------|---------|-------|
| D1 | Download full ASCADv2 raw campaign (8 shards, 800k × 1M) and verify | H | H | M | T-019 | — | **Done 2026-09-26**: 8/8 shards at 93.14 GiB, zero `.part`, each opens as `traces (100000, 1000000)` int8. Hourly `monitor_ascadv2_download.py` cron retired (it sent both SMTP and Telegram); crontab backed up to `~/ascad_crontab_backup_20260926.txt` |
| D2 | Diagnose the at-chance full raw run, then relaunch on our own hyperparameters | H | H | M | T-019 | — | **DONE 2026-10-04 end-to-end.** Cause was the learning rate, not the pipeline: label streaming proven correct (0/40,000 mismatches vs stored labels); on 512 raw traces lr 5e-5 cannot even memorise (19.31 vs chance 19.41). Relaunched 2026-09-29 at **lr 1e-3**, 40k probe PASSED (`rm` 98.34% held out), 34-head campaign generalised and its Table 11 is now delivered (`rm` 99.996% on 80k never-seen raw traces). Warmup+cosine only pays at their 150k-step budget |

| D3 | Make the trainer print the chance CE next to every loss line | M | M | H | T-019 | — | The stall hid for 3 hours because "19.4094, falling" looked like progress while still worse than the 19.4081 chance sum. **Shipped** (`[chance-check]` prints every epoch in `train_gpam_raw_full.py`) |
| D4 | T-019: §7.1 fixed-key GE curve + permutation/tie null controls on the ep64 checkpoint | H | M | M | T-019 | — | The last unstarted Step-4 item. `rm` at **99.996% held out** now satisfies the paper's own precondition for setting α by argmax, so the simulation is finally legitimate. One extra holdout pass (~7 min) storing `c`/`p` probabilities instead of only ranks. **Do not improvise the scoring rule** — the earlier local `compute_ge` was deleted for four documented faults; transcribe §7.1/[MS23] first. Expected to fail on `c` (0.679% vs the paper's 1.18%), and "GE does not reach 1" is a result, not a defect |
| D5 | T-019 hygiene: `--resume` in `train_gpam_raw_full.py`; its Telegram/holdout strings say batch 64 while the campaign used 32 | L | L | H | T-019 | — | The 76-epoch campaign could not be restarted after the external kill, which is why it ended at 51% of the paper's data exposure |
| C1 | C3: full-coverage first-order t-scan to settle masking | H | H | S | T-027 | — | **DONE 2026-09-29, OUTCOME E: no first-order leakage anywhere.** Global max abs t = 6.30 vs 6.44 threshold, and 6.30 is exactly the noise-only expectation √(2 ln 423M). Null confirmed 3 ways (`c3_scan_report.py`): t std 1.000, exceedances match normal, longest run above abs t>4 is 1–2 samples = same as the negative control. 113 min CPU, 100% coverage |
| C11 | C3: **ask the capturer what one 4,500,002-sample trace covers** | H | H | H | T-027 | — | Do this BEFORE any further compute. Adjacent samples are nearly independent and the rail is quiet at 4.9728 V ± 5.25 mV — looks like noise on a quiet supply, not switching activity. If a trace isn't one AES operation, nothing will be found |
| C12 | C3: second-order scan on the top abs t regions | H | M | M | T-027 | — | The natural next instrument: a masked target is *expected* to be null at first order. This is the test that would actually demonstrate masking |
| C13 | C3: full-length `patch=10,181` GPAM as a learned multivariate detector | M | L | L | T-027 | — | Only thing that can see interactions a univariate t-scan cannot. But with no first-order foothold it risks fitting noise. Deprioritised behind C11/C12 |
| C14 | C3: ~~`--pool all` for more power~~ | L | L | L | T-027 | — | **Skip.** Real leakage scales t as √n (38k→50k = ×1.15) while the noise ceiling barely moves; the max already sits exactly on the noise expectation, so this rescues nothing |
| C2 | C3: build the uint8 + trace-aligned cube (900 GB → 225 GB, bit-exact) | H | H | M | T-027 | — | **DONE 2026-09-26 23:02.** 10 shards, 227.5 GB, 225,000,100,000 elements bit-verified inline, 0 mismatches. `--phase verify` re-run + logged 2026-09-28 |
| C3 | C3: persist the seeded 76/20/4 split before training | H | M | H | T-027 | — | **DONE** — 38,000/10,000/2,000, seed 20260925. Lives with the cube at `c3_project_8bit/c3_split.npz`, NOT in `GPAM-c3/` |
| C4 | C3: confirm what the device actually computed | H | M | M | T-027 | — | **RESOLVED 2026-09-28**: construction is `E_K(RAND ⊕ OPc)`, AES-128, per `GPAM-c3/dpa_aes_opc.jpg` Fig. 5. `cipher_text == rand` is just the harness never recording a ciphertext. **Still open: is `OPc` derived from `K`? What is `auth`? What does one 4.5M-sample trace actually cover?** |
| C5 | Check/install `zstandard` or `blosc2` in `py10` | L | L | H | T-027 | — | **CLOSED — no longer needed.** Cube shipped uncompressed by decision: the 4× came from the dtype, and space is not scarce |
| C6 | C3: label correction — `SBOX[rand[b] ^ (K^OPc)[b]]`, never `^ key[b]` | H | H | H | T-027 | — | **DONE.** Old formula wrong on 100% of traces for 15/16 bytes (byte 0 survives, `OPc[0]=0x00`). Consequence: round 1 can only recover `K⊕OPc`; claiming `K` needs a round-2 head. Enforced in `c3_crypto.py`, checked by `c3_preflight.py` |
| C7 | C3: `c3_poi.py` — threshold, merge, snap to the 8192 lattice, pick (L, patch, steps) | H | M | M | T-027 | — | Blocked on C1. Choose L from the measured leak WIDTH, not a capacity ceiling — full-length `patch=10,181 → steps=442` is legal and costs ~0.95 GB |
| C8 | C3: single-head scaaml/TF trainer + vendored worst-tie ranking | H | M | L | T-027 | — | User chose TF-via-scaaml (paper-faithful) over a PyTorch port. Head name `sbox1_byte_b`. Ranking MUST use worst-ties — strict `>` reports rank 0 for a flat head (commit `b1d3d75`) |
| C9 | C3: ask the collaborator the 3 open questions | H | M | H | T-027 | — | (a) is `OPc = f(K)` CMAC-style? (b) what is `auth` (per-trace, 14 uniform + 2 constant columns)? (c) does one trace contain one AES operation? (c) is what the flat-variance result hinges on |
| C10 | C3: housekeeping batch from the plan review | M | L | H | T-027 | — | Mostly DONE 2026-09-28: stale `c3_level_table.npy` renamed `.STALE-do-not-use` + `c3_drift_check.py` now reads the cube's embedded table; CWD-relative paths fixed in 4 scripts (proven by running the loader from `/tmp`); fails-open `validate_sbox` now uses `cryptography` and hard-aborts; README §4/§5/§6/§8 corrected. Remaining: NOTES §14 ordering text, `docs/tasks/INDEX.md` line |

| M1 | Launch briefs A–D in `docs/2026-09-23/` before sleep; A owns the GPU | H | H | H | T-020 | 2026-09-24 14:00 | You, first 40 min. Do not kill the raw download |
| M2 | Morning: read `out/A-verdict.md` then C, B, D; fill `meetings/20260924-meeting.md`; no new runs | H | H | H | T-007 | 2026-09-24 14:00 | **Done**: 20260924-meeting.md fully populated with numbers, metrics, caveats & decisions |
| M3 | In the meeting, do NOT say the LLM family is dead on the strength of A2 | H | H | M | T-020 | 2026-09-24 14:00 | A2 got 0.30% on `alpha_mask`, but GPAM predicts that same label at **92.95%** on the same 1.5% crop (`GPAM/artifacts/C-gpam-eval.json`). Reword to "this architecture at this budget", not "mask labels are a dead end" |
| M5 | Say the two GPAM nulls before anyone quotes a low v2 GE | H | H | H | T-019 | 2026-09-24 14:00 | (1) a uniform head yields a constant score vector and strict `>` rank returns **0**; (2) predictions shuffled across traces — no valid pairing — reached rank 10 @2k vs our real 20 |
| M4 | Cite byte-4 GE only with the frequency-table caveat; byte 15 is not that cheat and still not a break | H | H | H | T-010 | 2026-09-24 14:00 | Source: `docs/notes/ascadv2-label-frequency.md`. Scoop twin byte 4 = 2.75 after 37,504 s vs trace-free prior = 3; byte 15's `ge_last` 0 needs **100,001** traces |
| L1 | Localizer sample trace visualization (before vs after subtraction) on v1 and v2 | H | M | H | T-025/T-026 | — | **Done**: `docs/lab-log/v1_active_patches_before_after.png` & `v2_active_patches_before_after.png` generated & sent via Telegram |
| T1 | Share `meetings/20260717-meeting-juncheng-shared.md` with Juncheng; use personal prep yourself | M | L | H | T-007 | — | Stale for 2026-09-24; brief is `meetings/20260924-meeting.md` |
| T2 | After Chen meeting: record confirmed RQ, metrics, TS-vs-graph choice in `notes.md` + `progress.md` | H | L | H | T-007 | — | July item; 2026-09-24 brief is separate |
| T3 | Prepare crisp questions: goal vs metrics; graphs for understanding vs ASCAD leaderboard | H | L | M | T-007 | — | Standing gap; not the 14:00 fork |
| T4 | One-slide / short exp results as **sandbox evidence** (not the whole story) | M | M | M | T-005 | 2026-09-24 14:00 | Use roadmap §1, not a new slide deck |
| T5 | Fair apples-to-apples result table (fixed epochs/batch/seeds) | H | L | L | T-005 | — | Publication / later slides |
| T6 | Side-by-side CNN/MLP vs `timesnet_ascad_head` ranks (same schema) | H | L | M | T-006 | — | Baseline JSON exists |
| T7 | Diagnose GAT/GCN non-learning; re-sweep GIN after NaN fix | H | L | L | T-003 | — | Plan A in T-003 |
| T8 | Segment GNN: multi-Z SNR nodes (masks) after Plan C SNR-path rank=8 | H | L | L | T-003 | — | Plan C path done; relation/combine smokes weak; see plan-snr-xor-adjacency.md |
| T9 | Improve desync100 without pure label overfit (vanilla TimesNet) | M | L | L | T-004 | — | After meeting direction |
| T10 | Optional rank-curve plot export for meetings/papers | M | L | M | T-005 | — | |
| T11 | Optional: generic meeting brief template under `meetings/` | L | L | H | T-007 | — | Process hygiene |
| T12 | Habit: append `progress.md` after meaningful sessions | M | L | H | T-007 | — | Ongoing |
| T13 | STM32_AES_v2: TimesNet multi eval done (ranks ~random); decide next experiment | H | L | M | T-009 | — | catalog in T-004; v2 success is black-box pooled byte 4, not multilabel |
| T14 | Scoop ASCADv2 finetune-only other-15 byte sweep (exclude byte 4): finish runs and publish success/failure report | H | L | M | T-010 | — | Superseded: repro-v2 all-16 report exists; only byte 15 hit `ge_last` 0, not a strict break |
| T16 | Official CE POI-searcher that localizes without HW/SNR in the loss | H | L | L | T-014 | — | 2O already rank-0 on desync0/100; CE path still flat |
| T17 | 3-seed the **banded** 2O localizer (ablation only) | L | L | H | T-014 | — | Claimed spec is no-band+order; banded is not the model |
| T18 | Match the five ASCADv1 crop spans to AVR opcodes in secAES-ATmega8515 | M | L | L | T-018 | — | Value map is done; opcodes are not in the HDF5 |
| T19 | Persist the 2O localizer rank table (re-run seeds 0,1,2 + restarts 3, both desyncs) so the headline v1 claim has a JSON | H | M | M | T-014 | after 2026-09-24 | Roadmap gap 1. Real, but do not take tonight's GPU from brief A. Note: `exp-v1-a2-two-order-baseline.json` has **no** rank/restarts/batch/seed key, so gap 2's "needs restarts=3 + batch 256" is prose inference, not recorded data |
| T21 | Add the 2026 "Is it Really Broken?" critique to `Scoop/Analyzing Scoop's ASCADv2 Attack.md` | H | M | H | T-010 | — | Roadmap gap 9: doc states the break as verified fact |
| T22 | Patch-name SimMIM probe on the v1 crop: 5-way head, must beat the 81.6% "no name" base rate per class | M | L | M | T-018 | — | Designed in `out/B-segment-meanings.md`, not run. Needs GPU — after 14:00 |
| T25 | Reconcile SNR(\(S\)): "≈0.02" is computed but unsaved (`exp_2o.py:347-348`) while `docs/lab-log/snr-peak-stability-desync0.json` reports `snr_max` 0.00613 / mean 0.00496 from `utils/snr.py:190` | M | L | M | T-014 | — | Roadmap gap 3. Different provenance; do not print either figure until one is named |
| T28 | **Scoop sweep + survey entry for learned localization.** (a) Search 2024–2026 for any network that emits **two ordered windows for a second-order attack** with **no mask label** — the exact T-014 claim; tonight's pass verified venues but read neither body end to end. (b) Fold AutoPOI / ACNN / Cao B-CNN / Perin-Wu-Picek / MS23 into `docs/survey-ascadv2-dl-2022-2026.md` as a localization section with one delta line each vs T-014. (c) Get the two unread texts: JCEN 13(2) (confirm W1/W2 are chosen offline only, no per-trace selection) and AutoPOI §5 (how many regions, fixed size?) — needs Springer access. (d) Check whether any of the four **already runs on ASCADv1 with a second-order attack**, which is the only way this is a real scoop rather than adjacent work | H | M | M | T-014 | after 2026-09-24 | If (a) or (d) comes back positive, the v1 paper option dies and we say so at the next meeting. If it comes back clean, "not scooped" stops being an absence of evidence and becomes a result. Detail: `out/D-paper-gap.md` + `docs/notes/ASCADv1-leakage-localization-prior-art.md` §7. **Overlaps T26** (same survey idea, registered minutes apart by another session) — merge them; T28 carries the verified venue list and the one question that would falsify us. **STATUS 14:15 (a) DONE → `docs/2026-09-23/out/T28-scoop-sweep.md`. Answer: the ordered-pair-into-2O-CPA slot is still empty, but BOTH selling points are occupied — learned selection (4 methods) and "no mask label" (OccPoIs says mask/share-agnostic on ASCADf; Marquet 2023/611 finds mask-as-label *helps*). Closest mechanism = Spatial Dependency Analysis (2021/904, two windows, no learning); closest desync model = EstraNet (TCHES 2024). Still open: (b) fold ~12 new papers into the survey doc, (c) OccPoIs §4–§5 + EstraNet desync trace counts, (d) confirm which ASCAD variant 2021/817 attacks |
| T29 | **Crop-prior ablation + the missing rank JSON, in one run.** Re-run the claimed order recipe on byte 2 × {desync 0, desync 100} × seeds 0,1,2 with `--no-bands --lambda-order 1 --order-margin 200` (baseline, must reproduce 0/0/0) and again with `--order-margin 0 --time-bias 0` (does the order hinge alone still recover?), then `json.dump` the rank table + the playback rank per claimed byte (T24's control) | H | L | M | T-014 | after 2026-09-24 | Closes roadmap gap 1 (no result file) and gap 2's real question in one sweep, and it is the experiment that decides whether "learns windows without dataset knowledge" survives review. **Correction to gap 2's wording:** `batch=256` is a default parameter of `train_2o` (`exp_2o.py:144`) that **no caller overrides**, so it was 256 in the FAILing a2 run too — it is not one of the differing knobs. `restarts` is. Model is ~56 KB of weights on 8k×700 traces, so this is minutes-to-an-hour on GPU, not a campaign; still ask before running and never before 14:00 (`.grok/rules/04-ask-before-long-runs.md`). **Distinct from T27**, which is a literature comparison matrix; T29 is the GPU run that makes our own numbers citable |
| T26 | Expanded ASCADv1 DL-SCA survey: search literature for additional learned POI/window selection and 2O DL papers | M | M | M | T-014 | — | Task D follow-up: expand beyond preliminary 6 papers |
| T27 | Literature benchmark comparison matrix: systematically compare published ASCADv1 results against 2O Localizer to identify needed experiments | M | M | M | T-014 | — | Task D follow-up: evaluate trace efficiency, profiling size, and desync robustness gaps |
| T30 | **Unsupervised band selection.** Find α's live 100-sample bands without labels (variance/covariance structure, not SNR-vs-ground-truth), then re-run the coverage curve on them | H | M | M | T-020 | — | `LLM/band_scan.py --select snr` uses ground-truth labels, so tonight's 93.95 % is an **oracle for understanding the data, not an attack**. Worst-ranked bands score 1.7×, so ranking itself works — proven |
| T31 | **Resolution-preserving front end.** Same 1,600 samples, but a model that never averages adjacent samples: strided conv with no pooling, or raw-value patches plus late attentive compression | H | M | M | T-020 | — | Mean-of-2 adjacent samples collapses 86.0 % → 0.83 % accuracy, so early pooling is the binding constraint. Explains `A2` (0.77×) and GPAM (93 %) in one variable. ~30 min, uses `LLM/band_scan.py` |
| T32 | **`rank()` now defaults to worst-among-ties — tell any session that quotes ranks.** `ASCAD_test_models.rank()`/`full_ranks()`/`ranks_from_log_scores()` gained `tie_break`, default `"worst"`; pass `"best"` explicitly when comparing to published ASCAD tables | H | H | H | T-028 | — | Reason: uniform-replacing all 34 heads reported rank 0 on 16/16 bytes. Stored JSONs predate the change but are unaffected where best==worst; re-check any rank 0 with `experiments/rank_degeneracy.py` |

---

## Done (recent)

| Date | Action | Task |
|------|--------|------|
| 2026-10-04 | **T-019 Table 11 holdout eval run and closed out**: 80,000 never-seen shard-8 traces, best checkpoint (metrics epoch 64) + `epoch_070` control → `rm` 99.996 / MeanRank 0.003, `rout` 10.480 / 10.517, `p[i]` 86.514 / 0.181, `c[i]` 0.679 / 108.631, all against the paper's 100.00 / 18.25 / 95.07 / 1.18. New evaluator `GPAM/artifacts/C-table11-holdout-eval.py` (6.1 min/checkpoint, inference only, ties counted against the target); report `GPAM/artifacts/C-table11-report.md`. Also corrected the stale "in flight, epoch 14" claims in `INDEX.md` and T-019 — the campaign was **killed externally at epoch 76**, it did not crash | T-019 |
| 2026-10-04 | T20: stray `T015-ascadv2-3o-localizer.md` retired to a stub and merged into T-013 (roadmap gap 8) | T-022 |
| 2026-10-04 | T15 cancelled with T-013: the 3O sibling's oracle gate failed on SNR(C)=0.016 and every label-free loss failed | T-013 |
| 2026-09-22 | Restored NVIDIA driver after kernel 7.0.0-30: DKMS install `nvidia/580.142` + `modprobe`; `nvidia-smi` and `py10` `torch.cuda` OK | T-021 |
| 2026-09-24 | Brief D delivered and **merged** into `docs/2026-09-23/out/D-paper-gap.md`: verdict **2 — Not yet**; six prior-art papers verified with DOIs; six citation errors in the repo note corrected (T23 below); v1 playback control run (T24 below) | T-014 |
| 2026-09-24 | T23 Prior-art note citation pass → **§7 of `docs/notes/ASCADv1-leakage-localization-prior-art.md`**. **AutoPOI added, and it is the closest localization neighbor** (Remmerswaal/Wu/Tiran/Mentens, JCEng 2023, ePrint 2023/008, DOI 10.1007/s13389-023-00328-y — PPO over feature regions, reward = template attack; datasets ASCAD_F/ASCAD_R/AES_HD/CHES_CTF via proxy; no second-order sentence). **MS23 settled after three readings tonight**: not a phantom, and not a *learned* localizer either — it is the **ASCADv2 paper** (Masure & Strullu, JCEng 13(2):129–147, 2023; preprint ePrint 2021/592) whose official extracted set **is** two windows, W1 `[205k,210k)` ‖ W2 `[455k,465k)` = 15k, but **frozen offline by code analysis + SNR**, with mask labels in training (worst-case). The intermediate "it has a learned region head" reading came from trusting an OpenAlex abstract **reconstruction** — do not cite rebuilt abstracts. Also: Cao B-CNN confirmed IEEE TIFS 2022 DOI 10.1109/TIFS.2022.3216959; RPOI/OPOI row = Perin/Wu/Picek TCHES 2022(4); 700-crop cite = ePrint 2018/053 / JCEng 2020 10(2):163–188 **not** "Maghrebi et al."; arXiv:2410.22425 is **synthetic-only** and "AdvMask" is our shorthand; the note's "\(W_1\|W_2\) from MS23" steal-list item re-attributed | T-014 |
| 2026-09-24 | T24 v1 frequency-playback control **measured** (`docs/2026-09-23/out/D-v1-playback-control.py` + `.json`): byte 2 playback rank at 10k = **216** (8k train) / **240** (50k train) vs chance 127.5 ⇒ the AOD prior cheat cannot fake our byte-2 rank 0. Label chi² equals the plaintext chi² (293.5, p=0.049) — v1 non-uniformity is sampling noise, not mask structure. Caution: **byte 7 hits rank 0 with no leakage at all**, so single-byte rank claims must print that byte's playback rank. Desync-invariant | T-014 |
| 2026-09-24 | Brief B delivered: 87-patch census of the v1 crop — **16 named / 71 unnamed** (81.6% "no name"); T-018's 28 segments / 8 valued reproduced exactly. `docs/2026-09-23/out/B-segment-meanings.md` | T-018 |
| 2026-09-23 | ASCADv1 value map: 5 spans, 19% of the 700-sample crop; notebook `visualize/segment_value_map.ipynb` | T-018 |
| 2026-08-20 | 2O windows: key_rank=0 on fixed-key desync100; FAQ in `docs/notes/v1-poi-faq.md` | T-014 |
| 2026-08-19 | Create `docs/notes/` and move datasets/metrics/models | T-015 |
| 2026-07-24 | Plan C SNR-path e75 rank_final=8; snr utils + edge modes | T-003 |
| 2026-07-24 | Plan B first cut: `seg_graphsage` v1 desync0 e75 (rank_final=7); plan+report in lab-log | T-003 |
| 2026-07-17 | Seed I/U/S triage rule + this `todo.md` | T-007 |
| 2026-07-17 | Meeting briefs (personal + shared) for Chen 1:1 | T-007 |
| 2026-07-17 | Lab-log folder + progress/notes init | T-007 |

---

## How to add an item

```markdown
| T13 | Short action verb phrase | H | M | H | T-00x | Chen / group / — | Done when … |
```

Then refresh **Priority snapshot** (sort by rule). Check off → move row to **Done (recent)** with date.
