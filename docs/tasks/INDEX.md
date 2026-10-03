# Task index

Registry of work units for this repository. Grok sessions should read this first, then open the active task file.

**Active focus:** [T-020](./T020-llm-transformer-sca-pipeline.md) — LLM/Transformer SCA pipeline (patchify → interpret → classify)

Survey (separate): [T-012](./T012-ascadv2-dl-survey.md) — canonical `docs/survey-ascadv2-dl-2022-2026.md` (root copies merged 2026-08-19)

Soft-window v2 (not active): [T-013](./T013-soft-window-abc.md) — plan `docs/superpowers/plans/2026-08-17-poi-searcher.md`

Recovered 2026-07-17 from old Grok session histories under `~/.grok/sessions/%2Fhome%2Flantian%2FASCAD/` plus git/repo artifacts. Many UI session titles are “primer” noise; research content is linked inside each task file by session id.

Task files are named `TNNN-short-slug.md` (id prefix, zero-padded, no date).

| ID | Title | Status | File | Last updated | Notes |
|----|--------|--------|------|--------------|-------|
| T-001 | Dataset visualization & understanding | in_progress | ./T001-dataset-visualization.md | 2026-09-25 | Segment notebooks executed 2026-09-25 and sent as HTML on Telegram; `docs/notes/datasets.md` I/O filled |
| T-002 | TF → pure PyTorch migration | done | ./T002-pytorch-migration.md | 2026-07-17 | Commit `1157ed5`; user-confirmed done |
| T-003 | GNN models on ASCAD | in_progress | ./T003-gnn-models.md | 2026-07-24 | Plan C weighted A: rank_final=35 (vs Plan B 7); multi-Z next |
| T-004 | TimesNet original (non-Scoop) | in_progress | ./T004-timesnet-original.md | 2026-09-23 | v1 head@1e-4/150e; v2 pooled byte4 flowchart; `pool_mode` avg/max |
| T-005 | SCA metrics & experiment analysis | in_progress | ./T005-sca-metrics-and-results.md | 2026-08-19 | `docs/notes/metrics.md` + GE in benchmark eval |
| T-006 | CNN / MLP baselines | in_progress | ./T006-cnn-mlp-baselines.md | 2026-07-17 | rank JSON done; side-by-side vs TimesNet/GNN open |
| T-007 | Lab log — progress & meetings | in_progress | ./T007-lab-log-and-meetings.md | 2026-09-24 | 14:00 speaking prep + blank notes page |
| T-008 | GPU training efficiency (util/memory) | in_progress | ./T008-gpu-training-efficiency.md | 2026-07-17 | Current Best Configs table filled from `DEFAULT_MODEL_PROFILES` |
| T-009 | STM32_AES_v2 key recovery | in_progress | ./T009-stm32-key-recovery.md | 2026-07-24 | e60 done rank~124; e90 skipped; meeting note |
| T-010 | Scoop ASCADv2 reproduction | in_progress | ./T010-scoop-ascadv2-reproduction.md | 2026-08-04 | v2 +25e continue on bytes 0–14; b15 already GE=0; single BYTE_SWEEP_REPORT.md |
| T-011 | TimesNet + Scoop ASCADv2 black-box | in_progress | ./T011-timesnet-scoop-blackbox.md | 2026-08-13 | Scoop pooled TimesNet bytes 0–4; b4 GE last 2.75, no GE→0 |
| T-012 | ASCADv2 DL survey (2022–2026) | in_progress | ./T012-ascadv2-dl-survey.md | 2026-08-19 | Canonical: `docs/survey-ascadv2-dl-2022-2026.md` (root copies merged; lit only; not T-011) |
| T-013 | Soft windows α/β/C + algebraic consistency | planned | ./T013-soft-window-abc.md | 2026-08-19 | v2 affine; per-trace \(h_\psi\); plan rewritten |
| T-014 | ASCADv1 leakage localizer (2O windows) | in_progress | ./T014-ascadv1-poi-searcher.md | 2026-09-23 | Claimed: no-band + order; spec §6.2 knobs / §8 loss; rank 0 desync0/100 |
| T-015 | Topic reference notes (`docs/notes/`) | done | ./T015-topic-reference-notes.md | 2026-08-20 | Plain-language wiki; notation + `v1-poi-faq.md` |
| T-016 | ASCADv1 Time-LLM GPT-2 standalone | in_progress | ./T016-ascadv1-timellm-gpt2.md | 2026-09-23 | Standalone train/eval script; frozen GPT-2 default; smoke run pending |
| T-017 | LLM methods organization & integration | active | ./T017-llm-methods-organization.md | 2026-09-23 | LLM/ package; SimMIM embed-level mask token on Route B |
| T-018 | ASCADv1 segment value map | in_progress | ./T018-ascadv1-segment-value-map.md | 2026-09-24 | Crop: 5 valued spans, 19% of samples; brief B added the patch census (16/87 named); opcodes still open |
| T-019 | GPAM ASCADv2 reproduction (paper-faithful) | active | ./T019-gpam-ascadv2-reproduction.md | 2026-09-29 | **raw 1M traces DO transfer at 40k: held-out `rm`(α) 98.340%** on shard 2, gate `PASSES`, ~83h 34-head campaign launched 21:23. α-only (β/C/π at chance); worst-case, not a break. Supersedes the 19:00 "DO NOT GENERALISE at 4k" reading | |
| T-020 | LLM/Transformer SCA pipeline | active | ./T020-llm-transformer-sca-pipeline.md | 2026-09-29 | α IS readable: 16 SNR bands (10.7% of crop) = 93.95%, matches full-crop GPAM `rm` 92.95%. A2's "family is dead" verdict **retracted** in `out/A-verdict.md`. Binding constraint is per-sample resolution: mean-of-2 adjacent samples → 0.83%. **Whole-family survey (variants tried + results + RESULT/ORACLE/SMOKE grading): `docs/notes/patch-pipeline-survey.md`** |
| T-021 | NVIDIA driver recovery (kernel mismatch) | done | ./T021-gpu-driver-kernel-mismatch.md | 2026-09-22 | Kernel 7.0.0-30 vs DKMS 6.17; rebuilt nvidia/580.142; RTX 5090 + py10 CUDA OK |
| T-022 | Break-ASCAD roadmap (status tree) | in_progress | ./T022-break-ascad-roadmap.md | 2026-09-23 | Scoreboard unchanged. 14:00 plan points at it from `docs/2026-09-23/`; no strict v2 break |
| T-023 | Professor Mentor Agent Definition & Setup | done | ./T023-professor-mentor-agent.md | 2026-09-24 | High-school level mentor agent defined; code reading + web search + .md reporting, no code writing |
| T-024 | TimesNet Multi-Task (sub_avg) vs GPAM | planned | ./T024-timesnet-multitask-subavg-vs-gpam.md | 2026-09-24 | 34-head joint supervision with mean-trace subtraction on extracted v2 |
| T-025 | ASCADv2 unsupervised variance-bulge idle/active localizer | active | ./T025-v2-variance-idle-active-localizer.md | 2026-09-24 | Zero-GT variance scanning on mean-centered v2 traces + patch classification |
| T-026 | ASCADv2 self-supervised autoencoder idle/active localizer | active | ./T026-v2-autoencoder-anomaly-localizer.md | 2026-09-24 | Zero-GT patch autoencoder anomaly scoring for idle vs active ARM cycles |
| T-028 | Non-worst-case ASCADv2 break via pseudo-label bootstrap | blocked | ./T028-v2-nonworstcase-pseudolabel-break.md | 2026-10-03 | **Phase 0 done, Phase 1 NO-GO.** Baseline verified genuine (34-head MultiResNet, 16/16 rank 0, median z +9.50) — but `rank()` reports rank 0 for TIED score vectors, so uniform-ablation "succeeded" too; added `experiments/rank_degeneracy.py`. π not essential (12/16 with π zeroed); C is the only data-dependent carrier, so "C decorative" is refuted. Label-free α: k-means/quantisation sit on the random-labelling floor (2.0%) because α owns 1.68% of trace variance vs 1.29% for a random 255-way split. Phase 2 blocked; Step 3.1 unexecutable as written (no shuffle-OFF ASCADv2). Code: `experiments/{ablation_heads,rank_degeneracy,unsup_alpha,unsup_alpha_residual,alpha_variance_budget}.py` |
| T-027 | C3 dataset — standalone break work (GPAM-c3) | active | ./T027-c3-standalone-break.md | 2026-09-28 | **Not ASCAD research.** `.sx` is plain HDF5: 50k × 4,500,002 float32. 8-bit ADC → 225 GB uint8 cube, **bit-exact** (225e9 elements verified, 0 mismatches). Construction is `E_K(RAND⊕OPc)` ⇒ label `SBOX[rand[b]^(K^OPc)[b]]`; round 1 recovers only `K⊕OPc`. Split 38k/10k/2k. Gate 1 scan running (J=94) |

## Status legend

`planned` · `active` · `in_progress` · `blocked` · `done` · `cancelled`

## Session → task map (recovered)

| Session id (prefix) | Model | Primary task(s) | Real user work (not primer) |
|---------------------|-------|-----------------|-----------------------------|
| `019f37f8-260e…` | grok-build | T-005 | Explain rank metrics → write `metrics.md` (now `docs/notes/metrics.md`) |
| `019f1d54-f0d6…` | grok-build / b | T-005 | Subagent: rank_final / full_ranks analysis |
| `019f38bb-996f…` | grok-build | T-001, T-003 | Segment stats; `find_segments_v2` |
| `019f3896-8203…` | grok-build | **T-008**, T-003 | GPU util/memory tuning; large batch; GPU-resident loader; GIN NaN under AMP |
| `019f3c93-824a…` | grok-build | T-006 | `run_cnn_mlp_best_with_delay.sh` (wait-for-idle; not throughput tuning) |
| `019f64f3-b502…` | grok-4.5 | T-005, T-004, T-003, T-006 | Multi-LR result analysis; TimesNet vs CNN complexity |
| `019f3b54-f3ef…` | grok-build | _(tooling)_ | Superpowers skills — not a research task |
| `019f64ec-8413…` | grok-4.5 | _(meta)_ | Task-memory scaffolding (this system) |
| `019f6bdd…` / `019f6bee…` | grok-4.5 | _(empty/primer)_ | No substantive research content |

## How to register

1. Copy `_template.md` to `TNNN-short-slug.md` (next free id, zero-pad to 3+ digits, e.g. `T009-…`).
2. Use display id `T-NNN` in the file title and INDEX row.
3. Add a row here; set **Active focus** when it is the current work.
