# Task index

Registry of work units for this repository. Grok sessions should read this first, then open the active task file.

**Active focus:** [T-028](./T028-v2-nonworstcase-pseudolabel-break.md) — non-worst-case ASCADv2 break (pseudo-label bootstrap); pending go/stop decision after Phase 1 NO-GO

In parallel (other agent): [T-019](./T019-gpam-ascadv2-reproduction.md) — GPAM raw campaign eval

Survey (separate): [T-012](./T012-ascadv2-dl-survey.md) — canonical `docs/survey-ascadv2-dl-2022-2026.md` (root copies merged 2026-08-19)

Recovered 2026-07-17 from old Grok session histories under `~/.grok/sessions/%2Fhome%2Flantian%2FASCAD/` plus git/repo artifacts. Many UI session titles are “primer” noise; research content is linked inside each task file by session id.

Task files are named `TNNN-short-slug.md` (id prefix, zero-padded, no date).

| ID | Title | Status | File | Last updated | Notes |
|----|--------|--------|------|--------------|-------|
| T-001 | Dataset visualization & understanding | paused | ./T001-dataset-visualization.md | 2026-10-04 | Paused: segment notebooks + `docs/notes/datasets.md` I/O done; resume when a figure is needed |
| T-002 | TF → pure PyTorch migration | done | ./T002-pytorch-migration.md | 2026-07-17 | Commit `1157ed5`; user-confirmed done |
| T-003 | GNN models on ASCAD | paused | ./T003-gnn-models.md | 2026-10-04 | Paused since 07-24: Plan C weighted A rank_final=35 (vs Plan B 7); segment-node Plan B awaits a meeting decision |
| T-004 | TimesNet original (non-Scoop) | paused | ./T004-timesnet-original.md | 2026-10-04 | Paused: catalog complete (v1 head@1e-4/150e; v2 pooled byte4); remaining runs optional, ask before GPU |
| T-005 | SCA metrics & experiment analysis | paused | ./T005-sca-metrics-and-results.md | 2026-10-04 | Paused: `docs/notes/metrics.md` + GE in eval. Owns the fair cross-model table (absorbs T-006's side-by-side) |
| T-006 | CNN / MLP baselines | paused | ./T006-cnn-mlp-baselines.md | 2026-10-04 | Paused: rank JSON done; side-by-side vs TimesNet/GNN tracked in T-005 |
| T-007 | Lab log — progress & meetings | in_progress | ./T007-lab-log-and-meetings.md | 2026-09-24 | 14:00 speaking prep + blank notes page |
| T-008 | GPU training efficiency (util/memory) | paused | ./T008-gpu-training-efficiency.md | 2026-10-04 | Paused: Current Best Configs table filled; remaining items optional tuning |
| T-009 | STM32_AES_v2 key recovery | done | ./T009-stm32-key-recovery.md | 2026-10-04 | Superseded by T-019 / T-024 / T-028; TimesNet multilabel ended at ~random ranks |
| T-010 | Scoop ASCADv2 reproduction | done | ./T010-scoop-ascadv2-reproduction.md | 2026-10-04 | All-16 v2 HPO sweep done; only byte 15 GE 0 (at 100,001 traces), not a strict break; `BYTE_SWEEP_REPORT.md` |
| T-011 | TimesNet + Scoop ASCADv2 black-box | done | ./T011-timesnet-scoop-blackbox.md | 2026-10-04 | Bytes 0–4 done; b4 GE last 2.75, no GE→0. More bytes = optional backlog |
| T-012 | ASCADv2 DL survey (2022–2026) | paused | ./T012-ascadv2-dl-survey.md | 2026-10-04 | Paused: canonical `docs/survey-ascadv2-dl-2022-2026.md`; coverage gaps listed in the task |
| T-013 | Soft windows α/β/C + algebraic consistency | cancelled | ./T013-soft-window-abc.md | 2026-10-04 | Cancelled: absorbed the stray 3O-localizer file (`T015-ascadv2-3o-localizer.md`, now a stub); gate FAIL on SNR(C)=0.016, label-free losses FAIL |
| T-014 | ASCADv1 leakage localizer (2O windows) | paused | ./T014-ascadv1-poi-searcher.md | 2026-10-04 | Paused: no-band + order recipe, rank 0 desync0/100 byte 2; persist 2O rank JSON + crop-prior ablation need GPU go-ahead |
| T-015 | Topic reference notes (`docs/notes/`) | done | ./T015-topic-reference-notes.md | 2026-08-20 | Plain-language wiki; notation + `v1-poi-faq.md` |
| T-016 | ASCADv1 Time-LLM GPT-2 standalone | done | ./T016-ascadv1-timellm-gpt2.md | 2026-10-04 | Negative result: 40-epoch reprogram at chance; family covered by T-020 survey |
| T-017 | LLM methods organization & integration | paused | ./T017-llm-methods-organization.md | 2026-10-04 | Paused since 09-23; `LLM/` package + SimMIM embed mask token. Results summarised in T-020 survey |
| T-018 | ASCADv1 segment value map | done | ./T018-ascadv1-segment-value-map.md | 2026-10-04 | `visualize/segment_value_map.ipynb` is the answer: crop 5 valued spans, 19% of samples |
| T-019 | GPAM ASCADv2 reproduction (paper-faithful) | active | ./T019-gpam-ascadv2-reproduction.md | 2026-09-29 | **raw 1M traces DO transfer at 40k: held-out `rm`(α) 98.340%** on shard 2, gate `PASSES`, ~83h 34-head campaign launched 21:23. α-only (β/C/π at chance); worst-case, not a break. Supersedes the 19:00 "DO NOT GENERALISE at 4k" reading | |
| T-020 | LLM/Transformer SCA pipeline | paused | ./T020-llm-transformer-sca-pipeline.md | 2026-10-04 | Paused (focus → T-028): α readable from 16 oracle SNR bands = 93.95%; open problem is label-free band finding. Survey: `docs/notes/patch-pipeline-survey.md` |
| T-021 | NVIDIA driver recovery (kernel mismatch) | done | ./T021-gpu-driver-kernel-mismatch.md | 2026-09-22 | Kernel 7.0.0-30 vs DKMS 6.17; rebuilt nvidia/580.142; RTX 5090 + py10 CUDA OK |
| T-022 | Break-ASCAD roadmap (status tree) | in_progress | ./T022-break-ascad-roadmap.md | 2026-09-23 | Scoreboard unchanged. 14:00 plan points at it from `docs/2026-09-23/`; no strict v2 break |
| T-023 | Professor Mentor Agent Definition & Setup | done | ./T023-professor-mentor-agent.md | 2026-09-24 | High-school level mentor agent defined; code reading + web search + .md reporting, no code writing |
| T-024 | TimesNet Multi-Task (sub_avg) vs GPAM | done | ./T024-timesnet-multitask-subavg-vs-gpam.md | 2026-10-04 | MultiResNet 34h + sub_avg: all 16 bytes rank 0 (z +3.89…+6.86σ, re-verified in T-028). TimesNet multi-task half never ran → backlog |
| T-025 | ASCADv2 unsupervised variance-bulge idle/active localizer | done | ./T025-v2-variance-idle-active-localizer.md | 2026-10-04 | Label-free idle/active localizer: variance scanner + patch autoencoder (T-026 merged in). AE 6.7% vs variance 90% active on v2 |
| T-026 | ASCADv2 self-supervised autoencoder idle/active localizer | done | ./T026-v2-autoencoder-anomaly-localizer.md | 2026-10-04 | Merged into T-025 (stub) |
| T-027 | C3 dataset — standalone break work (GPAM-c3) | blocked | ./T027-c3-standalone-break.md | 2026-10-04 | Blocked on collaborator: Gate 1 full first-order scan DONE 09-29, Outcome E — no first-order leakage (max abs t 6.30 vs 6.44). Need: what one trace covers, OPc=f(K)?, `auth` |
| T-028 | Non-worst-case ASCADv2 break via pseudo-label bootstrap | active | ./T028-v2-nonworstcase-pseudolabel-break.md | 2026-10-04 | **Phase 0 done, Phase 1 NO-GO.** Baseline verified genuine (34-head MultiResNet, 16/16 rank 0, median z +9.50) — but `rank()` reports rank 0 for TIED score vectors, so uniform-ablation "succeeded" too; added `experiments/rank_degeneracy.py`. π not essential (12/16 with π zeroed); C is the only data-dependent carrier, so "C decorative" is refuted. Label-free α: k-means/quantisation sit on the random-labelling floor (2.0%) because α owns 1.68% of trace variance vs 1.29% for a random 255-way split. Phase 2 blocked; Step 3.1 unexecutable as written (no shuffle-OFF ASCADv2). Code: `experiments/{ablation_heads,rank_degeneracy,unsup_alpha,unsup_alpha_residual,alpha_variance_budget}.py` |

## Status legend

`planned` · `active` · `in_progress` · `paused` · `blocked` · `done` · `cancelled`

`paused` = no current work, resumable; next steps in the task file. Last full status cleanup: 2026-10-04.

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
