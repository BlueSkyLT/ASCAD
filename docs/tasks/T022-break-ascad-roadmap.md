# T-022: Roadmap — breaking ASCADv1 / ASCADv2

## Status

`in_progress`

## Goal

Maintain `docs/roadmap-break-ascad.md`: one tree that lists every model, variation and probe aimed at
ASCADv1/ASCADv2, each with its highest status rung (`surveyed` → `implemented` → `ran@byte` →
`win@byte` → `multi-seed` / `all-16`) and an evidence grade (`A` artifact / `L` doc-log only /
`U` unverified).

## Context

- Roadmap file: `docs/roadmap-break-ascad.md`
- Covers: `leakage_localizer/`, `LLM/`, `Scoop/`, `GPAM/`, `TimesNet/`, `GNN/`, CNN/MLP baselines
- Reference facts: `docs/notes/metrics.md`, `docs/notes/datasets.md`, `docs/notes/v1-poi-faq.md`
- Constraint: this file is a *status tree*, not a session log. Narrative stays in the task files.

## Plan

- [x] (I:H U:H S:M) Audit all four attack families against artifacts on disk (done 2026-09-23 via
      four parallel read-only sweeps: TimesNet+baselines, Scoop+GPAM, leakage_localizer, LLM+GNN+support)
- [x] (I:H U:H S:M) Write the roadmap with scoreboard, per-dataset trees, dead-end index, gaps
- [x] (I:H U:M S:H) Verify the anchor numbers directly against their JSONs
      (`benchmark_runs/summary_full_fixedkey_gpu.json`, `benchmark_runs_cnn_mlp_baseline/`,
      `Scoop/artifacts/ascadv2_repro_v2/metrics/*b15*summary.json`)
- [ ] (I:H U:H S:M) Close gap 1: persist the 2O rank table so the headline v1 claim is artifact-backed
- [ ] (I:H U:M S:H) Close gaps 8–9: retire/renumber the stale `T015-ascadv2-3o-localizer.md`;
      fix `Scoop/Analyzing Scoop's ASCADv2 Attack.md` to state the 2026 critique
- [ ] (I:M U:L S:M) Re-score the tree when a branch changes; keep §1 scoreboard and §7 snapshot current

## Decisions

- 2026-09-23: The roadmap lives at `docs/roadmap-break-ascad.md` (docs root, like the survey), not in
  `docs/notes/` — notes are reference material, this is a status register.
- 2026-09-23: Every claim gets an evidence grade. Doc-only numbers are tagged `L`/`U` rather than
  deleted, so the gap list in §6 stays actionable.
- 2026-09-23: Two naming traps recorded up front — v1's pilot byte is **byte 2** (not 4), and
  "Scoop repro v1/v2" means our two configs, not the datasets.

## Working notes / session log

### 2026-09-23 — first full audit

- Summary: built the roadmap from four parallel artifact-level sweeps plus a data inventory. Net
  findings: three real ASCADv1 wins, all byte-2 only (2O localizer rank 0 × 3 seeds × both desyncs;
  `timesnet_ascad_head` rank mean 0.0/2.0/1.5; CNN baseline rank 0 @190). Nine mechanisms have touched
  ASCADv2 and none is a strict break — the single exception is Scoop repro-v2 byte 15
  (`ge_last 0.0`, `traces_to_ge0 100001`) on the contested re-split, single seed. The whole
  LLM/transformer family sits at chance (CE 5.545 = ln 256), consistent with SNR(S) ≈ 0.02.
  GPAM is blocked and has zero artifacts (`gpam_budget_run.log` is 0 bytes); raw v2 800k is ~40%
  downloaded (shard 8 complete, 1–4 partial, 5–7 empty).
- Files touched: created `docs/roadmap-break-ascad.md`, `docs/tasks/T022-break-ascad-roadmap.md`,
  this row in `docs/tasks/INDEX.md`, one item in `docs/lab-log/todo.md`.
- Open questions: does the 2O rank-0 survive a persisted re-run (gap 1)? What are v2 bytes 5/11
  `traces_to_rank0` 150/100 in the e60 artifact, which no doc mentions (gap 7)?

### 2026-09-23 — meeting uses the roadmap as the scoreboard

- The 2026-09-24 14:00 plan does not change the tree. It schedules one decision test (patch transformer vs `alpha_mask`) and three read-only briefs. GPAM Table 11 and the 2O rank JSON stay open. See `docs/2026-09-23/README.md`.

## Next actions

| Action | I | U | S | Notes |
|--------|---|---|---|-------|
| Persist 2O rank table (re-run seeds 0,1,2, restarts 3, both desyncs) | H | M | M | After 14:00. Tonight's GPU is the alpha probe, not this re-run |
| Retire/renumber stale `T015-ascadv2-3o-localizer.md` | M | M | H | roadmap gap 8 |
| Add the 2026 false-positive critique to `Scoop/Analyzing…md` | H | M | H | roadmap gap 9 |
| Save SNR(S)/SNR(r)/SNR(C) to JSON | H | M | H | roadmap gap 3 |

### Priority snapshot

- **Do now:** nothing on this file until `docs/2026-09-23/out/` exists. The 14:00 scoreboard is already §1.
- **Before next meeting:** read A2 before claiming anything new about LLM-on-v2.
- **Deep work (park):** persist the 2O JSON, GPAM after the raw download, AOD-GE, v2 bytes 5–15.

## Related

- Task ids: T-003, T-004, T-006, T-009, T-010, T-011, T-013, T-014, T-015, T-016, T-017, T-018, T-019, T-020
- `docs/lab-log/todo.md` items: T19 (roadmap maintenance), T14, T16, T17
- Files: `docs/roadmap-break-ascad.md`, `docs/survey-ascadv2-dl-2022-2026.md`
