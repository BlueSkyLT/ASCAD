# T-012: ASCADv2 deep-learning survey (2022–2026)

## Status

`in_progress`

## Goal

Produce a durable **literature survey** of deep learning side-channel attacks on AES using the **ASCADv2** dataset, fixed to roughly **2022–2026**, with **more recent works weighted higher**.

**Done when:** threat-model-separated comparison tables are filled for the main works; local `papers/` are read; known gaps (MS23, WPP23, remaining local PDFs, 2024–2026 ePrints) are either incorporated or explicitly marked out-of-scope; `docs/notes/models.md` ASCADv2 section reflects the survey.

This task is **literature only**. It is **not** TimesNet training (T-011), not Scoop MLP reproduction (T-010), and not multilabel STM32 runs (T-009).

## Context

| Path | Role |
|------|------|
| `docs/survey-ascadv2-dl-2022-2026.md` | **Main survey draft** (canonical; merged 2026-08-19) |
| `papers/` | Local PDFs (Scoop, Is it Really Broken?, GPAM, DS-NSCA, Rock…, Variously supervised…) |
| `docs/notes/models.md` | Short ASCADv2 model list to update from the survey |

Optional background only (other tasks, do not fold into survey goals): T-010 Scoop reproduction may *cite* the same papers; that does not make experiment code part of T-012.

## Plan

- [x] (I:H U:M S:H) Initial deep-research pass; write draft survey
- [ ] (I:H U:M S:M) Full-text remaining local PDFs: DS-NSCA, Variously supervised, Rock and a Hard Place
- [ ] (I:H U:M S:M) Fill secondary gaps: MS23, WPP23, Wu 2023/724, Yap & Jap 2024/167, etc.
- [ ] (I:H U:H S:H) Expand `docs/notes/models.md` ASCADv2 table (threat model, input, target, metric, AOD note)
- [ ] (I:M U:H S:H) Optional short meeting-facing literature map
- [ ] (I:H U:L S:L) Optional polish into a longer structured survey write-up

## Decisions

- 2026-08-19: Three survey files were one lineage, not three surveys. **Keep only** `docs/survey-ascadv2-dl-2022-2026.md`. Deleted repo-root `survey-ascadv2-dl-2022-2026.md` and `survey-ascadv2-dl-2022-2026_ver1.md` after merging unique bits (GPAM \(c[i]\) formula, WPP23 ~120× shuffle-on cost, SAC preproceedings URL, remaining-coverage / search queries).
- 2026-08-05: Canonical survey path is `docs/survey-ascadv2-dl-2022-2026.md` (not experiment plans under other tasks).
- 2026-08-05: Never collapse GE / traces-to-rank0 across Scoop black-box, GPAM multi-task, and DDPM collision settings.
- 2026-08-05: 2026 AOD critique (“Is it Really Broken?”) is first-class in the narrative, not a footnote.
- 2026-08-05: T-012 is independent of T-011 (TimesNet black-box experiments).

## Working notes / session log

### 2026-08-19 — merge duplicate survey files

- Compared `docs/survey-ascadv2-dl-2022-2026.md` (12.7k, 2026-08-05 partial), root `survey-ascadv2-dl-2022-2026.md` (52.8k, 08-06 complete), `survey-ascadv2-dl-2022-2026_ver1.md` (44.2k, 08-06 intermediate).
- Same document, later drafts strictly expand the earlier ones. Merged unique leftover facts into canonical `docs/` file; removed the two root copies.
- T-014 remains active focus; this was hygiene on T-012.

### 2026-08-05 — survey draft + task registration

- Deep-research workflow finished **Partial**.
- Core narrative: Scoop claimed non-worst-case full-15k break; ePrint 2026/338 labels classical GE success a false positive (AOD-GE / prior bias); GPAM/DDPM/multi-task are important but different threat models.
- Draft written to `docs/survey-ascadv2-dl-2022-2026.md`.
- Earlier handoff mistakenly mixed Scoop×TimesNet feasibility (unrelated plan); **removed from this task**.

## Next actions

| Action | I | U | S | Notes |
|--------|---|---|---|-------|
| Full-text remaining local `papers/` PDFs | H | M | M | DS-NSCA, Rock, Variously supervised |
| ePrint pass for 2024–2026 ASCADv2-only works | H | M | M | See survey § Coverage gaps |
| Update `docs/notes/models.md` from survey table | H | H | H | Meeting-facing |
| Meeting one-pager if needed | M | H | H | Optional |

### Priority snapshot

- **Do now:** continue deep search from survey “Coverage gaps”
- **Before next meeting:** `docs/notes/models.md` + short literature map if meeting needs it
- **Deep work:** full secondary census + polished write-up structure

## Related

- Survey draft: `docs/survey-ascadv2-dl-2022-2026.md`
- `docs/notes/models.md` ASCADv2 section
- Papers: ePrint 2025/498 (Scoop), 2026/338 (AOD critique), arXiv 2306.07249 (GPAM)

**Out of scope for T-012:** T-011 TimesNet, T-009 multilabel training, T-010 experiment runs (may cite papers only).
