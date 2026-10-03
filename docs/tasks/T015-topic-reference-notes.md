# T-015: Topic reference notes (`docs/notes/`)

## Status

`done`

## Goal

Give humans and agents a multi-file place for stable reference notes (notation, datasets, metrics, literature lists), separate from task logs and meeting prep.

## Context

- Catalog: `docs/notes/README.md`
- Rule: `.grok/rules/05-notes.md`
- Moved from repo root: `ascad_notes.md` → `docs/notes/datasets.md`; `metrics.md` → `docs/notes/metrics.md`; `models.md` → `docs/notes/models.md`
- Active research focus remains T-014

## Plan

- [x] (I:M U:L S:H) Create `docs/notes/` + catalog README
- [x] (I:M U:L S:H) Move existing root notes into the catalog
- [x] (I:M U:L S:H) Point `AGENTS.md` and a `.grok/rules/` file at the catalog
- [x] (I:M U:L S:H) Update live path references in tasks / lab-log / plans
- [x] (I:H U:L S:H) Write `docs/notes/notation.md` (α = \(r_m \neq r_{\mathrm{in}}\); byte ranges)

## Decisions

- 2026-08-19: Canonical reference wiki is `docs/notes/`, not repo-root Markdown and not `docs/lab-log/notes.md` (that file stays meeting background).
- 2026-08-19: Agents look up notes on demand via the catalog; they do not auto-load every note.
- 2026-08-19: `ascad_notes.md` renamed to `datasets.md` so the filename matches the topic.
- 2026-08-19: **α = \(r_m\), not \(r_{\mathrm{in}}\)**. α is `uint8` but only \(\{1,\ldots,255\}\`. β / \(r_{\mathrm{out}}\) and v1 \(r\) are full 256-way bytes. Locked in `docs/notes/notation.md` from `multilabelize` + local HDF5 + MS23 §1.3.1.
- 2026-08-20: Notes in `docs/notes/` are written for a reader **new to the field** (high-school level story first, then “in the files” names). Facts and HDF5 paths stay.

## Working notes / session log

### 2026-08-20 — rewrite notes in plain language

- User: `docs/notes` was too tech-heavy.
- Rewrote README, notation, datasets, metrics, models, v1-poi-faq: story first, keep numbers/paths in tables.
- Rule `05-notes.md`: same writing bar for future edits.
- T-014 remains active.

### 2026-08-20 — v1 POI FAQ

- Added `docs/notes/v1-poi-faq.md` from recurring T-014 questions (leak vs SNR, \(S^\star\) vs \(k\), HW, sample product vs byte Cov, desync \(t_0-d\)). Catalog + `05-notes.md` updated.

### 2026-08-19 — notation.md

- User: complete notation; guessed “α is \(r_{\mathrm{in}}\)” and “8-bit ⇒ 256 values”.
- Correction: α = \(r_m\) (`masks[18]`); \(r_{\mathrm{in}}\) is an additive S-box-input β. α never 0 on 20k v2 profiling traces. β = \(r_{\mathrm{out}}\) (`masks[17]`), 0…255.
- File: `docs/notes/notation.md`. T-014 remains active.

### 2026-08-19 — create `docs/notes/`

- User asked for a multi-file personal + agent reference location; implemented the layout agreed in chat.
- Files moved (untracked at repo root): `datasets.md`, `metrics.md`, `models.md`.
- T-014 left as INDEX active focus.

## Next actions

| Action | I | U | S | Notes |
|--------|---|---|---|-------|
| Grow catalog only when a topic is stable | L | L | H | Next candidate: v1 full mask-byte map |

### Priority snapshot

- **Do now:** none (this task is done)
- **Before next meeting:** —
- **Deep work (park):** grow the catalog only when a topic is stable

## Related

- Task ids: T-001 (datasets), T-005 (metrics), T-012 (models list), T-013 / T-014 (α,β,C vs r,C)
- `docs/lab-log/todo.md`: done row 2026-08-19
