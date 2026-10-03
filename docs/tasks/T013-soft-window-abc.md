# T-013: Soft learnable windows for α, β, C consistency

## Status

`planned`

## Goal

Validate whether a **per-trace proposer** \(h_\psi(X)\) can place three Gaussians on ASCADv2 traces for \(\alpha\), \(\beta\), and \(C=(\alpha\otimes S)\oplus\beta\), with CE + differentiable \(P(r=0\mid S)\). After training, save \((\psi,\phi)\), not a crop.

## Context

- Plan / math: `docs/superpowers/plans/2026-08-17-poi-searcher.md` (file on disk; INDEX previously said `window-searcher.md`)
- v1 leakage localizer (separate task): T-014 / `docs/superpowers/plans/2026-08-20-ascadv1-leakage-localizer.md`
- Data: `STM32_AES_v2/ASCAD_data/ASCAD_databases/ascadv2-extracted.h5` (15k already two POI crops)
- Existing inverse of the same algebra: `ASCAD_test_models.py` `proba_dissect_alpha` / `proba_dissect_beta`
- Labels: `ASCAD_generate.multilabelize` (`alpha_mask` = \(r_m\), `beta_mask` = \(r_{\mathrm{out}}\), `sbox_masked` vs `sbox_masked_with_perm`). \(\alpha\neq r_{\mathrm{in}}\): `docs/notes/notation.md`
- GNN (T-003) set aside for this idea
- Related: T-009 multilabel TimesNet (full-trace heads, no windows)

## Plan

- [x] (I:H U:L S:H) Lock design in `2026-08-17-poi-searcher.md` (per-trace \(h_\psi\); CE required; soft \(P(r=0)\); physical-slot \(C\); 15k crop; \(\alpha\neq 0\))
- [ ] (I:H U:L S:M) S0 synthetic: planted centres **jitter per trace**; \(h_\psi\) must follow
- [ ] (I:H U:L S:L) S1 on extracted 15k, small \(N\), band prior — **ask before GPU**
- [ ] (I:M U:L S:L) S2 \(\lambda\) ablation + GE via existing recombination
- [ ] (I:M U:L S:L) Shuffle / natural-order \(C\) only if S2 shows a window effect

## Decisions

- 2026-08-17: Idea agreed in principle (soft windows + explicit value heads + algebraic loss). Review: pure consistency is underdetermined; use distributional \(P(r=0)\); prototype without shuffle first.
- 2026-08-18: User rejected **fixed** windows after training. \(\theta=h_\psi(X)\) per trace; stored params are \(\psi,\phi\). Global \(\theta\) is only an ablation. Plan §2.1 / §2.4 updated.
- 2026-08-18: Assumption locked: inside one trace, \(\alpha,\beta,C\) at **different** times (three windows + repulsion). Across traces those times may **move** (per-trace proposer). Not a contradiction; it is the design.
- 2026-08-18: \(h_\psi\) I/O locked: input `X (B,T)` → 6 scalars → 3 Gaussians \((\mu,\sigma)\) → masks `(B,3,T)`. Not Transformer heads. Direct 3-mask attention is a later variant.
- 2026-08-18: User asked to update plans. Rewrote `2026-08-17-poi-searcher.md` so CE-optional / start-length / open-\(\mathcal{W}\) leftovers are gone. Aligned T-014 plan to the same per-trace \(h_\psi\).
- 2026-08-18: Collaborator \((\alpha-\bar\alpha)(\beta-\bar\beta)\): ensemble cov of **predicted bytes**, optional \(\mathcal{L}_{\mathrm{ind}}\). Cannot replace CE or GF consistency. Plan §2.4.1 / L10.
- 2026-08-18: User asked to use \((r-\bar r)(C-\bar C)=0\) as the v1 loss. That is the T-014 analogue of \(\mathcal{L}_{\mathrm{ind}}\), not a new GF identity. v1 default \(\lambda_{\mathrm{ind}}=1\); still not allowed as the sole term.

## Working notes / session log

### 2026-08-18 — plans rewritten

- v2 plan is now one locked document (pipeline, \(h_\psi\) I/O, CE+\(\lambda\), moving-centre S0).
- v1 plan (T-014) no longer uses global `nn.Parameter` windows as the method.

### 2026-08-18 — different locations is the assumption, not a blocker

- User asked whether “on each trace α, β, C sit at different places” is a big issue.
- Answer: different places *inside* a trace = why we have three windows. Places *changing across* traces = why \(\theta(X)\), and why a frozen crop would fail.
- Chip nuance: desync often shifts all three together; shuffle mainly moves \(C\).

### 2026-08-18 — per-trace windows (not fixed POI)

- User: after train we must **not** get one window location reused on future traces.
- Current math had global \(\theta\in(0,1)^6\). That *would* freeze. Locked: proposer \(h_\psi(X)\).
- Side effect: this is also how a \(C\) window can follow shuffle/desync.

### 2026-08-18 — user clarifications

- “Shuffle missing” ≠ window order. Windows may be in any order. Issue is only: a **global** \(C\) window vs per-trace S-box slot \(\pi\).
- \(r\) is the plan’s residual \(C\oplus(\alpha\otimes S)\oplus\beta\). \(256^2\) zeros = valid triples for one fixed \(S\).
- Recommended loss is **3×CE + \(\lambda\) consistency**, not CE alone and not consistency alone.
- \((\mu,\sigma)\) is only a nicer parameterization of the same soft window (center, width).

### 2026-08-18 — v1 split off

- ASCADv1 adaptation is **T-014**, not a rename of this task. Affine \(\alpha,\beta,C\) stays v2-only.

### 2026-08-17 — plan review

- Summary: User filled `docs/superpowers/plans/2026-08-17-window-searcher.md`. Reviewed implementability; appended §7 (locks + tricks + S0–S3 ladder).
- Files touched: plan file; this task; INDEX; `docs/lab-log/todo.md`
- Open questions: concrete \(\mathcal{W}\); shared trunk or not; whether CE on \(C\) is dropped in a later variant

## Next actions

| Action | I | U | S | Notes |
|--------|---|---|---|-------|
| S0 synthetic with **moving** planted centres | H | L | M | No HDF5; no long GPU |
| S1 real 15k smoke | H | L | L | Ask before run |

### Priority snapshot

- **Do now:** S0 when T-014 is not occupying the same hands
- **Before next meeting:** S0 result (windows move on synthetic) if they want a demo
- **Deep work (park):** S1–S3

## Related

- Task ids: T-009, T-011 (different I/O; do not mix)
- `docs/lab-log/todo.md` item T15
- Plan: `docs/superpowers/plans/2026-08-17-poi-searcher.md`
- v1 adaptation: T-014
