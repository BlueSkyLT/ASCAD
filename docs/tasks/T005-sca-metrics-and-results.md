# T-005: SCA metrics discussion and experiment result analysis

## Status

`in_progress`

## Goal

Define and document the attack metrics used in this repo, then interpret fixed-key GPU benchmark outputs across models, desync levels, learning rates, and seeds—so model choices are driven by **key recovery**, not training CE alone.

## Context

- Core metrics: **`rank_final`**, **`traces_to_rank0`**, plus **GE** fields (`ge_final`, `ge_at`, `mean_traces_to_rank0`, …) — see `docs/notes/metrics.md`.
- Computation: `benchmark_full_fixedkey_gpu.py` → `rank_metrics` / `ge_metrics` → `ASCAD_test_models.full_ranks` + `guessing_entropy`.
- Result JSONs under  
  `ATMEGA_AES_v1/ATM_AES_v1_fixed_key/ASCAD_data/ASCAD_trained_models/`:
  - `benchmark_runs/`
  - `benchmark_runs_lr1e-2/`
  - `benchmark_runs_lr1e-3/`
  - `benchmark_runs_lr1e-4/`
- Random 256-way CE ≈ **ln(256) ≈ 5.545**.

## Plan

- [x] Explain `rank_final` and `traces_to_rank0` from code
- [x] Write `metrics.md` for durable documentation (now `docs/notes/metrics.md`)
- [x] Analyze multi-folder fixed-key summaries (LR / architecture / desync)
- [x] Cross-link model implications for TimesNet / GNN / baselines
- [x] Implement multi-attack **guessing entropy** in eval path (`guessing_entropy` + benchmark wiring)
- [x] Re-eval all fixed-key summary checkpoints and merge GE into original JSONs (`scripts/reeval_summaries_with_ge.py`)
- [ ] Produce clean apples-to-apples table (fixed epochs, batch, seeds)
- [ ] Optional rank-curve plots export for paper/slides
- [ ] Optional: automated parser script from summary JSON → markdown/CSV

## Decisions

- 2026-07-06: Document metrics in repo root `metrics.md` (not only chat).
- 2026-08-19: Metric definitions live at `docs/notes/metrics.md` (T-015).
- Ranking semantics:
  - **`rank_final`**: rank of correct key byte after full attack budget (lower better; 0 ideal).
  - **`traces_to_rank0`**: first stepped trace count where rank hits 0; **null** if never.
  - Rank can hit 0 early then worsen → both metrics needed.
- 2026-07-18: No pre-existing GE implementation. Added classical multi-order GE:
  - Core: `ASCAD_test_models.guessing_entropy` (permutation trials).
  - Benchmark: `--ge-attacks` (default 100), `--ge-seed`; fields `ge_final`, `ge_at`, `mean_traces_to_rank0`, …
  - **Not** the same as seed-averaged `rank_final`. Endpoint `ge_final` is order-invariant (equals full-set rank); multi-attack value is the curve / MTD distribution.
- 2026-07-15 analysis conclusions (still the working scientific summary):
  1. **Train loss ≠ attack quality** (can overfit labels with terrible rank).
  2. Loss stuck ~5.545 + rank ~238 ⇒ model never learned.
  3. Desync0 ≪ desync100 difficulty in general.
  4. **Best overall so far:** `timesnet_ascad_head`, LR **1e-4**, **150 epochs**.
  5. **Avoid:** high LR (1e-2) for ASCAD head; trust incomplete `lr1e-3` folder blindly; GAT/GCN as currently configured.
  6. **GraphSAGE** is the only consistently interesting GNN in the 1e-4/75e sweep.

## Working notes / session log

### 2026-07-18 — GE implementation + full re-eval

- Confirmed no prior GE code in repo.
- Added `guessing_entropy` (+ helpers) to `ASCAD_test_models.py`; wired into `benchmark_full_fixedkey_gpu.py` eval stage; documented in `docs/notes/metrics.md`.
- Smoke-tested synthetic + real `ASCAD.h5` metadata path.
- Ran `scripts/reeval_summaries_with_ge.py` over all 7 fixed-key `summary*.json` dirs (91 ok rows; 6 non-ok in lr1e-3 skipped).
- Backups: `*.json.pre_ge.bak` next to each summary.
- GNN loaders: strip `torch.compile` `_orig_mod.` prefix; legacy GAT/GIN without residual/LN (`use_residual=False`).

### 2026-07-06 — metric definitions (session `019f37f8-260e-7202-bc73-b36c820da3eb`, `grok-build`)

- User had `summary_full_fixedkey_gpu.json` open; asked what metrics mean.
- Plan approved → wrote `metrics.md` (now `docs/notes/metrics.md`) with SCA background, definitions, edge cases, code pointers.
- Related subagent title: `019f1d54-f0d6…` “Analyzing Rank_final, Rank_metrics, Full_ranks…” (`grok-build` / `grok-build-b`).

### 2026-07-15 — multi-benchmark analysis (session `019f64f3-b502-7e52-8915-54d3fca95536`, `grok-4.5`)

User: analyze all full fixed-key summary JSONs (mainly different LRs) and explain what is happening.

**Folder map:**

| Folder | Intended LR | Contents (high level) |
|--------|-------------|------------------------|
| `benchmark_runs` | 1e-4 (profile default) | TimesNet×2, **150e**, seeds 0–1 — best attacks |
| `benchmark_runs_lr1e-2` | 1e-2 | TimesNet×2, 75e, seed 0 — head mostly dead |
| `benchmark_runs_lr1e-3` | 1e-3 | Incomplete: many eval-only / missing ckpts |
| `benchmark_runs_lr1e-4` | 1e-4 (TimesNet) | TimesNet 75e only after GNN split; **GNN rows → `gnn_runs/`** (actual GNN LR was Adam 1e-3) |
| `gnn_runs` | Adam 1e-3 historical | GNN checkpoints + `summary_gnn.json` / `aggregate_gnn.json` |

**Headline examples from that analysis:**

- TimesNet desync100 @ 1e-2: loss **0.14**, rank **234** (overfit, useless attack).
- TimesNet-ascad-head desync0 @ e150: loss **~5.2**, rank **0** in **~150–180** traces (CE near random, attack excellent).
- GAT/GCN @ 1e-4: stuck random CE and rank.
- GraphSAGE: best GNN ranks with tiny CE margin.

**Data-quality caveats:**

- `lr1e-3` mixes train/eval-only/error rows; some paths point at wrong folder names.
- Folders confound LR with epochs, batch size, seed count, model set.
- Some summaries record checkpoint paths under `benchmark_runs/` even when run dir is `lr1e-*`.

### Complexity cross-link

Same session compared TimesNet vs `cnn_best` FLOPs/params — stored primarily under T-004 / T-006.

## Next actions

1. When new runs finish, append a dated row here (model, desync, LR, epochs, seeds, rank_final mean/std, ttr0).
2. Build a single “fair comparison” matrix before writing papers.
3. Investigate GAT/GCN non-learning with systematic-debugging (ties to T-003).

## Related

- Sessions: `019f37f8-260e…`, `019f1d54-f0d6…`, `019f64f3-b502…`
- Doc: `docs/notes/metrics.md`
- Tasks: T-003, T-004, T-006
