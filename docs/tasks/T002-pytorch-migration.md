# T-002: Migrate ASCAD training/eval stack from TensorFlow to pure PyTorch

## Status

`done`

## Goal

Run ASCAD model training and key-rank evaluation fully in **PyTorch** (local research path), without depending on TensorFlow/Keras for day-to-day experiments.

## Context

- Upstream ANSSI scripts historically used TensorFlow 2 / Keras; CI workflows still reflect older TF paths.
- Local environment is **`py10`** with PyTorch (+ torch-geometric for GNNs).
- Commit landmark: `1157ed5` (“migrate from tf to full torch”).
- User confirmation (2026-07-17): this task is **DONE**.

## Plan

- [x] Port train path (`ASCAD_train_models.py`) to PyTorch modules / optim / checkpoints (`.pth`)
- [x] Port test / ranking path (`ASCAD_test_models.py`) to load `.pth` and compute ranks
- [x] Keep param files compatible (Python `eval()` dicts — not forced JSON)
- [x] Enable later extensions: GNN, TimesNet, unified GPU benchmark
- [x] Preserve HDF5 datasets read-only
- [ ] (Optional / out of scope unless requested) Retire or dual-run TF CI exclusively — not required for local research

## Decisions

- 2026-06-30: PyTorch is the **local source of truth**; do not reintroduce TF deps into the research path unless explicitly asked.
- Keep classic “best” model naming for CNN/MLP checkpoints; dual presence of old `.h5` reference files and new `.pth` is OK.
- Param files remain `eval()`-based for list comprehensions and existing examples.

## Working notes / session log

### 2026-07-17 — recovery note

- **No dedicated long Grok chat** in `~/.grok/sessions` fully reconstructs the migration turn-by-turn; evidence is primarily **git history + current codebase + copilot instructions**.
- Follow-on Grok sessions all assume PyTorch (`conda run -n py10`, `.pth`, `benchmark_full_fixedkey_gpu.py`).
- Dataset safety rules remain critical: never delete multi-GB `.h5` while cleaning TF-era artifacts.

### Artifacts that depend on the migration

- `ASCAD_train_models.py` / `ASCAD_test_models.py` PyTorch models
- `benchmark_full_fixedkey_gpu.py` multi-model sweep
- `GNN/`, `TimesNet/`, CNN/MLP `.pth` baselines

## Next actions

1. None required for migration completeness.
2. If documenting the project externally, mention PyTorch-local vs TF-upstream CI divergence.

## Related

- Commit: `1157ed5`
- Tasks: T-003, T-004, T-006 all built on this foundation.
- Avoid: installing TensorFlow into `py10` “for compatibility” without a concrete need.
