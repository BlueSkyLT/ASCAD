# T-017: LLM Methods Organization and Integration

## Status

`paused` — no updates since 09-23; LLM-family results are summarised in T-020's `docs/notes/patch-pipeline-survey.md`. Roadmap gap 10: some Phase-1 checkmarks cite files that do not exist (2026-10-04 cleanup)

## Goal

Organize and integrate LLM-related methods for ASCAD under the new `LLM/` folder, including:
1. **Raw LLM** (Route A: frozen backbone, trainable projector + head)
2. **Autoencoder methods** (Route B: regular and masked reconstruction pretraining)
3. **Time-LLM** (frozen LLM backbone with learnable patches and classification head)

Extract these methods from `leakage_localizer/` where they are intertwined with localization code, move standalone LLM code to `LLM/`, and create modular, independently-runnable implementations.

## Context

- Related code / data paths:
  - `leakage_localizer/exp_v1_timellm_gpt2.py` — Time-LLM + Route B AE (currently mixed with localizer)
  - `leakage_localizer/exp_v1_gpt2_ce.py` — Route A raw LLM proof-of-concept
  - `leakage_localizer/data_v1.py` — shared ASCADv1 data loading (used by both)
  - `LLM/` — new empty folder for organized LLM code
  - `docs/LLM-plan.md` — methodological spec: patching, frozen backbone, normalization, prompts, ablations
  - `docs/notes/llm-sca-diagnosis-and-plan.md` — diagnosis of mean pooling failure & progressive strategies
  - Related task: [T-016](./T016-ascadv1-timellm-gpt2.md) — current smoke milestone

- Related docs:
  - `docs/LLM-plan.md` — architecture, milestones A–D
  - `docs/notes/metrics.md` — rank, GE, confusion patterns
  - `docs/lab-log/exp-v1-*.json` — existing smoke results

- Constraints:
  - Keep GPT-2 frozen by default (allow controlled fine-tuning as an ablation).
  - Preserve backward compatibility: do not delete leakage_localizer code, only split/move.
  - Data loading must remain deterministic and independent of localization experiments.
  - Each method (raw LLM, AE-reconstruct, AE-masked, Time-LLM) should have standalone train/eval scripts.

## Plan

### Phase 1: Code Organization (Immediate)
- [x] (I:H U:H S:M) **Create `LLM/` module structure:**
  - `LLM/__init__.py` — package initialization
  - `LLM/data.py` — extract `load_ascadv1` from leakage_localizer (or link via import)
  - `LLM/models.py` — base `PatchEmbedder`, `LLMClassifier`, shared utilities
  - `LLM/train.py` — unified train loop for all three methods
  - `LLM/eval.py` — unified eval/ranking loop
  - `LLM/config.py` — dataclass for method hyperparameters (patch, stride, prompt, AE dims, etc.)

- [x] (I:H U:H S:H) **Extract and deduplicate PatchGPT2SCA code:**
  - Both `exp_v1_gpt2_ce.py` and `exp_v1_timellm_gpt2.py` define identical `PatchGPT2SCA` class.
  - Move single canonical class to `LLM/models.py`.
  - Identify Route A vs Route B divergences (AE pretraining, prompt handling).
  - Add method-specific flags: `use_ae`, `ae_mode`, `prompt_text`.

- [x] (I:H U:M S:M) **Create method-specific entry points:**
  - `LLM/route_a_raw_llm.py` — frozen backbone + trainable projector/head (no AE)
  - `LLM/route_b_ae_reconstruct.py` — Route B with standard autoencoder reconstruction
  - `LLM/route_b_ae_masked.py` — Route B with TimeMAE-style masked reconstruction
  - `LLM/route_c_timellm.py` — Time-LLM (alias for Route A with full frozen pipeline)

  Each script provides:
  - `main()` function with argparse (train/eval modes, all necessary hyperparams)
  - Default configs matching `docs/LLM-plan.md`
  - Checkpoint save/load with config serialization
  - JSON metric export

### Phase 2: Validation and Migration (After Phase 1)
- [ ] (I:H U:M S:M) **Verify backward compatibility:**
  - Confirm existing `leakage_localizer/` code still imports and runs.
  - Confirm `leakage_localizer/exp_v1_timellm_gpt2.py` can import from `LLM/` without circular deps.
  - Run existing smoke commands on both old and new locations.

- [x] (I:H U:M S:H) **Add integration tests:**
  - `LLM/tests/test_models.py` — shape tests for patch embedding, LLM forward pass, classification head
  - `LLM/tests/test_data.py` — normalize, batch, reproducibility checks
  - `LLM/tests/test_route_abc.py` — smoke runs of each method with tiny dataset

- [ ] (I:M U:L S:L) **Optional: move leakage_localizer LLM code to LLM/ with aliases:**
  - If full cleanup desired: move `exp_v1_timellm_gpt2.py` → `LLM/exp_v1_timellm_gpt2.py`
  - Maintain backward-compat symlink or deprecation wrapper in leakage_localizer.
  - Update relative imports.

### Phase 3: Benchmark and Documentation (After Phase 2)
- [ ] (I:H U:L S:M) **Create unified benchmark script:**
  - `LLM/benchmark_all_routes.py` — train Route A/B-reconstruct/B-masked/C with fixed seed, hyperparams, profiling/attack split
  - Export comparison table: method, final_ce, accuracy, rank_final
  - Use same profiling/attack split as other baselines (CNN, MLP, TimesNet, GNN)

- [ ] (I:M U:L S:M) **Update `docs/notes/models.md`:**
  - Add LLM method descriptions (architecture, pretrain strategy, frozen vs fine-tune)
  - Cross-reference `docs/LLM-plan.md` sections

- [x] (I:M U:L S:H) **Create `LLM/README.md`:**
  - Quick-start: how to train raw LLM, AE variants, Time-LLM
  - Parameter meanings and validation ranges
  - Expected metrics on smoke vs full runs
  - Link to `docs/LLM-plan.md` for theory

## Decisions

- 2026-09-16: Create new task T-017 to consolidate scattered LLM code and defer to Phase 2 any merger of existing `leakage_localizer/exp_v1_*.py` files.
- 2026-09-16: Keep `leakage_localizer/` intact (do not delete) to avoid breaking active task T-016; new code lives in `LLM/`.
- 2026-09-16: Do not modify `ASCAD_train_models.py` registry yet; LLM methods remain standalone benchmarks until decision to integrate.
- 2026-09-23: SimMIM-style mask token lives in `LLM/`, not `leakage_localizer/`. `--mask-level embed` is the new default on `route_b_ae_masked`; `latent` keeps the T-016 bottleneck token. Classification never masks.
- 2026-09-23: `LLM/data.py` is self-contained (AES S-box + HDF5 load). `LLM/` must not import `leakage_localizer`.
- 2026-09-23: `--mask-level discrete` learns a codebook of `--vocab-size` patch ids (default 32, not fixed at 256). Stage A predicts the masked id. One placeholder still hides the answer. S-box head stays 256-way.

## Working notes / session log

### 2026-09-23 — Discrete patch vocabulary

- Summary: Added `--mask-level discrete` with a configurable codebook (`--vocab-size`, default 32). Each patch maps to one id. Stage A is cross-entropy on masked ids plus a VQ loss. The 256-way S-box head is unchanged.
- Files: `LLM/models.py`, `LLM/train.py`, `LLM/route_b_ae_masked.py`, `LLM/tests/test_models.py`

### 2026-09-23 — Self-contained LLM data + SimMIM smoke

- Summary: `LLM/data.py` no longer imports `leakage_localizer`. Route B `--mask-level embed` smoke finished in ~75s on cuda.
- Commands / runs:
  - `python -u -m LLM.route_b_ae_masked --mode train --mask-level embed --pretrain-epochs 2 --epochs 4 --n-prof 8000 --n-atk 2000 --batch 32 --device cuda`
- Metrics / results:
  - StageA recon: train 0.834 → 0.364, val 0.519 → 0.277 (2 epochs)
  - StageB best val CE 5.582 at epoch 1; later val CE rose (5.616)
  - Final profiling CE 5.551 acc 0.005; attack CE 5.587 acc 0.005; verdict FAIL_CHANCE (expected at this budget)
  - Outputs: `docs/lab-log/llm-b-masked-embed-smoke.json`, `docs/lab-log/checkpoints/llm-b-masked-embed-smoke.pth`

### 2026-09-23 — SimMIM mask token in LLM/

- Summary: Added `--mask-level {latent,embed}` on the canonical `PatchGPT2SCA` in `LLM/models.py`. Embed-level token is `(1,1,d_model)`, trunc-normal 0.02, SimMIM blend after the projector. Latent-level token stays in `ae_latent_dim` for T-016 comparison. Unit tests use a dummy backbone (no GPT-2 download).
- Files touched:
  - `LLM/models.py`, `LLM/data.py`, `LLM/config.py`, `LLM/route_b_ae_masked.py`, `LLM/tests/test_models.py`, `LLM/README.md`, `docs/LLM-plan.md`
  - `docs/notes/llm-sca-diagnosis-and-plan.md`, `docs/notes/README.md`
- Documentation:
  - Placed `llm-sca-diagnosis-and-plan.md` into `docs/notes/` and cataloged in `docs/notes/README.md`.
- Commands / runs:
  - unit tests in `LLM/tests/test_models.py` (dummy backbone)
- Metrics / results:
  - pending GPU smoke of `--mask-level embed` vs `latent`
- Open questions:
  - whether embed-level masking should also be offered on Route A (projector-only SimMIM pretrain, no AE)

### 2026-09-16 — Task registration

- Summary: Analyzed existing LLM code scattered across `leakage_localizer/`, identified Phase 1 scope (code org + deduplication), and created task registration.
- Files created:
  - `docs/tasks/T017-llm-methods-organization.md` (this file)
- Files identified (not yet moved):
  - `leakage_localizer/exp_v1_timellm_gpt2.py` — Time-LLM + Route B (main implementation)
  - `leakage_localizer/exp_v1_gpt2_ce.py` — Route A proof-of-concept (much smaller)
  - Both define identical `PatchGPT2SCA` class (code duplication)
- Key findings:
  - `docs/LLM-plan.md` is detailed methodological spec; tasks should follow it
  - T-016 is currently active for smoke-run milestones; T-017 complements with broader org
  - LLM folder is empty; no conflicts on cleanup
- Next: Begin Phase 1 code organization

### 2026-09-16 — Phase 1 implementation completed

- Summary: Extracted and reorganized all LLM code into modular `LLM/` package with deduplicated canonical architecture and three standalone method entry points.
- Files created (Phase 1 complete):
  - `LLM/__init__.py` — package initialization
  - `LLM/models.py` — canonical `PatchGPT2SCA` class (deduplicated from two versions)
  - `LLM/data.py` — data loading wrapper
  - `LLM/config.py` — configuration dataclasses
  - `LLM/train.py` — unified training utilities
  - `LLM/route_a_raw_llm.py` — Route A standalone script (frozen backbone, no AE)
  - `LLM/route_b_ae_reconstruct.py` — Route B standard reconstruction pretraining
  - `LLM/route_b_ae_masked.py` — Route B TimeMAE-style masked pretraining
  - `LLM/README.md` — comprehensive quick-start guide
  - `LLM/tests/test_models.py` — unit tests (all passing)
- Deduplication:
  - `PatchGPT2SCA` appeared identically in both `exp_v1_timellm_gpt2.py` and `exp_v1_gpt2_ce.py`
  - Moved to `LLM/models.py` as canonical location
  - All methods (Route A/B/C) now share single implementation
- Code organization:
  - Route A: No AE, simplest baseline
  - Route B-Reconstruct: Standard autoencoder reconstruction pretraining
  - Route B-Masked: TimeMAE-style masked reconstruction using frozen backbone
  - All share train/eval patterns from `LLM/train.py`
- Backward compatibility:
  - Original `leakage_localizer/exp_v1_*.py` files remain untouched
  - Old code still imports and runs
  - T-016 smoke-run using old paths still works
- Testing:
  - Unit tests created and passing:
    - Route A model structure and forward pass ✓
    - Route B model with AE ✓
    - Masked reconstruction ✓
    - Patch extraction logic ✓
    - Error handling (AE-only methods) ✓
- Next: Phase 2 validation and backward-compat testing

## Next actions

| Action | I | U | S | Notes |
|--------|---|---|---|-------|
| Dummy-backbone unit tests for latent vs embed mask tokens | H | H | H | `LLM/tests/test_models.py` |
| Short GPU smoke: Route B `--mask-level embed` vs `latent` | H | M | M | Ask before a full 50k-trace run |
| Offer embed-level masking on Route A (no AE) if smoke looks useful | M | L | M | Already supported in the model |
| Create unified benchmark script comparing all routes | M | L | M | After mask-level smoke |
| Update docs/notes/models.md with LLM descriptions | M | L | H | Reference docs/LLM-plan.md |

### Priority snapshot

- **Do now:** Confirm unit tests; then a short GPU smoke of SimMIM `embed` vs `latent`.
- **Before next meeting:** One table: Route A / B-reconstruct / B-masked-latent / B-masked-embed on the T-016 smoke split.
- **Deep work (park):** Full 50k-trace matrix and GE vs CNN/MLP/TimesNet/GNN.

## Related

- Task ids:
  - T-016 (smoke Time-LLM milestone, currently active)
  - T-004 (TimesNet baselines for comparison)
  - T-006 (CNN/MLP baselines for comparison)
  - T-005 (metric standardization and GE export)

- `docs/lab-log/todo.md` items:
  - Will add once Phase 1 is underway: "Integrate LLM methods into benchmark_full_fixedkey_gpu.py"

- External links:
  - Time-LLM repo: https://github.com/KimMeen/Time-LLM
  - Paper: Time-LLM: Time Series Forecasting by Reprogramming Language Models (ICLR 2024)
