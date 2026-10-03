# T-006: CNN and MLP baselines on ASCAD fixed-key datasets

## Status

`paused` — baselines + rank JSON done; the side-by-side table vs TimesNet/GNN is tracked in T-005 (2026-10-04 cleanup)

## Goal

Train and evaluate the classic ASCAD **cnn_best** and **mlp_best** PyTorch baselines on ATMEGA fixed-key desync0/50/100, with GPU-friendly scheduling, so newer models (TimesNet/GNN) have trustworthy reference numbers.

## Context

- Models defined/trained via `ASCAD_train_models.py` (`CNNBest`, MLP configs).
- Historical TF weights may still exist as `.h5` alongside new `.pth`.
- Launcher: `run_cnn_mlp_best_with_delay.sh` (GPU util threshold + optional PID wait).
- Default training request from session: **75 epochs** for both; CNN batch 200; MLP node=200, 6 layers, batch 100.
- Output dir: `ATMEGA_AES_v1/ATM_AES_v1_fixed_key/ASCAD_data/ASCAD_trained_models/`.

## Plan

- [x] Confirm PyTorch cnn/mlp train+test paths work post-migration (T-002)
- [x] Add delay-aware runner mirroring GNN script patterns
- [x] Produce `.pth` for cnn_best and mlp_best on desync0/50/100 (75e naming for MLP)
- [x] Compare complexity of `cnn_best` vs TimesNet (params vs MACs)
- [x] Ensure rank curves / summary metrics for these baselines are in the same JSON schema as multi-model benchmark (or explicitly tabulated)
- [ ] Multi-seed baseline stats if paper needs variance
- [ ] Side-by-side rank comparison vs `timesnet_ascad_head` winner (T-004/T-005)

## Decisions

- 2026-07-07: Use **shared GPU etiquette** — wait until `nvidia-smi` util &lt; 15% (configurable) before starting, same as GNN runner.
- Match classic hyperparams where possible; MLP epochs set to **75** in the new runner (older reference `.h5` names may still say epochs200).
- Complexity takeaway (2026-07-15): `cnn_best` ~**66.7M** params but only ~**0.41G** MACs/sample — opposite of TimesNet (~8–27M params, ~**8.2G** MACs).

## Working notes / session log

### 2026-07-17 — baseline rank metrics written

Eval-only on Jul-7 `.pth` (seed labeled 0; single-run weights). Protocol matches TimesNet/GNN benchmarks: `eval_traces=10000`, `target_byte=2`, `rank_step=10`, CUDA.

**Artifacts**

- Summary: `ATMEGA_AES_v1/ATM_AES_v1_fixed_key/ASCAD_data/ASCAD_trained_models/benchmark_runs_cnn_mlp_baseline/summary_fixedkey_cnn_mlp_baseline.json`
- Aggregate: same dir `aggregate_fixedkey_cnn_mlp_baseline.json`
- Symlinks map benchmark names → existing `cnn_best_*` / `mlp_best_*` checkpoints
- Code: `mlp` profile + `predict_probs_mlp` in `benchmark_full_fixedkey_gpu.py`

| model | desync | rank_final | traces_to_rank0 |
|-------|--------|------------|-----------------|
| cnn | desync0 | **0** | **190** |
| cnn | desync50 | 64 | null |
| cnn | desync100 | 2 | null |
| mlp | desync0 | 183 | null |
| mlp | desync50 | 226 | null |
| mlp | desync100 | 212 | null |

Notes: MLP (75e) is weak vs CNN on fixed-key; CNN desync50 worse than desync100 on this single seed — treat as checkpoint-specific, not multi-seed mean.

### 2026-07-17 — eval inventory (metrics were missing; now fixed above)

- All 6 Jul-7 `.pth` present; prior `ASCAD_test_models.py` runs used `num_traces=2000` and did not persist rank JSON.

### 2026-07-07 — runner script (session `019f3c93-824a-7022-89dc-51aeff612bf3`, `grok-build`)

User request: script to run/test cnn best and mlp best (75 epochs) with delay when GPU util &lt; 15%.

Delivered `run_cnn_mlp_best_with_delay.sh`:

- Optional `PID=…`, `GPU_THRESHOLD`, `CHECK_INTERVAL`, `EVAL_ONLY`, `FORCE`
- Loops desync0/50/100 datasets
- Trains via `ASCAD_train_models.py`, tests via `ASCAD_test_models.py`
- Uses `py10` interpreter resolution consistent with other launchers

### 2026-07-17 — disk check during task recovery

Present under `ASCAD_trained_models/`:

- `cnn_best_ascad_desync{0,50,100}_epochs75_classes256_batchsize200.pth` (Jul 7)
- `mlp_best_ascad_desync{0,50,100}_node200_layernb6_epochs75_classes256_batchsize100.pth` (Jul 7)
- Older `.h5` reference cnn/mlp files still present (do not delete casually)

### 2026-07-15 — complexity vs TimesNet (session `019f64f3-b502…`)

- CNN size dominated by classifier FC after pool to length 21 (`10752→4096` alone ~44M).
- Fast per-sample compute relative to TimesNet despite more parameters.

## Next actions

1. Side-by-side rank comparison vs `timesnet_ascad_head` (T-004) / GNN (T-003) using the new summary JSON.
2. Optional: multi-seed retrain for variance (current baselines are single-run / no seed in filename).

## Related

- Sessions: `019f3c93-824a-7022-89dc-51aeff612bf3`, complexity discussion in `019f64f3-b502-7e52-8915-54d3fca95536`
- Paths: `run_cnn_mlp_best_with_delay.sh`, `ASCAD_train_models.py`, `ASCAD_test_models.py`
- Tasks: T-002 (done foundation), T-004, T-005
