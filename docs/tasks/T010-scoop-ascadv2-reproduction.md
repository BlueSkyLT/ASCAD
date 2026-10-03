# T-010: Scoop ASCADv2 reproduction

## Status

`done` — v2 HPO sweep finished on all 16 bytes; only byte 15 reached GE 0 (at 100,001 traces), not a strict break. Report: `byte_sweep/BYTE_SWEEP_REPORT.md` (2026-10-04 cleanup)

## Goal

Reproduce Scoop paper ASCADv2 results for Finetune MLP and Single-layer MLP, evaluate GE-to-zero behavior at 250k/300k traces, and keep all generated scripts, logs, models, and results under `Scoop/`.

## Context

- Related code / data paths:
  - `Scoop/scoop/*.py`
  - `Scoop/examples/ascadv2_profiling.ipynb`
  - `Scoop/examples/ascadv2_attack.ipynb`
  - `STM32_AES_v2/ASCAD_data/ASCAD_databases/ascadv2-extracted.h5`
- Constraints:
  - Try `py10` first; fallback to repo-recommended `py3.12` (`.venv`/uv) if needed.
  - ASCADv2 shapes/splits must follow paper scheme and be documented in final report.
  - Smoke runs must be deleted after baseline validation.
  - Extra seed reruns only after at least one successful baseline run.

## Plan

- [x] (I:H U:H S:H) Register dedicated task and set active focus in task index
- [ ] (I:H U:H S:M) Implement reproducible Scoop ASCADv2 runner with artifactized outputs under `Scoop/`
- [ ] (I:H U:H S:M) Validate data split/shape and GE pipeline with short smoke runs, then delete smoke artifacts
- [ ] (I:H U:H S:L) Run Single-layer MLP baseline and Finetune MLP baseline with paper-aligned settings
- [ ] (I:H U:H S:L) Compare GE results against 250k/300k targets and iterate with bounded retries if needed
- [ ] (I:H U:M S:M) Run extra seed reruns only if at least one baseline succeeds
- [ ] (I:H U:H S:M) Produce final report with GE curves, alpha/beta/C analysis, and one-trace prediction demonstration

## Decisions

- 2026-07-29: Scope includes both Finetune MLP and Single-layer MLP baselines.
- 2026-07-29: Artifacts must be written under `Scoop/` (scripts/logs/models/results).
- 2026-07-29: Success criterion tracked as GE reaching 0 by 250k/300k trace budgets.

## Working notes / session log

### 2026-07-29 — implementation kickoff

- Summary:
  - Started T-010, collected Scoop notebook settings and metric parameters.
  - Verified both environments are available (`py10` and `.venv` fallback).
- Files touched:
  - `docs/tasks/INDEX.md`
  - `docs/tasks/T010-scoop-ascadv2-reproduction.md`
- Commands / runs:
  - `conda run -n py10 python -V` and package probe succeeded.
  - `.venv` Python 3.12 package probe succeeded.
- Open questions:
  - Confirm final report filename and exact figure style once baseline runs complete.

### 2026-07-29 — runner implemented + dual-mode smoke validated

- Summary:
  - Implemented reproducible ASCADv2 runner and py10-first launcher under `Scoop/experiments/ascadv2_repro/`.
  - Verified both required experiment modes execute end-to-end on GPU:
    - `single_layer_mlp` smoke: completed, artifacts written.
    - `finetune_mlp` smoke: completed, artifacts written.
  - Smoke outputs generated under `Scoop/artifacts/ascadv2_repro/` and include checkpoint, GE arrays, GE curve image, and JSON summary.
- Files touched:
  - `Scoop/experiments/ascadv2_repro/run_repro.py`
  - `Scoop/experiments/ascadv2_repro/run_repro.sh`
- Commands / runs:
  - `Scoop/experiments/ascadv2_repro/run_repro.sh --mode single_layer_mlp --smoke --run-id single_layer_mlp_seed0_smoke`
  - `Scoop/experiments/ascadv2_repro/run_repro.sh --mode finetune_mlp --smoke --run-id finetune_mlp_seed0_smoke`
- Metrics / results:
  - Single-layer smoke summary: `Scoop/artifacts/ascadv2_repro/metrics/single_layer_mlp_seed0_smoke_summary.json`
  - Finetune smoke summary: `Scoop/artifacts/ascadv2_repro/metrics/finetune_mlp_seed0_smoke_summary.json`
- Open questions:
  - None blocking implementation. Proceeding to full baseline runs.

### 2026-07-29 — full baselines launched (in progress)

- Summary:
  - Launched both non-smoke baseline runs in background.
  - Confirmed both processes are alive and writing logs.
- Commands / runs:
  - `nohup Scoop/experiments/ascadv2_repro/run_repro.sh --mode single_layer_mlp --run-id single_layer_mlp_seed0_full > Scoop/artifacts/ascadv2_repro/logs/single_layer_mlp_seed0_full.log 2>&1 &` (PID 1097117)
  - `nohup Scoop/experiments/ascadv2_repro/run_repro.sh --mode finetune_mlp --run-id finetune_mlp_seed0_full > Scoop/artifacts/ascadv2_repro/logs/finetune_mlp_seed0_full.log 2>&1 &` (PID 1097126)
- Metrics / results:
  - Pending completion of full runs.

### 2026-07-29 — 3-hour autonomous watchdog armed

- Summary:
  - Added and launched a 3-hour watchdog script to monitor baseline health and intervene only on failure conditions.
  - Current behavior: sleep 3h, inspect process/log health, do nothing if healthy, auto-restart with safer parameters if errors are detected.
- Files touched:
  - `Scoop/experiments/ascadv2_repro/watchdog_3h.sh`
- Commands / runs:
  - `nohup Scoop/experiments/ascadv2_repro/watchdog_3h.sh > Scoop/artifacts/ascadv2_repro/logs/watchdog_3h_stdout.log 2>&1 &`
- Logs:
  - `Scoop/artifacts/ascadv2_repro/logs/watchdog_3h.log`

### 2026-07-29 — recovered full evaluation from saved checkpoints

- Summary:
  - Diagnosed the missing final report: both full runs crashed during GE evaluation because attack traces had been cached as `float16` while the MLP weights remained `float32`.
  - Patched `Scoop/experiments/ascadv2_repro/run_repro.py` to support evaluation-only reuse of existing checkpoints and to force `float32` inputs during GE evaluation.
  - Recovered full evaluation artifacts without retraining by running the saved full checkpoints through the patched evaluation path.
  - Baseline outcome is below the paper target in both modes:
    - Single-layer MLP never approaches GE 0; best recovered GE is `96.82` at `20,001` traces and final GE is `186.82` at `300,000` traces.
    - Finetune MLP improves substantially but still does not reach GE 0; best recovered GE is `7.55` at `250,001` traces and final GE is `8.64` at `300,000` traces.
- Files touched:
  - `Scoop/experiments/ascadv2_repro/run_repro.py`
- Commands / runs:
  - `Scoop/experiments/ascadv2_repro/run_repro.sh --mode single_layer_mlp --eval-only --run-id single_layer_mlp_seed0_full_eval --checkpoint-path Scoop/artifacts/ascadv2_repro/checkpoints/single_layer_mlp_seed0_full.pt --batch-size 24 --ge-n-tries 10 --ge-step 10000 --ge-n-attack 300000`
  - `Scoop/experiments/ascadv2_repro/run_repro.sh --mode finetune_mlp --eval-only --run-id finetune_mlp_seed0_full_eval --checkpoint-path Scoop/artifacts/ascadv2_repro/checkpoints/finetune_mlp_seed0_full.pt --batch-size 16 --ge-n-tries 10 --ge-step 10000 --ge-n-attack 300000`
- Metrics / results:
  - `Scoop/artifacts/ascadv2_repro/metrics/single_layer_mlp_seed0_full_eval_summary.json`
  - `Scoop/artifacts/ascadv2_repro/metrics/finetune_mlp_seed0_full_eval_summary.json`
  - `Scoop/artifacts/ascadv2_repro/figures/single_layer_mlp_seed0_full_eval_ge_curve.png`
  - `Scoop/artifacts/ascadv2_repro/figures/finetune_mlp_seed0_full_eval_ge_curve.png`
- Notes:
  - The original single-layer training log reached `Epoch 25/25 | Train loss: 7.9999 | Valid loss: 7.9997` before the GE crash.
  - The original finetune log available in `finetune_mlp_seed0_full.log` shows `Epoch 15/25 | Train loss: 8.0044 | Valid loss: 8.1533` before the GE crash; the saved full checkpoint timestamp indicates the watchdog later rewrote that checkpoint during restart.
  - Current recovered GE summaries were run with `ge_n_tries=10` for fast salvage; a stricter paper-aligned rerun should use `20` tries once the next iteration plan is chosen.

### 2026-07-29 — paper-aligned t20 GE rerun and metric cleanup

- Summary:
  - Reran GE in evaluation-only mode with `ge_n_tries=20` for both saved checkpoints.
  - Confirmed that neither checkpoint reaches GE `0` under the paper-aligned retry count.
  - Investigated the abnormal single-layer curve and found two separate issues:
    - a real reporting bug in `Scoop/scoop/metrics.py`, which divided by `n_tries + 1` instead of `n_tries`
    - a reproduction-runner reporting bug in `Scoop/experiments/ascadv2_repro/run_repro.py`, which multiplied GE by `2` in the plot, `ge_last`, and `traces_to_ge0` logic
  - After isolating those bugs, the single-layer instability still remains at the model level: a deterministic prefix-rank check on the saved checkpoint is also highly non-monotonic, improving only very late and ending near rank `18` at `290001` traces.
- Files touched:
  - `Scoop/scoop/metrics.py`
  - `Scoop/experiments/ascadv2_repro/run_repro.py`
- Commands / runs:
  - `Scoop/experiments/ascadv2_repro/run_repro.sh --mode single_layer_mlp --eval-only --run-id single_layer_mlp_seed0_full_eval_t20 --checkpoint-path Scoop/artifacts/ascadv2_repro/checkpoints/single_layer_mlp_seed0_full.pt --batch-size 24 --ge-n-tries 20 --ge-step 10000 --ge-n-attack 300000`
  - `Scoop/experiments/ascadv2_repro/run_repro.sh --mode finetune_mlp --eval-only --run-id finetune_mlp_seed0_full_eval_t20 --checkpoint-path Scoop/artifacts/ascadv2_repro/checkpoints/finetune_mlp_seed0_full.pt --batch-size 16 --ge-n-tries 20 --ge-step 10000 --ge-n-attack 300000`
- Metrics / results:
  - `single_layer_mlp_seed0_full_eval_t20`: reported `ge_last=380.38` in the pre-fix summary path because of the x2 display logic; the saved raw curve corresponds to a corrected final GE of about `199.70` after undoing both the `*2` display and the `n_tries + 1` averaging bug.
  - `finetune_mlp_seed0_full_eval_t20`: reported `ge_last=14.67` in the pre-fix summary path; corrected final GE is about `7.70` after undoing the same two issues.
  - Deterministic single-layer prefix ranks at `n={1,10001,...,290001}` are highly unstable and only reach top-10 very late (`rank 8` at `280001`, `rank 18` at `290001`), so the abnormal behavior is not only a plotting artifact.

### 2026-07-30 — finetune-only byte sweep (other 15 bytes) launched + recovery automation

- Summary:
  - Per user direction, focused on `finetune_mlp` only and launched a 15-byte sweep excluding current target byte `4`, with `n_epochs=25` and `attack_len=300000`.
  - Added a dedicated launcher that tracks per-byte status in TSV and writes a manifest for deterministic post-processing.
  - Added a byte-sweep report generator that classifies bytes as recovered, unsuccessful, pending, or failed-no-summary.
  - Observed byte `0` as `failed_no_summary` after checkpoint save; diagnosed termination during GE stage.
  - Reduced GE evaluation memory pressure by avoiding full-tensor float32 materialization and lowering GE prediction batch step.
  - Added and launched a background finalizer that waits for sweep completion, replays missing-summary bytes in `--eval-only` mode from saved checkpoints, then regenerates the final report.
- Files touched:
  - `Scoop/experiments/ascadv2_repro/run_finetune_other15_bytes.sh`
  - `Scoop/experiments/ascadv2_repro/byte_sweep_report.py`
  - `Scoop/experiments/ascadv2_repro/finalize_finetune_other15.py`
  - `Scoop/experiments/ascadv2_repro/run_repro.py`
  - `Scoop/scoop/metrics.py`
- Commands / runs:
  - `nohup bash Scoop/experiments/ascadv2_repro/run_finetune_other15_bytes.sh --seed 0 > Scoop/artifacts/ascadv2_repro/logs/finetune_other15_master_nohup.log 2>&1 &` (PID 1349671)
  - `nohup conda run -n py10 python Scoop/experiments/ascadv2_repro/finalize_finetune_other15.py --manifest Scoop/artifacts/ascadv2_repro/byte_sweep/finetune_other15_seed0_20260730_170636_manifest.json --status-tsv Scoop/artifacts/ascadv2_repro/byte_sweep/finetune_other15_seed0_20260730_170636_status.tsv --sweep-pid 1349671 --poll-seconds 120 > Scoop/artifacts/ascadv2_repro/logs/finetune_other15_finalizer_nohup.log 2>&1 &`
- Metrics / results (intermediate):
  - Manifest: `Scoop/artifacts/ascadv2_repro/byte_sweep/finetune_other15_seed0_20260730_170636_manifest.json`
  - Status TSV: `Scoop/artifacts/ascadv2_repro/byte_sweep/finetune_other15_seed0_20260730_170636_status.tsv`
  - Live report: `Scoop/artifacts/ascadv2_repro/byte_sweep/BYTE_SWEEP_REPORT_finetune_other15_seed0_20260730_170636.md`
  - Current status: byte `0` = `failed_no_summary`; byte `1` running; remaining bytes pending.

### 2026-07-31 — why we are slower than paper’s “10 min / byte”

- Summary:
  - Paper (TCHES 2025 §6.3.1–6.3.2, RTX 4500 Ada): **fine-tuned MLP trains in ~10 minutes**; the **3 h** figure is **hyperparameter fine-tuning budget**, not one train. Single-layer MLP is **~10 hours**, not 10 min.
  - Our `finetune_other15` byte sweep on **RTX 5090** (typical 25-epoch byte, e.g. b12):
    - **Train ≈ 36–38 min**
    - **Attack/GE ≈ 36 min** (tqdm GE bar ~36:08; model forward for 300k preds only ~9 s)
    - **Total wall ≈ 70–75 min** (`wall_time_sec` in metadata)
  - Outliers: b02 **125 epochs**, b10 **95 epochs** because `finetuning=True` continues past `n_epochs` while `val_loss ≤ entropy` → wall 2–3.7 h.
  - Root causes vs paper 10 min train:
    1. **Architecture mismatch (main train cost):** we use `4 × 3000` (~**73M** params, 278MB ckpt); paper describes a **narrow** MLP with hidden width **2^8 = 256** (~**4M** params) → ~18× more weights.
    2. **GE is ~half wall clock:** `Scoop/scoop/metrics.py` pure-Python nested loops over `n_tries × Σ n_attack` (tries=20, step=10k, 300k) — CPU-bound, not helped by 5090.
    3. `verbose=True` logs every of 6250 iters/epoch to file (~156k lines / ~9MB per byte).
    4. Scoop `loss.backward(create_graph=True)` + Hutchinson every `hessian_update=10` (same algorithm as paper; not the main gap).
- Files / evidence:
  - `papers/Scoop_TCHES2025_3_03.pdf` §6.3 (“~10 minutes to train”, Table 5 GPU Time 3h = HPO)
  - Byte metas under `Scoop/artifacts/ascadv2_repro/logs/finetune_mlp_seed0_b*_…_metadata.json`
  - GE-only baselines: `full_eval` ~18 min (`ge_n_tries=10`), `full_eval_t20` ~38 min (`ge_n_tries=20`)
- Open questions:
  - Recover paper’s actual HPO-chosen `(n_linear, linear_size)` if available; default `3000` is almost certainly wrong for the 10-min claim.
  - Vectorize GE scoring (NumPy gather over 256 key hyps) to cut attack from ~36 min toward seconds–few minutes.

## Next actions

| Action | I | U | S | Notes |
|--------|---|---|---|-------|
| Align finetune width with paper (`linear_size≈256`) or document why 3000 | H | H | H | Likely needed for both speed and GE-to-0 |
| Vectorize GE in `Scoop/scoop/metrics.py` | H | M | M | Attack currently ≈ train time |
| Disable/redirect per-iter verbose when logging to files | M | L | H | Easy wall-time win |
| Publish final byte success/failure report (excluding byte 4) | H | H | H | Sweep finished; none reached GE 0 |
| Clean smoke artifacts once report is stable | H | M | H | User requested cleanup after validation |

### 2026-07-31 — v2 paper-narrow HPO byte sweep launched

- Summary:
  - Labeled prior fixed **4×3000** sweep as **v1** (`Scoop/artifacts/ascadv2_repro`).
  - Implemented **v2** paper-aligned path: width **256**, random HPO budget **2h/byte**, 25 epochs, keep **best-by-GE** checkpoint only, full GE eval + report.
  - Vectorized `Scoop/scoop/metrics.py` GE scoring (was ~36 min pure-Python).
  - Bytes: **1,2,3,4,9,10,11,12**. Artifacts: `Scoop/artifacts/ascadv2_repro_v2/`.
  - Smoke on byte1 OK; full sweep launched via `run_finetune_v2_byte_sweep.sh`.
- Files:
  - `Scoop/experiments/ascadv2_repro/run_finetune_v2_paper.py`
  - `Scoop/experiments/ascadv2_repro/run_finetune_v2_byte_sweep.sh`
  - `Scoop/experiments/ascadv2_repro/byte_sweep_report_v2.py`
  - `Scoop/scoop/metrics.py`
- Commands:
  - `nohup bash Scoop/experiments/ascadv2_repro/run_finetune_v2_byte_sweep.sh --seed 0 --n-epochs 25 --hpo-budget-sec 7200 --bytes 1,2,3,4,9,10,11,12 > Scoop/artifacts/ascadv2_repro_v2/logs/finetune_v2_master_nohup.log 2>&1 &`
- Expected wall: ~8 × (≤2h HPO + final GE) ≈ **half-day to ~1 day** on one GPU.

### 2026-08-03 — complete v2 remaining bytes

- Wave1 done: bytes **1,2,3,4,9,10,11,12** (none GE→0; best e.g. b12 ge_last=3.55).
- Remaining launched: **0,5,6,7,8,13,14,15** via `run_finetune_v2_remaining_and_report.sh` (2h HPO/byte, 25 epochs).
- Full merged report builder: `build_full_v2_report.py` → `BYTE_SWEEP_REPORT.md`.
- Monitor: `Scoop/artifacts/ascadv2_repro_v2/logs/finetune_v2_remaining_nohup.log`

### 2026-08-04 — v2 continue +25 epochs on bytes 0–14

- Full v2 HPO finished 0–15; only **byte 15** reached GE=0 (`traces_to_ge0=100001`).
- Launched continue: reload best_params + checkpoint, train **+25 epochs**, full GE; skip b15.
- Report: glossary for **HPO** / **n_hpo**; single file `byte_sweep/BYTE_SWEEP_REPORT.md`.
- Scripts: `run_finetune_v2_continue.py`, `run_finetune_v2_continue_bytes.sh`.
- Log: `Scoop/artifacts/ascadv2_repro_v2/logs/finetune_v2_continue_nohup.log`

### Priority snapshot

- **Do now:** Monitor continue wave for bytes 0–14; refresh GE vs pre-continue `source_ge_last`.
- **Before next meeting:** Compare v1 vs v2 GE; highlight b15 recovery + continue gains.
- **Deep work (park):** Multi-seed once more bytes reach GE 0.

## Related

- Task ids: T-009
- `docs/lab-log/todo.md` items: T13
- External links: Scoop paper (local PDF), Thales Scoop GitHub
