# T-026: ASCADv2 self-supervised autoencoder idle/active localizer

## Status

`active`

## Goal

Train a self-supervised patch autoencoder on mean-centered ASCADv2 traces to learn low-entropy idle ARM opcode representations. Use patch reconstruction error (MSE anomaly score) to identify active cryptographic leakage cycles without access to secret keys or masks.

## Context

- Related code / data paths: `leakage_localizer/v2_autoencoder_localizer.py`, `STM32_AES_v2/ASCAD_data/ASCAD_databases/ascadv2-extracted.h5`
- Related notes: `docs/notes/trace-signal-anatomy.md`
- Related tasks: T-020, T-025
- Constraints: Self-contained, lightweight architecture (e.g. 1D Conv/MLP autoencoder), fast training.

## Plan

- [x] (I:H U:H S:H) Step 1: Design patch autoencoder for 16-sample patches ($16 \to 4 \to 16$ bottleneck).
- [x] (I:H U:H S:H) Step 2: Implement `leakage_localizer/v2_autoencoder_localizer.py` with unsupervised training on mean-centered residuals.
- [x] (I:H U:H S:M) Step 3: Compute reconstruction error curve across the 1,874 patch positions and correlate top-reconstruction-error patches with known ground-truth POIs.

## Decisions

- 2026-09-24: Subtract mean trace $\bar{X}(t)$ prior to patch autoencoding so the model reconstructs residual variance rather than memorizing the 159-unit clock wave.
- 2026-09-24: Use bottleneck compression to force model to capture the dominant Gaussian idle noise distribution; high-entropy multi-modal leaking patches will naturally exhibit higher reconstruction loss.

## Working notes / session log

### 2026-09-24 — Planning, implementation & execution

- Summary: Implemented self-supervised patch autoencoder in `leakage_localizer/v2_autoencoder_localizer.py`.
- Execution on v1 (`ASCAD.h5`, 700 samples, 87 patches):
  - Median idle reconstruction MSE: $0.716$.
  - Top anomalous patches align with $m_0$ span (p17 MSE 3.35, p18 MSE 2.20), $m_{14}$ span (p76 MSE 2.55), and early $m_0$ (p11 MSE 2.38).
  - Output saved to `docs/lab-log/v1_autoencoder_localizer.json`.
- Execution on v2 (`ascadv2-extracted.h5`, 15,000 samples, 1,874 patches):
  - Median idle reconstruction MSE: $37.23$.
  - Over $93.4\%$ of patches fit the low-dimensional idle subspace cleanly.
  - Detected $\beta$ mask at sample 341 (p42, MSE 94.2) as anomaly; top anomalies at patches 1488 (MSE 177.9), 1851 (MSE 155.1), 125 (MSE 140.1).
  - Output saved to `docs/lab-log/v2_autoencoder_localizer.json`.
- Communication: Formatted results and patch comparison figure sent to user via Telegram (`scripts/send_telegram.py`).

### 2026-09-25 — Trace Before vs. After Subtraction Visualizations
- Implemented `scripts/plot_active_patches_before_after.py` generating 2-panel comparison plots:
  - Top panel: Option 1 (Variance Bulge Scanner active patch spans).
  - Bottom panel: Option 2 (Autoencoder Anomaly active patch spans).
  - Traces: Raw trace #0 in dashed line (`--`), mean-centered residual trace in solid line (`-`).
- Output generated and verified:
  - `docs/lab-log/v1_active_patches_before_after.png`: Autoencoder flags 11 high-anomaly patches (12.6% of patches) sharply concentrated around $m_0$ and $m_{14}$.
  - `docs/lab-log/v2_active_patches_before_after.png`: Autoencoder flags 126 anomaly patches (6.7% of patches), pinpointing $\beta$ leakage at sample 341 and filtering out over 93.3% of idle opcode cycles.
- Sent high-resolution figures to user via Telegram.

## Next actions

| Action | I | U | S | Notes |
|--------|---|---|---|-------|
| Run standalone autoencoder training and test anomaly scoring | H | H | H | Done; verified on v1 and v2 |
| Benchmark anomaly scoring vs variance scanner (T-025) | H | M | M | Autoencoder is 10x more selective on v2 (6.7% vs 90.0% active) |

### Priority snapshot

- **Do now:** Deliver visualization comparison report to user.
- **Before next meeting:** Compare T-025 (variance) and T-026 (autoencoder).
- **Deep work (park):** Explore masked autoencoder (SimMIM) variant.

## Related

- Task ids: T-020, T-025
- `docs/lab-log/todo.md` items: None
- External links: Anomaly Detection via Autoencoders
