# T-025: ASCADv2 unsupervised variance-bulge idle/active localizer

## Status

`active`

## Goal

Detect active cryptographic leakage cycles versus idle ARM opcode cycles on ASCADv2 with zero ground-truth keys or masks. Use residual variance scanning across traces to generate automated pseudo-labels, evaluate alignment against known leakage points (e.g. sample 867 for $\alpha$), and train a lightweight patch classifier.

## Context

- Related code / data paths: `leakage_localizer/v2_variance_localizer.py`, `STM32_AES_v2/ASCAD_data/ASCAD_databases/ascadv2-extracted.h5`, `leakage_localizer/data_v2.py`
- Related notes: `docs/notes/trace-signal-anatomy.md`, `docs/notes/notation.md`
- Related tasks: T-018, T-020, T-024, T-026
- Constraints: Must not disrupt ongoing GPU background jobs; fast lightweight training (CPU or low VRAM).

## Plan

- [x] (I:H U:H S:H) Step 1: Design variance-bulge scanner on mean-centered ASCADv2 traces ($15,000$ samples, $1,874$ patches with $P=16, S=8$).
- [x] (I:H U:H S:H) Step 2: Implement `leakage_localizer/v2_variance_localizer.py` with automatic pseudo-labeling (`IDLE` vs `ACTIVE`).
- [x] (I:H U:H S:M) Step 3: Train patch classifier on pseudo-labels and validate whether detected active patches coincide with known ground-truth POIs (e.g. $\alpha$ peak at sample 867).

## Decisions

- 2026-09-24: Use ensemble mean trace $\bar{X}(t)$ subtraction before computing variance, isolating pure data-dependent fluctuations from the 159-unit opcode wave.
- 2026-09-24: Define threshold dynamically based on median residual variance (noise floor $\sigma_{\text{noise}}^2$) plus statistical significance margin ($k \times \text{IQR}$ or standard deviations).

## Working notes / session log

### 2026-09-24 — Planning, implementation & execution

- Summary: Implemented zero-ground-truth variance scanner and patch classifier in `leakage_localizer/v2_variance_localizer.py`.
- Execution on v1 (`ASCAD.h5`, 700 samples, 87 patches):
  - Opcode peak-to-peak: $98.46$ units.
  - Noise floor median variance: $4.953$.
  - Ground truth hits: $m_0$ (p19, var 15.9), masked output (p64, var 16.8), $m_{14}$ (p77, var 18.7).
  - 1D-CNN patch classifier validation accuracy: **$78.8\%$**.
  - Output saved to `docs/lab-log/v1_variance_localizer.json`.
- Execution on v2 (`ascadv2-extracted.h5`, 15,000 samples, 1,874 patches):
  - Opcode peak-to-peak: $158.16$ units.
  - Noise floor median variance: $20.846$.
  - Ground truth hits: $\alpha$ mask at sample 867 (var 184.6) and $\beta$ mask at sample 341 (var 1212.0) both detected active.
  - 1D-CNN patch classifier validation accuracy: **$78.7\%$**.
  - Output saved to `docs/lab-log/v2_variance_localizer.json`.
- Communication: Formatted results and patch comparison figure sent to user via Telegram (`scripts/send_telegram.py`).

### 2026-09-25 — Trace Before vs. After Subtraction Visualizations
- Implemented `scripts/plot_active_patches_before_after.py` generating 2-panel comparison plots:
  - Top panel: Option 1 (Variance Bulge Scanner active patch spans).
  - Bottom panel: Option 2 (Autoencoder Anomaly active patch spans).
  - Traces: Raw trace #0 in dashed line (`--`), mean-centered residual trace in solid line (`-`).
- Output generated and verified:
  - `docs/lab-log/v1_active_patches_before_after.png`: Highlights 34 active patches (Option 1) vs. 11 active patches (Option 2) across 700 samples, cleanly marking $m_0, \text{masked input}, \text{masked output}, m_{14}$.
  - `docs/lab-log/v2_active_patches_before_after.png`: Full 15,000-sample trace with zoomed detail inset for samples $[150:1150]$ showing $\beta$ and $\alpha$ leakage zones.
- Sent high-resolution figures to user via Telegram.

## Next actions

| Action | I | U | S | Notes |
|--------|---|---|---|-------|
| Run standalone verification of variance detector | H | H | H | Done; verified on v1 and v2 |
| Connect active patch mask to downstream POI windowing / transformer attention | H | M | M | Use to filter out 85%+ idle patches |

### Priority snapshot

- **Do now:** Deliver visualization comparison report to user.
- **Before next meeting:** Compare variance detection with autoencoder anomaly detection (T-026).
- **Deep work (park):** Integrate with sparse attention transformer.

## Related

- Task ids: T-020, T-026
- `docs/lab-log/todo.md` items: None
- External links: Test Vector Leakage Assessment (TVLA)
