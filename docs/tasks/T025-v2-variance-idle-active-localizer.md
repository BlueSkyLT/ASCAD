# T-025: Label-free idle/active localizer (variance scanner + patch autoencoder)

## Status

`done` — both detectors built, run on v1 and v2, and compared in one figure (2026-09-25).
**T-026 (autoencoder) merged into this file on 2026-10-04.** Downstream use as a label-free
α finder was later tested in T-028 Phase 1 and closed NO-GO: α owns only 1.3× the trace
variance of a random 255-way split, so spread-based selection cannot isolate it.

## Goal

Detect active cryptographic leakage cycles versus idle ARM opcode cycles on ASCADv1/v2 with zero
ground-truth keys or masks, two ways:

1. **Variance scanner** (was T-025): residual variance across traces after mean-trace
   subtraction → automatic `IDLE`/`ACTIVE` pseudo-labels → lightweight patch classifier.
2. **Patch autoencoder** (was T-026): self-supervised 16→4→16 bottleneck on mean-centred
   patches; reconstruction MSE as an anomaly score.

## Context

- Code: `leakage_localizer/v2_variance_localizer.py`, `leakage_localizer/v2_autoencoder_localizer.py`,
  `scripts/plot_active_patches_before_after.py`
- Data: `STM32_AES_v2/ASCAD_data/ASCAD_databases/ascadv2-extracted.h5` (15,000 samples,
  1,874 patches at P=16, S=8); ASCADv1 `ASCAD.h5` (700 samples, 87 patches)
- Notes: `docs/notes/trace-signal-anatomy.md`, `docs/notes/notation.md`
- Constraints: CPU or low VRAM; must not disrupt GPU background jobs.

## Plan

- [x] (I:H U:H S:H) Variance-bulge scanner on mean-centred traces with dynamic threshold
- [x] (I:H U:H S:M) Patch classifier on variance pseudo-labels; check hits vs known POIs
- [x] (I:H U:H S:H) Patch autoencoder (16→4→16) on mean-centred residuals
- [x] (I:H U:H S:M) Reconstruction-error curve over all patches; check top anomalies vs POIs
- [x] (I:H U:H S:H) Before/after-subtraction figure comparing both detectors (v1, v2)

## Decisions

- 2026-09-24: Subtract the ensemble mean trace before both methods, so they see data-dependent
  fluctuation rather than the 159-unit opcode/clock wave.
- 2026-09-24: Variance threshold = median residual variance (noise floor) plus a significance
  margin (k × IQR or standard deviations).
- 2026-09-24: Autoencoder bottleneck forces it to model the dominant Gaussian idle noise, so
  multi-modal leaking patches reconstruct worse.
- 2026-10-04: Merged T-026 into this task (same goal, same dates, shared figure); closed as done.

## Working notes / session log

### 2026-10-04 — merged and closed

- T-026 content folded in below; `T026-v2-autoencoder-anomaly-localizer.md` is now a stub.
- Downstream follow-up (label-free α discovery) belongs to T-028, where Phase 1 measured it NO-GO.

### 2026-09-25 — before vs after subtraction figures (both detectors)

- `scripts/plot_active_patches_before_after.py`: top panel = variance scanner spans, bottom =
  autoencoder anomaly spans; raw trace #0 dashed, mean-centred residual solid.
- `docs/lab-log/v1_active_patches_before_after.png`: variance 34 active patches vs autoencoder
  11 (12.6%), both marking \(m_0\), masked input, masked output, \(m_{14}\).
- `docs/lab-log/v2_active_patches_before_after.png`: autoencoder flags 126 patches (6.7%),
  pinpointing β near sample 341; zoom inset on samples [150:1150] for β and α zones.
- **Selectivity on v2:** autoencoder 6.7% active vs variance scanner 90.0% — ~10× more selective.
- Figures sent via Telegram.

### 2026-09-24 — variance scanner (was T-025)

- v1: opcode peak-to-peak 98.46, noise-floor median variance 4.953. Hits \(m_0\) (p19, var 15.9),
  masked output (p64, var 16.8), \(m_{14}\) (p77, var 18.7). Patch classifier val acc **78.8%**.
  `docs/lab-log/v1_variance_localizer.json`.
- v2: opcode peak-to-peak 158.16, noise-floor median variance 20.846. α at sample 867
  (var 184.6) and β at sample 341 (var 1212.0) both flagged active. Patch classifier val acc
  **78.7%**. `docs/lab-log/v2_variance_localizer.json`.

### 2026-09-24 — patch autoencoder (was T-026)

- v1: median idle reconstruction MSE 0.716. Top anomalies on \(m_0\) (p17 MSE 3.35, p18 2.20),
  \(m_{14}\) (p76 2.55), early \(m_0\) (p11 2.38). `docs/lab-log/v1_autoencoder_localizer.json`.
- v2: median idle MSE 37.23; 93.4% of patches fit the idle subspace. β at sample 341 (p42, MSE
  94.2) flagged; top anomalies p1488 (177.9), p1851 (155.1), p125 (140.1).
  `docs/lab-log/v2_autoencoder_localizer.json`.

## Next actions

None required. Backlog only:

| Action | I | U | S | Notes |
|--------|---|---|---|-------|
| Use the active-patch mask to prune input for a patch transformer (T-020) | M | L | M | Filters ~93% idle patches on v2; only if T-020 resumes |
| Masked-autoencoder (SimMIM) variant of the anomaly score | L | L | M | Park |

## Related

- Task ids: T-020 (patch pipeline), T-028 (label-free α discovery, Phase 1 NO-GO), T-018
- Merged from: T-026 (`./T026-v2-autoencoder-anomaly-localizer.md`, now a stub)
- `docs/lab-log/todo.md` items: L1 (done)
