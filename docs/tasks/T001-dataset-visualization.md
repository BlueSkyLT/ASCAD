# T-001: Visualization and understanding of ASCAD datasets

## Status

`in_progress`

## Goal

Build tooling and intuition for ASCAD power traces (ATMEGA fixed-key + STM32) so model design (CNN/GNN/TimesNet, segmentation, POIs) is grounded in data properties rather than architecture alone.

## Context

- Primary processed ATMEGA fixed-key traces: 700 samples, profiling 50k / attack 10k, 256 S-box classes, target byte 2 by default.
- Desync variants: `ASCAD.h5`, `ASCAD_desync50.h5`, `ASCAD_desync100.h5`.
- STM32 extracted traces much longer (~15k samples) — segmentation stats differ.
- Code: `visualize/` toolkit; segment helpers in `utils/find_segments.py`, `utils/analyze_trace_segments.py`.
- Commit landmark: `01e56eb` (“visualize the dataset”).

## Plan

- [x] Load processed HDF5 (profiling/attack, labels, metadata)
- [x] Plot individual / multi-class traces, means, heatmaps, basic leakage-oriented views
- [x] Compare sync vs desync datasets
- [x] Interactive notebook walkthrough (`explore_traces.ipynb`)
- [x] Segment-based analysis of traces (local-minima heuristic)
- [x] Diagnose short-segment bug in `find_segments` final tail
- [x] Add wavelet-inspired `find_segments_v2` + re-run stats
- [ ] Use segment graphs as GNN nodes (feature-per-segment) end-to-end — **full exploration plan in T-003 Plan B** (deferred)
- [ ] Optional: richer POI / alignment visualizations for paper figures

## Decisions

- 2026-06-30: Keep visualization tooling under `visualize/` with `py10` + matplotlib/jupyter; do not rewrite HDF5 datasets.
- 2026-07-06/07: Treat segment discovery as a path toward **segment-as-node** graphs (vs sample-as-node complete graphs).
- 2026-07-07: Original `find_segments` **always appends a final segment without min-length check** → can report length 2 even when `threshold=10`.
- 2026-07-07: `find_segments_v2` uses smoothing (approx/scaling idea) + prominence-style minima rather than only raw local minima.
- 2026-07-17: Segment-as-node first-cut design frozen in T-003 Plan B (`find_segments_v2`, fixed-N path graph, GraphSAGE, `out_dim=12`); not scheduled yet.

## Working notes / session log

### 2026-09-25 — ran segment notebooks, sent HTML on Telegram

- Executed all cells of `visualize/segment_cross_label.ipynb` and `visualize/segment_detection.ipynb` (py10, cwd `visualize/`, ASCAD.h5 profiling 50k×700). Source notebooks left unchanged.
- Cross-label: top-3 SNR peaks at samples 10 / 50 / 75 (SNR ≈ 0.0059–0.0060). Best window `[50, 100)`, max |Pearson| 0.093.
- Detection (`find_segments_v2`, threshold=10, prominence=5): 28 segments on every one of 100 traces, mean length 24.96, coverage 99.9%. SNR on this POI file is nearly flat (0.0042–0.0061).
- HTML reports sent with `sendDocument` to the configured Telegram chat.

### 2026-07-22 — SNR Q&A in `ascad_notes.md` (now `docs/notes/datasets.md`)

- Documented: SNR needs many traces + known intermediate \(Z\) (labels and/or p/k/masks depending on which curve); not attack-time alone.

### 2026-07-22 — `visualize/snr_poi_xor.ipynb`

- Notebook: multi-trace SNR for \(S\), \(r=\texttt{masks}[:,0]\), \(S\oplus r\); overlay POI segments on a chosen trace; pairwise + canonical XOR matrices showing \((S\oplus r)\oplus r = S\).
- Empirically on fixed-key ASCAD: strong peaks ~`r` near t≈156, `S⊕r` near t≈517; unmasked \(S\) first-order SNR weak.

### 2026-07-22 — SNR section in `ascad_notes.md` (now `docs/notes/datasets.md`)

- Added **SNR** subsection: SCA definition (between/within class variance), formula, NumPy snippet aligned with `visualize/comparative_analysis.py`, other intermediates via masks, note that SNR2/SNR3-style peaks are **not** labelled in any of the three HDF5s.

### 2026-07-17 — finish `ascad_notes.md` (inputs/outputs)

- Completed repo-root **`ascad_notes.md`** (now **`docs/notes/datasets.md`**): cross-campaign comparison table, HDF5 layout, raw vs POI windows, **X/y** for ATMEGA fixed/variable and STM32 multilabel, label formulas verified on local HDF5.
- Corrections: variable-key attack **100k** (not 20k), T=**1400**; variable profiling keys unique, attack key fixed `00..FF`; STM32 T=**15k**, structured multilabel heads.
- Verified v1: `labels == AES_Sbox[p[:,2]^k[:,2]]` on fixed + variable processed files.

### 2026-07-17 — recovery from old Grok sessions + repo state

- **Grok sessions with research content:** mostly `019f38bb-996f…` (segments + datasets). Visualization folder itself predates those chats (git commit), so early explore work may not have full chat logs in `~/.grok/sessions`.
- Toolkit present: `ascad_loader.py`, `trace_visualizer.py`, `comparative_analysis.py`, `explore_traces.ipynb`, `visualize/README.md`.

### 2026-07-06/07 — segment investigation (session `019f38bb-996f-7950-89da-f7df9badc54a`, model `grok-build`)

- Built `utils/analyze_trace_segments.py` (stream HDF5 safely; report avg segments/trace, min/max/avg lengths).
- **ATMEGA desync100** (`threshold=10`, 5000 profiling traces): ~**27.7 segs/trace**, avg len ~**26.2**, min len **2**, max **38**.
- **STM32 ascadv2-extracted** (500 traces): ~**516 segs/trace**, avg len ~**30**, min **11**, max **279**.
- v2 offset/mean issues: `trace[i] < 0` gate fires less without `--center`; centering roughly doubles segment count on v2.
- Explained min-length anomaly; implemented `find_segments_v2` (wavelet-inspired) and re-ran comparable stats.

## Next actions

1. (I:M U:L S:H) When writing: pull 1–2 mean-trace figures (desync0 vs desync100) into `docs/figures/` or cite `docs/notes/datasets.md`.
2. (I:H U:L S:L) When segment-GNN starts: T-003 Plan B0 — freeze v1/v2 segment stats (desync0/50/100 ± centering); paste numbers here.
3. (I:L U:L S:M) Optional: ATMEGA mask-byte map from ANSSI paper if leakage models need it (noted in `docs/notes/datasets.md`).

## Related

- Sessions: `019f38bb-996f-7950-89da-f7df9badc54a` (primary); weak/indirect mentions in other chats via notebook/paths.
- Tasks: T-003 (GNN graph construction), T-005 (how desync affects attack metrics).
- Paths: `docs/notes/datasets.md`, `visualize/`, `utils/find_segments.py`, `utils/analyze_trace_segments.py`.
