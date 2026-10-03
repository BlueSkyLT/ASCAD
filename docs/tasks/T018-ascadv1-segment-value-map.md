# T-018: ASCADv1 segment value map

## Status

`active`

## Goal

Say, from the ground truth ASCAD actually stores, which stretches of an ASCADv1 trace carry which byte — and which stretches cannot be named. Deliverable: `visualize/segment_value_map.ipynb`.

## Context

- Related code / data paths: `visualize/segment_value_map.ipynb`, `visualize/segment_label_groups.ipynb`, fixed-key `ASCAD.h5` (700-sample crop) and `ATMega8515_raw_traces.h5`
- Related notes: `docs/notes/datasets.md`, `docs/notes/notation.md`, `docs/notes/v1-poi-faq.md`
- Constraints: read-only on HDF5; no new training run

## Plan

- [x] (I:H U:L S:H) Measure which computable bytes leak in the 700-sample crop, and which labels are the same partition on a fixed key
- [x] (I:H U:L S:H) Notebook: one trace, spans labeled with that trace's byte, plus a long-trace mask pairing
- [ ] (I:M U:L S:L) Optional later: match each span to an AVR opcode in secAES-ATmega8515. The HDF5 does not contain the program counter

## Decisions

- 2026-09-23: Label the **value**, not an opcode. `LD(α=…)` is the wrong vocabulary for ASCADv1 (no multiplicative mask, no instruction trace).
- 2026-09-23: A sample is labeled only when identity SNR ≥ 0.25 and ≥ 1.5× the runner-up, then spans shorter than 8 samples are dropped. Noise floor of a non-leaking byte on 8,000 crop traces is ~0.10.
- 2026-09-23: Fixed-key `S`, `p[2]`, and `p[2]⊕k[2]` are one partition (measured max |ΔSNR| ~ 1e-16). They are a single catalog entry.

## Working notes / session log

### 2026-09-23 — value map

- Summary: With profiling masks we can name the byte on 133/700 samples of the published crop (5 spans on trace 42). The other waveform wiggles have no winning byte. Opcodes are not in the file.
- Crop spans (8,000 profiling traces, trace 42): `m0=F1` at [97,106) SNR 0.34 and [148,186) SNR 1.33 (plain HW); `(p⊕k)⊕m14=13` at [252,291) SNR 7.90 (plain HW fails, HW(value⊕0x69) corr −0.70); `S2⊕m0=A0` at [513,550) SNR 6.59 (HW corr 0.84); `m14=63` at [617,627) SNR 0.44. Unmasked S stays under the floor.
- Waveform segments from `find_segments`: 28 on trace 42, 8 overlap a valued span.
- Long trace (1,000 traces, stride 20): for bytes 2–15, `S⊕m[b-2]` beats every other column (gap usually >10×). Byte 1 matches nothing. Byte 0's best is a weak `m15`. Strongest time of a pair is often outside the crop (later reuse).
- Files touched: `visualize/segment_value_map.ipynb`, `visualize/v1_trace42_value_map.png`, `visualize/v1_raw_mask_snr.png`, `visualize/v1_raw_sbox_mask_pairing.png`, `docs/notes/datasets.md`
- Commands / runs: read-only HDF5 SNR, no training
- Open questions: which paper name (`r_in`, state mask, `r_out`) each column is; what constant `0x69` is in the masked-input HW model

### 2026-09-24 — patch-name census (brief B)

- Summary: Cut the 700-sample crop into the 87 patches `LLM/patch_transformer_sca.py` actually builds (len 16, stride 8, `1+ceil((700-16)/8)`, trace padded to 704). With "≥ 8 of 16 samples inside a span", **16 patches carry a name, 71 do not** (18.4% / 81.6%) against 19.0% of samples.
- Named patches: p12 (`m0`, weak span [97,106) only 9 samples wide), p18–p22 (`m0`), p31–p35 (masked input), p64–p67 (masked output), p77 (`m14`). Ten near-misses at 1–7 samples (p11, 13, 17, 23, 30, 36, 63, 68, 76, 78).
- Longest unnamed stretch = 28 consecutive patches, samples [288,520) — the middle of the crop where every SNR curve sits under the 0.25 floor while the waveform looks equally busy.
- Checked, not replaced: `find_segments(trace 42, threshold=10)` reproduces T-018's **28 segments / 8 overlapping a valued span** exactly. Same 28/8 raw, mean-centred or z-scored; a 10-point detrend changes it to 40/12, so state the preprocessing whenever this is quoted. Crop traces are already signed (−62…+37).
- Segments ≠ patches: 8 of 28 waveform humps have a name, but the trainable grid is the 87 patches; only the patch census is usable as a mask-token target.
- Files touched: `docs/2026-09-23/out/B-segment-meanings.md` (deliverable), `out/B_patch_overlap.py`, `out/B-patch-table.json`. Notebook and HDF5 untouched, CPU only, no training.
- Naming guard restated for anyone reusing this: these five values are v1 Boolean shares / masked bytes. None is \(\alpha\).

### 2026-09-24 — names rebuilt from ground truth + first v2 probe

- Summary: `docs/2026-09-23/out/B-patch-names-explained.ipynb` recomputes the value map from stored GT (49 candidates: 16 masks, `S2`, 16 masked outputs, 16 masked inputs) instead of copying it. Reproduces T-018 **exactly** — 5 spans, 133/700 = 19.0%, 16/87 patches, peaks 0.34/1.33/7.90/6.59/0.44. Unmasked `S2` peak SNR 0.042.
- **v2 is now partly answered:** same recipe on 3 stored labels (2,000 traces, desync 0, 15,000 samples) → `alpha_mask` peak 2.66 @ 867 (1,103 samples ≥ 0.25), `beta_mask` 1.72 @ 341, `sbox_masked[2]` 0.26 (1 sample). All four v1 rules → **47 spans, 1,598/15,000 = 10.7%**, and a **shuffled-label control named 0 samples**, so it is signal, not the ~8-traces-per-class artefact. α is the best-naming v2 field; the masked S-box names essentially nothing — the same asymmetry brief A tested on GPU.
- Caveat to repeat if quoted: 3 candidates vs 49, 2,000 traces vs 8,000, floor 0.25 is v1-calibrated. **Not a v2 value map.** A real one needs the full derived candidate list from `masks[38]`/`plaintext`/`key`.
- Caught in-session: a hand-written count-weighted SNR gave 11 spans / 55% (3× overstatement) and only failed because a winner came out `S2^m15` vs the recorded `S2⊕m0`. Canonical definition is `var(class means)/mean(within-class var)` — copy it, do not rewrite it.
- Files: `out/B-patch-names-explained.ipynb` (nbformat-valid, 0 errors), `out/B-patch-names-meeting.png`, `out/build_B_notebook.py`, `out/B-status.md`. CPU only; GPU never touched.

## Next actions

| Action | I | U | S | Notes |
|--------|---|---|---|-------|
| Read secAES-ATmega8515 v1 listing against these five spans | M | L | L | Only if we need opcode names. Do not invent them from SNR. |
| Keep using the crop map when a figure needs “what leaks where” on byte 2 | H | L | H | Notebook is the source |
| Run the patch-name SimMIM probe (5-way, beat the 81.6% “no name” base rate) | M | L | M | Designed in `docs/2026-09-23/out/B-segment-meanings.md`, not run. Needs GPU time — not before the 14:00 meeting. |

### Priority snapshot

- **Do now:** nothing further for this question; the notebook is the answer
- **Before next meeting:** show trace 42 figure if localization comes up; the patch-name counts are ready to read aloud
- **Deep work (park):** opcode alignment from the assembly listing

## Related

- Task ids: T-001, T-014, T-015
- `docs/lab-log/todo.md` items: T18
- External links: ANSSI secAES-ATmega8515 (boolean masking, v1)
