# T-027: C3 dataset — standalone break work (GPAM-c3)

## Status

`in_progress` — **compression stage COMPLETE and verified** (2026-09-26 23:02); next is the
full-coverage first-order scan.

### Build result (final)

`c3_project_8bit` = 10 shards × 5,000 traces, 22,754,498,728 B each (227.5 GB total).

| Audit field | Value |
|---|---|
| `elements_checked` | **225,000,100,000** (100% of the cube) |
| `bit_mismatches_total` | **0** |
| `trace_ids_covered` | **50,000** (exactly 0..49999) |
| `level_table_size` | 256 (8.0 bits/level) |
| level range | 4.889977 … 5.071311 V |
| elapsed | 58.2 min; per-shard 356.7 … 616.1 s |

Independent `--phase verify`: structural 50,000 ids / problems **none**; re-read decode
80 blocks / **0** mismatches. `c3_loader.py` self-test: cross-shard reads decode
**bit-exact vs the original source**. Size 900,671,706,384 → 227,544,987,280 B (3.959×).

- 2026-09-26: Confirmed the attack setting (`c3_label_check.py`): `key` is **1 distinct row
  for all 50k traces** (`00..0f`), `opc` is **1 distinct row**, `rand` is **50,000 distinct
  rows**. All 256 classes present in every split — train 119/148/179, test 21/39/57,
  val **1**/7.8/16 per class. So a 256-way head is well-supported on train, but **per-class
  val accuracy is unusable at 4%**; early-stop on aggregate ranking.
- 2026-09-26: Because `key` and `opc` are constant, they cannot be the source of per-trace
  variation — the label must derive from `rand` or `auth`. Only `rand` is aliased to
  `plain_text`/`cipher_text`, so `rand` is the likely block input, but this stays an
  inference until the capturer confirms it.


## Goal

Stand up `GPAM-c3/` as a self-contained project for attacking `c3_project.sx`
(50,000 traces x 4,500,002 samples), determine whether the target is masked at first
order, and train a single-head GPAM-shaped network on it. **Not part of ASCAD research.**

## Context

- Data: `/media/lantian/1.44.1-72806/c3_project.sx` — 838.8 GiB, read-only to us
  (`0664`, owner `daijingyang`), symlinked at `GPAM-c3/c3_project.sx`.
- Actually HDF5. `traces` (50000, 4500002) float32, chunks (49, 4395), uncompressed;
  `metadata/` = key, opc, rand, plain_text, cipher_text, auth, each (50000,16) uint8.
- Code: `GPAM-c3/` (gitignored). Probes: `inspect_c3.py`, `c3_compress_probe.py`,
  `c3_level_check.py`, `c3_roundtrip_check.py`, `c3_drift_check.py`,
  `c3_firstorder_scan.py`.
- Constraints: 62 GB RAM, 240 GB free local NVMe, 1.3 TB free on the source drive,
  ~480 MB/s device throughput, one full streaming pass ~31 min.

## Plan

- [x] (I:H U:M S:H) Map the container: shape, dtype, chunking, metadata semantics.
- [x] (I:H U:M S:H) Quantization + lossless-compression feasibility → proved 8-bit ADC,
      bit-exact uint8+LUT round trip.
- [x] (I:H U:M S:H) Ruled out acquisition drift → validates a random split.
- [x] (I:H U:M S:M) Write `GPAM-c3/README.md` (shape, handling, split, masking, plan).
- [x] (I:H U:H S:M) **Stage 1**: build the uint8, trace-aligned rechunked cube.
      **DONE 2026-09-26 23:02** — 10 shards, 227.5 GB, 225,000,100,000 elements
      bit-verified inline, 0 mismatches; `--phase verify` re-run 2026-09-28 and logged to
      `GPAM-c3/verify_8bit.log` (structural: 50,000 ids, problems none; 80 blocks, 0 mismatches).
- [x] (I:H U:H S:H) Persist the 76/20/4 split → `c3_split.npz`, seed 20260925.
- [x] (I:H U:H S:H) Correct the label to `SBOX[rand[b] ^ (K^OPc)[b]]` after the
      `E_K(RAND⊕OPc)` construction was supplied 2026-09-28.
- [x] (I:H U:H S:H) **Stage 2**: full-coverage first-order scan, J=94, on the uint8 cube.
      **DONE 2026-09-29 01:00** (6,758 s CPU, 100% of 4,500,002 positions, 38,000 traces).
      **Outcome E: no first-order leakage anywhere.** Global max|t| = 6.30 vs a 6.44
      Bonferroni threshold — and 6.30 is *exactly* √(2 ln 423M), the noise-only expectation.
      Verified a genuine null three ways (`c3_scan_report.py`): t std = 1.000 for every
      group; exceedance counts at 3σ/4σ within 1.01× of normal for the real hypotheses too;
      longest contiguous run above |t|>4 is **1–2 samples**, identical to the negative
      control. Top-20 peaks scattered over a 3.55 M-sample span, no co-location.
      **Positive control** (`c3_positive_control.py`): a known HW template injected into real
      data is detected from **0.15 mV** up (t=6.52 vs the 6.44 threshold); 0.00 mV is correctly
      not detected. The instrument can see leakage at this level and does not invent it, so the
      null is a measurement, not an absence of evidence. Empirical floor ~0.15 mV on a 5.25 mV
      noise floor = |r| 0.029, matching the theoretical 0.033.
      **This also refuted my own concern** that autocorrelation made Bonferroni too
      conservative — adjacent samples are nearly independent, so the threshold was right.
- [x] (I:H U:H S:M) **Stage 3** `c3_poi.py`: **not applicable** — there are no POIs to
      extract. Do NOT crop-and-train; there is no window, and a 256-way head on an
      arbitrary crop of 4.5 M positions manufactures rank-0-looking numbers from noise.
- [ ] (I:H U:H S:M) **New Stage 3a**: second-order scan on the top-|t| regions — this is
      what a masked target would produce, and the natural next instrument.
- [ ] (I:H U:H S:H) **New Stage 0b (do first)**: ask the capturer what one 4,500,002-sample
      trace actually covers (sampling rate, window, is it one operation). If the premise is
      wrong, no scan or network will find anything.

### Gate 0 complete

`c3_preflight.py` — 28 checks, all passing: derived S-box matches a real AES on 256 vectors
**and** matches the repo's transcribed `AES_Sbox` (read by AST-parsing the literal, not by
importing `ASCAD_test_models.py`, which would drag in matplotlib/GNN/TimesNet); cube LUT is
256/256 finite and matches `c3_audit.json`; splits partition 0..49999; all 256 classes
present in train for bytes 0/5/15; the old `SBOX[rand^K] label differs on 38,000/38,000
train traces.

## Decisions

- 2026-09-28: **The construction is `E_K(RAND ⊕ OPc)` — AES-128 with the known `RAND`
  pre-XORed by `OPc`** (source: `GPAM-c3/dpa_aes_opc.jpg`, Fig. 5, supplied by collaborator).
  Round-1 SubBytes input is `(RAND ⊕ OPc) ⊕ K = RAND ⊕ (K ⊕ OPc)`, so:
  - **Correct label: `SBOX[rand[b] ^ (K^OPc)[b]]`**, not `SBOX[rand[b] ^ key[b]]`.
  - `K^OPc = 00 10 20 30 … f0` is the **effective first-round key**.
  - A round-1 attack can therefore only ever recover **`K ⊕ OPc`**, never `K` alone —
    they appear only as a sum. Getting `K` needs a **round-2** attack (RK2 enters
    non-linearly). Our single-head plan is round-1 only → deliverable is `K⊕OPc`.
  - **This invalidates the 2026-09-25 coarse first-order scan**: it used `K` for all 16
    bytes, and `OPc[b] != 0` for b=1..15, so 15 of 16 hypothesis bytes were wrong (only
    byte 0 survives, `OPc[0]=0x00`). Its clean null was measuring a wrong hypothesis, not
    the absence of leakage. `c3_firstorder_scan.py` / `c3_label_check.py` now use `K^OPc`.
  - Also resolves the "is this AES?" and `cipher_text != AES(key,rand)` puzzles: `E_K` is
    AES-128, and the harness recorded `RAND` into `plain_text`/`cipher_text`/`rand` and
    never recorded a ciphertext. The true block input is `RAND⊕OPc`.
  - Masking status is now **untested**, not "probed and negative".

- 2026-09-26: **Bug found and fixed mid-build.** `c3_build_8bit.py` skipped every
  even-numbered shard: the resume check tested the 0-based loop index `j` against a list
  of **1-based** "done" labels, so `j=1` (shard 2) matched `done=[1]` and was skipped.
  Run stopped at 35% having written only shards 1/3/5 = **15,000 of 50,000 traces**. The
  three completed shards are individually valid (each verified 0 mismatches), so they are
  kept and the remaining 7 are being written on resume. Fix: "already done" is now derived
  **from the filesystem**, never from a remembered counter, and `finalize()` refuses to run
  unless all 10 shards exist, each is `(5000, 4500002) uint8` with matching `level_table`,
  and covered trace ids are **exactly** `0..49999`. `--phase verify` re-opens the written
  files, decodes a random sample and bit-compares against the source independently of the
  write path. Lesson: a completeness assertion belongs in the builder, not in my reading
  of its logs — `% of cube` in the log was cumulative-by-work, which hid the gap.
- 2026-09-26: Split is **not** baked into the file layout (4% of a 5,000-trace shard is
  200 traces; indices keep the ratio changeable and every shard usable for eval).
- 2026-09-26: Cube is **uncompressed** uint8. The 4× came from the dtype, not entropy
  coding; training re-reads it every epoch and the drive has 1.3 TB free.

- 2026-09-25: `GPAM-c3/` is gitignored and standalone — no imports to/from the ASCAD or
  `GPAM/` packages. The ASCAD loaders expect flat POI vectors and do not fit 4.5M-sample
  traces, so C3 gets its own loader.
- 2026-09-25: Split is 76/20/4 = 38,000 / 10,000 / 2,000 traces, seeded permutation,
  indices persisted to `GPAM-c3/c3_split.npz` so the split is never re-derived.
- 2026-09-25: Do **not** use `GPAM/train_gpam_ascadv2_budget.py`'s GE path for c3 — it
  `SystemExit`s by design (commit `b1d3d75`); use the §7.1 ranking instead.
- 2026-09-25: Never recompress or rewrite `c3_project.sx` in place; derive a new file.
- 2026-09-25: Compression is **lossless, not lossy** — uint8 codes + 256-entry float32
  LUT reconstructs the original bit patterns exactly (proved on ~245M samples).

## Working notes / session log

### 2026-09-25 — session

- Summary: `.sx` turned out to be plain HDF5, openable with h5py. Full structure mapped.
  Headline discovery: the traces are an **8-bit ADC stored in float32** — 254 distinct
  levels file-wide, LSB 0.711 mV, DC 4.9728 V, per-trace std 5.25 mV. So 900 GB can
  become 225 GB with zero information loss.
- Files touched: `GPAM-c3/README.md` (new), `GPAM-c3/c3_project.sx` (symlink, new),
  six probe scripts (new), `.gitignore` (+`*.sx`, +`GPAM-c3/`).
- Commands / runs: `inspect_c3.py`, `c3_compress_probe.py`, `c3_level_check.py`,
  `c3_roundtrip_check.py`, `c3_drift_check.py` all completed;
  `c3_firstorder_scan.py` running.
- Metrics / results:
  - chunk lattice (49, 4395) → 1,045,504 chunks of 0.86 MB; a single-trace full read
    pulls 882 MB to return 18 MB (**49x read amplification**), 1.83 s.
  - device ~480 MB/s; 49-aligned column-window reads hit ~2 GB/s (cached).
  - level-index entropy H0 = 4.685 bits/sample → ~132 GB floor.
  - bit-exact uint8+LUT round trip: **True** on 5 dispersed regions.
  - no drift: decile means within 0.5 mV, non-monotone; per-trace mean std 0.247 mV.
  - variance is flat across all 2,250 2k-sample bins — **zero** "active" bins.
  - `plain_text == cipher_text == rand` byte-identically; `cipher_text != AES(key, rand)`
    (0/20 rows). `key` fixed 00..0f, `opc` fixed 00 11 22 .. ff.
  - **First-order scan (20 windows, 87,900 positions, 50k traces, 32 hyp):** per-window
    `max|t|` 4.21–5.06, mean ~4.60; **0/20** above the 5.5 Bonferroni threshold. The noise-only
    expectation at 140,640 tests is ~4.87, so the null is clean, not a dead pipeline. S-box
    built from GF(2⁸)+affine and validated via full AES-128 vs `cryptography`, 256/256.
    Masking therefore **neither established nor excluded** — needs full-coverage scan, and the
    recorded operation is itself unverified (`cipher_text != AES(key,rand)`).
- Open questions:
  - What did the device actually compute? The metadata's `cipher_text` is not an AES
    output, so the input->operation relation is unverified from metadata alone.
  - Is `auth` a per-trace id/counter (2 constant columns, skewed histogram)?
  - Where, if anywhere, is the AES operation inside 4.5M samples?
  - Are `zstandard` / `blosc2` available in `py10`? (Neither is installed today.)

## Next actions

| Action | I | U | S | Notes |
|--------|---|---|---|-------|
| Run corrected full-coverage first-order scan | H | H | S | **Written + validated, NOT run.** `GPAM-c3/c3_fullscan.py`; blocked because GPAM raw training holds 30.4/32.6 GB of the 5090. Script refuses to start if GPU memory is short. Design + power table: `GPAM-c3/NOTES-fullscan-design.md`. Run `--pilot` first |
| Decide round-1-only vs adding a round-2 head | H | M | M | Round 1 can only ever yield `K^OPc`, never `K` |
| Ask collaborator: what are `OPc` / `auth`? Is `OPc` derived from `K`? | H | M | H | If `OPc=f(K)` the two secrets are not independent |
| ~~Build uint8 cube~~ **DONE** 2026-09-26 | — | — | — | 225 GB, 0 mismatches over 225,000,100,000 elements |
| ~~Persist `c3_split.npz`~~ **DONE** | — | — | — | 38k/10k/2k, seed 20260925 |
| Check/install `zstandard` or `blosc2` in `py10` | L | L | H | No longer needed — cube is uncompressed by decision |

### Priority snapshot

- **Do now:** finish the first-order scan; then the uint8 conversion (gating step).
- **Before next meeting:** a POI/t-profile answer on whether C3 has first-order leakage.
- **Deep work (park):** single-head trainer + ranking, until localization answers.

## Related

- Task ids: T-019 (GPAM ASCADv2 reproduction — architecture reference only), T-022
  (Break-ASCAD roadmap — separate dataset, do not merge scoreboards).
- **`GPAM-c3/NOTES-c3-construction-and-confusions.md` — read before writing any label,
  hypothesis byte, or training loop.** Records the `E_K(RAND⊕OPc)` construction, the
  `K⊕OPc` round-1 degeneracy, the invalidated 2026-09-25 scan, and the open questions.
- `GPAM-c3/REPORT-compression-stage.md` — full record of the uint8 build + its bug history.
- `GPAM-c3/README.md` — dataset reference (shape, handling, split, cube layout).
- `docs/notes/datasets.md` — C3 is **not** an ASCAD dataset; keep it out of that catalog
  unless the user asks for a cross-reference.
- `docs/lab-log/todo.md` — C3 items mirrored (C1–C5).
