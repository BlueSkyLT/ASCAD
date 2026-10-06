# Which part of a c3 trace can carry leakage, and what licenses discarding the rest

**Short answer.** The only positions worth searching for a key-dependent signal are the first
**~50,814 samples of 4,500,002 — 1.13% of the trace**; the remaining **98.87% is inert** in every
sense a first-order instrument can measure, and the strongest single piece of evidence is that the
**global maximum of all 423 million first-order t-values sits inside the discarded region and is
exactly the noise-only expectation: observed 6.303, predicted √(2 ln 423,000,188) = 6.303.**

But "discard" is licensed to different degrees by different claims. This note separates them, gives
the criterion, the artifact and the number for each, and lists what the tail is **not** proven to be
free of. Everything below was recomputed from the on-disk profiles rather than quoted from prose;
the commands are at the end so any of it can be re-derived.

---

## 1. What "useful" has to mean, before measuring it

A position in a trace can be discarded for leakage search on three independent grounds. They are
not interchangeable, and conflating them is how the first version of this claim went wrong:

| sense | what it means | is the c3 tail empty of it? |
|---|---|---|
| **(S1) activity** | traces differ from each other at this position (no cross-trace dispersion ⇒ nothing to condition on, whatever the secret) | **yes**, at ≥1.5× median, to within one boundary position |
| **(S2) first-order label dependence** | `E[x \| label]` moves here | **yes**, over 100% of positions and all 94 hypotheses |
| **(S3) any information at all** | the region is uninformative about everything | **no** — it encodes the capture state, see §6 |

S1 is the cheap, label-free screen; S2 is the actual question; S3 is what people hear when someone
says "98.9% of the trace is noise", and it is **false**. The honest sentence is: *the tail is inert
with respect to the labels and carries information about the acquisition.*

## 2. S1 — activity, measured without labels

`c3_activity.py` (2026-10-02). For each sample position, the standard deviation across traces. No
label, no hypothesis, no key: a position where every trace reads the same value cannot be
conditioned on anything, so this is a legitimate *screen* before any leak test.

Profile: `c3_project_8bit/c3_activity_profile.npy`, shape `(2, 4,500,002)` float32 — per-sample
`std` and `mean` across traces, **all 4.5M positions**.

| | value |
|---|---|
| median per-sample std | **5.237 mV** |
| p99.9 | 6.777 mV |
| max | **30.839 mV at sample 45,000** = 5.89× median |
| positions > 1.5× median | **1,950**, last one at **50,814** |
| positions > 2× median | 385, last at 46,471 |
| positions > 3× median | 14, last at 45,001 |
| **positions > 1.5× median beyond 50,814** | **0** (re-checked: the slice `std[50814:]` reports 1, and that one is 50,814 itself — a boundary of the slice, not a straggler) |
| max ratio anywhere in the remaining 4,449,188 positions | **1.504×**, against 5.89× at the peak |

A threshold is a threshold, so two things make this one defensible: the cut is placed where the
distribution has already collapsed (0.3% above the line versus 489% at the peak), and S2 below does
not depend on it at all — the first-order scan is full coverage, so nothing outside the active
region is excluded from the leak test by this choice.

Structure inside the active window: hot positions come in clumps of 7–8 consecutive samples with
onsets 332, 1000, 1665, 2333, 2999, 3666 … (gaps 668, 665, 668, 666, 667), and the activity
profile's autocorrelation gives **ac[667] = +0.901** against ac[668] = +0.262, ac[700] = −0.156.
So ~76 events on a 667-sample cadence, all inside the first 51k, nothing after.

## 3. S2 — first-order label dependence, measured everywhere

`c3_fullscan.py`: 94 hypotheses × **all 4,500,002 positions** × 38,000 train traces, 6,758 s.
Profile kept whole: `c3_project_8bit/c3_firstorder_tprofile.npy`, `(94, 4,500,002)` float32,
1.69 GB. This is the artifact that makes the discard claim, because it does not need a window.

| | value |
|---|---|
| tests | 94 × 4,500,002 = **423,000,188** |
| Bonferroni bar, α = 0.05 | **6.44** |
| noise-only expected max\|t\| = √(2 ln M) | **6.303** |
| observed max\|t\|, whole trace | **6.303**, at sample **1,656,004** |
| positions with \|t\| > 6.44 | **0** |
| observed max\|t\| **within the discarded tail** | **6.303** (the global maximum *is* in the tail) |
| positions with max_J\|t\| > 4 | 25,375, forming **25,234 runs** — 96% are isolated single positions, mean run length 1.006, **longest run 2 samples** (a real leak persists over tens of consecutive positions; the active clumps are 7–8) |

The coincidence of observed max with the noise-only expectation to three decimals is the whole
argument: the most extreme first-order statistic the dataset can produce is what 423 million
independent Gaussians produce, and it happens 1.6M samples past anything the activity profile flags.

Because the region was chosen by a *label-free* criterion (S1), tightening the correction to the
active region is legitimate — no hypothesis information entered the selection, so there is no
selection bias in the leak test:

| | over all 4.5M positions | over the active 51k |
|---|---|---|
| Bonferroni bar (J = 94) | 6.44 | **5.72** |
| observed max\|t\| | 6.303 | **5.16** |

Nothing crosses under either correction, so the null is not an artifact of over-correcting.

## 4. S2 without a leakage model — 256-class SNR, everywhere

A t-scan needs a *template* (Hamming weight, LSB, …). The model-free 256-class SNR needs only the
class assignment, so it cannot be defeated by the wrong weight model:

    SNR_j = var of the 256 class means at position j / mean of their variances

`c3_snr_scan.py`, windows of 16,384 across **all 4.5M positions**, 16 real byte labels + 16
label-shuffled controls; profiles kept whole: `c3_snr_profile.npy` and `c3_snr_between.npy`
(each `(32, 4,500,002)`).

| | value |
|---|---|
| max SNR, real labels / shuffled | 0.9045 / 0.7532 (ratio artifacts; see the absolute scale below) |
| **in the discarded tail**: max SNR real / shuffled | **0.5041 / 0.5756** — the *shuffled* labels separate better |
| tail max absolute between-class variance | real **4.53e-07**, shuffled **4.42e-07** V² — indistinguishable |
| real/null ratio at the peak | 1.20×, priced at p = 0.0215 against exchangeable re-partitioning (README §6b: never a signal) |
| absolute class separation, max over positions | real **2.7454 mV** (byte 7) / shuffled **2.7334 mV** → ratio **1.0044**; the scan's own `real_vs_null_abs_ratio` field reads **1.0088** |
| noise-only expectation of that ratio | **1.005–1.013** (`c3_snr_sensitivity.py --numerics`, leak-free synthetic data of the same shape) — both measurements sit inside it |
| this instrument's own measured floor | **0.3 mV** at a known site |

"Shuffled labels do better than real ones in the tail" is the cleanest null result available: it is
what happens when the quantity being grouped on has no relation to the signal at all.

## 5. Why the earlier "variance is flat everywhere" claim was wrong, and survived being wrong

The first pass (`inspect_c3.py`) binned the 4.5M axis into 2,000-sample bins and judged flatness
with a 3× threshold. Bin-averaging a 7–8-sample excursion into 2,000 samples destroys it: a 5.9×
spike becomes ~1.02×. Per sample, the picture is §2. This matters as a method warning, not because
the conclusion changed — the flatness claim was *more* dismissive than the truth, and the full-
coverage scans had already covered the tail regardless.

## 6. What the tail is NOT proven free of

Say these out loud whenever the discard is invoked, because each is a live limitation:

1. **Second-order leakage in the tail is unmeasured.** Second-order coverage so far is one 667-sample
   slot plus a 76-slot cadence-aligned tiling = samples **0–50,692**. Beyond that, no pair of
   positions has ever been multiplied. The activity screen does not close this hole: a dual-rail
   style cancellation can leave per-sample variance flat while the product of two samples carries the
   secret, which is precisely why the second-order family was run at all. Full second-order coverage
   of the tail costs ~13 s per slot per pass × 6,679 additional slots ≈ **2 days**, or 17.1 h over
   every offset; nobody has authorised it.
2. **The tail is not information-free.** It predicts the acquisition: a free 256-class fit of the
   per-trace **std** of samples 100,000–100,667 on `auth[15]` gives **F = 10.31** (against 0.977 for
   `rand[15]` and 0.959 for the byte-5 label), and at samples 0–667 **F = 18.41**, while that window's
   **mean** is untouched (**F = 1.23**). See README §6e. Note *why* the activity screen could not see
   this: it is position-wise, and a per-trace scalar modulation raises every inert position equally,
   so it never shows up as a position-dependent excursion. A screen blind to scalar effects is fine
   for S1 and blind for this.
3. **Every test here groups traces by `SBOX[rand[b] ^ (K^OPc)[b]]`.** If `rand` is not the block the
   device consumed, or the whitening differs from `RAND ^ OPc`, then S2 and §4 are blind *for reasons
   unrelated to position*, and a null is not evidence about the key. That is question C9/C11, and no
   amount of scanning answers it.
4. **Multivariate models saw only crops.** Geometry A is samples 0–49,152 (96.7% of the activity
   region), geometry B is 0–51,200 (100%), and both came back at chance. So the multivariate null now
   covers the whole active region — but it is still a statement about 51,200 of 4,500,002 samples, and
   B's null has no ranking number attached to it at all, because B's test-split evaluation is an open
   decision (todo C29).

## 7. What this buys in practice

- Reading cost. The cube stores 225 GB uncompressed; one 49,152-sample crop per 38,000 traces is
  **1.87 GB** of the **225 GB** a full-trace pipeline would re-read every epoch — a ~120× I/O
  reduction, and the difference between a 30-epoch run being 51 minutes or indefensible.
- Multiple-testing power. Restricting to the active region cuts the Bonferroni bar from 6.44 to 5.72
  (§3) at no cost in honesty, because the restriction is label-free.
- Model capacity. A 128- or 768-token input is a searchable hypothesis space; 5,493 tokens at
  `patch=64` over the full trace is mostly a request to fit noise.
- The right reading of the second-order result. The one non-null localisation found so far is at
  samples 43,667/43,783 — **inside** the kept region. Discarding the tail has not cost us the only
  real signal in the dataset; keeping it would have cost 120× the I/O to look for nothing new with
  S1/S2 instruments.

## 8. Decision rule to use until contradicted

1. Crop to the activity region, but **record coverage as a number in every artifact**
   (`activity_coverage_pct`, `activity_gap_samples`) so a null can always be read as windowed.
2. Never let "we cropped" carry a second-order or S3 claim. Second-order claims need second-order
   coverage; S3 claims need the collaborator, not a scan.
3. Any covariance-shaped statistic on c3 must control for per-trace noise scale and for alignment,
   or it will find `auth` and report it as leakage. That is the cost of §6.2, measured.

## Verification — re-derive every number here

```bash
PY=/home/lantian/miniconda3/envs/py10/bin/python
$PY - <<'EOF'
import numpy as np
P = "/media/lantian/1.44.1-72806/c3_project_8bit/"
a = np.load(P + "c3_activity_profile.npy"); std = a[0]; med = float(np.median(std))
print(med*1e3, np.percentile(std, 99.9)*1e3, std.max()*1e3, int(np.argmax(std)))
hot = np.where(std > 1.5*med)[0]; print(len(hot), hot.max(), len(hot[hot > 50814]))
t = np.load(P + "c3_firstorder_tprofile.npy", mmap_mode="r")      # (94, 4_500_002)
at = np.abs(np.asarray(t[:, 1_640_000:1_680_000], dtype=np.float32))   # or stream the whole file
print(float(np.abs(np.asarray(t[:], dtype=np.float32)).max()))    # 6.303, bar 6.44, M = 423,000,188
s = np.load(P + "c3_snr_profile.npy", mmap_mode="r"); b = np.load(P + "c3_snr_between.npy", mmap_mode="r")
print(np.asarray(s[:16, 50815:], dtype=np.float64).max(), np.asarray(s[16:, 50815:], dtype=np.float64).max())
print(np.asarray(b[:16, 50815:], dtype=np.float64).max(), np.asarray(b[16:, 50815:], dtype=np.float64).max())
EOF
```

Sources: `GPAM-c3/c3_activity.py`, `c3_fullscan.py`, `c3_snr_scan.py`, `c3_snr_decompose.py`,
`c3_2o_scan.py` (one slot + tiling), `c3_2o_auth_causal.py` (§6.2), `c3_train_gpam.py` (windows).
Narrative write-ups: `GPAM-c3/README.md` §6b (activity + first-order null), §6c (the network,
geometry A), §6d–§6e (second order, and the capture-state finding), §6f (geometry B, which is the
crop that contains this whole region). Task history:
[T-027](../tasks/T027-c3-standalone-break.md), [T-029](../tasks/T029-c3-gpam-single-head.md).
