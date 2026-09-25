# Brief C — talking points for the 14:00 meeting

Written 2026-09-24, ~01:30. Scope note: this covers **more than brief C as written**.
The brief asked for a 90-minute pipeline smoke plus a reading of two notes. It
explicitly said the smoke's accuracy is not a result and that no Table 11 number
should be proposed. The 2026-09-24 01:14 instruction widened that to "run the
smoke, then test it — predictions, comparison to the paper, GE and non-GE
metrics" with a 3-hour ceiling. Both are reported, and they are kept apart.

## GPAM

### What the paper actually reports, and it is not GE

Table 11 (PDF p.21 / printed p.492, §7.1) is **machine-learning metrics on the
80,000-trace holdout**, not key recovery. Per head, `Accuracy[%] | MeanRank | MaxRank`:

| Head | Meaning | Acc % | Mean rank | Max rank | vs chance |
|---|---|---|---|---|---|
| `rm` | multiply hide α (`masks[18]`) | **100.00** | 0 | 0 | chance 0.39 % |
| `rout` | add hide β (`masks[17]`) | 18.25 | 4.78 | 65 | chance 0.39 % |
| `c[i]` | masked S-box byte, avg over 16 | **1.18** | 80.65 | 255 | chance 0.39 % |
| `p[i]` | shuffle index, avg over 16 | 95.07 | 0.055 | 4.44 | chance 6.25 % |

Read the third row out loud. **The paper's own masked-S-box head is at 1.18 % —
about 3× chance, mean rank 80.65 of 255.** GPAM's ASCADv2 result is carried by
predicting the *mask* (`rm` at 100 %), not by seeing the S-box. That is the same
conclusion we reached from the other direction with the frequency table.

Which of these our numbers can be put next to, directly: `alpha_mask` and
`beta_mask` are stored as **one value per trace** (shape `(1,)`), not per byte, so
our `rm` and `rout` head accuracies are like-for-like against those two rows.
`sbox_masked` and `perm_index` are stored 16-wide, and the paper's `c[i]` / `p[i]`
rows are **averages over 16 bytes** — our byte-0 numbers are one draw from that
average, not the average itself.

The `~80 traces` line is **not in any table**. It is prose in §7.1: "Following
[MS23] we simulate a fixed key (all zero) by replacing pt by its XOR with the
corresponding random key. Since rm has very high accuracy we use it directly
instead of the whole probability distribution. We sample 10,000 times to
estimate the GE and need roughly 80 traces to have GE of all key bytes under 1,
comparable with the 60 traces needed by [MS23]."

Three things hide in that sentence, and each one is a reason our number is not
the same quantity:

1. **ASCADv2 has no fixed key.** I checked the file: the Attack split has 2,000
   distinct keys in 2,000 sampled traces, and Profiling has 2,000 in 2,000. Every
   trace carries its own random key. You cannot accumulate evidence across
   traces unless you *manufacture* a fixed key by XOR-ing the plaintext with that
   trace's own key. That is exactly what they do — it is a simulation, and it is
   legitimate, but it is not the classical "one unknown key, many traces" attack.
2. **`rm` is taken by argmax**, i.e. treated as known-good, because it is 100 %
   accurate for them. If our `rm` head is not near 100 %, the whole chain breaks
   at the first step and the 80-trace figure is not the right comparison.
3. Threat model is **First Threat plus white-box**: full knowledge of the
   protection parameters during training. Mask bytes are *prediction targets*.
   This is not the strict no-mask-label bar in `docs/survey-ascadv2-dl-2022-2026.md`.

Table 1 is **hyper-parameters only** — no accuracy, no metric: batch 64, 1,000
steps/epoch, **150 epochs**, target LR 5e-5, merge filters 0/0, trace length
1,000,000, **patch size 400**. Table 2: ASCADv2 training takes **20 hours on an
RTX 4090**; hyper-tuning was 29 days cumulative. Table 3: 640,000 train / 80,000
test / 80,000 holdout, 0.9 TB. §7.1 uses all 1,000,000 points with no
preprocessing, specifically instead of an SNR crop to 15,000 points.

### What is actually true on this machine, tonight

- `scaaml` imports; `get_gpam_model` builds. Our smoke makes a 1.75 M-parameter
  model over 2,500 patches of 400 samples.
- **The raw shard-8 smoke completes a training epoch, with no special flags.** It
  has now been run three times, all passing: by `overnight-baton.sh` at 00:56:49
  (exit 0 in ~7 s; numbers recorded in T019), and twice from this session at
  01:15 and 01:17. `run_ascadv2_gpam_baseline.py --dataset-kind raw
  --patch-size 400 --n-train 32 --n-val 16 --epochs 1 --batch-size 8` prints
  `Loaded 32 train / 16 val traces`, builds `(None, 2500, 400)` (2,500 even
  steps, 1.75 M params), runs 4 steps at ~210 ms/step, 1 epoch.
- Numbers from the three runs, which is the honest way to report them because
  they bounce around: loss 5.5684 / val_loss 5.5442 / val_acc 0.0 (00:56), loss
  5.5249 / val_acc 0.0625 (01:15), loss 5.5482 / val_loss 5.5312 / val_acc
  0.0625 (01:17). Chance CE for 256 classes is ln 256 = 5.545, so every one of
  these is **at chance**, and `val_acc 0.0625` is one correct trace out of
  sixteen.
- One loose end to own up to: the 01:15 re-run used `>` on
  `out/C-gpam-smoke.log` and **truncated the baton's original log**. The
  baton's numbers survive in T019's session log, so nothing is lost, but the
  file at that path is now this session's run, not the 00:56 one.
- This is a pipeline check, not a result. Shard 8 is traces 700,000–799,999,
  which sits on the paper's validation tail and its entire 80,000-trace holdout.
- `GPAM/gpam_budget_run.log` is still **0 bytes** and there is no
  `GPAM/gpam_budget_results.json`. So "rank 0 in ~80 traces" remains the paper's
  claim, not ours. `docs/roadmap-break-ascad.md` §6 gap 4 says this and is
  correct except that it predates the smoke log now existing.
- Both GPAM scripts still hard-code `run_eagerly=True`
  (`run_ascadv2_gpam_baseline.py:163`, `train_gpam_ascadv2_budget.py:90`). Graph
  mode is planned, not done.

### `compute_ge` in the budget script is wrong, three independent ways

`GPAM/train_gpam_ascadv2_budget.py:95–121` must not be used to produce a v2 GE
number. Each of these alone invalidates it:

1. It scores hypotheses with `AES_SBOX[plaintext[byte] ^ key_hyp]` — the
   **unmasked** S-box at the **unpermuted** index. The head predicts
   `α·S[p[π]⊕k[π]] ⊕ β`. Without dividing out α, XOR-ing off β, and routing
   through π, the hypothesis and the label are different quantities. §7 gives the
   correction as a XOR-convolution, `P[s=b] = Σ_i P[x=b⊕i]·P[y=i]`, and the repo
   already implements that dissection correctly in
   `ASCAD_test_models.py:687–743` (`proba_dissect_beta/alpha/permind`).
2. It reads `true_key = attack_meta[0]["key"][byte_idx]` — trace 0's key — and
   applies it to every trace. With per-trace random keys there is no such thing.
3. It accumulates log-probabilities across attack traces, which only means
   something under the §7.1 fixed-key simulation it does not perform.

Any GE it has ever printed for v2 is an artifact. This is worth a sentence at the
meeting because it is the cheapest real finding of the night: we had a script
that reports a GE, and the number is not a GE.

### One correctness win

Our **label algebra is right**. `sbox_masked[b] == gf256_mul(masks[18],
AES_SBOX[plaintext[perm_indices(b, masks[0..3])] ^ key[same]]) ^ masks[17]`
reproduced the stored labels exactly on rows 0–2 of both the Profiling and Attack
splits, and `perm_indices` matches the stored `perm_index`. Two caveats found
while checking: `docs/notes/datasets.md` says α ∈ 1–255 and never 0, but the
full profiling set does contain α = 0 rows (5 in 500,000) — my 2,000-trace
strided sample showed min 1, so this only bites if someone samples the whole
split, and division by α is undefined there; and raw shards have **no labels at
all**, only `key`, `plaintext`, `masks`, `ciphertext`.

### What we are NOT claiming

We have not reproduced Table 11. Our comparison run differs from the paper in
every load-bearing dimension: ~4 heads vs ~35, `output_relations=[]`, tens of
thousands of training traces vs 640,000, 8 epochs vs 150×1,000 steps, patch 150
on the 15,000-point extracted file vs patch 400 on 1,000,000 raw points. The next
cited run waits on the full raw download; shards 1–7 are still `.part`.

## The caveat that applies to our own GPAM numbers

Every model we have ever trained on ASCADv2 — brief A's patch transformer,
TimesNet, Scoop, and the GPAM run reported below — trains on
`ascadv2-extracted.h5`, which is **15,000 samples wide**. That file is a crop:
`STM32_AES_v2/Readme.md:56` calls it "the 15,000 points of interest from the raw
dataset", and `docs/notes/datasets.md:150` says it "keeps **15,000** interesting
samples, two raw windows glued together".

15,000 / 1,000,000 = **1.5 %** — the exact number in GPAM §7.1, where the authors
say they "perform an attack by using all 1,000,000 points of each trace **without
preprocessing**, instead of using an SNR analysis to use only 15,000 points
(1.5 %)". `docs/notes/2o-localizer-vs-gpam.md:37` already records "Selling point
of the paper: **no SNR crop**".

So GPAM's headline result and our entire local setup differ by one variable we
have not controlled: **98.5 % of the input.**

**I drafted this section as an excuse for brief A, and my own run then refuted
it.** Keep the caveat for what it still covers, and drop it where it no longer
applies:

- It still covers the *comparison to the paper's config*: 40,000 traces and 8
  epochs against 640,000 and 150×1,000 steps, 4 heads against ~35, patch 150 on
  the crop against patch 400 on 1M raw points. Failing to hit the paper's `c` and
  `p` numbers on 1.5 % of the input is not a refutation of GPAM.
- It **does not** cover α. Our `rm` head reaches **92.95 %** on that same crop,
  against the paper's 100.00 %. The crop plainly is not preventing mask
  prediction. So the reading "A2 failed because we only gave it 1.5 % of the
  trace" is wrong, and nobody should say it at 14:00 — we now have a positive
  result on the same file and the same label. What separated A2 from our run was
  the architecture and the budget, and that is a much more useful thing to know.
- It does not cover `c` or `p` in the other direction either: `p` at 8.6 % against
  their 95.07 % is a real gap we have not explained, and we do not know how much
  of it the crop is responsible for.

A raw 1,000,000-point run is still the only way to close the config question, and
that is the gate T-019 is already blocked on: shards 1–7 were still downloading
at 01:14 and only shard 8 is complete. Multi-day, not a morning.

## What our own GPAM run measured

GPAM itself (`scaaml.get_gpam_model`), 4 heads, extracted ASCADv2, byte 0, seed 0,
40,000 profiling traces, 8 epochs, patch 150 (100 steps), batch 64, Adafactor
lr 1e-3. Wall 1,098 s. Holdout = traces 400,000–401,999 (never trained on);
attack = the public 2,000. Script `out/C-gpam-eval.py`, numbers
`out/C-gpam-eval.json`.

### Non-GE: our four heads against Table 11's four rows

| Head | Our acc % (holdout / attack) | Our mean rank (h / a) | Our max rank | Paper acc % | Paper mean rank | Paper max rank | Call |
|---|---|---|---|---|---|---|---|
| `rm` | **92.950 / 92.350** | **0.083 / 0.102** | 3 / 4 | 100.00 | 0 | 0 | **Reproduces.** 93 %, top-5 = 100 %, worst miss ranked 3rd |
| `rout` | 2.750 / 2.750 | 36.05 / 41.01 | 161 / 190 | 18.25 | 4.78 | 65 | 7× chance but ~7× worse than the paper |
| `c` | 0.650 / 0.500 | **126.50 / 125.09** | 255 / 255 | 1.18 | 80.65 | 255 | **At chance** (127.5 is the true null). Paper's is 80.65 |
| `p` | 8.600 / 9.550 | 6.51 / 6.47 | 15 / 15 | 95.07 | 0.055 | 4.44 | **Fails badly** — 1.4× chance vs their 15× |

`rm` argmax equals the stored `alpha_mask` on **92.35 %** of attack traces, and
`perm_indices(b, masks[0..3])` reproduces the stored `perm_index` label on
**100.00 %** of 2,000 attack traces — so our label algebra is right at scale, not
just on the six rows checked earlier.

### The result that reorders the meeting

**GPAM's architecture predicts the multiply hide α at 93 % on the 1.5 % crop.**
Brief A's patch transformer, on the *same file and the same label*, scored 0.30 %
— chance. So the crop does not explain A2, and neither does the target being
invisible. The two variables that actually differ are the architecture (GAU stack,
patch 150, 100 tokens vs patch 16 / stride 8, ~1,872 tokens, attention pooling) and
the budget. **The "drop this family" conclusion was about one model, not about
ASCADv2.** I wrote a caveat above claiming the crop probably explained A2; my own
run refutes it, and the caveat section now says so rather than being deleted.

What that buys us is narrow and must be stated precisely: α is a **mask label**.
Predicting it at 93 % is worst-case / First Threat evidence. It is not a key, and
it is not a break.

### GE: we do not reproduce ~80 traces, and the curve is not a curve

§7.1 simulation — `pt_eff = pt ^ key` so the true key byte is hypothesis 0, `rm`
taken by argmax, `rout` marginalised as a XOR-convolution, all masks **handed as
truth** (the most generous regime, worse threat model than the paper's attack
phase):

| Traces | 1 | 10 | 50 | **80** | 200 | 500 | 1000 | 2000 |
|---|---|---|---|---|---|---|---|---|
| Perfect `c` head (control) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Ours, all masks true | 208 | 254 | 229 | **231** | 133 | 2 | 10 | 20 |
| Ours, all predicted | — | — | — | 35 | 132 | 2 | 4 | 12 |

Read the paper's own operating point: **at 80 traces our rank is 231 out of 255 —
worse than a coin flip — even handed every true mask label.** The perfect-`c`
row proves the machinery is correct: given a real head it returns rank 0 at a
single trace. So this is the model, not the code.

And the sequence is **non-monotone** (231 → 133 → 2 → 10 → 20). More traces must
not make a working attack worse; the "2 at 500" is a dip in noise, not a
convergence.

Two nulls that make that concrete, both worth repeating at the meeting:

- **Tie artifact.** With `c` replaced by a uniform distribution the score vector is
  *exactly* constant (all 256 hypotheses at −2772.5887) and a strict `>` rank
  returns **rank 0** — an entirely uninformative model appears to recover the key.
  Any GE implementation without tie handling can report a break it does not have.
  This is the same class of bug as `compute_ge`.
- **Signal-free band.** Randomly permuting which trace each prediction belongs to
  — destroying the trace↔value link while keeping the head's marginals — gave
  ranks at 2,000 traces of `[10, 29, 42, 85, 91, 166, 198, 203, 210, 238, 239, 244]`,
  median 182. **Ours is 20.** One control, with no valid trace pairing at all, beat
  us. Our number is inside the noise band of a model that cannot see the trace.

The trace-independent prior control here gave rank 194 at 2,000 traces. Note this
is the *masked* value's prior, which is nearly flat because α and β are uniform —
not the unmasked-Z frequency table that produced byte-4 rank 3. Do not line the
two up.

### Where the chain actually breaks

Not at `rm` — that one reproduces. It breaks at **`c`**, the masked S-box head,
which sits at chance (mean rank 126.5 against a null of 127.5, versus the paper's
80.65), and at **`p`**, the shuffle index (8.6 % against their 95.07 %), which
routes each trace to the right key byte. With `c` uninformative and `p` wrong on
90 % of traces there is nothing for the GE accumulation to compound. `rout` is
weak in both campaigns and is not the blocker.

So the one sentence to take from this run: **we reproduced the mask head and
failed the value head and the shuffle head** — which is the opposite of what the
architecture was supposed to be bad at, and it says exactly which head to work on
next if GPAM is pursued at all.

One more guard on the `p` claim, because "8.6 % against 6.25 % chance" is the kind
of number that flatters itself: `perm_index` for byte 0 is genuinely uniform on
the attack set (class counts 111–141 around an expected 125, within binomial
noise), and the **most frequent single class alone scores 7.05 % with no trace at
all.** Our 9.55 % is 2.5 points over that. So `p` is not mildly working — it is
essentially unlearned, against the paper's 95.07 %.

### What this run does NOT license

40,000 traces and 8 epochs against the paper's 640,000 and 150×1,000 steps;
4 heads against ~35; `output_relations=[]`; byte 0 against their 16-byte averages
for `c` and `p`; the 1.5 % POI crop against no crop. On this evidence we can say
our `rm` head is close and our `c`/`p` heads are not working. We cannot say GPAM
fails, and we cannot say it works.

## Which numbers to defend

The measured fact from `docs/notes/ascadv2-label-frequency.md`: a model that
**never looks at a trace** and plays back the training frequencies of the
unmasked Z = Sbox(p⊕k) puts the true key at **rank 3 on byte 4** after 300,000
attack traces. The tilt behind it is mild — the favourite value on byte 4 is
0.42 % of traces against 0.391 % if perfectly flat, about **1.1× fair**. Full
16-byte prior-only table in that note; the rows that matter tonight:

| Byte | prior-only rank @10k | @100k | @200k | @300k |
|---|---|---|---|---|
| **4** | 74 | **13** | **7** | **3** |
| 15 | 255 | 255 | 242 | 213 |
| 9 | **0** | 18 | 10 | 71 |
| 0 | 232 | 232 | 253 | 218 |

Note row 9: the prior alone reaches **rank 0 at 10,000 traces**. Nobody should
claim byte 9 without an AOD-GE number.

| Result | Where it lives | Verdict |
|---|---|---|
| Scoop twin, byte 4, `ge_last` 2.75, `best_ge` 2.75, 37,504 s wall | `Scoop/artifacts/timesnet_blackbox/logs/timesnet_pooled_scoop_b04_seed0_full.log` | **Same neighborhood as the prior.** A network burned ten hours of wall clock to reach GE 2.75; a lookup table that never reads a trace reaches rank 3. We cannot tell them apart until AOD-GE is run. Do not say the network *is* the prior — the note explicitly does not prove that. Say we cannot distinguish them. |
| TimesNet pooled, byte 4, `ge_last` 3.65, `best_ge` 2.9 @ 190,001, `traces_to_ge0 = None` | `.../timesnet_pooled_b04_seed0_full.log` | **Same neighborhood as the prior**, same caveat. It also never reached GE 0 by 300k, so it is not a break on its own terms. |
| Scoop repro-v2, byte 15, `ge_last` 0.0, `best_ge` 0.0, `traces_to_ge0 = 100,001` | `Scoop/artifacts/ascadv2_repro_v2/byte_sweep/BYTE_SWEEP_REPORT.md` | **Not this cheat** — the byte-15 prior-only rank is 213–255 across the whole range. Still not a strict break: one seed, a contested re-split, and **100,001 attack traces**, which is three orders of magnitude from GPAM's ~80. Also `c_top1_acc = 0.0036`, i.e. chance-level top-1 while GE reads 0 — that combination needs explaining before we lean on it. |
| Scoop repro-v2, byte 4 `ge_last` 15.75; byte 9 `ge_last` 60.3 | same report | Byte 4 is above the prior's reach at that budget but adjacent enough to say the metric. Byte 9's prior hits rank 0, so quote its number only with the prior's number next to it. |
| v1 `timesnet_ascad_head`, CNN baseline, 2O localizer `rank_final` | fixed-key ASCADv1, 10k attack traces | **Different metric, say the metric.** Different dataset, different budget. The 300k-trace prior does not move these claims. The 2O claim separately still lacks a persisted rank JSON (roadmap gap 1) — `exp_2o.py` prints it; six checkpoints exist. That is a citation caveat, not a GE caveat. |

**AOD-GE** (divide the posterior by Pr(Z)) is **not implemented in this repo**. Do
not implement it tonight. It is the metric to add before any future v2 GE is
called a break.

## Sentences the speaker should not say

1. **"We reproduced GPAM"** or **"we got GE under 1 in ~80 traces."** We did
   neither. At the paper's own 80-trace operating point we are at rank 231 of 255
   with every true mask handed to us. The 80-trace figure is the CHES paper's.
2. **"TimesNet breaks byte 4 of ASCADv2."** The number we are proudest of, GE
   2.75 after 37,504 s of wall clock, sits inside what a trace-free frequency
   table reaches on that same byte (rank 3).
3. **"The patch transformer failed, so this architecture family is dead"** or
   **"α cannot be learned."** We now predict α at 92.95 % with GPAM's own model on
   the same file. A2's chance result was about that model at that budget.
4. **"We got a low GE, therefore we recovered the key."** Our own shuffled
   control — predictions assigned to the wrong traces — produced rank 10 at 2,000
   traces, beating our real 20. And a uniform head produces rank 0 outright
   through tie handling. On this dataset a low GE number is not evidence until it
   clears a signal-free control and AOD-GE.
5. **"This is a strict break."** Every head here predicts mask labels. That is
   worst-case First Threat, not the no-mask-label bar in
   `docs/survey-ascadv2-dl-2022-2026.md`.

## Next actions this produces

| Action | Why | Cost |
|---|---|---|
| Fix the tie handling in every GE path, ours and `compute_ge`'s | A uniform head currently reports rank 0 | ~lines |
| Treat `p` (shuffle index) as the blocking head, not `rm` | 8.6 % vs their 95.07 %, and it routes every key byte | needs a run |
| Add AOD-GE before any v2 GE is called a break | Byte 4's 2.75 vs prior's 3 is currently unresolvable | not implemented |
| Re-run `rm` at the paper's budget only if raw shards land | Answers the config question honestly | days, blocked on shards 1–7 |

