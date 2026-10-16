# How we know an attack worked

Canonical path: `docs/notes/metrics.md`. Catalog: [`README.md`](./README.md).

Imagine 256 contestants (every possible value of **one key byte**, 0 through 255). After looking at attack traces, we sort them from “most likely” to “least likely.”

**Rank** = how far down the list the *true* key byte sits.

- Rank **0** = it is in first place. We recovered that byte.
- Rank **255** = it is in last place. We did worse than a coin flip.

We care about this ranking, **not** “what percent of single traces did we classify correctly.” A model can be a bad classroom tester and still, over many traces, push the true key to the top.

The usual pipeline:

1. The network looks at **one** trace and outputs 256 numbers: “how much do I believe each possible \(S\)?” (\(S\) is the S-box output; see [`notation.md`](./notation.md).)
2. We know the **plaintext** byte. For each guessed key byte we can turn that into a guess of \(S\), and add up (in log space) how much the network liked it.
3. Repeat on more traces. The true key should climb the leaderboard.

This is the classic ASCAD ranking, implemented in `ASCAD_test_models.py` and automated by `benchmark_full_fixedkey_gpu.py`.

---

## Fair experiment: profiling vs attack

**“Legal” here means a fair profiled-SCA split**, the same protocol the ASCAD paper and later DL-SCA work use. It does **not** mean “the loss omitted Hamming weight.”

ASCAD is **profiled**:

1. **Profiling set** — clone device, **known key** (and, in the public files, known plaintext and masks). Training, SNR, POI picking, template building all happen here. Supervising with \(S^\star=\mathrm{Sbox}(p\oplus k)\), with \(\mathrm{HW}(S^\star)\), or (worst-case) with mask bytes is standard. The official ASCAD CNN uses 256-way \(S^\star\), which is *more* information than HW.
2. **Attack set** — **held-out** traces. The attacker gets traces and plaintext, **not** the key. Every key byte \(0\ldots255\) is scored (network \(P(S)\) or a CPA correlation). Rank / GE is the position of the true key in that list. Computing that rank after the fact is evaluation, not feeding the secret into the attack.

| Fair | Not fair |
|------|----------|
| Train on profiling labels (\(S^\star\), HW(\(S^\star\)), optional masks) | Train, or pick POIs, on the attack traces / attack key |
| Attack with 256 **guesses** of \(k\) | Score the attack set with the true \(S^\star\) or true \(k\) |
| SNR / POI on the **profiling** split | SNR on the attack split used as the attack itself |
| Use `desync` only to *debug* whether windows followed the shift | Feed `desync` into the model at attack time and call it a jitter attack |

**Non-profiled** CPA (no training set; only HW of key guesses) is a different, harder setting. We are not in that setting when we train on `Profiling_traces`.

A lab preference such as “do not train the searcher with an HW-correlation loss” is a **method design** choice (e.g. to match CE+GE, or to avoid rediscovering 2O CPA). It is not the fairness bar. Recurring wording: [`v1-poi-faq.md`](./v1-poi-faq.md) §0.

---

## Two headline numbers

They appear in `summary_full_fixedkey_gpu.json`.

### `rank_final`

Where the true key sits **after we have used the whole attack set**, in the **fixed order** those traces are stored.

- Lower is better. **0** is perfect.
- Integer, or `null` if something broke.

Example: `"rank_final": 12` means after 10 000 attack traces the true key is still only 12th.

### `traces_to_rank0`

How many traces (counted in steps of `rank_step`) until the true key **first** hits rank 0.

- Smaller is better (faster recovery).
- `null` if it never hits 0 in the budget.

Example: `"traces_to_rank0": 590` — first time we were sure was after 590 traces.

Ranks are allowed to **wiggle**. A model can hit 0 early, then lose first place later. Then you can see both:

```json
"rank_final": 12,
"traces_to_rank0": 590
```

`rank_final` is only the **end** of the budget, not the best moment in the middle.

---

## Guessing entropy (GE) — “what if we shuffled the attack traces?”

One ranking depends on **order**. Papers usually report **guessing entropy**: the **average** rank if you repeat the attack many times, each time using the traces in a **random order**.

\[
\mathrm{GE}(n) = \text{average rank of the true key after } n \text{ traces}.
\]

We estimate that with many random shuffles (default **100**). Code: `ASCAD_test_models.guessing_entropy`, called from the benchmark as `ge_metrics`.

| Field | Meaning in English |
|-------|---------------------|
| `ge_final` | Average rank after **all** `eval_traces` |
| `ge_at` | Snapshots, e.g. at 100 / 1 000 / 10 000 traces |
| `ge_traces_to_0` | Smallest \(n\) where **every** shuffle has rank 0 |
| `mean_traces_to_rank0` | Average, over shuffles that succeed, of “when did this shuffle first hit 0” |
| `median_traces_to_rank0` | Median of those times |
| `frac_reached_rank0` | Fraction of shuffles that ever hit 0 |
| `success_rate_final` | Fraction that are rank 0 at the **end** |
| `ge_n_attacks` | How many shuffles (`--ge-attacks`, default 100) |

### Easy to mix up

| | `rank_final` | Average of `rank_final` across training seeds | `ge_*` |
|--|--------------|-----------------------------------------------|--------|
| What is random? | Nothing (file order) | We **retrained** the network | We **reordered attack traces** |
| Is it paper GE? | No | No | **Yes** for the GE *curve* |

Adding log-scores does not care about order once you have used **the same full set** of traces. So `ge_final` should match `rank_final` (up to step size / rounding). The extra GE numbers matter for the **middle** of the curve and for “how many traces until success.”

Turn GE off with `--ge-attacks 0`.

```bash
python benchmark_full_fixedkey_gpu.py ... --ge-attacks 100 --ge-seed 12345
python benchmark_full_fixedkey_gpu.py ... --ge-attacks 0
```

The shuffle seed is `ge_seed + training_seed`, so different training seeds stay reproducible and distinct.

---

## SNR: read it against its own noise floor before comparing two targets

Per-sample $\mathrm{SNR}(t)=\mathrm{Var}_{between}(t)/\mathrm{Var}_{within}(t)$, grouping $N$
profiling traces by class label. **Under a null (no signal) this statistic is not 0 — it is
$\chi^2_{K-1}/N$, whose mean is $(K-1)/N$.** So its floor scales with the number of classes $K$:
with $N=16{,}384$, a 256-class target starts at $0.0156$ and a 16-class target at $0.00092$, i.e.
the 256-class number is inflated 17× by pure sampling noise.

Two consequences, both of which bit us:

- **Never rank targets by raw SNR when their $K$ differ.** "SNR(π) 0.0008 < SNR(C) 0.0155, so the
  shuffle is the invisible one" is exactly backwards: after subtracting each floor, π's peak excess
  is $0.2075$ (class-mean separation $0.46\sigma$) against C's $0.0566$ ($0.24\sigma$).
- **Fixed thresholds are also biased** by the same $K$: "samples above 0.1" is 6.4× C's floor but
  109× π's. Report $\mathrm{SNR}-(K-1)/N$, or the per-sample $\chi^2$ significance, instead.

SNR is a *per-sample, univariate, linear* readability measure. It is not a learnability bound: a
model integrates over samples (correlated, so the count of significant samples over-states the
information) and exploits cross-head structure that a per-target profile cannot see. Achievable
accuracy also depends on $K$ — $0.46\sigma$ of separation across 16 classes is worth $\approx 10\%$
top-1 / $36\%$ top-4 from **one** sample, while $1.41\sigma$ across 256 classes is worth only
$\approx 1.6\%$. Measure: `GPAM/artifacts/C-snr-profile.py` (arrays in `snr_{tag}_{name}.npy`).

---

## Knobs that change the numbers

Saved on every benchmark row:

| Field | Default | Meaning |
|-------|---------|---------|
| `eval_traces` | 10000 | How many attack traces we use |
| `rank_step` | 10 | How often we write down the rank |
| `target_byte` | 2 | Which of the 16 key bytes (0–15) |
| `ge_n_attacks` | 100 | Number of shuffles (`0` = skip GE) |
| `ge_seed` | 12345 | Base seed for those shuffles |

**ASCADv2 shuffle routing (since 2026-10-17, T-041).** The 16 shuffle heads each guess "which byte is processed at this time slot" on their own, so together they can claim the same byte twice. Every v2 attack decoder now first converts them into the exact probabilities of a valid shuffle, in which each byte occurs once (`perm_routing.apply_pi_routing`). Measured on 7 models, this raises median z by +0.08 to +0.55 and never loses a byte. Environment variable `ASCAD_PI_ROUTING=raw` restores the old routing and reproduces every number recorded before 2026-10-17. Numbers from before and after this date are not directly comparable; say which routing a table uses.

---

## Averaging several training seeds

`aggregate_full_fixedkey_gpu.json` stores mean and standard deviation across **retrains**. Failed / `null` rows are dropped.

That is **not** GE. Seed average = “we trained again.” GE = “we shuffled the attack list.”

---

## Reading the numbers

- Smaller `traces_to_rank0` → faster.
- Smaller `rank_final` / `ge_final` → more sure at the end.
- `traces_to_rank0` is `null` but `rank_final` is 5–20 → close, not quite first place.
- Compare **desync0 / 50 / 100** separately; jitter makes the problem harder.
- This is **one key byte**. A full AES key has 16 bytes.

Same ranking functions: `ASCAD_test_models.py` (`full_ranks`, `rank`, `guessing_entropy`) and the GPU benchmark. Plots from the test script are “number of traces” vs “rank.”

Paper: *Study of Deep Learning Techniques for Side-Channel Analysis and Introduction to ASCAD Database* — https://eprint.iacr.org/2018/053.pdf

## Where the numbers live

- `benchmark_full_fixedkey_gpu.py`
- `ASCAD_test_models.py`
- `…/ASCAD_trained_models/benchmark_runs/summary_full_fixedkey_gpu.json`
- `…/aggregate_full_fixedkey_gpu.json`
- `run_full_fixedkey_gpu.sh`
