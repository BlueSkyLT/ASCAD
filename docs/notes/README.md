# Notes (plain-language lab wiki)

These files explain the project in ordinary language (aimed at someone new to crypto and hardware), then give the **exact names** used in code and data files.

You do **not** need a cryptography background. Start with [`notation.md`](./notation.md).

This is **not** a diary of experiments. Plans and session logs live in `docs/tasks/`. Meeting prep lives in `docs/lab-log/`.

## Which file?

| Open this | When you want to know |
|-----------|------------------------|
| [`notation.md`](./notation.md) | What \(S\), \(\alpha\), \(r\), a “byte”, a “trace” are. **α is not \(r_{\mathrm{in}}\).** |
| [`datasets.md`](./datasets.md) | What is in the big data files: sizes, labels, keys, SNR |
| [`metrics.md`](./metrics.md) | How we *score* an attack; **fair profiling vs attack split** |
| [`ascadv2-label-frequency.md`](./ascadv2-label-frequency.md) | Is the S-box label uneven? The “always guess 99” story vs what we counted |
| [`models.md`](./models.md) | Other papers’ models (GPAM, Scoop, …) |
| [`v1-poi-faq.md`](./v1-poi-faq.md) | Recurring v1 leakage-localizer questions (loss: spec §8; claimed model = order \(\mu_r<\mu_C\)) |
| [`gpam-algorithm-diagram.txt`](./gpam-algorithm-diagram.txt) | How GPAM actually recovers a key, start to finish — 7-act ASCII diagram (masking → shuffle → trace → heads → algebraic dissection → log-prob ranking). Includes why mask labels are training-only and why a tied rank 0 means nothing |
| [`2o-localizer-vs-gpam.md`](./2o-localizer-vs-gpam.md) | How the 2O localizer is / is not like GPAM |
| [`trace-signal-anatomy.md`](./trace-signal-anatomy.md) | Three-tier signal decomposition (opcode vs leakage vs noise), patch dynamics, and blind extraction |
| [`llm-sca-diagnosis-and-plan.md`](./llm-sca-diagnosis-and-plan.md) | Why naive LLM+pooling fails at chance on masked SCA, and progressive strategies |
| [`c3-usable-trace-region.md`](./c3-usable-trace-region.md) | **Why only the first ~50.8k of c3's 4.5M samples can carry leakage** — the three senses of "useful", the full-coverage artifacts behind each, and what the discarded tail is NOT proven free of. **Lab-mate project (`GPAM-c3/`). Not an input to the TCHES draft.** |
| [`ascadv2-trace-usage.md`](./ascadv2-trace-usage.md) | **Which parts of the 15k ASCADv2 trace a working model uses** (occlusion, 2026-10-13): what a "window" is, the seam at sample 5,000, almost nothing idle but highly redundant, and why a trained model can't be cropped |
| [`ascadv2-results-ledger.md`](./ascadv2-results-ledger.md) | **Look a number up here before quoting it** (2026-10-19): current valid results at 300k–500k, the 50k smoke results kept apart, head accuracy by scale, and every withdrawn or corrected statement. Every number carries a regime tag (`[SMOKE 50k]` / `[VALID 300k/400k]`) and a metric tag (`MTD1` / `GE<1-500` / `MTDw`) |
| [`ascadv2-profiling-drift.md`](./ascadv2-profiling-drift.md) | **The ASCADv2 profiling traces drift** (2026-10-18): the last profiling traces look most like the attack traces (they were recorded right before them); sub-sample clock slip + DC move together; why `[:n]` subsets and random validation splits mislead |
| [`patch-pipeline-survey.md`](./patch-pipeline-survey.md) | **Every patch → attention → classification variant we built and what it scored** (survey dated 2026-09-29). Which numbers are RESULT vs ORACLE vs SMOKE, and the withdrawn-claim register |

## How to use

- **You:** read the story at the top of a file; skip the “in the files” tables until you need a name.
- **Agents:** still use the tables. Do not invent metric names or dataset sizes.
- **Do not put here:** today’s experiment log, todos, meeting briefs, secrets, huge file paths.

## Related

- Task: [T-015](../tasks/T015-topic-reference-notes.md)
- Rule: `.grok/rules/05-notes.md`
- Meeting background (people, cadence): [`docs/lab-log/notes.md`](../lab-log/notes.md) — different file
