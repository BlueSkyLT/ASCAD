# Patch → feature/attention → classification: what has been tried, and what came back

**Survey date 2026-09-29.** Read-only pass over the code, `docs/tasks/`, `docs/notes/`,
`docs/roadmap-break-ascad.md`, and the ~60 persisted run artifacts. No run was launched to
produce this file; every number is quoted from an artifact or a task note, with its path.

## How to read this

**Chance bars.** 256-way heads: `chance_ce = 5.5451774` (= ln 256), `chance_acc = 0.00390625`.
For `alpha_mask` / `beta_mask` class 0 never occurs, so the correct accuracy bar is
1/255 = 0.00392157 (`docs/2026-09-23/out/A-verdict.md:46`). Permutation `p` heads are 16-way,
chance 6.25 %.

**Grade, not just number.** Every row below carries one of:

| Grade | Meaning |
|---|---|
| **RESULT** | learned from traces only, scored on held-out attack traces |
| **ORACLE** | used ground-truth labels or hand-placed windows; characterises the data, is not an attack |
| **SMOKE** | plumbing check on a tiny subset; the recorded "verdict" is meaningless |

**Metric names are the repo's own** — `rank_final`, `traces_to_rank0`, `ge_last`,
`traces_to_ge0`, `mean_rank`, `acc_pct`, `val_loss`. Nothing here introduces a new metric.

## TL;DR

1. **Roughly 40 distinct patch/attention configurations were built and ~30 recorded runs
   exist. Not one produced a key rank of 0 by learned means on either dataset.** 46 files in
   the repo contain the string `FAIL_CHANCE`.
2. **Explicit averaging is fatal — but that is not what our patch models did, and it is not why
   they failed.** Holding the *same* 1,600 samples fixed: raw 0.8603 → mean-of-2 **0.0083** top-1
   on α. GPAM groups samples into blocks 10–25× wider than our patches and still reaches 98.34 %,
   so grouping is not the problem. The likelier cause is that every patch run on a readable label
   was early-stopped at 6–7 epochs, and GPAM's own breakthrough came at epoch 6. See §7b, §7c.
3. **The label is readable, and now on raw data too.** GPAM — grouped attention over
   non-overlapping blocks, no patch-mean — gets `rm`(= α) **92.95 %** held-out on the extracted
   crop (`GPAM/artifacts/C-gpam-eval.json:53`) and **98.340 %** on raw 1 M-sample traces
   transferred to a disjoint shard (§10). The 2026-09-24 "this family cannot see a leak with
   SNR ≈ 2" verdict was **retracted the same night** (`out/A-verdict.md:3-20`). Both are
   worst-case: α is a mask label.
4. **The only rank-0 in the patch family is an oracle** (`PW-probe-v1-b2`: hand-placed
   patches at known-leaking positions, `rank_final 0`, `traces_to_rank0 120`).
5. **Three traps generated most false optimism**: label-based band selection (oracle), the
   label-frequency prior (a trace-ignoring model reaches key rank 3), and smoke checks whose
   `ABOVE_CHANCE` string survives in the artifacts (`WIRING.json`, `OVERFIT-flatten.json`).

## The tally

| Family | Built | Recorded runs | Best honest outcome |
|---|---|---|---|
| A. patch → frozen GPT-2 → mean pool | ~10 configs (3 CLIs) | ~12 | CE 5.5724–5.5941, acc ≤ 0.0055 — **at chance**; rank never computed |
| B. patch → trainable transformer → attn/bilinear/flatten | 7 (3 strategies × pools) | ~8 | `rank_final` 94–245 — **at chance** |
| C. resolution-preserving readouts (soft window, pair words) | ~10 | ~9 | acc 0.0354 (9.1× chance) on β, **no key rank**; rank 0 only as ORACLE |
| D. sliding-window proposers (2O / 3O) | ~10 | ~10 | **2O `rank_final 0`** on v1 b2 — but no persisted JSON (roadmap gap 1) |
| E. GPAM grouped attention | 3 scripts | 4 | `rm` **98.340 %** held-out on raw 1 M traces (§10); 92.95 % on crop (mask label on ⇒ worst-case, not a break) |
| F. contrast: TimesNet / MultiResNet / CNN / GNN-segments | 6 + GNN | several | v1 CNN `rank_final 0` @190 traces; v2 all-16 rank 0 **uncontrolled** |

---

## 1. Family A — patch → frozen GPT-2 → mean pool → 256-way

`LLM/models.py:42` `PatchGPT2SCA`. Patchify is `unfold(1, patch, step)` with zero-pad or
truncate (`models.py:238-245`), `n_patches = 1 + ceil((trace_len − patch)/step)`
(`models.py:145`); default patch 16 / stride 8; **no positional embedding** — instead a frozen
GPT-2 *text prompt* is prepended and sliced off (`models.py:269-279`). Backbone fully frozen
(`models.py:151-153`); trainable = a linear projector + `LayerNorm→Linear(d,256)` head.
Pooling is **mean over patch states** (`models.py:441`).

Pretraining variants, all reachable: `use_ae` reconstruction (`route_b_ae_reconstruct.py`),
`mask_level="latent"` TimeMAE token, `mask_level="embed"` SimMIM token,
`mask_level="discrete"` VQ codebook with `--vocab-size 32` (`route_b_ae_masked.py:400-405`).
`mask_level="none"` is only exercised by tests.

Duplicated standalone versions live in `leakage_localizer/`: `exp_v1_timellm_gpt2.py` (adds a
Time-LLM `ReprogrammingLayer`, cross-attention over 1,000 learned vocab prototypes,
`:45,:174`), `exp_v1_gpt2_ce.py` (patch 25), `exp_v1_e1_scratch.py` (0.306 M-param scratch
encoder).

**Results — all SMOKE-grade for key recovery, because rank was never computed.** Every
artifact records `chance_ce 5.545` and attack accuracy 0.0025–0.0055; `docs/roadmap-break-ascad.md:169`
headers the section "CE + accuracy ONLY". Notable: `reprogram-p16s8-e40-train.json:335`
CE 5.5769 / acc 0.004 with **val accuracy collapsing to 0 after epoch 11** at 50.55 M trainable
params against 124.44 M frozen. The one genuine positive in this family is not a classifier:
`exp_v1_a2_tokens_to_HW_reader` reaches **acc 0.28** on a 9-way Hamming-weight target, proving
the tokens *do* carry r and C — the 256-way head is where it dies.

## 2. Family B — patch → trainable transformer → attention / bilinear / flatten

`LLM/patch_transformer_sca.py`, three strategies: `trainable` (`:126`, 4-layer bidirectional
`nn.TransformerEncoder`, d_model 128, 4 heads, learnable `pos_embed` `:158`), `gpt2-fixed`
(`:238`, GPT-2 frozen except `LayerNorm` and `wpe` — GPT4TS style), `cnn-transformer`
(`:401`, Conv1d 11/7/5 tokenizer + encoder). Pooling: `AttentionPooling` `:67`,
`BilinearInteractionPooling` `:91`, `flatten` `:226-227`.

| Variant | Target | Metric | Grade |
|---|---|---|---|
| trainable, attention, v1 b2, 50 k/10 k | unmasked S\* | atk CE 5.5467, acc 0.0039, `final_rank` **168** | RESULT (fail) |
| trainable, bilinear, same budget | S\* | CE 5.5471, acc 0.0038, `final_rank` **182** | RESULT (fail) |
| earlier attn / bilinear / cnn-transformer | S\* | `final_rank` **230 / 94 / 233** | RESULT (fail) |
| `A1-v1-flatten`, 20 k/5 k, 4.32 M | S\* | CE 5.5476, acc 0.0032, `final_rank` **245** | RESULT (fail) |
| `A2-v2-alpha`, 20 k/2 k, stopped at e6 | `alpha_mask` | atk CE **5.54696**, acc **0.0030** vs 1/255 bar | RESULT (fail) |
| `patch_transformer_v2_alpha`, p32/s16 | `alpha_mask` | CE 5.5468, acc 0.0060 | RESULT (fail) |
| `cnn_transformer_v2_alpha` | `alpha_mask` | CE 5.5460, acc 0.0030 | RESULT (fail) |

Sources: `out/patch_transformer_v1_results.json:205,210,4213`,
`out/A1-v1-flatten.json:140,2143`, `out/A2-v2-alpha.json:115,120,122`,
`docs/lab-log/patch_transformer_trainable_v1_{attn,bilinear}.json`.

**Two dead ends in this file, verified:** `MultiLabelWrapper` (`:487`) is defined and never
instantiated anywhere in the repo; `--use-lora` builds `self.lora_modules` (`:296,:302`) but
`forward()` never calls them, so the flag changes nothing. `LLM/config.py` is entirely
unimported.

**Doc/artifact conflict:** `out/A-verdict.md:60-62` and `T020:104` say A1 "crashed, never
produced a number", but `out/A1-v1-flatten.json` carries 20 epochs of full metrics. The prose
is stale. Separately `T020:156` cites "bilinear rank = 94" while the 09-26 artifact says 182 —
two runs reported as one.

## 3. Family C — resolution-preserving and structured readouts

`LLM/soft_window.py`: conv stem with total stride 16 (patch-equivalent tokens, `:83-89`),
readout ∈ {`gate`, `max`, `both`, `topk`} (`:106-117`), multi-task `--targets` in GPAM style
(`:130-132`). `LLM/pair_words.py`: `patchify` `:50` → `PatchFeatures` (per-patch MLP over the
patch plus 6 amplitude statistics `:62`, conv smoothing, LayerNorm) → `PairWords` ordered
centred pair atoms, `separable` `:134` vs full N×N `:139`.

| Variant | Target | Result | Grade |
|---|---|---|---|
| `SW-beta` whole-trace learned gate | v2 β | atk acc **0.0354** ≈ 9.1× chance, `fit_acc_4096` 0.0564, no rank | RESULT (above chance, incomplete) |
| `SW-alpha` / `SW3-alpha-topk` / `SW-alpha-sharp` / `SW2-beta-both` | v2 | acc 0.0042 / 0.0042 / 0.0036 / 0.0032 | RESULT (fail) |
| `SW-alpha` gate pathology | — | gate **collapsed onto padded sample 14992**, entropy −0.0, `eff_positions` 1.0 | mechanism, not accuracy |
| PW / PW2 / PW3-lin / PW4-long | v1 b2 S\* | `rank_final` 199 / 190 / 228 / 220; `control_rank_const_trace` 196 / 190 / 80 / 254 | RESULT — **PW and PW2 rank equal the constant-trace control: zero trace dependence** |
| `PW-probe-v1-b2` product_only | v1 b2 S\* | `rank_final` **0**, `traces_to_rank0` **120** | **ORACLE** — hand-placed p18-22 × p64-67 (`A-followup:205-207`) |

The `SW-alpha` padding collapse and the PW-vs-constant-trace control are the two most
informative negative results in the whole survey: one shows the learned gate finding nothing
better than a padded sample, the other shows a pair-interaction head scoring the same as a
model given a **single constant trace**.

## 4. Family D — sliding-window proposers (the non-patch route)

`leakage_localizer/model.py`: `BandAttentionProposer` conv-score → per-band softmax (`:327`),
`TwoOrderWindows` (`:436`), `ThreeOrderWindows` (`:463`), plus
`SoftWindowBooleanSCA` / `DirectWindowSHead` / `AttentionBooleanSCA` / `GlobalThetaSCA` built
via `build_poi_searcher:534` — of which only `AttentionBooleanSCA` is reachable from
`ASCAD_train_models.py:333`.

This family keeps per-sample resolution and composes the two windows **multiplicatively**
(centred product), which is what second-order analysis needs. `exp_2o.py` claims
`rank_final 0` on fixed-key byte 2 for desync 0 and 100, three seeds — **but the rank table is
printed, never persisted; six `.pth` checkpoints, no JSON** (roadmap gap 1). `exp_3o.py` is a
clean failure: `FAIL_FLAT`, corr ≈ 0.003 in fair/synth/aux. `exp_3o_wc.py` `PASS_LOCALIZE`
means the windows landed on POIs, **not** that the key was recovered. `exp_a2_tokens_to_S.py`
FAIL_CHANCE 0.004 vs `exp_a2_tokens_to_HW.py` PASS 0.28 — same tokens, different target
granularity.

## 5. Family E — GPAM grouped attention

`scaaml.models.get_gpam_model` (`scaaml/models/gpam.py:586`): **non-overlapping** blocks via
`Reshape((steps, patch_size))` (`:649`) — no patch mean — rescale to [−1,1], `Dense(192)`,
3× GAU blocks with **RoPE** attention (`:664,:136`), concatenated intermediate outputs,
`GlobalAveragePooling1D` (`:697`), one independent head per label (`_make_head:422`).
34-head mode = `rm`(α, 256) + `rout`(β, 256) + 16×`c_b`(masked S, 256) + 16×`p_b`(π, 16)
(`GPAM/train_gpam_raw_full.py:157`); raw patch 400 over 1 M samples → 2,500 blocks.
`run_ascadv2_gpam_baseline.py:228` is the single-head version.
`train_gpam_ascadv2_budget.py` is **disabled**: `main()` raises `SystemExit(GE_BROKEN)`
(`:183`) because `compute_ge()` cannot produce valid ASCADv2 guessing entropy (`:129-130`).

| Run | Metric | Grade |
|---|---|---|
| GPAM crop, 4 heads, 40 k traces, 8 epochs, patch 150, v2 b0 | `rm`(α) **92.95 %** holdout / 92.35 % attack, `mean_rank` 0.0825, top-5 100 % (chance 0.3906) | **RESULT, worst-case** — mask labels on, so not a break (`C-talking-points:19-21`) |
| same, `rout`(β) / `c`(masked S) / `p`(π) | 2.75 % / **0.65 %, mean_rank 126.5 vs null 127.5 = at chance** / 8.60 % (prior 7.05 %) | RESULT — β and masked-S fail; π marginal |
| `ge_fixed_key_sim` @ 80 traces | rank 35 (paper claims < 1); the control reached 10 | noise band, not a break |

This is the project's **only reproduction-grade GPAM number**, and it is also the cleanest
refutation of "the label is unreadable": same file, same label, different front end.

## 6. Family F — contrast baselines that are not patch transformers

`TimesNet/` uses no patches — `TimesBlock` reshapes by period (`model.py:83`) and pools to
`pool_segments` bins; variants `baseline / ascad_head / pooled / ascad_head_pooled / multi /
ascad_head_multi` (`build_timesnet_model:392`), with `TimesNetMultiTask` emitting the 34
`MULTILABEL_KEYS` heads through a MultiResNet-style `_BranchHead` (`ASCAD_train_models.py:166`).
`Scoop/` is **not a pipeline at all** — `Scoop(Optimizer)`, a second-order optimizer
(`scoop/scoop.py:21`). `GNN/segment_graph.py` is the sibling idea to patching:
`find_segments_v2` local-minima bounds → each segment resampled to a fixed 12-dim vector →
**one graph node**, padded to 32 nodes; classifiers `GCN / GAT / GIN / GraphSAGE /
SegmentGraphSAGEClassifier` (`classifiers.py:492`).

| Model | Result | Grade |
|---|---|---|
| `timesnet_ascad_head` v1 b2 | `rank_final` 0.0 / 2.0 / 1.5 over seeds, `traces_to_rank0` 180,150 / 5,930,4750 / 9,230,2360 | RESULT (v1 win) |
| CNN baseline v1 d0 | `rank_final` **0**, `traces_to_rank0` **190**, `ge_final` 0.0 | RESULT |
| TimesNet v2 multi, ×34 heads | mean rank 141.12 → 124.25 ± 81.1 (e60) | RESULT (v2 fail) |
| TimesNetPooled b4 black-box | `ge_last` 3.65, best GE 2.9 | **ambiguous** — inside the frequency-prior neighbourhood, §8 |
| MultiResNet 34-head `sub_avg` | **all 16 bytes `rank_final` 0**; π top-1 67.03 %; α 100.00 %, β 5.84 %, C 0.69 % (recomputed 2026-10-04) | **UNCONTROLLED** — no playback/AOD control persisted; the same tie artifact is documented at `T-019:138` |
| Single-head sub-avg TimesNet v2 b0–b4 | min key rank 7 / 88 / 21 / **1** / 15 | RESULT (close, no rank 0) |
| Scoop repro-v2 b15 | `ge_last` 0.0, `traces_to_ge0` 100001 | not a strict break (1/16 bytes, seed 0, contested split) |
| GNN segment-graph Plan B / Plan C | `rank_final` 7 / 35 (Plan C weighted A) | RESULT |

## 7. Why the patch family failed

**The measurement that matters** (`out/A-followup-where-alpha-leaks.md:78-94`, artifacts
`out/BANDCTL16-contiguous.json`, `out/RESOL-contig16-b{2,4,8}.json`). Same 1,600 samples,
same label, same learner — only the representation changes:

| Representation of the same 1,600 samples | features | top-1 on α | × chance |
|---|---|---|---|
| raw, one feature per sample | 1600 | **0.8603** | 219× |
| mean of adjacent 2 | 800 | **0.0083** | 2.1× |
| mean of adjacent 4 | 400 | 0.0065 | 1.7× |
| mean of adjacent 8 | 200 | 0.0090 | 2.3× |

> "Averaging two adjacent samples destroys 99 % of what the model had extracted."
> — `A-followup-where-alpha-leaks.md:89`

A patch **is** a group of adjacent samples that gets projected jointly, and mean-pooling over
patches averages again. `A-followup:91-94` draws the conclusion: any front end whose first move
reduces temporal resolution — strided pooling, patch means, average subsampling — discards α
before a learner can use it.

**The pooling argument.** `docs/notes/llm-sca-diagnosis-and-plan.md:20` — mean pooling with a
linear head "is a sum of independent per-patch classifiers. Since no single patch leaks S, each
contributes uniform noise. Sum of uniform = uniform", escalated at `:35` to "mean pooling is
mathematically incompatible with second-order attacks… not a hyperparameter problem — it's an
architectural impossibility." `T020:191-193` records this as HELD & REINFORCED.

**An unresolved contradiction in the notes.** `docs/notes/trace-signal-anatomy.md:95` frames the
problem as *location* ("Spatial Dilution Trap… 81.6 %–89.3 % contains no signal"; 39.4× at 100
samples vs 1.4× whole-trace). `A-followup:68-73` refutes that framing directly: "Localization is
not the main obstacle; pooling is" — contiguous 0–1600 reaches 0.8603 and even **random** 16-band
draws reach 0.5622. The note (mtime 09-24) was never updated after the 09-25 correction. Treat
the "dilution" framing as superseded.

**And the retraction that follows from all this.** `out/A-verdict.md:3-20` strikes its own
earlier sentence "this family cannot see a leak with SNR about 2", on three grounds: SNR(α) =
1.986 confirmed and saved as `out/SNR-alpha-curve.npy`; GPAM reads the same label on the same
file at 92.95 %; and the pooling result above. `T020:33`: "the 'drop the family' verdict is
retracted."

## 7b. Correction to §7: GPAM also attends, and also groups samples

§7's mechanism claim was overstated in this file and in conversation. A patch transformer's
`Linear(16 → d_model)` does **not** average its 16 samples — every sample stays a separate
input, exactly as GPAM's `Reshape((steps, 400)) → Dense(192)` keeps all 400. GPAM even ends with
`GlobalAveragePooling1D` (`scaaml/models/gpam.py:697`), over block outputs. And GPAM's tokens are
**far coarser** than ours — 150 or 400 samples per token versus 16. If coarse grouping were the
killer, GPAM should be worse than A2. It is 98.34 % and A2 is chance.

So the RESOL result and the GPAM result are in tension, and the honest reading is:

| front end | outcome |
|---|---|
| explicit mean over adjacent samples (RESOL b2/b4/b8) | fatal — 0.86 → 0.008 |
| group samples into blocks, project, attend (GPAM, 150–400/token) | **works** — 92.95 % crop, 98.34 % raw |
| group into 16-sample patches, project, attend, linear head (our models) | fails — at chance |

Grouping is therefore **not** the discriminator; averaging is. The remaining candidates, in
order of how much weight the evidence gives them, are **training length**, token count, head
depth, and multi-scale readout — see §7c.

## 7c. The patch family was stopped at the epoch GPAM needed to work

Every patch-transformer run on a label that is *known readable* (α, extracted v2) was
terminated at ≤ 7 epochs:

| run | configured | ran | best epoch | atk acc on α |
|---|---|---|---|---|
| `A2-v2-alpha` | 12, patience 4 | **6 (early-stopped)** | 2 | 0.0030 |
| `patch_transformer_v2_alpha` | 12, patience 4 | 7 | 3 | 0.0060 |
| `cnn_transformer_v2_alpha` | 12, patience 4 | ~5 | 5 | 0.0030 |

Tonight's GPAM raw probe was **at chance for five epochs** (`val rm` 0.098–0.488 %), broke
through at epoch 6 (17.87 %), and reached 98.34 % by epoch 8. A2's early-stop rule — patience 4
against a best epoch of 2 — would have killed it at epoch 6 of a GPAM-shaped curve. **We do not
have evidence that the patch architecture cannot learn α; we have evidence that it had not
learned it within 6–7 epochs, on a target where 6–8 epochs is exactly the transition window.**

The other differences are real but untested here: GPAM uses 100 tokens on the 15 k crop against
A2's 1,874 (α leaks at one place, ~sample 867, so this is search difficulty); a 3-Dense head per
label against our single `Linear(d, 256)`; concatenated intermediate block outputs against
final-layer-only; RoPE against additive learned `pos_embed`.

**Falsification test, not yet run:** A2's exact architecture, α target, early stopping disabled,
30+ epochs. A2 cost 49 s/epoch, so this is ~25 min of GPU — cheap enough that "the patch family
is dead" should not be asserted until it exists. Blocked behind the 83 h campaign.

## 8. The three traps that produced false optimism

**8a. Oracle trap.** Any band chosen *using* the labels is a characterisation, not an attack.
`T020:59-60`: "Band choice here used ground-truth labels via SNR, so it is an oracle for
understanding the data, not an attack." The 240× / 93.95 % headline (`out/BANDCTL16-snr.json`)
is exactly that. Same category: `PW-probe` hand-placed windows (`A-followup:205-207`), the v1
2O oracle gate on 156×517 (`v1-poi-faq.md:36`), v2 3O Loss 2/3 with known π
(`roadmap:249-251`), and the T-009 ground-truth one-hot recombination. GPAM's 92.95 % is not an
oracle but *is* worst-case: it is trained on mask labels.

**8b. Label-frequency prior.** `docs/notes/ascadv2-label-frequency.md:5` — a model that
**ignores the power trace** and only memorises training frequencies puts the true key at
**rank 3** on byte 4 after 300,000 traces. `:167` says plainly that this is "the same
neighborhood as pooled TimesNet + RAdam on byte 4 (best GE 2.9, last GE 3.65)… It proves the
cheat **can** get there." So every byte-4 GE near 3 is ambiguous until AOD-GE is run. v1 has the
same hazard: `todo.md:94` records that **byte 7 hits rank 0 with no leakage at all**.
`A-verdict.md:56` names the same failure mode for A2: train CE below chance with flat validation
is "the head fitting the label marginal, not leakage."
Note `C-talking-points:107-123` reports AOD-GE artifacts *do* exist on disk
(`Scoop/artifacts/ascadv2_repro_v2/metrics/alt_metrics_ge_lt10.json`, untracked) and that both
runs go to **AOD-GE 255**, with byte 15's classical GE curve unchanged when the first `Linear`
layer is zeroed — i.e. that curve does not use the trace. That contradicts `roadmap:330`
("AOD-GE is unimplemented") and weakens `roadmap:68`.

**8c. Smoke checks that look like wins.**

| Artifact | Why it is not a result |
|---|---|
| `out/WIRING.json` `verdict: "ABOVE_CHANCE"` | 256 profiling / 64 attack traces, 1 epoch, d_model 32, 1 layer, `device: cpu`. `final_attack.acc = 0.0`, `final_rank: null`. Only the 256-trace *profiling* accuracy (0.0195) tripped the script's 2×-chance test. Disowned at `A-verdict.md:113`: "plumbing, not a result… Cite nothing from WIRING" |
| `out/OVERFIT-flatten.json` `ABOVE_CHANCE` | 512 traces, 200 epochs; attack acc 0.0039 = exactly chance; `final_rank` 163 on 512 traces. A memorisation check |
| `GPAM/artifacts/C-dryrun.json` | 512 traces / 1 epoch; `rm` acc 0.0 % |
| T-019 512-trace 34-head "memorisation" (`rm` 94.3 %) | **Withdrawn** — the label-shuffled control scored `rm` **98.05 %**, better than real labels (`T019:63`) |

## 9. Register of withdrawn, unbacked, or contradicted claims

| Claim | Status | Source |
|---|---|---|
| "This family cannot see a leak with SNR ≈ 2" | **retracted** same night | `A-verdict.md:3-20`, `T020:33` |
| 34-head `p` is intrinsically hard | **withdrawn** by shuffled control | `T019:63` |
| "monotone decline" of raw val loss | **a running-average artifact** | `T019:25` |
| Raw 1 M result generalises | **DO NOT GENERALISE at 4 k traces** | `T019:5-14` |
| 2026-09-25 coarse first-order scan | **invalidated** — 15/16 hypothesis bytes wrong | `T027:77-80` |
| 2O `rank_final 0` × 3 seeds | real but **no persisted JSON** (gap 1) | `roadmap:79-80` |
| GPAM GPU smoke succeeded | `[U]` — 0-byte log | `roadmap` gap 4 |
| `timesnet_lr1e-3` numbers | **DO-NOT-CITE** | `roadmap:12` |
| "rank_final@1k = 102", "Scoop-slice 1k = 17" | doc-only `[U]`, no artifact | `roadmap:224` |
| Plan C rank ≈ 5 | `[U]` no artifact; contradicts `llm-sca-diagnosis:31` "✅ Rank ~5" | `roadmap:144` |
| MLPBest "✅ Rank 0" | contradicted: rank 183/226/212, "v1 FAIL (weak)" | `llm-sca-diagnosis:28` vs `roadmap:117` |
| A1 crashed with no number | contradicted by its own artifact | §2 |
| additive-access rank 233 | artifact says **238** | `A-followup:191` vs `out/PW-probe-v1-b2.json` |
| sub-avg's gain is information gain | **voided**: "subtracting the profiling mean trace is a provable no-op here" | `A-followup:97-104`, contradicting `trace-signal-anatomy.md:97` and `T-024:16` |
| MultiResNet all-16 `rank_final` 0 | **uncontrolled** — heads at chance, no playback/AOD control persisted | `roadmap:67`, `T-024:108-113` |

## 10. The result that arrived while this survey was being written

**GPAM on raw 1 M-sample ASCADv2 transfers: `rm`(= α) 98.340 % held-out on a disjoint shard.**
This is the project's strongest raw-data number and it supersedes every interim reading in this
section's history.

Gate: `GPAM/artifacts/C-raw-generalization-test.py` — 4 heads, 40 k raw traces from shard 1, 8
epochs, lr 1e-3, evaluated on shard 2 rows 0–1023, launch bar `rm > 2 %` (5× chance).
Log `GPAM/artifacts/logs_raw/probe40k_20260929_202123.log`:

| epoch | train | val (shard 2) | gap vs 19.4081 | `val rm` |
|---|---|---|---|---|
| 1–5 | 19.4154 → 19.3623 | 19.4079 → 19.4944 | +0.0002 → **−0.0863** | 0.293 / 0.488 / 0.098 / 0.391 / 0.293 % |
| 6 | 18.7306 | 17.0840 | +2.3241 | **17.871 %** |
| 7 | 15.5788 | 14.6365 | +4.7717 | 76.270 % |
| 8 | 13.9707 | **13.4763** | **+5.9318** | **98.340 %** |

`PROBE GATE VERDICT: PASSES` at 21:23:14; the chain launched the ~83 h 34-head raw campaign
(`GPAM/train_gpam_raw_full.py`, pid 1083690) two seconds later. The gate is fail-closed
(`gpam_probe_then_campaign.sh:35,:39`: empty verdict, `FAILS`, or non-zero probe exit ⇒ no
launch).

**Controls run on this number.** Train/eval disjointness was checked on the exact slices used:
all 40,000 train and 1,024 eval `(key, plaintext, 38-byte masks)` signatures are unique and the
**overlap is 0**, so these are distinct acquisitions, not re-labelled copies. `alpha_mask` spans
255 distinct values in train and 251 in eval. The trace-free frequency prior caps a
label-marginal model at ~0.48 %, so 98.34 % cannot be the §8b trap. Not checked: raw waveform
equality between metadata-distinct rows — that would be a dataset-generation bug, not a split
bug.

**What it does not mean.** Only `rm` learned. At epoch 8, `rout`(β) 0.879 %, `c_0` 0.391 %, and
`p_0` 6.84 % against a 6.25 % chance bar are all flat — the same α-only hierarchy the 15 k crop
run showed (`rm` 92.95 %, `rout` 2.75 %, `c` 0.65 %, `p` 8.60 %), now on raw traces and *better*
(98.34 % vs 92.95 %). α is a **mask label**: this is worst-case / First Threat by §4's own
definition, not a key recovery and not eligible as a strict break. The paper's Table 11 claim
(rank 0 in ~80 traces) is still not reproduced here. One seed, one shard pair, 40 k of the
paper's 640 k.

**Methodological record, because it matters for how to read GPAM curves.** Five epochs in, this
run looked dead — `val rm` oscillating at chance and the held-out loss drifting *worse* than
chance (gap −0.0863 at epoch 5), and an interim note here said "the gate is not passing". That
reading was wrong and was reversed by a 37× jump one epoch later. `T019:12` already warns "do
not over-read this"; this is the strongest available demonstration of it. The related header
claim at `T019:5-14`, "DO NOT GENERALISE at 4 k traces", is **not** contradicted — it was about
4 k traces, and this is 40 k. What is new is that the transfer appears somewhere between those
two budgets, which makes the budget the interesting variable.

## 11. What the project proposes instead

Not abandonment — **re-posing**. `T020:60-63`: "The unsolved question is finding the live bands
*without* labels — attention, hierarchical/Coarse-to-fine gating, or sliding-window compression
to a short sequence that keeps per-sample detail." Concretely `todo.md:83` (T31): "**Resolution-
preserving front end.** Same 1,600 samples, but a model that never averages adjacent samples:
strided conv with no pooling, or raw-value patches plus late attentive compression", plus
`todo.md:82` (T30) unsupervised band selection, and `T025:63` "connect active patch mask to
downstream POI windowing / transformer attention". The frozen-GPT-2/mean-pool and 50 k-budget
scratch-transformer branches stay **DEAD** (`roadmap:279,:282`) — though `roadmap:189` still
attributes the family's failure to "the SNR(S) ≈ 0.02 diagnosis rather than a bug", which is the
framing §7 refutes.

---

### Provenance

Read-only survey. Sources: `LLM/`, `leakage_localizer/`, `GPAM/`, `TimesNet/`, `Scoop/`, `GNN/`,
`utils/find_segments.py`, `docs/roadmap-break-ascad.md`, `docs/notes/{llm-sca-diagnosis-and-plan,
ascadv2-label-frequency,trace-signal-anatomy,v1-poi-faq,2o-localizer-vs-gpam}.md`,
`docs/tasks/{T016,T017,T019,T020,T022,T024,T025,T026,T027}*.md`, `docs/2026-09-23/out/*`
(55 JSON artifacts), `docs/lab-log/todo.md`. Spot-verified by hand: the §7 collapse table,
`C-gpam-eval.json:53`, `A-verdict.md:3-20`, the three dead-code claims, and the live campaign
logs. No file outside `docs/notes/` and the task index was modified.
