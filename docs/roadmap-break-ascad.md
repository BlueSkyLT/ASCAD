# Roadmap — breaking ASCADv1 and ASCADv2

Status date: **2026-09-23**. Living inventory of every model, variation and probe this repo has
surveyed, implemented or actually run against the two masked-AES ASCAD datasets, and where each one
stands.

Scope: `leakage_localizer/`, `LLM/`, `Scoop/`, `GPAM/`, `TimesNet/`, `GNN/`, the CNN/MLP baselines in
`ASCAD_train_models.py` / `benchmark_full_fixedkey_gpu.py`, and the support work they depend on.

This file is the *tree of what exists and its verdict*. Per-session narrative stays in
[`docs/tasks/`](./tasks/INDEX.md); reference facts stay in [`docs/notes/`](./notes/README.md).

---

## 0. How to read this

### Status ladder

Each node carries the **highest** rung it has reached. The rungs are deliberately not cumulative in
the last three: a model can be `win@b2` and still be `v2 FAIL`.

| Tag | Meaning |
|-----|---------|
| `surveyed` | Read about it (paper / survey). No code of ours. |
| `implemented` | Code exists in this repo. No evidence it was ever run. |
| `ran@bN` | Trained **and** evaluated on pilot byte *N*. |
| `win@bN` | On that byte: `rank_final = 0` (v1) or `GE -> 0` i.e. `traces_to_ge0` non-null (v2). |
| `multi-seed` | ≥ 2 seeds on the same recipe, results reported as mean/standalone per seed. |
| `all-16` | Swept bytes 0–15. |
| `v1 OK` / `v2 FAIL` | Dataset-level verdict for that branch. |
| `DEAD` | Recorded dead end — do not re-run without a new idea. |
| `BLOCKED` | Waiting on data, a decision, or an unimplemented prerequisite. |

### Evidence grade

| Tag | Meaning |
|-----|---------|
| `A` | **Artifact** — a JSON/NPY/log on disk that I can point at. |
| `L` | **Log** — the number appears only in a task file or plan markdown; no persisted result. |
| `U` | **Unverified / disputed** — claim conflicts with an artifact or nothing backs it. |

### Two naming traps

1. **The pilot byte differs by dataset.** ASCADv1 work targets **byte 2** (the fixed-key 700-sample
   POI crop leaks byte 2); ASCADv2 work targets **byte 4**. When a v1 note says "byte 4", it is wrong.
2. **"v1/v2" is used for two different things.** *ASCADv1/v2* = the datasets. *Scoop repro v1/v2* =
   our two reproduction configs (`ascadv2_repro` = fixed width-3000 baseline, `ascadv2_repro_v2` =
   paper-narrow width-256 + searched depth). Both Scoop repros attack the **ASCADv2** dataset.

### What counts as "broken"

`docs/survey-ascadv2-dl-2022-2026.md:23` — a **strict** ASCADv2 break needs: full public 15k (or full
raw), affine **and** shuffle **on**, model trained on an intermediate, key recovered via GE, **no mask
labels**. Section II.6: public non-worst-case full-15k ASCADv2 "remains unbroken under AOD".
For ASCADv1 the bar is simply `rank_final = 0` on a byte with a fair profiling/attack split
(`docs/notes/metrics.md`).

---

## 1. Scoreboard — what we can actually claim today

| Branch | Dataset | Best measured result | Rungs reached | Break? | Evid. |
|--------|---------|----------------------|---------------|--------|-------|
| 2O window localizer | v1 desync0 + desync100 | `rank_final = 0`, byte 2, seeds 0/1/2 | `ran@b2` `win@b2` `multi-seed` | **v1 OK** (2O CPA, no mask labels) | `L` (ckpt `A`, rank table not persisted) |
| TimesNet `ascad_head` | v1 d0/d50/d100 | `rank_final` mean **0.0 / 2.0 / 1.5** | `ran@b2` `win@b2` `multi-seed` | **v1 OK** (single byte) | `A` |
| CNN baseline (official recipe) | v1 desync0 | `rank_final = 0`, `traces_to_rank0 = 190` | `ran@b2` `win@b2` | **v1 OK** (byte 2, seed 0) | `A` |
| MultiResNet (34-Head, `sub_avg`) | v2 extracted 15k | `rank_final = 0` across **ALL 16 bytes** (z = +3.89σ to +6.86σ @1k traces; shuffled null = 143.8) | `ran@all-16` `win@all-16` `all-16` | **v2 multi-task OK** (ANSSI reference break reproduced) | `A` |
| Scoop repro v2, byte 15 | v2 extracted 15k | `ge_last = 0.0`, `traces_to_ge0 = 100001` | `ran@b15` `win@b15` `all-16` | **not a break** — 1/16 bytes, seed 0, contested re-split | `A` |
| Everything else on v2 (black-box) | v2 | best `ge_last` 2.75–3.65 on b4, **never** `GE -> 0` | `ran@b4` | **v2 FAIL** (black-box) | `A` |

**MultiResNet with mean-waveform subtraction (`sub_avg`) breaks all 16 AES bytes to Rank 0 on ASCADv2** under the official multi-task supervision setting (8.6 min training, 50k traces). On ASCADv1, three genuine wins exist for byte 2. Strict black-box (no mask labels) break on v2 remains open.


---

## 2. Support layer (prerequisites the attacks sit on)

```
Support
├── Datasets on disk
│   ├── ASCADv1 fixed key ....... ASCAD.h5 (desync0) + desync50 + desync100 (44 MB each), ATMega8515_raw_traces.h5 (5.9 GB) ......... [A][done]
│   ├── ASCADv1 variable key .... 3 × 418 MB + raw 70 GB (unused by any attack above) ................................................. [A][done]
│   ├── ASCADv2 public extracted  ascadv2-extracted.h5 (7.4 GB) — what every v2 run above used ....................................... [A][done]
│   └── ASCADv2 raw 800k .......... shard 8 only (93 GiB, complete). shard 1 = 18 GiB + a 21 GiB .part; shards 2–4 .part only;
│                                   shards 5–7 = 0 bytes. Monitor's last tick 2026-09-23: 88.1 GB of ~220 GB (decimal)
│                                   → hard blocker for GPAM and for any full-raw strict-break claim .................. [A][BLOCKED]
├── Labeled POI HDF5 generation ... ASCAD_generate.py (upstream, outputs present) ...................................................... [done]
├── Metrics vocabulary ............ docs/notes/metrics.md — rank_final, traces_to_rank0, ge_final, success_rate_final;
│                                   fair profiling vs attack split enforced here ....................................................... [done]
├── Trace/segment understanding ... visualize/ (explore_traces.ipynb, snr_poi_xor.ipynb executed),
│                                   find_segments_v2 = a function in utils/find_segments.py:90
│                                   (27.7 segments @desync100; ~516 on v2) ................................................... [A][done]
├── v1 segment value map .......... 5 valued spans = 133/700 ≈ 19% of the crop carries data-dependent value;
│                                   visualize/segment_value_map.ipynb + PNGs ................................................... [A][done]
├── Soft windows α/β/C (T-013) ... design locked in docs/superpowers/plans/2026-08-17-poi-searcher.md,
│                                   zero artifacts ....................................................... [planned][L][parked]
├── ASCADv2 literature base ....... docs/survey-ascadv2-dl-2022-2026.md (15-work master table + Part II critique) ................ [done]
└── Environment ................... py10 conda env, RTX 5090, CUDA restored 2026-09-22 (T-021, kernel/DKMS mismatch fixed) ..... [done]
```

---

## 3. ASCADv1 (1st-order masked ATmega8515) — attack tree

**Structural finding that governs this whole branch:** in v1 the *unmasked* intermediate `S` has
essentially no direct leakage (SNR(S) ≈ 0.02), while the two mask shares leak strongly and locally —
`r` at t≈156 (SNR 1.32) and `C` at t≈517 (SNR 6.38). Any model that tries to learn `S` directly from
the crop is fighting the physics, not its hyperparameters. Stated in `docs/notes/v1-poi-faq.md`,
`docs/notes/llm-sca-diagnosis-and-plan.md`, plan §0/§1.4, and re-measured in T-018 (1.33 / 6.59).
Caveat: SNR(S) itself is `L` — `exp_2o.py` computes it but never saves it.

```
ASCADv1
├── Classic baselines (byte 2, eval-only, seed 0 only)
│   ├── cnn_best ....... ran@b2 win@b2 v1 OK ......... desync0 rank_final 0 / traces_to_rank0 190; desync50 64; desync100 2
│   │                                              [A] .../ASCAD_trained_models/benchmark_runs_cnn_mlp_baseline/summary_fixedkey_cnn_mlp_baseline.json
│   └── mlp_best ....... ran@b2 v1 FAIL (weak) ....... rank_final 183 / 226 / 212 (d0/d50/d100)
│                                              [A] same file. 183 is the 75-epoch budget; the published ASCAD MLP succeeds at 200+ epochs.
│                                                  No training log, no multi-seed variance (T-006 open).
├── TimesNet family (T-004) — byte 2, 10 000 eval traces, RAdam, 150 e unless noted
│   ├── timesnet (vanilla, emb k=3, ~8.1M) ... ran@b2 multi-seed(0,1) v1 FAIL
│   │     rank_final mean 42.5 / 19.5 / 179.5 (d0/d50/d100), traces_to_rank0 null, train CE 0.42/0.48/0.36 = label overfit
│   │     [A] .../benchmark_runs/summary_full_fixedkey_gpu.json (+ aggregate_…)
│   ├── timesnet_ascad_head (emb k=11, 1024→1024→256, ~26.6M) ... ran@b2 win@b2 multi-seed(0,1) v1 OK  ★ best v1 classifier
│   │     rank_final mean 0.0 / 2.0 / 1.5 · per-seed traces_to_rank0 180,150 / 5930,4750 / 9230,2360 · frac_reached_rank0 1.0 (d0)
│   │     [A] same summary JSON + 6 checkpoints
│   ├── LR 1e-4, 75 e ... ran@b2 partial — head d0 rank 1.5 (ttr0 3230); vanilla d0 22; head d50 87, d100 103.5 → 75 e too short  [A]
│   ├── LR 1e-2, 75 e, seed 0 ... ran@b2 v1 FAIL — head 232–234 at CE 5.544 (never learned); vanilla d100 234 at CE 0.141        [A]
│   ├── LR 1e-3, mixed e50/e75 ... ran@b2 DO-NOT-CITE — 6 rows status=error ("Checkpoint not found"); T-004 says one rank-0 seed,
│   │     artifacts show two (s0 e75 ttr0 3720, s2 e50 ttr0 1810)  .............................................. [A][U]
│   └── timesnet_pooled / ascad_head_pooled on v1 ... implemented, never wired into ASCAD_train_models.is_timesnet_network
│         or benchmark_full_fixedkey_gpu.py:185 (they exist only in TimesNet/model.py:281,367)  ............... [implemented]
├── GNN family (T-003, last edited 2026-07-24 — byte 2, desync0, 75 e)
│   ├── sample-node K700 GCN/GAT .... ran@b2 DEAD (non-learning; diagnosed, T17 backlog)
│   ├── sample-node GIN ............. ran@b2 v1 FAIL, seed lottery (also had an AMP NaN bug, fixed)
│   ├── GraphSAGE sample-node ....... ran@b2 win@b2 multi-seed(0,1) × 3 desyncs — desync0 rank 0–1, ttr0 4.2–5.7k; desync100 rank 9–11
│   │     [A] .../ASCAD_trained_models/gnn_runs/summary_gnn.json  ⚠ file mixes a 2-epoch wiring row (rank 43) into the 75-e index
│   ├── Plan A (GAT/GIN arch-fix sweep) ... ran multi-seed v1 FAIL — 18 runs, rank 182–240
│   │     [A] gnn_runs/gat_gin_fix/summary_full_fixedkey_gpu.json
│   ├── Plan B segment-as-node (N=32/F=12, path edges) ... ran@b2 — CE 5.482, rank_final 7  ★ best GNN so far
│   │     [A] gnn_runs/segment_node/summary_seg_graphsage_desync0_e75.json + logs/seg_graphsage_e75_v1*.log. Never desync50/100, never v2.
│   ├── Plan C SNR-product nodes ... ran@b2 — binary path rank_final 8; weighted product A CE 5.498, rank_final 35 (curve best 8)
│   │     [A] gnn_runs/segment_node_snr/summary_seg_graphsage_snr_{path,weighted}_desync0_e75.json
│   │     Verdict: better CE ≠ better rank; Plan C never beat Plan B. (T-020's "Plan C rank ~5" has no artifact.) [U]
│   ├── relation/combine edge modes . ran@b2 — 2-e smokes, checkpoints only, no rank summary .................. [implemented→ran]
│   └── multi-Z SNR nodes ........... planned only — checkbox in docs/lab-log/plan-snr-xor-adjacency.md §4, no code  [planned]
├── Leakage localizer (T-014) — the branch that actually breaks v1, byte 2, 8k prof / 2k attack
│   ├── CE POI-searcher (SoftWindowBooleanSCA, λ_hw) ... ran@b2 DEAD — CE_S pinned at 5.545 = ln 256 = chance;
│   │     band-floor variants S1/S1b/S1c all failed, catalogued in docs/superpowers/plans/2026-08-18-ascadv1-poi-searcher.md
│   │     [A] ckpt poi_searcher_desync0_s1.pth + .config.json
│   ├── 2O oracle gate (frozen windows 156×517) ... ran@b2 win@b2 v1 OK — 2O CPA rank 0, abs_corr 0.216;
│   │     positive control corr −0.1998, z = −17.4. RPOI/OPOI-snr/NOPOI baselines rank 79–239 (they fail)
│   │     [A] docs/lab-log/exp-v1-b1-poi-baselines.json, oracle-3o-calibrated.json
│   ├── 2O learned, no-band + order hinge (µr < µC) ★★ THE CLAIMED v1 RESULT
│   │     ├── desync0,  seeds 0/1/2, 3 restarts ... ran@b2 win@b2 multi-seed v1 OK — rank 0 / 0 / 0   [L; ckpt A]
│   │     ├── desync100, seeds 0/1/2 ............. ran@b2 win@b2 multi-seed v1 OK — rank 0 / 0 / 0   [L; ckpt A]
│   │     ├── orderdec variant ................... ran@b2 — checkpoints only, no verdict recorded      [implemented→ran]
│   │     └── ⚠ exp_2o.py prints its rank table and never persists it → the 0/0/0 is backed by six .pth
│   │           provenance + task/plan logs, not a result file.  [A] .../ASCAD_trained_models/poi_searcher_runs/exp2o_desync{0,100}_order_rst3_b2_s{0,1,2}.pth
│   ├── 2O no-band + repulsion ..... ran@b2 multi-seed v1 FAIL — ranks 12 / 88 / 25, windows park on edges → DEAD
│   │     [A] exp2o_desync0_noband_rep_b2_s{0,1,2}.pth; docs/superpowers/plans/2026-08-20-ascadv1-leakage-localizer.md §12.5
│   ├── 2O banded (hand bands r[40,280] / C[400,640]) ... ran@b2 — desync100 abs single unseeded run; ablation only, not the claimed model
│   ├── all-16-byte sweep (--bytes all) ... ran@all-16 — only byte 2 recovers; the crop is byte-2-only → all-16 is NOT reachable here
│   └── A2 certification of the 2O tokens (byte 2)
│       ├── tokens → 9-way HW reader ......... ran@b2 win@b2 — acc 0.28, windows land 157/516 → tokens really carry the shares  [A] exp-v1-a2-tokens-to-HW-reader{,-recheck}.json
│       ├── tokens → 256-way S reader ........ ran@b2 DEAD — FAIL_CHANCE (MLP, tiny_tf, ftwin) → no direct S from tokens  [A] exp-v1-a2-tokens-to-S-reader*.json
│       └── a2 lightweight 2O re-run ......... ran@b2 v1 FAIL — attack corr 0.021; rank 0 depends on restarts=3 + batch 256,
│             so the headline is not reproducible from a single cheap run .............................. [A][U] exp-v1-a2-two-order-baseline.json
└── LLM / transformer probes (all byte 2, 8k/2k, CE + accuracy ONLY — rank/GE was never computed on any of them)
    ├── frozen GPT-2 Route A raw (projector + prompt + mean pool) ... ran@b2 v1 FAIL — smokes 8 e, all FAIL_CHANCE (CE ln 256 = 5.545)
    │     [A] docs/lab-log/exp-v1-gpt2-ce-smoke{,-p16s8,-p16s8-prompt}.json · LLM/route_a_raw_llm.py itself never run
    ├── Route B.1 AE reconstruct ............ ran@b2 v1 FAIL — only via leakage_localizer (exp-v1-gpt2-routeb-ae-p16s8.json); LLM/ script unrun [A]
    ├── Route B.2 masked AE / latent token ... ran@b2 v1 FAIL — 8 e (exp-v1-timellm-gpt2-routeb-masked-p16s8-{train,eval}.json)  [A]
    ├── Route B.2 SimMIM --mask-level embed . ran@b2 v1 FAIL — 4 e / 2 e pretrain ~75 s smoke, CE 5.551/5.587.
    │     The ONLY route ever run from an LLM/ script (LLM/models.py 16:39 → run 18:18, same day) ............... [A] docs/lab-log/llm-b-masked-embed-smoke.{json,log}
    ├── Time-LLM GPT-2 reprogramming layer ... ran@b2 v1 FAIL DEAD — 8 e and 40 e, val_acc → 0. Lives in
    │     leakage_localizer/exp_v1_timellm_gpt2.py, NOT in LLM/ (LLM/route_c_timellm.py was never created) .... [A] exp-v1-timellm-gpt2-reprogram-p16s8-e40-train.json
    ├── E1 scratch Transformer (0.3M) ....... ran@b2 v1 FAIL — FAIL_CHANCE ....................................................... [A] exp-v1-e1-scratch-encoder.json
    ├── soft-slots tokenizer ................ ran@b2 DEAD — stage-A pick collapsed ............................................... [A] exp-v1-soft-slots-stageA.json
    └── Strategy 2 trainable patch transformer (no LLM) ★ the only LLM-family model given a full budget
          ran@b2 v1 FAIL — 50k/10k, ≤50 e, early stop: attn final_rank 230, bilinear 94, cnn+bilinear 233
          [A] docs/lab-log/patch_transformer_{trainable_v1_attn,trainable_v1_bilinear,cnn_v1_bilinear}.json + ckpts
          Strategy 1 (gpt2-fixed) implemented, never run ............................................................ [implemented]
```

**v1 conclusion.** A 1st-order-masked v1 byte *is* recoverable in this repo by two independent
mechanisms — a strong classifier at byte 2 (`ascad_head`, CNN baseline) and the 2O window localizer
(order hinge, 3 seeds, both desyncs). Nothing in the LLM/transformer family reached even the chance
bar, which is consistent with the SNR(S) ≈ 0.02 diagnosis rather than a bug in those models.

---

## 4. ASCADv2 (higher-order masked STM32) — attack tree

```
ASCADv2
├── Gray-box, multi-label heads (T-004/T-009) — multilabel=2, no-permind, 100k Profiling, stride 5 / pool 32,
│   batch 64, RAdam 1e-4, eval on 1000 Attack traces, ALL 16 bytes reported as mean rank over bytes
│   ├── timesnet_multi ....... ran@all-16 v2 FAIL — mean rank 141.12 (e22) → 129.62 (e30) → 124.25 ± 81.1 (e60)
│   │   [A] e60 STM32_AES_v2/ASCAD_data/ASCAD_trained_models/summary_full_fixedkey_gpu.json + *_s5p32_t100k_e60.pth;
│   │       e22/e30 JSON were overwritten by the e60 eval → logs only (logs/t009_timesnet_v2/eval_summary.log, retrain_s5p32_eval.log) [L]
│   ├── timesnet_ascad_head_multi ... ran@all-16 v2 FAIL — 133.19 (e22) → 126.44 (e30) → 137.19 ± 95.2 (e60)   [A][L] same
│   │   ⚠ artifact rows say "epochs": 30 while the checkpoint is *_e60.pth (e60 = resume from e30); input_stride/pool_segments null. [U]
│   │   ⚠ byte 5 traces_to_rank0 150 and byte 11 = 100 are non-null in the e60 artifact and are mentioned by NO task doc,
│   │     while T-004/T-009 call the whole run a plain fail — worth a look before repeating that claim. .......... [A][U]
│   ├── e90 ... planned then stopped — watch_e60_then_maybe_e90.sh rule "mean ≥ 110 → STOP" fired at e60 ................ [L]
│   ├── alpha-overfit sanity (512 traces, 30 e, LR 1e-3): CE 5.570 → 5.214 — proves the head fits, not an attack ....... [A]
│   └── MultiResNet 34-Head (pure PyTorch, sub_avg, 137.4M params) ... ran@all-16 win@all-16 all-16 v2 multi-task OK ★ FULL 16-BYTE BREAK
│         50k profiling, 15 epochs (8.6 min wall-clock on RTX 5090), RAdam + AMP. Evaluated on 5k attack traces:
│         ALL 16 BYTES reach Rank 0 (Byte 0–15 z-scores: +3.89σ to +6.86σ @1k traces; shuffled trace control mean rank = 143.8, 0/16 bytes at rank 0).
│         Permutation index top-1 accuracy = 67.42% (chance 6.25%).
│         [A] STM32_AES_v2/ASCAD_data/ASCAD_trained_models/multiresnet_34h_50k_subavg.pth (+ config, eval.json, 16bytes_rank.png)

├── Black-box, S-box target + Scoop optimizer (T-011) — seed 0, Scoop split 200k train / 1k val / 300k GE, no mask labels
│   ├── TimesNetPooledClassifier, byte 4 ... ran@b4 v2 FAIL — ge_last 3.65 @290001, best_ge 2.9 @190001, traces_to_ge0 null,
│   │     train CE → NaN after epoch 15 ......... [A] Scoop/artifacts/timesnet_blackbox/metrics/timesnet_pooled_b04_seed0_full_summary.json + ckpt
│   ├── same model, Scoop optimizer, byte 4 ... ran@b4 v2 FAIL — ge_last 2.75 @290001, 2 385 472 params, 37 504 s wall
│   │     [A] Scoop/artifacts/timesnet_blackbox/metrics/timesnet_pooled_scoop_b04_seed0_full_summary.json
│   ├── bytes 0–3 ... ran v2 FAIL — ge_last 190.6 / 111.5 / 123.0 / 116.6 (Scoop-optimizer variant 211.35 / 111.4 / 64.3 / 70.7)
│   │     [A] Scoop/artifacts/timesnet_blackbox/byte_sweep/*_status.tsv
│   ├── bytes 5–15 ... never run ........................................................ [gap]
│   ├── pool_mode avg vs max (5 e smokes, bytes 0–4) ... ran — all 10 summary JSONs reproduce the T-004 2026-09-23 tables
│   │     (e.g. avg b0 ge_last 3.6 vs max 10.0) ......... [A] Scoop/artifacts/timesnet_blackbox/metrics/*_e5_smoke_summary.json; knob is config-only (TimesNet/config.py)
│   └── doc-only numbers: "rank_final@1k = 102" and "Scoop-slice 1k = 17" appear only in markdown (phase_1_faq.md §6);
│         no rank artifact exists in metrics/ or runs/ .................................... [U]
├── Scoop reproduction (our re-implementation, ASCADv2 extracted 15k)
│   ├── upstream code .................. [implemented] Scoop/scoop/*, examples/ascadv2_attack.ipynb, ascadv2_profiling.ipynb — notebooks NEVER executed;
│   │     the upstream >900 MB pre-trained model is absent. Only upstream file we touched: scoop/metrics.py (+54/−22,
│   │     vectorized GE and a fixed n_tries+1 bug — this changes every "pre-fix" number below)
│   ├── repro-A single-layer ~231M (linear_size 15171, ckpt 926 MB) ... ran@b4 v2 FAIL — ge_last 199.7 (pre-fix twin said 380.38)
│   │     [A] Scoop/artifacts/ascadv2_repro/metrics/single_layer_mlp_seed0_full_eval_t20_fixed_summary.json
│   ├── repro-v1 finetune 4×3000 (~73M, fixed width) ... ran@b4 v2 FAIL — ge_last 7.7 (pre-fix 14.67), no traces_to_ge0
│   │     [A] Scoop/artifacts/ascadv2_repro/metrics/finetune_mlp_seed0_full_eval_t20_fixed_summary.json
│   ├── repro-v1 other-15 sweep (byte 4 excluded, seed 0) ... ran@all-15 v2 FAIL — recovered [] ; ge_last 14.60 (b10) → 248.35 (b7)
│   │     [A] Scoop/artifacts/ascadv2_repro/byte_sweep/BYTE_SWEEP_REPORT_finetune_other15_seed0_20260730_170636.md
│   └── repro-v2 paper-narrow (width 256 = 2^8, depth searched 1–6, 2 h HPO/byte, 25 e, GE to 300k)
│       ├── bytes 0–15, seed 0 ... ran@all-16 win@b15 — recovered [15] only: ge_last 0.0, traces_to_ge0 100001, n_hpo 14.
│       │     Unsuccessful 0–14; best ge_last b8 3.15, b10 14.55, b12 12.70, b4 15.75; val_top1 ≈ 0.003
│       │     [A] Scoop/artifacts/ascadv2_repro_v2/byte_sweep/BYTE_SWEEP_REPORT.md + metrics/*b15*summary.json
│       ├── +25-epoch continue (e50) on b0–14 ... ran — b4 43.55 → 15.75, b8 25.35 → 3.15, still no GE → 0 ....... [A] stage=continue_plus_25 rows
│       └── ⚠ single seed, and byte 15 sits on Scoop's contested re-split. Per survey §II this is NOT a strict break;
│             T-011's own next-actions say do not claim GE = 0 or 16-byte recovery. ................. [A but not-a-break]
├── 3-window 3O localizer (T-015, byte 4, 20k prof / 5k attack, T = 15k) — the v2 mirror of the v1 2O idea
│   ├── oracle gate (frozen windows) ... ran@b4 v2 FAIL — prod_vs_m3 corr 0.0009 (z = 0.12), pairwise ≈ 0,
│   │     SNR(α) 2.0, SNR(β) 1.5, but SNR(C) = 0.016 → the third share is not findable where the plan expected it
│   │     [A] docs/lab-log/oracle-3o-calibrated.json, oracle-pairwise-calibrated.json, oracle-3o-controls.json
│   ├── Loss 1 (3O correlation; fair / synthetic / aux-HW) ... ran@b4 v2 FAIL — corr_m3 flat ≤ 0.003 ............. [A] exp3o-{fair-smoke,B-synthetic,C-auxhw,D-fair-s1}.json
│   ├── Loss 2 (head CE, non-worst-case, no mask labels) ... ran@b4 v2 FAIL — FAIL_CHANCE, acc 0.0054 .......... [A] exp3o-F-loss2-10e.json
│   ├── Loss 2 + known π (permutation fed as input) ... ran@b4 v2 FAIL — FAIL_CHANCE; probe only, not an attack . [A] exp3o-G-loss2-knownpi-10e.json
│   ├── Loss 3 (GF share CE — USES mask labels ⇒ worst-case) ... ran@b4 PASS_LOCALIZE (acc_a ≈ 0.06, b/c ≈ 0.003) but
│   │     not eligible as a break by definition .................................................................... [A] exp3o-H-loss3-10e.json
│   └── verdict ... v2 3-window route is DEAD on public 15k. T-015's own status line ("planned, don't train windows
│         until the gate") is stale: the gate and Loss 1/2/3 smokes all ran 2026-09-07/08. .......................... [U]
└── GPAM foundation model (T-019) ................................................. [surveyed][implemented][BLOCKED]
    ├── run_ascadv2_gpam_baseline.py (smoke, byte head, patch 150) + train_gpam_ascadv2_budget.py (100k/10k/1k, bytes 0–3, GE/epoch)
    ├── GPAM/gpam_budget_run.log is 0 bytes; no gpam_budget_results.json, no smoke log/JSON anywhere
    ├── "GPU smoke succeeded (16 s PTX warm-up, val acc 0.0)" — INDEX/T-019 claim with NO artifact ................ [U]
    ├── "smoke/GPU works via graph mode" — jit_compile=False + run_eagerly=False is a *planned* decision;
    │   both scripts still hard-code run_eagerly=True ........................................................... [U]
    └── Phase 0 blocker: full raw 800k (only shard 8 valid; see §2). The "Rank 0 in ~80 traces" figure that
        T-020's matrix marks ✅ is the CHES paper's claim, not ours.
```

**v2 conclusion.** Nine independent mechanisms have touched public ASCADv2 (TimesNet gray-box,
TimesNet black-box ×2 optimizers, Scoop single-layer, Scoop finetune, Scoop paper-narrow, 3O
localizer Losses 1–3, GPAM, multi-ResNet reference). **None recovered a byte under strict
shuffle-on / no-mask-label conditions except Scoop repro-v2 byte 15 on the contested re-split.** The
raw-800k dataset, which any full break needs, is still ~40% downloaded.

---

## 5. Dead-ends index (do not re-run without changing the premise)

| Dead end | Why it died | Evidence |
|---|---|---|
| v1 CE POI-searcher (λ_hw) | CE stuck at 5.545 = chance | ckpt + `plans/2026-08-18-…` |
| v1 2O with repulsion loss | ranks 12/88/25, windows park on edges | ckpts, plan §12.5 |
| v1 tokens → 256-way S reader | no direct S leakage in tokens (matches SNR(S)≈0.02) | `exp-v1-a2-tokens-to-S-reader*.json` |
| Frozen-GPT-2 + mean pooling (Routes A / B.1 / B.2 / SimMIM) | every run FAIL_CHANCE | 5 JSONs, §3 |
| Time-LLM vocab reprogramming | val_acc → 0 at 40 e | `…reprogram-p16s8-e40-train.json` |
| Soft-slots tokenizer | stage-A pick collapsed | `exp-v1-soft-slots-stageA.json` |
| Scratch patch transformer at 50k budget | rank 94–233 with full budget | `patch_transformer_*.json` |
| v1 sample-node GCN/GAT, GIN, Plan A sweep, Plan C | ≤ Plan B (rank 7); Plan A 182–240 | `gnn_runs/…` |
| v2 3-window localizer (all 3 losses) | SNR(C) = 0.016 on real 15k; Loss 3 needs mask labels | `oracle-3o-*.json`, `exp3o-*.json` |
| v2 TimesNet gray-box and black-box | mean rank ~124–141 / ge_last 2.75–190, never GE→0 | §4 |
| v2 LR 1e-2 TimesNet on v1 (cross-ref) | CE 5.544, never learned | §3 |

---

## 6. Evidence gaps (fix these before citing this file in a paper)

1. **`exp_2o.py` never persists its rank table.** The headline v1 claim (rank 0 × 3 seeds × 2 desyncs)
   rests on six checkpoints plus task/plan prose. Re-run with `--seeds 0,1,2` and write a JSON.
2. **The a2 re-run did not reproduce it** (attack corr 0.021) — rank 0 is sensitive to `restarts=3` and
   batch 256. Any claim must state those two knobs.
3. **SNR(S) ≈ 0.02 is computed but never saved**, yet it is the premise of the entire LLM diagnosis.
4. **GPAM has zero artifacts** (0-byte log) while INDEX claims a successful GPU smoke; and "graph mode"
   is not what the code does.
5. **v2 e22/e30 summaries were overwritten** by the e60 eval; only `.log` text survives for those points.
6. **e60 rows carry `"epochs": 30`** and null `input_stride` / `pool_segments`; the e60 number is real,
   the row metadata is not.
7. **Non-null `traces_to_rank0` on v2 bytes 5 and 11** (150 / 100) in the e60 artifact is documented nowhere.
8. **`docs/tasks/T015-ascadv2-3o-localizer.md` is stale and unregistered** — it collides with
   `T015-topic-reference-notes.md` (different task, same id) and contradicts its own artifacts.
9. **`Scoop/Analyzing Scoop's ASCADv2 Attack.md`** restates the paper's break as verified fact and never
   mentions the 2026 false-positive critique, contradicting our survey and every run of ours.
10. **`T-017` Phase-1 checkmarks** cite `LLM/eval.py`, `LLM/route_c_timellm.py`,
    `LLM/tests/test_route_abc.py` — none of those files exist.
11. **`scripts/run_cnn_mlp_best_with_delay.sh` is broken** (its `cd "$ROOT_DIR"` lands in `scripts/`, so the
    relative `ASCAD_train_models.py` / HDF5 paths fail) and is untracked, while AGENTS.md/T-006 cite it.
12. **The `timesnet_lr1e-3` v1 rows** are 6 × `status=error`; do not cite that sweep.

---

## 7. Next actions

Tagged I / U / S per `.grok/rules/03-priority-triage.md`. Meeting-facing copies live in
[`docs/lab-log/todo.md`](./lab-log/todo.md).

The 2026-09-24 14:00 plan (one GPU decision test, three read-only briefs, no new claims) is
[`docs/2026-09-23/README.md`](./2026-09-23/README.md). It does not change the scoreboard above.

| Action | I | U | S | Where | Done when |
|--------|---|---|---|-------|-----------|
| Persist the 2O rank table: re-run `exp_2o.py --seeds 0,1,2 --restarts 3` on desync0 + desync100, write JSON | H | H | M | T-014 | `poi_searcher_runs/exp2o_*_rank.json` exists and matches or corrects the 0/0/0 claim |
| Add SNR(S)/SNR(r)/SNR(C) save to the localizer so the v1 diagnosis is artifact-backed | H | M | H | T-014/T-020 | one JSON per desync |
| Resolve gap 8: renumber or retire `T015-ascadv2-3o-localizer.md`, then update its status from the artifacts | M | M | H | T-015 | INDEX has no id collision and file status matches `exp3o-*.json` |
| Fix `Scoop/Analyzing…md` to state the 2026 critique (gap 9) | H | M | H | T-010/T-012 | the doc no longer asserts an unqualified break |
| v2 black-box bytes 5–15 (finish the 16-byte sweep we started) | H | L | L | T-011 | all-16 report exists |
| Implement AOD-GE + first-layer ablation so "is it really broken?" is answerable locally, not by citation | H | L | M | T-012 | a run of AOD-GE on Scoop b15 exists |
| Check v2 e60 bytes 5/11 `traces_to_rank0` (gap 7) before repeating "plain fail" | M | L | H | T-004/T-009 | explained or dismissed |
| Unblock GPAM: finish raw 800k download, then set `run_eagerly=False` + `jit_compile=False` and keep the log | M | L | L | T-019 | non-empty `gpam_budget_results.json` |
| v2 multi-ResNet reference weights (permind baseline is still not local) | M | L | L | T-009 | weights on disk + one eval |
| Multi-seed + all-byte v1 for CNN/MLP/GraphSAGE so "v1 OK" is not byte-2/seed-0 only | M | L | L | T-006/T-003 | aggregate JSON with ≥3 seeds |

### Priority snapshot

- **Do now:** persist the 2O rank table (it is the headline v1 result and currently the only claim without an artifact).
- **Before next meeting:** the v2 byte-level story — Scoop repro-v2 recovered b15 only; everything else fails; AOD-GE is unimplemented.
- **Deep work (park):** GPAM raw data, AOD-GE implementation, v2 bytes 5–15, multi-seed v1 baselines.

---

## 8. Maintenance

- Update the scoreboard (§1) and the relevant tree when a run finishes; move anything disproven into §5
  or §6 rather than deleting it.
- One line per run, not one paragraph. Numbers must use the real field names from
  [`docs/notes/metrics.md`](./notes/metrics.md) — `rank_final`, `traces_to_rank0`, `ge_last`,
  `best_ge`, `traces_to_ge0`, `val_top1_acc`. Never invent a metric name.
- Keep the evidence tag honest: `A` requires a file path a stranger could open.
- Related tasks: T-003, T-004, T-006, T-009, T-010, T-011, T-013, T-014, T-015, T-016, T-017, T-018,
  T-019, T-020. Roadmap task: [T-022](./tasks/T022-break-ascad-roadmap.md).
