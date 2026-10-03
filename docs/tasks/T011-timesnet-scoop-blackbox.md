# T-011: TimesNet + Scoop on ASCADv2 (black-box)

## Status

`done` — bytes 0–4 complete (seed 0), no GE→0; bytes 5–15 / extra seeds are optional backlog (ask before GPU) (2026-10-04 cleanup)

## Goal

Train the **same pooled TimesNet** as the T-004 black-box baseline, but with the **Scoop optimizer** (`create_graph` + Hutchinson), and report GE vs RAdam TimesNet and vs Scoop MLP. This file is **only** TimesNet-with-Scoop. Non-Scoop TimesNet (v1, v2 multilabel, v2 black-box RAdam) is **[T-004](./T004-timesnet-original.md)**.

## Context

| Item | Value |
|------|--------|
| Dataset | `STM32_AES_v2/ASCAD_data/ASCAD_databases/ascadv2-extracted.h5` |
| Attack model | **Black-box** unmasked `Sbox(p_b ⊕ k_b)` — Scoop paper labels, not T-009 multilabel |
| **Model** | **`TimesNetPooledClassifier`** (`pooled`) — **not** ASCAD-head, **not** multilabel |
| Encoder | 2× TimesBlock, `d_model=32`, `d_ff=64`, `top_k=5`, emb kernel 3 |
| Length | L=15 000, `input_stride=5` (L_eff=3000), `pool_segments=32` |
| Head | Linear `d_model → 256` |
| Params | **2 385 472** |
| Split | train 200k Profiling / val 1k Attack / GE on Profiling [200k:500k] (300k); zero-trace fix |
| Runner | `Scoop/experiments/ascadv2_repro/run_timesnet_blackbox.py --optimizer scoop` |
| Wrapper | `Scoop/experiments/ascadv2_repro/run_timesnet_scoop.sh` |
| Sweep | `run_timesnet_scoop_byte_sweep.sh` |
| Artifacts | `Scoop/artifacts/timesnet_blackbox/` |
| RAdam baseline | T-004 §C / `REPORT_phase1.md` (same I/O, optimizer ≠ Scoop) |

Scoop hyperparams (Phase 2 full): `lr=1e-4`, `beta1=0.965`, `beta2=0.99`, `rho=0.04`, `weight_decay=0.2`, `hessian_iter=1`, `estimator=low_variance`, `hessian_update=10`, AMP **off**, `grad_clip=0`, loss in **bits** (NLL/log2). Epochs 25, batch 32, GE `n_tries=20`.

Random CE in nats ≈ 5.545 ≡ **8 bits**. Phase 2 train/val bits stay ~8.00 (no real CE learning).

---

## Results (Scoop optimizer)

Smoke (`timesnet_pooled_scoop_b04_seed0_smoke`): OK, ~37 s, peak CUDA ~3.9 GB; bits ~8.03→8.00; GE not meaningful.

### Byte 4 — full (`timesnet_pooled_scoop_b04_seed0_full`)

| Metric | TimesNet + **Scoop** | TimesNet + **RAdam** (T-004 Phase 1) |
|--------|----------------------|--------------------------------------|
| `ge_last` @ ~290k | **2.75** | 3.65 |
| `best_ge` | **2.75 @ 290 001** | **2.9 @ 190 001** |
| GE → 0 | **no** | no |
| Wall | **~10.4 h** (`37504` s) | ~4.6 h |
| Peak CUDA | ~3.9 GB | ~1.3 GB |
| Train loss | 7.994 bits | CE ~5.545 then NaN (AMP) |

Scoop is **slightly better** on this seed/byte at the endpoint, still **not** paper GE=0, and **~2× wall** (create_graph). Same 2.39M pooled model.

### Bytes 0–3 — Phase 2 sweep (`timesnet_p2_scoop_seed0_20260812_124826`)

| Byte | Opt | ge_last | best_ge | GE→0 | Wall |
|------|-----|---------|---------|------|------|
| 0 | Scoop | 211.35 | 110.35 @ 1 | no | ~10.4 h |
| 1 | Scoop | 111.4 | 73.1 @ 150k | no | ~10.4 h |
| 2 | Scoop | 64.3 | 62.0 @ 230k | no | ~10.3 h |
| 3 | Scoop | 70.7 | 50.4 @ 220k | no | ~10.2 h |
| **4** | Scoop | **2.75** | **2.75 @ 290k** | no | ~10.4 h |
| 5–15 | — | — | — | — | **not run** |

RAdam (T-004) on the same bytes, for comparison:

| Byte | RAdam ge_last | RAdam best_ge | Scoop ge_last | Scoop better? |
|------|---------------|---------------|---------------|---------------|
| 0 | 190.6 | 114.0 | 211.4 | no |
| 1 | 111.5 | 90.5 | 111.4 | tie / tiny |
| 2 | 123.0 | 104.0 | **64.3** | yes (still far from 0) |
| 3 | 116.6 | 76.0 | **70.7** | yes (still far from 0) |
| 4 | 3.65 | 2.9 | **2.75** | yes, small |

**No byte recovered.** Byte 4 remains the only single-digit GE. Scoop did not turn failing bytes into recoveries; it modestly helped 2, 3, and 4.

Vs local Scoop **finetune MLP** byte 4 (`ge_last≈7.7` t20 fixed): TimesNet+Scoop **2.75** is better GE on this seed, with far fewer params. Paper target (GE=0 by 250k/300k) is **not** met.

Report (template mixed RAdam notes — trust JSON/TSV): `Scoop/artifacts/timesnet_blackbox/byte_sweep/timesnet_p2_scoop_seed0_20260812_124826_status.tsv`. Summaries: `…/metrics/timesnet_pooled_scoop_b0{0-4}_*_summary.json`. Ckpts: `…/checkpoints/timesnet_pooled_scoop_b0*_*.pt`.

---

## Did Scoop-on-TimesNet “achieve something”?

**Same narrow story as RAdam TimesNet, not a Scoop breakthrough.**

- **Yes (weak):** byte **4**, pooled TimesNet, black-box Sbox labels, GE last **2.75** (RAdam was 3.65).
- **No:** GE never 0; bytes 0–3 unsuccessful; 5–15 untested; loss stays ~8 bits.
- **Model is not** `timesnet_ascad_head` and **not** `timesnet_multi`. It is **`TimesNetPooledClassifier`** with stride 5 + 32-segment pool + linear 256-way head.

Do not cite Phase 1 RAdam numbers as “Scoop + TimesNet”. Those are T-004.

---

## Plan

- [x] (I:H U:H S:H) `TimesNetPooledClassifier` + black-box runner matching Scoop split
- [x] (I:H U:H S:M) Phase 2 code: `--optimizer scoop` (`PHASE2_IMPL.md`)
- [x] (I:H U:H S:H) Phase 2 smoke byte 4
- [x] (I:H U:H S:L) Phase 2 full byte 4
- [x] (I:H U:H S:L) Phase 2 bytes 0–3
- [ ] (I:H U:L S:L) Bytes 5–15 Scoop — **ask first** (~10 h/byte)
- [ ] (I:M U:L S:M) `ascad_head_pooled` + Scoop (never trained)
- [ ] (I:M U:L S:L) Extra seeds on byte 4

## Decisions

- 2026-07-31: Phase 1 = RAdam black-box (now catalogued in T-004). Phase 2 = Scoop on same I/O.
- 2026-07-31: Model = `pooled` TimesNet, byte 4 first, 200k / 25e / GE 300k.
- 2026-08-06: AMP off + bits loss for Scoop; WD 0.2; no grad clip.
- 2026-08-13: This task file reports **only** Scoop-on-TimesNet. Non-Scoop TimesNet → T-004.

## Working notes / session log

### 2026-08-13 — Phase 2 bytes 0–3 complete

- Sweep `timesnet_p2_scoop_seed0_20260812_124826`: all four unsuccessful; ge_last 211 / 111 / 64 / 71.
- Auto-generated `PHASE2_SCOOP_BYTE_SWEEP_REPORT_latest.md` wrongly says “RAdam” and uses Phase 1 byte-4 as `phase1_ref` — **ignore that note**; byte 4 Scoop full is a separate run (`timesnet_pooled_scoop_b04_seed0_full`, ge_last 2.75).

### 2026-08-12 — Phase 2 full b4 + chain to 0–3

- Full b4 ~10 h; then `run_timesnet_scoop_byte_sweep.sh` for 0–3.
- Chain log: `Scoop/artifacts/timesnet_blackbox/logs/phase2_after_b4_then_b0-3.log`

### 2026-08-06 — Phase 2 smoke + code

- Smoke OK. Implementation: `create_graph` + `hutchinson_hessian`, NLL/log2, AMP off.
- Notes: `Scoop/artifacts/timesnet_blackbox/PHASE2_IMPL.md`

### 2026-08-05 — Phase 1 RAdam (moved to T-004)

- Byte 4 RAdam GE 2.9 / 3.65 is the **non-Scoop** baseline, not a Scoop result.

## Next actions

| Action | I | U | S | Notes |
|--------|---|---|---|-------|
| Cite byte-4 Scoop GE 2.75 vs RAdam 3.65 carefully (single seed) | H | H | H | meeting |
| Do **not** claim GE=0 or 16-byte recovery | H | H | H | |
| Bytes 5–15 / extra seeds | M | L | L | ask before GPU |
| Fix sweep report template (says RAdam on Scoop runs) | L | L | H | `timesnet_byte_sweep_report.py` |

### Priority snapshot

- **Do now:** use T-004 for architecture catalog; this file for optimizer swap
- **Before next meeting:** table RAdam vs Scoop GE bytes 0–4
- **Deep work (park):** more bytes, head-pooled, HPO on rho/wd

## Related

- Non-Scoop TimesNet catalog + byte-4 flowchart: [T-004](./T004-timesnet-original.md)
- Scoop MLP reproduction: [T-010](./T010-scoop-ascadv2-reproduction.md)
- v2 multilabel (not this threat model): [T-009](./T009-stm32-key-recovery.md)
- Code: `TimesNet/model.py` → `TimesNetPooledClassifier`; `Scoop/scoop/scoop.py`
