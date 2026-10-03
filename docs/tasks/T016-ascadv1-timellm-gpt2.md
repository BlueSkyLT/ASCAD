# T-016: ASCADv1 Time-LLM GPT-2 standalone

## Status

`done` — negative result: the 40-epoch reprogram run stayed at chance. Family survey and follow-up live in T-020 (`docs/notes/patch-pipeline-survey.md`) (2026-10-04 cleanup)

## Goal

Deliver a standalone Time-LLM-style GPT-2 classifier for ASCADv1 (desync0 first), with train/eval CLI, checkpointing, and smoke-run metrics.

## Context

- Related code / data paths:
  - leakage_localizer/exp_v1_timellm_gpt2.py
  - leakage_localizer/exp_v1_gpt2_ce.py
  - leakage_localizer/data_v1.py
  - ATMEGA_AES_v1/ATM_AES_v1_fixed_key/ASCAD_data/ASCAD_databases/ASCAD.h5
- Related docs:
  - docs/LLM-plan.md
- Constraints:
  - Keep GPT-2 backbone frozen by default.
  - Do not edit baseline ASCAD train/test registry yet.
  - Keep to smoke scope first (5-10 epochs).

## Plan

- [x] (I:H U:H S:H) Create standalone script with train/eval modes and GPT-2 default.
- [x] (I:H U:H S:M) Add reproducible outputs: JSON metrics and checkpoint save/load.
- [x] (I:H U:H S:M) Run smoke training/eval and capture results in lab-log JSON.
- [x] (I:M U:M S:M) Add brief usage notes and next-step expansion points.

## Decisions

- 2026-09-16: Implement as standalone under leakage_localizer first; defer integration into ASCAD_train_models.py.
- 2026-09-16: Default model remains `gpt2` with frozen backbone and trainable adapter/head.

## Working notes / session log

### 2026-09-16 — implementation start

- Summary: Added a dedicated Time-LLM-style script with train/eval split, checkpointing, and JSON reporting.
- Files touched:
  - leakage_localizer/exp_v1_timellm_gpt2.py
- Commands / runs:
  - pending smoke run
- Metrics / results:
  - pending
- Open questions:
  - whether to keep per-trace z-score from data_v1 vs train-stat normalization for later larger runs.

### 2026-09-16 — smoke run completed

- Summary: New standalone script trained and evaluated end-to-end with GPT-2 frozen backbone.
- Files touched:
  - leakage_localizer/exp_v1_timellm_gpt2.py
  - docs/tasks/INDEX.md
- Commands / runs:
  - conda run -n py10 python leakage_localizer/exp_v1_timellm_gpt2.py --mode train --epochs 5 --n-prof 8000 --n-atk 2000 --batch 32 --patch 16 --step 8 --device cuda --model gpt2 --out-json docs/lab-log/exp-v1-timellm-gpt2-smoke-train.json --ckpt docs/lab-log/checkpoints/exp-v1-timellm-gpt2-smoke.pth
  - conda run -n py10 python leakage_localizer/exp_v1_timellm_gpt2.py --mode eval --n-prof 8000 --n-atk 2000 --batch 32 --device cuda --model gpt2 --out-json docs/lab-log/exp-v1-timellm-gpt2-smoke-eval.json --ckpt docs/lab-log/checkpoints/exp-v1-timellm-gpt2-smoke.pth
- Metrics / results:
  - Profiling CE: 5.5499, acc: 0.0050 (n=8000)
  - Attack CE: 5.5835, acc: 0.0050 (n=2000)
  - Chance baseline: CE=ln(256)=5.545, acc=0.0039
  - Verdict: FAIL_CHANCE (expected for short smoke; objective was execution validation)
- Open questions:
  - should next iteration enable AE pretraining by default for stronger early-signal learning.

### 2026-09-16 — masked pretraining added and validated

- Summary: Added TimeMAE-style masked StageA pretraining support to the standalone script and validated train/eval path.
- Files touched:
  - leakage_localizer/exp_v1_timellm_gpt2.py
- Commands / runs:
  - conda run -n py10 python leakage_localizer/exp_v1_timellm_gpt2.py --mode train --use-ae --ae-pretrain-mode masked --mask-ratio 0.5 --ae-latent-dim 8 --ae-hidden-dim 32 --pretrain-epochs 2 --pretrain-lr 1e-3 --epochs 8 --lr 1e-3 --patch 16 --step 8 --n-prof 8000 --n-atk 2000 --batch 32 --device cuda --model gpt2 --prompt "Classify the ASCAD side-channel trace and infer its S-box output." --out-json docs/lab-log/exp-v1-timellm-gpt2-routeb-masked-p16s8-train.json --ckpt docs/lab-log/checkpoints/exp-v1-timellm-gpt2-routeb-masked-p16s8.pth
  - conda run -n py10 python leakage_localizer/exp_v1_timellm_gpt2.py --mode eval --n-prof 8000 --n-atk 2000 --batch 32 --device cuda --model gpt2 --out-json docs/lab-log/exp-v1-timellm-gpt2-routeb-masked-p16s8-eval.json --ckpt docs/lab-log/checkpoints/exp-v1-timellm-gpt2-routeb-masked-p16s8.pth
- Metrics / results:
  - StageA masked reconstruction improved: train 0.9151 -> 0.4330, val 0.6389 -> 0.3062 (2 epochs)
  - Final profiling CE: 5.5515, acc: 0.0050 (n=8000)
  - Final attack CE: 5.5866, acc: 0.0050 (n=2000)
  - Verdict: FAIL_CHANCE at smoke budget

### 2026-09-22 — reprogramming layer validation (40 epochs)

- Summary: Added ReprogrammingLayer + mapping_layer (Time-LLM-style) and ran extended training to validate learning capacity.
- Files touched:
  - leakage_localizer/exp_v1_timellm_gpt2.py (added ReprogrammingLayer, vocab-report mode)
- Commands / runs:
  - conda run -n py10 python leakage_localizer/exp_v1_timellm_gpt2.py --mode train --use-reprogram --num-tokens 1000 --reprogram-heads 8 --reprogram-dim 32 --epochs 40 --lr 1e-3 --patch 16 --step 8 --n-prof 8000 --n-atk 2000 --batch 32 --device cuda --model gpt2 --out-json docs/lab-log/exp-v1-timellm-gpt2-reprogram-p16s8-e40-train.json --ckpt docs/lab-log/checkpoints/exp-v1-timellm-gpt2-reprogram-p16s8-e40.pth
- Metrics / results:
  - Trainable params: 50.55M (reprogramming + classification head), frozen: 124.44M (GPT-2)
  - Training CE: 5.613→5.509 (slight decrease), acc: 0.0054→0.0079
  - Validation acc: 0.0063→0.0000 after epoch 11 (collapsed to zero, not improving)
  - Final profiling: CE=5.545, acc=0.0055 (n=8000)
  - Final attack: CE=5.577, acc=0.004 (n=2000)
  - Verdict: **FAIL_CHANCE** — model did not learn above chance even with 40 epochs
- Interpretation:
  - Frozen GPT-2 embeddings appear incompatible with power trace domain
  - Reprogramming layer (cross-attention mapping) insufficient to bridge the gap
  - Validation collapse suggests overfitting to noise or fundamental architecture mismatch
- Open questions:
  - Should we unfreeze GPT-2 layers (defeats LLM transfer learning premise)?
  - Try different LLM backbones (smaller, domain-agnostic)?
  - Abandon LLM approach for this task and return to CNN/TimesNet baselines?

### Usage notes

- Route A (no AE):
  - conda run -n py10 python leakage_localizer/exp_v1_timellm_gpt2.py --mode train --epochs 8 --patch 16 --step 8 --n-prof 8000 --n-atk 2000 --batch 32 --device cuda --model gpt2 --out-json docs/lab-log/exp-v1-timellm-gpt2-smoke-train.json --ckpt docs/lab-log/checkpoints/exp-v1-timellm-gpt2-smoke.pth
- Route B masked pretrain (TimeMAE-like):
  - add --use-ae --pretrain-epochs 2 --ae-pretrain-mode masked --mask-ratio 0.5
- Route C reprogramming (Time-LLM-style):
  - add --use-reprogram --num-tokens 1000 --reprogram-heads 8 --reprogram-dim 32
- Vocab analysis:
  - conda run -n py10 python leakage_localizer/exp_v1_timellm_gpt2.py --mode vocab-report --ckpt <checkpoint.pth> --vocab-topk 1000 --out-json <vocab.json>
- Eval:
  - conda run -n py10 python leakage_localizer/exp_v1_timellm_gpt2.py --mode eval --batch 32 --device cuda --ckpt <checkpoint.pth> --out-json <eval.json>

## Next actions

| Action | I | U | S | Notes |
|--------|---|---|---|-------|
| Decide on Time-LLM viability: continue debugging or pivot | H | H | M | **Current blocker**: 40-epoch reprogram failed at chance |
| Try unfreezing GPT-2 top layers as ablation | M | M | M | Test if frozen backbone is the bottleneck |
| Compare to TimesNet baseline on same data split | H | M | M | Establish competitive performance bar |
| Document negative result and close task | M | M | H | If LLM approach is fundamentally unsuitable |

### Priority snapshot

- Do now: run smoke train/eval and log metrics.
- Before next meeting: provide reproducible command block and result JSON paths.
- Deep work (park): full integration into benchmark pipeline and key-rank export.

## Related

- Task ids:
  - T-004, T-014
- docs/lab-log/todo.md items:
  - add once smoke metrics are collected
- External links:
  - https://github.com/KimMeen/Time-LLM
