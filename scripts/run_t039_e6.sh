#!/bin/bash
# T-039 E6 label-ablation ladder (user go-ahead 2026-10-15). Design frozen in
# docs/tasks/T039-project-retrospective.md ("Pre-registration E6").
#   Recipe = E3's 50k reference (Compact B, h64, 15 epochs) + decoder NLL weight 1.0.
#   Rungs (labels dropped): R0 none, R1 pi, R2 beta, R3 alpha, R4 beta+C, R5 all. Seeds 1, 2.
# Each checkpoint (best + last): w6_nochead_eval (heads/decode) + t039_controls (works / works_v2).
set -u

PY=/home/lantian/miniconda3/envs/py10/bin/python
REPO_ROOT=/home/lantian/ASCAD
cd "$REPO_ROOT"
M=STM32_AES_v2/ASCAD_data/ASCAD_trained_models
O=experiments/out/t039_e6
mkdir -p "$O"
exec > >(tee -a "$O/pipeline.log") 2>&1

COMMON=(--compact --bottleneck-dim 0 --head-hidden-dim 64 --batch-size 64 --lr 1e-4
        --sub-avg --c-loss-weight 0.5 --save-last --train-len 50000 --epochs 15
        --decoder-loss-weight 1.0)
# RC / RCpi added 2026-10-19 (T-039 E6-A): drop C (labels of alpha, beta, pi kept) / drop C and pi (alpha, beta kept).
declare -A DROP=([R0]="" [R1]="pi" [R2]="beta" [R3]="alpha" [R4]="beta,C" [R5]="alpha,beta,C,pi"
                 [RC]="C" [RCpi]="C,pi" [RCpiU]="C,pi" [3head]="C,pi" [3headA]="C,pi" [3headAB]="C,pi" [3headW]="C,pi" [3headBit]="C,pi")
# RCpiU (2026-10-20): RCpi with the decoder routing fixed at R = 1/16 (--uniform-pi); scored with uniform pi.

# 3head (2026-10-20): 3headResNet = alpha, beta, ONE C head, no pi heads; uniform routing (pi rows are exact zeros -> 1/16).
# 3headA (2026-10-21): 3head with label annealing: the beta CE is ON for epochs 1-3 only (--anneal-drop beta --anneal-epochs 3).
declare -A UNIFORM_PI=([RCpiU]="--uniform-pi" [3head]="--uniform-pi --three-head"
                       [3headA]="--uniform-pi --three-head --anneal-drop beta --anneal-epochs 3"
                       [3headAB]="--uniform-pi --three-head --anneal-drop alpha,beta --anneal-epochs 3"
                       [3headW]="--uniform-pi --three-head --c-head-hidden-dim 1024"
                       [3headBit]="--uniform-pi --three-head --c-head-form bit")
# 3headBit (2026-10-29): 3head with a bit-linear C head (logit(v) = sum_i bit_i(v) w_i; 8 outputs).
# 3headW (2026-10-28): 3head with the single C head widened 64 -> 1024 (same C parameter count as RCpiU's 16 heads).
# 3headAB (2026-10-26): alpha AND beta CE on for epochs 1-3 only; afterwards only the decoder loss on S remains.

score() {  # $1 label, $2 checkpoint
    local label=$1 ckpt=$2
    if [[ "$ckpt" == *_last.pth ]]; then
        local base=${ckpt%_last.pth}.pth
        cp -n "$base.config.json" "$ckpt.config.json"
        cp -n "$base.mean_wf.npy" "$ckpt.mean_wf.npy"
    fi
    echo "=== [$(date -Is)] score $label ($ckpt)"
    local UNI=
    [[ "$label" == e6_RCpiU* ]] && UNI=1
    "$PY" -m experiments.w6_nochead_eval --label "$label" --model-file "$ckpt" \
        --save-npz "$O/${label}_heads.npz" --out-json "$O/${label}_eval.json" \
        || { echo "!!! w6 eval failed: $label"; return 1; }
    if [ -n "$UNI" ]; then  # untrained pi heads: score with exactly uniform pi (keeps the raw npz)
        "$PY" - "$O/${label}_heads.npz" <<'PYEOF'
import sys, shutil, numpy as np
p = sys.argv[1]; shutil.copy(p, p.replace(".npz", "_rawpi.npz"))
d = dict(np.load(p))
for i in range(16):
    d[f"permind_{i}_output"] = np.full_like(d[f"permind_{i}_output"], 1.0 / 16)
np.savez(p, **d)
PYEOF
    fi
    "$PY" experiments/t039_controls.py --label "$label" --model-file "$ckpt" \
        --npz "$O/${label}_heads.npz" --out-json "$O/${label}_controls.json" \
        || echo "!!! controls failed: $label"
}

echo "=== [$(date -Is)] T-039 E6 pipeline start"
# Order: R2 seed 1 (in flight at the descoping), then R1, then the R0 control.
# Descoped 2026-10-15 at the user's request: only rungs with >= 40% prior success (R0, R1); R2 seed 1
# was already running and is kept. R3, R4, R5 not run. R2 seed 2 and a 40-epoch R2 (E6_TAG=L,
# E6_EPOCHS=40) were added after R2 seed 1 passed (user go-ahead for the long run 2026-10-15).
# E6b (2026-10-15): R1 at 300k via E6_SPECS='R1:1 R1:2' E6_TAG=T300k E6_TRAIN_LEN=300000 E6_C4=1.
for spec in ${E6_SPECS:-R2:1 R2:2 R1:1 R1:2 R0:1 R0:2}; do
    r=${spec%:*}; s=${spec#*:}
    ck=$M/t039_e6_${r}${E6_TAG:-}_seed$s.pth
    if [ ! -f "$ck" ]; then
        echo "=== [$(date -Is)] train $r (drop '${DROP[$r]}') seed $s"
        "$PY" MultiResNet/train.py "${COMMON[@]}" ${E6_EPOCHS:+--epochs $E6_EPOCHS} \
            ${E6_TRAIN_LEN:+--train-len $E6_TRAIN_LEN} ${E6_HEADS_FP32:+--heads-fp32} --drop-labels "${DROP[$r]}" ${UNIFORM_PI[$r]:-} --seed "$s" \
            --out-model "$ck" > "$O/${r}${E6_TAG:-}_seed${s}_train.log" 2>&1
        echo "=== [$(date -Is)] train $r seed $s exit $?"
    fi
    [ -f "$ck" ] || { echo "!!! missing $ck"; continue; }
    [ -f "$O/e6_${r}${E6_TAG:-}_s${s}_last_controls.json" ] && { echo "=== $r seed $s already scored"; continue; }
    score "e6_${r}${E6_TAG:-}_s${s}_best" "$ck"
    score "e6_${r}${E6_TAG:-}_s${s}_last" "${ck%.pth}_last.pth"
    if [ -n "${E6_C4:-}" ]; then  # E6b: does the attack use the learned routing? (A0 vs uniform/oracle pi)
        L=e6_${r}${E6_TAG:-}_s${s}_best
        "$PY" experiments/c4_oracle_swaps.py --add-model "$L:$O/${L}_heads.npz" --models "$L" \
            --variants A0_as_trained,B4_oracle_pi,F2_uniform_pi,D0_oracle_all \
            --out-json "$O/${L}_c4.json" > "$O/${L}_c4.log" 2>&1 || echo "!!! c4 failed: $L"
        echo "=== [$(date -Is)] c4 done $L"
    fi
done

echo "=== [$(date -Is)] T-039 E6 pipeline done"
touch "$O/DONE${E6_TAG:+_$E6_TAG}"
