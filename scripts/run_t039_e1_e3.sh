#!/bin/bash
# T-039 E1 + E3 (user go-ahead 2026-10-13). Design frozen in docs/tasks/T039-project-retrospective.md.
#   E1: CompactMultiResNet Variant B, 300k profiling, seed 2 (seed 1 = compact_mresnet_b_cw05_t300k_seed1)
#   E3: mask-label efficiency, N = 1k/5k/20k labelled traces x seeds 1,2, gradient steps held ~constant
# Each checkpoint (best + last): w6_nochead_eval (heads/decode/GE) + t039_controls (nulls, AOD).
set -u

PY=/home/lantian/miniconda3/envs/py10/bin/python
REPO_ROOT=/home/lantian/ASCAD
cd "$REPO_ROOT"
M=STM32_AES_v2/ASCAD_data/ASCAD_trained_models
O=experiments/out/t039
mkdir -p "$O"
exec > >(tee -a "$O/pipeline.log") 2>&1

COMMON=(--compact --bottleneck-dim 0 --head-hidden-dim 64 --batch-size 64 --lr 1e-4
        --sub-avg --c-loss-weight 0.5 --save-last)

score() {  # $1 label, $2 checkpoint
    local label=$1 ckpt=$2
    if [[ "$ckpt" == *_last.pth ]]; then
        local base=${ckpt%_last.pth}.pth
        cp -n "$base.config.json" "$ckpt.config.json"
        cp -n "$base.mean_wf.npy" "$ckpt.mean_wf.npy"
    fi
    echo "=== [$(date -Is)] score $label ($ckpt)"
    "$PY" -m experiments.w6_nochead_eval --label "$label" --model-file "$ckpt" \
        --save-npz "$O/${label}_heads.npz" --out-json "$O/${label}_eval.json" \
        || { echo "!!! w6 eval failed: $label"; return 1; }
    "$PY" experiments/t039_controls.py --label "$label" --model-file "$ckpt" \
        --npz "$O/${label}_heads.npz" --out-json "$O/${label}_controls.json" \
        || echo "!!! controls failed: $label"
}

carriage() {  # $1 label
    "$PY" experiments/c4_oracle_swaps.py \
        --add-model "$1:$O/${1}_heads.npz" --models "$1" \
        --variants A0_as_trained,B3_oracle_C,E3_all_but_C,D0_oracle_all \
        --out-json "$O/${1}_c4.json" || echo "!!! c4 failed: $1"
}

echo "=== [$(date -Is)] T-039 pipeline start"

# ---------------- E1: 300k seed 2 ----------------
E1=$M/compact_mresnet_b_cw05_t300k_seed2.pth
if [ ! -f "$E1" ]; then
    echo "=== [$(date -Is)] E1 train 300k seed 2"
    "$PY" MultiResNet/train.py "${COMMON[@]}" --train-len 300000 --epochs 15 --seed 2 \
        --out-model "$E1" > "$O/e1_t300k_seed2_train.log" 2>&1
    echo "=== [$(date -Is)] E1 train exit $?"
fi
for s in 1 2; do
    B=$M/compact_mresnet_b_cw05_t300k_seed$s
    for v in best last; do
        ck=$B.pth; [ "$v" = last ] && ck=${B}_last.pth
        [ -f "$ck" ] || { echo "!!! missing $ck"; continue; }
        score "e1_t300k_s${s}_$v" "$ck" && carriage "e1_t300k_s${s}_$v"
    done
done

# ---------------- E3: label efficiency ----------------
for s in 1 2; do  # N = 50k reference: T-033's two seeds, re-scored with the same controls
    for v in best last; do
        ck=experiments/out/compact_multiresnet_direct_h64_seed$s.pth
        [ "$v" = last ] && ck=experiments/out/compact_multiresnet_direct_h64_seed${s}_last.pth
        score "e3_n50000_s${s}_$v" "$ck"
    done
done
for spec in 1000:750 5000:150 20000:38; do
    N=${spec%:*}; EP=${spec#*:}
    for s in 1 2; do
        ck=$M/t039_labeff_n${N}_seed$s.pth
        if [ ! -f "$ck" ]; then
            echo "=== [$(date -Is)] E3 train N=$N epochs=$EP seed $s"
            "$PY" MultiResNet/train.py "${COMMON[@]}" --train-len "$N" --epochs "$EP" --seed "$s" \
                --out-model "$ck" > "$O/e3_n${N}_seed${s}_train.log" 2>&1
            echo "=== [$(date -Is)] E3 train N=$N seed $s exit $?"
        fi
        [ -f "$ck" ] || { echo "!!! missing $ck"; continue; }
        score "e3_n${N}_s${s}_best" "$ck"
        score "e3_n${N}_s${s}_last" "${ck%.pth}_last.pth"
    done
done

echo "=== [$(date -Is)] T-039 pipeline done"
touch "$O/DONE"
