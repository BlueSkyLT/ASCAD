#!/bin/bash
# T-039 400k Benchmark: CompactMultiResNet Variant B with Decoder Loss vs MS23 Baseline
# Profiling size: 400,000 traces (matching MS23 paper training size exactly)
# Epochs: 15, Batch Size: 64, Optimizer: RAdam (lr 1e-4)
# Config: Compact Variant B (15.65M, direct unshared slim heads h64), w_C = 0.5, decoder_loss_weight = 1.0
set -u

PY=/home/lantian/miniconda3/envs/py10/bin/python
REPO_ROOT=/home/lantian/ASCAD
cd "$REPO_ROOT"

M=STM32_AES_v2/ASCAD_data/ASCAD_trained_models
O=experiments/out/t039_400k
mkdir -p "$O"
exec > >(tee -a "$O/pipeline.log") 2>&1

COMMON=(--compact --bottleneck-dim 0 --head-hidden-dim 64 --batch-size 64 --lr 1e-4
        --sub-avg --c-loss-weight 0.5 --save-last --train-len 400000 --epochs 15
        --decoder-loss-weight 1.0)

score() {  # $1 label, $2 checkpoint
    local label=$1 ckpt=$2
    if [[ "$ckpt" == *_last.pth ]]; then
        local base=${ckpt%_last.pth}.pth
        cp -n "$base.config.json" "$ckpt.config.json" 2>/dev/null || true
        cp -n "$base.mean_wf.npy" "$ckpt.mean_wf.npy" 2>/dev/null || true
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

echo "=== [$(date -Is)] 400k Benchmark pipeline start"

for s in ${SEEDS:-1 2}; do
    ck=$M/compact_mresnet_b_cw05_dec_t400k_seed$s.pth
    if [ ! -f "$ck" ]; then
        echo "=== [$(date -Is)] Train CompactMultiResNet 400k + Decoder Loss (seed $s)"
        "$PY" MultiResNet/train.py "${COMMON[@]}" --seed "$s" \
            --out-model "$ck" > "$O/t400k_dec_seed${s}_train.log" 2>&1
        echo "=== [$(date -Is)] Train seed $s exit code $?"
    fi

    [ -f "$ck" ] || { echo "!!! missing $ck"; continue; }
    
    for v in best last; do
        target_ck=$ck
        [ "$v" = last ] && target_ck=${ck%.pth}_last.pth
        [ -f "$target_ck" ] || continue
        label="t400k_dec_s${s}_$v"
        score "$label" "$target_ck" && carriage "$label"
    done
done

echo "=== [$(date -Is)] 400k Benchmark pipeline finished successfully"
touch "$O/DONE"
