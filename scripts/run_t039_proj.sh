#!/bin/bash
# T-039 (2026-10-29, user go-ahead "先b后a"): post-hoc bit-linear / HW projection of the C posteriors on every cached
# [VALID 300k] checkpoint, scored with t039_controls (plain decode + works_v2 nulls; the constant-trace null is projected
# the same way via --project-c). No retraining; alpha, beta, pi untouched. best checkpoints first, then last.
# PROJ_KINDS='tempmatch:bit' PROJ_ONLY='...' PROJ_LOG=temp.log: structure-free control (C log-probs scaled to the
# bit projection's spread).
set -u
PY=/home/lantian/miniconda3/envs/py10/bin/python
cd /home/lantian/ASCAD
M=STM32_AES_v2/ASCAD_data/ASCAD_trained_models
O=experiments/out/t039_e6/proj
mkdir -p "$O"
exec > >(tee -a "$O/${PROJ_LOG:-controls.log}") 2>&1

declare -a JOBS=()
for s in 1 2; do JOBS+=("e1_t300k_s${s}|experiments/out/t039/e1_t300k_s${s}|$M/compact_mresnet_b_cw05_t300k_seed${s}"); done
for m in RCpiT300k RCpiUT300k 3headT300k 3headAT300k R1T300k; do
  for s in 1 2; do JOBS+=("${m}_s${s}|experiments/out/t039_e6/e6_${m}_s${s}|$M/t039_e6_${m}_seed${s}"); done
done
for m in R2T300k 3headWT300k 3headABT300k; do JOBS+=("${m}_s1|experiments/out/t039_e6/e6_${m}_s1|$M/t039_e6_${m}_seed1"); done

for which in best last; do
  for job in "${JOBS[@]}"; do
    [ -n "${PROJ_ONLY:-}" ] && [[ " $PROJ_ONLY " != *" ${job%%|*} "* ]] && continue
    IFS='|' read -r label npzbase ckbase <<< "$job"
    if [ "$which" = best ]; then ck=$ckbase.pth; else ck=${ckbase}_last.pth; fi
    for k in ${PROJ_KINDS:-bit hw}; do
      tag=${label}_${which}_${k/:/}
      [ -f "$O/${tag}_controls.json" ] && { echo "skip $tag"; continue; }
      kk=$($PY experiments/t039_c_projection_npz.py "${npzbase}_${which}_heads.npz" "$O/${tag}.npz" $k | tee /dev/stderr | sed -n 's/.*(\(.*\))$/\1/p')
      $PY -m experiments.t039_controls --label "$tag" --model-file "$ck" --npz "$O/${tag}.npz" \
          --project-c "$kk" --out-json "$O/${tag}_controls.json" 2>&1 | grep -E "^\[|Error|error"
      rm -f "$O/${tag}.npz"
    done
  done
done
echo "=== [$(date -Iseconds)] PROJ ALLDONE"
