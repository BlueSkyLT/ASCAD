#!/bin/bash
# T-039 held-out check (2026-11-01, user go-ahead "可以,请进行验证"; pre-registration "HOLD-OUT CHECK FROZEN" in the
# T-039 task file). Attack traces 5000-9999. Per checkpoint: w6 inference (GPU) -> heads npz; t039_controls on the
# original heads, then on the bit-projected heads (primary); afterwards the matched-temperature control and HW.
set -u
PY=/home/lantian/miniconda3/envs/py10/bin/python
cd /home/lantian/ASCAD
M=STM32_AES_v2/ASCAD_data/ASCAD_trained_models
H=experiments/out/t039_e6/holdout
S=5000
mkdir -p "$H"
exec > >(tee -a "$H/holdout.log") 2>&1

declare -a JOBS=()
for s in 1 2; do JOBS+=("e1_t300k_s${s}|$M/compact_mresnet_b_cw05_t300k_seed${s}"); done
for m in RCpiT300k RCpiUT300k 3headT300k 3headAT300k R1T300k; do
  for s in 1 2; do JOBS+=("${m}_s${s}|$M/t039_e6_${m}_seed${s}"); done
done
for m in R2T300k 3headWT300k 3headBitT300k 3headABT300k; do JOBS+=("${m}_s1|$M/t039_e6_${m}_seed1"); done
TEMP_SET=" RCpiT300k_s1 RCpiT300k_s2 RCpiUT300k_s1 RCpiUT300k_s2 3headT300k_s1 3headT300k_s2 3headAT300k_s1 3headAT300k_s2 3headWT300k_s1 R1T300k_s1 "

controls() {  # $1 tag, $2 ckpt, $3 npz, $4 project-c or ""
  [ -f "$H/$1_controls.json" ] && { echo "skip $1"; return; }
  $PY -m experiments.t039_controls --label "$1" --model-file "$2" --npz "$3" --attack-start $S \
      ${4:+--project-c "$4"} --out-json "$H/$1_controls.json" 2>&1 | grep -E "^\[|Error|error"
}

transformed() {  # $1 label_which, $2 ckpt, $3 kind
  local tag=$1_${3/:/}
  [ -f "$H/${tag}_controls.json" ] && { echo "skip $tag"; return; }
  local kk
  kk=$($PY experiments/t039_c_projection_npz.py "$H/$1_heads.npz" "$H/$tag.npz" "$3" | sed -n 's/.*(\(.*\))$/\1/p')
  echo "$tag -> $kk"
  controls "$tag" "$2" "$H/$tag.npz" "$kk"
  rm -f "$H/$tag.npz"
}

for pass in main temp hw; do
  for which in best last; do
    for job in "${JOBS[@]}"; do
      IFS='|' read -r label ckbase <<< "$job"
      if [ "$which" = best ]; then ck=$ckbase.pth; else ck=${ckbase}_last.pth; fi
      lw=${label}_${which}
      if [ "$pass" = main ]; then
        if [ ! -f "$H/${lw}_heads.npz" ]; then
          echo "=== [$(date -Is)] infer $lw ($ck)"
          $PY -m experiments.w6_nochead_eval --label "${lw}_ho" --model-file "$ck" --attack-start $S \
              --save-npz "$H/${lw}_heads.npz" --out-json "$H/${lw}_eval.json" 2>&1 | grep -E "as-trained|Error|error"
          if [[ "$label" == RCpiUT300k* ]]; then   # untrained pi heads: exactly uniform pi, as in run_t039_e6.sh
            $PY - "$H/${lw}_heads.npz" <<'PYEOF'
import sys, shutil, numpy as np
p = sys.argv[1]; shutil.copy(p, p.replace(".npz", "_rawpi.npz"))
d = dict(np.load(p))
for i in range(16):
    d[f"permind_{i}_output"] = np.full_like(d[f"permind_{i}_output"], 1.0 / 16)
np.savez(p, **d)
PYEOF
          fi
        fi
        controls "${lw}_orig" "$ck" "$H/${lw}_heads.npz" ""
        transformed "$lw" "$ck" bit
      elif [ "$pass" = temp ]; then
        [[ "$TEMP_SET" == *" $label "* ]] && transformed "$lw" "$ck" tempmatch:bit
      else
        transformed "$lw" "$ck" hw
      fi
    done
  done
  echo "=== [$(date -Is)] pass $pass done"
done
echo "=== [$(date -Is)] HOLDOUT ALLDONE"
