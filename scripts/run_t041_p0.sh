#!/bin/bash
# T-041 P0 (CPU): permutation-aware pi routing at attack time. Pre-registered in docs/tasks/T041-*.md.
set -u
PY=/home/lantian/miniconda3/envs/py10/bin/python
cd /home/lantian/ASCAD
O=experiments/out/t041
H5=STM32_AES_v2/ASCAD_data/ASCAD_databases/ascadv2-extracted.h5
exec > >(tee -a "$O/pipeline.log") 2>&1
declare -A SRC=(
  [e1_s1]=experiments/out/t039/e1_t300k_s1_best_heads.npz
  [e1_s2]=experiments/out/t039/e1_t300k_s2_best_heads.npz
  [e3_s1]=experiments/out/t039/e3_n50000_s1_best_heads.npz
  [e3_s2]=experiments/out/t039/e3_n50000_s2_best_heads.npz
  [r0_s1]=experiments/out/t039_e6/e6_R0_s1_best_heads.npz
  [r0_s2]=experiments/out/t039_e6/e6_R0_s2_best_heads.npz
  [r1t_s2]=experiments/out/t039_e6/e6_R1T300k_s2_best_heads.npz)
for m in r1t_s2 e3_s1 e3_s2 r0_s1 r0_s2 e1_s1 e1_s2; do
  echo "=== [$(date -Is)] routings $m"
  "$PY" experiments/t041_p0_routing.py --npz "${SRC[$m]}" --label "$m" --out-dir "$O" --perm-labels "$H5" || { echo "!!! routing failed $m"; continue; }
  ADD=(); NAMES=""
  for r in raw hung sink exact; do ADD+=(--add-model "${m}_$r:$O/${m}_${r}_heads.npz"); NAMES+="${m}_$r,"; done
  "$PY" experiments/c4_oracle_swaps.py "${ADD[@]}" --models "${NAMES%,}" \
      --variants A0_as_trained,B4_oracle_pi,F2_uniform_pi --out-json "$O/${m}_c4.json" > "$O/${m}_c4.log" 2>&1 \
      || echo "!!! c4 failed $m"
  grep -E "A0_as_trained|B4_oracle_pi|F2_uniform_pi" "$O/${m}_c4.log" | grep -v "_hung\].*B4\|_sink\].*B4\|_exact\].*B4\|_hung\].*F2\|_sink\].*F2\|_exact\].*F2"
done
echo "=== [$(date -Is)] T-041 P0 done"; touch "$O/DONE_P0"
