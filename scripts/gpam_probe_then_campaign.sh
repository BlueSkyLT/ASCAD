#!/usr/bin/env bash
# Gate the 83-hour 34-head campaign on a ~80-minute evidence check.
#
# Probe: 40k raw traces, 8 epochs, 4 heads, lr 1e-3, held-out eval on shard 2.
# This is the IDENTICAL budget to the crop run that reached rm 92.95%, differing
# only in input length (1M vs 15k samples), so it isolates the dilution question.
#
# Gate: held-out rm accuracy must clear 2% (5x the 0.391% chance level) before
# spending multi-day GPU time on the full campaign. Every prior positive number
# for raw was train=eval memorisation, which a 17.8M-param model also achieves on
# SHUFFLED labels.
set -uo pipefail
cd /home/lantian/ASCAD

PY=/home/lantian/miniconda3/envs/py10/bin/python
LOGDIR=GPAM/artifacts/logs_raw
PROBE_LOG="$LOGDIR/probe40k_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$LOGDIR"

notify() {
  "$PY" -c "
import sys
sys.path.insert(0, 'scripts')
from send_telegram import send_message
send_message(sys.argv[1])
" "$1" || echo "telegram send failed (continuing)"
}

echo "[$(date '+%F %T')] probe start: 40k raw traces, 8 epochs, lr 1e-3, eval shard2" | tee -a "$PROBE_LOG"
"$PY" -u GPAM/artifacts/C-raw-generalization-test.py \
  --lr 0.001 --n-train 40000 --n-eval 1024 --epochs 8 >> "$PROBE_LOG" 2>&1
PROBE_EXIT=$?
echo "[$(date '+%F %T')] probe exit=$PROBE_EXIT" | tee -a "$PROBE_LOG"

VERDICT=$(grep -aoE "PASSES|FAILS" "$PROBE_LOG" | tail -1 || true)
RMROW=$(grep -aE "shard-2 held-out rm acc" "$PROBE_LOG" | tail -1 || true)
LASTEP=$(grep -aoE "val rm +[0-9.]+%" "$PROBE_LOG" | tail -1 || true)

if [ "$PROBE_EXIT" -ne 0 ] || [ "$VERDICT" != "PASSES" ]; then
  notify "❌ [ASCAD GPAM] 40k raw probe FAILED - full campaign NOT launched.
${RMROW}
last epoch ${LASTEP} (chance 0.391%)
exit=${PROBE_EXIT}
Log: ${PROBE_LOG}
Meaning: at the exact budget that gave 92.95% on the 15k POI crop, raw 1M-point traces do not transfer. The 83h campaign is unjustified."
  echo "[$(date '+%F %T')] gate FAILED - campaign not launched" | tee -a "$PROBE_LOG"
  exit 0
fi

notify "✅ [ASCAD GPAM] 40k raw probe PASSED (${LASTEP}) - launching the 34-head raw campaign at lr 1e-3, ~83h.
Log: GPAM/artifacts/logs_raw/campaign34_1e-3.log"
echo "[$(date '+%F %T')] gate PASSED - launching 34-head campaign at lr 1e-3" | tee -a "$PROBE_LOG"

setsid nohup "$PY" -u GPAM/train_gpam_raw_full.py \
  --target-bytes 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 \
  --epochs 150 --steps-per-epoch 2000 --val-steps 50 \
  --batch-size 32 --chunk-size 128 --lr 1e-3 --patch-size 400 \
  --checkpoint-dir GPAM/artifacts/checkpoints_raw --log-dir GPAM/artifacts/logs_raw \
  --report-interval-sec 3600 --eval-holdout-batches 2500 \
  > GPAM/artifacts/logs_raw/campaign34_1e-3.log 2>&1 < /dev/null &
echo "[$(date '+%F %T')] campaign pid $!" | tee -a "$PROBE_LOG"
