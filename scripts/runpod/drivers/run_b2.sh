#!/usr/bin/env bash
set -uo pipefail
export PATH="/root/shim:$PATH"; cd /root/fed-pulse
TS=$(date -u +%Y%m%dT%H%M%SZ)
ART="backend/artifacts/experiments"; LOG="$HOME/runpod-sweep-logs"
SUM="$LOG/_b2_summary_${TS}.log"
note(){ printf '[b2] %s\n' "$*" | tee -a "$SUM"; }
run(){ local k=$1; shift; note "===== $k start $(date -u) ====="; if "$@" 2>&1 | tee "$LOG/b2_${k}_${TS}.log"; then note "$k OK"; else note "$k FAILED"; fi; }
note "start $(date -u) HEAD=$(git rev-parse --short HEAD)"
run xbank_aux_stance_masked python -m app.data.finetune_pilot_b2 \
    --training-package-id canonical --output $ART/runpod_b2_xbank_aux_${TS}.json \
    --encoder-alias finbert_fed_adjacent_xbank_aux_stance_masked --seeds 11 29 47 71 97 --epochs 5
run xbank_aux_weighted python -m app.data.finetune_pilot_b2 \
    --training-package-id canonical --output $ART/runpod_b2_xbank_aux_weighted_${TS}.json \
    --encoder-alias finbert_fed_adjacent_xbank_aux_weighted --seeds 11 29 47 71 97 --epochs 5
note "===== b2 finished $(date -u) ====="; note "B2_COMPLETE_MARKER"
ls -la $ART/runpod_b2_xbank_aux_${TS}.json $ART/runpod_b2_xbank_aux_weighted_${TS}.json 2>&1 | tee -a "$SUM"
