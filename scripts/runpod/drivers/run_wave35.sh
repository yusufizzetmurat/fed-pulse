#!/usr/bin/env bash
set -uo pipefail
export PATH="/root/shim:$PATH"
cd /root/fed-pulse
TS=$(date -u +%Y%m%dT%H%M%SZ)
ART="backend/artifacts/experiments"; LOG="$HOME/runpod-sweep-logs"
SUM="$LOG/_wave35_summary_${TS}.log"
note(){ printf '[wave3.5] %s\n' "$*" | tee -a "$SUM"; }
note "start $(date -u) HEAD=$(git rev-parse --short HEAD)"
run(){ local k=$1; shift; note "===== $k start $(date -u) ====="; if "$@" 2>&1 | tee "$LOG/wave35_${k}_${TS}.log"; then note "$k OK"; else note "$k FAILED"; fi; }
run focal_loss python -m scripts.run_dual_head_comparison --training-package-id canonical \
    --output $ART/runpod_wave3_focal_loss_${TS}.json --seeds 11 29 47 71 97 --epochs 40 \
    --regime-loss focal --focal-gamma 2.0
run class_balanced python -m scripts.run_dual_head_comparison --training-package-id canonical \
    --output $ART/runpod_wave3_class_balanced_${TS}.json --seeds 11 29 47 71 97 --epochs 40 \
    --regime-loss class_balanced --class-balanced-beta 0.999
note "===== wave3.5 finished $(date -u) ====="; note "WAVE35_COMPLETE_MARKER"
ls -la $ART/runpod_wave3_focal_loss_${TS}.json $ART/runpod_wave3_class_balanced_${TS}.json 2>&1 | tee -a "$SUM"
