#!/usr/bin/env bash
# Pre-run the slowest of the 9 green arms (canonical) at full seeds/epochs while
# #524 + the LM dict land. Does NOT touch the statement_delta/vote/press code
# path. Output name matches launch_sweeps.sh so sync_results.sh picks it up; this
# arm is therefore EXCLUDED from the later 12-arm batch.
set -uo pipefail
export PATH="/root/shim:$PATH"          # bake the 3.12 shim in (tmux env-gotcha safe)
cd /root/fed-pulse

TS=$(date -u +%Y%m%dT%H%M%SZ)
ART="backend/artifacts/experiments"
LOG_DIR="$HOME/runpod-sweep-logs"
mkdir -p "$ART" "$LOG_DIR"
LOG="$LOG_DIR/canonical_${TS}.log"
OUT="$ART/runpod_canonical_${TS}.json"

{
  echo "[canonical] start $(date -u)  HEAD=$(git rev-parse --short HEAD)  python=$(command -v python) ($(python --version 2>&1))"
  echo "[canonical] output -> $OUT"
  python -m scripts.run_dual_head_comparison \
      --training-package-id canonical \
      --output "$OUT" \
      --seeds 11 29 47 71 97 \
      --epochs 40 \
      --regression-alpha 0.5
  rc=$?
  echo "[canonical] exit=$rc $(date -u)"
  if [ "$rc" -eq 0 ] && [ -f "$OUT" ]; then
    echo "[canonical] OK artefact=$OUT"
  else
    echo "[canonical] FAILED rc=$rc (artefact present: $([ -f "$OUT" ] && echo yes || echo no))"
  fi
  echo "CANONICAL_DONE_MARKER rc=$rc"
} 2>&1 | tee "$LOG"
