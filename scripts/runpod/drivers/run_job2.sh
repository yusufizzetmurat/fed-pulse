#!/usr/bin/env bash
set -uo pipefail
export PATH="/root/shim:$PATH"; cd /root/fed-pulse
TS=$(date -u +%Y%m%dT%H%M%SZ); ART="backend/artifacts/experiments"; LOG="$HOME/runpod-sweep-logs"
SUM="$LOG/_job2_summary_${TS}.log"; RR="$LOG/arch_sweep_job2_${TS}"
note(){ printf '[job2] %s\n' "$*" | tee -a "$SUM"; }
note "start $(date -u) HEAD=$(git rev-parse --short HEAD) TS=$TS samples=8 (reduced from 20 for budget)"
note "===== arch sweep start $(date -u) ====="
if python scripts/run_regime_architecture_sweep.py --training-package-id canonical \
    --random-search-samples 8 --no-llm-features --report-root "$RR" > "$LOG/job2_archsweep_${TS}.log" 2>&1; then
  note "arch sweep OK -> $RR"
else note "arch sweep FAILED (see job2_archsweep_${TS}.log)"; fi
note "archs done: $(find "$RR" -name forecaster_sweep_results.json 2>/dev/null | sed 's#.*/##;s#/.*##' | wc -l)"
find "$RR" -name forecaster_sweep_results.json 2>/dev/null | sed 's#.*/canonical/##;s#/forecaster.*##' | tr '\n' ' ' | xargs -I{} note "arch dirs: {}"
note "===== ensemble aggregate $(date -u) ====="
if python -m app.evaluation.ensemble_aggregator --arch-sweep-dir "$RR" \
    --output-json "$ART/pod_arch_ensemble_${TS}.json" --output-markdown "$RR/ensemble_${TS}.md" \
    > "$LOG/job2_aggregate_${TS}.log" 2>&1; then
  note "ensemble OK -> pod_arch_ensemble_${TS}.json"
else note "ensemble FAILED (see job2_aggregate_${TS}.log)"; fi
note "===== JOB2 DONE $(date -u) ====="; note "JOB2_COMPLETE_MARKER"
