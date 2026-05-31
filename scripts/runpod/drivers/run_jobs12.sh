#!/usr/bin/env bash
# JOB 1 (#499 horizon 5/10/20, via --target-horizon, direct) then
# JOB 2 (#217 arch ensemble, direct script + aggregator). 17:30Z budget guard
# before JOB 2 (ship JOB 1 alone if past). JOB 3 (VIX) blocked on HF dataset rev.
set -uo pipefail
export PATH="/root/shim:$PATH"; cd /root/fed-pulse
TS=$(date -u +%Y%m%dT%H%M%SZ)
ART="backend/artifacts/experiments"; LOG="$HOME/runpod-sweep-logs"
SUM="$LOG/_jobs12_summary_${TS}.log"
GUARD=$(date -u -d 'today 17:30' +%s)
note(){ printf '[jobs12] %s\n' "$*" | tee -a "$SUM"; }
ex(){ PATH=/root/shim:$PATH python - "$1" <<'PY'
import json,sys
try:
    s=json.load(open(sys.argv[1]))["summary"]
    def m(h,k):
        v=s.get(h,{}).get(k); return f"{v['mean']:.4f}" if isinstance(v,dict) else "na"
    print(f"dual_f1={m('dual','regime_f1_macro')} cls_f1={m('classification','regime_f1_macro')} reg_rmse={m('regression','regression_rmse_log_rv')}")
except Exception as e: print("parse-error",e)
PY
}
run(){ local k=$1; shift; note "===== $k start $(date -u) ====="; if "$@" 2>&1 | tee "$LOG/jobs12_${k}_${TS}.log" >/dev/null; then note "$k OK"; return 0; else note "$k FAILED"; return 1; fi; }

note "start $(date -u) TS=$TS HEAD=$(git rev-parse --short HEAD) guard=17:30Z"

# --- JOB 1: horizon sensitivity 5/10/20 -------------------------------------
note "########## JOB 1 — #499 horizon 5/10/20 ##########"
for H in 5 10 20; do
  out="$ART/pod_horizon_${H}d_${TS}.json"
  if run "horizon_${H}d" python -m scripts.run_dual_head_comparison --training-package-id canonical \
      --target-horizon "$H" --output "$out" --seeds 11 29 47 71 97 --epochs 40; then
    note "horizon_${H}d: $(ex "$out")"
  fi
done
note "JOB1 horizon table:"
for H in 5 10 20; do note "  ${H}d -> $(ex "$ART/pod_horizon_${H}d_${TS}.json" 2>/dev/null)"; done
note "JOB1_DONE"

# --- JOB 2: arch ensemble (budget-guarded) ----------------------------------
if [ "$(date -u +%s)" -lt "$GUARD" ]; then
  note "########## JOB 2 — #217 arch ensemble (5 archs × 5 samples) ##########"
  ASD="$LOG/job2_arch_sweep_${TS}"
  if run arch_sweep python scripts/run_regime_architecture_sweep.py --training-package-id canonical \
      --architectures lstm lstm_attn gru tcn transformer --random-search-samples 5 \
      --no-llm-features --report-root "$ASD"; then
    run ensemble_aggregate python -m app.evaluation.ensemble_aggregator \
      --arch-sweep-dir "$ASD" --output-json "$ART/pod_arch_ensemble_${TS}.json" || true
    [ -f "$ART/pod_arch_ensemble_${TS}.json" ] && note "arch ensemble JSON written: pod_arch_ensemble_${TS}.json"
  fi
  note "JOB2_DONE"
else
  note "17:30Z guard passed before JOB 2 — shipping JOB 1 only (JOB2 skipped)"
fi

note "===== JOBS12 DONE $(date -u) ====="; note "JOBS12_COMPLETE_MARKER"
