#!/usr/bin/env bash
set -uo pipefail
export PATH="/root/shim:$PATH"; cd /root/fed-pulse
TS=$(date -u +%Y%m%dT%H%M%SZ); ART="backend/artifacts/experiments"; LOG="$HOME/runpod-sweep-logs"
SUM="$LOG/_job1_summary_${TS}.log"
note(){ printf '[job1] %s\n' "$*" | tee -a "$SUM"; }
ex(){ PATH=/root/shim:$PATH python - "$1" <<'PY'
import json,sys
s=json.load(open(sys.argv[1]))["summary"]
def m(h):
    v=s.get(h,{}).get("regime_f1_macro"); return f"{v['mean']:.4f}" if isinstance(v,dict) else "na"
def r(h):
    v=s.get(h,{}).get("regression_rmse_log_rv"); return f"{v['mean']:.4f}" if isinstance(v,dict) else "na"
print(f"dual_f1={m('dual')} cls_f1={m('classification')} reg_rmse={r('dual')}")
PY
}
note "start $(date -u) HEAD=$(git rev-parse --short HEAD) TS=$TS"
for H in 5 10 20; do
  out="$ART/pod_horizon_${H}d_${TS}.json"
  note "===== horizon ${H}d start $(date -u) ====="
  if python -m scripts.run_dual_head_comparison --training-package-id canonical \
      --target-horizon $H --output "$out" --seeds 11 29 47 71 97 --epochs 40 \
      2>&1 | tee "$LOG/job1_h${H}d_${TS}.log" >/dev/null; then
    note "horizon ${H}d OK: $(ex "$out")"
  else note "horizon ${H}d FAILED"; fi
done
note "===== JOB1 DONE $(date -u) ====="; note "JOB1_COMPLETE_MARKER"
