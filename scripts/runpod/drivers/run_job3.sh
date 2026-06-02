#!/usr/bin/env bash
# JOB 3 — §6.41 VIX-on encoder bake-off (control for the text-as-vol-proxy claim).
# Waits for JOB1+JOB2 (JOBS12_COMPLETE_MARKER), then runs on canonical_vix with
# --use-vix-features: baseline (no text) + bge/finbert/xbank (+text). nomic/voyage
# remain run-path-blocked (trust_remote_code / API from_pretrained), skipped.
# 17:30Z budget guard before each arm.
set -uo pipefail
export PATH="/root/shim:$PATH"; cd /root/fed-pulse
TS=$(date -u +%Y%m%dT%H%M%SZ)
ART="backend/artifacts/experiments"; LOG="$HOME/runpod-sweep-logs"
SUM="$LOG/_job3_summary_${TS}.log"; GUARD=$(date -u -d 'today 17:30' +%s)
note(){ printf '[job3] %s\n' "$*" | tee -a "$SUM"; }
ex(){ PATH=/root/shim:$PATH python - "$1" <<'PY'
import json,sys
try:
    s=json.load(open(sys.argv[1]))["summary"]
    def m(h):
        v=s.get(h,{}).get("regime_f1_macro"); return f"{v['mean']:.4f}" if isinstance(v,dict) else "na"
    print(f"dual={m('dual')} cls={m('classification')}")
except Exception as e: print("err",e)
PY
}
run(){ local k=$1; shift; note "===== $k start $(date -u) ====="; if "$@" 2>&1 | tee "$LOG/job3_${k}_${TS}.log" >/dev/null; then note "$k OK"; return 0; else note "$k FAILED"; return 1; fi; }

# wait for JOB1+JOB2
J12=$(ls -t "$LOG"/_jobs12_summary_*.log 2>/dev/null | head -1)
note "waiting for JOB1+JOB2 (JOBS12_COMPLETE_MARKER) before JOB3..."
until [ -n "$J12" ] && grep -q JOBS12_COMPLETE_MARKER "$J12" 2>/dev/null; do sleep 30; J12=$(ls -t "$LOG"/_jobs12_summary_*.log 2>/dev/null | head -1); done
note "JOB1+2 done; starting JOB3 $(date -u) TS=$TS"

# VIX-on baseline (no text) — the new reference number
if [ "$(date -u +%s)" -lt "$GUARD" ]; then
  run vixon_baseline python -m scripts.run_dual_head_comparison --training-package-id canonical_vix \
    --use-vix-features --output "$ART/pod_vixon_baseline_${TS}.json" --seeds 11 29 47 71 97 --epochs 40
  [ -f "$ART/pod_vixon_baseline_${TS}.json" ] && note "vixon_baseline: $(ex "$ART/pod_vixon_baseline_${TS}.json")"
fi

# VIX-on + text, valid encoders only (caches built locally in salvage)
for enc in bge_large_en_v15 finbert_fed_adjacent finbert_fed_adjacent_xbank; do
  [ "$(date -u +%s)" -lt "$GUARD" ] || { note "17:30Z guard; stopping JOB3"; break; }
  out="$ART/pod_vixon_${enc}_${TS}.json"
  if run "vixon_${enc}" python -m scripts.run_dual_head_comparison --training-package-id canonical_vix \
      --use-vix-features --text-encoder "$enc" --use-text-embeddings \
      --output "$out" --seeds 11 29 47 71 97 --epochs 40; then
    note "vixon_${enc}: $(ex "$out")"
  fi
done
note "NOTE: nomic + voyage skipped (run-path blocked: trust_remote_code / API from_pretrained); independent of VIX."
note "===== JOB3 DONE $(date -u) ====="; note "JOB3_COMPLETE_MARKER"