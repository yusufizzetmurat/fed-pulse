#!/usr/bin/env bash
# Stage-2 HP refine (F) + F2(top-5 × bge). E/E2 moved laptop-side.
# F: 40 random samples from the FULL 5-dim grid
#   hidden{32,64,128} × ralpha{0.3,0.5,0.7} × gamma{1,1.5,2} × dropout{0.1,0.2,0.3} × lr{1e-4,3e-4,1e-3}
# Locked: --no-mp-surprise --use-doc-length --regime-loss focal. (dropout/lr via local CLI tweak.)
# Stop: THRESHOLD_HIT marker if a cell dual>=0.50 or cls>=0.45; 8h hard cap -> stop + leave partial.
set -uo pipefail
export PATH="/root/shim:$PATH"; cd /root/fed-pulse
TS=$(date -u +%Y%m%dT%H%M%SZ); START=$(date +%s); DEADLINE=$((START + 28800))
ART="backend/artifacts/experiments"; LOG="$HOME/runpod-sweep-logs"
SUM="$LOG/_stage2_summary_${TS}.log"; CSV="$LOG/stage2_ranking_${TS}.csv"
CFG="$LOG/stage2_configs_${TS}.txt"
note(){ printf '[stage2] %s\n' "$*" | tee -a "$SUM"; }
S="11 29 47 71 97"; EP=40
ok(){ [ "$(date +%s)" -lt "$DEADLINE" ]; }
ex(){ PATH=/root/shim:$PATH python - "$1" <<'PY'
import json,sys,math
try:
    s=json.load(open(sys.argv[1]))["summary"]
    def m(h):
        v=s.get(h,{}).get("regime_f1_macro"); return v["mean"] if isinstance(v,dict) else float("nan")
    du,cl=m("dual"),m("classification")
    gm=math.sqrt(du*cl) if du==du and cl==cl and du>0 and cl>0 else float("nan")
    print(f"{du:.4f} {cl:.4f} {gm:.4f}")
except Exception: print("nan nan nan")
PY
}
run(){ local key=$1; shift; note "===== $key start $(date -u) ====="; if "$@" 2>&1 | tee "$LOG/stage2_${key}_${TS}.log" >/dev/null; then note "$key OK"; return 0; else note "$key FAILED"; return 1; fi; }

# deterministic 40-sample draw from the 243-combo 5-dim grid
PATH=/root/shim:$PATH python - "$CFG" <<'PY'
import random, itertools, sys
random.seed(42)
grid=list(itertools.product([32,64,128],[0.3,0.5,0.7],[1.0,1.5,2.0],[0.1,0.2,0.3],[1e-4,3e-4,1e-3]))
with open(sys.argv[1],"w") as f:
    for hs,ra,fg,do,lr in random.sample(grid,40):
        f.write(f"{hs} {ra} {fg} {do} {lr}\n")
PY

note "start $(date -u) TS=$TS HEAD=$(git rev-parse --short HEAD) hardcap=+8h  (F=40 samples 5-dim)"
printf 'cell,hidden,ralpha,gamma,dropout,lr,dual,cls,geomean\n' > "$CSV"

# --- F: 40-sample sweep -----------------------------------------------------
while read -r hs ra fg dp lr; do
  ok || { note "8h cap reached; stopping F sweep"; break; }
  cell="h${hs}_a${ra}_g${fg}_d${dp}_l${lr}"; out="$ART/runpod_stage2_${cell}_${TS}.json"
  if run "F_${cell}" python -m scripts.run_dual_head_comparison --training-package-id canonical \
      --output "$out" --seeds $S --epochs $EP --no-mp-surprise --use-doc-length --regime-loss focal \
      --hidden-size "$hs" --regression-alpha "$ra" --focal-gamma "$fg" --dropout "$dp" --learning-rate "$lr"; then
    read du cl gm <<< "$(ex "$out")"
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' "$cell" "$hs" "$ra" "$fg" "$dp" "$lr" "$du" "$cl" "$gm" >> "$CSV"
    note "F ${cell}: dual=$du cls=$cl geomean=$gm"
    awk -v d="$du" -v c="$cl" 'BEGIN{exit !((d+0)>=0.50 || (c+0)>=0.45)}' && { note "*** THRESHOLD_HIT ${cell} dual=$du cls=$cl ***"; touch "$LOG/THRESHOLD_HIT_${TS}"; }
  fi
done < "$CFG"

note "===== F ranking by geomean ====="
{ head -1 "$CSV"; tail -n +2 "$CSV" | sort -t, -k9 -gr; } | tee -a "$SUM"

# --- F2: top-5 × bge --------------------------------------------------------
note "===== F2: top-5 cells × bge ====="
mapfile -t top5 < <(tail -n +2 "$CSV" | sort -t, -k9 -gr | head -5)
for row in "${top5[@]}"; do
  ok || { note "8h cap; stopping F2"; break; }
  IFS=, read cell hs ra fg dp lr du cl gm <<< "$row"
  out="$ART/runpod_stage2_bge_${cell}_${TS}.json"
  if run "F2_${cell}_bge" python -m scripts.run_dual_head_comparison --training-package-id canonical \
      --output "$out" --seeds $S --epochs $EP --no-mp-surprise --use-doc-length --regime-loss focal \
      --hidden-size "$hs" --regression-alpha "$ra" --focal-gamma "$fg" --dropout "$dp" --learning-rate "$lr" \
      --text-encoder bge_large_en_v15 --use-text-embeddings; then
    note "F2 ${cell}+bge: $(ex "$out")"
  fi
done

note "===== STAGE2 DONE $(date -u) ====="; note "STAGE2_COMPLETE_MARKER"
