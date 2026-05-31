#!/usr/bin/env bash
# Wave 4 — focal-gamma sweep + #81 encoder bake-off (runnable subset).
# voyage_finance_2 is EXCLUDED here (no VOYAGE_API_KEY / voyageai SDK on pod;
# run as a follow-on once the key is staged). B2 xbank-aux arms are blocked on
# the registry HF-SHA PR and handled separately.
set -uo pipefail
export PATH="/root/shim:$PATH"
cd /root/fed-pulse

TS=$(date -u +%Y%m%dT%H%M%SZ)
ART="backend/artifacts/experiments"; LOG_DIR="$HOME/runpod-sweep-logs"
mkdir -p "$ART" "$LOG_DIR"
SUMMARY="$LOG_DIR/_wave4_summary_${TS}.log"
note(){ printf '[wave4] %s\n' "$*" | tee -a "$SUMMARY"; }
run_arm(){ local k=$1; shift; local lf="$LOG_DIR/wave4_${k}_${TS}.log"; note "===== $k start $(date -u) ====="; if "$@" 2>&1 | tee "$lf"; then note "$k OK -> $lf"; else note "$k FAILED (continuing) -> $lf"; fi; }

note "start $(date -u) TS=$TS HEAD=$(git rev-parse --short HEAD)"
SEEDS="${SEEDS:-11 29 47 71 97}"; EPOCHS="${EPOCHS:-40}"

# --- focal-gamma sweep ------------------------------------------------------
for g in 0.5 1.0 2.0 5.0; do
    run_arm "focal_gamma_${g}" \
        python -m scripts.run_dual_head_comparison \
            --training-package-id canonical \
            --output "$ART/runpod_wave4_focal_gamma_${g}_${TS}.json" \
            --seeds $SEEDS --epochs $EPOCHS \
            --regime-loss focal --focal-gamma "$g"
done

# --- #81 encoder bake-off (voyage excluded) ---------------------------------
for enc in nomic_embed_text_v15 bge_large_en_v15 finbert_fed_adjacent; do
    run_arm "bakeoff_${enc}" \
        python -m scripts.run_dual_head_comparison \
            --training-package-id canonical \
            --output "$ART/runpod_wave4_bakeoff_${enc}_${TS}.json" \
            --seeds $SEEDS --epochs $EPOCHS \
            --text-encoder "$enc" --use-text-embeddings
done

note "===== wave4 finished $(date -u) ====="; note "WAVE4_BATCH_COMPLETE_MARKER"
ls -la "$ART"/runpod_wave4_*_${TS}.json 2>&1 | tee -a "$SUMMARY"
