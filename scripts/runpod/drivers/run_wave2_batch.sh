#!/usr/bin/env bash
# Wave 2 runpod batch driver — 7 new arms against the canonical TP.
#
# Arms (canonical re-run + 6 new):
#   canonical_v2         baseline dual-head canonical with the post-#529/#531 runner
#   garch_residual       --vol-target-mode garch_residual on dual head
#   horizon_5d           --target-horizon 5  (forward_realized_vol_5d target)
#   horizon_20d          --target-horizon 20 (forward_realized_vol_20d target)
#   ordinal_ce           --regime-loss ordinal_ce (bin-distance-weighted CE)
#   confounder_ablation  5-cell ablation: baseline + year_fe + meeting_type_fe + doc_length + all_three
#   arch_sweep           regime architecture sweep across 5 archs (LSTM / LSTM-attn / GRU / TCN / Transformer)
#
# Resilient serial driver: each arm runs in isolation; one failure does not
# abort the rest. Output JSONs land at $ART/runpod_wave2_<arm>_<TS>.json.

set -uo pipefail

export PATH="/root/shim:$PATH"
cd /root/fed-pulse

TS=$(date -u +%Y%m%dT%H%M%SZ)
ART="backend/artifacts/experiments"
LOG_DIR="$HOME/runpod-sweep-logs"
mkdir -p "$ART" "$LOG_DIR"
SUMMARY="$LOG_DIR/_wave2_summary_${TS}.log"

note() { printf '[wave2] %s\n' "$*" | tee -a "$SUMMARY"; }

note "start $(date -u) TS=$TS host=$(hostname)"
note "python -> $(PATH=/root/shim:$PATH command -v python) ($(PATH=/root/shim:$PATH python --version 2>&1))"
note "dev rev -> $(git rev-parse --short HEAD)"

SEEDS="${SEEDS:-11 29 47 71 97}"
EPOCHS="${EPOCHS:-40}"

run_arm() {
    local key=$1
    shift
    local log_file="$LOG_DIR/wave2_${key}_${TS}.log"
    note "===== $key start $(date -u) ====="
    if "$@" 2>&1 | tee "$log_file"; then
        note "$key OK -> $log_file"
    else
        note "$key FAILED (continuing) — see $log_file"
    fi
}

# --- canonical_v2 (re-baseline against the post-#529/#531 runner) -----------
run_arm canonical_v2 \
    python -m scripts.run_dual_head_comparison \
        --training-package-id canonical \
        --output $ART/runpod_wave2_canonical_v2_${TS}.json \
        --seeds $SEEDS --epochs $EPOCHS \
        --regression-alpha 0.5

# --- garch_residual ---------------------------------------------------------
run_arm garch_residual \
    python -m scripts.run_dual_head_comparison \
        --training-package-id canonical \
        --output $ART/runpod_wave2_garch_residual_${TS}.json \
        --seeds $SEEDS --epochs $EPOCHS \
        --vol-target-mode garch_residual

# --- horizon_5d -------------------------------------------------------------
run_arm horizon_5d \
    python -m scripts.run_dual_head_comparison \
        --training-package-id canonical \
        --output $ART/runpod_wave2_horizon_5d_${TS}.json \
        --seeds $SEEDS --epochs $EPOCHS \
        --target-horizon 5

# --- horizon_20d ------------------------------------------------------------
run_arm horizon_20d \
    python -m scripts.run_dual_head_comparison \
        --training-package-id canonical \
        --output $ART/runpod_wave2_horizon_20d_${TS}.json \
        --seeds $SEEDS --epochs $EPOCHS \
        --target-horizon 20

# --- ordinal_ce -------------------------------------------------------------
run_arm ordinal_ce \
    python -m scripts.run_dual_head_comparison \
        --training-package-id canonical \
        --output $ART/runpod_wave2_ordinal_ce_${TS}.json \
        --seeds $SEEDS --epochs $EPOCHS \
        --regime-loss ordinal_ce

# --- confounder_ablation (5 cells inside one runner) ------------------------
run_arm confounder_ablation \
    python -m scripts.run_confounder_ablation \
        --training-package-id canonical \
        --output $ART/runpod_wave2_confounder_ablation_${TS}.json \
        --seeds $SEEDS --epochs $EPOCHS \
        --cells baseline year_fe meeting_type_fe doc_length all_three

# --- arch_sweep (stage-1 screening) -----------------------------------------
# 5 archs (lstm / lstm_attn / gru / tcn / transformer), 5 HP samples each
# (stage-1 coarse screen, not full HP refinement). report-root under LOG_DIR so
# sync_results.sh bundles it; ensemble_aggregator then writes one wave-2 JSON.
# NB: aggregator flag is --output-json (not --output); report-root via rglob
# picks up the <root>/canonical/<arch>/forecaster_sweep_results.json layout.
run_arm arch_sweep \
    bash -c "PATH=/root/shim:\$PATH python scripts/run_regime_architecture_sweep.py \
        --training-package-id canonical \
        --architectures lstm lstm_attn gru tcn transformer \
        --random-search-samples 5 \
        --no-llm-features \
        --report-root $LOG_DIR/arch_sweep_${TS} \
     && PATH=/root/shim:\$PATH python -m app.evaluation.ensemble_aggregator \
        --arch-sweep-dir $LOG_DIR/arch_sweep_${TS} \
        --output-json $ART/runpod_wave2_arch_sweep_${TS}.json"

note "===== wave2 batch finished $(date -u) ====="
note "WAVE2_BATCH_COMPLETE_MARKER"
ls -la "$ART"/runpod_wave2_*_${TS}.json 2>&1 | tee -a "$SUMMARY"
