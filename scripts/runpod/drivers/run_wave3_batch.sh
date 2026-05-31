#!/usr/bin/env bash
# Wave 3 runpod batch driver — 7 new arms against the canonical TP +
# the seq_len_60 arm against the 60-bar TP variant.
#
# Arms:
#   focal_loss             --regime-loss focal --focal-gamma 2.0
#   class_balanced         --regime-loss class_balanced --class-balanced-beta 0.999
#   multi_horizon_aux      --aux-horizons 5,20 --aux-horizon-alpha 0.3
#   absolute_regime        --vol-regime-label-mode absolute (12% / 22% annualised)
#   vix_features           --use-vix-features (requires TP rebuild with VIX columns)
#   seq_len_60             --sequence-length 60 against canonical_60bar TP
#   cross_bank_revisited   --text-encoder finbert_fed_adjacent_xbank
#
# Default: every arm runs at 5 seeds × 5 folds × 3 head modes via
# run_dual_head_comparison.py. seq_len_60 uses canonical_60bar TP;
# everything else uses canonical TP.

set -uo pipefail

export PATH="/root/shim:$PATH"
cd /root/fed-pulse

TS=$(date -u +%Y%m%dT%H%M%SZ)
ART="backend/artifacts/experiments"
LOG_DIR="$HOME/runpod-sweep-logs"
mkdir -p "$ART" "$LOG_DIR"
SUMMARY="$LOG_DIR/_wave3_summary_${TS}.log"

note() { printf '[wave3] %s\n' "$*" | tee -a "$SUMMARY"; }

note "start $(date -u) TS=$TS host=$(hostname)"
note "python -> $(PATH=/root/shim:$PATH command -v python) ($(PATH=/root/shim:$PATH python --version 2>&1))"
note "dev rev -> $(git rev-parse --short HEAD)"

SEEDS="${SEEDS:-11 29 47 71 97}"
EPOCHS="${EPOCHS:-40}"
SWEEPS="${SWEEPS:-focal_loss class_balanced multi_horizon_aux absolute_regime vix_features seq_len_60 cross_bank_revisited}"

run_arm() {
    local key=$1
    shift
    local log_file="$LOG_DIR/wave3_${key}_${TS}.log"
    note "===== $key start $(date -u) ====="
    if "$@" 2>&1 | tee "$log_file"; then
        note "$key OK -> $log_file"
    else
        note "$key FAILED (continuing) — see $log_file"
    fi
}

for key in $SWEEPS; do
    case $key in
    focal_loss)
        run_arm focal_loss \
            python -m scripts.run_dual_head_comparison \
                --training-package-id canonical \
                --output $ART/runpod_wave3_focal_loss_${TS}.json \
                --seeds $SEEDS --epochs $EPOCHS \
                --regime-loss focal --focal-gamma 2.0
        ;;

    class_balanced)
        run_arm class_balanced \
            python -m scripts.run_dual_head_comparison \
                --training-package-id canonical \
                --output $ART/runpod_wave3_class_balanced_${TS}.json \
                --seeds $SEEDS --epochs $EPOCHS \
                --regime-loss class_balanced --class-balanced-beta 0.999
        ;;

    multi_horizon_aux)
        # aux heads share the primary log-RV head architecture, so they
        # require head_mode regression/dual; the classification cell raises
        # in factory.build_forecaster. Restrict head-modes accordingly.
        run_arm multi_horizon_aux \
            python -m scripts.run_dual_head_comparison \
                --training-package-id canonical \
                --output $ART/runpod_wave3_multi_horizon_aux_${TS}.json \
                --seeds $SEEDS --epochs $EPOCHS \
                --head-modes regression dual \
                --aux-horizons 5,20 --aux-horizon-alpha 0.3
        ;;

    absolute_regime)
        # Annualised thresholds: calm < 12%, normal 12-22%, high >= 22%.
        # Runner converts to per-period internally.
        run_arm absolute_regime \
            python -m scripts.run_dual_head_comparison \
                --training-package-id canonical \
                --output $ART/runpod_wave3_absolute_regime_${TS}.json \
                --seeds $SEEDS --epochs $EPOCHS \
                --vol-regime-label-mode absolute \
                --absolute-calm-max 12.0 --absolute-high-min 22.0
        ;;

    vix_features)
        # NB: requires canonical TP rebuild with the 6 new VIX columns
        # before the loader returns non-None values. On the current TP
        # every row hits the missing-flag path -> arm runs but produces
        # canonical-equivalent results. Worth running once the TP
        # rebuild lands; flagged here so the operator notices.
        run_arm vix_features \
            python -m scripts.run_dual_head_comparison \
                --training-package-id canonical \
                --output $ART/runpod_wave3_vix_features_${TS}.json \
                --seeds $SEEDS --epochs $EPOCHS \
                --use-vix-features
        ;;

    seq_len_60)
        run_arm seq_len_60 \
            python -m scripts.run_dual_head_comparison \
                --training-package-id canonical_60bar \
                --output $ART/runpod_wave3_seq_len_60_${TS}.json \
                --seeds $SEEDS --epochs $EPOCHS \
                --sequence-length 60
        ;;

    cross_bank_revisited)
        # Re-run the canonical dual-head cell with the cross-bank-
        # supervised encoder. PR #231's prior null used the deprecated
        # target convention + classification-only head; this re-tests
        # the same encoder choice under strict-forward + dual head.
        run_arm cross_bank_revisited \
            python -m scripts.run_dual_head_comparison \
                --training-package-id canonical \
                --output $ART/runpod_wave3_cross_bank_revisited_${TS}.json \
                --seeds $SEEDS --epochs $EPOCHS \
                --text-encoder finbert_fed_adjacent_xbank
        ;;

    mp_off)
        # Follow-up control: --no-mp-surprise isolates the -0.072 dual-vs-cls
        # mp_surprise effect found in the bisect. Output keeps the runpod_mp_off_
        # prefix (not wave3_) per the operator's spec.
        run_arm mp_off \
            python -m scripts.run_dual_head_comparison \
                --training-package-id canonical \
                --output $ART/runpod_mp_off_${TS}.json \
                --seeds $SEEDS --epochs $EPOCHS \
                --no-mp-surprise
        ;;

    *)
        note "unknown arm: $key (skipping)"
        ;;
    esac
done

note "===== wave3 batch finished $(date -u) ====="
note "WAVE3_BATCH_COMPLETE_MARKER"
ls -la "$ART"/runpod_wave3_*_${TS}.json 2>&1 | tee -a "$SUMMARY"
