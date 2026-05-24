#!/usr/bin/env bash
# Lambda sweep for the gated InfoNCE multi-modal forecaster (#235).
# Locks the HP cell that won the post-correction regime_arch_sweep on
# the supplied training package, then trains the gated_infonce variant
# at three lambda values against that cell so the comparison against
# the concat baseline is apples-to-apples (same arch, same HP, only
# fusion mode differs).
#
# Usage:
#
#     bash scripts/run_infonce_lambda_sweep.sh \
#         tp_v3_macro_aug_2026_05_23_sentiment_market_core_v1.1_epv1_v1.0 \
#         transformer
#
# Expected wall-clock: ~3-5 min per trial × 3 lambda values × 4 folds
# × 1 seed = ~60-90 min on a single GPU.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

TRAINING_PACKAGE_ID="${1:-tp_v3_macro_aug_2026_05_23_sentiment_market_core_v1.1_epv1_v1.0}"
ARCHITECTURE="${2:-transformer}"
TEXT_ENCODER="${TEXT_ENCODER:-finbert_fed_adjacent}"
SEEDS="${SEEDS:-97}"
FOLDS="${FOLDS:-wf_fold_1 wf_fold_2 wf_fold_3 wf_fold_4}"
LAMBDAS=("${LAMBDA_LIST[@]:-0.05 0.1 0.3}")

# Locked HP cell. Use the Path B winning HP from §6.7
# (transformer hidden=128, layers=2, dropout=0.1, lr=3e-4, wd=1e-4).
# These match the sweep grid the rest of the pipeline uses.
HIDDEN_SIZE="${HIDDEN_SIZE:-128}"
NUM_LAYERS="${NUM_LAYERS:-2}"
DROPOUT="${DROPOUT:-0.1}"
LEARNING_RATE="${LEARNING_RATE:-0.0003}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
TEXT_ADAPTER_DIM="${TEXT_ADAPTER_DIM:-64}"
INFONCE_LATENT_DIM="${INFONCE_LATENT_DIM:-64}"
INFONCE_TEMPERATURE="${INFONCE_TEMPERATURE:-0.07}"
EPOCHS="${EPOCHS:-40}"

LOG_DIR="${LOG_DIR:-/tmp/fed-pulse/infonce_lambda_sweep/$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$LOG_DIR"
echo "[infonce-sweep] training package: $TRAINING_PACKAGE_ID"
echo "[infonce-sweep] architecture:     $ARCHITECTURE"
echo "[infonce-sweep] logs:             $LOG_DIR"

for lambda_value in "${LAMBDAS[@]}"; do
  cell_id="lambda${lambda_value//./p}"
  report_path="/data/artifacts/infonce_lambda_sweep/${TRAINING_PACKAGE_ID}/${ARCHITECTURE}/${cell_id}/forecaster_sweep_results.json"
  log_file="$LOG_DIR/${ARCHITECTURE}_${cell_id}.log"
  echo "[infonce-sweep] starting cell ${ARCHITECTURE}/${cell_id} -> $report_path"
  docker compose --profile gpu run --rm backend-gpu \
    python -m app.train_forecaster \
      --training-package-id "$TRAINING_PACKAGE_ID" \
      --architecture "$ARCHITECTURE" \
      --seeds $SEEDS \
      --folds $FOLDS \
      --hidden-size "$HIDDEN_SIZE" \
      --num-layers "$NUM_LAYERS" \
      --dropout "$DROPOUT" \
      --learning-rate "$LEARNING_RATE" \
      --weight-decay "$WEIGHT_DECAY" \
      --epochs "$EPOCHS" \
      --output-mode classification \
      --vol-regime-classes 3 \
      --rich-features \
      --text-encoder "$TEXT_ENCODER" \
      --text-adapter-dim "$TEXT_ADAPTER_DIM" \
      --fusion-mode gated_infonce \
      --infonce-lambda "$lambda_value" \
      --infonce-temperature "$INFONCE_TEMPERATURE" \
      --infonce-latent-dim "$INFONCE_LATENT_DIM" \
      --report-path "$report_path" \
      2>&1 | tee "$log_file"
done

echo "[infonce-sweep] all cells complete"
echo "[infonce-sweep] aggregate macro-F1 across cells:"

docker compose --profile gpu run --rm backend-gpu python - <<PY
import json
from pathlib import Path

root = Path("/data/artifacts/infonce_lambda_sweep/${TRAINING_PACKAGE_ID}/${ARCHITECTURE}")
for cell_dir in sorted(root.glob("lambda*")):
    report = cell_dir / "forecaster_sweep_results.json"
    if not report.exists():
        print(f"{cell_dir.name}: missing report")
        continue
    payload = json.loads(report.read_text())
    trials = payload.get("trials", [])
    if not trials:
        print(f"{cell_dir.name}: no trials")
        continue
    test_metrics = [
        float((t.get("summary") or {}).get("test_metrics", {}).get("regime_f1_macro") or 0.0)
        for t in trials
        if (t.get("summary") or {}).get("test_metrics") is not None
    ]
    if not test_metrics:
        print(f"{cell_dir.name}: no test_metrics")
        continue
    mean = sum(test_metrics) / len(test_metrics)
    print(f"{cell_dir.name}: n={len(test_metrics)} mean_test_macro_f1={mean:.4f}")
PY

echo "[infonce-sweep] done."
