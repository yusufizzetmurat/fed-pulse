#!/usr/bin/env bash
# Overnight multi-axis classifier training: 3 seeds × 8 epochs × all 6
# gtfintechlab central-bank datasets (18 000 rows). Each seed writes
# a separate checkpoint; the lowest val-loss run wins the canonical
# slot at backend/models/text_multi_axis_best.pt that the /analyze
# service reads.
#
# Usage (kick off in the background; output goes to the log file
# printed at the top of stdout):
#
#     bash scripts/train_text_multi_axis_overnight.sh
#
# Expected wall-clock: ~20 min per seed on a single GPU; ~60 min for
# the three seeds plus the post-run "pick best" step.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SEEDS=(97 11 47)
EPOCHS="${EPOCHS:-8}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
ENCODER_ALIAS="${ENCODER_ALIAS:-finbert_fed_adjacent}"

LOG_DIR="$REPO_ROOT/data/artifacts/text_multi_axis_training/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$LOG_DIR"
echo "[overnight] logs and per-seed checkpoints land under $LOG_DIR"

# Each seed writes its best-epoch checkpoint to a seed-specific path
# inside backend/models/ so we can compare val loss after the run and
# promote the winner to text_multi_axis_best.pt.
for seed in "${SEEDS[@]}"; do
  ckpt="/app/models/text_multi_axis_seed${seed}.pt"
  log_file="$LOG_DIR/seed${seed}.log"
  echo "[overnight] starting seed=$seed -> $ckpt"
  docker compose --profile gpu run --rm backend-gpu \
    python -m app.data.train_text_multi_axis_classifier \
      --data-source gtfintechlab_hf \
      --encoder-alias "$ENCODER_ALIAS" \
      --epochs "$EPOCHS" \
      --seed "$seed" \
      --batch-size "$BATCH_SIZE" \
      --learning-rate "$LEARNING_RATE" \
      --output-checkpoint "$ckpt" \
      2>&1 | tee "$log_file"
done

echo "[overnight] all seeds complete; picking best by val loss"

# Pull the best_val_loss off each per-seed checkpoint payload and copy
# the winner to the canonical slot the /analyze service reads. Falls
# back to seed 97 if none of the checkpoints carry a metrics entry.
docker compose --profile gpu run --rm backend-gpu python - <<'PY'
import shutil
from pathlib import Path

import torch

CANDIDATES = [
    ("/app/models/text_multi_axis_seed97.pt", 97),
    ("/app/models/text_multi_axis_seed11.pt", 11),
    ("/app/models/text_multi_axis_seed47.pt", 47),
]
CANONICAL = Path("/app/models/text_multi_axis_best.pt")

best_seed = None
best_path = None
best_loss = float("inf")
for path, seed in CANDIDATES:
    if not Path(path).exists():
        print(f"[pick-best] missing checkpoint for seed={seed} at {path}")
        continue
    payload = torch.load(path, map_location="cpu", weights_only=False)
    val_loss = float((payload.get("metrics") or {}).get("val_loss", float("inf")))
    print(f"[pick-best] seed={seed} val_loss={val_loss:.4f}")
    if val_loss < best_loss:
        best_loss = val_loss
        best_seed = seed
        best_path = path

if best_path is not None:
    shutil.copyfile(best_path, CANONICAL)
    print(f"[pick-best] promoted seed={best_seed} val_loss={best_loss:.4f} -> {CANONICAL}")
else:
    print("[pick-best] no candidate checkpoints found; canonical slot untouched")
PY

echo "[overnight] done."
