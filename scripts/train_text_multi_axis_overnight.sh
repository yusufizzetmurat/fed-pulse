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

# Logs land under /tmp because backend/data/artifacts/ is root-owned
# (Docker writes as root by default); the container itself still
# writes checkpoints to /app/models which bind-mounts back to
# backend/models on the host.
LOG_DIR="${LOG_DIR:-/tmp/fed-pulse/text_multi_axis_training/$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$LOG_DIR"
echo "[overnight] logs land under $LOG_DIR; checkpoints under backend/models/"

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

# Pull the per-seed val loss off each checkpoint payload (key:
# ``metrics['val_loss']``, written by the trainer's ``_save_checkpoint``)
# and copy the winner to the canonical slot the /analyze service
# reads. CANDIDATES is derived from the SEEDS env / default above so
# the two stay in lockstep — adding a seed to SEEDS automatically
# picks it up here.
seeds_csv=$(IFS=,; echo "${SEEDS[*]}")
docker compose --profile gpu run --rm -e SEEDS_CSV="$seeds_csv" backend-gpu python - <<'PY'
import os
import shutil
from pathlib import Path

import torch

seeds_csv = os.environ.get("SEEDS_CSV", "")
seeds = [int(s.strip()) for s in seeds_csv.split(",") if s.strip()]
CANDIDATES = [(f"/app/models/text_multi_axis_seed{seed}.pt", seed) for seed in seeds]
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
