#!/usr/bin/env bash
set -euo pipefail

# Resolve repo root: inside the backend container the script is mounted at
# /app/scripts/, where /app is the backend package and /data is the data
# volume. On the host the script is at <repo>/scripts/ and the backend lives
# at <repo>/backend/.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -d "/app" && -d "/data" && -f "/app/main.py" ]]; then
  # in-container layout
  PYTHONPATH="/app${PYTHONPATH:+:$PYTHONPATH}"
  DATA_DIR="${FED_PULSE_DATA_DIR:-/data}"
  TESTS_DIR="/app/tests"
  SCRIPTS_DIR="/app/scripts"
  cd /
else
  REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
  PYTHONPATH="${REPO_ROOT}/backend${PYTHONPATH:+:$PYTHONPATH}"
  DATA_DIR="${FED_PULSE_DATA_DIR:-${REPO_ROOT}/data}"
  TESTS_DIR="${REPO_ROOT}/tests"
  SCRIPTS_DIR="${REPO_ROOT}/scripts"
  cd "${REPO_ROOT}"
fi
export PYTHONPATH
export FED_PULSE_DATA_DIR="${DATA_DIR}"

echo "[verify] 1/4 build toy snapshot (50 events)"
python "${SCRIPTS_DIR}/build_toy_snapshot.py" --n-events 50 --out-dir "${DATA_DIR}/interim/toy_snapshot"

echo "[verify] 2/4 unit + property + contract + regression tests"
pytest "${TESTS_DIR}/unit" "${TESTS_DIR}/properties" "${TESTS_DIR}/contract" "${TESTS_DIR}/regression" -q --maxfail=3

echo "[verify] 3/4 forecaster determinism smoke (seed 11)"
pytest "${TESTS_DIR}/regression/test_forecaster_determinism.py" -q -k bit_identical

echo "[verify] 4/4 import-graph sanity"
python - <<'PY'
import importlib

for module in (
    "app.main",
    "app.config",
    "app.determinism",
    "app.services.forecaster",
    "app.services.market_data",
    "app.services.text_encoder",
    "app.data.dense_daily_dataset",
    "app.data.dense_forecast_train",
    "app.data.dense_fomc_text",
    "app.data.polygon_spx",
    "app.data.intraday_event_builder",
    "app.data.intraday_direction_train",
    "app.data.intraday_magnitude_train",
    "app.training.manifest",
    "app.models.registry",
):
    importlib.import_module(module)
print("all modules imported cleanly")
PY

echo "[verify] OK"
