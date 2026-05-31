#!/usr/bin/env bash
# Resilient serial sweep driver for a single-GPU Runpod pod.
#
# - Reuses the official /workspace/launch_sweeps.sh command templates, one key
#   per invocation, so a failure in one sweep does NOT abort the rest (the
#   upstream script's `set -e` only scopes a single key here).
# - cross_source is run via the module directly because `make
#   cross-source-transfer` shells out to `docker compose`, and docker is not
#   usable on this pod.
set -uo pipefail   # deliberately NOT -e: continue past per-sweep failures

export PATH="/root/shim:$PATH"   # bare `python` -> system 3.12 (has the stack)
cd /root/fed-pulse

TS=$(date -u +%Y%m%dT%H%M%SZ)
ART="backend/artifacts/experiments"
LOG_DIR="$HOME/runpod-sweep-logs"
mkdir -p "$ART" "$LOG_DIR"
SUMMARY="$LOG_DIR/_batch_summary_${TS}.log"

note() { printf '[batch] %s\n' "$*" | tee -a "$SUMMARY"; }

note "start $(date -u) TS=$TS host=$(hostname)"
note "python -> $(PATH=/root/shim:$PATH command -v python) ($(PATH=/root/shim:$PATH python --version 2>&1))"

# canonical already completed in a pre-run (runpod_canonical_*.json present) and
# is byte-identical under #524 for the no-flag base path, so it is EXCLUDED here.
LM_CSV="data/external/loughran_mcdonald/lm_master_2025__master_dictionary.csv"
if [ -f "$LM_CSV" ]; then
    note "LM dict present ($LM_CSV) — lm arm will run"
else
    note "WARNING: LM dict missing ($LM_CSV) — lm arm will FAIL (FileNotFoundError); other arms unaffected"
fi
if ! ls "$ART"/runpod_canonical_*.json >/dev/null 2>&1; then
    note "WARNING: no runpod_canonical_*.json found — canonical was expected from the pre-run"
fi

# --- template-driven sweeps (isolated per key) ------------------------------
for key in surprise retrieval regime derived b2 phrasebank \
           statement_delta vote_tally lm press_conf mtl_verify; do
    note "===== $key start $(date -u) ====="
    if TP_ID=canonical FANOUT=serial SWEEPS="$key" bash /workspace/launch_sweeps.sh; then
        note "$key OK"
    else
        note "$key FAILED (continuing) — see $LOG_DIR/${key}_*.log"
    fi
done

# --- cross_source: direct module (docker workaround) ------------------------
note "===== cross_source start $(date -u) (direct module; docker unusable) ====="
CS_DIR="$LOG_DIR/cross_source_out_${TS}"
CS_LOG="$LOG_DIR/cross_source_${TS}.log"
if python -m app.evaluation.cross_source_transfer \
        --training-package-id canonical \
        --encoder-checkpoints finbert_fed_adjacent=yusufizzetmurat/finbert-fed-adjacent \
        --output-dir "$CS_DIR" > "$CS_LOG" 2>&1; then
    if [ -f "$CS_DIR/matrix.json" ]; then
        cp "$CS_DIR/matrix.json" "$ART/runpod_cross_source_${TS}.json"
        note "cross_source OK -> runpod_cross_source_${TS}.json"
    else
        note "cross_source ran but matrix.json missing (see $CS_LOG)"
    fi
else
    note "cross_source FAILED (continuing) — see $CS_LOG"
fi

# --- wrap up ----------------------------------------------------------------
note "DONE $(date -u)"
note "artefacts written:"
ls -1 "$ART"/runpod_*.json 2>/dev/null | tee -a "$SUMMARY" || note "  (none)"
note "BATCH_COMPLETE_MARKER"
