#!/usr/bin/env bash
# Post-B salvage: build local embedding caches (full GPU, after B frees it),
# then re-run C-finbert / D-cross_bank / C-voyage with real embeddings.
# Caches stay LOCAL (no HF push per operator). nomic skipped (cache builder
# needs trust_remote_code — tomorrow).
set -uo pipefail
export PATH="/root/shim:$PATH"; cd /root/fed-pulse
LOG="$HOME/runpod-sweep-logs"; ART="backend/artifacts/experiments"
TS=$(date -u +%Y%m%dT%H%M%SZ)
SUM="$LOG/_salvage_summary_${TS}.log"
note(){ printf '[salvage] %s\n' "$*" | tee -a "$SUM"; }
VKEY='pa-IBlhX6M_h-5iZL7vqQuz8PmT0fGIqXnSfsGOIDZHT_6'

# 0. wait for B / the queue to finish so cache builds get full GPU
QSUM=$(ls -t "$LOG"/_queue_summary_*.log | head -1)
note "waiting for queue (B) to finish before salvage..."
until grep -q QUEUE_COMPLETE_MARKER "$QSUM" 2>/dev/null; do sleep 20; done
note "queue done; starting salvage $(date -u)"

# 1. build BERT caches (full GPU, generous timeout)
for enc in finbert_fed_adjacent finbert_fed_adjacent_xbank; do
  note "build cache: $enc"
  if timeout 1800 python -m app.data.embedding_cache --encoder "$enc" \
        --training-package-id canonical --allow-network > "$LOG/cache_${enc}_${TS}.log" 2>&1; then
    note "  cache $enc OK"
  else
    note "  cache $enc FAILED (see cache_${enc}_${TS}.log)"
  fi
done

# 2. build voyage cache via API (key inline; not paying retail per operator)
note "build cache: voyage_finance_2 (API)"
if VOYAGE_API_KEY="$VKEY" timeout 1800 python scripts/cache_voyage_embeddings.py \
      --encoder-alias voyage_finance_2 --training-package-id canonical \
      --api-key "$VKEY" --allow-network > "$LOG/cache_voyage_${TS}.log" 2>&1; then
  note "  cache voyage OK"
else
  note "  cache voyage FAILED (see cache_voyage_${TS}.log)"
fi

# 3. re-run the three arms with now-local caches (real embeddings)
run(){ local k=$1; shift; note "===== rerun $k $(date -u) ====="; if "$@" 2>&1 | tee "$LOG/salvage_${k}_${TS}.log"; then note "$k OK"; else note "$k FAILED"; fi; }
run C_finbert_cache python -m scripts.run_dual_head_comparison --training-package-id canonical \
    --output "$ART/runpod_bakeoff_v546cache_finbert_fed_adjacent_${TS}.json" \
    --seeds 11 29 47 71 97 --epochs 40 --text-encoder finbert_fed_adjacent --use-text-embeddings
run D_cross_bank_cache python -m scripts.run_dual_head_comparison --training-package-id canonical \
    --output "$ART/runpod_cross_bank_revisited_v546cache_${TS}.json" \
    --seeds 11 29 47 71 97 --epochs 40 --text-encoder finbert_fed_adjacent_xbank --use-text-embeddings
run C_voyage_cache python -m scripts.run_dual_head_comparison --training-package-id canonical \
    --output "$ART/runpod_bakeoff_v546cache_voyage_finance_2_${TS}.json" \
    --seeds 11 29 47 71 97 --epochs 40 --text-encoder voyage_finance_2 --use-text-embeddings

note "===== salvage finished $(date -u) ====="; note "SALVAGE_COMPLETE_MARKER"
ls -la "$ART"/runpod_bakeoff_v546cache_*_${TS}.json "$ART"/runpod_cross_bank_revisited_v546cache_${TS}.json 2>&1 | tee -a "$SUM"
