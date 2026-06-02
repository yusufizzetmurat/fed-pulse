#!/usr/bin/env bash
# Post-#545/#546 queue: A (doc_length×mp_off) -> C (bake-off rerun, fixed) ->
# D (cross_bank rerun, fixed) -> B (2^3 composition matrix).
# nomic excluded from C (needs trust_remote_code=True); voyage excluded (key).
# E (vix on canonical_vix) is handled separately once the HF SHA lands.
set -uo pipefail
export PATH="/root/shim:$PATH"; cd /root/fed-pulse
TS=$(date -u +%Y%m%dT%H%M%SZ)
ART="backend/artifacts/experiments"; LOG="$HOME/runpod-sweep-logs"
mkdir -p "$ART" "$LOG"; SUM="$LOG/_queue_summary_${TS}.log"
note(){ printf '[queue] %s\n' "$*" | tee -a "$SUM"; }
run(){ local k=$1; shift; note "===== $k start $(date -u) ====="; if "$@" 2>&1 | tee "$LOG/q_${k}_${TS}.log"; then note "$k OK"; else note "$k FAILED (continuing)"; fi; }
note "start $(date -u) HEAD=$(git rev-parse --short HEAD) TS=$TS"
S="11 29 47 71 97"; E=40

# A: doc_length × mp_off
run A_doc_length_mp_off python -m scripts.run_dual_head_comparison \
    --training-package-id canonical --output $ART/runpod_compo_doc_length_mp_off_${TS}.json \
    --seeds $S --epochs $E --use-doc-length --no-mp-surprise

# C: bake-off rerun under #546 (finbert, bge; nomic/voyage excluded)
for enc in finbert_fed_adjacent bge_large_en_v15; do
    run C_bakeoff_${enc}_v546 python -m scripts.run_dual_head_comparison \
        --training-package-id canonical --output $ART/runpod_bakeoff_v546_${enc}_${TS}.json \
        --seeds $S --epochs $E --text-encoder $enc --use-text-embeddings
done

# D: cross_bank_revisited rerun under #546
run D_cross_bank_revisited_v546 python -m scripts.run_dual_head_comparison \
    --training-package-id canonical --output $ART/runpod_cross_bank_revisited_v546_${TS}.json \
    --seeds $S --epochs $E --text-encoder finbert_fed_adjacent_xbank --use-text-embeddings

# B: 2^3 composition matrix (mp_surprise × doc_length × focal)
for mp in "" "--no-mp-surprise"; do
  for dl in "" "--use-doc-length"; do
    for fl in ce focal; do
      extra=""; [ "$fl" = "focal" ] && extra="--focal-gamma 2.0"
      label="mp${mp:+_off}_dl${dl:+_on}_loss_${fl}"
      run B_${label} python -m scripts.run_dual_head_comparison \
          --training-package-id canonical --output $ART/runpod_compo_matrix_${label}_${TS}.json \
          --seeds $S --epochs $E --regime-loss $fl $extra $mp $dl
    done
  done
done

note "===== queue finished $(date -u) ====="; note "QUEUE_COMPLETE_MARKER"
ls -la $ART/runpod_compo_*_${TS}.json $ART/runpod_bakeoff_v546_*_${TS}.json $ART/runpod_cross_bank_revisited_v546_${TS}.json 2>&1 | tee -a "$SUM"
