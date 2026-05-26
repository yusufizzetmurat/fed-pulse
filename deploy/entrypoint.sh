#!/bin/sh
# Production entrypoint for the fed-pulse droplet container.
#
# Responsibilities:
#   1. Log in to Hugging Face Hub with the HF_TOKEN env var if present.
#   2. Eager-pull the hot-path artefacts via the registry resolver so
#      the first /analyze request does not wait on a checkpoint
#      download.
#   3. Hand off to s6-overlay which supervises uvicorn (backend on
#      :8000) and the Next.js standalone server (FE on :3000).
#
# Boot budget on the 8 GB / 4 vCPU droplet: under 90 seconds. The
# eager-pull step is what dominates — local SSD plus a fast NIC make
# the actual snapshot_download the slow part. If a non-eager encoder
# is needed at request time, the lazy-fetch path in
# app/data/embedding_cache.py (ensure_local) handles it.

set -e

if [ -n "${HF_TOKEN}" ]; then
    # huggingface-cli login --token is silent; we redirect to /dev/null
    # so the token never lands in a process listing. Set
    # HUGGINGFACE_HUB_TOKEN as a fallback for any tool that doesn't see
    # the cached token.
    export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}"
    python -c "from huggingface_hub import login; import os; login(token=os.environ['HF_TOKEN'], add_to_git_credential=False)" >/dev/null 2>&1 || true
fi

# Mask the s6 service set so the FE container does not also boot a
# uvicorn worker and vice versa. The "user" bundle's contents.d
# directory enumerates the active longrun services.
if [ -n "${FED_PULSE_SVC_MASK}" ]; then
    case "${FED_PULSE_SVC_MASK}" in
        backend)
            rm -f /etc/s6-overlay/s6-rc.d/user/contents.d/frontend
            ;;
        frontend)
            rm -f /etc/s6-overlay/s6-rc.d/user/contents.d/backend
            ;;
    esac
fi

if [ "${FED_PULSE_SKIP_EAGER_PULL:-0}" != "1" ]; then
    echo "[entrypoint] eager-pull hot-path artefacts..."
    python - <<'PY'
import sys
import time

from app.models.registry import eager_artefacts, resolve_hf_uri

start = time.time()
for ref in eager_artefacts():
    print(f"[entrypoint]   {ref.name} <- {ref.hf_uri}", flush=True)
    try:
        resolve_hf_uri(ref.hf_uri)
    except Exception as exc:  # pragma: no cover — log + continue
        print(f"[entrypoint]   ! skipped {ref.name}: {exc!r}", file=sys.stderr, flush=True)
elapsed = time.time() - start
print(f"[entrypoint] eager-pull done in {elapsed:.1f}s", flush=True)
PY
fi

exec "$@"
