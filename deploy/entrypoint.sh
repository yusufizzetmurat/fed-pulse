#!/bin/sh
# Production entrypoint for the fed-pulse droplet container.
#
# Responsibilities:
#   1. Log in to Hugging Face Hub with the HF_TOKEN env var if present.
#      Failure is fatal — uvicorn will not start on a broken token because
#      the eager-pull would silently fall back to anonymous rate limits.
#   2. Eager-pull the hot-path artefacts via the registry resolver so
#      the first /analyze request does not wait on a checkpoint
#      download.
#   3. Hand off to s6-overlay which supervises uvicorn (backend on
#      :8000) and the Next.js standalone server (FE on :3000). Each
#      service drops to the ``fedpulse`` UID via ``s6-setuidgid``
#      before exec'ing the workload (see deploy/s6-services/*/run).
#
# Boot budget on the 8 GB / 4 vCPU droplet: under 90 seconds. The
# eager-pull step is what dominates — local SSD plus a fast NIC make
# the actual snapshot_download the slow part. If a non-eager encoder
# is needed at request time, the lazy-fetch path in
# app/data/embedding_cache.py (ensure_local) handles it.
#
# MP-surprise parquet refresh runs weekly via
# .github/workflows/refresh-mp-surprise.yml; entrypoint reads the
# latest committed parquet.

set -e

if [ -n "${HF_TOKEN}" ]; then
    # Pin HUGGINGFACE_HUB_TOKEN so any tool that doesn't see the cached
    # token (e.g. transformers' from_pretrained called outside the
    # huggingface_hub login session) still authenticates. Failures here
    # are fatal — a stale or revoked HF token degrades the eager-pull
    # to anonymous rate-limited reads which silently break the first
    # /analyze request. Capture stderr so the operator sees the actual
    # huggingface_hub error message in the deploy logs.
    export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}"
    if ! python -c "from huggingface_hub import login; import os; login(token=os.environ['HF_TOKEN'], add_to_git_credential=False)"; then
        echo "[entrypoint] HF login failed — verify HF_TOKEN secret is set and not expired" >&2
        exit 1
    fi
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
