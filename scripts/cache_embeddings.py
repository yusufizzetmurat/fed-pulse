"""Build (or rebuild) the per-encoder embedding cache for a training package.

Examples
--------
Single encoder::

    python scripts/cache_embeddings.py \
        --encoder finbert_fomc \
        --training-package-id tp_ds_v2 \
        --allow-network

The Makefile target ``cache-embeddings`` wraps this script for the common case.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_PATH = REPO_ROOT / "backend"
if str(BACKEND_PATH) not in sys.path:
    sys.path.insert(0, str(BACKEND_PATH))

from app.data.embedding_cache import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
