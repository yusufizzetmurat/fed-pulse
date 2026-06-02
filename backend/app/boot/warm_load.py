"""Boot-time warm-load of the credibility module inputs.

The credibility tile on /analyze reads from two on-disk caches:

- the per-encoder embedding parquet under ``data/raw/embeddings/`` —
  produced once via ``make cache-embeddings`` and mirrored to the
  ``yusufizzetmurat/fed-pulse-embedding-caches`` HF dataset for the
  deploy lane.
- the FRED ``DFF`` series under ``data/external/fred/`` — refreshed
  via :func:`app.services.fred_client.fetch_fred_series` and cached
  as JSON.

A fresh container has neither, so /analyze used to surface an empty
"signals unavailable" state for every passage. This helper runs from
the FastAPI lifespan hook and pulls each input best-effort. It never
raises out — every call site is wrapped in ``try / except`` so a
missing token, a 404, or a network blip does not block uvicorn boot.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger("app.boot.warm_load")


def _warm_embedding_cache(encoder_alias: str) -> Path | None:
    """Best-effort warm of the canonical encoder's embedding parquet.

    Returns the resolved parquet path when present (either pre-existing
    or freshly pulled), ``None`` otherwise. Never raises.
    """

    try:
        from app.data.embedding_cache import (
            EmbeddingCacheUnavailable,
            ensure_local,
            resolve_cache_paths,
        )
        from app.models.registry import encoder_ref
    except Exception:  # noqa: BLE001
        logger.warning("warm_load: embedding_cache imports failed", exc_info=True)
        return None

    ref = encoder_ref(encoder_alias)
    if ref is None or not ref.revision:
        logger.info(
            "warm_load: encoder %r absent from registry; skipping",
            encoder_alias,
        )
        return None

    paths = resolve_cache_paths(encoder_alias, revision=ref.revision)
    if paths.parquet.exists():
        logger.info("warm_load: embedding cache present at %s", paths.parquet)
        return paths.parquet

    # The HF dataset pull only runs when a token is set — otherwise
    # the lazy-fetch returns an anonymous rate-limited error and the
    # tile stays empty until an operator triggers ``make cache-embeddings``.
    if not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")):
        logger.info(
            "warm_load: HF_TOKEN absent; embedding cache will degrade to "
            "zero drift until the parquet lands on disk"
        )
        return None
    try:
        return ensure_local(encoder_alias, revision=ref.revision)
    except (EmbeddingCacheUnavailable, FileNotFoundError) as exc:
        logger.warning("warm_load: embedding cache fetch failed: %s", exc)
    except Exception:  # noqa: BLE001
        logger.warning("warm_load: embedding cache fetch raised", exc_info=True)
    return None


def _warm_fred_series(series_id: str = "DFF") -> Path | None:
    """Best-effort warm of the FRED rate-history JSON cache.

    Looks at the on-disk cache first — a fresh checkout that committed
    the JSON under ``data/external/fred/`` resolves instantly. Falls
    back to a network pull only when ``FRED_API_KEY`` is set so the
    container that ships without the env var still boots quietly.
    """

    try:
        from app.services.fred_client import DEFAULT_CACHE_DIR, fetch_fred_series
    except Exception:  # noqa: BLE001
        logger.warning("warm_load: fred_client imports failed", exc_info=True)
        return None

    cache_path = Path(DEFAULT_CACHE_DIR) / f"{series_id}.json"
    if cache_path.exists():
        logger.info("warm_load: FRED %s cache present at %s", series_id, cache_path)
        return cache_path

    if not (os.environ.get("FRED_API_KEY") or os.environ.get("FRED_TOKEN")):
        logger.info(
            "warm_load: FRED_API_KEY absent; %s realized series will "
            "degrade to neutral until the cache JSON lands on disk",
            series_id,
        )
        return None
    try:
        fetch_fred_series(series_id)
    except Exception:  # noqa: BLE001
        logger.warning("warm_load: FRED %s fetch raised", series_id, exc_info=True)
        return None
    return cache_path if cache_path.exists() else None


def warm_credibility_inputs(
    *,
    encoder_alias: str = "finbert_fed_adjacent_xbank_dapt",
    fred_series_id: str = "DFF",
) -> None:
    """Run both warm steps in sequence; never raises out.

    The two helpers are independent — the FRED warm proceeds even when
    the embedding warm fails so the realized-vs-stated axis can still
    fire on a host that lacks the embedding parquet.
    """

    _warm_embedding_cache(encoder_alias)
    _warm_fred_series(fred_series_id)


__all__ = ["warm_credibility_inputs"]
