"""Encoder-provenance sidecars for embedding artifacts.

Writes a ``<artifact>.encoder.json`` next to any built embedding file recording
which encoder produced it (model id, revision, hidden size, build time). This
makes the encoder behind a cached embedding artifact permanently auditable, so
the silent-fallback contamination risk cannot leave an unverifiable artifact
behind (the bug that left ``fomc_embeddings.parquet`` ambiguous).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

_SIDECAR_SUFFIX = ".encoder.json"


def _sidecar_path(artifact_path: Path | str) -> Path:
    p = Path(artifact_path)
    return p.parent / (p.name + _SIDECAR_SUFFIX)


def write_encoder_sidecar(
    artifact_path: Path | str,
    provenance: dict[str, Any],
    *,
    built_at: str | None = None,
) -> Path:
    """Write the encoder provenance for ``artifact_path`` and return the sidecar path.

    ``provenance`` should carry at least ``model_id``; ``revision`` and
    ``hidden_size`` are recorded when present. ``built_at`` defaults to the
    current UTC time (override for deterministic tests).
    """
    record = dict(provenance)
    record.setdefault("built_at", built_at or datetime.now(timezone.utc).isoformat())
    sidecar = _sidecar_path(artifact_path)
    sidecar.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return sidecar


def read_encoder_sidecar(artifact_path: Path | str) -> dict[str, Any] | None:
    """Return the recorded provenance for ``artifact_path``, or None if absent."""
    sidecar = _sidecar_path(artifact_path)
    if not sidecar.exists():
        return None
    return cast("dict[str, Any]", json.loads(sidecar.read_text()))
