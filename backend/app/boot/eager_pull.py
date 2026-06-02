"""Boot-time eager pull of artefacts pinned ``eager: true`` in registry.yaml.

Runs from the container entrypoint before uvicorn starts. For each
eager artefact mapped in :data:`_ARTEFACT_FILES`, downloads the pinned
revision via ``huggingface_hub.snapshot_download`` and copies the named
files into ``MODELS_DIR`` only when the destination is absent. A dev
box with a freshly trained checkpoint keeps it — the shim never
clobbers a file already on disk. The shim never raises out: on
missing token / network failure / 404 it logs and returns so the
cold-start bootstrap in :mod:`app.main` still runs on first
``/analyze``.

Mapping policy: only artefacts whose files are read directly out of
``MODELS_DIR`` appear in :data:`_ARTEFACT_FILES`. Each entry is either
a flat filename (snapshot path == destination path relative to
``MODELS_DIR``) or a ``(snapshot_name, dst_relpath)`` pair when the
file must land in a sub-directory of ``MODELS_DIR``; the tuple form
is what ``volume_har_canonical`` uses to drop its JSON spec under
``models/volume_har/``. ``encoder_canonical``, ``retrieval``,
``trajectory`` lazy-load via their own caches and are intentionally
absent so they do not trip the copy step.
``rates_heads_canonical`` historically pointed at the same
``forecaster_best.pt`` file as ``forecaster_canonical``; with the LSTM
canonical revision (``7ab0a873``) rates heads are absent, so we leave
``rates_heads_canonical`` out of the copy map to avoid clobbering the
forecaster path.
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger("app.boot.eager_pull")

# Hoist the HF Hub import to module load so the test suite can
# monkeypatch ``snapshot_download`` deterministically. ``None`` is a
# sentinel for "huggingface_hub absent in this env"; ``hydrate`` then
# logs and skips.
snapshot_download: Callable[..., str] | None
try:
    from huggingface_hub import snapshot_download as _snapshot_download

    snapshot_download = _snapshot_download
except Exception:  # pragma: no cover - import-time defensive
    snapshot_download = None

# Artefact-name -> tuple of filenames to extract from the snapshot and
# copy into MODELS_DIR. Anything not listed is left in the HF cache.
#
# Each entry is either a flat filename (snapshot path == destination
# path relative to ``MODELS_DIR``) or a ``(snapshot_name, dst_relpath)``
# pair when the destination needs to land in a sub-directory under
# ``MODELS_DIR``. The ``volume_har_canonical`` artefact uses the pair
# form so the JSON spec sits in ``models/volume_har/`` where
# :mod:`app.services.volume_forecaster` reads it.
_ARTEFACT_FILES: dict[str, tuple[str | tuple[str, str], ...]] = {
    "forecaster_canonical": (
        "forecaster_best.pt",
        "forecaster_best.pt.inference_contract.json",
        "forecaster_best.conformal.json",
        "forecaster_best.pt.lora_adapter.pt",
        "forecaster_calibration_fresh.pt",
    ),
    "volume_har_canonical": (("volume_har_artifact.json", "volume_har/volume_har_artifact.json"),),
}


def _split_entry(entry: str | tuple[str, str]) -> tuple[str, str]:
    """Return ``(snapshot_relpath, dst_relpath)`` for one mapping entry.

    Plain strings collapse to ``(entry, entry)`` so the legacy flat-file
    mapping is byte-identical.
    """

    if isinstance(entry, tuple):
        return entry[0], entry[1]
    return entry, entry


def _hydrate_one(  # noqa: PLR0913 - six injected params is the natural shape here
    artefact: Any,
    files: tuple[str | tuple[str, str], ...],
    models_dir: Path,
    token: str,
    parse_hf_uri: Callable[[str], Any],
    download_snapshot: Callable[..., str],
) -> None:
    try:
        ref = parse_hf_uri(artefact.hf_uri)
    except Exception:
        logger.exception(
            "eager-pull: parse_hf_uri failed for %r (%s); skip",
            artefact.name,
            artefact.hf_uri,
        )
        return

    pairs = [_split_entry(entry) for entry in files]
    snapshot_names = [src_name for src_name, _ in pairs]

    revision = artefact.revision or None
    try:
        snapshot_dir = download_snapshot(
            repo_id=ref.repo_id,
            repo_type=ref.repo_type,
            revision=revision,
            allow_patterns=snapshot_names,
            token=token,
        )
    except Exception as exc:
        logger.warning(
            "eager-pull: snapshot_download failed for %s @ %s (%s); "
            "cold-start bootstrap will fill any gap on first /analyze",
            artefact.hf_uri,
            revision or "main",
            exc,
        )
        return

    for src_name, dst_relpath in pairs:
        src = Path(snapshot_dir) / src_name
        dst = models_dir / dst_relpath
        if not src.exists():
            logger.warning(
                "eager-pull: %r missing from snapshot of %s; skip",
                src_name,
                artefact.hf_uri,
            )
            continue
        if dst.exists():
            logger.info("eager-pull: %s already present; not overwriting", dst_relpath)
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        logger.info(
            "eager-pull: hydrated %s <- %s @ %s",
            dst_relpath,
            artefact.hf_uri,
            revision or "main",
        )


def hydrate() -> None:
    """Pull every mapped eager artefact, copy each named file if absent."""

    try:
        from app.models.config import MODELS_DIR
        from app.models.registry import eager_artefacts, parse_hf_uri
    except Exception:
        logger.exception("eager-pull: registry import failed; skipping")
        return

    download_snapshot = snapshot_download
    if download_snapshot is None:
        logger.warning(
            "eager-pull: huggingface_hub not importable; skipping (cold-start will bootstrap)"
        )
        return

    token = os.environ.get("HF_TOKEN")
    if not token:
        logger.info(
            "eager-pull: HF_TOKEN absent; skipping (cold-start will bootstrap on first /analyze)"
        )
        return

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for artefact in eager_artefacts():
        files = _ARTEFACT_FILES.get(artefact.name)
        if files is None:
            logger.debug("eager-pull: %r has no MODELS_DIR copy mapping; skip", artefact.name)
            continue
        _hydrate_one(artefact, files, MODELS_DIR, token, parse_hf_uri, download_snapshot)


def main() -> int:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    try:
        hydrate()
    except Exception:
        logger.exception("eager-pull: unexpected error; continuing to uvicorn boot")
    return 0


if __name__ == "__main__":
    sys.exit(main())
