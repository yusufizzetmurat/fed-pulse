"""Boot-time eager pull of artefacts pinned ``eager: true`` in registry.yaml.

Runs from the container entrypoint before uvicorn starts. For each
eager artefact mapped in :data:`_ARTEFACT_FILES`, downloads the pinned
revision via ``huggingface_hub.snapshot_download`` and copies the named
files into ``MODELS_DIR`` or ``DATA_DIR``. The copy is conditional on
content drift: an existing destination that matches the snapshot
byte-for-byte (same size + sha256) is skipped, but a drifted
destination is OVERWRITTEN so a stale checkpoint baked into a base
image cannot mask the pinned artefact. The shim never raises out: on
missing token / network failure / 404 it logs and returns so the
cold-start bootstrap in :mod:`app.main` still runs on first
``/analyze``.

Mapping policy: each entry is either a flat filename (snapshot path
== destination path relative to ``MODELS_DIR``) or a
``(snapshot_name, dst_relpath)`` pair when the file must land in a
sub-directory, or a ``(snapshot_name, dst_relpath, dst_root)`` triple
when the destination is rooted at ``DATA_DIR`` instead of
``MODELS_DIR``. ``dst_root`` is ``"MODELS"`` (default, backwards
compatible) or ``"DATA"``. ``volume_har_canonical`` uses the pair
form to drop its JSON spec under ``models/volume_har/``;
``trajectory_bundle`` and ``retrieval_bundle`` use the triple form so
their files land under ``data/artifacts/...`` where
:mod:`app.services.trajectory` and :mod:`app.services.analogs` read
them. ``encoder_canonical`` lazy-loads via the HF cache and is
intentionally absent. ``rates_heads_canonical`` historically pointed
at the same ``forecaster_best.pt`` file as ``forecaster_canonical``;
with the LSTM canonical revision (``7ab0a873``) rates heads are
absent, so we leave ``rates_heads_canonical`` out of the copy map to
avoid clobbering the forecaster path.
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
# copy into ``MODELS_DIR`` or ``DATA_DIR``. Anything not listed is left
# in the HF cache.
#
# Each entry is one of:
#   - a flat filename (snapshot path == destination path relative to
#     ``MODELS_DIR``);
#   - a ``(snapshot_name, dst_relpath)`` pair when the destination
#     needs to land in a sub-directory under ``MODELS_DIR``
#     (``volume_har_canonical`` uses this form);
#   - a ``(snapshot_name, dst_relpath, dst_root)`` triple when the
#     destination is rooted at ``DATA_DIR`` instead of ``MODELS_DIR``;
#     ``dst_root`` is ``"MODELS"`` (default) or ``"DATA"``.
#
# ``trajectory_bundle`` and ``retrieval_bundle`` use the triple form
# so their files land under ``data/artifacts/...`` where the trajectory
# and analogs services read them.
_TRAJECTORY_BUNDLE_RELDIR = "artifacts/trajectory/trajectory_transformer"
_RETRIEVAL_BUNDLE_RELDIR = "artifacts/retrieval/finbert_fed_adjacent_xbank_dapt_retrieval"

_ARTEFACT_FILES: dict[str, tuple[str | tuple[str, str] | tuple[str, str, str], ...]] = {
    "forecaster_canonical": (
        "forecaster_best.pt",
        "forecaster_best.pt.inference_contract.json",
        "forecaster_best.conformal.json",
        "forecaster_best.pt.lora_adapter.pt",
        "forecaster_calibration_fresh.pt",
    ),
    "volume_har_canonical": (("volume_har_artifact.json", "volume_har/volume_har_artifact.json"),),
    "trajectory_bundle": (
        ("embedding_index.parquet", f"{_TRAJECTORY_BUNDLE_RELDIR}/embedding_index.parquet", "DATA"),
        ("embedding_index.npz", f"{_TRAJECTORY_BUNDLE_RELDIR}/embedding_index.npz", "DATA"),
        ("model.pt", f"{_TRAJECTORY_BUNDLE_RELDIR}/model.pt", "DATA"),
        ("manifest.json", f"{_TRAJECTORY_BUNDLE_RELDIR}/manifest.json", "DATA"),
        ("conformal.json", f"{_TRAJECTORY_BUNDLE_RELDIR}/conformal.json", "DATA"),
        ("metrics.json", f"{_TRAJECTORY_BUNDLE_RELDIR}/metrics.json", "DATA"),
    ),
    "retrieval_bundle": (
        ("embeddings.npy", f"{_RETRIEVAL_BUNDLE_RELDIR}/embeddings.npy", "DATA"),
        ("index.parquet", f"{_RETRIEVAL_BUNDLE_RELDIR}/index.parquet", "DATA"),
        ("manifest.json", f"{_RETRIEVAL_BUNDLE_RELDIR}/manifest.json", "DATA"),
        ("training_args.json", f"{_RETRIEVAL_BUNDLE_RELDIR}/training_args.json", "DATA"),
        ("checkpoint/config.json", f"{_RETRIEVAL_BUNDLE_RELDIR}/checkpoint/config.json", "DATA"),
        (
            "checkpoint/config_sentence_transformers.json",
            f"{_RETRIEVAL_BUNDLE_RELDIR}/checkpoint/config_sentence_transformers.json",
            "DATA",
        ),
        (
            "checkpoint/model.safetensors",
            f"{_RETRIEVAL_BUNDLE_RELDIR}/checkpoint/model.safetensors",
            "DATA",
        ),
        ("checkpoint/modules.json", f"{_RETRIEVAL_BUNDLE_RELDIR}/checkpoint/modules.json", "DATA"),
        (
            "checkpoint/sentence_bert_config.json",
            f"{_RETRIEVAL_BUNDLE_RELDIR}/checkpoint/sentence_bert_config.json",
            "DATA",
        ),
        (
            "checkpoint/tokenizer.json",
            f"{_RETRIEVAL_BUNDLE_RELDIR}/checkpoint/tokenizer.json",
            "DATA",
        ),
        (
            "checkpoint/tokenizer_config.json",
            f"{_RETRIEVAL_BUNDLE_RELDIR}/checkpoint/tokenizer_config.json",
            "DATA",
        ),
        (
            "checkpoint/1_Pooling/config.json",
            f"{_RETRIEVAL_BUNDLE_RELDIR}/checkpoint/1_Pooling/config.json",
            "DATA",
        ),
        ("checkpoint/README.md", f"{_RETRIEVAL_BUNDLE_RELDIR}/checkpoint/README.md", "DATA"),
    ),
}


# Inventory of artefact-name -> ``.pt`` files the inference container
# reads out of the HF cache. Distinct from :data:`_ARTEFACT_FILES`
# above, which controls the boot-time copy into ``MODELS_DIR``: this
# table is metadata for the settings page so it can surface every
# registered checkpoint file regardless of whether the eager-pull shim
# copied it locally or the service is consuming it straight from the HF
# snapshot cache (the multi-axis classifier path lazy-fetches via
# :func:`huggingface_hub.hf_hub_download` and never lands under
# ``MODELS_DIR``). Entries here must list the snapshot-side filename
# only; the cache resolver supplies the absolute path.
ARTEFACT_PT_INVENTORY: dict[str, tuple[str, ...]] = {
    "forecaster_canonical": (
        "forecaster_best.pt",
        "forecaster_best.pt.lora_adapter.pt",
        "forecaster_calibration_fresh.pt",
    ),
    "multi_axis_text_classifier": ("text_multi_axis_best.pt",),
}


# ``dst_root`` values recognised in the triple-form entry. Anything
# else is logged and the entry is skipped.
_DST_ROOT_MODELS = "MODELS"
_DST_ROOT_DATA = "DATA"


def _split_entry(
    entry: str | tuple[str, str] | tuple[str, str, str],
) -> tuple[str, str, str]:
    """Return ``(snapshot_relpath, dst_relpath, dst_root)`` for one mapping entry.

    Plain strings collapse to ``(entry, entry, "MODELS")`` so the legacy
    flat-file mapping is byte-identical. Two-tuples expand to a default
    ``"MODELS"`` root so the existing ``volume_har_canonical`` mapping
    keeps working unchanged.
    """

    if isinstance(entry, tuple):
        if len(entry) == 3:
            return entry[0], entry[1], entry[2]
        return entry[0], entry[1], _DST_ROOT_MODELS
    return entry, entry, _DST_ROOT_MODELS


_HASH_CHUNK_BYTES = 1024 * 1024


def _same_content(a: Path, b: Path) -> bool:
    """Return True iff ``a`` and ``b`` share the same size and sha256.

    The size guard short-circuits the common case (a file baked into a
    base image vs the HF snapshot of a different revision will almost
    always differ in size) without paying the streamed-hash cost. When
    sizes match we fall back to a streamed sha256 because byte-identical
    files do exist across revisions (e.g. config JSONs that did not
    change) and we want the skip-copy fast path there.
    """

    import hashlib

    try:
        if a.stat().st_size != b.stat().st_size:
            return False
    except OSError:
        return False
    try:
        ah = hashlib.sha256()
        bh = hashlib.sha256()
        with a.open("rb") as af, b.open("rb") as bf:
            while True:
                ablock = af.read(_HASH_CHUNK_BYTES)
                bblock = bf.read(_HASH_CHUNK_BYTES)
                if not ablock and not bblock:
                    break
                ah.update(ablock)
                bh.update(bblock)
        return ah.digest() == bh.digest()
    except OSError:
        return False


def _hydrate_one(  # noqa: PLR0913 - seven injected params is the natural shape here
    artefact: Any,
    files: tuple[str | tuple[str, str] | tuple[str, str, str], ...],
    models_dir: Path,
    data_dir: Path,
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

    triples = [_split_entry(entry) for entry in files]
    snapshot_names = [src_name for src_name, _, _ in triples]

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

    roots = {_DST_ROOT_MODELS: models_dir, _DST_ROOT_DATA: data_dir}
    for src_name, dst_relpath, dst_root in triples:
        root = roots.get(dst_root)
        if root is None:
            logger.warning(
                "eager-pull: unknown dst_root %r for %s in %s; skip",
                dst_root,
                dst_relpath,
                artefact.hf_uri,
            )
            continue
        src = Path(snapshot_dir) / src_name
        dst = root / dst_relpath
        if not src.exists():
            logger.warning(
                "eager-pull: %r missing from snapshot of %s; skip",
                src_name,
                artefact.hf_uri,
            )
            continue
        # Drift guard: when the destination already exists and matches
        # the snapshot byte-for-byte (size + sha256) we skip the copy.
        # When it exists but differs, we OVERWRITE -- otherwise a stale
        # checkpoint baked into a base image or left over from a prior
        # revision masks the pinned artefact and the live serving model
        # silently drifts away from the registry. Production hit exactly
        # this: a regression-mode ``forecaster_best.pt`` in MODELS_DIR
        # outlived a revision bump to a classification-mode pin and
        # ``/analyze`` started returning ``regime_classification: null``
        # because the live model's output_mode mismatched.
        if dst.exists() and _same_content(dst, src):
            logger.info("eager-pull: %s already present and matches snapshot; skip", dst_relpath)
            continue
        if dst.exists():
            logger.info(
                "eager-pull: %s already present but DIFFERS from snapshot; overwriting",
                dst_relpath,
            )
        dst.parent.mkdir(parents=True, exist_ok=True)
        # Copy to a temp sibling and rename atomically so a mid-copy
        # crash leaves the target either fully intact or absent, never
        # half-written. Matters most for the forecaster checkpoint where
        # a torn write surfaces as a cryptic state_dict load error.
        dst_tmp = Path(str(dst) + ".tmp")
        shutil.copy2(src, dst_tmp)
        os.replace(dst_tmp, dst)
        logger.info(
            "eager-pull: hydrated %s <- %s @ %s",
            dst_relpath,
            artefact.hf_uri,
            revision or "main",
        )


def hydrate() -> None:
    """Pull every mapped eager artefact, copying each named file when
    absent or drifted (same size + sha256 skips, anything else
    overwrites). See the module docstring for the rationale."""

    try:
        from app.config import DATA_DIR
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

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        logger.info(
            "eager-pull: HF_TOKEN/HUGGINGFACE_HUB_TOKEN absent; "
            "skipping (cold-start will bootstrap on first /analyze)"
        )
        return

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for artefact in eager_artefacts():
        files = _ARTEFACT_FILES.get(artefact.name)
        if files is None:
            logger.debug("eager-pull: %r not in copy map; skip", artefact.name)
            continue
        _hydrate_one(artefact, files, MODELS_DIR, DATA_DIR, token, parse_hf_uri, download_snapshot)


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
