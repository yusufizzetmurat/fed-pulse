"""SHA-256 sidecar for training-package immutability guarantees.

The benchmark policy under ``docs/benchmark-policy.md`` promises that
published training packages are never silently replaced. Without an
explicit hash, a rebuild that lands at the same directory name would
violate the promise undetectably. This module writes a
``dataset_metadata.sha256`` sidecar at publish time and verifies it at
load time.

The hash covers ``dataset_metadata.json`` only — the manifest itself
already references the contents (row counts, source counts, fold
boundaries, drift values), so any change to the underlying parquet
that should be visible to a downstream consumer must round-trip through
the manifest.

Loader behaviour:

- Sidecar present and matches: pass silently.
- Sidecar present and mismatches: raise ``ManifestShaMismatch``.
- Sidecar missing: log a warning so unsidecarred packages surface for
  backfill, but do not block loads. Existing pre-sidecar packages
  remain usable until they are rebuilt.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

_MANIFEST_NAME = "dataset_metadata.json"
_SIDECAR_NAME = "dataset_metadata.sha256"

_logger = logging.getLogger(__name__)


class ManifestShaMismatch(RuntimeError):
    """Raised when the manifest hash does not match the sidecar."""


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compute_manifest_sha(package_dir: Path) -> str:
    """Return the SHA-256 hex digest of the package's manifest file."""

    manifest_path = package_dir / _MANIFEST_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found at {manifest_path}")
    return _hash_file(manifest_path)


def write_manifest_sha(package_dir: Path) -> str:
    """Compute the manifest hash and write it to the sidecar.

    Returns the computed hex digest so callers can log or surface it.
    Overwrites any existing sidecar — the canonical hash is whatever
    the current manifest evaluates to at write time.
    """

    digest = compute_manifest_sha(package_dir)
    (package_dir / _SIDECAR_NAME).write_text(digest + "\n", encoding="utf-8")
    return digest


def verify_manifest_sha(package_dir: Path) -> bool:
    """Verify ``dataset_metadata.sha256`` matches ``dataset_metadata.json``.

    Returns ``True`` when the sidecar exists and matches. Returns
    ``False`` when the sidecar is absent (logged as a warning for
    backfill follow-up). Raises :class:`ManifestShaMismatch` when the
    sidecar exists but disagrees with the manifest — that case
    indicates a silent replacement, which the policy forbids.
    """

    sidecar = package_dir / _SIDECAR_NAME
    if not sidecar.exists():
        _logger.warning(
            "manifest_sha_sidecar_missing",
            extra={"package_dir": str(package_dir)},
        )
        return False
    expected = sidecar.read_text(encoding="utf-8").strip()
    actual = compute_manifest_sha(package_dir)
    if expected != actual:
        raise ManifestShaMismatch(
            f"manifest hash mismatch in {package_dir.name}: "
            f"sidecar reports {expected[:12]}…, manifest hashes to {actual[:12]}… "
            "— the package has been replaced after publish, which "
            "docs/benchmark-policy.md forbids. Rebuild and republish, "
            "or restore the original manifest."
        )
    return True
