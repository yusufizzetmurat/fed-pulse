#!/usr/bin/env python3
"""Offline audit for training-package manifest sidecars.

Walks ``data/processed/`` and verifies every directory containing a
``dataset_metadata.json`` has a matching ``dataset_metadata.sha256``.
Missing sidecars print a backfill hint; mismatches exit non-zero.

Usage::

    python scripts/verify_training_package_manifests.py [--backfill]

With ``--backfill``, packages missing a sidecar get one written from
their current manifest hash. The flag is for one-shot migration of
packages built before the sidecar landed; do not use it as a routine
"make the check pass" knob — the whole point is that a mismatch
flags a silent replacement.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.data.manifest_sha import (  # noqa: E402
    ManifestShaMismatch,
    verify_manifest_sha,
    write_manifest_sha,
)


def _iter_package_dirs(processed_dir: Path) -> list[Path]:
    return sorted(
        path for path in processed_dir.iterdir()
        if path.is_dir() and (path / "dataset_metadata.json").exists()
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=ROOT / "data" / "processed",
        help="Root of training packages (default: data/processed/).",
    )
    parser.add_argument(
        "--backfill",
        action="store_true",
        help="Write sidecars for packages currently missing one. "
        "Use only for one-shot migration of pre-sidecar packages.",
    )
    args = parser.parse_args()

    if not args.processed_dir.exists():
        print(f"no training packages at {args.processed_dir}; nothing to verify")
        return 0

    packages = _iter_package_dirs(args.processed_dir)
    if not packages:
        print(f"no packages found in {args.processed_dir}")
        return 0

    failures: list[str] = []
    missing: list[Path] = []
    for package_dir in packages:
        try:
            present = verify_manifest_sha(package_dir)
        except ManifestShaMismatch as exc:
            failures.append(f"{package_dir.name}: {exc}")
            continue
        if not present:
            missing.append(package_dir)
            continue
        print(f"OK    {package_dir.name}")

    for package_dir in missing:
        if args.backfill:
            digest = write_manifest_sha(package_dir)
            print(f"BACKFILL  {package_dir.name}  sha256={digest[:12]}…")
        else:
            print(f"MISSING   {package_dir.name}  (run with --backfill to create sidecar)")

    if failures:
        print()
        print("MISMATCH detected — manifest disagrees with sidecar:")
        for entry in failures:
            print(f"  - {entry}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
