"""End-to-end reproducibility smoke (#302 Stage 5).

Reads the canonical training-package + embedding-cache URIs out of
``backend/app/models/registry.yaml``, pulls them via
``huggingface_hub.snapshot_download`` (training package) and
``hf_hub_download`` (per-encoder caches), then runs a one-epoch
forecaster training pass to confirm the data + code path is intact.

Designed for a fresh machine with only Docker + an HF token. The
expected wall time on the 8 GB / 4 vCPU droplet is ~15 minutes (~10
min for the artefact pull on a cold cache, ~5 min for the one-epoch
training).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.config import DATA_DIR  # noqa: E402
from app.models.registry import (  # noqa: E402
    artefact_ref,
    is_hf_uri,
    resolve_hf_uri,
)

CANONICAL_TP_ID = os.environ.get("FED_PULSE_REPRODUCE_TP_ID", "canonical")


def _ensure_training_package() -> Path:
    """Pull the canonical training-package dataset into ``data/processed/<id>/``."""

    ref = artefact_ref("training_package")
    if ref is None:
        raise SystemExit("artefacts.training_package missing from registry.yaml")
    if not is_hf_uri(ref.hf_uri):
        raise SystemExit(f"training_package.hf_uri is not an hf:// URI: {ref.hf_uri!r}")

    print(f"[reproduce] pulling {ref.hf_uri} ...", flush=True)
    snapshot_dir = resolve_hf_uri(ref.hf_uri)

    target = DATA_DIR / "processed" / CANONICAL_TP_ID
    target.mkdir(parents=True, exist_ok=True)
    # Copy unconditionally. The previous ``if dst.exists(): continue``
    # guard meant a second run after a revision bump silently kept the
    # old files — exactly the failure mode this script is supposed to
    # detect. ``shutil.copy2`` overwrites; ``copytree(dirs_exist_ok=True)``
    # recurses without complaining about partial copies.
    for src in snapshot_dir.iterdir():
        dst = target / src.name
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)
    print(f"[reproduce]   -> {target}", flush=True)
    return target


def _run_forecaster_smoke() -> int:
    cmd = [
        sys.executable,
        "-m",
        "app.train_forecaster",
        "--training-package-id",
        CANONICAL_TP_ID,
        "--seed",
        "11",
        "--epochs",
        "1",
    ]
    print(f"[reproduce] running {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=str(BACKEND_DIR), env={**os.environ, "PYTHONPATH": str(BACKEND_DIR)})
    return result.returncode


def main() -> int:
    start = time.time()
    if not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")):
        print("[reproduce] HF_TOKEN env var is not set — public repos will still work but rate limits apply", flush=True)

    _ensure_training_package()

    exit_code = _run_forecaster_smoke()
    elapsed = time.time() - start
    print(f"[reproduce] forecaster smoke exit={exit_code} elapsed={elapsed:.1f}s", flush=True)
    if exit_code != 0:
        print("[reproduce] FAILED — see backend logs above", flush=True)
        return exit_code
    print("[reproduce] OK — pipeline reproduced end-to-end", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
