from __future__ import annotations

import hashlib
import json
import platform
import socket
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

MANIFEST_FILENAME = "run_manifest.json"
MANIFEST_VERSION = 1


def _git_sha(repo_root: Path | None = None) -> str | None:
    cwd = repo_root or Path(__file__).resolve().parents[2]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
            timeout=2.0,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    sha = result.stdout.strip()
    return sha or None


def _git_dirty(repo_root: Path | None = None) -> bool | None:
    cwd = repo_root or Path(__file__).resolve().parents[2]
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
            timeout=2.0,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return bool(result.stdout.strip())


def _gpu_descriptor() -> str | None:
    try:
        import torch
    except Exception:
        return None
    if not torch.cuda.is_available():
        return None
    try:
        return torch.cuda.get_device_name(0)
    except Exception:
        return None


def _torch_versions() -> dict[str, str | None]:
    try:
        import torch
    except Exception:
        return {"torch": None, "cuda": None, "cudnn": None}
    cuda = torch.version.cuda if hasattr(torch, "version") else None
    cudnn = None
    if torch.backends.cudnn.is_available():
        try:
            cudnn = str(torch.backends.cudnn.version())
        except Exception:
            cudnn = None
    return {"torch": torch.__version__, "cuda": cuda, "cudnn": cudnn}


def _hash_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_inputs(inputs: Iterable[str | Path]) -> dict[str, str]:
    resolved: dict[str, str] = {}
    for raw in inputs:
        path = Path(raw)
        if not path.exists() or not path.is_file():
            continue
        resolved[str(path)] = _hash_file(path)
    return resolved


@dataclass(frozen=True)
class RunManifest:
    manifest_version: int
    run_id: str
    written_at: str
    git_sha: str | None
    git_dirty: bool | None
    hostname: str
    python_version: str
    platform: str
    gpu: str | None
    library_versions: Mapping[str, str | None]
    cli_argv: Sequence[str]
    version_ids: Mapping[str, str]
    seeds: Sequence[int]
    hyperparameters: Mapping[str, Any]
    input_sha256: Mapping[str, str]
    extra: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_run_manifest(
    *,
    run_id: str,
    version_ids: Mapping[str, str] | None = None,
    seeds: Sequence[int] | None = None,
    hyperparameters: Mapping[str, Any] | None = None,
    inputs: Iterable[str | Path] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> RunManifest:
    return RunManifest(
        manifest_version=MANIFEST_VERSION,
        run_id=str(run_id),
        written_at=datetime.now(timezone.utc).isoformat(),
        git_sha=_git_sha(),
        git_dirty=_git_dirty(),
        hostname=socket.gethostname(),
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        gpu=_gpu_descriptor(),
        library_versions=_torch_versions(),
        cli_argv=list(sys.argv),
        version_ids=dict(version_ids or {}),
        seeds=list(seeds or ()),
        hyperparameters=dict(hyperparameters or {}),
        input_sha256=_resolve_inputs(inputs or ()),
        extra=dict(extra or {}),
    )


def write_run_manifest(
    target_dir: str | Path,
    *,
    run_id: str,
    version_ids: Mapping[str, str] | None = None,
    seeds: Sequence[int] | None = None,
    hyperparameters: Mapping[str, Any] | None = None,
    inputs: Iterable[str | Path] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    out_dir = Path(target_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_run_manifest(
        run_id=run_id,
        version_ids=version_ids,
        seeds=seeds,
        hyperparameters=hyperparameters,
        inputs=inputs,
        extra=extra,
    )
    out_path = out_dir / MANIFEST_FILENAME
    out_path.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")
    return out_path
