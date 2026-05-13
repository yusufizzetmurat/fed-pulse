from __future__ import annotations

import hashlib
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import settings


_LOCK = threading.Lock()


def _resolve_audit_path(audit_path: Path | str | None = None) -> Path:
    if audit_path is not None:
        return Path(audit_path)
    return Path(settings.data_dir) / "artifacts" / "audit.log"


def append_audit_entry(
    action: str,
    *,
    run_id: str | None = None,
    before_hash: str | None = None,
    after_hash: str | None = None,
    metadata: dict[str, Any] | None = None,
    audit_path: Path | str | None = None,
) -> dict[str, Any]:
    """Append a single audit row as JSONL. Returns the row written.

    Designed for low-volume use (checkpoint write, benchmark publish, train-job
    finalisation). Locking serialises writes from the daemon thread and the
    request thread.
    """

    path = _resolve_audit_path(audit_path)
    entry: dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "action": action,
        "run_id": run_id,
        "before_hash": before_hash,
        "after_hash": after_hash,
    }
    if metadata:
        entry["metadata"] = dict(metadata)

    with _LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, default=str))
            handle.write("\n")
    return entry


def hash_file(path: Path | str) -> str | None:
    path = Path(path)
    if not path.exists():
        return None
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            sha.update(chunk)
    return sha.hexdigest()


def read_audit_entries(audit_path: Path | str | None = None) -> list[dict[str, Any]]:
    path = _resolve_audit_path(audit_path)
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out
