from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

MODEL_REGISTRY_PATH = Path(__file__).resolve().parent / "registry.yaml"


@dataclass(frozen=True)
class EncoderRef:
    alias: str
    repo: str
    revision: str
    gated: bool
    task: str
    description: str


@lru_cache(maxsize=1)
def load_registry(path: Path | None = None) -> dict[str, EncoderRef]:
    target = path or MODEL_REGISTRY_PATH
    raw = yaml.safe_load(target.read_text(encoding="utf-8"))
    encoders = (raw or {}).get("encoders") or {}
    by_repo: dict[str, EncoderRef] = {}
    for alias, fields in encoders.items():
        if not isinstance(fields, dict):
            continue
        ref = EncoderRef(
            alias=alias,
            repo=str(fields["repo"]),
            revision=str(fields["revision"]),
            gated=bool(fields.get("gated", False)),
            task=str(fields.get("task", "classification")),
            description=str(fields.get("description", "")),
        )
        by_repo[ref.repo] = ref
        by_repo[ref.alias] = ref
        for repo_alias in fields.get("repo_aliases") or ():
            by_repo[str(repo_alias)] = ref
    return by_repo


def revision_for(repo_or_alias: str) -> str | None:
    registry = load_registry()
    ref = registry.get(repo_or_alias)
    if ref is None:
        return None
    return ref.revision


def encoder_ref(repo_or_alias: str) -> EncoderRef | None:
    return load_registry().get(repo_or_alias)


def from_pretrained_kwargs(repo_or_alias: str, **extra: Any) -> dict[str, Any]:
    revision = revision_for(repo_or_alias)
    if revision is None:
        raise ValueError(
            f"Model '{repo_or_alias}' is not pinned in models/registry.yaml. "
            "Add a revision before loading."
        )
    return {"revision": revision, **extra}
