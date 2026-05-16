from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


_ALLOWED_KEYS = {
    "name",
    "description",
    "text_channel",
    "zero_text",
    "calendar_only",
    "feature_overrides",
    "embedding_adapter_dim",
}


# Candidate ``configs/`` directories searched in order. The loader works
# from both the host repo root and the ``/app`` container layout:
#
# - Host: ``backend/app/training/config_loader.py`` ``parents[3]`` is the
#   repo root, where ``configs/`` lives alongside ``backend/``.
# - Container: the same path resolves to ``/`` (``backend`` is mounted
#   at ``/app``), so the repo-root candidate misses. Compose mounts
#   ``./configs`` at ``/app/configs`` (``parents[2] / "configs"``) so
#   the second candidate finds the file.
_CONFIGS_DIR_CANDIDATES: tuple[Path, ...] = (
    (Path(__file__).resolve().parents[3] / "configs"),
    (Path(__file__).resolve().parents[2] / "configs"),
)


def _resolve_config_path(path: Path | str) -> Path:
    """Resolve an ablation-config path against a known configs directory.

    Three resolution strategies are tried in order:

    1. Absolute paths and paths that exist as given (relative to the
       current cwd) are returned unchanged.
    2. Bare filenames (e.g. ``configs/ablation_no_text.yaml`` or
       ``ablation_no_text.yaml``) are tried against every entry in
       :data:`_CONFIGS_DIR_CANDIDATES`. The leading ``configs/``
       segment is stripped before the join so both forms resolve to the
       same file.
    3. As a last resort the path is walked upward from the cwd until a
       sibling ``configs/`` directory containing the file is found. This
       covers test runners invoked from arbitrary subdirectories.

    The returned path is not guaranteed to exist; the caller is
    responsible for the ``FileNotFoundError`` raise so the existing
    error message stays stable.
    """

    candidate = Path(path)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    if candidate.exists():
        return candidate

    bare_name = candidate.name
    parts = candidate.parts
    relative_under_configs: Path
    if parts and parts[0] == "configs":
        relative_under_configs = Path(*parts[1:]) if len(parts) > 1 else Path(bare_name)
    else:
        relative_under_configs = candidate

    for configs_dir in _CONFIGS_DIR_CANDIDATES:
        attempt = configs_dir / relative_under_configs
        if attempt.exists():
            return attempt
        bare_attempt = configs_dir / bare_name
        if bare_attempt.exists():
            return bare_attempt

    # Walk upward from cwd looking for a sibling ``configs/`` directory
    # that holds the file; this covers the rare case of the loader being
    # invoked from a deeply nested path.
    walker = Path.cwd().resolve()
    for parent in [walker, *walker.parents]:
        attempt = parent / "configs" / relative_under_configs
        if attempt.exists():
            return attempt

    return candidate


@dataclass(frozen=True)
class AblationConfig:
    name: str
    text_channel: str = "scalar"
    zero_text: bool = False
    calendar_only: bool = False
    embedding_adapter_dim: int = 128
    feature_overrides: dict[str, Any] = field(default_factory=dict)
    description: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "text_channel": self.text_channel,
            "zero_text": self.zero_text,
            "calendar_only": self.calendar_only,
            "embedding_adapter_dim": self.embedding_adapter_dim,
            "feature_overrides": dict(self.feature_overrides),
            "description": self.description,
        }


def load_ablation_config(path: Path | str) -> AblationConfig:
    resolved = _resolve_config_path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Ablation config not found: {path}")
    raw = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Ablation config must be a YAML mapping: {resolved}")
    unknown = set(raw.keys()) - _ALLOWED_KEYS
    if unknown:
        raise ValueError(f"Unknown keys in {resolved}: {sorted(unknown)}")
    name = raw.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError(f"Ablation config {resolved} requires a non-empty 'name' field.")
    text_channel = raw.get("text_channel", "scalar")
    if text_channel not in {"scalar", "embeddings"}:
        raise ValueError(
            f"Ablation config {resolved}: text_channel must be 'scalar' or 'embeddings'."
        )
    overrides = raw.get("feature_overrides", {}) or {}
    if not isinstance(overrides, dict):
        raise ValueError(f"Ablation config {resolved}: feature_overrides must be a mapping.")
    return AblationConfig(
        name=name,
        text_channel=text_channel,
        zero_text=bool(raw.get("zero_text", False)),
        calendar_only=bool(raw.get("calendar_only", False)),
        embedding_adapter_dim=int(raw.get("embedding_adapter_dim", 128)),
        feature_overrides=dict(overrides),
        description=raw.get("description"),
    )
