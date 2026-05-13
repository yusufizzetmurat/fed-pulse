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
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Ablation config not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Ablation config must be a YAML mapping: {path}")
    unknown = set(raw.keys()) - _ALLOWED_KEYS
    if unknown:
        raise ValueError(f"Unknown keys in {path}: {sorted(unknown)}")
    name = raw.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError(f"Ablation config {path} requires a non-empty 'name' field.")
    text_channel = raw.get("text_channel", "scalar")
    if text_channel not in {"scalar", "embeddings"}:
        raise ValueError(
            f"Ablation config {path}: text_channel must be 'scalar' or 'embeddings'."
        )
    overrides = raw.get("feature_overrides", {}) or {}
    if not isinstance(overrides, dict):
        raise ValueError(f"Ablation config {path}: feature_overrides must be a mapping.")
    return AblationConfig(
        name=name,
        text_channel=text_channel,
        zero_text=bool(raw.get("zero_text", False)),
        calendar_only=bool(raw.get("calendar_only", False)),
        embedding_adapter_dim=int(raw.get("embedding_adapter_dim", 128)),
        feature_overrides=dict(overrides),
        description=raw.get("description"),
    )
