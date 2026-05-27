"""Integrity pin for the B1 (#212) LLM-as-features cache (#337).

The cache parquet is the authoritative artefact for the §6.6 Tier 4 /
Tier 5 results — it was extracted against the Claude Sonnet 4.6 model
snapshot at temperature 0 and is not re-extractable from a future
model once Anthropic deprecates `claude-sonnet-4-6`. The integrity
contract is the `llm_features:` block in
``backend/app/models/registry.yaml``: this test asserts the block
carries every required key and, when the cache file is on disk, that
the pinned SHA matches the file. CI runs without the cache mounted —
the second assertion skips cleanly in that environment.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

from app.models.registry import MODEL_REGISTRY_PATH

_REQUIRED_KEYS = (
    "cache_path",
    "sha256",
    "size_bytes",
    "model",
    "temperature",
    "pinned_at",
    "immutability_note",
)


def _load_llm_features_block() -> dict:
    raw = yaml.safe_load(MODEL_REGISTRY_PATH.read_text(encoding="utf-8"))
    assert "llm_features" in raw, (
        "registry.yaml must carry a top-level `llm_features:` block (#337)"
    )
    return raw["llm_features"]


def test_registry_carries_llm_features_block_with_required_keys() -> None:
    block = _load_llm_features_block()
    missing = [k for k in _REQUIRED_KEYS if k not in block]
    assert not missing, f"llm_features block missing required keys: {missing!r}"


def test_llm_features_block_field_shapes() -> None:
    block = _load_llm_features_block()
    sha = str(block["sha256"]).strip()
    assert len(sha) == 64 and all(c in "0123456789abcdef" for c in sha), (
        f"sha256 must be a 64-char lowercase hex digest; got {sha!r}"
    )
    assert int(block["size_bytes"]) > 0, "size_bytes must be a positive int"
    assert block["model"] == "claude-sonnet-4-6", (
        "model snapshot pin must read 'claude-sonnet-4-6' — the cache binds "
        "reproducibility to this exact Anthropic snapshot"
    )
    assert float(block["temperature"]) == 0.0, "temperature must be 0"
    # Smoke check on the immutability note: it must mention the ADR
    # escape hatch so a future contributor cannot silently regenerate.
    assert "ADR" in str(block["immutability_note"]), (
        "immutability_note must reference the ADR escape hatch"
    )


def _resolve_cache_path(cache_path_rel: str) -> Path | None:
    # registry.yaml is at <repo>/backend/app/models/registry.yaml; the
    # cache path is repo-relative. Walk up four parents to land at the
    # repo root, then join.
    repo_root = MODEL_REGISTRY_PATH.resolve().parents[3]
    candidate = repo_root / cache_path_rel
    if candidate.exists():
        return candidate
    # Fall back to the CWD-relative path (CI runners sometimes mount
    # the repo at the working directory rather than the conventional
    # location).
    cwd_candidate = Path.cwd() / cache_path_rel
    if cwd_candidate.exists():
        return cwd_candidate
    return None


def test_pinned_sha_matches_cache_when_present() -> None:
    block = _load_llm_features_block()
    cache_path_rel = str(block["cache_path"])
    cache_path = _resolve_cache_path(cache_path_rel)
    if cache_path is None:
        pytest.skip(
            f"LLM-features cache not present at {cache_path_rel!r}; the "
            "registry pin is asserted on the dev / GPU host only."
        )
    expected_sha = str(block["sha256"]).strip().lower()
    expected_size = int(block["size_bytes"])
    actual_size = cache_path.stat().st_size
    assert actual_size == expected_size, (
        f"cache size drift: pinned {expected_size}, on-disk {actual_size} "
        f"({cache_path}); the cache file must not be regenerated outside "
        "an ADR (see docs/data-and-training-contracts.md)"
    )
    hasher = hashlib.sha256()
    with cache_path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            hasher.update(chunk)
    actual_sha = hasher.hexdigest()
    assert actual_sha == expected_sha, (
        f"cache sha drift: pinned {expected_sha}, on-disk {actual_sha} "
        f"({cache_path}); the cache file must not be regenerated outside "
        "an ADR (see docs/data-and-training-contracts.md)"
    )
