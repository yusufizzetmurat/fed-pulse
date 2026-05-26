"""Tests for the ``hf://`` URI resolver on the model registry (#302)."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.models import registry


def test_is_hf_uri_detects_prefix() -> None:
    assert registry.is_hf_uri("hf://owner/name")
    assert registry.is_hf_uri("hf://datasets/owner/name")
    assert not registry.is_hf_uri("/data/checkpoint")
    assert not registry.is_hf_uri("ProsusAI/finbert")


def test_parse_hf_uri_model_repo() -> None:
    ref = registry.parse_hf_uri("hf://yusufizzetmurat/fed-pulse-forecaster")
    assert ref.repo_id == "yusufizzetmurat/fed-pulse-forecaster"
    assert ref.repo_type == "model"
    assert ref.revision is None
    assert not ref.is_dataset


def test_parse_hf_uri_dataset_with_revision() -> None:
    ref = registry.parse_hf_uri(
        "hf://datasets/yusufizzetmurat/fed-pulse-training-package:abc1234"
    )
    assert ref.repo_id == "yusufizzetmurat/fed-pulse-training-package"
    assert ref.repo_type == "dataset"
    assert ref.revision == "abc1234"
    assert ref.is_dataset


def test_parse_hf_uri_rejects_missing_owner() -> None:
    with pytest.raises(ValueError, match="owner/name"):
        registry.parse_hf_uri("hf://just-a-name")


def test_parse_hf_uri_rejects_empty_body() -> None:
    with pytest.raises(ValueError, match="Empty hf:// URI body"):
        registry.parse_hf_uri("hf://")


def test_parse_hf_uri_rejects_non_hf_uri() -> None:
    with pytest.raises(ValueError, match="Not an hf:// URI"):
        registry.parse_hf_uri("/local/path/to/checkpoint")


def test_parse_hf_uri_rejects_path_traversal() -> None:
    # ``..`` in either segment must not be accepted — the resolver
    # would otherwise pass the path through to snapshot_download where
    # a malformed value could land artefacts outside the cache root.
    for malformed in (
        "hf://../../etc/passwd",
        "hf://owner/..",
        "hf://../name",
        "hf://datasets/../escape",
    ):
        with pytest.raises(ValueError):
            registry.parse_hf_uri(malformed)


def test_parse_hf_uri_rejects_trailing_slash() -> None:
    with pytest.raises(ValueError, match="trailing slash"):
        registry.parse_hf_uri("hf://owner/name/")


def test_parse_hf_uri_rejects_multi_colon() -> None:
    with pytest.raises(ValueError, match="multiple ':'"):
        registry.parse_hf_uri("hf://owner/name:rev1:rev2")


def test_parse_hf_uri_rejects_empty_revision() -> None:
    with pytest.raises(ValueError, match="empty revision"):
        registry.parse_hf_uri("hf://owner/name:")


def test_parse_hf_uri_rejects_empty_repo_id_side() -> None:
    # ``/name`` (empty owner) and ``owner/`` (empty name) must both fail.
    for malformed in ("hf:///name", "hf://owner/"):
        with pytest.raises(ValueError):
            registry.parse_hf_uri(malformed)


def test_parse_hf_uri_rejects_illegal_revision_characters() -> None:
    # Revisions are restricted to ``[a-zA-Z0-9._-]+`` — slashes, spaces
    # and shell metacharacters all reject. Tags and branches in the wild
    # do contain ``/`` but huggingface_hub itself treats those as
    # path-like references; the resolver intentionally rejects them so
    # the registry stays auditable.
    for malformed in (
        "hf://owner/name:rev/with/slash",
        "hf://owner/name:rev with space",
        "hf://owner/name:rev;rm",
    ):
        with pytest.raises(ValueError):
            registry.parse_hf_uri(malformed)


def test_resolve_hf_uri_invokes_snapshot_download(monkeypatch, tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []

    def fake_snapshot_download(**kwargs: object) -> str:
        calls.append(dict(kwargs))
        return str(tmp_path / "snapshot")

    import huggingface_hub  # type: ignore[import-not-found]

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    monkeypatch.setenv("HF_TOKEN", "fake-token-for-test")

    out = registry.resolve_hf_uri(
        "hf://yusufizzetmurat/fed-pulse-forecaster:rev-aaa",
        cache_dir=tmp_path / "cache",
    )

    assert out == tmp_path / "snapshot"
    assert len(calls) == 1
    call = calls[0]
    assert call["repo_id"] == "yusufizzetmurat/fed-pulse-forecaster"
    assert call["repo_type"] == "model"
    assert call["revision"] == "rev-aaa"
    assert call["cache_dir"] == str(tmp_path / "cache")
    assert call["token"] == "fake-token-for-test"


def test_resolve_hf_uri_dataset_passes_repo_type(monkeypatch, tmp_path: Path) -> None:
    seen: dict[str, object] = {}

    def fake_snapshot_download(**kwargs: object) -> str:
        seen.update(kwargs)
        return str(tmp_path / "ds")

    import huggingface_hub  # type: ignore[import-not-found]

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)

    out = registry.resolve_hf_uri(
        "hf://datasets/yusufizzetmurat/fed-pulse-training-package"
    )

    assert out == tmp_path / "ds"
    assert seen["repo_type"] == "dataset"
    assert seen["repo_id"] == "yusufizzetmurat/fed-pulse-training-package"
    assert "revision" not in seen  # no pin on this URI
    assert "token" not in seen  # no env, no explicit token


def test_resolve_repo_passes_through_local_path(monkeypatch) -> None:
    # Local paths and plain HF repo ids should be returned verbatim — no
    # network call. The dev stack relies on this so the registry resolver
    # is back-compat by default.
    import huggingface_hub  # type: ignore[import-not-found]

    def fail(*_args: object, **_kwargs: object) -> str:
        raise AssertionError("snapshot_download must not be invoked for a local path")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fail)

    assert registry.resolve_repo("/data/artifacts/encoder") == "/data/artifacts/encoder"
    assert registry.resolve_repo("ProsusAI/finbert") == "ProsusAI/finbert"


def test_resolve_repo_dispatches_hf_uri_to_resolver(monkeypatch, tmp_path: Path) -> None:
    import huggingface_hub  # type: ignore[import-not-found]

    monkeypatch.setattr(
        huggingface_hub,
        "snapshot_download",
        lambda **_kwargs: str(tmp_path / "resolved"),
    )
    monkeypatch.setenv("HF_TOKEN", "token-for-resolve")

    out = registry.resolve_repo(
        "hf://yusufizzetmurat/fed-pulse-forecaster",
        cache_dir=tmp_path / "cache",
    )
    assert out == str(tmp_path / "resolved")


def test_artefact_registry_loads_eager_set() -> None:
    # The boot entrypoint pulls these before uvicorn starts. The set is
    # the hot path: canonical DAPT encoder + canonical forecaster +
    # rates heads + retrieval bundle + trajectory bundle. The training
    # package + embedding caches stay lazy.
    eager = {a.name for a in registry.eager_artefacts()}
    assert "encoder_canonical" in eager
    assert "forecaster_canonical" in eager
    assert "retrieval_bundle" in eager
    assert "trajectory_bundle" in eager
    assert "rates_heads_canonical" in eager
    assert "training_package" not in eager
    assert "embedding_caches" not in eager


def test_artefact_registry_hf_uri_schema() -> None:
    ref = registry.artefact_ref("encoder_canonical")
    assert ref is not None
    assert ref.hf_uri.startswith("hf://yusufizzetmurat/")
    assert registry.is_hf_uri(ref.hf_uri)
    parsed = registry.parse_hf_uri(ref.hf_uri)
    assert parsed.repo_id == "yusufizzetmurat/finbert-fed-adjacent-xbank-dapt"
    assert not parsed.is_dataset
