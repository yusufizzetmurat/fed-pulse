"""Tests for the HF-Hub lazy-fetch path on the embedding cache (#302)."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.data import embedding_cache


def test_ensure_local_returns_existing_file_without_network(tmp_path: Path, monkeypatch) -> None:
    parquet = tmp_path / "finbert_fed_adj_20260515T104.parquet"
    parquet.write_bytes(b"local-content")

    import huggingface_hub  # type: ignore[import-not-found]

    def fail(**_kwargs: object) -> str:
        raise AssertionError("hf_hub_download must not be invoked when the cache hits")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fail)

    out = embedding_cache.ensure_local(
        "finbert_fed_adj",
        revision="20260515T104824Z",
        cache_dir=tmp_path,
    )
    assert out == parquet
    assert out.read_bytes() == b"local-content"


def test_ensure_local_falls_through_to_hf_on_miss(tmp_path: Path, monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_hf_hub_download(**kwargs: object) -> str:
        calls.append(dict(kwargs))
        local_dir = Path(str(kwargs["local_dir"]))
        local_dir.mkdir(parents=True, exist_ok=True)
        target = local_dir / str(kwargs["filename"])
        target.write_bytes(b"remote-bytes")
        return str(target)

    import huggingface_hub  # type: ignore[import-not-found]

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_hf_hub_download)
    monkeypatch.setenv("HF_TOKEN", "fake-token")

    out = embedding_cache.ensure_local(
        "bge_large_en_v15",
        revision="d4aa6901d3a4",
        cache_dir=tmp_path,
    )

    assert out.exists()
    assert out.read_bytes() == b"remote-bytes"
    assert len(calls) == 1
    call = calls[0]
    assert call["repo_type"] == "dataset"
    assert call["repo_id"] == embedding_cache.HF_EMBEDDING_CACHE_DATASET
    assert call["filename"] == "bge_large_en_v15_d4aa6901d3a4.parquet"
    assert call["token"] == "fake-token"

    # Second call hits the local cache — no further network.
    monkeypatch.setattr(
        huggingface_hub,
        "hf_hub_download",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("second call must hit local cache"),
        ),
    )
    second = embedding_cache.ensure_local(
        "bge_large_en_v15",
        revision="d4aa6901d3a4",
        cache_dir=tmp_path,
    )
    assert second == out


def test_ensure_local_raises_when_fetch_disabled(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Embedding cache missing"):
        embedding_cache.ensure_local(
            "bge_large_en_v15",
            revision="d4aa6901d3a4",
            cache_dir=tmp_path,
            allow_hf_fetch=False,
        )


def test_ensure_local_omits_token_when_unset(tmp_path: Path, monkeypatch) -> None:
    seen: dict[str, object] = {}

    def fake_hf_hub_download(**kwargs: object) -> str:
        seen.update(kwargs)
        local_dir = Path(str(kwargs["local_dir"]))
        local_dir.mkdir(parents=True, exist_ok=True)
        target = local_dir / str(kwargs["filename"])
        target.write_bytes(b"")
        return str(target)

    import huggingface_hub  # type: ignore[import-not-found]

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_hf_hub_download)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)

    embedding_cache.ensure_local(
        "nomic_embed",
        revision="b0753ae763",
        cache_dir=tmp_path,
    )
    assert "token" not in seen
