"""Unit tests for app.data.embedding_cache (encoder-keyed cache)."""

from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.data import embedding_cache as ec


@pytest.fixture(autouse=True)
def _clear_lru_cache():
    yield


def _write_registry(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_resolve_cache_paths_embeds_short_revision(tmp_path: Path) -> None:
    paths = ec.resolve_cache_paths("finbert_fomc", revision="abc1234567890def" * 4, cache_dir=tmp_path)
    assert paths.parquet.name == "finbert_fomc_abc123456789.parquet"
    assert paths.sources_lock == tmp_path / "SOURCES.lock"


def test_resolve_cache_paths_handles_unpinned_revision(tmp_path: Path) -> None:
    paths = ec.resolve_cache_paths("finbert_fed_adjacent", revision=None, cache_dir=tmp_path)
    assert paths.parquet.name == "finbert_fed_adjacent_unpinned.parquet"


def test_resolve_cache_paths_normalises_hyphens_in_revision(tmp_path: Path) -> None:
    """Voyage-style revisions carry hyphens; the cache writer normalises
    them to underscores. The resolver must apply the same normalisation
    so the loader finds the parquet on disk."""

    paths = ec.resolve_cache_paths(
        "voyage_finance_2", revision="voyage-finance-2", cache_dir=tmp_path
    )
    assert paths.parquet.name == "voyage_finance_2_voyage_finan.parquet"


def test_build_cache_hard_fails_without_allow_network(tmp_path: Path, monkeypatch) -> None:
    """The training-time invariant: never silently download at runtime."""

    monkeypatch.setattr(
        ec, "encoder_ref",
        lambda alias: types.SimpleNamespace(
            alias=alias, repo="ProsusAI/finbert", revision="rev1234567890ab", gated=False,
            task="classification", description="",
        ),
    )
    registry = tmp_path / "processed" / "pkg" / "registry_normalized.jsonl"
    _write_registry(registry, [{"text": "hello", "event_date": "2024-01-01"}])

    with pytest.raises(RuntimeError, match="Embedding cache missing"):
        ec.build_cache(
            encoder_alias="finbert",
            training_package_id="pkg",
            data_dir=tmp_path,
            cache_dir=tmp_path / "embeddings",
            allow_network=False,
        )


def test_build_cache_raises_on_unpinned_encoder(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        ec, "encoder_ref",
        lambda alias: types.SimpleNamespace(
            alias=alias, repo="local/finbert-fed-adjacent", revision="", gated=False,
            task="masked_lm", description="",
        ),
    )
    with pytest.raises(ValueError, match="no pinned revision"):
        ec.build_cache(
            encoder_alias="finbert_fed_adjacent",
            training_package_id="pkg",
            data_dir=tmp_path,
            cache_dir=tmp_path / "embeddings",
            allow_network=True,
        )


def test_build_cache_writes_parquet_and_sources_lock(tmp_path: Path, monkeypatch) -> None:
    """End-to-end: builder calls a faked encoder, writes the parquet, and
    appends a SOURCES.lock line capturing the artefact SHA-256."""

    monkeypatch.setattr(
        ec, "encoder_ref",
        lambda alias: types.SimpleNamespace(
            alias=alias, repo="ProsusAI/finbert", revision="rev1234567890ab", gated=False,
            task="classification", description="",
        ),
    )

    class _FakeTokenizer:
        model_max_length = 512

        def encode(self, text, add_special_tokens=False):
            return list(range(len(text.split())))

        def decode(self, ids, skip_special_tokens=True):
            return "chunk-" + str(len(ids))

        def __call__(self, inputs, **kwargs):
            import torch as _torch

            batch = len(inputs)
            return {
                "input_ids": _torch.zeros((batch, 4), dtype=_torch.long),
                "attention_mask": _torch.ones((batch, 4), dtype=_torch.long),
            }

    class _FakeModel:
        def __init__(self):
            import torch as _torch

            self._weight = _torch.nn.Parameter(_torch.zeros(1))

        def parameters(self):
            return iter([self._weight])

        def eval(self):
            return self

        def __call__(self, **kwargs):
            import torch as _torch

            batch = kwargs["input_ids"].shape[0]
            hidden = _torch.zeros((batch, 4, 6))
            for b in range(batch):
                hidden[b, 0, :] = float(b + 1)  # distinguishable CLS row per item
            return types.SimpleNamespace(last_hidden_state=hidden)

    monkeypatch.setattr(
        ec, "_load_encoder", lambda ref: (_FakeTokenizer(), _FakeModel())
    )

    registry = tmp_path / "processed" / "pkg" / "registry_normalized.jsonl"
    _write_registry(
        registry,
        [
            {
                "text": "hawkish phrasing about the policy outlook",
                "event_date": "2024-01-01",
                "record_id": "rec-1",
            },
            {
                "text": "dovish phrasing about the policy outlook",
                "event_date": "2024-02-01",
                "record_id": "rec-2",
            },
        ],
    )

    result = ec.build_cache(
        encoder_alias="finbert",
        training_package_id="pkg",
        data_dir=tmp_path,
        cache_dir=tmp_path / "embeddings",
        batch_size=2,
        min_text_chars=0,
        max_length=8,
        allow_network=True,
    )

    assert result.parquet_path.exists()
    assert result.encoder_revision == "rev1234567890ab"
    assert result.encoder_alias == "finbert"

    df = pd.read_parquet(result.parquet_path)
    assert set(df.columns) == {"record_id", "doc_id", "event_date", "chunk_index", "chunk_preview", "embedding"}
    assert len(df) == 2

    assert result.sources_lock_path.exists()
    line = result.sources_lock_path.read_text(encoding="utf-8").strip().splitlines()[-1]
    entry = json.loads(line)
    assert entry["encoder_alias"] == "finbert"
    assert entry["encoder_revision"] == "rev1234567890ab"
    assert entry["row_count"] == 2
    expected_sha = hashlib.sha256(result.parquet_path.read_bytes()).hexdigest()
    assert entry["parquet_sha256"] == expected_sha


def test_require_cache_exists_error_message_is_actionable(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="make cache-embeddings ENCODER=finbert_fomc"):
        ec.require_cache_exists(
            "finbert_fomc",
            revision="abc1234567890def",
            cache_dir=tmp_path,
        )


def test_build_cache_reuses_existing_parquet_without_network(tmp_path: Path, monkeypatch) -> None:
    """Second invocation reads the cached parquet without invoking the encoder."""

    monkeypatch.setattr(
        ec, "encoder_ref",
        lambda alias: types.SimpleNamespace(
            alias=alias, repo="ProsusAI/finbert", revision="rev1234567890ab", gated=False,
            task="classification", description="",
        ),
    )
    paths = ec.resolve_cache_paths("finbert", revision="rev1234567890ab", cache_dir=tmp_path / "embeddings")
    paths.parquet.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        [
            {"record_id": "x", "doc_id": "x", "event_date": "2024-01-01", "chunk_index": 0,
             "chunk_preview": "y", "embedding": [0.1, 0.2]},
        ]
    )
    df.to_parquet(paths.parquet, index=False)

    registry = tmp_path / "processed" / "pkg" / "registry_normalized.jsonl"
    _write_registry(registry, [{"text": "x", "event_date": "2024-01-01"}])

    def _explode(_ref):
        raise AssertionError("encoder loader should not be called on a cache hit")

    monkeypatch.setattr(ec, "_load_encoder", _explode)

    result = ec.build_cache(
        encoder_alias="finbert",
        training_package_id="pkg",
        data_dir=tmp_path,
        cache_dir=tmp_path / "embeddings",
        allow_network=False,
    )
    assert result.row_count == 1


def test_resolve_cache_device_prefers_cuda_when_available(monkeypatch) -> None:
    """#553: default to CUDA when ``torch.cuda.is_available()`` returns True.

    Pre-#553 ``_load_encoder`` never moved the model off CPU even on a pod
    with an idle GPU; rebuilding the wave-4 bake-off caches ran ~10× slower
    than necessary. The default-on CUDA selection is the operational fix.
    """

    import torch

    monkeypatch.delenv("FED_PULSE_EMBEDDING_CACHE_DEVICE", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert ec._resolve_cache_device() == "cuda"


def test_resolve_cache_device_falls_back_to_cpu_without_cuda(monkeypatch) -> None:
    """CPU fallback keeps the laptop / CI builds byte-identical to pre-#553."""

    import torch

    monkeypatch.delenv("FED_PULSE_EMBEDDING_CACHE_DEVICE", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert ec._resolve_cache_device() == "cpu"


def test_resolve_cache_device_env_override_wins(monkeypatch) -> None:
    """``FED_PULSE_EMBEDDING_CACHE_DEVICE`` forces a specific device.

    Useful when the GPU is occupied by a training run and the operator
    wants to share rather than queue. The override beats the
    is_available probe in both directions.
    """

    import torch

    monkeypatch.setenv("FED_PULSE_EMBEDDING_CACHE_DEVICE", "cpu")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert ec._resolve_cache_device() == "cpu"

    monkeypatch.setenv("FED_PULSE_EMBEDDING_CACHE_DEVICE", "cuda")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert ec._resolve_cache_device() == "cuda"
