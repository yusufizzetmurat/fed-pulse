from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.data import llm_embedding_store


class _StubEmbeddingClient:
    def __init__(self, vectors):
        self._vectors = list(vectors)

    def embed_content(self, content, **kwargs):
        v = self._vectors.pop(0)
        wrapper = type("R", (), {"embedding": type("E", (), {"values": v})()})
        return wrapper()


def _write_registry(path: Path, n: int = 3) -> None:
    rows = []
    for i in range(n):
        rows.append(
            {
                "record_id": f"doc{i}",
                "event_date": f"2024-01-{i+1:02d}",
                "text": f"document {i} body",
                "source_type": "fomc_minutes",
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as h:
        for r in rows:
            h.write(json.dumps(r) + "\n")


def test_precompute_embeddings_writes_one_row_per_document(tmp_path: Path) -> None:
    registry = tmp_path / "registry.jsonl"
    _write_registry(registry, n=3)
    output = tmp_path / "llm_embeddings.parquet"
    client = _StubEmbeddingClient(
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
        ]
    )

    written = llm_embedding_store.precompute_embeddings(
        registry_path=registry,
        output_path=output,
        embedding_client=client,
    )

    assert written == 3
    # Output may be parquet or jsonl-fallback if pyarrow missing
    assert output.exists() or output.with_suffix(".jsonl").exists()


def test_precompute_embeddings_skips_empty_text_rows(tmp_path: Path) -> None:
    registry = tmp_path / "registry.jsonl"
    rows = [
        {"record_id": "d0", "event_date": "2024-01-01", "text": "real text", "source_type": "fomc_minutes"},
        {"record_id": "d1", "event_date": "2024-01-02", "text": "", "source_type": "fomc_minutes"},
        {"record_id": "d2", "event_date": "2024-01-03", "text": "another", "source_type": "fomc_minutes"},
    ]
    registry.parent.mkdir(parents=True, exist_ok=True)
    with registry.open("w", encoding="utf-8") as h:
        for r in rows:
            h.write(json.dumps(r) + "\n")

    output = tmp_path / "llm_embeddings.parquet"
    client = _StubEmbeddingClient([[0.1, 0.2], [0.3, 0.4]])  # only 2 vectors needed

    written = llm_embedding_store.precompute_embeddings(
        registry_path=registry,
        output_path=output,
        embedding_client=client,
    )
    assert written == 2  # the empty-text row is skipped


def test_parse_args_requires_no_extra_args() -> None:
    """All args have defaults; bare _parse_args([]) should not raise."""

    args = llm_embedding_store._parse_args([])
    assert args.embedding_model == "gemini-embedding-001"
    assert args.request_interval_seconds == 0.0
    assert args.max_rows == 0
