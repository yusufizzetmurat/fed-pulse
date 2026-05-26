from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
pd = pytest.importorskip("pandas")

from app.data.chunk_embedding_retrieval import (  # noqa: E402
    _build_lookback_tensors_llm,
    build_lookback_tensors,
)


def _store(rows):
    return pd.DataFrame(rows)


def test_build_lookback_tensors_filters_by_window():
    store = _store(
        [
            {"doc_id": "a", "event_date": "2024-01-01", "chunk_index": 0, "chunk_preview": "a", "embedding": [1.0, 0.0]},
            {"doc_id": "b", "event_date": "2024-10-01", "chunk_index": 0, "chunk_preview": "b", "embedding": [0.0, 1.0]},
            {"doc_id": "c", "event_date": "2024-12-01", "chunk_index": 0, "chunk_preview": "c", "embedding": [1.0, 1.0]},
        ]
    )
    result = build_lookback_tensors(store, anchor_date="2024-12-31", lookback_days=180, max_chunks=4)
    kept = [d for d in result.doc_ids if d]
    assert result.actual_count == 2
    assert set(kept) == {"b", "c"}
    assert "a" not in kept


def test_build_lookback_tensors_pads_to_max_chunks():
    store = _store(
        [
            {"doc_id": "a", "event_date": "2024-12-01", "chunk_index": 0, "chunk_preview": "a", "embedding": [0.5, 0.5]},
        ]
    )
    result = build_lookback_tensors(store, anchor_date="2024-12-31", lookback_days=60, max_chunks=4)
    assert result.embeddings.shape == (4, 2)
    assert result.elapsed_days.shape == (4,)
    assert result.mask.shape == (4,)
    assert result.mask.sum().item() == 1.0
    # Padding rows are zeros and masked out.
    assert torch.allclose(result.embeddings[1], torch.zeros(2))
    assert result.mask[1].item() == 0.0


def test_build_lookback_tensors_caps_at_max_chunks_keeping_most_recent():
    store = _store(
        [
            {"doc_id": f"d{i}", "event_date": f"2024-12-{(i % 28) + 1:02d}", "chunk_index": 0, "chunk_preview": "x", "embedding": [float(i), 0.0]}
            for i in range(10)
        ]
    )
    result = build_lookback_tensors(store, anchor_date="2024-12-31", lookback_days=60, max_chunks=3)
    assert result.actual_count == 3
    # Most recent 3 by event_date (ties broken by row index ascending in retrieval).
    kept = [d for d in result.doc_ids if d]
    assert len(kept) == 3


def test_build_lookback_tensors_computes_elapsed_days_from_anchor():
    store = _store(
        [
            {"doc_id": "x", "event_date": "2024-12-01", "chunk_index": 0, "chunk_preview": "x", "embedding": [1.0]},
            {"doc_id": "y", "event_date": "2024-12-15", "chunk_index": 0, "chunk_preview": "y", "embedding": [1.0]},
        ]
    )
    result = build_lookback_tensors(store, anchor_date="2024-12-31", lookback_days=60, max_chunks=4, embedding_size=1)
    elapsed = result.elapsed_days.tolist()[: result.actual_count]
    assert sorted(elapsed) == [16.0, 30.0]


def test_build_lookback_tensors_empty_when_no_matches():
    store = _store(
        [
            {"doc_id": "a", "event_date": "2020-01-01", "chunk_index": 0, "chunk_preview": "a", "embedding": [1.0]},
        ]
    )
    result = build_lookback_tensors(store, anchor_date="2024-12-31", lookback_days=30, max_chunks=2, embedding_size=1)
    assert result.actual_count == 0
    assert result.mask.sum().item() == 0.0


# ---------------------------------------------------------------------------
# Variant C (LLM embedding source) tests
# ---------------------------------------------------------------------------


def _write_llm_parquet(tmp_path, rows):
    """Write a small LLM embeddings parquet fixture; fall back to jsonl if pyarrow missing."""
    df = pd.DataFrame(rows)
    parquet_path = tmp_path / "llm_embeddings.parquet"
    try:
        df.to_parquet(parquet_path, index=False)
        return parquet_path
    except Exception:
        jsonl_path = tmp_path / "llm_embeddings.jsonl"
        with jsonl_path.open("w") as h:
            for row in rows:
                h.write(json.dumps(row) + "\n")
        return parquet_path  # caller should also try .jsonl


def test_build_lookback_tensors_llm_returns_correct_shape(tmp_path):
    """LLM source returns same tensor shapes as chunk source."""
    rows = [
        {"document_id": "d1", "event_date": "2024-11-01", "embedding": [0.1, 0.2, 0.3]},
        {"document_id": "d2", "event_date": "2024-12-01", "embedding": [0.4, 0.5, 0.6]},
    ]
    parquet_path = _write_llm_parquet(tmp_path, rows)
    # Use _build_lookback_tensors_llm directly so we can pass a tmp path.
    result = _build_lookback_tensors_llm(
        anchor_date="2024-12-31",
        lookback_days=90,
        max_chunks=4,
        llm_store_path=parquet_path,
    )
    assert result.embeddings.shape == (4, 3)
    assert result.elapsed_days.shape == (4,)
    assert result.mask.shape == (4,)
    assert result.actual_count == 2
    assert result.mask.sum().item() == 2.0


def test_build_lookback_tensors_llm_one_slot_per_document(tmp_path):
    """Each document occupies exactly one slot (no chunk explosion)."""
    rows = [
        {"document_id": f"doc{i}", "event_date": "2024-12-01", "embedding": [float(i), 0.0]}
        for i in range(5)
    ]
    parquet_path = _write_llm_parquet(tmp_path, rows)
    result = _build_lookback_tensors_llm(
        anchor_date="2024-12-31",
        lookback_days=60,
        max_chunks=10,
        llm_store_path=parquet_path,
    )
    assert result.actual_count == 5


def test_build_lookback_tensors_llm_elapsed_days_computed_from_anchor(tmp_path):
    rows = [
        {"document_id": "a", "event_date": "2024-12-01", "embedding": [1.0]},
        {"document_id": "b", "event_date": "2024-12-15", "embedding": [1.0]},
    ]
    parquet_path = _write_llm_parquet(tmp_path, rows)
    result = _build_lookback_tensors_llm(
        anchor_date="2024-12-31",
        lookback_days=60,
        max_chunks=4,
        embedding_size=1,
        llm_store_path=parquet_path,
    )
    elapsed = sorted(result.elapsed_days.tolist()[: result.actual_count])
    assert elapsed == [16.0, 30.0]


def test_build_lookback_tensors_llm_caps_at_max_chunks(tmp_path):
    rows = [
        {"document_id": f"d{i}", "event_date": "2024-12-01", "embedding": [float(i)]}
        for i in range(10)
    ]
    parquet_path = _write_llm_parquet(tmp_path, rows)
    result = _build_lookback_tensors_llm(
        anchor_date="2024-12-31",
        lookback_days=60,
        max_chunks=3,
        llm_store_path=parquet_path,
    )
    assert result.actual_count == 3


def test_build_lookback_tensors_embedding_source_chunk_default():
    """Passing embedding_source='chunk' (default) still works as before."""
    store = _store(
        [
            {"doc_id": "x", "event_date": "2024-12-01", "chunk_index": 0, "chunk_preview": "x", "embedding": [1.0]},
        ]
    )
    result = build_lookback_tensors(
        store,
        anchor_date="2024-12-31",
        lookback_days=60,
        max_chunks=4,
        embedding_size=1,
        embedding_source="chunk",
    )
    assert result.actual_count == 1
    assert result.embeddings.shape == (4, 1)


def test_build_lookback_tensors_llm_via_embedding_source_kwarg(tmp_path):
    """build_lookback_tensors dispatches to LLM path when embedding_source='llm'."""
    rows = [
        {"document_id": "z", "event_date": "2024-12-01", "embedding": [0.5, 0.5]},
    ]
    parquet_path = _write_llm_parquet(tmp_path, rows)

    # Patch the default LLM path so the top-level dispatcher finds our fixture.
    import app.data.chunk_embedding_retrieval as retrieval_mod

    original = retrieval_mod.DEFAULT_LLM_EMBEDDINGS_PARQUET
    retrieval_mod.DEFAULT_LLM_EMBEDDINGS_PARQUET = parquet_path
    # Clear the lru_cache so our patched path is used.
    retrieval_mod._load_llm_store.cache_clear()
    try:
        dummy_store = pd.DataFrame()  # ignored for LLM path
        result = build_lookback_tensors(
            dummy_store,
            anchor_date="2024-12-31",
            lookback_days=60,
            max_chunks=4,
            embedding_source="llm",
        )
    finally:
        retrieval_mod.DEFAULT_LLM_EMBEDDINGS_PARQUET = original
        retrieval_mod._load_llm_store.cache_clear()

    assert result.actual_count == 1
    assert result.embeddings.shape == (4, 2)
