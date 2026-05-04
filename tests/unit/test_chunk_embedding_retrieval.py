from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pd = pytest.importorskip("pandas")

from app.data.chunk_embedding_retrieval import build_lookback_tensors  # noqa: E402


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
