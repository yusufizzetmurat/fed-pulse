"""Verify ``/analyze`` surfaces a populated credibility block.

Before this change the response schema declared a ``credibility``
field but ``_build_analyze_response`` never assigned to it, so the
workspace rendered the "credibility signals unavailable" empty state
for every passage. These tests pin the new behaviour:

- a real loader response (embedding parquet present, FRED cache
  present) lands in the ``credibility`` slot with axes populated;
- a missing FRED / embedding pair falls back to None per axis so the
  frontend renders N/A without 5xx;
- the drift trend sparkline carries one entry per usable prior.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("torch")

import app.main as main_mod  # noqa: E402
from app.schemas import AnalyzeRequest  # noqa: E402


def _write_embeddings(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def _make_payload() -> AnalyzeRequest:
    return AnalyzeRequest(
        text="Statement text about the federal funds rate.",
        date="2024-05-01",
        symbol="^GSPC",
        horizon="3d",
    )


def test_build_credibility_block_returns_none_axes_when_no_inputs(monkeypatch, tmp_path):
    """No embedding cache, no FRED cache -> realized/months_since are None."""

    monkeypatch.setattr(main_mod, "DATA_DIR", tmp_path)
    import app.services.fred_client as fred_mod

    monkeypatch.setattr(fred_mod, "DEFAULT_CACHE_DIR", tmp_path / "fred_missing")
    import app.data.embedding_cache as embedding_cache_mod

    monkeypatch.setattr(
        embedding_cache_mod, "DEFAULT_CACHE_DIR", tmp_path / "raw" / "embeddings"
    )

    payload = _make_payload()
    block = main_mod._build_credibility_block(
        payload, sentiment={"label": "neutral", "score": 0.0}
    )
    assert block is not None
    assert block["drift_score"] == 0.0
    assert block["realized_vs_stated_gap"] is None
    assert block["market_implied_gap"] is None
    # months_since stays None because only the current run is in
    # ``stance_by_date`` (history table is empty under the in-memory
    # session this test boots against).
    assert block["months_since_reversal"] is None
    assert block["drift_trend"] == []


def test_build_credibility_block_empty_fred_dir_leaves_realized_none(monkeypatch, tmp_path):
    """An existing-but-empty FRED cache directory (no ``DFF.json``) must
    still suppress the realized-vs-stated axis.

    ``fetch_fred_series`` creates ``data/external/fred/`` on the first
    call even without an API key, so the previous directory-existence
    guard let a degenerate 0.0 reading leak through on subsequent runs.
    """

    monkeypatch.setattr(main_mod, "DATA_DIR", tmp_path)
    import app.services.fred_client as fred_mod

    empty_fred_dir = tmp_path / "fred"
    empty_fred_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(fred_mod, "DEFAULT_CACHE_DIR", empty_fred_dir)
    import app.data.embedding_cache as embedding_cache_mod

    monkeypatch.setattr(
        embedding_cache_mod, "DEFAULT_CACHE_DIR", tmp_path / "raw" / "embeddings"
    )

    payload = _make_payload()
    block = main_mod._build_credibility_block(
        payload, sentiment={"label": "neutral", "score": 0.0}
    )
    assert block is not None
    assert block["realized_vs_stated_gap"] is None


def test_build_credibility_block_populates_from_embedding_parquet(monkeypatch, tmp_path):
    """A real parquet under the canonical encoder slug must drive the
    drift score above zero and seed a non-empty drift_trend."""

    from app.models.registry import encoder_ref

    encoder_alias = "finbert_fed_adjacent_xbank_dapt"
    ref = encoder_ref(encoder_alias)
    assert ref is not None and ref.revision, "registry must pin the canonical encoder"

    from app.data.embedding_cache import resolve_cache_paths

    # Pivot the embedding cache root onto tmp_path so the resolver
    # produces a parquet path we control.
    monkeypatch.setattr(main_mod, "DATA_DIR", tmp_path)
    import app.data.embedding_cache as embedding_cache_mod

    monkeypatch.setattr(
        embedding_cache_mod, "DEFAULT_CACHE_DIR", tmp_path / "raw" / "embeddings"
    )

    paths = resolve_cache_paths(encoder_alias, revision=ref.revision)
    _write_embeddings(
        paths.parquet,
        [
            {"event_date": "2023-09-01", "embedding": [1.0, 0.0, 0.0]},
            {"event_date": "2023-11-01", "embedding": [0.95, 0.05, 0.0]},
            {"event_date": "2024-01-01", "embedding": [0.9, 0.1, 0.0]},
            {"event_date": "2024-03-01", "embedding": [0.85, 0.15, 0.0]},
            {"event_date": "2024-04-15", "embedding": [0.0, 1.0, 0.0]},  # orthogonal pivot
        ],
    )

    payload = _make_payload()
    block = main_mod._build_credibility_block(
        payload, sentiment={"label": "neutral", "score": 0.0}
    )
    assert block is not None
    assert block["drift_score"] > 0.5, (
        f"expected a large drift after the orthogonal pivot; got {block['drift_score']}"
    )
    assert len(block["drift_trend"]) >= 2
