"""Unit tests for app.services.credibility_loader."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from app.features.credibility import CredibilityVector
from app.services.credibility_loader import (
    CredibilityInputs,
    assemble_inputs,
    load_credibility_for_run,
)
from app.services.fred_client import FredObservation, FredSeriesResponse


def _write_embeddings(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def _fred_response(observations: list[tuple[str, float]]) -> FredSeriesResponse:
    return FredSeriesResponse(
        series_id="DFF",
        realtime_start="2024-01-01",
        realtime_end="2024-12-31",
        observation_start="2024-01-01",
        observation_end="2024-12-31",
        count=len(observations),
        observations=[
            FredObservation(date=d, value=v, realtime_start="2024-01-01", realtime_end="2024-12-31")
            for d, v in observations
        ],
    )


def test_assemble_inputs_collects_drift_inputs_from_embedding_cache(tmp_path: Path) -> None:
    path = tmp_path / "embeddings.parquet"
    _write_embeddings(
        path,
        [
            {"event_date": "2023-09-01", "embedding": [1.0, 0.0, 0.0]},
            {"event_date": "2023-11-01", "embedding": [0.9, 0.1, 0.0]},
            {"event_date": "2024-01-01", "embedding": [0.8, 0.2, 0.0]},
            {"event_date": "2024-03-01", "embedding": [0.7, 0.3, 0.0]},
            {"event_date": "2024-05-01", "embedding": [0.0, 1.0, 0.0]},
        ],
    )
    inputs = assemble_inputs(
        as_of_ts="2024-05-01",
        embedding_path=path,
        drift_window=4,
    )
    assert isinstance(inputs, CredibilityInputs)
    assert inputs.current_embedding == [0.0, 1.0, 0.0]
    assert len(inputs.prior_embeddings) == 4
    assert inputs.prior_embeddings[0] == [1.0, 0.0, 0.0]
    assert inputs.prior_embeddings[-1] == [0.7, 0.3, 0.0]


def test_assemble_inputs_handles_missing_embedding_path(tmp_path: Path) -> None:
    inputs = assemble_inputs(
        as_of_ts="2024-05-01",
        embedding_path=tmp_path / "does_not_exist.parquet",
    )
    assert inputs.current_embedding == []
    assert inputs.prior_embeddings == []


def test_assemble_inputs_filters_stance_history_to_on_or_before(tmp_path: Path) -> None:
    inputs = assemble_inputs(
        as_of_ts="2024-05-01",
        embedding_path=None,
        stance_by_date=[
            ("2024-01-01", 0.5),
            ("2024-03-01", -0.3),
            ("2024-05-15", 0.8),  # post-as-of — must be excluded
        ],
    )
    assert inputs.stance_history == [0.5, -0.3]


def test_assemble_inputs_filters_fred_series_to_trailing_window(tmp_path: Path) -> None:
    response = _fred_response(
        [
            ("2024-01-01", 4.0),
            ("2024-02-01", 4.5),
            ("2024-03-01", 5.0),
            ("2024-04-01", 5.25),
            ("2024-05-01", 5.5),
            ("2024-06-01", 5.5),  # post-as-of — excluded
        ]
    )
    inputs = assemble_inputs(
        as_of_ts="2024-05-01",
        embedding_path=None,
        stance_by_date=[
            ("2024-02-01", 0.3),
            ("2024-04-01", -0.1),
        ],
        fred_response=response,
        realized_window_days=200,
    )
    assert inputs.realized_path == [4.0, 4.5, 5.0, 5.25, 5.5]
    # stated_path is clipped to the matched horizon from stance_history
    assert inputs.stated_path == [0.3, -0.1]


def test_load_credibility_for_run_returns_vector_with_neutral_defaults(tmp_path: Path) -> None:
    vector = load_credibility_for_run(
        as_of_ts="2024-05-01",
        embedding_path=None,
        stance_by_date=[],
        fred_response=None,
    )
    assert isinstance(vector, CredibilityVector)
    # All axes degrade to 0 / 0 when no inputs are available.
    assert vector.drift_score == 0.0
    assert vector.realized_vs_stated_gap == 0.0
    assert vector.market_implied_gap == 0.0
    assert vector.months_since_reversal == 0


def test_load_credibility_for_run_invokes_drift_when_embeddings_present(tmp_path: Path) -> None:
    path = tmp_path / "embeddings.parquet"
    _write_embeddings(
        path,
        [
            {"event_date": "2024-01-01", "embedding": [1.0, 0.0]},
            {"event_date": "2024-03-01", "embedding": [1.0, 0.0]},
            {"event_date": "2024-05-01", "embedding": [0.0, 1.0]},  # orthogonal to priors
        ],
    )
    vector = load_credibility_for_run(
        as_of_ts="2024-05-01",
        embedding_path=path,
    )
    assert vector.drift_score == pytest.approx(1.0, abs=1e-6)


def test_invalid_as_of_ts_raises() -> None:
    with pytest.raises(ValueError):
        assemble_inputs(as_of_ts="not-a-date", embedding_path=None)
