"""Tests for the Phase 8 training-package forecaster loader.

The loader reads ``events.parquet`` (per-event prior-bar windows) and
``splits_train_val_test.parquet`` (per-text-hash partition tags) from a
Phase 8 training package and returns sequence groups ready for the
``train_model`` consumer. These tests synthesise a tiny package
fixture on disk, drive the loader through it, and assert the
filtering / ordering / shape contracts documented in the loader
docstring.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")

from app.models.config import SEQUENCE_LENGTH
from app.training import loaders


_TRAINING_PACKAGE_ID = "tp_unit_test_loader_v1.0"


def _synth_prior_bars(*, base_close: float, base_vol: float, start_day: int) -> str:
    """Emit a 20-bar JSON window with strictly-increasing dates / closes."""

    payload = []
    for offset in range(SEQUENCE_LENGTH):
        day = start_day + offset
        payload.append(
            {
                "date": f"2024-01-{day:02d}",
                "close": round(base_close + offset * 1.5, 10),
                "volume": 1_000_000.0,
                "vol_5d": round(base_vol + offset * 0.0001, 10),
                "cum_return_20d": round(offset * 0.001, 10),
            }
        )
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


def _event_row(
    *,
    event_date: str,
    text_hash: str,
    axis_stance: str | None,
    realized_return: float,
    realized_date: str,
    base_close: float,
    horizon: int = 1,
) -> dict:
    return {
        "event_date": event_date,
        "event_kind": "statement",
        "document_id": text_hash[:16],
        "text_hash": text_hash,
        "source": "scraped_fed",
        "source_record_id": f"src:{text_hash[:8]}",
        "as_of_ts": f"{event_date}T19:00:00Z",
        "text": "FOMC body",
        "token_count": 2,
        "axis_stance": axis_stance,
        "axis_time": None,
        "axis_certainty": None,
        "axis_factor": None,
        "axis_topic": None,
        "credibility_drift_score": 0.0,
        "credibility_realized_vs_stated_gap": 0.0,
        "credibility_market_implied_gap": 0.0,
        "credibility_months_since_reversal": 0,
        "prior_window_sha256": "0" * 64,
        "prior_bars_json": _synth_prior_bars(
            base_close=base_close, base_vol=0.012, start_day=1
        ),
        "asset_symbol": "^GSPC",
        "horizon": int(horizon),
        "realized_return": float(realized_return),
        "abnormal_return": float(realized_return),
        "alpha": 0.0,
        "beta": 1.0,
        "direction_t1d": 1 if realized_return > 0 else (-1 if realized_return < 0 else 0),
        "volatility_shift": 0.0,
        "concurrent_macro_release": False,
        "intra_meeting_stance_shift": 0.0,
        "intra_meeting_certainty_shift": 0.0,
        "intra_meeting_factor_shift": 0.0,
        "realized_date": realized_date,
    }


@pytest.fixture
def training_package_dir(tmp_path: Path, monkeypatch) -> Path:
    """Materialise a tiny five-event training package under ``tmp_path``."""

    processed_root = tmp_path / "processed"
    package_dir = processed_root / _TRAINING_PACKAGE_ID
    package_dir.mkdir(parents=True)

    # Point the loader's DATA_DIR at tmp_path so ``<DATA_DIR>/processed/<id>``
    # resolves to the synthetic package above.
    monkeypatch.setattr(loaders, "DATA_DIR", tmp_path)

    events = [
        _event_row(
            event_date="2024-02-15",
            text_hash="hash_b",
            axis_stance="hawkish",
            realized_return=0.012,
            realized_date="2024-02-16",
            base_close=4500.0,
        ),
        _event_row(
            event_date="2024-01-31",
            text_hash="hash_a",
            axis_stance="dovish",
            realized_return=-0.008,
            realized_date="2024-02-01",
            base_close=4400.0,
        ),
        _event_row(
            event_date="2024-03-20",
            text_hash="hash_c",
            axis_stance="neutral",
            realized_return=0.0,
            realized_date="2024-03-21",
            base_close=4600.0,
        ),
        _event_row(
            event_date="2024-04-30",
            text_hash="hash_excluded",
            axis_stance="hawkish",
            realized_return=0.005,
            realized_date="2024-05-01",
            base_close=4700.0,
        ),
        _event_row(
            event_date="2024-05-15",
            text_hash="hash_d",
            axis_stance=None,
            realized_return=0.002,
            realized_date="2024-05-16",
            base_close=4800.0,
        ),
    ]
    events_frame = pd.DataFrame(events)
    events_frame.to_parquet(package_dir / "events.parquet", index=False)

    # One row per text_hash with a partition tag. One row is tagged
    # ``excluded_from_training`` so the filter contract is exercised.
    split_rows = [
        {"text_hash": "hash_a", "partition": "train"},
        {"text_hash": "hash_b", "partition": "train"},
        {"text_hash": "hash_c", "partition": "val"},
        {"text_hash": "hash_excluded", "partition": "excluded_from_training"},
        {"text_hash": "hash_d", "partition": "test"},
    ]
    splits_frame = pd.DataFrame(split_rows)
    splits_frame.to_parquet(
        package_dir / "splits_train_val_test.parquet", index=False
    )
    return package_dir


def test_load_training_sequences_from_package_filters_and_orders(
    training_package_dir: Path,
) -> None:
    sequences = loaders.load_training_sequences_from_package(_TRAINING_PACKAGE_ID)

    # 5 fixture events, 1 excluded => 4 surviving sequences.
    assert len(sequences) == 4

    # Each sequence carries SEQUENCE_LENGTH prior bars + 1 event-day
    # target frame, so the downstream window slicer materialises one
    # supervised pair per event.
    for inner in sequences:
        assert len(inner) == SEQUENCE_LENGTH + 1

    # Sort contract: surviving events ordered by event_date ascending.
    event_dates = [seq[0].date[:10] for seq in sequences]
    # The first bar in each sequence is the earliest prior bar (2024-01-01);
    # all four sequences therefore start on the same date. Verify ordering
    # via the appended target frame's date instead, which is the
    # event-specific ``realized_date``.
    target_dates = [seq[-1].date[:10] for seq in sequences]
    assert target_dates == sorted(target_dates)
    # Sanity-check that the excluded row's target date is absent.
    assert "2024-05-01" not in target_dates
    # Sanity-check that the surviving target dates match the four
    # non-excluded fixture events.
    assert target_dates == ["2024-02-01", "2024-02-16", "2024-03-21", "2024-05-16"]
    # event_dates is the first bar's calendar date and is the same
    # across sequences because the prior windows share a synthetic
    # 2024-01 start; assert it for completeness.
    assert all(d == "2024-01-01" for d in event_dates)


def test_load_training_sequences_from_package_excludes_partition(
    training_package_dir: Path,
) -> None:
    sequences = loaders.load_training_sequences_from_package(_TRAINING_PACKAGE_ID)
    # The excluded event's realized_date is 2024-05-01; no surviving
    # sequence should land on that date.
    for inner in sequences:
        assert inner[-1].date[:10] != "2024-05-01"


def test_load_training_sequences_from_package_encodes_axis_stance(
    training_package_dir: Path,
) -> None:
    sequences = loaders.load_training_sequences_from_package(_TRAINING_PACKAGE_ID)
    # Sequences are sorted by realized_date / event_date; the second
    # surviving sequence is the 2024-02-15 hawkish event.
    by_target_date = {seq[-1].date[:10]: seq for seq in sequences}
    hawkish = by_target_date["2024-02-16"]
    dovish = by_target_date["2024-02-01"]
    neutral = by_target_date["2024-03-21"]
    none_stance = by_target_date["2024-05-16"]

    assert hawkish[0].sentiment_score == pytest.approx(1.0)
    assert dovish[0].sentiment_score == pytest.approx(-1.0)
    assert neutral[0].sentiment_score == pytest.approx(0.0)
    assert none_stance[0].sentiment_score == pytest.approx(0.0)


def test_load_training_sequences_from_package_appends_target_close(
    training_package_dir: Path,
) -> None:
    sequences = loaders.load_training_sequences_from_package(_TRAINING_PACKAGE_ID)
    # For the 2024-02-15 event (base_close=4500, last prior bar close
    # = 4500 + 19 * 1.5 = 4528.5, realized_return = 0.012), the
    # appended target close should be 4528.5 * 1.012 = 4582.722.
    by_target_date = {seq[-1].date[:10]: seq for seq in sequences}
    hawkish = by_target_date["2024-02-16"]
    last_prior_close = 4500.0 + 19 * 1.5
    expected_target_close = last_prior_close * (1.0 + 0.012)
    assert hawkish[-1].market_close == pytest.approx(expected_target_close)


def test_load_training_sequences_from_package_missing_package_raises(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(loaders, "DATA_DIR", tmp_path)
    with pytest.raises(FileNotFoundError):
        loaders.load_training_sequences_from_package("does_not_exist")
