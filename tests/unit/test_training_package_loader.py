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

    # One row per text_hash with a partition tag. Uses ``split_tag`` —
    # the column the production training-package builder actually emits.
    # The mix covers every partition the contract recognises: train
    # (only one that survives), val + test (forward-looking holdouts),
    # and the explicit excluded_from_training sentinel.
    split_rows = [
        {"text_hash": "hash_a", "split_tag": "train"},
        {"text_hash": "hash_b", "split_tag": "train"},
        {"text_hash": "hash_c", "split_tag": "val"},
        {"text_hash": "hash_excluded", "split_tag": "excluded_from_training"},
        {"text_hash": "hash_d", "split_tag": "test"},
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

    # 5 fixture events: 2 train + 1 val + 1 test + 1 excluded sentinel.
    # Only the two train rows feed the loss; val / test / excluded all
    # drop. Walk-forward leakage on val and test is the contract.
    assert len(sequences) == 2

    # Each sequence carries SEQUENCE_LENGTH prior bars + 1 event-day
    # target frame, so the downstream window slicer materialises one
    # supervised pair per event.
    for inner in sequences:
        assert len(inner) == SEQUENCE_LENGTH + 1

    # Sort contract: surviving events ordered by event_date ascending.
    event_dates = [seq[0].date[:10] for seq in sequences]
    target_dates = [seq[-1].date[:10] for seq in sequences]
    assert target_dates == sorted(target_dates)
    # The two surviving rows are the train-tagged hash_a (dovish,
    # 2024-02-01 target) and hash_b (hawkish, 2024-02-16 target).
    assert target_dates == ["2024-02-01", "2024-02-16"]
    # val / test / excluded targets must all be absent.
    for missing in ("2024-03-21", "2024-05-01", "2024-05-16"):
        assert missing not in target_dates
    # event_dates is the first bar's calendar date and is the same
    # across sequences because the prior windows share a synthetic
    # 2024-01 start; assert it for completeness.
    assert all(d == "2024-01-01" for d in event_dates)


def test_load_training_sequences_from_package_excludes_non_train_partitions(
    training_package_dir: Path,
) -> None:
    sequences = loaders.load_training_sequences_from_package(_TRAINING_PACKAGE_ID)
    # val / test / excluded_from_training targets must never appear on
    # the training side. The fixture maps each of those tags to a
    # specific event-day target date; none should survive.
    survivor_target_dates = {seq[-1].date[:10] for seq in sequences}
    for non_train_date in ("2024-03-21", "2024-05-01", "2024-05-16"):
        assert non_train_date not in survivor_target_dates


def test_load_training_sequences_from_package_encodes_axis_stance(
    training_package_dir: Path,
) -> None:
    sequences = loaders.load_training_sequences_from_package(_TRAINING_PACKAGE_ID)
    # Only the two train-tagged rows survive: hash_b (hawkish, target
    # 2024-02-16) and hash_a (dovish, target 2024-02-01). The other
    # stance encodings are validated against a separate fixture below.
    by_target_date = {seq[-1].date[:10]: seq for seq in sequences}
    hawkish = by_target_date["2024-02-16"]
    dovish = by_target_date["2024-02-01"]

    assert hawkish[0].sentiment_score == pytest.approx(1.0)
    assert dovish[0].sentiment_score == pytest.approx(-1.0)


def test_load_training_sequences_neutral_and_none_axis_stance(
    tmp_path: Path, monkeypatch
) -> None:
    """Neutral and None stance encodings rely on val / test rows being
    relabelled to train; the canonical fixture excludes them so the
    main test stays focused on walk-forward correctness."""

    import pandas as pd

    package_id = "tp_test_stance_v0"
    processed_root = tmp_path / "processed"
    package_dir = processed_root / package_id
    package_dir.mkdir(parents=True)
    monkeypatch.setattr(loaders, "DATA_DIR", tmp_path)

    events = [
        _event_row(
            event_date="2024-03-20",
            text_hash="hash_neutral",
            axis_stance="neutral",
            realized_return=0.0,
            realized_date="2024-03-21",
            base_close=4600.0,
        ),
        _event_row(
            event_date="2024-05-15",
            text_hash="hash_none",
            axis_stance=None,
            realized_return=0.002,
            realized_date="2024-05-16",
            base_close=4800.0,
        ),
    ]
    pd.DataFrame(events).to_parquet(package_dir / "events.parquet", index=False)
    splits = [
        {"text_hash": "hash_neutral", "split_tag": "train"},
        {"text_hash": "hash_none", "split_tag": "train"},
    ]
    pd.DataFrame(splits).to_parquet(
        package_dir / "splits_train_val_test.parquet", index=False
    )

    sequences = loaders.load_training_sequences_from_package(package_id)
    by_target_date = {seq[-1].date[:10]: seq for seq in sequences}
    assert by_target_date["2024-03-21"][0].sentiment_score == pytest.approx(0.0)
    assert by_target_date["2024-05-16"][0].sentiment_score == pytest.approx(0.0)


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


# ---------------------------------------------------------------------------
# Target-frame derivation tests
#
# The event-study target frame replaces the pre-fix realized-return /
# identity-copy target. These tests pin the close + volatility values
# produced by each ``target_mode`` against synthetic event rows whose
# ``abnormal_return`` and ``volatility_shift`` are deliberately
# distinguishable from ``realized_return`` and the prior vol_5d, so the
# two modes cannot accidentally produce the same numbers.
# ---------------------------------------------------------------------------


_TARGET_PACKAGE_ID = "tp_unit_target_modes_v0"


@pytest.fixture
def target_mode_package_dir(tmp_path: Path, monkeypatch) -> Path:
    """Minimal one-event package with distinct event-study / realized fields.

    ``abnormal_return`` (0.005) and ``realized_return`` (0.020) are
    chosen far apart so the close target lands at clearly different
    values under each mode. ``volatility_shift`` (-0.002) does the
    same for the volatility column against the last prior bar's
    vol_5d (0.012 + 4 * 0.0001 = 0.01240 with the synthesiser's
    five-bar window, but the actual last-bar value depends on
    SEQUENCE_LENGTH; the test reconstructs it from the same formula).
    """

    package_dir = tmp_path / "processed" / _TARGET_PACKAGE_ID
    package_dir.mkdir(parents=True)
    monkeypatch.setattr(loaders, "DATA_DIR", tmp_path)

    row = _event_row(
        event_date="2024-02-15",
        text_hash="hash_targets",
        axis_stance="hawkish",
        realized_return=0.020,
        realized_date="2024-02-16",
        base_close=4500.0,
    )
    # Override the event-study columns: the helper defaults to
    # ``abnormal_return == realized_return`` and ``volatility_shift =
    # 0.0`` so the two modes would coincide; the test needs them
    # distinct to lock the per-mode arithmetic.
    row["abnormal_return"] = 0.005
    row["volatility_shift"] = -0.002

    pd.DataFrame([row]).to_parquet(package_dir / "events.parquet", index=False)
    pd.DataFrame(
        [{"text_hash": "hash_targets", "split_tag": "train"}]
    ).to_parquet(package_dir / "splits_train_val_test.parquet", index=False)
    return package_dir


def _expected_last_prior_bar_close(base_close: float) -> float:
    return base_close + (SEQUENCE_LENGTH - 1) * 1.5


def _expected_last_prior_bar_vol() -> float:
    return 0.012 + (SEQUENCE_LENGTH - 1) * 0.0001


def test_target_uses_abnormal_return_in_event_study_mode(
    target_mode_package_dir: Path,
) -> None:
    sequences = loaders.load_training_sequences_from_package(
        _TARGET_PACKAGE_ID, target_mode="event_study"
    )
    assert len(sequences) == 1
    target = sequences[0][-1]

    last_close = _expected_last_prior_bar_close(4500.0)
    last_vol = _expected_last_prior_bar_vol()
    expected_close = last_close * (1.0 + 0.005)
    expected_volatility = last_vol + (-0.002)

    assert target.market_close == pytest.approx(expected_close)
    assert target.market_volatility == pytest.approx(expected_volatility)
    # Sanity: under realized_return the values would be different, so
    # the test really does discriminate the two modes.
    legacy_close = last_close * (1.0 + 0.020)
    assert target.market_close != pytest.approx(legacy_close)
    assert target.market_volatility != pytest.approx(last_vol)


def test_target_falls_back_when_abnormal_return_is_nan(
    tmp_path: Path, monkeypatch
) -> None:
    package_id = "tp_unit_target_nan_fallback_v0"
    package_dir = tmp_path / "processed" / package_id
    package_dir.mkdir(parents=True)
    monkeypatch.setattr(loaders, "DATA_DIR", tmp_path)

    row = _event_row(
        event_date="2024-02-15",
        text_hash="hash_nan",
        axis_stance="hawkish",
        realized_return=0.020,
        realized_date="2024-02-16",
        base_close=4500.0,
    )
    row["abnormal_return"] = float("nan")
    row["volatility_shift"] = float("nan")

    pd.DataFrame([row]).to_parquet(package_dir / "events.parquet", index=False)
    pd.DataFrame(
        [{"text_hash": "hash_nan", "split_tag": "train"}]
    ).to_parquet(package_dir / "splits_train_val_test.parquet", index=False)

    with pytest.warns(UserWarning):
        sequences = loaders.load_training_sequences_from_package(
            package_id, target_mode="event_study"
        )

    assert len(sequences) == 1
    target = sequences[0][-1]
    last_close = _expected_last_prior_bar_close(4500.0)
    last_vol = _expected_last_prior_bar_vol()
    # NaN abnormal_return falls back to realized_return; NaN
    # volatility_shift falls back to the identity copy of the last
    # prior bar's vol_5d.
    assert target.market_close == pytest.approx(last_close * (1.0 + 0.020))
    assert target.market_volatility == pytest.approx(last_vol)


def test_target_mode_realized_return_reproduces_legacy_behaviour(
    target_mode_package_dir: Path,
) -> None:
    sequences = loaders.load_training_sequences_from_package(
        _TARGET_PACKAGE_ID, target_mode="realized_return"
    )
    assert len(sequences) == 1
    target = sequences[0][-1]
    last_close = _expected_last_prior_bar_close(4500.0)
    last_vol = _expected_last_prior_bar_vol()
    # Legacy formula: close * (1 + realized_return) and identity copy
    # of the last prior bar's vol_5d. Locks the back-compat path.
    assert target.market_close == pytest.approx(last_close * (1.0 + 0.020))
    assert target.market_volatility == pytest.approx(last_vol)


def test_target_mode_invalid_raises(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(loaders, "DATA_DIR", tmp_path)
    with pytest.raises(ValueError):
        loaders.load_training_sequences_from_package(
            "any_id", target_mode="not_a_mode"  # type: ignore[arg-type]
        )
