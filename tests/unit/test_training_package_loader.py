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

from app.models.config import (
    FEATURE_SIZE,
    RICH_CREDIBILITY_DIM,
    RICH_CREDIBILITY_SLICE,
    RICH_FEATURE_SIZE,
    RICH_LINGUISTIC_DIM,
    RICH_LINGUISTIC_SLICE,
    RICH_MP_SURPRISE_DIM,
    RICH_MP_SURPRISE_SLICE,
    RICH_MULTI_AXIS_DIM,
    RICH_MULTI_AXIS_SLICE,
    SEQUENCE_LENGTH,
)
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


# ---------------------------------------------------------------------------
# Rich-feature loader tests (PR #173)
#
# The training-package loader joins linguistic_features.parquet on
# text_hash, mp_surprises.parquet on event_date, reads the credibility +
# multi-axis fields straight off the events parquet, and broadcasts the
# event-level signal onto every bar in the 20-day prior window plus the
# appended event-day target frame. Per-bar feature size grows from
# FEATURE_SIZE (6) to RICH_FEATURE_SIZE (35) and is emitted through
# ``FeatureVector.as_rich_list``.
# ---------------------------------------------------------------------------


_RICH_PACKAGE_ID = "tp_unit_rich_features_v0"


def _linguistic_row(text_hash: str, base: float) -> dict:
    """Build a synthetic linguistic-feature row with the full 15 columns.

    Values are chosen to be unique per-text_hash so the broadcaster's
    join-on-text_hash contract is observable: each surviving sequence
    should carry exactly the row keyed by its event's text_hash.
    """

    return {
        "text_hash": text_hash,
        "topic_share_inflation": base + 0.01,
        "topic_share_employment": base + 0.02,
        "topic_share_financial_stability": base + 0.03,
        "topic_share_growth": base + 0.04,
        "topic_share_balance_sheet": base + 0.05,
        "topic_share_misc_1": base + 0.06,
        "topic_share_misc_2": base + 0.07,
        "topic_share_misc_3": base + 0.08,
        "hedge_density": base + 0.10,
        "comparison_density": base + 0.11,
        "forward_density": base + 0.12,
        "concrete_ratio": base + 0.13,
        "hawk_dove_asymmetry": base + 0.14,
        "log_token_count": base + 0.15,
        "pivot_distance": base + 0.16,
    }


def _mp_surprise_row(event_date: str, base: float) -> dict:
    """Build a synthetic mp_surprises row keyed by ``event_date``."""

    return {
        "event_date": event_date,
        "mp_surprise_level": base + 0.001,
        "mp_surprise_path_factor": base + 0.002,
        "fed_info_factor": base + 0.003,
        "is_intermeeting": bool(base > 0.5),
    }


@pytest.fixture
def rich_feature_package_dir(tmp_path: Path, monkeypatch) -> Path:
    """Materialise a small package with linguistic + mp-surprise parquets.

    Two train-tagged events with known credibility / multi-axis /
    linguistic / mp-surprise values let the rich-feature tests pin
    each slice against an arithmetically reconstructible expectation.
    A third event drops its linguistic row so the missing-join
    fallback test has a candidate to validate against.
    """

    package_dir = tmp_path / "processed" / _RICH_PACKAGE_ID
    package_dir.mkdir(parents=True)
    monkeypatch.setattr(loaders, "DATA_DIR", tmp_path)

    events = []
    for text_hash, event_date, realized_date, base_close, base, factor, certainty in (
        ("hash_a", "2024-01-31", "2024-02-01", 4400.0, 0.10, 0.25, 0.5),
        ("hash_b", "2024-02-15", "2024-02-16", 4500.0, 0.20, -0.40, 0.8),
        ("hash_no_ling", "2024-03-15", "2024-03-16", 4600.0, 0.30, 0.10, 0.2),
    ):
        row = _event_row(
            event_date=event_date,
            text_hash=text_hash,
            axis_stance="hawkish",
            realized_return=0.001,
            realized_date=realized_date,
            base_close=base_close,
        )
        # Pin credibility + multi-axis to distinguishable, non-zero
        # values so the per-bar slice assertions are observable.
        row["credibility_drift_score"] = base + 0.001
        row["credibility_realized_vs_stated_gap"] = base + 0.002
        row["credibility_market_implied_gap"] = base + 0.003
        row["credibility_months_since_reversal"] = int(base * 10)
        row["axis_factor"] = factor
        row["axis_certainty"] = certainty
        row["axis_time"] = base + 0.5
        events.append(row)
    pd.DataFrame(events).to_parquet(package_dir / "events.parquet", index=False)
    pd.DataFrame(
        [
            {"text_hash": "hash_a", "split_tag": "train"},
            {"text_hash": "hash_b", "split_tag": "train"},
            {"text_hash": "hash_no_ling", "split_tag": "train"},
        ]
    ).to_parquet(package_dir / "splits_train_val_test.parquet", index=False)

    # Linguistic parquet covers hash_a and hash_b but NOT hash_no_ling,
    # so the join-fallback test can assert that the third event's
    # linguistic slice is all zeros.
    pd.DataFrame(
        [
            _linguistic_row("hash_a", 0.10),
            _linguistic_row("hash_b", 0.20),
        ]
    ).to_parquet(package_dir / "linguistic_features.parquet", index=False)

    # MP-surprise parquet keyed on event_date. Covers all three events.
    pd.DataFrame(
        [
            _mp_surprise_row("2024-01-31", 0.10),
            _mp_surprise_row("2024-02-15", 0.20),
            _mp_surprise_row("2024-03-15", 0.30),
        ]
    ).to_parquet(package_dir / "mp_surprises.parquet", index=False)

    return package_dir


def test_rich_features_emit_35_per_bar(rich_feature_package_dir: Path) -> None:
    sequences = loaders.load_training_sequences_from_package(
        _RICH_PACKAGE_ID, rich_features=True
    )
    assert len(sequences) == 3
    # Per-bar size on the rich emitter is the documented constant; the
    # legacy emitter stays at FEATURE_SIZE for any non-rich vector.
    for sequence in sequences:
        for vector in sequence:
            assert vector.rich_payload is True
            assert len(vector.as_rich_list()) == RICH_FEATURE_SIZE
            # Back-compat: the 6-dim slice is still emitted.
            assert len(vector.as_list()) == FEATURE_SIZE
            # Positions [0:6] of as_rich_list match as_list bit-for-bit
            # so models built on the legacy slice keep seeing the same
            # values when widened to RICH_FEATURE_SIZE.
            assert vector.as_rich_list()[:FEATURE_SIZE] == vector.as_list()

    # Pin one full slice composition against the synthetic fixture so
    # the per-family broadcast and the column-ordering contract are
    # locked together. hash_a: base = 0.10, factor = 0.25, certainty
    # = 0.5, axis_time = 0.60.
    by_target_date = {seq[-1].date[:10]: seq for seq in sequences}
    hash_a_target = by_target_date["2024-02-01"][-1]
    rich = hash_a_target.as_rich_list()
    cred_slice = rich[RICH_CREDIBILITY_SLICE]
    assert cred_slice == pytest.approx([0.101, 0.102, 0.103, 1.0])
    ling_slice = rich[RICH_LINGUISTIC_SLICE]
    expected_ling = [
        0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18,
        0.20, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26,
    ]
    assert ling_slice == pytest.approx(expected_ling)
    mp_slice = rich[RICH_MP_SURPRISE_SLICE]
    # hash_a has base=0.10 in the mp parquet (is_intermeeting=False).
    assert mp_slice == pytest.approx([0.101, 0.102, 0.103, 0.0])
    multi_axis_slice = rich[RICH_MULTI_AXIS_SLICE]
    # All three axes present -> missing flags zero.
    assert multi_axis_slice == pytest.approx([0.25, 0.0, 0.5, 0.0, 0.6, 0.0])


def test_rich_features_missing_linguistic_row_zeros_and_flags(
    rich_feature_package_dir: Path,
) -> None:
    sequences = loaders.load_training_sequences_from_package(
        _RICH_PACKAGE_ID, rich_features=True
    )
    # hash_no_ling has no linguistic-parquet row -- its linguistic
    # slice must be all zeros on every bar of the sequence. The
    # ablation flag is *not* set (the family is still on); the row's
    # absence is what zeros the slice.
    by_target_date = {seq[-1].date[:10]: seq for seq in sequences}
    no_ling_sequence = by_target_date["2024-03-16"]
    for vector in no_ling_sequence:
        ling_slice = vector.as_rich_list()[RICH_LINGUISTIC_SLICE]
        assert ling_slice == [0.0] * RICH_LINGUISTIC_DIM

    # Sanity: the credibility + mp-surprise + multi-axis slices are
    # still populated (only the missing family zeros out), so the
    # broadcaster did not also drop unrelated families.
    target = no_ling_sequence[-1].as_rich_list()
    assert target[RICH_CREDIBILITY_SLICE][0] == pytest.approx(0.301)
    assert target[RICH_MP_SURPRISE_SLICE][0] == pytest.approx(0.301)
    assert target[RICH_MULTI_AXIS_SLICE][2] == pytest.approx(0.2)


def test_rich_features_per_family_ablation_zeros_correct_slice(
    rich_feature_package_dir: Path,
) -> None:
    sequences = loaders.load_training_sequences_from_package(
        _RICH_PACKAGE_ID,
        rich_features=True,
        use_credibility=False,
    )
    assert len(sequences) == 3
    # Credibility slice on every bar must be zero. The per-bar
    # feature size stays at RICH_FEATURE_SIZE so the model input
    # shape is unchanged -- the ablation measures lift by zeroing
    # the slice rather than shrinking the input.
    for sequence in sequences:
        for vector in sequence:
            rich = vector.as_rich_list()
            assert len(rich) == RICH_FEATURE_SIZE
            assert rich[RICH_CREDIBILITY_SLICE] == [0.0] * RICH_CREDIBILITY_DIM

    # Other families stay populated: pick hash_a and verify the
    # linguistic + mp-surprise + multi-axis slices match the
    # all-on baseline above.
    by_target_date = {seq[-1].date[:10]: seq for seq in sequences}
    hash_a_target = by_target_date["2024-02-01"][-1].as_rich_list()
    assert hash_a_target[RICH_LINGUISTIC_SLICE][0] == pytest.approx(0.11)
    assert hash_a_target[RICH_MP_SURPRISE_SLICE][0] == pytest.approx(0.101)
    assert hash_a_target[RICH_MULTI_AXIS_SLICE][0] == pytest.approx(0.25)


def test_rich_features_multi_axis_missing_flips_missing_flag(
    tmp_path: Path, monkeypatch
) -> None:
    """NaN multi-axis fields collapse to zero and flip the missing flag."""

    package_id = "tp_unit_multi_axis_missing_v0"
    package_dir = tmp_path / "processed" / package_id
    package_dir.mkdir(parents=True)
    monkeypatch.setattr(loaders, "DATA_DIR", tmp_path)

    row = _event_row(
        event_date="2024-04-30",
        text_hash="hash_nan_axes",
        axis_stance="neutral",
        realized_return=0.0,
        realized_date="2024-05-01",
        base_close=4700.0,
    )
    # axis_factor / axis_certainty / axis_time stay None (default
    # from ``_event_row``); the loader must collapse them to 0.0 and
    # flip each *_missing flag to 1.0.
    pd.DataFrame([row]).to_parquet(package_dir / "events.parquet", index=False)
    pd.DataFrame(
        [{"text_hash": "hash_nan_axes", "split_tag": "train"}]
    ).to_parquet(package_dir / "splits_train_val_test.parquet", index=False)

    sequences = loaders.load_training_sequences_from_package(
        package_id, rich_features=True
    )
    assert len(sequences) == 1
    target = sequences[0][-1].as_rich_list()
    multi_axis = target[RICH_MULTI_AXIS_SLICE]
    # axis_factor, axis_factor_missing, axis_certainty,
    # axis_certainty_missing, axis_time, axis_time_missing
    assert multi_axis == pytest.approx([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])


def test_no_rich_features_reproduces_legacy_6dim_path(
    rich_feature_package_dir: Path,
) -> None:
    """``rich_features=False`` is byte-identical to the pre-PR-#173 output."""

    legacy = loaders.load_training_sequences_from_package(
        _RICH_PACKAGE_ID, rich_features=False
    )
    assert len(legacy) == 3
    for sequence in legacy:
        for vector in sequence:
            # Rich payload flag stays at the dataclass default.
            assert vector.rich_payload is False
            assert len(vector.as_list()) == FEATURE_SIZE
            # ``as_rich_list`` on a non-rich vector falls back to
            # ``as_list`` plus zero-padding, so its 6-dim prefix is
            # unchanged. The remaining 29 dims are zeros plus the
            # default multi-axis missing flags (1.0 each).
            rich = vector.as_rich_list()
            assert len(rich) == RICH_FEATURE_SIZE
            assert rich[:FEATURE_SIZE] == vector.as_list()
            assert rich[RICH_CREDIBILITY_SLICE] == [0.0] * RICH_CREDIBILITY_DIM
            assert rich[RICH_LINGUISTIC_SLICE] == [0.0] * RICH_LINGUISTIC_DIM
            assert rich[RICH_MP_SURPRISE_SLICE] == [0.0] * RICH_MP_SURPRISE_DIM


def test_rich_features_ignored_when_mp_parquet_missing(
    tmp_path: Path, monkeypatch
) -> None:
    """No mp_surprises.parquet -> mp-surprise slice is zero, other
    families still flow. Confirms the loader degrades cleanly when one
    of the optional side-tables is absent."""

    package_id = "tp_unit_no_mp_v0"
    package_dir = tmp_path / "processed" / package_id
    package_dir.mkdir(parents=True)
    monkeypatch.setattr(loaders, "DATA_DIR", tmp_path)

    row = _event_row(
        event_date="2024-06-15",
        text_hash="hash_no_mp",
        axis_stance="hawkish",
        realized_return=0.001,
        realized_date="2024-06-16",
        base_close=4800.0,
    )
    row["credibility_drift_score"] = 0.5
    pd.DataFrame([row]).to_parquet(package_dir / "events.parquet", index=False)
    pd.DataFrame(
        [{"text_hash": "hash_no_mp", "split_tag": "train"}]
    ).to_parquet(package_dir / "splits_train_val_test.parquet", index=False)
    # NO linguistic_features.parquet, NO mp_surprises.parquet.

    sequences = loaders.load_training_sequences_from_package(
        package_id, rich_features=True
    )
    assert len(sequences) == 1
    target = sequences[0][-1].as_rich_list()
    # Credibility still populated from the events row.
    assert target[RICH_CREDIBILITY_SLICE][0] == pytest.approx(0.5)
    # Linguistic + mp-surprise slices zero (parquet missing).
    assert target[RICH_LINGUISTIC_SLICE] == [0.0] * RICH_LINGUISTIC_DIM
    assert target[RICH_MP_SURPRISE_SLICE] == [0.0] * RICH_MP_SURPRISE_DIM


def test_rebuilt_assets_drive_nonzero_mp_surprise_and_pivot_distance(
    rich_feature_package_dir: Path,
) -> None:
    """End-to-end contract: post-rebuild parquets put nonzero values in
    both the MP-surprise slice and the ``pivot_distance`` position of
    the linguistic slice.

    The rich_feature_package_dir fixture mirrors the shape produced by
    the post-PR-#173 emitters (``mp_surprises.parquet`` keyed on
    ``event_date`` and ``linguistic_features.parquet`` with the
    15-column schema including ``pivot_distance``). This test confirms
    that the rebuilt artefacts feed the rich-feature slice without
    falling back to the all-zeros path that the pre-PR-#173 loader hit.
    """

    sequences = loaders.load_training_sequences_from_package(
        _RICH_PACKAGE_ID, rich_features=True
    )
    assert len(sequences) == 3

    # ``pivot_distance`` sits at the trailing slot of the linguistic
    # 15-vector (column index 14 within ``_LINGUISTIC_FEATURE_COLUMNS``).
    pivot_distance_offset = RICH_LINGUISTIC_DIM - 1
    pivot_distance_global = RICH_LINGUISTIC_SLICE.start + pivot_distance_offset

    nonzero_pivot_seen = False
    nonzero_mp_seen = False
    for sequence in sequences:
        for vector in sequence:
            if vector.date.startswith("2024-03"):
                # hash_no_ling has no linguistic row -> the slice
                # collapses to zero by design; skip when validating
                # pivot_distance is wired.
                continue
            rich = vector.as_rich_list()
            if rich[pivot_distance_global] != 0.0:
                nonzero_pivot_seen = True
            if rich[RICH_MP_SURPRISE_SLICE][0] != 0.0:
                nonzero_mp_seen = True
    assert nonzero_pivot_seen, (
        "linguistic_features.parquet rebuild did not feed pivot_distance "
        "into the rich-feature slice"
    )
    assert nonzero_mp_seen, (
        "mp_surprises.parquet did not feed the MP-surprise slice"
    )


# ---------------------------------------------------------------------------
# Text-embedding tests (PR #176)
#
# The four tests below exercise the prior-4 statement pool, the missing-
# flag semantics on the first event in the corpus, the encoder-driven
# in_dim, and the byte-identical no-text path. They use a synthetic
# embedding parquet under ``tmp_path`` so no real encoder fires; the
# test patches ``revision_for`` and ``resolve_cache_paths`` so the
# loader resolves to the synthetic file.
# ---------------------------------------------------------------------------


def _write_synthetic_embedding_parquet(
    cache_dir: Path,
    encoder_alias: str,
    *,
    rows: list[dict[str, "Any"]],
    revision: str = "abc1234",
) -> Path:
    """Materialise a synthetic embedding-cache parquet on disk."""

    import pandas as pd

    cache_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = cache_dir / f"{encoder_alias}_{revision[:12]}.parquet"
    pd.DataFrame(rows).to_parquet(parquet_path, index=False)
    return parquet_path


def _patch_encoder_registry(
    monkeypatch: pytest.MonkeyPatch,
    *,
    encoder_alias: str,
    revision: str = "abc1234",
    cache_dir: Path | None = None,
) -> None:
    """Patch the registry + cache paths to point at a tmp_path artefact."""

    from app.data import embedding_cache as embedding_cache_module
    from app.training import loaders as loaders_module

    def _revision_for(_alias: str) -> str:
        return revision

    monkeypatch.setattr(loaders_module, "_logger", loaders_module._logger)

    # The loader does ``from app.models.registry import revision_for``
    # inside ``_read_chunk_embedding_lookup`` -- patch the registry
    # module itself so the import resolves to our stub.
    import app.models.registry as registry_module

    monkeypatch.setattr(registry_module, "revision_for", _revision_for)
    monkeypatch.setattr(
        embedding_cache_module, "DEFAULT_CACHE_DIR", cache_dir or embedding_cache_module.DEFAULT_CACHE_DIR
    )


def test_text_embedding_pool_weighting_decays_with_age(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The most recent prior statement gets the largest weight.

    Synthesise four prior statements at Delta t = 0, 30, 60, 90 days.
    The current event is the fifth statement, dated 90 days after the
    oldest. With ``lambda_inv_days = 30`` the most recent (Delta t = 0)
    statement contributes >0.5 of the pooled mass and the farthest
    (Delta t = 90) contributes <0.1.
    """

    import numpy as np

    package_id = "tp_unit_text_embed_weighting"
    package_dir = tmp_path / "processed" / package_id
    package_dir.mkdir(parents=True)
    cache_dir = tmp_path / "raw" / "embeddings"

    monkeypatch.setattr(loaders, "DATA_DIR", tmp_path)
    _patch_encoder_registry(monkeypatch, encoder_alias="finbert", cache_dir=cache_dir)
    monkeypatch.setattr(loaders, "_logger", loaders._logger)

    # Five statements spaced 30 days apart, base_close stays constant.
    dates = ["2024-01-01", "2024-01-31", "2024-03-01", "2024-03-31", "2024-04-30"]
    text_hashes = [f"hash_{i}" for i in range(5)]
    events = []
    for date_str, hash_str in zip(dates, text_hashes):
        events.append(
            _event_row(
                event_date=date_str,
                text_hash=hash_str,
                axis_stance="hawkish",
                realized_return=0.001,
                realized_date=f"{date_str[:8]}{int(date_str[8:]) + 1:02d}",
                base_close=4500.0,
            )
        )
    pd.DataFrame(events).to_parquet(package_dir / "events.parquet", index=False)
    pd.DataFrame(
        [{"text_hash": h, "split_tag": "train"} for h in text_hashes]
    ).to_parquet(package_dir / "splits_train_val_test.parquet", index=False)

    # Build four distinct embedding vectors so the pooled mean is
    # identifiable. The fifth statement (current event) carries its
    # own embedding too, but the pooler reads only the four priors.
    in_dim = 8
    embedding_rows = []
    for i, (date_str, hash_str) in enumerate(zip(dates, text_hashes)):
        vec = np.zeros(in_dim, dtype=np.float32)
        vec[i] = 1.0
        embedding_rows.append(
            {
                "record_id": hash_str,
                "doc_id": hash_str,
                "event_date": date_str,
                "chunk_index": 0,
                "chunk_preview": "",
                "embedding": vec.tolist(),
            }
        )
    _write_synthetic_embedding_parquet(
        cache_dir, "finbert", rows=embedding_rows
    )

    sequences = loaders.load_training_sequences_from_package(
        package_id,
        text_encoder="finbert",
        text_adapter_dim=64,
        text_pool_lambda_inv_days=30.0,
        text_embedding_cache_dir=cache_dir,
    )
    # 5 events all tagged train -> 5 sequences.
    assert len(sequences) == 5

    # Final event (hash_4) has 4 priors at Delta t = 30, 60, 90, 120.
    # Expected softmax weights at lambda=30: exp(-1, -2, -3, -4)
    # normalised. Most recent (hash_3, idx=3) > 0.5, oldest (hash_0,
    # idx=0) < 0.1.
    target_event = sequences[-1][0]
    pooled = np.asarray(target_event.text_embedding_pooled, dtype=np.float32)
    assert pooled.shape == (in_dim,)
    # Weights on (hash_3, hash_2, hash_1, hash_0) -- the four prior
    # statements in chronological order on the lookup. The most
    # recent (hash_3 at Delta t = 30) places the largest mass on
    # vec[3]; the oldest (hash_0 at Delta t = 120) gets the smallest.
    weight_recent = float(pooled[3])
    weight_oldest = float(pooled[0])
    assert weight_recent > 0.5
    assert weight_oldest < 0.1
    assert target_event.text_embedding_missing == pytest.approx(0.0)


def test_text_embedding_missing_when_fewer_than_one_prior(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The chronologically-earliest event has no prior to pool over."""

    import numpy as np

    package_id = "tp_unit_text_embed_missing"
    package_dir = tmp_path / "processed" / package_id
    package_dir.mkdir(parents=True)
    cache_dir = tmp_path / "raw" / "embeddings"

    monkeypatch.setattr(loaders, "DATA_DIR", tmp_path)
    _patch_encoder_registry(monkeypatch, encoder_alias="finbert", cache_dir=cache_dir)

    events = [
        _event_row(
            event_date="2024-01-01",
            text_hash="hash_first",
            axis_stance="hawkish",
            realized_return=0.001,
            realized_date="2024-01-02",
            base_close=4500.0,
        ),
        _event_row(
            event_date="2024-02-01",
            text_hash="hash_second",
            axis_stance="dovish",
            realized_return=-0.001,
            realized_date="2024-02-02",
            base_close=4500.0,
        ),
    ]
    pd.DataFrame(events).to_parquet(package_dir / "events.parquet", index=False)
    pd.DataFrame(
        [
            {"text_hash": "hash_first", "split_tag": "train"},
            {"text_hash": "hash_second", "split_tag": "train"},
        ]
    ).to_parquet(package_dir / "splits_train_val_test.parquet", index=False)

    in_dim = 8
    embedding_rows = [
        {
            "record_id": h,
            "doc_id": h,
            "event_date": d,
            "chunk_index": 0,
            "chunk_preview": "",
            "embedding": np.eye(in_dim)[i].astype(np.float32).tolist(),
        }
        for i, (h, d) in enumerate(
            [("hash_first", "2024-01-01"), ("hash_second", "2024-02-01")]
        )
    ]
    _write_synthetic_embedding_parquet(
        cache_dir, "finbert", rows=embedding_rows
    )

    sequences = loaders.load_training_sequences_from_package(
        package_id,
        text_encoder="finbert",
        text_adapter_dim=64,
        text_embedding_cache_dir=cache_dir,
    )
    assert len(sequences) == 2

    by_event_date = {seq[0].date[:10]: seq for seq in sequences}
    # First sequence's event_date doesn't have a chronological prior
    # statement -> missing flag 1.0, pooled empty.
    first = sequences[0]
    assert first[0].text_embedding_missing == pytest.approx(1.0)
    assert first[0].text_embedding_pooled == []
    del by_event_date
    # Second sequence has hash_first as its prior; pool should land on
    # the eye[0] basis vector at full weight.
    second = sequences[1]
    assert second[0].text_embedding_missing == pytest.approx(0.0)
    assert len(second[0].text_embedding_pooled) == in_dim
    assert second[0].text_embedding_pooled[0] == pytest.approx(1.0)


def test_text_embedding_encoder_choice_changes_input_dim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pooled embedding inherits the encoder's native dim."""

    import numpy as np

    package_id = "tp_unit_text_embed_dim"
    package_dir = tmp_path / "processed" / package_id
    package_dir.mkdir(parents=True)
    cache_dir = tmp_path / "raw" / "embeddings"

    monkeypatch.setattr(loaders, "DATA_DIR", tmp_path)

    events = [
        _event_row(
            event_date=f"2024-{month:02d}-15",
            text_hash=f"hash_{i}",
            axis_stance="hawkish",
            realized_return=0.001,
            realized_date=f"2024-{month:02d}-16",
            base_close=4500.0,
        )
        for i, month in enumerate([1, 2, 3])
    ]
    pd.DataFrame(events).to_parquet(package_dir / "events.parquet", index=False)
    pd.DataFrame(
        [{"text_hash": e["text_hash"], "split_tag": "train"} for e in events]
    ).to_parquet(package_dir / "splits_train_val_test.parquet", index=False)

    def _row_with_dim(dim: int) -> list[dict]:
        return [
            {
                "record_id": e["text_hash"],
                "doc_id": e["text_hash"],
                "event_date": e["event_date"],
                "chunk_index": 0,
                "chunk_preview": "",
                "embedding": np.ones(dim, dtype=np.float32).tolist(),
            }
            for e in events
        ]

    # FinBERT path: in_dim = 768.
    _patch_encoder_registry(monkeypatch, encoder_alias="finbert", cache_dir=cache_dir)
    _write_synthetic_embedding_parquet(
        cache_dir, "finbert", rows=_row_with_dim(768)
    )
    finbert_sequences = loaders.load_training_sequences_from_package(
        package_id,
        text_encoder="finbert",
        text_adapter_dim=64,
        text_embedding_cache_dir=cache_dir,
    )
    second_event = finbert_sequences[1][0]
    assert len(second_event.text_embedding_pooled) == 768

    # voyage_finance_2 path: in_dim = 1024. Clear caches so the second
    # invocation re-reads the synthetic parquet rather than reusing
    # the FinBERT 768-dim payload.
    _patch_encoder_registry(monkeypatch, encoder_alias="voyage_finance_2", cache_dir=cache_dir)
    _write_synthetic_embedding_parquet(
        cache_dir, "voyage_finance_2", rows=_row_with_dim(1024)
    )
    voyage_sequences = loaders.load_training_sequences_from_package(
        package_id,
        text_encoder="voyage_finance_2",
        text_adapter_dim=64,
        text_embedding_cache_dir=cache_dir,
    )
    voyage_second = voyage_sequences[1][0]
    assert len(voyage_second.text_embedding_pooled) == 1024


def test_no_text_embeddings_zeros_slice_but_shape_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``use_text_embeddings=False`` skips the encoder lookup entirely."""

    import numpy as np

    package_id = "tp_unit_text_embed_off"
    package_dir = tmp_path / "processed" / package_id
    package_dir.mkdir(parents=True)
    cache_dir = tmp_path / "raw" / "embeddings"

    monkeypatch.setattr(loaders, "DATA_DIR", tmp_path)
    _patch_encoder_registry(monkeypatch, encoder_alias="finbert", cache_dir=cache_dir)

    events = [
        _event_row(
            event_date="2024-02-15",
            text_hash="hash_a",
            axis_stance="hawkish",
            realized_return=0.001,
            realized_date="2024-02-16",
            base_close=4500.0,
        ),
        _event_row(
            event_date="2024-03-15",
            text_hash="hash_b",
            axis_stance="dovish",
            realized_return=-0.001,
            realized_date="2024-03-16",
            base_close=4500.0,
        ),
    ]
    pd.DataFrame(events).to_parquet(package_dir / "events.parquet", index=False)
    pd.DataFrame(
        [{"text_hash": "hash_a", "split_tag": "train"}, {"text_hash": "hash_b", "split_tag": "train"}]
    ).to_parquet(package_dir / "splits_train_val_test.parquet", index=False)

    _write_synthetic_embedding_parquet(
        cache_dir,
        "finbert",
        rows=[
            {
                "record_id": "hash_a",
                "doc_id": "hash_a",
                "event_date": "2024-02-15",
                "chunk_index": 0,
                "chunk_preview": "",
                "embedding": np.ones(768, dtype=np.float32).tolist(),
            }
        ],
    )

    sequences_off = loaders.load_training_sequences_from_package(
        package_id,
        text_encoder="finbert",
        text_adapter_dim=64,
        use_text_embeddings=False,
        text_embedding_cache_dir=cache_dir,
    )
    # text path off -> no pooled vector, missing-flag stays at the
    # ``FeatureVector`` default (1.0).
    for sequence in sequences_off:
        for vector in sequence:
            assert vector.text_embedding_pooled == []
            assert vector.text_embedding_missing == pytest.approx(1.0)

    sequences_on = loaders.load_training_sequences_from_package(
        package_id,
        text_encoder="finbert",
        text_adapter_dim=64,
        use_text_embeddings=True,
        text_embedding_cache_dir=cache_dir,
    )
    # Both flag values produce the same number of sequences and the
    # same per-bar scalar 35-dim slice. The only difference is the
    # pooled list + missing-flag pair on the FeatureVector.
    assert len(sequences_on) == len(sequences_off)
    on_scalar = sequences_on[0][0].as_rich_list()
    off_scalar = sequences_off[0][0].as_rich_list()
    assert on_scalar == off_scalar
    # hash_b (chronologically second) hits the pool with hash_a as the
    # prior; the pooled embedding lands at the eye[0] basis vector.
    chronological = sorted(
        sequences_on, key=lambda seq: seq[-1].date[:10]
    )
    second_event = chronological[1][0]
    assert len(second_event.text_embedding_pooled) == 768
    assert second_event.text_embedding_missing == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Walk-forward split tests
#
# ``load_walk_forward_split`` returns three pre-partitioned sequence
# lists (train + val + test) plus per-list event dates. The tests
# below exercise both the single-fold path (split-tag partition off
# ``splits_train_val_test.parquet``) and the multi-fold path
# (chronological partition off ``fold_manifest_expanding_walk_forward.json``).
# ---------------------------------------------------------------------------


_WALK_FORWARD_PACKAGE_ID = "tp_unit_walk_forward_v0"


def _wf_event_row(*, event_date: str, text_hash: str, base_close: float) -> dict:
    """Build a minimal event row for the walk-forward tests."""

    return _event_row(
        event_date=event_date,
        text_hash=text_hash,
        axis_stance="neutral",
        realized_return=0.001,
        realized_date=event_date,
        base_close=base_close,
    )


@pytest.fixture
def walk_forward_package_dir(tmp_path: Path, monkeypatch) -> Path:
    """Materialise a tiny package with split tags + a two-fold manifest."""

    package_dir = tmp_path / "processed" / _WALK_FORWARD_PACKAGE_ID
    package_dir.mkdir(parents=True)
    monkeypatch.setattr(loaders, "DATA_DIR", tmp_path)

    # Six events spaced across two fold windows. The fold manifest
    # below makes wf_fold_1 train cover the first two events and
    # wf_fold_2 train cover the first four (expanding window).
    events = [
        _wf_event_row(event_date="2020-01-15", text_hash="hash_t1", base_close=4400.0),
        _wf_event_row(event_date="2020-02-15", text_hash="hash_t2", base_close=4410.0),
        _wf_event_row(event_date="2020-03-15", text_hash="hash_v1", base_close=4420.0),
        _wf_event_row(event_date="2020-04-15", text_hash="hash_v2", base_close=4430.0),
        _wf_event_row(event_date="2020-05-15", text_hash="hash_te1", base_close=4440.0),
        _wf_event_row(event_date="2020-06-15", text_hash="hash_te2", base_close=4450.0),
    ]
    pd.DataFrame(events).to_parquet(package_dir / "events.parquet", index=False)

    splits = [
        {"text_hash": "hash_t1", "split_tag": "train"},
        {"text_hash": "hash_t2", "split_tag": "train"},
        {"text_hash": "hash_v1", "split_tag": "val"},
        {"text_hash": "hash_v2", "split_tag": "val"},
        {"text_hash": "hash_te1", "split_tag": "test"},
        {"text_hash": "hash_te2", "split_tag": "test"},
    ]
    pd.DataFrame(splits).to_parquet(
        package_dir / "splits_train_val_test.parquet", index=False
    )

    manifest = {
        "evaluation_protocol": "evaluation_protocol_v1",
        "folds": [
            {
                "fold_id": "wf_fold_1",
                "train_start": "2020-01-01",
                "train_end": "2020-02-29",
                "val_start": "2020-03-01",
                "val_end": "2020-03-31",
                "test_start": "2020-04-01",
                "test_end": "2020-04-30",
            },
            {
                "fold_id": "wf_fold_2",
                "train_start": "2020-01-01",
                "train_end": "2020-04-30",
                "val_start": "2020-05-01",
                "val_end": "2020-05-31",
                "test_start": "2020-06-01",
                "test_end": "2020-06-30",
            },
        ],
    }
    (package_dir / "fold_manifest_expanding_walk_forward.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return package_dir


def test_walk_forward_split_single_fold_partitions_from_split_tag(
    walk_forward_package_dir: Path,
) -> None:
    split = loaders.load_walk_forward_split(
        _WALK_FORWARD_PACKAGE_ID, rich_features=False
    )
    assert split.fold_id is None
    assert split.protocol == "single-fold"
    # Six events: 2 train, 2 val, 2 test.
    assert len(split.train) == 2
    assert len(split.val) == 2
    assert len(split.test) == 2
    # Sum equals the events.parquet total.
    assert len(split.train) + len(split.val) + len(split.test) == 6
    # Event-date ordering matches the per-list count.
    assert split.train_event_dates == ["2020-01-15", "2020-02-15"]
    assert split.val_event_dates == ["2020-03-15", "2020-04-15"]
    assert split.test_event_dates == ["2020-05-15", "2020-06-15"]


def test_walk_forward_split_multi_fold_partitions_from_manifest(
    walk_forward_package_dir: Path,
) -> None:
    fold_1 = loaders.load_walk_forward_split(
        _WALK_FORWARD_PACKAGE_ID, fold_id="wf_fold_1", rich_features=False
    )
    fold_2 = loaders.load_walk_forward_split(
        _WALK_FORWARD_PACKAGE_ID, fold_id="wf_fold_2", rich_features=False
    )
    assert fold_1.protocol == "walk-forward"
    assert fold_1.fold_id == "wf_fold_1"
    # fold_1: train covers everything before val_start=2020-03-01 ->
    # hash_t1 (2020-01-15) + hash_t2 (2020-02-15) -> 2 sequences.
    assert len(fold_1.train) == 2
    assert fold_1.train_event_dates == ["2020-01-15", "2020-02-15"]
    # fold_1 val: 2020-03-01 to 2020-03-31 -> hash_v1.
    assert len(fold_1.val) == 1
    assert fold_1.val_event_dates == ["2020-03-15"]
    # fold_1 test: 2020-04-01 to 2020-04-30 -> hash_v2.
    assert len(fold_1.test) == 1
    assert fold_1.test_event_dates == ["2020-04-15"]

    # fold_2: train covers everything before 2020-05-01 ->
    # hash_t1 + hash_t2 + hash_v1 + hash_v2 -> 4 sequences.
    assert len(fold_2.train) == 4
    assert fold_2.train_event_dates == [
        "2020-01-15",
        "2020-02-15",
        "2020-03-15",
        "2020-04-15",
    ]
    # Expanding-window contract: train_2 strictly contains train_1.
    assert set(fold_1.train_event_dates).issubset(set(fold_2.train_event_dates))
    assert len(fold_2.train) > len(fold_1.train)
    # fold_2 val and test.
    assert fold_2.val_event_dates == ["2020-05-15"]
    assert fold_2.test_event_dates == ["2020-06-15"]


def test_walk_forward_split_unknown_fold_raises(
    walk_forward_package_dir: Path,
) -> None:
    with pytest.raises(ValueError, match="not found in fold manifest"):
        loaders.load_walk_forward_split(
            _WALK_FORWARD_PACKAGE_ID, fold_id="wf_fold_does_not_exist"
        )


def test_walk_forward_split_no_test_events_raises(
    tmp_path: Path, monkeypatch
) -> None:
    """A fold whose test window is empty must raise loudly."""

    package_id = "tp_unit_empty_test_window"
    package_dir = tmp_path / "processed" / package_id
    package_dir.mkdir(parents=True)
    monkeypatch.setattr(loaders, "DATA_DIR", tmp_path)

    events = [
        _wf_event_row(event_date="2020-01-15", text_hash="hash_only", base_close=4400.0),
    ]
    pd.DataFrame(events).to_parquet(package_dir / "events.parquet", index=False)
    pd.DataFrame(
        [{"text_hash": "hash_only", "split_tag": "train"}]
    ).to_parquet(package_dir / "splits_train_val_test.parquet", index=False)

    manifest = {
        "folds": [
            {
                "fold_id": "wf_fold_1",
                "train_start": "2020-01-01",
                "train_end": "2020-02-29",
                "val_start": "2020-03-01",
                "val_end": "2020-03-31",
                "test_start": "2020-04-01",
                "test_end": "2020-04-30",
            }
        ],
    }
    (package_dir / "fold_manifest_expanding_walk_forward.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="empty test partition"):
        loaders.load_walk_forward_split(package_id, fold_id="wf_fold_1")


def test_walk_forward_split_back_compat_wrapper_emits_deprecation(
    walk_forward_package_dir: Path,
) -> None:
    """The legacy ``load_training_sequences_from_package`` is deprecated.

    It still returns the train partition (the pre-PR semantics) so the
    callers that have not migrated keep working, but every call emits
    a DeprecationWarning pointing at ``load_walk_forward_split``.
    """

    with pytest.warns(DeprecationWarning, match="load_walk_forward_split"):
        sequences = loaders.load_training_sequences_from_package(
            _WALK_FORWARD_PACKAGE_ID, rich_features=False
        )
    # Two train-tagged events survive the legacy wrapper.
    assert len(sequences) == 2
    target_dates = sorted(seq[-1].date[:10] for seq in sequences)
    # Synthesised realized_date was the same date as event_date in the
    # fixture, so the appended target frame uses each event's own date.
    assert target_dates == ["2020-01-15", "2020-02-15"]
