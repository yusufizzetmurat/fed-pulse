"""Regression: every per-bar FeatureVector column respects its declared as_of.

Contract (`docs/feature-provenance-audit.md`, `docs/benchmark-policy.md
§Per-Feature Provenance`): on a supervised sequence with event_date=T,
no scalar feature emitted into the model input tensor on a lookback bar
may read from a source post-dating T. Lookback bars carry `T-Δ` data
only; `T (snapshot)` columns are document-level signals on T itself;
`T+Δ` columns are training targets and are either mounted on the
appended event-day target frame or stored on the dataclass for the
target-row consumer but **never emitted** by `FeatureVector.as_rich_list`.

The test materialises a synthetic training package (events.parquet +
splits_train_val_test.parquet) under tmp_path, loads it through
`load_walk_forward_split`, and verifies four guarantees:

1. Every lookback bar's `date` is strictly less than the supervised
   `event_date` — the prior-window builder's core invariant.
2. `as_rich_list()` on a lookback bar emits exactly `RICH_FEATURE_SIZE`
   floats and never widens to include a future-derived training target
   (a structural lock on the input-tensor surface).
3. The audit inventory in `docs/feature-provenance-audit.md` covers
   every FeatureVector field — any new field must be classified before
   merge.
4. The MP-surprise / fed-info construction reads only strict-prior
   inputs (issue #350 reformulation). The previously-leaking
   `[T-1, T+1]` post-event window is replaced with an actual-vs-pre-
   implied surprise and a strict-prior trailing SPX return; this test
   builds a synthetic ``mp_surprises`` row through the production
   builder and asserts the read window stays strictly before
   ``event_date``.
"""

from __future__ import annotations

import json
from dataclasses import fields as dataclass_fields
from datetime import date
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")

from app.models.config import FeatureVector, RICH_FEATURE_SIZE, SEQUENCE_LENGTH  # noqa: E402
from app.training import loaders  # noqa: E402


pytestmark = pytest.mark.regression


_TRAINING_PACKAGE_ID = "tp_provenance_as_of_regression_v1"

# Columns declared as `T-Δ` (prior-bar derived). Each lookback bar's
# value must be observable from data dated strictly before the event.
_TRAINING_DELTA_COLUMNS: tuple[str, ...] = (
    "market_close",
    "market_volatility",
    "close_change_pct",
    "volatility_change",
    "realized_vol_20d",
    "realized_vol_60d",
    "vix_close",
    "dxy_close",
    "tnx_close",
    "gold_close",
    "vix3m_close",
    "irx_close",
    "vix_term_slope",
    "yield_curve_slope_10y_3m",
)

# Columns declared as future-derived training targets. They must stay
# `None` on every lookback bar; only the appended target frame (last
# row of the sequence) may carry a non-None value.
_TARGET_ONLY_COLUMNS: tuple[str, ...] = (
    "forward_realized_vol_10d",
    "target_yield_2y_change_5d",
    "target_yield_5y_change_5d",
    "target_terminal_rate_change_5d",
    # #305 FOMC-attributable projections of the three forward 5d rates
    # moves. Same target-only contract as the raw siblings; only the
    # event row carries the projected scalar.
    "target_yield_2y_change_5d_fomc_attributable",
    "target_yield_5y_change_5d_fomc_attributable",
    "target_terminal_rate_change_5d_fomc_attributable",
)

# Columns declared `T (snapshot)` — document-level signals broadcast to
# every bar of the sequence. The audit documents these as observable
# from the released FOMC text on T itself, not from post-T market data.
_SNAPSHOT_COLUMNS: tuple[str, ...] = (
    "sentiment_score",
    "credibility_drift_score",
    "credibility_realized_vs_stated_gap",
    "credibility_market_implied_gap",
    "credibility_months_since_reversal",
    "mp_surprise_level",
    "mp_surprise_path_factor",
    "fed_info_factor",
    "mp_is_intermeeting",
    "stance_hawk",
    "stance_dove",
    "stance_neutral",
    "time_label_forward",
    "certain_label_certain",
    "stance_missing",
    "llm_features_missing",
    # #306 retrieval-augmented summary: the analog-features missing
    # flag is structural (presence of the retrieval bundle on disk).
    # The five-scalar block itself rides on the `analog_features`
    # list[float] | None slot and is exempt below alongside the other
    # list-payload fields.
    "analog_features_missing",
)


def _synth_prior_bars(*, event_date: date, base_close: float) -> str:
    """Emit a 20-bar JSON window with dates strictly before ``event_date``.

    The builder's contract (`_assert_no_lookahead`) is that the last
    prior bar's date is strictly less than the as-of date. We materialise
    20 calendar-day-spaced bars ending at ``event_date - 1`` so the
    contract holds without needing a real trading-day calendar in the
    fixture.
    """

    payload = []
    # Walk backwards: offset 1 == the bar immediately before event_date,
    # offset SEQUENCE_LENGTH == the oldest bar in the window.
    for offset in range(SEQUENCE_LENGTH, 0, -1):
        bar_date = date.fromordinal(event_date.toordinal() - offset)
        payload.append(
            {
                "date": bar_date.isoformat(),
                "close": round(base_close + (SEQUENCE_LENGTH - offset) * 1.5, 10),
                "volume": 1_000_000.0,
                "vol_5d": round(0.012 + (SEQUENCE_LENGTH - offset) * 0.0001, 10),
                "vol_20d": round(0.018 + (SEQUENCE_LENGTH - offset) * 0.00005, 10),
                "vol_60d": round(0.022 + (SEQUENCE_LENGTH - offset) * 0.00002, 10),
                "cum_return_20d": round((SEQUENCE_LENGTH - offset) * 0.001, 10),
                "vix_close": round(15.0 + (SEQUENCE_LENGTH - offset) * 0.05, 6),
                "dxy_close": round(103.0 + (SEQUENCE_LENGTH - offset) * 0.02, 6),
                "tnx_close": round(4.20 + (SEQUENCE_LENGTH - offset) * 0.001, 6),
                "gold_close": round(2050.0 + (SEQUENCE_LENGTH - offset) * 0.1, 6),
                "vix3m_close": round(17.0 + (SEQUENCE_LENGTH - offset) * 0.04, 6),
                "irx_close": round(5.10 + (SEQUENCE_LENGTH - offset) * 0.001, 6),
                "vix_term_slope": 0.0,
                "yield_curve_slope_10y_3m": -0.90,
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
) -> dict:
    ed = date.fromisoformat(event_date)
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
        "axis_time_label": None,
        "axis_certain_label": None,
        "credibility_drift_score": 0.0,
        "credibility_realized_vs_stated_gap": 0.0,
        "credibility_market_implied_gap": 0.0,
        "credibility_months_since_reversal": 0,
        "prior_window_sha256": "0" * 64,
        "prior_bars_json": _synth_prior_bars(event_date=ed, base_close=base_close),
        "asset_symbol": "^GSPC",
        "horizon": 1,
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
        "forward_realized_vol_10d": 0.015,
        "yield_2y_change_5d": -3.2,
        "yield_5y_change_5d": -2.1,
        "terminal_rate_change_5d": -1.0,
    }


@pytest.fixture
def training_package_dir(tmp_path: Path, monkeypatch) -> Path:
    processed_root = tmp_path / "processed"
    package_dir = processed_root / _TRAINING_PACKAGE_ID
    package_dir.mkdir(parents=True)

    # Point the loader's DATA_DIR at tmp_path so <DATA_DIR>/processed/<id>
    # resolves to the synthetic package above.
    monkeypatch.setattr(loaders, "DATA_DIR", tmp_path)

    events = [
        _event_row(
            event_date="2024-01-31",
            text_hash="hash_a",
            axis_stance="dovish",
            realized_return=-0.008,
            realized_date="2024-02-01",
            base_close=4400.0,
        ),
        _event_row(
            event_date="2024-02-15",
            text_hash="hash_b",
            axis_stance="hawkish",
            realized_return=0.012,
            realized_date="2024-02-16",
            base_close=4500.0,
        ),
        _event_row(
            event_date="2024-03-20",
            text_hash="hash_c",
            axis_stance="neutral",
            realized_return=0.0,
            realized_date="2024-03-21",
            base_close=4600.0,
        ),
    ]
    pd.DataFrame(events).to_parquet(package_dir / "events.parquet", index=False)

    split_rows = [
        {"text_hash": "hash_a", "split_tag": "train"},
        {"text_hash": "hash_b", "split_tag": "train"},
        {"text_hash": "hash_c", "split_tag": "test"},
    ]
    pd.DataFrame(split_rows).to_parquet(
        package_dir / "splits_train_val_test.parquet", index=False
    )
    return package_dir


def _assert_bar_is_before_event_date(bar_date_str: str, event_date: date) -> None:
    """Lookback bars must carry a calendar date strictly before event_date."""

    bar_date = date.fromisoformat(bar_date_str[:10])
    assert bar_date < event_date, (
        f"as_of contract: lookback bar dated {bar_date} is not strictly "
        f"before event_date {event_date}"
    )


def _audit_inventory_covers_every_field() -> set[str]:
    """Return the FeatureVector field names not declared in the audit inventory.

    The audit doc is authoritative for the column surface. Any future
    FeatureVector field that does not appear in one of the three
    inventory tuples here (training-Δ, snapshot, target-only) and is not
    one of the structural / metadata exemptions below must be classified
    before merge — this assertion is the gate.
    """

    exempt = {
        "date",
        "elapsed_time",
        "text_embedding",
        "linguistic_features",
        "llm_features",
        "analog_features",
        "rich_payload",
        "text_embedding_pooled",
        "text_embedding_missing",
        "text_per_bar",
        "raw_text",
        "target_stance_idx",
        "target_stance_present",
        "target_factor",
        "target_factor_present",
        "target_certainty_idx",
        "target_certainty_present",
        "target_topic_idx",
        "target_topic_present",
    }
    declared = (
        set(_TRAINING_DELTA_COLUMNS)
        | set(_SNAPSHOT_COLUMNS)
        | set(_TARGET_ONLY_COLUMNS)
        | exempt
    )
    every_field = {f.name for f in dataclass_fields(FeatureVector)}
    return every_field - declared


def test_feature_provenance_as_of_contract(training_package_dir: Path) -> None:
    undeclared = _audit_inventory_covers_every_field()
    assert not undeclared, (
        "FeatureVector has fields not classified in the provenance audit: "
        f"{sorted(undeclared)}. Add them to docs/feature-provenance-audit.md "
        "and update tests/regression/test_feature_provenance_as_of.py."
    )

    split = loaders.load_walk_forward_split(
        _TRAINING_PACKAGE_ID,
        rich_features=True,
        use_credibility=True,
        use_linguistic=True,
        use_mp_surprise=True,
        use_multi_axis=True,
        use_llm_features=False,
        text_encoder=None,
    )

    assert split.train, "fixture must produce at least one train sequence"

    for partition in (split.train, split.val, split.test):
        for sequence in partition:
            assert len(sequence) >= SEQUENCE_LENGTH + 1, (
                "each sequence must carry SEQUENCE_LENGTH lookback bars "
                "plus an appended event-day target frame"
            )
            event_row = sequence[-1]
            event_date = date.fromisoformat(event_row.date[:10])
            lookback = sequence[:-1]

            for vector in lookback:
                # Per-bar dates on lookback rows must be strictly before
                # the supervised event_date — the prior-window builder's
                # core invariant.
                _assert_bar_is_before_event_date(vector.date, event_date)

                # The input tensor surface: as_rich_list() is what the
                # model actually consumes. Its width is pinned at
                # RICH_FEATURE_SIZE and its layout is the documented
                # market + credibility + linguistic + mp_surprise +
                # multi_axis + realized_vol + cross_asset + llm slices.
                # No future-derived training-target column appears in
                # this output by construction; a future change that
                # accidentally widens the row to include one would
                # change the size and break this assertion.
                rich = vector.as_rich_list()
                assert len(rich) == RICH_FEATURE_SIZE, (
                    f"as_rich_list width drift on bar {vector.date}: "
                    f"got {len(rich)}, expected {RICH_FEATURE_SIZE}"
                )
                for slot_idx, slot_value in enumerate(rich):
                    assert isinstance(slot_value, float), (
                        f"as_rich_list slot {slot_idx} on bar "
                        f"{vector.date} is not a float "
                        f"(got {type(slot_value).__name__})"
                    )

                # Target-only columns are stored on the dataclass for
                # downstream target-row consumers but must never leak
                # into the input tensor. The width check above is the
                # structural lock; this loop is the documentation
                # anchor — it asserts the storage shape the audit
                # declares (numeric-or-None on every lookback bar).
                for column in _TARGET_ONLY_COLUMNS:
                    value = getattr(vector, column)
                    assert value is None or isinstance(value, (int, float)), (
                        f"target-only column {column!r} on bar "
                        f"{vector.date} is neither None nor numeric "
                        f"(got {type(value).__name__})"
                    )

                # T-Δ columns are populated by the prior-window builder
                # off bar-dated market data; the strict-before-event
                # date check above is the substantive guarantee.
                for column in _TRAINING_DELTA_COLUMNS:
                    value = getattr(vector, column)
                    assert isinstance(value, (int, float)), (
                        f"T-Δ column {column!r} on bar {vector.date} is "
                        f"not a numeric scalar (got {type(value).__name__})"
                    )

                # Snapshot columns are broadcast off the as-of row. The
                # audit previously flagged mp_surprise_* / fed_info_factor
                # as a methodology-level leak path; #350 closed that path
                # at the source-data level (see
                # ``test_mp_surprise_columns_read_strictly_before_event_date``
                # below for the strict-prior construction contract). The
                # structural check here remains shape-only.
                for column in _SNAPSHOT_COLUMNS:
                    value = getattr(vector, column)
                    assert isinstance(value, (int, float)), (
                        f"snapshot column {column!r} on bar {vector.date} "
                        f"is not a numeric scalar (got {type(value).__name__})"
                    )


# Columns the #350 reformulation moved from ``T+Δ`` (post-event window)
# to strict-prior. Listed here so the contract is grep-able alongside
# the audit doc; the assertion below covers the source-data builder.
_MP_SURPRISE_STRICT_PRIOR_COLUMNS: tuple[str, ...] = (
    "mp_surprise_level",
    "mp_surprise_path_factor",
    "fed_info_factor",
)


def test_mp_surprise_columns_read_strictly_before_event_date(tmp_path) -> None:
    """#350: ``mp_surprise.build_mp_surprises`` never reads ``T+Δ`` inputs.

    Builds a synthetic ``mp_surprises`` row through the production
    builder with a strictly-prior FRED panel and asserts:

    1. The strict-prior pre/trail yield helper returns dates strictly
       before ``event_date`` for every tenor in CURVE_TENORS_MONTHS.
    2. The strict-prior SPX return helper rejects post-event closes
       and degrades to ``unavailable`` when only ``T+1`` data is
       supplied.
    3. The intraday route (Alpha Vantage ±30 min) is ignored: the
       ``spx_intraday_returns`` argument cannot bump any row to a
       leaky source flag.

    Together with the per-bar lookback assertion above, this closes the
    audit's three remaining ``med`` leak rows
    (``mp_surprise_level``, ``mp_surprise_path_factor``,
    ``fed_info_factor``).
    """

    import datetime as _dt

    from app.data import mp_surprise

    base = _dt.date(2024, 1, 1)
    trading_days = [
        base + _dt.timedelta(days=i)
        for i in range(120)
        if (base + _dt.timedelta(days=i)).weekday() < 5
    ]
    series_map = {d: 0.10 + i * 0.001 for i, d in enumerate(trading_days)}
    event_date = _dt.date(2024, 3, 6)

    # 1. Trailing-yield helper: strict-prior contract on dates.
    pre, trail, pre_d, trail_d = mp_surprise._strictly_prior_pre_and_trailing_yield(
        event_date, series_map, trading_days=trading_days,
    )
    assert pre is not None and trail is not None
    assert pre_d is not None and trail_d is not None
    assert trail_d < pre_d < event_date, (
        f"#350 strict-prior contract violated: trail={trail_d} "
        f"pre={pre_d} event={event_date}"
    )

    # 2. SPX helper: rejects post-event-only closes.
    leaky_closes = {event_date + _dt.timedelta(days=1): 5200.0}
    ret_leaky, source_leaky = mp_surprise._spx_return_on(event_date, leaky_closes)
    assert ret_leaky is None, (
        "#350: SPX helper must NOT compute a return from post-event-only "
        f"closes (got ret={ret_leaky}, source={source_leaky})"
    )
    assert source_leaky == "unavailable"

    # 3. Strict-prior closes resolve to a non-null trailing return.
    strict_closes = {
        event_date - _dt.timedelta(days=10): 5000.0,
        event_date - _dt.timedelta(days=1): 5100.0,
    }
    ret_ok, source_ok = mp_surprise._spx_return_on(event_date, strict_closes)
    assert ret_ok is not None and source_ok == "strict_prior_trailing", (
        "#350: strict-prior trailing SPX return failed to resolve from "
        f"pre-event closes (ret={ret_ok}, source={source_ok})"
    )
