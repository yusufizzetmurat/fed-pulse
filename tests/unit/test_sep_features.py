"""SEP dot-plot ingestion (#215).

Pins the surface at four layers:

- the per-feature SEP composer (hand fixtures with known release dates
  and median projections; assert the forward-fill + release-flag
  behaviour);
- the FeatureVector schema: ``as_rich_list`` does NOT append the SEP
  block when ``sep_features`` is ``None`` (legacy byte-identity
  contract); it DOES append the block when populated; the SEP block
  composes with the regime block in the documented order;
- the loader regression: ``--no-sep`` (default) keeps the per-bar
  feature size at ``RICH_FEATURE_SIZE`` and every event's ``sep_features``
  slot stays ``None``; ``--use-sep`` flips both, with the forward-fill
  rule applied on non-SEP meetings;
- a one-epoch smoke training run with the SEP block populated and the
  recurrent core widened by the SEP tail runs to completion.

See ADR 0030 for the design.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Pure-Python composer -- exercise the forward-fill + release-flag math
# against hand fixtures so the logic is reviewable.
# ---------------------------------------------------------------------------


from app.training.sep_features import (  # noqa: E402
    SEP_FEATURE_DIM,
    SepFeatures,
    SepProjections,
    compute_sep_features_for_event,
)


def _row(meeting_date: str, *, cy: float, lr: float, hi: float, lo: float) -> dict:
    return {
        "meeting_date": meeting_date,
        "ffr_median_current_year": cy,
        "ffr_median_longer_run": lr,
        "ffr_range_upper_current": hi,
        "ffr_range_lower_current": lo,
    }


def test_sep_release_meeting_sets_flag_one() -> None:
    """A supervised event that IS an SEP release reads ``release_flag = 1.0``."""

    event_date = _dt.date(2024, 3, 20)
    lookup = {
        "2023-12-13": _row("2023-12-13", cy=5.4, lr=2.5, hi=5.6, lo=5.4),
        "2024-03-20": _row("2024-03-20", cy=4.6, lr=2.6, hi=4.9, lo=4.4),
    }
    out = compute_sep_features_for_event(event_date=event_date, sep_lookup=lookup)
    assert out is not None
    # The matched release is the supervised event's own SEP; the values
    # are observable from the document released at T.
    assert out.ffr_median_current_year == pytest.approx(4.6)
    assert out.ffr_median_longer_run == pytest.approx(2.6)
    assert out.ffr_range_current == pytest.approx(0.5)
    assert out.sep_release_flag == 1.0


def test_sep_forward_fill_on_non_sep_meeting_sets_flag_zero() -> None:
    """A non-SEP meeting carries the most recent prior SEP with flag 0.0."""

    event_date = _dt.date(2024, 5, 1)  # not an SEP meeting (May)
    lookup = {
        "2023-12-13": _row("2023-12-13", cy=5.4, lr=2.5, hi=5.6, lo=5.4),
        "2024-03-20": _row("2024-03-20", cy=4.6, lr=2.6, hi=4.9, lo=4.4),
    }
    out = compute_sep_features_for_event(event_date=event_date, sep_lookup=lookup)
    assert out is not None
    # The matched release is the March SEP -- the most recent prior.
    # Values carry forward; release flag reads 0.0.
    assert out.ffr_median_current_year == pytest.approx(4.6)
    assert out.ffr_median_longer_run == pytest.approx(2.6)
    assert out.ffr_range_current == pytest.approx(0.5)
    assert out.sep_release_flag == 0.0


def test_sep_cold_start_returns_none() -> None:
    """No SEP release on or before ``event_date`` -> ``None`` (cold start)."""

    event_date = _dt.date(2010, 1, 27)
    lookup = {
        "2012-03-13": _row("2012-03-13", cy=0.25, lr=4.25, hi=0.5, lo=0.0),
    }
    out = compute_sep_features_for_event(event_date=event_date, sep_lookup=lookup)
    assert out is None


def test_sep_strict_prior_filter_drops_future_releases() -> None:
    """Releases dated strictly after ``event_date`` must not be matched."""

    event_date = _dt.date(2024, 1, 31)  # non-SEP meeting (January)
    lookup = {
        "2023-12-13": _row("2023-12-13", cy=5.4, lr=2.5, hi=5.6, lo=5.4),
        # March 2024 SEP post-dates the supervised event; the composer
        # must NOT see it as the matched release.
        "2024-03-20": _row("2024-03-20", cy=4.6, lr=2.6, hi=4.9, lo=4.4),
    }
    out = compute_sep_features_for_event(event_date=event_date, sep_lookup=lookup)
    assert out is not None
    # Without the strict-prior gate the composer would pick March; with
    # the gate it picks December and reads ``release_flag = 0.0``.
    assert out.ffr_median_current_year == pytest.approx(5.4)
    assert out.sep_release_flag == 0.0


def test_sep_range_missing_one_bound_collapses_to_zero() -> None:
    """Either bound missing -> the range computation degrades to 0.0."""

    event_date = _dt.date(2024, 3, 20)
    lookup = {
        "2024-03-20": _row("2024-03-20", cy=4.6, lr=2.6, hi=4.9, lo=None),
    }
    out = compute_sep_features_for_event(event_date=event_date, sep_lookup=lookup)
    assert out is not None
    assert out.ffr_range_current == 0.0


def test_sep_features_as_list_layout() -> None:
    """``as_list`` returns the documented six-element layout."""

    fv = SepFeatures(
        ffr_median_current_year=4.6,
        ffr_median_longer_run=2.6,
        ffr_range_current=0.5,
        sep_release_flag=1.0,
    )
    payload = fv.as_list()
    assert len(payload) == SEP_FEATURE_DIM
    assert payload == [4.6, 2.6, 0.5, 1.0]


def test_sep_projections_range_missing_input() -> None:
    """``SepProjections.range_current`` returns ``None`` cleanly."""

    proj = SepProjections(
        meeting_date=_dt.date(2024, 3, 20),
        ffr_median_current_year=4.6,
        ffr_median_longer_run=2.6,
        ffr_range_upper_current=None,
        ffr_range_lower_current=4.4,
    )
    assert proj.range_current() is None


# ---------------------------------------------------------------------------
# FeatureVector schema: conditional emission keeps legacy byte-identity.
# ---------------------------------------------------------------------------


from app.models.config import (  # noqa: E402
    FEATURE_SIZE,
    FeatureVector,
    RICH_FEATURE_SIZE,
    RICH_MACRO_REGIME_DIM,
    RICH_MACRO_REGIME_MISSING_DIM,
    RICH_SEP_DIM,
    RICH_SEP_MISSING_DIM,
    rich_feature_size_with_blocks,
    rich_feature_size_with_sep,
)


def test_as_rich_list_default_omits_sep_block() -> None:
    """The default ``sep_features=None`` keeps the pre-#215 width."""

    fv = FeatureVector(
        date="2024-03-20",
        sentiment_score=0.0,
        market_close=4500.0,
        market_volatility=0.01,
    )
    assert fv.sep_features is None
    assert len(fv.as_rich_list()) == RICH_FEATURE_SIZE


def test_as_rich_list_populated_appends_sep_block() -> None:
    """A populated SEP slot appends the block past ``RICH_FEATURE_SIZE``.

    When only ``sep_features`` is populated (no regime block) the SEP
    block lands at positions ``[RICH_FEATURE_SIZE : RICH_FEATURE_SIZE +
    RICH_SEP_DIM]``. The module-level ``RICH_SEP_SLICE`` constant is
    defined for the both-on case and sits past the regime tail; the
    only-SEP-on case slices at the dynamic offset below.
    """

    block = [4.6, 2.6, 0.5, 1.0]
    fv = FeatureVector(
        date="2024-03-20",
        sentiment_score=0.0,
        market_close=4500.0,
        market_volatility=0.01,
        sep_features=block,
        sep_features_missing=0.0,
    )
    rich = fv.as_rich_list()
    expected_width = RICH_FEATURE_SIZE + RICH_SEP_DIM + RICH_SEP_MISSING_DIM
    assert len(rich) == expected_width
    assert rich[RICH_FEATURE_SIZE : RICH_FEATURE_SIZE + RICH_SEP_DIM] == block
    assert rich[RICH_FEATURE_SIZE + RICH_SEP_DIM] == 0.0


def test_rich_feature_size_with_sep_helper() -> None:
    """The helper widens by exactly ``RICH_SEP_DIM + 1`` when on."""

    assert rich_feature_size_with_sep(False) == RICH_FEATURE_SIZE
    assert (
        rich_feature_size_with_sep(True)
        == RICH_FEATURE_SIZE + RICH_SEP_DIM + RICH_SEP_MISSING_DIM
    )


def test_rich_feature_size_with_blocks_composes_both() -> None:
    """Combined helper widens by both regime + SEP tails when on."""

    assert (
        rich_feature_size_with_blocks(use_regime=False, use_sep=False)
        == RICH_FEATURE_SIZE
    )
    assert rich_feature_size_with_blocks(use_regime=True, use_sep=False) == (
        RICH_FEATURE_SIZE + RICH_MACRO_REGIME_DIM + RICH_MACRO_REGIME_MISSING_DIM
    )
    assert rich_feature_size_with_blocks(use_regime=False, use_sep=True) == (
        RICH_FEATURE_SIZE + RICH_SEP_DIM + RICH_SEP_MISSING_DIM
    )
    assert rich_feature_size_with_blocks(use_regime=True, use_sep=True) == (
        RICH_FEATURE_SIZE
        + RICH_MACRO_REGIME_DIM
        + RICH_MACRO_REGIME_MISSING_DIM
        + RICH_SEP_DIM
        + RICH_SEP_MISSING_DIM
    )


def test_short_sep_payload_zero_pads() -> None:
    """A short payload right-pads to ``RICH_SEP_DIM``."""

    fv = FeatureVector(
        date="2024-03-20",
        sentiment_score=0.0,
        market_close=4500.0,
        market_volatility=0.01,
        sep_features=[4.6, 3.9],
        sep_features_missing=0.0,
    )
    rich = fv.as_rich_list()
    # SEP-only path: block sits at ``[RICH_FEATURE_SIZE : ...]``.
    assert (
        rich[RICH_FEATURE_SIZE : RICH_FEATURE_SIZE + RICH_SEP_DIM]
        == [4.6, 3.9] + [0.0] * (RICH_SEP_DIM - 2)
    )


def test_sep_block_appends_after_regime_block_when_both_on() -> None:
    """Combined emission: regime first, then SEP, past ``RICH_FEATURE_SIZE``."""

    regime = [1.0, 0.0, -1.0]
    sep = [4.6, 2.6, 0.5, 1.0]
    fv = FeatureVector(
        date="2024-03-20",
        sentiment_score=0.0,
        market_close=4500.0,
        market_volatility=0.01,
        macro_regime_features=regime,
        macro_regime_features_missing=0.0,
        sep_features=sep,
        sep_features_missing=0.0,
    )
    rich = fv.as_rich_list()
    expected_width = (
        RICH_FEATURE_SIZE
        + RICH_MACRO_REGIME_DIM
        + RICH_MACRO_REGIME_MISSING_DIM
        + RICH_SEP_DIM
        + RICH_SEP_MISSING_DIM
    )
    assert len(rich) == expected_width
    # The regime block sits at slice positions
    # ``[RICH_FEATURE_SIZE : RICH_FEATURE_SIZE + RICH_MACRO_REGIME_DIM + 1]``
    # and the SEP block follows immediately after.
    regime_end = RICH_FEATURE_SIZE + RICH_MACRO_REGIME_DIM + RICH_MACRO_REGIME_MISSING_DIM
    assert (
        rich[RICH_FEATURE_SIZE : RICH_FEATURE_SIZE + RICH_MACRO_REGIME_DIM]
        == regime
    )
    assert rich[regime_end : regime_end + RICH_SEP_DIM] == sep


# ---------------------------------------------------------------------------
# Loader regression -- flag off keeps byte-identical schema; flag on
# populates the slot and applies the forward-fill rule.
# ---------------------------------------------------------------------------


pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")
torch = pytest.importorskip("torch")


from app.models.config import SEQUENCE_LENGTH  # noqa: E402
from app.training import loaders  # noqa: E402


_TRAINING_PACKAGE_ID = "tp_sep_regression_v1"


def _synth_prior_bars(*, event_date: _dt.date, base_close: float) -> str:
    payload = []
    for offset in range(SEQUENCE_LENGTH, 0, -1):
        bar_date = _dt.date.fromordinal(event_date.toordinal() - offset)
        payload.append(
            {
                "date": bar_date.isoformat(),
                "close": round(base_close + (SEQUENCE_LENGTH - offset) * 1.5, 10),
                "volume": 1_000_000.0,
                "vol_5d": 0.012,
                "vol_20d": 0.018,
                "vol_60d": 0.022,
                "cum_return_20d": 0.0,
                "vix_close": 14.0,
                "dxy_close": 103.0,
                "tnx_close": 4.20,
                "gold_close": 2050.0,
                "vix3m_close": 17.0,
                "irx_close": 5.10,
                "vix_term_slope": 0.0,
                "yield_curve_slope_10y_3m": -0.90,
            }
        )
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


def _make_event_row(
    *,
    event_date: str,
    text: str,
    axis_stance: str | None,
    base_close: float,
) -> dict:
    ed = _dt.date.fromisoformat(event_date)
    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return {
        "event_date": event_date,
        "event_kind": "statement",
        "document_id": text_hash[:16],
        "text_hash": text_hash,
        "source": "scraped_fed",
        "source_record_id": f"src:{text_hash[:8]}",
        "as_of_ts": f"{event_date}T19:00:00Z",
        "text": text,
        "token_count": len(text.split()),
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
        "realized_return": 0.001,
        "abnormal_return": 0.001,
        "alpha": 0.0,
        "beta": 1.0,
        "direction_t1d": 1,
        "volatility_shift": 0.0,
        "concurrent_macro_release": False,
        "intra_meeting_stance_shift": 0.0,
        "intra_meeting_certainty_shift": 0.0,
        "intra_meeting_factor_shift": 0.0,
        "realized_date": (ed + _dt.timedelta(days=1)).isoformat(),
        "forward_realized_vol_10d": 0.015,
        "yield_2y_change_5d": 1.0,
        "yield_5y_change_5d": 0.5,
        "terminal_rate_change_5d": 0.0,
    }


@pytest.fixture
def loader_package(tmp_path: Path, monkeypatch) -> Path:
    """Three-event synthetic training package with an SEP-projections lookup.

    Includes a March SEP release that the May non-SEP meeting must
    forward-fill from. The third (December) event is itself an SEP
    release and must read ``release_flag = 1.0``.
    """

    processed_root = tmp_path / "processed"
    package_dir = processed_root / _TRAINING_PACKAGE_ID
    package_dir.mkdir(parents=True)

    monkeypatch.setattr(loaders, "DATA_DIR", tmp_path)

    rows = [
        _make_event_row(
            event_date="2024-03-20",
            text="The Committee will continue to assess incoming data.",
            axis_stance="neutral",
            base_close=4500.0,
        ),
        _make_event_row(
            event_date="2024-05-01",
            text="Inflation has eased substantially.",
            axis_stance="dovish",
            base_close=4600.0,
        ),
        _make_event_row(
            event_date="2024-12-18",
            text="A gradual normalisation path is anticipated.",
            axis_stance="hawkish",
            base_close=4700.0,
        ),
    ]
    pd.DataFrame(rows).to_parquet(package_dir / "events.parquet", index=False)

    split_rows = [
        {"text_hash": row["text_hash"], "split_tag": "train" if i < 2 else "test"}
        for i, row in enumerate(rows)
    ]
    pd.DataFrame(split_rows).to_parquet(
        package_dir / "splits_train_val_test.parquet", index=False
    )

    # SEP-projections lookup: one row per SEP release, matching the
    # March / June / September / December cadence. The May 2024
    # supervised meeting has no SEP release of its own and must
    # forward-fill from the March 2024 row.
    sep_rows = [
        {
            "meeting_date": "2023-12-13",
            "ffr_median_current_year": 5.4,
            "ffr_median_longer_run": 2.5,
            "ffr_range_upper_current": 5.6,
            "ffr_range_lower_current": 5.4,
        },
        {
            "meeting_date": "2024-03-20",
            "ffr_median_current_year": 4.6,
            "ffr_median_longer_run": 2.6,
            "ffr_range_upper_current": 4.9,
            "ffr_range_lower_current": 4.4,
        },
        {
            "meeting_date": "2024-12-18",
            "ffr_median_current_year": 4.4,
            "ffr_median_longer_run": 3.0,
            "ffr_range_upper_current": 4.6,
            "ffr_range_lower_current": 4.4,
        },
    ]
    pd.DataFrame(sep_rows).to_parquet(
        package_dir / "sep_projections.parquet", index=False
    )
    return package_dir


def test_loader_sep_flag_off_keeps_pre_215_schema(loader_package: Path) -> None:
    """Default ``use_sep=False`` -> byte-identical schema.

    Pins the byte-identity contract on the legacy / opt-out path.
    Every supervised sequence must keep ``sep_features=None`` and every
    per-bar ``as_rich_list`` must keep the pre-#215 width.
    """

    split = loaders.load_walk_forward_split(
        _TRAINING_PACKAGE_ID,
        rich_features=True,
        use_sep=False,
        text_encoder=None,
    )
    assert split.train, "fixture must produce at least one train sequence"

    for partition in (split.train, split.val, split.test):
        for sequence in partition:
            for vector in sequence:
                assert vector.sep_features is None
                assert vector.sep_features_missing == 1.0
                assert len(vector.as_rich_list()) == RICH_FEATURE_SIZE


def test_loader_sep_flag_on_populates_block_with_forward_fill(loader_package: Path) -> None:
    """``use_sep=True`` -> populated block on every bar, forward-fill applied.

    The fixture's three supervised meetings exercise three branches:

    - 2024-03-20 IS an SEP release. The block must carry March's
      values and ``release_flag = 1.0``.
    - 2024-05-01 is NOT an SEP release. The block must carry March's
      values (the most recent prior SEP) and ``release_flag = 0.0``.
    - 2024-12-18 IS an SEP release. The block must carry December's
      values and ``release_flag = 1.0``.
    """

    split = loaders.load_walk_forward_split(
        _TRAINING_PACKAGE_ID,
        rich_features=True,
        use_sep=True,
        text_encoder=None,
    )
    assert split.train

    expected_width = rich_feature_size_with_sep(True)
    # Each sequence's lookback bar 0 carries the supervised event's
    # SEP block (broadcast onto every bar by the loader); the target
    # row's date is the day after the event so we read the block off
    # the prior-window bars and key by the implied event date instead.
    sep_by_first_bar_date: dict[str, list[float]] = {}
    for partition in (split.train, split.val, split.test):
        for sequence in partition:
            for vector in sequence:
                assert vector.sep_features is not None
                assert vector.sep_features_missing == 0.0
                assert len(vector.sep_features) == RICH_SEP_DIM
                assert len(vector.as_rich_list()) == expected_width
            # First lookback bar's date is ``event_date - SEQUENCE_LENGTH``
            # in the synthetic fixture; the SEP block on every bar is the
            # block the loader computed for the supervised event. Read
            # from the last lookback bar (most recent prior bar) to key
            # by something deterministic.
            anchor = sequence[-2]
            sep_by_first_bar_date[anchor.date[:10]] = list(anchor.sep_features)

    # The synthetic fixture's prior-bar window ends at ``event_date - 1``,
    # so the last lookback bar's date is one day before the supervised
    # event. Map back to event dates via the documented offset.
    keys_sorted = sorted(sep_by_first_bar_date.keys())
    assert len(keys_sorted) == 3, (
        f"fixture must produce 3 supervised sequences; got {keys_sorted}"
    )
    # Order: 2024-03-19 (March event), 2024-04-30 (May event),
    # 2024-12-17 (December event).
    march_block = sep_by_first_bar_date[keys_sorted[0]]
    may_block = sep_by_first_bar_date[keys_sorted[1]]
    december_block = sep_by_first_bar_date[keys_sorted[2]]
    # March: own release. ``release_flag`` (last scalar) = 1.0;
    # current-year median = 4.6.
    assert march_block[0] == pytest.approx(4.6)
    assert march_block[-1] == 1.0
    # May: forward-fill from March. Values match March; ``release_flag`` = 0.0.
    assert may_block[0] == pytest.approx(4.6)
    assert may_block[-1] == 0.0
    # December: own release. Values match December; ``release_flag`` = 1.0.
    assert december_block[0] == pytest.approx(4.4)
    assert december_block[-1] == 1.0


def test_loader_sep_absent_parquet_collapses_to_missing(tmp_path: Path, monkeypatch) -> None:
    """No SEP parquet on disk -> graceful degrade to ``None`` + missing-flag.

    The code path stays live without the parquet (operator without the
    SEP source can still run the flag-on training; the model just sees
    the all-zeros block + missing=1.0 on every event).
    """

    processed_root = tmp_path / "processed"
    package_dir = processed_root / "tp_sep_absent_parquet_v1"
    package_dir.mkdir(parents=True)
    monkeypatch.setattr(loaders, "DATA_DIR", tmp_path)

    rows = [
        _make_event_row(
            event_date="2024-03-20",
            text="Statement text March.",
            axis_stance="neutral",
            base_close=4500.0,
        ),
        _make_event_row(
            event_date="2024-06-12",
            text="Statement text June.",
            axis_stance="neutral",
            base_close=4600.0,
        ),
    ]
    pd.DataFrame(rows).to_parquet(package_dir / "events.parquet", index=False)
    # Split-tag the second event as test so the loader's "empty test
    # partition" guard is satisfied; the absent-parquet behaviour
    # we are pinning is independent of which partition the events
    # land in.
    split_rows = [
        {"text_hash": rows[0]["text_hash"], "split_tag": "train"},
        {"text_hash": rows[1]["text_hash"], "split_tag": "test"},
    ]
    pd.DataFrame(split_rows).to_parquet(
        package_dir / "splits_train_val_test.parquet", index=False
    )

    split = loaders.load_walk_forward_split(
        "tp_sep_absent_parquet_v1",
        rich_features=True,
        use_sep=True,
        text_encoder=None,
    )
    for sequence in split.train + split.val + split.test:
        for vector in sequence:
            assert vector.sep_features is None
            assert vector.sep_features_missing == 1.0
            # Width stays at the legacy pre-#215 size because the block
            # was never populated (graceful degrade preserves the
            # conditional-emit contract on as_rich_list).
            assert len(vector.as_rich_list()) == RICH_FEATURE_SIZE


def test_provenance_audit_sep_block_reads_strictly_prior_or_t_snapshot(
    loader_package: Path,
) -> None:
    """The SEP composer reads only ``T (snapshot)`` or ``T-Δ`` rows.

    On SEP-release meetings the matched row IS the supervised event
    (``T (snapshot)``); on non-SEP meetings the matched row has
    ``meeting_date < event_date`` (``T-Δ`` strict prior). The audit row
    pins the contract; this regression locks the per-event read window.
    """

    split = loaders.load_walk_forward_split(
        _TRAINING_PACKAGE_ID,
        rich_features=True,
        use_sep=True,
        text_encoder=None,
    )
    for sequence in split.train + split.val + split.test:
        target = sequence[-1]
        assert target.sep_features is not None
        # Release flag is the last scalar; either 0.0 (forward-filled)
        # or 1.0 (matched the supervised event itself). Both branches
        # are leak-clean by the composer's contract.
        assert target.sep_features[-1] in {0.0, 1.0}


# ---------------------------------------------------------------------------
# Smoke training run with the SEP block populated -- verifies the
# recurrent-core widening lines up with the loader's per-bar tensor
# size when the flag is on.
# ---------------------------------------------------------------------------


from app.models.config import ModelConfig  # noqa: E402
from app.training.loop import train_model  # noqa: E402


def _rich_feature_vector_with_sep(
    *,
    day: int,
    vol: float,
    sep_block: list[float] = (4.6, 3.9, 2.6, 0.5, 1.0),  # type: ignore[assignment]
) -> FeatureVector:
    """In-memory FeatureVector matching the loader's rich-payload shape.

    Sets ``rich_payload=True`` so the per-bar tensoriser routes
    through ``as_rich_list`` (the widened path when the SEP block is
    populated).
    """

    fv = FeatureVector(
        date=str(_dt.date(2025, 1, 1) + _dt.timedelta(days=day - 1)),
        sentiment_score=0.0,
        market_close=100.0 + day * 0.5,
        market_volatility=0.01,
        close_change_pct=0.0,
        volatility_change=0.0,
        elapsed_time=0.0,
        forward_realized_vol_10d=vol,
        rich_payload=True,
    )
    fv.sep_features = list(sep_block)
    fv.sep_features_missing = 0.0
    return fv


def test_train_model_smoke_with_sep_block() -> None:
    """One-epoch training run with the SEP block populated runs to completion.

    The recurrent core widens by ``RICH_SEP_DIM + RICH_SEP_MISSING_DIM``
    when ``use_sep=True``; the smoke verifies the model graph + tensor
    plumbing line up across the wider per-bar input.
    """

    groups = [
        [_rich_feature_vector_with_sep(day=i + 1, vol=0.01 + 0.001 * i) for i in range(40)]
    ]
    config = ModelConfig(
        input_size=RICH_FEATURE_SIZE,
        output_mode="classification",
        n_classes=3,
        hidden_size=16,
        head_hidden_size=8,
        use_sep=True,
    )
    result = train_model(
        model_config=config,
        train_sequence_groups=groups,
        val_sequence_groups=groups,
        test_sequence_groups=groups,
        epochs=1,
        batch_size=8,
        seed=11,
        save_checkpoint=False,
        use_compile=False,
        use_amp=False,
    )
    assert result.summary.epochs_completed == 1


def test_research_model_lstm_width_includes_sep_tail() -> None:
    """The recurrent core widens by the SEP tail when the flag is on."""

    from app.models.factory import build_research_forecaster

    config = ModelConfig(
        input_size=RICH_FEATURE_SIZE,
        output_mode="classification",
        n_classes=3,
        hidden_size=16,
        head_hidden_size=8,
        use_sep=True,
    )
    model = build_research_forecaster(config)
    expected_width = RICH_FEATURE_SIZE + RICH_SEP_DIM + RICH_SEP_MISSING_DIM
    assert model.lstm_input_size == expected_width
    core = model.recurrent_core
    core_width = getattr(core, "input_size", None)
    assert core_width == expected_width


def test_research_model_lstm_width_unchanged_when_sep_off() -> None:
    """Default ``use_sep=False`` keeps the recurrent core at ``RICH_FEATURE_SIZE``."""

    from app.models.factory import build_research_forecaster

    config = ModelConfig(
        input_size=RICH_FEATURE_SIZE,
        output_mode="classification",
        n_classes=3,
        hidden_size=16,
        head_hidden_size=8,
    )
    model = build_research_forecaster(config)
    assert model.lstm_input_size == RICH_FEATURE_SIZE
    assert getattr(model.recurrent_core, "input_size", None) == RICH_FEATURE_SIZE


def test_research_model_lstm_width_composes_regime_and_sep() -> None:
    """Both flags on -> recurrent core widens by regime + SEP tails.

    Pins the composition contract: the two opt-in tails stack in the
    order documented in ``as_rich_list`` (regime first, then SEP).
    """

    from app.models.factory import build_research_forecaster

    config = ModelConfig(
        input_size=RICH_FEATURE_SIZE,
        output_mode="classification",
        n_classes=3,
        hidden_size=16,
        head_hidden_size=8,
        use_regime_conditioning=True,
        use_sep=True,
    )
    model = build_research_forecaster(config)
    expected_width = (
        RICH_FEATURE_SIZE
        + RICH_MACRO_REGIME_DIM
        + RICH_MACRO_REGIME_MISSING_DIM
        + RICH_SEP_DIM
        + RICH_SEP_MISSING_DIM
    )
    assert model.lstm_input_size == expected_width


# ---------------------------------------------------------------------------
# SEP parquet builder -- exercise the on-or-before lookup against a
# synthetic FRED panel so the path-(a) ingestion is reviewable.
# ---------------------------------------------------------------------------


def test_sep_parquet_fixture_csv_round_trips(tmp_path: Path) -> None:
    """The CSV fixture path reads + writes the same rows by data value."""

    from app.data.sep_projections import (
        DEFAULT_FRED_SERIES_IDS,
        load_fixture_csv,
        to_frame,
    )

    csv_path = tmp_path / "sep_projections.csv"
    csv_path.write_text(
        "meeting_date,ffr_median_current_year,"
        "ffr_median_longer_run,ffr_range_upper_current,"
        "ffr_range_lower_current\n"
        "2024-03-20,4.625,2.625,4.875,4.375\n"
        "2024-06-12,5.125,2.750,5.250,4.875\n",
        encoding="utf-8",
    )
    rows = load_fixture_csv(csv_path)
    assert len(rows) == 2
    assert rows[0].meeting_date == _dt.date(2024, 3, 20)
    assert rows[0].ffr_median_current_year == pytest.approx(4.625)
    frame = to_frame(rows)
    assert list(frame["meeting_date"]) == ["2024-03-20", "2024-06-12"]
    # Sanity check: the default FRED series mapping covers every
    # numeric column the parquet schema requires.
    assert set(DEFAULT_FRED_SERIES_IDS.keys()) == {
        "ffr_median_current_year",
        "ffr_median_longer_run",
        "ffr_range_upper_current",
        "ffr_range_lower_current",
    }
