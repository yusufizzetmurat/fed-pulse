"""Tests for the workspace futures-consensus serving wrapper.

The serving wrapper translates short-end DGS FRED observations into
the descriptive panel's wire response. The probability bucketing math
and the sign convention are the core invariants — the FRED layer is
stubbed via the ``fetch_dgs`` / ``fetch_target`` keyword seams so the
tests do not touch ``httpx`` or the on-disk cache.
"""

from __future__ import annotations

from datetime import date

import pytest

from app.services import futures_consensus
from app.services.fred_client import FredObservation, FredSeriesResponse


def _series(series_id: str, value_pct: float) -> FredSeriesResponse:
    """One-observation FRED response in the units the parser emits.

    Values are percent (e.g. ``5.25`` for 5.25%) to match the real
    FRED schema; the consensus builder multiplies by 100 to move
    into basis points.
    """

    return FredSeriesResponse(
        series_id=series_id,
        realtime_start="2026-05-15",
        realtime_end="2026-05-15",
        observation_start="2026-05-10",
        observation_end="2026-05-15",
        count=1,
        observations=[
            FredObservation(
                date="2026-05-15",
                value=value_pct,
                realtime_start="2026-05-15",
                realtime_end="2026-05-15",
            ),
        ],
    )


def _multi_obs_series(
    series_id: str,
    rows: list[tuple[str, float | None]],
) -> FredSeriesResponse:
    return FredSeriesResponse(
        series_id=series_id,
        realtime_start="2026-05-15",
        realtime_end="2026-05-15",
        observation_start=rows[0][0],
        observation_end=rows[-1][0],
        count=len(rows),
        observations=[
            FredObservation(
                date=row_date,
                value=row_value,
                realtime_start="2026-05-15",
                realtime_end="2026-05-15",
            )
            for row_date, row_value in rows
        ],
    )


def _stub_factories(dgs_pct: dict[str, float], lower_pct: float, upper_pct: float):
    def fetch_dgs(*, cache_dir=None):
        return {sid: _series(sid, pct) for sid, pct in dgs_pct.items()}

    def fetch_target(series_id: str, *, cache_dir=None):
        if series_id == futures_consensus.TARGET_LOWER_SERIES:
            return _series(series_id, lower_pct)
        if series_id == futures_consensus.TARGET_UPPER_SERIES:
            return _series(series_id, upper_pct)
        raise AssertionError(f"unexpected series {series_id}")

    return fetch_dgs, fetch_target


def test_probabilities_sum_to_one_at_each_horizon() -> None:
    fetch_dgs, fetch_target = _stub_factories(
        dgs_pct={"DGS1MO": 5.33, "DGS3MO": 5.40, "DGS6MO": 5.20},
        lower_pct=5.25,
        upper_pct=5.50,
    )

    out = futures_consensus.get_consensus(
        date(2026, 4, 1),
        fetch_dgs=fetch_dgs,
        fetch_target=fetch_target,
    )

    assert len(out.horizons) == 3
    for horizon in out.horizons:
        total = (
            horizon.probability_hike
            + horizon.probability_cut
            + horizon.probability_pause
        )
        assert total == pytest.approx(1.0, abs=1e-9)
        assert 0.0 <= horizon.probability_hike <= 1.0
        assert 0.0 <= horizon.probability_cut <= 1.0
        assert 0.0 <= horizon.probability_pause <= 1.0


def test_hawkish_curve_above_target_yields_hike_bias() -> None:
    # DGS3MO at 5.80% vs target band 5.25-5.50 (midpoint 5.375%) implies
    # roughly +42.5 bps of policy tightening at the 3-month horizon —
    # well above the 25 bps hike threshold and 12.5 bps sigma.
    fetch_dgs, fetch_target = _stub_factories(
        dgs_pct={"DGS1MO": 5.40, "DGS3MO": 5.80, "DGS6MO": 5.90},
        lower_pct=5.25,
        upper_pct=5.50,
    )

    out = futures_consensus.get_consensus(
        date(2026, 4, 1),
        fetch_dgs=fetch_dgs,
        fetch_target=fetch_target,
    )

    h3 = next(h for h in out.horizons if h.horizon_label == "3m")
    assert h3.change_vs_current_bps > 0
    assert h3.probability_hike > 0.9
    assert h3.probability_cut < 0.01


def test_dovish_curve_below_target_yields_cut_bias() -> None:
    # DGS6MO at 4.80% vs target midpoint 5.375% implies -57.5 bps,
    # comfortably past the -25 bps cut threshold.
    fetch_dgs, fetch_target = _stub_factories(
        dgs_pct={"DGS1MO": 5.30, "DGS3MO": 5.10, "DGS6MO": 4.80},
        lower_pct=5.25,
        upper_pct=5.50,
    )

    out = futures_consensus.get_consensus(
        date(2026, 4, 1),
        fetch_dgs=fetch_dgs,
        fetch_target=fetch_target,
    )

    h6 = next(h for h in out.horizons if h.horizon_label == "6m")
    assert h6.change_vs_current_bps < 0
    assert h6.probability_cut > 0.9
    assert h6.probability_hike < 0.01


def test_curve_on_target_yields_pause_bias() -> None:
    # All three tenors exactly at the midpoint -> implied change is
    # zero. With sigma=12.5 and threshold=25 the pause probability
    # works out to Phi(2) - Phi(-2) ~= 0.9545.
    fetch_dgs, fetch_target = _stub_factories(
        dgs_pct={"DGS1MO": 5.375, "DGS3MO": 5.375, "DGS6MO": 5.375},
        lower_pct=5.25,
        upper_pct=5.50,
    )

    out = futures_consensus.get_consensus(
        date(2026, 4, 1),
        fetch_dgs=fetch_dgs,
        fetch_target=fetch_target,
    )

    for horizon in out.horizons:
        assert horizon.change_vs_current_bps == pytest.approx(0.0, abs=1e-9)
        assert horizon.probability_pause > 0.9
        assert horizon.probability_hike == pytest.approx(
            horizon.probability_cut, abs=1e-9
        )


def test_implied_rate_uses_percent_to_bps_conversion() -> None:
    fetch_dgs, fetch_target = _stub_factories(
        dgs_pct={"DGS1MO": 5.33, "DGS3MO": 5.40, "DGS6MO": 5.20},
        lower_pct=5.25,
        upper_pct=5.50,
    )

    out = futures_consensus.get_consensus(
        date(2026, 4, 1),
        fetch_dgs=fetch_dgs,
        fetch_target=fetch_target,
    )

    h1 = next(h for h in out.horizons if h.horizon_label == "1m")
    assert h1.implied_rate_bps == pytest.approx(533.0)
    # Target midpoint = (525 + 550) / 2 = 537.5 -> change = -4.5 bps.
    assert h1.change_vs_current_bps == pytest.approx(-4.5)
    assert out.current_target_lo_bps == pytest.approx(525.0)
    assert out.current_target_hi_bps == pytest.approx(550.0)


def test_response_carries_methodology_and_data_source() -> None:
    fetch_dgs, fetch_target = _stub_factories(
        dgs_pct={"DGS1MO": 5.33, "DGS3MO": 5.40, "DGS6MO": 5.20},
        lower_pct=5.25,
        upper_pct=5.50,
    )

    out = futures_consensus.get_consensus(
        date(2026, 4, 1),
        fetch_dgs=fetch_dgs,
        fetch_target=fetch_target,
    )

    assert out.data_source == "FRED"
    assert "Treasury constant-maturity proxy" in out.methodology
    assert "OIS-clean" in out.methodology


def test_meeting_date_aligns_with_next_scheduled_fomc() -> None:
    fetch_dgs, fetch_target = _stub_factories(
        dgs_pct={"DGS1MO": 5.33, "DGS3MO": 5.40, "DGS6MO": 5.20},
        lower_pct=5.25,
        upper_pct=5.50,
    )

    # April 1, 2026 -> next scheduled FOMC is 2026-04-28 per the
    # bundled calendar in app.services.fomc_calendar.
    out = futures_consensus.get_consensus(
        date(2026, 4, 1),
        fetch_dgs=fetch_dgs,
        fetch_target=fetch_target,
    )
    assert out.meeting_date == "2026-04-28"


def test_latest_observation_skips_trailing_nulls() -> None:
    # Saturday / Sunday observations come back as ``None`` (FRED's
    # literal ``.``). The latest-observation walker should skip them
    # and pick the most recent real number.
    def fetch_dgs(*, cache_dir=None):
        return {
            "DGS1MO": _multi_obs_series(
                "DGS1MO",
                [("2026-05-13", 5.33), ("2026-05-14", None), ("2026-05-15", None)],
            ),
            "DGS3MO": _series("DGS3MO", 5.40),
            "DGS6MO": _series("DGS6MO", 5.20),
        }

    def fetch_target(series_id: str, *, cache_dir=None):
        return _series(
            series_id,
            5.25 if series_id == futures_consensus.TARGET_LOWER_SERIES else 5.50,
        )

    out = futures_consensus.get_consensus(
        date(2026, 4, 1),
        fetch_dgs=fetch_dgs,
        fetch_target=fetch_target,
    )

    h1 = next(h for h in out.horizons if h.horizon_label == "1m")
    assert h1.implied_rate_bps == pytest.approx(533.0)


def test_missing_dgs_tenor_raises_unavailable() -> None:
    def fetch_dgs(*, cache_dir=None):
        # Drop DGS6MO entirely -- the builder should refuse to assemble
        # a partial response rather than silently render two columns.
        return {
            "DGS1MO": _series("DGS1MO", 5.33),
            "DGS3MO": _series("DGS3MO", 5.40),
        }

    def fetch_target(series_id: str, *, cache_dir=None):
        return _series(
            series_id,
            5.25 if series_id == futures_consensus.TARGET_LOWER_SERIES else 5.50,
        )

    with pytest.raises(futures_consensus.FuturesConsensusUnavailable):
        futures_consensus.get_consensus(
            date(2026, 4, 1),
            fetch_dgs=fetch_dgs,
            fetch_target=fetch_target,
        )


def test_network_error_in_fetch_raises_unavailable() -> None:
    def fetch_dgs(*, cache_dir=None):
        raise RuntimeError("FRED 503")

    def fetch_target(series_id: str, *, cache_dir=None):
        return _series(series_id, 5.25)

    with pytest.raises(futures_consensus.FuturesConsensusUnavailable):
        futures_consensus.get_consensus(
            date(2026, 4, 1),
            fetch_dgs=fetch_dgs,
            fetch_target=fetch_target,
        )


def test_calendar_with_no_upcoming_meeting_raises_unavailable() -> None:
    # An as-of date past the bundled calendar's last scheduled meeting
    # has no upcoming row to anchor the panel header. The builder
    # should fail loudly so the API can return 503.
    fetch_dgs, fetch_target = _stub_factories(
        dgs_pct={"DGS1MO": 5.33, "DGS3MO": 5.40, "DGS6MO": 5.20},
        lower_pct=5.25,
        upper_pct=5.50,
    )
    with pytest.raises(futures_consensus.FuturesConsensusUnavailable):
        futures_consensus.get_consensus(
            date(2099, 1, 1),
            fetch_dgs=fetch_dgs,
            fetch_target=fetch_target,
        )


def test_probability_math_directly_matches_normal_cdf() -> None:
    # +25 bps change at sigma 12.5 -> Phi(0) = 0.5 hike probability;
    # cut probability is Phi(-4) ~= 3.17e-5; pause is the residual.
    p_hike, p_cut, p_pause = futures_consensus._hike_cut_pause_probabilities(25.0)
    assert p_hike == pytest.approx(0.5, abs=1e-6)
    assert p_cut < 1e-4
    assert p_pause == pytest.approx(1.0 - p_hike - p_cut, abs=1e-9)


def test_zero_sigma_is_rejected() -> None:
    with pytest.raises(ValueError):
        futures_consensus._hike_cut_pause_probabilities(0.0, sigma_bps=0.0)
