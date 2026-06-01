"""Workspace descriptive panel: fed-funds path consensus via DGS proxy.

The frontend's *FRED futures consensus* panel renders three horizon
columns (1-month / 3-month / 6-month) showing the level the short-end
of the Treasury curve is pricing in at each tenor, the change versus
the current target band midpoint, and a stacked hike / cut / pause
probability bar.

Methodology rationale.
    True fed-funds futures (CME ZQ contracts) are the cleanest read of
    expected policy, but the public dataset access path used in this
    project is FRED rather than a CME settlements feed. The short-end
    Treasury constant-maturity series ``DGS1MO`` / ``DGS3MO`` /
    ``DGS6MO`` embed a small term premium on top of the expected
    rate path; this proxy is well-known in the rate-surprise
    literature (see e.g. Kuttner 2001 and the BIS quarterly review's
    discussion of short-end term premia under quiet macro regimes).
    Under typical conditions the term premium at these tenors is on
    the order of a few basis points; treating the level as a
    *direction* indicator is sound, but the absolute number should be
    read as a proxy, never as an OIS-clean expectation.

Probability bucketing.
    The change vs. the current target midpoint is mapped to a
    discrete ``{cut, pause, hike}`` distribution via a normal CDF
    centered on the implied change. The probability of a 25 bp hike
    is ``Phi((change - 25) / sigma)``; the probability of a 25 bp cut
    is ``Phi((-25 - change) / sigma)`` (i.e. the lower tail at the
    negative-25 threshold); pause is the residual mass between the
    two thresholds. ``sigma = 12.5`` bps approximates the standard
    deviation of the daily change in the short Treasury yield around
    pre-meeting windows and keeps the bar from collapsing to a
    degenerate {0, 1} on small misses.

Sign convention.
    Positive ``change_vs_current_bps`` is hawkish (curve pricing in a
    higher target). Negative is dovish. The current target midpoint is
    read from the existing FRED ``DFEDTARL`` / ``DFEDTARU`` series so
    the panel agrees with the rest of the rate UI.

The Workspace spine renders this under ``WorkspaceSection
variant="descriptive"`` — the panel never feeds the forecast cards
(HAR-tercile, QLIKE-RV, Expected Volume). Callers in
:mod:`app.main` translate :class:`FuturesConsensusUnavailable` into a
``503`` with a structured detail; the front-end then degrades to an
"unavailable" placeholder.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

from app.schemas import FuturesConsensusHorizon, FuturesConsensusResponse
from app.services.fomc_calendar import get_calendar
from app.services.fred_client import (
    DGS_SHORT_SERIES,
    FredSeriesResponse,
    fetch_dgs_short,
    fetch_fred_series,
)


# Horizon label rendered in the panel header for each DGS tenor. Tuple
# ordering mirrors :data:`DGS_SHORT_SERIES` so the panel renders 1m /
# 3m / 6m left-to-right.
HORIZON_LABELS: dict[str, str] = {
    "DGS1MO": "1m",
    "DGS3MO": "3m",
    "DGS6MO": "6m",
}

# Hike / cut threshold in basis points. The Fed has moved in 25 bp
# increments throughout the dataset window (with a handful of 50 / 75
# exceptions during the 2022-23 tightening cycle); using 25 keeps the
# panel labels readable and matches the discretization the
# mp-surprise module uses for the strict-prior comparison.
HIKE_THRESHOLD_BPS: float = 25.0

# Standard deviation of the implied change in bps. 12.5 bps -- one
# half of the hike threshold -- is a deliberately mild prior on noise
# in the short Treasury proxy; it keeps the probability bar from
# pinning to {0, 1} on a small miss while still moving meaningfully
# when the curve has clearly repriced.
DIRECTION_SIGMA_BPS: float = 12.5

# FRED target-band series. ``DFEDTARL`` is the lower bound of the fed
# funds target range; ``DFEDTARU`` is the upper bound. Values are in
# percent (e.g. 5.25 -> 525 bps).
TARGET_LOWER_SERIES = "DFEDTARL"
TARGET_UPPER_SERIES = "DFEDTARU"

# Methodology footnote shown in the panel's fine print. Single source
# of truth -- the panel renders this verbatim under the columns.
METHODOLOGY_TEXT = (
    "Treasury constant-maturity proxy (DGS1MO/3MO/6MO). Embeds a term "
    "premium; treat as a level proxy, not an OIS-clean expectation."
)

DATA_SOURCE_LABEL = "FRED"


class FuturesConsensusUnavailable(RuntimeError):
    """Raised when FRED is unreachable or the FOMC calendar is empty."""


@dataclass(frozen=True)
class _LatestPoint:
    date: str
    value: float  # percent (e.g. 5.33)


def _latest_observation(response: FredSeriesResponse) -> _LatestPoint:
    """Pick the most recent non-null observation from a FRED response.

    FRED encodes missing days as ``None`` (the parser already coerces
    the literal ``"."``). We walk the observation list in reverse so
    weekends / holidays at the tail do not collapse the panel to
    ``unavailable``.
    """

    for obs in reversed(response.observations):
        if obs.value is None:
            continue
        return _LatestPoint(date=obs.date, value=float(obs.value))
    raise FuturesConsensusUnavailable(
        f"no non-null observation in FRED series {response.series_id!r}"
    )


def _midpoint_bps_from_responses(
    lower: FredSeriesResponse,
    upper: FredSeriesResponse,
) -> tuple[float, float]:
    """Return the (lower_bps, upper_bps) of the current target range.

    Both series are reported in percent on FRED; we multiply by 100 to
    move to basis points (e.g. 5.25% -> 525 bps).
    """

    lo = _latest_observation(lower)
    hi = _latest_observation(upper)
    return lo.value * 100.0, hi.value * 100.0


def _normal_cdf(x: float) -> float:
    """Standard-normal CDF via :func:`math.erf`; pure stdlib."""

    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _hike_cut_pause_probabilities(
    change_bps: float,
    *,
    threshold_bps: float = HIKE_THRESHOLD_BPS,
    sigma_bps: float = DIRECTION_SIGMA_BPS,
) -> tuple[float, float, float]:
    """Bucket the implied change into ``(p_hike, p_cut, p_pause)``.

    The implied change is treated as the mean of a normal distribution
    with standard deviation ``sigma_bps``. A hike is defined as the
    target rising by at least ``threshold_bps``; a cut is the
    symmetric negative move. The pause probability is the residual
    mass between the two thresholds, clamped to ``[0, 1]`` to absorb
    floating-point drift on the boundaries.
    """

    if sigma_bps <= 0:
        raise ValueError("sigma_bps must be positive")
    p_hike = 1.0 - _normal_cdf((threshold_bps - change_bps) / sigma_bps)
    p_cut = _normal_cdf((-threshold_bps - change_bps) / sigma_bps)
    p_pause = max(0.0, 1.0 - p_hike - p_cut)
    # Renormalize to defuse a sub-ulp drift before serialization.
    total = p_hike + p_cut + p_pause
    if total <= 0:
        return 0.0, 0.0, 1.0
    return p_hike / total, p_cut / total, p_pause / total


def _next_meeting_date(as_of: date | None) -> date:
    """Return the next scheduled FOMC meeting on or after ``as_of``.

    Re-uses :func:`app.services.fomc_calendar.get_calendar` so the
    panel header agrees with the calendar surface elsewhere in the UI.
    """

    calendar = get_calendar(as_of=as_of, upcoming_limit=1, past_limit=0)
    upcoming = calendar.get("upcoming") or []
    if not upcoming:
        raise FuturesConsensusUnavailable(
            "FOMC calendar has no upcoming meetings on or after the as-of date"
        )
    return upcoming[0].meeting_date


def _build_horizon(
    series_id: str,
    response: FredSeriesResponse,
    *,
    target_midpoint_bps: float,
) -> FuturesConsensusHorizon:
    latest = _latest_observation(response)
    implied_bps = latest.value * 100.0
    change_bps = implied_bps - target_midpoint_bps
    p_hike, p_cut, p_pause = _hike_cut_pause_probabilities(change_bps)
    return FuturesConsensusHorizon(
        horizon_label=HORIZON_LABELS[series_id],
        implied_rate_bps=implied_bps,
        change_vs_current_bps=change_bps,
        probability_hike=p_hike,
        probability_cut=p_cut,
        probability_pause=p_pause,
    )


def get_consensus(
    as_of_date: date | None = None,
    *,
    cache_dir: Path | None = None,
    fetch_dgs: Callable[..., dict[str, "FredSeriesResponse"]] = fetch_dgs_short,
    fetch_target: Callable[..., "FredSeriesResponse"] = fetch_fred_series,
) -> FuturesConsensusResponse:
    """Build the workspace futures-consensus response.

    The ``fetch_dgs`` and ``fetch_target`` seams are kept as keyword
    parameters so the test suite can stub the FRED layer without
    touching ``httpx`` or the on-disk cache. Production callers leave
    them at the default and rely on the existing retry / backoff and
    cache infrastructure inside :mod:`app.services.fred_client`.
    """

    reference = as_of_date or date.today()
    try:
        dgs = fetch_dgs(cache_dir=cache_dir)
        lower = fetch_target(TARGET_LOWER_SERIES, cache_dir=cache_dir)
        upper = fetch_target(TARGET_UPPER_SERIES, cache_dir=cache_dir)
    except FuturesConsensusUnavailable:
        raise
    except Exception as exc:  # network, missing key, cache miss, etc.
        raise FuturesConsensusUnavailable(
            f"FRED data unavailable for futures-consensus panel: {exc}"
        ) from exc

    missing = [sid for sid in DGS_SHORT_SERIES if sid not in dgs]
    if missing:
        raise FuturesConsensusUnavailable(
            f"FRED response missing required DGS tenors: {sorted(missing)}"
        )

    lower_bps, upper_bps = _midpoint_bps_from_responses(lower, upper)
    midpoint_bps = 0.5 * (lower_bps + upper_bps)
    meeting_date = _next_meeting_date(reference)

    horizons = [
        _build_horizon(sid, dgs[sid], target_midpoint_bps=midpoint_bps)
        for sid in DGS_SHORT_SERIES
    ]

    generated_at = datetime.now(timezone.utc).isoformat()

    return FuturesConsensusResponse(
        meeting_date=meeting_date.isoformat(),
        generated_at=generated_at,
        current_target_lo_bps=lower_bps,
        current_target_hi_bps=upper_bps,
        horizons=horizons,
        methodology=METHODOLOGY_TEXT,
        data_source=DATA_SOURCE_LABEL,
    )


__all__ = [
    "DATA_SOURCE_LABEL",
    "DIRECTION_SIGMA_BPS",
    "FuturesConsensusUnavailable",
    "HIKE_THRESHOLD_BPS",
    "HORIZON_LABELS",
    "METHODOLOGY_TEXT",
    "TARGET_LOWER_SERIES",
    "TARGET_UPPER_SERIES",
    "get_consensus",
]
