"""Macro-regime conditioning features for the forecaster (#307).

Three strict-prior scalar indicators per event:

- ``policy_cycle_phase_score`` -- {-1, 0, +1} signed direction of the
  realised fed-funds target over the prior twelve months. Hiking,
  cutting, or extended-pause regime. Derived from the strict-prior
  meetings on the MP-surprise lookup; reads ``ff_target_prior`` (the
  band midpoint observable strictly before each prior meeting), so the
  whole quantity is strictly prior to ``event_date`` by construction.

- ``vix_level_regime_score`` -- {-1, 0, +1} tertile bucket of the T-1
  VIX close against the trailing 20-bar window of VIX values that are
  already in the supervised sequence's prior bars. The 20-bar window
  is strictly before ``event_date`` (daily-bar resolution; no T-0
  reading) so the tertile cutoffs are themselves strict-prior.

- ``term_spread_sign`` -- {-1, 0, +1} sign of the 10y-3m yield-curve
  slope at T-1 (last prior bar). Reads ``tnx_close`` (^TNX, 10y) and
  ``irx_close`` (^IRX, 13-week T-bill) off the last prior bar of the
  event's supervised sequence.

The block is opt-in via ``--use-regime-conditioning`` on
``app.train_forecaster``. When the flag is off, the loader leaves the
``macro_regime_features`` slot ``None`` on every event and
``FeatureVector.as_rich_list`` does NOT append the block, so the
default per-bar feature size is byte-identical to pre-#307.

Strict-prior contract is documented per-feature in
``docs/feature-provenance-audit.md`` and exercised by
``tests/unit/test_macro_regime_features.py``.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


# Threshold (in basis points) above / below which the rolling 12-month
# fed-funds target change scores as a hike / cut regime. Chosen as a
# single quarter-point move so any single 25-bp action in the trailing
# year is enough to flip the regime off "holding". Documented in
# ADR 0029 alongside the policy-cycle-phase derivation.
POLICY_CYCLE_THRESHOLD_BPS: float = 25.0

# Trailing window length (in calendar days) for the rolling
# policy-cycle phase. Twelve months is the literature default for
# "current cycle phase"; matches the post-#350 strict-prior trailing
# horizons used elsewhere in the codebase.
POLICY_CYCLE_LOOKBACK_DAYS: int = 365

# Tertile cutoffs the VIX bucket reads off the trailing prior-bar
# window. Lower tertile -> low-vol regime; upper tertile -> high-vol
# regime; middle -> normal.
VIX_LOW_TERTILE: float = 1.0 / 3.0
VIX_HIGH_TERTILE: float = 2.0 / 3.0

# Number of derived regime scalars emitted per event. Re-exported as
# ``RICH_MACRO_REGIME_DIM`` from ``app.models.config`` so the schema
# constants stay one source of truth; this alias gives the helper a
# local name.
REGIME_FEATURE_DIM: int = 3


@dataclass(frozen=True)
class MacroRegimeFeatures:
    """Per-event macro-regime indicator scalars.

    Three signed scalars in {-1.0, 0.0, +1.0}; the loader writes them
    into ``FeatureVector.macro_regime_features`` in the documented
    order, and ``as_rich_list`` appends the block past the legacy
    ``RICH_FEATURE_SIZE`` width when the macro_regime slot is populated.
    """

    policy_cycle_phase_score: float
    vix_level_regime_score: float
    term_spread_sign: float

    def as_list(self) -> list[float]:
        return [
            float(self.policy_cycle_phase_score),
            float(self.vix_level_regime_score),
            float(self.term_spread_sign),
        ]


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out:  # NaN guard
        return None
    return out


def compute_policy_cycle_phase_score(
    *,
    event_date: datetime.date,
    mp_surprise_lookup: Mapping[str, Mapping[str, Any]],
    lookback_days: int = POLICY_CYCLE_LOOKBACK_DAYS,
    threshold_bps: float = POLICY_CYCLE_THRESHOLD_BPS,
) -> float:
    """Score the realised fed-funds path over the strict-prior trailing window.

    Walks the MP-surprise lookup for every meeting whose ``event_date``
    sits strictly before the supervised event and within
    ``lookback_days`` calendar days. The window's prior-band reading is
    the first eligible meeting's ``ff_target_prior``; the window's
    latest-band reading is the most recent eligible meeting's
    ``ff_target_prior`` (the band published the day before each
    meeting's announcement; every read is strictly prior to
    ``event_date`` by construction since prior meetings themselves
    pre-date the supervised event).

    The score is ``+1.0`` when ``(latest - earliest) * 100 >= threshold_bps``
    (hiking), ``-1.0`` when the change is ``<= -threshold_bps``
    (cutting), and ``0.0`` otherwise (extended pause / holding).

    Returns ``0.0`` when the trailing window carries fewer than two
    eligible meetings (cold-start contract; the "no signal" default
    collapses to the holding regime so an event at the start of the
    corpus is not misclassified as a regime change).
    """

    if not mp_surprise_lookup:
        return 0.0
    window_start = event_date - datetime.timedelta(days=int(lookback_days))
    eligible: list[tuple[datetime.date, float]] = []
    for date_str, payload in mp_surprise_lookup.items():
        try:
            meeting_date = datetime.date.fromisoformat(str(date_str)[:10])
        except ValueError:
            continue
        if meeting_date >= event_date:
            # Strict-prior contract: the supervised event itself and
            # any later meeting are excluded.
            continue
        if meeting_date < window_start:
            continue
        target_prior = _coerce_float(payload.get("ff_target_prior"))
        if target_prior is None:
            continue
        eligible.append((meeting_date, target_prior))
    if len(eligible) < 2:
        return 0.0
    eligible.sort(key=lambda item: item[0])
    earliest_band = eligible[0][1]
    latest_band = eligible[-1][1]
    change_bps = (float(latest_band) - float(earliest_band)) * 100.0
    if change_bps >= float(threshold_bps):
        return 1.0
    if change_bps <= -float(threshold_bps):
        return -1.0
    return 0.0


def compute_vix_level_regime_score(
    *,
    prior_bar_vix_values: Sequence[float],
) -> float:
    """Score the T-1 VIX against the strict-prior trailing-bar tertiles.

    ``prior_bar_vix_values`` is the ordered ``vix_close`` series from
    the supervised event's prior 20-bar window. Bars are already
    strictly before ``event_date`` by the loader contract
    (``_assert_no_lookahead`` on the events-builder side). The T-1
    value (last entry) is the most recent strictly-prior observation;
    the tertile cutoffs are taken over the same prior-bar series so
    the bucketing is itself a strict-prior computation.

    Returns ``+1.0`` when T-1 VIX > upper tertile, ``-1.0`` when below
    the lower tertile, ``0.0`` otherwise. Falls back to ``0.0`` when
    the window has zero or one valid value (no tertile defined) or
    when every value is ``0.0`` (placeholder rows on pre-A3 events
    parquets).
    """

    cleaned = [float(v) for v in prior_bar_vix_values if v is not None]
    cleaned = [v for v in cleaned if v == v and v > 0.0]
    if len(cleaned) < 2:
        return 0.0
    t_minus_one = cleaned[-1]
    sorted_values = sorted(cleaned)
    n = len(sorted_values)
    low_idx = max(0, min(n - 1, int(VIX_LOW_TERTILE * n)))
    high_idx = max(0, min(n - 1, int(VIX_HIGH_TERTILE * n)))
    low_cutoff = sorted_values[low_idx]
    high_cutoff = sorted_values[high_idx]
    if t_minus_one > high_cutoff:
        return 1.0
    if t_minus_one < low_cutoff:
        return -1.0
    return 0.0


def compute_term_spread_sign(
    *,
    t_minus_one_tnx_close: float | None,
    t_minus_one_irx_close: float | None,
) -> float:
    """Sign of the 10y-3m yield-curve slope at T-1.

    Reads the last prior bar's ``tnx_close`` (^TNX, 10y) and
    ``irx_close`` (^IRX, 13-week T-bill) and returns the sign of the
    difference. Both inputs are strictly-prior by the loader contract
    (daily-bar resolution, bars dated strictly before the supervised
    event). Returns ``0.0`` when either input is missing or when the
    spread is exactly zero (no inversion signal).
    """

    tnx = _coerce_float(t_minus_one_tnx_close)
    irx = _coerce_float(t_minus_one_irx_close)
    if tnx is None or irx is None:
        return 0.0
    spread = float(tnx) - float(irx)
    if spread > 0.0:
        return 1.0
    if spread < 0.0:
        return -1.0
    return 0.0


def compute_macro_regime_features(
    *,
    event_date: datetime.date,
    mp_surprise_lookup: Mapping[str, Mapping[str, Any]],
    prior_bar_vix_values: Sequence[float],
    t_minus_one_tnx_close: float | None,
    t_minus_one_irx_close: float | None,
) -> MacroRegimeFeatures:
    """Compose the three regime indicators for one supervised event.

    All inputs are strict-prior to ``event_date`` by construction; see
    each component helper's docstring for the per-feature contract.
    The composer is a pure data transform -- it does no I/O and the
    loader can therefore call it inside the per-event for-loop without
    touching disk a second time.
    """

    return MacroRegimeFeatures(
        policy_cycle_phase_score=compute_policy_cycle_phase_score(
            event_date=event_date,
            mp_surprise_lookup=mp_surprise_lookup,
        ),
        vix_level_regime_score=compute_vix_level_regime_score(
            prior_bar_vix_values=prior_bar_vix_values,
        ),
        term_spread_sign=compute_term_spread_sign(
            t_minus_one_tnx_close=t_minus_one_tnx_close,
            t_minus_one_irx_close=t_minus_one_irx_close,
        ),
    )


__all__ = [
    "MacroRegimeFeatures",
    "POLICY_CYCLE_LOOKBACK_DAYS",
    "POLICY_CYCLE_THRESHOLD_BPS",
    "REGIME_FEATURE_DIM",
    "VIX_HIGH_TERTILE",
    "VIX_LOW_TERTILE",
    "compute_macro_regime_features",
    "compute_policy_cycle_phase_score",
    "compute_term_spread_sign",
    "compute_vix_level_regime_score",
]
