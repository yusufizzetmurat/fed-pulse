"""Summary of Economic Projections (SEP) dot-plot features (#215).

Five strict-prior scalars per supervised FOMC event, drawn from the
SEP series FRED actually publishes:

- ``sep_ffr_median_current_year`` -- median FFR projection for the
  current calendar-year end (``FEDTARMD``). Quarterly cadence (March /
  June / September / December meetings).
- ``sep_ffr_median_next_year`` -- median FFR projection for the next
  calendar-year end. FRED publishes this as a year-specific series
  ``FEDTARMD<YYYY>`` per vintage rather than a single rolling line;
  the parquet builder pivots per release, pulling
  ``FEDTARMD<year(release)+1>`` at each SEP meeting and writing the
  on-or-before value into the row. Pre-2014 vintages lack the year-
  specific series entirely and the slot collapses to ``0.0`` with
  the per-row missing flag carrying the signal (#415 restoration of
  the slot dropped during the #215 reviewer pass).
- ``sep_ffr_median_longer_run`` -- median longer-run FFR projection
  (``FEDTARMDLR``); the FOMC's neutral-rate estimate.
- ``sep_ffr_range_current`` -- upper minus lower of the full
  all-participants range for the current-year projection
  (``FEDTARRH`` - ``FEDTARRL``). Dispersion measure capturing how
  tightly the Committee's views cluster. The two FRED series are full
  range bounds (not central-tendency bounds, which trim three high
  and three low and would require ``FEDTARCT*`` series this loader
  does not currently pull).
- ``sep_release_flag`` -- ``1.0`` when the supervised meeting itself
  released a fresh SEP (March / June / September / December); ``0.0``
  when the values are forward-filled from a prior SEP meeting. Lets
  the model learn the interaction between "fresh projections" and the
  reaction to the released document.

Provenance contract -- ``T (snapshot)`` for SEP-release meetings,
``T-Δ`` for forward-filled meetings. The SEP is released simultaneously
with the FOMC statement at the meeting, so its values are observable
from the document released on ``T`` (same band as the existing
``stance_*`` text features). On non-SEP meetings the slot carries the
most recent prior SEP's values (forward-fill is strict-prior by
construction), and ``sep_release_flag = 0.0`` distinguishes the two
cases. No future-derived inputs.

The block is opt-in via ``--use-sep`` on ``app.train_forecaster``.
When the flag is off, the loader leaves the ``sep_features`` slot
``None`` on every event and ``FeatureVector.as_rich_list`` does NOT
append the block, so the default per-bar feature size is byte-identical
to pre-#215. See ADR 0030.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import Any, Mapping


# Number of values the SEP composer emits per event, including the
# release flag in the last slot. The ``RICH_SEP_DIM`` constant on
# ``app.models.config`` is the single source of truth; this alias
# gives the helper a local name.
SEP_FEATURE_DIM: int = 5


@dataclass(frozen=True)
class SepProjections:
    """One SEP release's reported medians + range dispersion.

    Field units are percentage points (e.g. ``5.375`` = 5.375%). The
    dataclass is frozen so a builder cannot accidentally mutate a
    cached row from the projections lookup.
    """

    meeting_date: datetime.date
    ffr_median_current_year: float | None
    ffr_median_next_year: float | None
    ffr_median_longer_run: float | None
    ffr_range_upper_current: float | None
    ffr_range_lower_current: float | None

    def range_current(self) -> float | None:
        """Upper minus lower of the current-year all-participants range.

        Returns ``None`` when either bound is missing so the composer
        stamps a 0.0 without poisoning the dispersion signal with an
        inferred-zero spread.
        """

        if self.ffr_range_upper_current is None or self.ffr_range_lower_current is None:
            return None
        return float(self.ffr_range_upper_current) - float(self.ffr_range_lower_current)


@dataclass(frozen=True)
class SepFeatures:
    """Per-event SEP feature block, as the loader writes onto a FeatureVector.

    The first four slots are the scalar SEP statistics (current-year
    median, next-year median, longer-run median, current-year range);
    the fifth is the release flag. ``as_list`` returns the five-element
    list the loader broadcasts onto every bar of the supervised
    sequence.
    """

    ffr_median_current_year: float
    ffr_median_next_year: float
    ffr_median_longer_run: float
    ffr_range_current: float
    sep_release_flag: float

    def as_list(self) -> list[float]:
        return [
            float(self.ffr_median_current_year),
            float(self.ffr_median_next_year),
            float(self.ffr_median_longer_run),
            float(self.ffr_range_current),
            float(self.sep_release_flag),
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


def _parse_iso_date(value: Any) -> datetime.date | None:
    if value is None:
        return None
    try:
        return datetime.date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _projection_from_record(record: Mapping[str, Any]) -> SepProjections | None:
    """Coerce one lookup row to a ``SepProjections`` dataclass.

    Returns ``None`` when the row's meeting date cannot be parsed; the
    composer then skips it instead of poisoning the forward-fill chain
    with a malformed anchor.
    """

    md = _parse_iso_date(record.get("meeting_date"))
    if md is None:
        return None
    return SepProjections(
        meeting_date=md,
        ffr_median_current_year=_coerce_float(record.get("ffr_median_current_year")),
        ffr_median_next_year=_coerce_float(record.get("ffr_median_next_year")),
        ffr_median_longer_run=_coerce_float(record.get("ffr_median_longer_run")),
        ffr_range_upper_current=_coerce_float(record.get("ffr_range_upper_current")),
        ffr_range_lower_current=_coerce_float(record.get("ffr_range_lower_current")),
    )


def compute_sep_features_for_event(
    *,
    event_date: datetime.date,
    sep_lookup: Mapping[str, Mapping[str, Any]],
) -> SepFeatures | None:
    """Compose the SEP block for one supervised event.

    Walks ``sep_lookup`` for the most recent SEP release whose
    ``meeting_date <= event_date``. When the matching release IS the
    supervised event itself, the release flag is ``1.0`` (the meeting
    refreshed projections at ``T``, observable from the SEP document
    released simultaneously with the FOMC statement). Otherwise the
    flag is ``0.0`` (forward-fill from a strictly-prior SEP).

    Returns ``None`` when the lookup carries no SEP release on or
    before ``event_date`` (cold-start at the beginning of the corpus);
    the caller treats this as "no signal" and flips the missing flag
    to ``1.0``.

    Missing scalars inside the matched SEP collapse to ``0.0`` so the
    block shape stays fixed; the per-row missing flag is the only
    signal the caller emits for absent-block cases.
    """

    if not sep_lookup:
        return None
    eligible: list[SepProjections] = []
    for _key, record in sep_lookup.items():
        proj = _projection_from_record(record)
        if proj is None:
            continue
        if proj.meeting_date > event_date:
            continue
        eligible.append(proj)
    if not eligible:
        return None
    eligible.sort(key=lambda p: p.meeting_date)
    latest = eligible[-1]
    release_flag = 1.0 if latest.meeting_date == event_date else 0.0
    range_current = latest.range_current()
    return SepFeatures(
        ffr_median_current_year=float(latest.ffr_median_current_year or 0.0),
        ffr_median_next_year=float(latest.ffr_median_next_year or 0.0),
        ffr_median_longer_run=float(latest.ffr_median_longer_run or 0.0),
        ffr_range_current=float(range_current or 0.0),
        sep_release_flag=release_flag,
    )


__all__ = [
    "SEP_FEATURE_DIM",
    "SepFeatures",
    "SepProjections",
    "compute_sep_features_for_event",
]
