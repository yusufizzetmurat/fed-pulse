"""Serve the latest monetary-policy surprise as a descriptive panel.

The Workspace MP-surprise chip reads the most recent FOMC row from the
``mp_surprises.parquet`` table built by :mod:`app.data.mp_surprise`. The
surprise level is a strict-prior, basis-point quantity (see the module
docstring on ``app.data.mp_surprise`` for the construction); the serving
layer here only picks the latest row by ``event_date`` and translates the
columns to the wire shape consumed by the descriptive chip.

Sign convention (matches the upstream construction in
``app.data.mp_surprise``): positive ``mp_surprise_level`` is a *hawkish*
surprise — actual change minus the pre-event 1-month-ahead implied path
came in above expectations. Negative is a *dovish* surprise. The
"no surprise" band is symmetric around zero at ``|x| <= 2.5 bps``: the
treasury-OIS proxy and the daily-window curve discretization together
produce small drifts on every meeting, so a tighter band would label
near-zero noise as a directional signal.

The descriptive surface is intentionally text- / realized-only — it
reports a measured quantity for context and never feeds the forecast
cards. The Workspace spine renders it under ``WorkspaceSection
variant="descriptive"``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.config import DATA_DIR
from app.schemas import MonetaryPolicySurpriseResponse


# Symmetric no-surprise band in basis points. Anything inside the band is
# rendered as ``no_surprise`` so the chip does not flash a directional
# headline on near-zero meetings. 2.5 bps is half the discretization
# floor of the daily Treasury proxy used in :mod:`app.data.mp_surprise`.
NO_SURPRISE_BAND_BPS: float = 2.5

# Canonical on-disk location for the parquet (see
# :data:`app.data.fed_comms_dataset.DEFAULT_MP_SURPRISE_PARQUET`).
DEFAULT_MP_SURPRISE_PARQUET: Path = DATA_DIR / "external" / "fred" / "mp_surprises.parquet"


class MpSurpriseUnavailable(RuntimeError):
    """Raised when the parquet is missing or empty."""


@dataclass(frozen=True)
class _LatestRow:
    event_date: str
    mp_surprise_level_bps: float
    is_intermeeting: bool
    ff_target_prior_bps: float | None


def _classify(level_bps: float) -> tuple[str, float]:
    """Bucket the signed level into ``hawkish`` / ``dovish`` / ``no_surprise``.

    Returns the discrete direction label and the absolute magnitude in
    basis points. Positive levels are hawkish (actual policy path
    tighter than priced), negative are dovish, and anything inside the
    symmetric ``|x| <= NO_SURPRISE_BAND_BPS`` band is reported as
    ``no_surprise`` so the chip stays quiet on small-drift meetings.
    """

    magnitude = abs(float(level_bps))
    if magnitude <= NO_SURPRISE_BAND_BPS:
        return "no_surprise", magnitude
    return ("hawkish" if level_bps > 0 else "dovish"), magnitude


def _read_latest(path: Path) -> _LatestRow:
    """Read the latest event_date row from the parquet."""

    if not path.exists():
        raise MpSurpriseUnavailable(
            f"mp_surprises parquet not found at {path}; rebuild via "
            "`python -m app.data.mp_surprise`."
        )

    # Imported lazily so the API process does not pay pandas startup
    # cost when this endpoint is never hit.
    import pandas as pd  # noqa: WPS433 - lazy import is intentional

    df = pd.read_parquet(path)
    if df.empty:
        raise MpSurpriseUnavailable(f"mp_surprises parquet at {path} is empty")

    required = {"event_date", "mp_surprise_level", "is_intermeeting"}
    missing = required.difference(df.columns)
    if missing:
        raise MpSurpriseUnavailable(
            f"mp_surprises parquet missing required columns: {sorted(missing)}"
        )

    # Sort by event_date ascending and pick the last row. Sorting on a
    # string column is fine here — the dates are zero-padded ISO-8601.
    latest = df.sort_values("event_date").iloc[-1]

    level = latest["mp_surprise_level"]
    if level is None or (isinstance(level, float) and level != level):  # NaN check
        raise MpSurpriseUnavailable(
            "latest mp_surprises row has a null mp_surprise_level"
        )

    # Defensive guard for partially-populated rows: the latest event
    # row can carry a usable mp_surprise_level but a malformed
    # event_date (NaN / empty) or NaN is_intermeeting flag — both of
    # which would silently coerce to misleading wire values
    # (``"nan"`` and ``True``). Surfacing the missing-data condition
    # here lets the API translate to a structured 503 instead of
    # leaking a corrupt row into the descriptive chip.
    event_date_raw = latest["event_date"]
    if event_date_raw is None or (
        isinstance(event_date_raw, float) and event_date_raw != event_date_raw
    ):
        raise MpSurpriseUnavailable(
            "latest mp_surprises row has a null event_date"
        )
    event_date_str = str(event_date_raw).strip()
    if not event_date_str:
        raise MpSurpriseUnavailable(
            "latest mp_surprises row has an empty event_date"
        )

    is_intermeeting_raw = latest["is_intermeeting"]
    if is_intermeeting_raw is None or (
        isinstance(is_intermeeting_raw, float)
        and is_intermeeting_raw != is_intermeeting_raw
    ):
        raise MpSurpriseUnavailable(
            "latest mp_surprises row has a null is_intermeeting flag"
        )

    prior_pct = latest.get("ff_target_prior") if "ff_target_prior" in df.columns else None
    prior_bps: float | None
    if prior_pct is None or (isinstance(prior_pct, float) and prior_pct != prior_pct):
        prior_bps = None
    else:
        # Parquet column is in percent (e.g. 3.625 -> 362.5 bps).
        prior_bps = float(prior_pct) * 100.0

    return _LatestRow(
        event_date=event_date_str,
        mp_surprise_level_bps=float(level),
        is_intermeeting=bool(is_intermeeting_raw),
        ff_target_prior_bps=prior_bps,
    )


def load_latest_mp_surprise(
    path: Path | str | None = None,
) -> MonetaryPolicySurpriseResponse:
    """Return the latest FOMC monetary-policy surprise as a wire response.

    Raises :class:`MpSurpriseUnavailable` when the parquet is missing or
    its latest row carries a null surprise level. Callers in
    :mod:`app.main` translate that into a 503 with a structured detail.
    """

    parquet_path = Path(path) if path is not None else DEFAULT_MP_SURPRISE_PARQUET
    row = _read_latest(parquet_path)
    direction, magnitude = _classify(row.mp_surprise_level_bps)
    return MonetaryPolicySurpriseResponse(
        event_date=row.event_date,
        mp_surprise_level_bps=row.mp_surprise_level_bps,
        direction=direction,  # type: ignore[arg-type]
        magnitude_bps=magnitude,
        is_intermeeting=row.is_intermeeting,
        ff_target_prior_bps=row.ff_target_prior_bps,
    )


__all__ = [
    "DEFAULT_MP_SURPRISE_PARQUET",
    "MpSurpriseUnavailable",
    "NO_SURPRISE_BAND_BPS",
    "load_latest_mp_surprise",
]
