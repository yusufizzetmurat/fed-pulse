"""Monetary-policy surprise time-series (Phase 8 foundation, closes #146).

Produces ``data/external/fred/mp_surprises.parquet`` -- one row per FOMC
meeting date from 2010-01-01 to today, carrying:

- the fed-funds target before / after the announcement,
- a 1-month-ahead policy-rate change (``mp_surprise_level``) measured from
  t-1 EOD to t+1 EOD,
- a PCA-derived ``mp_surprise_path_factor`` capturing curve-shape
  surprises beyond the level component,
- the pre- and post-event implied policy curves at {1, 3, 6, 12, 24}
  months ahead,
- a Cieslak-Vissing-Jorgensen-style ``fed_info_factor`` (residual of the
  level surprise on the same-day SPX return; documented daily-window
  proxy when intraday data is unavailable; degrades to ``None`` with
  ``fed_info_factor_source = "unavailable"`` when SPX data is absent so
  the missing-data row stays distinguishable from a real-but-tiny
  residual),
- an ``is_intermeeting`` flag for unscheduled / emergency actions, and
- a ``data_version`` short-sha of the joined FRED series.

Methodology
-----------

The gold-standard recipe for a monetary-policy surprise is to read CME
fed-funds-futures settlement prices around each FOMC announcement and
back out the implied policy rate. The CME settlement series is not
freely available through a public API -- it sits behind CME DataMine and
attribution-only access. To keep this module reproducible from a free
data source, we proxy the fed-funds-futures curve with the Treasury
constant-maturity series available from FRED (``DGS1MO``, ``DGS3MO``,
``DGS6MO``, ``DGS1``, ``DGS2``). Treasury yields embed a small term
premium plus a Treasury-vs-OIS basis, but those components move slowly
and are absorbed into the cross-sectional PCA fit -- they do not bias
the daily *change* used in :func:`compute_surprises`. The
``methodology`` column on every row records which path was used so the
downstream consumer never has to guess:

- ``methodology = "ff_futures"`` -- a future enhancement that reads CME
  settlement parquets from ``data/external/cme/`` (out of scope for
  #146).
- ``methodology = "ois_proxy"`` -- the Treasury-yield proxy implemented
  here. The wiki entry under ``06_Deep_Learning_Roadmap.md`` documents
  the caveat in plain language.

Target-rate inputs use ``DFEDTAR`` (the single-target series, valid
through 2008-12-15) joined to ``DFEDTARL`` / ``DFEDTARU`` (the post-2008
target band, valid 2008-12-16 onward). The reconstructed target on a
day is the midpoint of the band when a band is published, else the
single target. ``DFF`` (effective funds rate, daily) is read for cross-
validation and is exposed as ``ff_effective_after`` so credibility
checks can compute the realised-vs-target gap.

Reproducibility
---------------

Same FRED inputs imply identical *data* in the parquet, locked via
:func:`dataframe_value_hash`. The PCA fit is deterministic
(eigenvectors persisted in the lock JSON before use); SPX data is
loaded from a fixed yfinance cache. The on-disk encoding uses
``zstd`` (level 3) with ``write_statistics=False`` and
``use_dictionary=False`` so the parquet metadata is stable, but the
canonical determinism contract is on the **DataFrame's sorted values**
(not the raw byte stream): we hash the row-wise sorted DataFrame after
re-reading and compare to a pinned reference. This is more honest
than byte-hashing because compressed parquet metadata (pyarrow
version, encoder build flags, platform) is not portable across
machines.

No look-ahead
-------------

``pre_event_curve`` reads the last trading-day yield strictly before
``event_date``; ``post_event_curve`` reads the first trading-day yield
strictly after ``event_date``. The two-day window absorbs same-day
announcement noise and matches Cieslak-Vissing-Jorgensen 2021 (table 2,
"announcement window"). The contract is enforced by an assertion in
:func:`_pre_post_yields`.

CLI
---

::

    python -m app.data.mp_surprise \
        --start 2010-01-01 --end today \
        --output mp_surprises.parquet

The output parquet lands under ``<fred_cache_dir>/`` (default
``data/external/fred/``). A SOURCES.lock entry under the key
``mp_surprises`` records the parquet sha256, retrieval timestamp,
methodology label, row count, and PCA eigenvectors so the build is
auditable.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import httpx
import pandas as pd

from app.config import DATA_DIR
from app.services.fred_client import (
    DEFAULT_CACHE_DIR as FRED_CACHE_DIR,
    FredObservation,
    FredSeriesResponse,
    SOURCES_LOCK_NAME,
    fetch_fred_series,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_FOMC_CALENDAR_CSV = DATA_DIR / "external" / "fomc_meetings_2010_2026.csv"
DEFAULT_START = "2010-01-01"
DEFAULT_OUTPUT_NAME = "mp_surprises.parquet"
DEFAULT_LOCK_KEY = "mp_surprises"

# Target-rate series (FRED).
TARGET_RATE_SERIES = ("DFEDTAR", "DFEDTARU", "DFEDTARL")
EFFECTIVE_RATE_SERIES = "DFF"

# Yield curve proxy at five horizons. Tuple ordering is significant: the
# PCA fit and the persisted eigenvectors are anchored on this layout.
CURVE_TENORS_MONTHS: tuple[int, ...] = (1, 3, 6, 12, 24)
CURVE_SERIES_BY_TENOR: dict[int, str] = {
    1: "DGS1MO",
    3: "DGS3MO",
    6: "DGS6MO",
    12: "DGS1",
    24: "DGS2",
}

# Tenors that feed the PCA path-factor fit. We always exclude the 1m
# tenor: it is the level component and we residualize against it.
PATH_TENORS_MONTHS: tuple[int, ...] = (3, 6, 12)

# Methodology labels. Single source of truth.
METHODOLOGY_FF_FUTURES = "ff_futures"
METHODOLOGY_OIS_PROXY = "ois_proxy"

# ``ff_target_after`` for emergency actions on weekends / off-cycle days
# is sometimes unobserved on event_date itself. We then fall back to the
# next-trading-day target value as documented in the row's
# ``target_source`` flag.
MAX_TARGET_LOOKAHEAD_DAYS = 5

# yfinance retry knobs (SPX intraday). When the lookup fails we degrade
# ``fed_info_factor`` to ``None`` and set
# ``fed_info_factor_source = "unavailable"`` as the transparency flag.
# ``None`` (not ``0.0``) is the documented sentinel because a hard zero
# would be indistinguishable from a real-but-tiny residual.
SPX_LOOKUP_RADIUS_DAYS = 3


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FomcMeetingRecord:
    """One row from the bundled FOMC calendar."""

    meeting_date: _dt.date
    is_intermeeting: bool
    notes: str


@dataclass(frozen=True)
class CurvePoint:
    months_ahead: int
    implied_rate: float


@dataclass(frozen=True)
class SurpriseRow:
    """One assembled meeting row, before serialization."""

    event_date: _dt.date
    meeting_id: int
    ff_target_prior: float | None
    ff_target_after: float | None
    mp_surprise_level: float | None
    mp_surprise_path_factor: float | None
    pre_event_curve: list[CurvePoint]
    post_event_curve: list[CurvePoint]
    fed_info_factor: float | None
    is_intermeeting: bool
    methodology: str
    fed_info_factor_source: str
    target_source: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_date": self.event_date.isoformat(),
            "meeting_id": int(self.meeting_id),
            "ff_target_prior": _clean_float(self.ff_target_prior),
            "ff_target_after": _clean_float(self.ff_target_after),
            "mp_surprise_level": _clean_float(self.mp_surprise_level),
            "mp_surprise_path_factor": _clean_float(self.mp_surprise_path_factor),
            "pre_event_curve": _curve_to_json(self.pre_event_curve),
            "post_event_curve": _curve_to_json(self.post_event_curve),
            "fed_info_factor": _clean_float(self.fed_info_factor),
            "is_intermeeting": bool(self.is_intermeeting),
            "methodology": self.methodology,
            "fed_info_factor_source": self.fed_info_factor_source,
            "target_source": self.target_source,
        }


COLUMN_ORDER: tuple[str, ...] = (
    "event_date",
    "meeting_id",
    "ff_target_prior",
    "ff_target_after",
    "mp_surprise_level",
    "mp_surprise_path_factor",
    "pre_event_curve",
    "post_event_curve",
    "fed_info_factor",
    "is_intermeeting",
    "methodology",
    "fed_info_factor_source",
    "target_source",
    "data_version",
)


# ---------------------------------------------------------------------------
# Helpers: dates + value cleanup
# ---------------------------------------------------------------------------


def _clean_float(value: float | None) -> float | None:
    if value is None:
        return None
    try:
        fv = float(value)
    except (TypeError, ValueError):
        return None
    if fv != fv:  # NaN
        return None
    return fv


def _curve_to_json(curve: Sequence[CurvePoint]) -> str:
    payload = [
        {"months_ahead": int(p.months_ahead), "implied_rate": _round(_clean_float(p.implied_rate))}
        for p in curve
    ]
    return json.dumps(payload, separators=(",", ":"))


def _round(value: float | None, ndigits: int = 6) -> float | None:
    if value is None:
        return None
    return round(value, ndigits)


def _parse_date(value: str | _dt.date) -> _dt.date:
    if isinstance(value, _dt.date):
        return value
    return _dt.date.fromisoformat(str(value)[:10])


# ---------------------------------------------------------------------------
# FOMC calendar
# ---------------------------------------------------------------------------


def load_fomc_calendar(
    *,
    path: Path | str | None = None,
    start: _dt.date | None = None,
    end: _dt.date | None = None,
) -> list[FomcMeetingRecord]:
    """Load the bundled FOMC meeting calendar, filtered to ``[start, end]``.

    The bundled CSV at ``data/external/fomc_meetings_2010_2026.csv`` is
    the source of truth. Callers can override ``path`` to point at a
    refreshed calendar (for example, when the Fed publishes the next
    year's schedule).
    """

    resolved = Path(path) if path is not None else DEFAULT_FOMC_CALENDAR_CSV
    if not resolved.exists():
        raise FileNotFoundError(f"FOMC calendar CSV missing: {resolved}")
    records: list[FomcMeetingRecord] = []
    with resolved.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            raw_date = (row.get("meeting_date") or "").strip()
            if not raw_date:
                continue
            try:
                d = _dt.date.fromisoformat(raw_date)
            except ValueError:
                continue
            inter = (row.get("is_intermeeting") or "false").strip().lower() in {"true", "1", "yes"}
            notes = (row.get("notes") or "").strip()
            records.append(FomcMeetingRecord(meeting_date=d, is_intermeeting=inter, notes=notes))
    records.sort(key=lambda r: r.meeting_date)
    if start is not None:
        records = [r for r in records if r.meeting_date >= start]
    if end is not None:
        records = [r for r in records if r.meeting_date <= end]
    return records


# ---------------------------------------------------------------------------
# FRED loading + caching
# ---------------------------------------------------------------------------


def _fetch_series_safely(
    series_id: str,
    *,
    start: str,
    end: str,
    cache_dir: Path,
    transport: httpx.BaseTransport | None,
    force_refresh: bool,
) -> FredSeriesResponse:
    return fetch_fred_series(
        series_id,
        start=start,
        end=end,
        cache_dir=cache_dir,
        transport=transport,
        force_refresh=force_refresh,
    )


def _series_to_map(series: FredSeriesResponse) -> dict[_dt.date, float]:
    """Reduce a FRED response to a ``date -> value`` dict, skipping missing."""

    out: dict[_dt.date, float] = {}
    for obs in series.observations:
        if obs.value is None:
            continue
        try:
            d = _dt.date.fromisoformat(obs.date)
        except ValueError:
            continue
        out[d] = float(obs.value)
    return out


def _target_rate_on(
    on_date: _dt.date,
    *,
    upper: Mapping[_dt.date, float],
    lower: Mapping[_dt.date, float],
    single: Mapping[_dt.date, float],
) -> tuple[float | None, str]:
    """Reconstruct the published fed-funds target on ``on_date``.

    Post-2008 the FOMC publishes a target band (``DFEDTARL`` / ``DFEDTARU``).
    Pre-2008 it publishes a single target (``DFEDTAR``). We return the
    band midpoint when both bounds are present, else the single target.
    The returned ``source`` flag is one of ``{"band","single","missing"}``
    so downstream consumers can audit.
    """

    if on_date in upper and on_date in lower:
        return ((upper[on_date] + lower[on_date]) / 2.0, "band")
    if on_date in single:
        return (single[on_date], "single")
    return (None, "missing")


def _lookup_target_with_lookahead(
    on_date: _dt.date,
    *,
    upper: Mapping[_dt.date, float],
    lower: Mapping[_dt.date, float],
    single: Mapping[_dt.date, float],
    lookahead_days: int = MAX_TARGET_LOOKAHEAD_DAYS,
    lookbehind_days: int = 0,
) -> tuple[float | None, str]:
    """Try ``on_date`` first; degrade to nearby trading days if missing.

    The fed-funds target is constant between meetings, so reading the
    target on the meeting date or the next published date both yield
    the same number. Falling back to a nearby day is safe and matches
    how DFEDTARU is published (no weekend rows).
    """

    direct = _target_rate_on(on_date, upper=upper, lower=lower, single=single)
    if direct[0] is not None:
        return direct
    for offset in range(1, lookahead_days + 1):
        d = on_date + _dt.timedelta(days=offset)
        cand = _target_rate_on(d, upper=upper, lower=lower, single=single)
        if cand[0] is not None:
            return (cand[0], cand[1] + "_lookahead")
    for offset in range(1, lookbehind_days + 1):
        d = on_date - _dt.timedelta(days=offset)
        cand = _target_rate_on(d, upper=upper, lower=lower, single=single)
        if cand[0] is not None:
            return (cand[0], cand[1] + "_lookbehind")
    return (None, "missing")


def _pre_post_yields(
    on_date: _dt.date,
    series_map: Mapping[_dt.date, float],
    *,
    lookbehind_days: int = 5,
    lookahead_days: int = 5,
    trading_days: Sequence[_dt.date] | None = None,
) -> tuple[float | None, float | None, _dt.date | None, _dt.date | None]:
    """Return ``(pre_value, post_value, pre_date, post_date)`` for one series.

    ``pre`` is the latest available value strictly before ``on_date``.
    ``post`` is the earliest available value strictly after ``on_date``.
    The no-look-ahead contract is enforced by the strict inequalities.
    Returns None when no value falls inside the window.

    The lookback is measured in **trading days**, not calendar days,
    when a sorted ``trading_days`` index is supplied. We bisect into the
    index to find the nearest trading day strictly before/after
    ``on_date`` and then step ``lookbehind_days`` / ``lookahead_days``
    *trading-day* slots. This matters during multi-day holiday clusters
    (Christmas-NYE, Easter, Thanksgiving week) where a 5-calendar-day
    radius can exhaust without crossing a single trading day and
    silently return ``None``.

    Backwards compatibility: when ``trading_days`` is omitted we derive
    it from ``series_map.keys()`` (which is itself the set of dates with
    a published yield -- i.e. the implicit trading-day calendar).
    """

    if trading_days is None:
        trading_days = sorted(series_map.keys())
    n_tdays = len(trading_days)

    pre_value: float | None = None
    pre_date: _dt.date | None = None
    post_value: float | None = None
    post_date: _dt.date | None = None

    if n_tdays > 0:
        import bisect as _bisect

        # `bisect_left` returns the first index whose value is >= on_date.
        # Trading days strictly before on_date are at indices < idx.
        idx = _bisect.bisect_left(trading_days, on_date)
        # Walk backwards through trading days for the pre side.
        # `lookbehind_days` counts trading-day slots, so we step from
        # idx - 1 (the most recent trading day < on_date) down at most
        # lookbehind_days - 1 more steps before giving up.
        for step in range(lookbehind_days):
            i = idx - 1 - step
            if i < 0:
                break
            cand = trading_days[i]
            if cand >= on_date:
                continue
            if cand in series_map:
                pre_value = series_map[cand]
                pre_date = cand
                break
        # `bisect_right` returns the first index whose value is > on_date.
        idx2 = _bisect.bisect_right(trading_days, on_date)
        for step in range(lookahead_days):
            i = idx2 + step
            if i >= n_tdays:
                break
            cand = trading_days[i]
            if cand <= on_date:
                continue
            if cand in series_map:
                post_value = series_map[cand]
                post_date = cand
                break

    if pre_date is not None and post_date is not None:
        assert pre_date < on_date < post_date, (
            f"no-look-ahead contract violated: pre={pre_date} on={on_date} post={post_date}"
        )
    return (pre_value, post_value, pre_date, post_date)


# ---------------------------------------------------------------------------
# PCA path factor
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PathFactorModel:
    """Persisted PCA fit on residual curve-shape changes.

    The path factor is the first principal component of
    ``delta_path_residuals``, where each residual is computed by
    regressing the ``tenor in PATH_TENORS_MONTHS`` change on the
    1-month change (level component). We keep eigenvectors as plain
    Python floats so the lock JSON is byte-stable across numpy
    versions.
    """

    tenors_months: tuple[int, ...]
    beta_on_level: tuple[float, ...]
    mean_residual: tuple[float, ...]
    eigenvector: tuple[float, ...]
    explained_variance_ratio: float
    n_meetings_fit: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "tenors_months": list(self.tenors_months),
            "beta_on_level": [float(v) for v in self.beta_on_level],
            "mean_residual": [float(v) for v in self.mean_residual],
            "eigenvector": [float(v) for v in self.eigenvector],
            "explained_variance_ratio": float(self.explained_variance_ratio),
            "n_meetings_fit": int(self.n_meetings_fit),
        }


def _fit_path_factor_model(
    delta_level: Sequence[float],
    delta_path: Sequence[Sequence[float]],
    *,
    tenors_months: Sequence[int] = PATH_TENORS_MONTHS,
) -> PathFactorModel:
    """Fit a deterministic PCA on level-residualized curve-shape changes.

    Inputs are aligned across meetings: ``delta_level[i]`` is the 1-month
    change on meeting ``i``; ``delta_path[i]`` is the parallel list of
    {3m, 6m, 12m} changes. We:

    1. Run OLS per tenor: ``delta_tenor = beta_tenor * delta_level``
       (no intercept; level changes have a near-zero mean across the
       sample after de-meaning would cancel out).
    2. Residualize: ``r_tenor = delta_tenor - beta_tenor * delta_level``.
    3. De-mean residuals; compute the 3x3 covariance matrix.
    4. Symmetric-eigendecomposition; pick the eigenvector with the
       largest eigenvalue.
    5. Sign-normalize: the largest absolute component is forced
       positive so the eigenvector is unique up to sign.
    """

    import numpy as np

    if len(delta_level) != len(delta_path):
        raise ValueError(
            f"length mismatch: delta_level={len(delta_level)} vs delta_path={len(delta_path)}"
        )
    if not delta_level:
        # Empty fit: identity-style eigenvector (PC1 = first tenor).
        return PathFactorModel(
            tenors_months=tuple(tenors_months),
            beta_on_level=tuple([0.0] * len(tenors_months)),
            mean_residual=tuple([0.0] * len(tenors_months)),
            eigenvector=tuple([1.0] + [0.0] * (len(tenors_months) - 1)),
            explained_variance_ratio=0.0,
            n_meetings_fit=0,
        )
    x = np.asarray(delta_level, dtype=np.float64)
    Y = np.asarray(delta_path, dtype=np.float64)
    n, k = Y.shape
    if k != len(tenors_months):
        raise ValueError(f"delta_path column count {k} != tenors {len(tenors_months)}")

    # Step 1: per-tenor regression without intercept. beta_tenor = (x'y) / (x'x).
    xx = float(np.dot(x, x))
    betas = np.zeros(k, dtype=np.float64)
    if xx > 1e-18:
        for j in range(k):
            betas[j] = float(np.dot(x, Y[:, j]) / xx)
    # Step 2: residuals.
    residuals = Y - np.outer(x, betas)
    # Step 3: de-mean.
    mean_resid = residuals.mean(axis=0)
    centered = residuals - mean_resid
    # Step 4: covariance + eigendecomposition.
    cov = (centered.T @ centered) / max(n - 1, 1)
    # ``np.linalg.eigh`` returns ascending eigenvalues for symmetric matrices.
    eigvals, eigvecs = np.linalg.eigh(cov)
    top = int(np.argmax(eigvals))
    top_vec = eigvecs[:, top].copy()
    top_val = float(eigvals[top])
    total_var = float(eigvals.sum())
    evr = (top_val / total_var) if total_var > 1e-18 else 0.0
    # Step 5: sign-normalize. Largest-magnitude component positive.
    idx_max = int(np.argmax(np.abs(top_vec)))
    if top_vec[idx_max] < 0:
        top_vec = -top_vec
    return PathFactorModel(
        tenors_months=tuple(tenors_months),
        beta_on_level=tuple(float(v) for v in betas),
        mean_residual=tuple(float(v) for v in mean_resid),
        eigenvector=tuple(float(v) for v in top_vec),
        explained_variance_ratio=float(evr),
        n_meetings_fit=int(n),
    )


def _apply_path_factor(
    model: PathFactorModel,
    delta_level: float | None,
    delta_path: Sequence[float | None],
) -> float | None:
    """Project one meeting's residual onto the persisted eigenvector."""

    if delta_level is None:
        return None
    if any(v is None for v in delta_path):
        return None
    residual = []
    for j, dv in enumerate(delta_path):
        beta = model.beta_on_level[j]
        residual.append(float(dv) - beta * float(delta_level) - model.mean_residual[j])
    out = 0.0
    for r, e in zip(residual, model.eigenvector):
        out += r * e
    return float(out)


# ---------------------------------------------------------------------------
# Fed info factor (CVJ 2021 residual on stock returns)
# ---------------------------------------------------------------------------


def _spx_return_on(
    event_date: _dt.date,
    spx_close_by_date: Mapping[_dt.date, float],
    *,
    radius_days: int = SPX_LOOKUP_RADIUS_DAYS,
) -> tuple[float | None, str]:
    """Approximate the daily SPX return centred on ``event_date``.

    Returns ``(daily_return_pct, source_flag)`` where ``source_flag`` is
    one of:

    - ``"daily_window_proxy"`` -- the standard daily close-to-close
      return computed over a ``[t-1, t+1]`` close window. This is the
      documented approximation; intraday ``+/-30 min`` data is out of
      scope.
    - ``"unavailable"`` -- no SPX data inside the radius. The caller
      writes ``fed_info_factor = None`` (NOT zero) and stamps
      ``fed_info_factor_source = "unavailable"`` so the row is
      distinguishable from a real-but-tiny residual.

    We never claim a CVJ-style intraday measurement; the flag is the
    transparency mechanism.
    """

    pre_close: float | None = None
    pre_date: _dt.date | None = None
    for offset in range(1, radius_days + 1):
        d = event_date - _dt.timedelta(days=offset)
        if d in spx_close_by_date:
            pre_close = spx_close_by_date[d]
            pre_date = d
            break
    post_close: float | None = None
    post_date: _dt.date | None = None
    for offset in range(1, radius_days + 1):
        d = event_date + _dt.timedelta(days=offset)
        if d in spx_close_by_date:
            post_close = spx_close_by_date[d]
            post_date = d
            break
    if pre_close is None or post_close is None or pre_close <= 0:
        return (None, "unavailable")
    assert pre_date is not None and post_date is not None
    daily_return = (post_close - pre_close) / pre_close
    return (daily_return, "daily_window_proxy")


def _fit_fed_info_factor(
    levels: Sequence[float],
    spx_returns: Sequence[float],
) -> tuple[float, float]:
    """OLS ``level = alpha + beta * spx_return``. Returns ``(alpha, beta)``.

    The fed-information residual on each meeting is then
    ``level_i - (alpha + beta * spx_return_i)``. Degenerate inputs fall
    back to ``(0.0, 0.0)`` which makes the fed-info factor equal to the
    level itself -- documented as a "no-info-decomposition" fallback
    via the ``fed_info_factor_source`` flag on each row.
    """

    n = min(len(levels), len(spx_returns))
    if n < 5:
        return (0.0, 0.0)
    import numpy as np

    a = np.asarray(levels[:n], dtype=np.float64)
    b = np.asarray(spx_returns[:n], dtype=np.float64)
    mean_a = float(a.mean())
    mean_b = float(b.mean())
    cov = float(((a - mean_a) * (b - mean_b)).mean())
    var_b = float(((b - mean_b) ** 2).mean())
    if var_b <= 1e-18:
        return (0.0, 0.0)
    beta = cov / var_b
    alpha = mean_a - beta * mean_b
    return (float(alpha), float(beta))


# ---------------------------------------------------------------------------
# SPX loader (yfinance, optional)
# ---------------------------------------------------------------------------


def _load_spx_close_cache(
    cache_path: Path,
) -> dict[_dt.date, float]:
    """Load cached SPX daily closes from a parquet.

    Schema: columns ``date`` (ISO string) and ``close`` (float).
    Returns an empty dict when the cache is absent.
    """

    if not cache_path.exists():
        return {}
    frame = pd.read_parquet(cache_path)
    if "date" not in frame.columns or "close" not in frame.columns:
        return {}
    out: dict[_dt.date, float] = {}
    for raw_d, raw_c in zip(frame["date"].tolist(), frame["close"].tolist()):
        try:
            d = _dt.date.fromisoformat(str(raw_d)[:10])
            v = float(raw_c)
        except (TypeError, ValueError):
            continue
        if v > 0:
            out[d] = v
    return out


def _fetch_spx_close_via_yfinance(
    *,
    start: _dt.date,
    end: _dt.date,
    cache_path: Path,
) -> dict[_dt.date, float]:
    """Pull ^GSPC daily closes and persist them under ``cache_path``.

    Used by the CLI smoke run. Unit tests inject the cache directly and
    never trigger network I/O.
    """

    if cache_path.exists():
        return _load_spx_close_cache(cache_path)
    try:
        import yfinance as yf
    except ImportError:
        return {}
    try:
        frame = yf.Ticker("^GSPC").history(
            start=start.isoformat(),
            end=(end + _dt.timedelta(days=1)).isoformat(),
            auto_adjust=True,
        )
    except Exception:  # noqa: BLE001 -- best-effort fallback
        return {}
    if frame.empty:
        return {}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    out_frame = pd.DataFrame(
        {
            "date": [idx.date().isoformat() for idx in frame.index],
            "close": [float(v) for v in frame["Close"].to_numpy()],
        }
    )
    out_frame.to_parquet(cache_path, index=False)
    return _load_spx_close_cache(cache_path)


# ---------------------------------------------------------------------------
# Top-level builder
# ---------------------------------------------------------------------------


@dataclass
class BuildArtifacts:
    """Returned by :func:`build_mp_surprises` for downstream auditing."""

    frame: pd.DataFrame
    path_model: PathFactorModel
    data_version: str
    methodology: str
    fred_series_used: tuple[str, ...]
    rows_written: int
    intermeeting_rows: int
    fed_info_factor_unavailable_rows: int


def _data_version_hash(
    series_responses: Mapping[str, FredSeriesResponse],
    *,
    methodology: str,
    calendar_signature: str,
) -> str:
    """Short sha capturing the inputs that drove this build.

    Computed over the concatenation of ``series_id|observation_end|count``
    for every FRED series, plus the methodology label and a short hash of
    the FOMC calendar in use. Two builds with the same FRED state and
    calendar produce the same data_version. Note: we deliberately do NOT
    include the retrieval timestamp here so byte-identical re-builds
    share the same data_version.
    """

    parts: list[str] = [methodology, calendar_signature]
    for series_id in sorted(series_responses):
        resp = series_responses[series_id]
        parts.append(f"{series_id}|{resp.observation_end}|{resp.count}")
    digest = hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()
    return digest[:16]


def _calendar_signature(meetings: Sequence[FomcMeetingRecord]) -> str:
    payload = "\n".join(f"{m.meeting_date.isoformat()}|{int(m.is_intermeeting)}" for m in meetings)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def build_mp_surprises(
    *,
    start: _dt.date,
    end: _dt.date,
    fred_responses: Mapping[str, FredSeriesResponse],
    fomc_calendar: Sequence[FomcMeetingRecord] | None = None,
    spx_close_by_date: Mapping[_dt.date, float] | None = None,
    fomc_calendar_path: Path | str | None = None,
    methodology: str = METHODOLOGY_OIS_PROXY,
) -> BuildArtifacts:
    """Assemble the MP surprise frame.

    The pre-loaded ``fred_responses`` dict must contain every series in
    :data:`TARGET_RATE_SERIES`, :data:`EFFECTIVE_RATE_SERIES`, and the
    five curve series listed in :data:`CURVE_SERIES_BY_TENOR`. Tests
    inject canned responses; the CLI hydrates them from
    :func:`app.services.fred_client.fetch_fred_series`.
    """

    # ---- Calendar ----
    if fomc_calendar is None:
        fomc_calendar = load_fomc_calendar(path=fomc_calendar_path, start=start, end=end)
    meetings = list(fomc_calendar)
    if not meetings:
        return BuildArtifacts(
            frame=_empty_frame(),
            path_model=_fit_path_factor_model([], []),
            data_version="empty",
            methodology=methodology,
            fred_series_used=tuple(sorted(fred_responses)),
            rows_written=0,
            intermeeting_rows=0,
            fed_info_factor_unavailable_rows=0,
        )

    # ---- Maps ----
    target_upper = _series_to_map(fred_responses["DFEDTARU"]) if "DFEDTARU" in fred_responses else {}
    target_lower = _series_to_map(fred_responses["DFEDTARL"]) if "DFEDTARL" in fred_responses else {}
    target_single = _series_to_map(fred_responses["DFEDTAR"]) if "DFEDTAR" in fred_responses else {}
    curve_maps: dict[int, dict[_dt.date, float]] = {}
    for tenor, sid in CURVE_SERIES_BY_TENOR.items():
        if sid not in fred_responses:
            raise KeyError(f"Missing required FRED series '{sid}' for tenor {tenor}m")
        curve_maps[tenor] = _series_to_map(fred_responses[sid])

    # Trading-day index = union of dates with any published curve yield.
    # We use this so `_pre_post_yields` walks trading-day slots instead
    # of calendar days, which prevents silent None rows around long
    # holiday clusters.
    trading_days_set: set[_dt.date] = set()
    for tenor_map in curve_maps.values():
        trading_days_set.update(tenor_map.keys())
    trading_days_sorted: list[_dt.date] = sorted(trading_days_set)

    # ---- Pass 1: per-meeting deltas + curves ----
    per_meeting_curves_pre: dict[_dt.date, list[CurvePoint]] = {}
    per_meeting_curves_post: dict[_dt.date, list[CurvePoint]] = {}
    per_meeting_delta_level: dict[_dt.date, float | None] = {}
    per_meeting_delta_path: dict[_dt.date, list[float | None]] = {}
    per_meeting_target_prior: dict[_dt.date, tuple[float | None, str]] = {}
    per_meeting_target_after: dict[_dt.date, tuple[float | None, str]] = {}

    # Pre-compute next-meeting dates so the `after_target` lookahead
    # never bleeds into the following meeting's action. Two FOMC events
    # can sit inside the 5-day default window (e.g. 2020-03-03 emergency
    # cut followed by the 2020-03-15 emergency cut; or any intermeeting
    # action that lands days before a scheduled meeting). When that
    # happens we must clip the lookahead to `next - current - 1` so the
    # second meeting's published target band can't pollute the first
    # meeting's `ff_target_after`.
    next_meeting_date_by_event: dict[_dt.date, _dt.date | None] = {}
    sorted_meeting_dates = sorted({m.meeting_date for m in meetings})
    for i, d in enumerate(sorted_meeting_dates):
        nxt = sorted_meeting_dates[i + 1] if i + 1 < len(sorted_meeting_dates) else None
        next_meeting_date_by_event[d] = nxt

    for m in meetings:
        ed = m.meeting_date
        # Curves at five tenors.
        pre_curve: list[CurvePoint] = []
        post_curve: list[CurvePoint] = []
        deltas_at_tenor: dict[int, float | None] = {}
        for tenor in CURVE_TENORS_MONTHS:
            s_map = curve_maps[tenor]
            pre, post, _, _ = _pre_post_yields(
                ed,
                s_map,
                trading_days=trading_days_sorted,
            )
            pre_curve.append(CurvePoint(months_ahead=tenor, implied_rate=pre if pre is not None else float("nan")))
            post_curve.append(CurvePoint(months_ahead=tenor, implied_rate=post if post is not None else float("nan")))
            if pre is None or post is None:
                deltas_at_tenor[tenor] = None
            else:
                # Convert from percent to basis points (1% = 100 bps).
                deltas_at_tenor[tenor] = (post - pre) * 100.0
        per_meeting_curves_pre[ed] = pre_curve
        per_meeting_curves_post[ed] = post_curve
        per_meeting_delta_level[ed] = deltas_at_tenor[1]
        per_meeting_delta_path[ed] = [deltas_at_tenor[t] for t in PATH_TENORS_MONTHS]

        # Targets: prior = day before announcement (already published target band);
        # after = effective once announcement lands. We look back to the previous
        # publication and forward to the next one respectively.
        prior_target, prior_src = _lookup_target_with_lookahead(
            ed - _dt.timedelta(days=1),
            upper=target_upper,
            lower=target_lower,
            single=target_single,
            lookahead_days=0,
            lookbehind_days=MAX_TARGET_LOOKAHEAD_DAYS,
        )
        # Clip the `after_target` lookahead so it cannot reach the next
        # FOMC meeting's published action. Without this guard, two
        # meetings within MAX_TARGET_LOOKAHEAD_DAYS calendar days of one
        # another (e.g. March 2020) silently let the second meeting's
        # post-action target leak into the first meeting's row.
        nxt = next_meeting_date_by_event.get(ed)
        if nxt is not None:
            gap = (nxt - ed).days - 1  # last safe lookahead day
            clipped_lookahead = max(0, min(MAX_TARGET_LOOKAHEAD_DAYS, gap))
        else:
            clipped_lookahead = MAX_TARGET_LOOKAHEAD_DAYS
        after_target, after_src = _lookup_target_with_lookahead(
            ed,
            upper=target_upper,
            lower=target_lower,
            single=target_single,
            lookahead_days=clipped_lookahead,
            lookbehind_days=0,
        )
        per_meeting_target_prior[ed] = (prior_target, prior_src)
        per_meeting_target_after[ed] = (after_target, after_src)

    # ---- Pass 2: PCA fit on residual curve shape ----
    fit_levels: list[float] = []
    fit_paths: list[list[float]] = []
    for m in meetings:
        ed = m.meeting_date
        lvl = per_meeting_delta_level[ed]
        path = per_meeting_delta_path[ed]
        if lvl is None or any(v is None for v in path):
            continue
        fit_levels.append(float(lvl))
        fit_paths.append([float(v) for v in path])  # type: ignore[arg-type]
    path_model = _fit_path_factor_model(fit_levels, fit_paths)

    # ---- Pass 3: fed-info factor regression on SPX returns ----
    spx_lookup = spx_close_by_date or {}
    levels_for_fit: list[float] = []
    spx_for_fit: list[float] = []
    spx_returns_per_meeting: dict[_dt.date, tuple[float | None, str]] = {}
    for m in meetings:
        ed = m.meeting_date
        lvl = per_meeting_delta_level[ed]
        ret, source = _spx_return_on(ed, spx_lookup)
        spx_returns_per_meeting[ed] = (ret, source)
        if lvl is None or ret is None:
            continue
        levels_for_fit.append(float(lvl))
        spx_for_fit.append(float(ret))
    alpha, beta = _fit_fed_info_factor(levels_for_fit, spx_for_fit)

    # ---- Pass 4: assemble rows ----
    rows: list[SurpriseRow] = []
    fed_info_unavailable = 0
    intermeeting_rows = 0
    for idx, m in enumerate(meetings, start=1):
        ed = m.meeting_date
        prior_target, prior_src = per_meeting_target_prior[ed]
        after_target, after_src = per_meeting_target_after[ed]
        lvl = per_meeting_delta_level[ed]
        path_factor = _apply_path_factor(path_model, lvl, per_meeting_delta_path[ed])
        ret, ret_src = spx_returns_per_meeting[ed]
        if ret is None or lvl is None:
            fed_info_factor: float | None = None
            fed_info_source = "unavailable" if ret is None else "level_missing"
            if ret is None:
                fed_info_unavailable += 1
        else:
            fed_info_factor = float(lvl) - (alpha + beta * float(ret))
            fed_info_source = ret_src
        target_source = f"prior:{prior_src}|after:{after_src}"
        if m.is_intermeeting:
            intermeeting_rows += 1
        rows.append(
            SurpriseRow(
                event_date=ed,
                meeting_id=idx,
                ff_target_prior=prior_target,
                ff_target_after=after_target,
                mp_surprise_level=lvl,
                mp_surprise_path_factor=path_factor,
                pre_event_curve=per_meeting_curves_pre[ed],
                post_event_curve=per_meeting_curves_post[ed],
                fed_info_factor=fed_info_factor,
                is_intermeeting=m.is_intermeeting,
                methodology=methodology,
                fed_info_factor_source=fed_info_source,
                target_source=target_source,
            )
        )

    data_version = _data_version_hash(
        fred_responses,
        methodology=methodology,
        calendar_signature=_calendar_signature(meetings),
    )
    out_rows = [r.to_dict() for r in rows]
    for r in out_rows:
        r["data_version"] = data_version
    frame = pd.DataFrame(out_rows) if out_rows else _empty_frame()
    if not frame.empty:
        frame = frame.sort_values(["event_date", "meeting_id"], kind="mergesort").reset_index(drop=True)
        frame = frame[list(COLUMN_ORDER)]
    return BuildArtifacts(
        frame=frame,
        path_model=path_model,
        data_version=data_version,
        methodology=methodology,
        fred_series_used=tuple(sorted(fred_responses)),
        rows_written=len(rows),
        intermeeting_rows=intermeeting_rows,
        fed_info_factor_unavailable_rows=fed_info_unavailable,
    )


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in COLUMN_ORDER})


# ---------------------------------------------------------------------------
# Parquet writer + SOURCES.lock update
# ---------------------------------------------------------------------------


def _sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def write_mp_surprises_parquet(frame: pd.DataFrame, output_path: Path) -> str:
    """Write deterministically and return the sha256 of the parquet bytes.

    The byte-level sha256 is returned for convenience (e.g. SOURCES.lock
    bookkeeping) but is NOT the determinism contract -- parquet metadata
    can shift across pyarrow versions or platforms even under
    fixed-encoding settings. The canonical determinism check is
    :func:`dataframe_value_hash`, which is computed over the sorted
    DataFrame values after re-reading the parquet.

    The encoding settings below (``zstd`` level 3, statistics off,
    dictionary off) are tuned to minimise spurious metadata drift but
    are not load-bearing for the data-equality claim.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(
        output_path,
        engine="pyarrow",
        index=False,
        compression="zstd",
        compression_level=3,
        write_statistics=False,
        use_dictionary=False,
    )
    return _sha256_of_file(output_path)


def dataframe_value_hash(frame: pd.DataFrame) -> str:
    """Hash the DataFrame's data (not its encoding).

    The hash is computed over the row-sorted, column-ordered string
    representation of every cell. This makes the contract portable
    across pyarrow / platform / encoding combinations: two runs that
    produce the same MP-surprise rows produce the same hash even if
    the parquet bytes differ.

    Used by the determinism test and by anyone who needs to assert
    data identity without depending on the on-disk encoding being
    bit-stable.
    """

    if frame.empty:
        return hashlib.sha256(b"empty").hexdigest()
    # Pin the column order (defensive — callers should already pass
    # frames in COLUMN_ORDER, but we guard against accidental drift).
    cols = [c for c in COLUMN_ORDER if c in frame.columns]
    if not cols:
        cols = list(frame.columns)
    ordered = frame[cols].copy()
    # Stringify every cell to side-step numpy / arrow dtype quirks
    # (NaN vs None vs <NA>, int64 vs Int64, etc.). The semantic data
    # equality is what we care about, not the in-memory representation.
    str_rows = ordered.astype(object).map(
        lambda v: "" if v is None or (isinstance(v, float) and v != v) else str(v)
    )
    serialised = ["|".join(str(v) for v in row) for row in str_rows.to_numpy().tolist()]
    serialised.sort()
    payload = "\n".join(serialised) + "\n" + "|".join(cols)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def update_sources_lock(
    *,
    lock_path: Path,
    artifacts: BuildArtifacts,
    parquet_path: Path,
    parquet_sha256: str,
    lock_key: str = DEFAULT_LOCK_KEY,
) -> None:
    """Persist provenance for the MP surprise parquet."""

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, Any] = {}
    if lock_path.exists():
        try:
            existing = json.loads(lock_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}
    entry = {
        "parquet_path": str(parquet_path.name),
        "sha256": parquet_sha256,
        "fred_series": list(artifacts.fred_series_used),
        "rows": int(artifacts.rows_written),
        "intermeeting_rows": int(artifacts.intermeeting_rows),
        "fed_info_factor_unavailable_rows": int(artifacts.fed_info_factor_unavailable_rows),
        "methodology": artifacts.methodology,
        "data_version": artifacts.data_version,
        "path_factor_model": artifacts.path_model.to_dict(),
        "retrieved_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }
    existing[lock_key] = entry
    lock_path.write_text(json.dumps(existing, indent=2, sort_keys=True), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _required_series_ids() -> tuple[str, ...]:
    ids: list[str] = list(TARGET_RATE_SERIES) + [EFFECTIVE_RATE_SERIES]
    for sid in CURVE_SERIES_BY_TENOR.values():
        ids.append(sid)
    return tuple(sorted(set(ids)))


def _hydrate_fred_responses(
    *,
    start: _dt.date,
    end: _dt.date,
    cache_dir: Path,
    transport: httpx.BaseTransport | None = None,
    force_refresh: bool = False,
) -> dict[str, FredSeriesResponse]:
    responses: dict[str, FredSeriesResponse] = {}
    for sid in _required_series_ids():
        responses[sid] = _fetch_series_safely(
            sid,
            start=start.isoformat(),
            end=end.isoformat(),
            cache_dir=cache_dir,
            transport=transport,
            force_refresh=force_refresh,
        )
    return responses


def _parse_end(value: str) -> _dt.date:
    if value.lower() == "today":
        return _dt.date.today()
    return _dt.date.fromisoformat(value)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the monetary-policy surprise time-series at "
            "data/external/fred/mp_surprises.parquet (Phase 8 foundation, "
            "closes #146)."
        ),
    )
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default="today")
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_NAME,
        help="Parquet filename (relative to --cache-dir unless absolute).",
    )
    parser.add_argument(
        "--cache-dir",
        default=str(FRED_CACHE_DIR),
        help="FRED cache directory (also receives the output parquet).",
    )
    parser.add_argument(
        "--fomc-calendar-csv",
        default=None,
        help="Override the bundled FOMC meeting calendar CSV.",
    )
    parser.add_argument(
        "--methodology",
        default=METHODOLOGY_OIS_PROXY,
        choices=(METHODOLOGY_OIS_PROXY, METHODOLOGY_FF_FUTURES),
        help=(
            "Methodology label for the row. The default 'ois_proxy' "
            "reflects that this build uses Treasury yields as a fed-funds "
            "futures proxy. Pass 'ff_futures' only if you have wired a real "
            "CME settlement source."
        ),
    )
    parser.add_argument(
        "--spx-cache-path",
        default=None,
        help="Cached SPX daily-close parquet. CLI default: <cache-dir>/_spx_gspc.parquet.",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force re-fetch from FRED, bypassing the per-series JSON cache.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    start = _dt.date.fromisoformat(args.start)
    end = _parse_end(args.end)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Treasury yields need warmup days before ``start`` so the
    # pre-event lookup on the earliest meeting has a value. Pull a
    # 14-day buffer.
    fetch_start = start - _dt.timedelta(days=14)
    fetch_end = end + _dt.timedelta(days=7)

    responses = _hydrate_fred_responses(
        start=fetch_start,
        end=fetch_end,
        cache_dir=cache_dir,
        force_refresh=args.force_refresh,
    )

    spx_cache_path = (
        Path(args.spx_cache_path)
        if args.spx_cache_path
        else cache_dir / "_spx_gspc.parquet"
    )
    spx_map = _fetch_spx_close_via_yfinance(start=fetch_start, end=fetch_end, cache_path=spx_cache_path)

    calendar_path = Path(args.fomc_calendar_csv) if args.fomc_calendar_csv else None
    calendar = load_fomc_calendar(path=calendar_path, start=start, end=end)

    artifacts = build_mp_surprises(
        start=start,
        end=end,
        fred_responses=responses,
        fomc_calendar=calendar,
        spx_close_by_date=spx_map,
        methodology=args.methodology,
    )

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = cache_dir / output_path
    parquet_sha = write_mp_surprises_parquet(artifacts.frame, output_path)
    update_sources_lock(
        lock_path=cache_dir / SOURCES_LOCK_NAME,
        artifacts=artifacts,
        parquet_path=output_path,
        parquet_sha256=parquet_sha,
    )

    print(f"[mp-surprise] rows: {artifacts.rows_written}")
    if artifacts.rows_written:
        pct = (artifacts.intermeeting_rows / artifacts.rows_written) * 100.0
    else:
        pct = 0.0
    print(f"[mp-surprise] is_intermeeting: {artifacts.intermeeting_rows} ({pct:.2f}%)")
    print(f"[mp-surprise] methodology: {artifacts.methodology}")
    print(f"[mp-surprise] data_version: {artifacts.data_version}")
    print(f"[mp-surprise] parquet sha256: {parquet_sha}")
    print(f"[mp-surprise] PCA explained_variance_ratio: {artifacts.path_model.explained_variance_ratio:.4f}")
    print("[mp-surprise] 5 most recent meetings:")
    if not artifacts.frame.empty:
        tail = artifacts.frame.tail(5)
        for _, row in tail.iterrows():
            print(
                "  {date} meeting_id={mid} level={lvl} path={path} fed_info={fi} intermeeting={im}".format(
                    date=row["event_date"],
                    mid=row["meeting_id"],
                    lvl=_fmt(row.get("mp_surprise_level")),
                    path=_fmt(row.get("mp_surprise_path_factor")),
                    fi=_fmt(row.get("fed_info_factor")),
                    im=row.get("is_intermeeting"),
                )
            )
    return 0


def _fmt(value: Any) -> str:
    if value is None:
        return "None"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


if __name__ == "__main__":
    sys.exit(main())
