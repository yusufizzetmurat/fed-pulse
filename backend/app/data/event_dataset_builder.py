"""Phase 8 event-row dataset builder.

Produces one row per ``(FOMC event date x event_kind x forecast horizon)``
for downstream event-study forecasting. Each row carries the document text,
multi-axis labels (where available), a 4-vector credibility check, a 20
trading-day prior market window, and abnormal-return targets at horizons
``h in {1d, 5d, 10d, 30d}``.

Two outputs are emitted side-by-side in the same CLI run:

- ``data/processed/<pkg>/events.parquet``        -- collapsed view, one row
  per ``(event_date, event_kind, asset_symbol, horizon)``. Multi-source
  duplicates are pinned to one preferred source via ``_SOURCE_PREFERENCE``.
- ``data/processed/<pkg>/events_full.parquet``    -- full view, one row per
  ``(event_date, event_kind, source, source_record_id, asset_symbol, horizon)``.
  Keeps sentence-level data from every source for source-stratified and
  sentence-level analyses.

Both parquets share the same column schema; downstream consumers pick the
view that fits their question.

Schema (one row per event x kind x horizon x asset):

- ``event_date``                ISO date (string)
- ``event_kind``                One of ``{statement, minutes, speech,
                                 testimony, press_conference}``
- ``document_id``               sha256(source|event_date|event_kind)[:16]
- ``text_hash``                 sha256 of the concatenated document text
- ``source``                    Registry source (preferred order documented
                                 in ``_SOURCE_PREFERENCE``)
- ``as_of_ts``                  ISO 8601 timestamp at which the event
                                 announcement becomes public knowledge.
                                 Placeholder convention:
                                 ``T19:00:00Z`` for FOMC statements /
                                 minutes / press conferences / testimonies
                                 (2pm ET = 19:00 UTC during US standard time
                                 -- we use the wall-clock 19:00Z throughout
                                 so the dataset is reproducible regardless
                                 of DST), and ``T14:00:00Z`` for speeches
                                 (typical morning ET delivery).
- ``text``                      Raw concatenated document text
- ``token_count``               Whitespace-token count of ``text`` (used as
                                 a coarse truncation budget upstream)
- ``axis_stance``               ``hawkish | dovish | neutral`` or None
- ``axis_time``, ``axis_certainty``, ``axis_factor``, ``axis_topic``
                                Multi-axis labels (None when unavailable)
- ``credibility_*``             Four credibility-vector axes
- ``prior_window_sha256``       sha256 over the concatenated prior bars,
                                 written for reproducibility
- ``prior_bars_json``           JSON-encoded list of 20 prior bars
                                 (``close``, ``volume``, ``vol_5d``,
                                 ``cum_return_20d``) -- parquet does not
                                 store nested arrays cleanly across
                                 versions, so JSON keeps the bytes stable.
- ``asset_symbol``              Default ``^GSPC``; schema supports per-asset
                                 expansion in future sweeps without
                                 changing the contract
- ``horizon``                   Trading-day horizon (1, 5, 10, 30)
- ``realized_return``           Raw close-to-close return over horizon
- ``abnormal_return``           ``realized_return - (alpha + beta * SPX)``;
                                 when asset is ``^GSPC`` itself this is
                                 just the raw return (alpha=0, beta=1)
- ``alpha``, ``beta``           Market-model parameters fit on the
                                 trailing 252-day window ending strictly
                                 before ``as_of_ts``
- ``direction_t1d``             Sign of the t+1d realized return:
                                 ``-1``, ``0``, ``+1``
- ``volatility_shift``          10d post-event realized vol minus 10d
                                 pre-event realized vol
- ``concurrent_macro_release``  Boolean -- True when a major macro release
                                 (CPI, NFP, ISM) falls within +/-2 trading
                                 days. Heuristic dates only; we flag, never
                                 drop.

Methodological constraints, enforced by assertions in the builder:

1. No look-ahead. The last bar in the 20-day prior window has a date
   strictly less than ``as_of_ts.date()``. The market-model regression
   window ends strictly before that same date.
2. No survivorship filter. Every FOMC event with text + event_date + a
   usable prior window produces a row, regardless of whether the post-
   event move was large.
3. Deterministic: builds on the same inputs are byte-identical. Rows are
   sorted by ``(event_date, event_kind, asset_symbol, horizon)`` before
   writing, and parquet is written with snappy compression (which has no
   timestamp metadata) via pandas.
4. Idempotent: re-running with the same training package leaves the output
   parquet content unchanged.

CLI:
    python -m app.data.event_dataset_builder \
        --training-package-id <id> --asset ^GSPC --output events.parquet
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

from app.config import DATA_DIR as DEFAULT_DATA_DIR
from app.data.macro_releases import (
    DEFAULT_MACRO_RELEASES_CSV,
    MacroReleaseCalendar,
    build_heuristic_calendar,
    load_macro_release_calendar,
)
from app.features.credibility import CredibilityVector
from app.services.credibility_loader import load_credibility_for_run

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_ASSET = "^GSPC"
DEFAULT_BENCHMARK = "^GSPC"
DEFAULT_HORIZONS = (1, 5, 10, 30)
PRIOR_WINDOW_DAYS = 20
MARKET_MODEL_WINDOW_DAYS = 252
VOL_WINDOW_DAYS = 10
ROLLING_VOL_DAYS = 5
CONCURRENT_MACRO_TRADING_DAY_RADIUS = 2

# Two announcement-time placeholders. Documented in module docstring; if a
# subsequent issue lands a real per-event timestamp source (#146 OIS surprise)
# replace these constants and re-build.
FOMC_AS_OF_TIME = "T19:00:00Z"
SPEECH_AS_OF_TIME = "T14:00:00Z"

# Map registry document_type values (raw, mixed-case) onto the canonical
# event_kind taxonomy. Anything not listed is dropped silently and counted
# in the build summary.
_EVENT_KIND_MAP: dict[str, str] = {
    "statement": "statement",
    "Statement": "statement",
    "minutes": "minutes",
    "Minutes": "minutes",
    "meeting_transcript": "minutes",  # FOMC transcripts share the minutes window
    "press_conference": "press_conference",
    "congressional_testimony": "testimony",
    "chair_speech": "speech",
    "governor_speech": "speech",
}

# Speech-like kinds use the speech time placeholder; everything else uses
# the FOMC 2pm ET placeholder.
_SPEECH_KINDS = frozenset({"speech", "testimony"})

# When multiple registry sources cover the same (event_date, event_kind),
# pick the row with the highest preference rank. This avoids near-duplicate
# event rows while keeping the choice deterministic. Higher-quality / more
# complete document sources are preferred over sentence-level shards.
_SOURCE_PREFERENCE: tuple[str, ...] = (
    "scraped_fed",
    "vtasca_fomc_archive",
    "op_fed",
    "gtfintechlab_federal_reserve_system",
    "hf_fomc_communication",
    "kaggle_fed_statements_minutes",
    "gss_factor",
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class _RegistryRow:
    source: str
    source_record_id: str
    document_type: str
    event_date: str
    text: str
    mapped_label: str | None
    multi_axis_extras: dict[str, Any]
    axes: dict[str, Any]


@dataclass
class _EventDoc:
    """One aggregated document for a single (source, event_date, event_kind)."""

    source: str
    event_date: str
    event_kind: str
    text: str
    record_ids: list[str]
    multi_axis: dict[str, str | None]

    @property
    def source_record_id(self) -> str:
        """Stable identifier for one (source, event_date, event_kind) shard.

        Uses the concatenation of all aggregated ``source_record_id`` values
        joined by ``|``. Sorted at aggregation time so the value is
        deterministic.
        """

        return "|".join(self.record_ids)


@dataclass
class _PriorBar:
    date: _dt.date
    close: float
    volume: float
    vol_5d: float
    cum_return_20d: float


@dataclass
class _BuildSummary:
    rows_written: int = 0
    events_emitted: int = 0
    dropped_no_text: int = 0
    dropped_unmapped_kind: int = 0
    dropped_no_prior_window: int = 0
    dropped_no_targets: int = 0
    concurrent_macro_release_rows: int = 0
    per_source_rows: dict[str, int] = field(default_factory=dict)
    per_kind_rows: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Registry loading + event aggregation
# ---------------------------------------------------------------------------


def _load_registry_rows(package_dir: Path) -> list[_RegistryRow]:
    path = package_dir / "registry_normalized.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing registry: {path}")
    rows: list[_RegistryRow] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        text = str(payload.get("text", "") or "").strip()
        event_date = str(payload.get("event_date", "") or "").strip()
        if not text or not event_date:
            continue
        mapped = payload.get("mapped_label")
        rows.append(
            _RegistryRow(
                source=str(payload.get("source", "") or ""),
                source_record_id=str(payload.get("source_record_id", "") or ""),
                document_type=str(payload.get("document_type", "") or ""),
                event_date=event_date,
                text=text,
                mapped_label=str(mapped).lower() if mapped else None,
                multi_axis_extras=dict(payload.get("multi_axis_extras") or {}),
                axes=dict(payload.get("axes") or {}),
            )
        )
    return rows


def _aggregate_events(rows: Iterable[_RegistryRow]) -> list[_EventDoc]:
    """Aggregate registry rows into one document per ``(source, date, kind)``.

    Sentence-level shards (TDW, gtfintechlab, GSS) get concatenated in
    ``source_record_id`` order; full-document sources stay as-is.
    """

    groups: dict[tuple[str, str, str], list[_RegistryRow]] = defaultdict(list)
    for row in rows:
        kind = _EVENT_KIND_MAP.get(row.document_type)
        if kind is None:
            continue
        groups[(row.source, row.event_date, kind)].append(row)

    docs: list[_EventDoc] = []
    for (source, event_date, kind), bucket in groups.items():
        bucket_sorted = sorted(bucket, key=lambda r: r.source_record_id)
        text = "\n".join(r.text for r in bucket_sorted if r.text)
        if not text.strip():
            continue
        multi_axis: dict[str, str | None] = {
            "stance": None,
            "time": None,
            "certainty": None,
            "factor": None,
            "topic": None,
        }
        # Use the first row whose mapped_label or axes carry a value as the
        # document-level label. Deterministic because bucket is sorted.
        for r in bucket_sorted:
            if multi_axis["stance"] is None and r.mapped_label:
                multi_axis["stance"] = r.mapped_label
            for axis_name in ("time", "certainty", "factor", "topic"):
                if multi_axis[axis_name] is None:
                    val = r.axes.get(axis_name)
                    if val:
                        multi_axis[axis_name] = str(val)
            # multi_axis_extras (Op-Fed opinion etc.) is recorded only via
            # axes; the raw extras dict is not lifted into event rows.
        docs.append(
            _EventDoc(
                source=source,
                event_date=event_date,
                event_kind=kind,
                text=text,
                record_ids=[r.source_record_id for r in bucket_sorted],
                multi_axis=multi_axis,
            )
        )
    return docs


def _choose_preferred(docs: list[_EventDoc]) -> list[_EventDoc]:
    """Collapse multi-source duplicates to one chosen ``_EventDoc`` per event.

    For the same ``(event_date, event_kind)``, keep the source ranked highest
    in ``_SOURCE_PREFERENCE``; ties broken by source name (sorted).
    """

    rank = {src: i for i, src in enumerate(_SOURCE_PREFERENCE)}

    def _key(doc: _EventDoc) -> tuple[int, str]:
        return (rank.get(doc.source, len(_SOURCE_PREFERENCE)), doc.source)

    grouped: dict[tuple[str, str], list[_EventDoc]] = defaultdict(list)
    for doc in docs:
        grouped[(doc.event_date, doc.event_kind)].append(doc)

    chosen: list[_EventDoc] = []
    for bucket in grouped.values():
        bucket.sort(key=_key)
        chosen.append(bucket[0])
    chosen.sort(key=lambda d: (d.event_date, d.event_kind, d.source))
    return chosen


# ---------------------------------------------------------------------------
# Market series
# ---------------------------------------------------------------------------


def _date(value: str | _dt.date) -> _dt.date:
    if isinstance(value, _dt.date):
        return value
    return _dt.date.fromisoformat(str(value)[:10])


@dataclass
class _CloseSeries:
    """Lightweight close+volume series indexed by trading-day date."""

    dates: list[_dt.date]
    close: list[float]
    volume: list[float]

    def __len__(self) -> int:
        return len(self.dates)

    def index_strictly_before(self, target: _dt.date) -> int:
        """Largest index with ``dates[i] < target``; -1 if none."""
        lo, hi = 0, len(self.dates)
        while lo < hi:
            mid = (lo + hi) // 2
            if self.dates[mid] < target:
                lo = mid + 1
            else:
                hi = mid
        return lo - 1

    def index_on_or_after(self, target: _dt.date) -> int:
        """Smallest index with ``dates[i] >= target``; ``len`` if none."""
        lo, hi = 0, len(self.dates)
        while lo < hi:
            mid = (lo + hi) // 2
            if self.dates[mid] < target:
                lo = mid + 1
            else:
                hi = mid
        return lo


def _fetch_close_series(
    symbol: str,
    *,
    start: _dt.date,
    end: _dt.date,
    cache_dir: Path | None = None,
    force_refresh: bool = False,
) -> _CloseSeries:
    """Fetch (or load cached) daily close+volume for ``symbol``.

    The cache layout mirrors :mod:`app.services.fred_client`: one parquet per
    symbol in ``cache_dir`` with a ``SOURCES.lock`` recording the SHA. Tests
    write a parquet directly and never hit yfinance; the smoke run hits live
    yfinance once and reuses the cache forever after.
    """

    cache_path = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        safe = symbol.replace("^", "").replace("=", "_").replace("/", "_").replace(":", "_")
        cache_path = cache_dir / f"{safe}.parquet"
        if cache_path.exists() and not force_refresh:
            frame = pd.read_parquet(cache_path)
            return _frame_to_series(frame)

    import yfinance as yf

    ticker = yf.Ticker(symbol)
    frame = ticker.history(
        start=start.isoformat(),
        end=(end + _dt.timedelta(days=1)).isoformat(),
        auto_adjust=True,
    )
    if frame.empty:
        raise RuntimeError(f"yfinance returned no data for {symbol} in [{start}, {end}]")
    dates = [idx.date() for idx in frame.index]
    close = [float(v) for v in frame["Close"].to_numpy()]
    volume_series = frame["Volume"] if "Volume" in frame.columns else None
    if volume_series is not None:
        volume = [float(v) for v in volume_series.to_numpy()]
    else:
        volume = [0.0] * len(close)
    series = _CloseSeries(dates=dates, close=close, volume=volume)
    if cache_path is not None:
        out = pd.DataFrame(
            {
                "symbol": symbol,
                "date": [d.isoformat() for d in series.dates],
                "close": series.close,
                "volume": series.volume,
            }
        )
        out.to_parquet(cache_path, index=False)
        _update_sources_lock(cache_path, symbol)
    return series


def _frame_to_series(frame: pd.DataFrame) -> _CloseSeries:
    if "date" not in frame.columns or "close" not in frame.columns:
        raise RuntimeError(
            "Cached market parquet must contain 'date' and 'close' columns; "
            f"got {list(frame.columns)}"
        )
    parsed = sorted(
        zip(
            [_date(d) for d in frame["date"].tolist()],
            [float(c) for c in frame["close"].tolist()],
            [float(v) for v in (frame["volume"].tolist() if "volume" in frame.columns else [0.0] * len(frame))],
        ),
        key=lambda t: t[0],
    )
    return _CloseSeries(
        dates=[t[0] for t in parsed],
        close=[t[1] for t in parsed],
        volume=[t[2] for t in parsed],
    )


def _update_sources_lock(parquet_path: Path, symbol: str) -> None:
    lock_path = parquet_path.parent / "SOURCES.lock"
    digest = hashlib.sha256(parquet_path.read_bytes()).hexdigest()
    if lock_path.exists():
        try:
            payload = json.loads(lock_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
    else:
        payload = {}
    entries = payload.setdefault("entries", {})
    entries[symbol] = {
        "parquet_path": parquet_path.name,
        "sha256": digest,
    }
    payload["format_version"] = 1
    lock_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


# ---------------------------------------------------------------------------
# Market computations: prior bars, market-model, targets, vol shift
# ---------------------------------------------------------------------------


def _log_returns(closes: Sequence[float]) -> list[float]:
    import math

    out: list[float] = []
    for i in range(1, len(closes)):
        prev = closes[i - 1]
        cur = closes[i]
        if prev <= 0 or cur <= 0:
            out.append(0.0)
        else:
            out.append(math.log(cur / prev))
    return out


def _build_prior_window(
    series: _CloseSeries,
    as_of: _dt.date,
    *,
    window_days: int = PRIOR_WINDOW_DAYS,
    vol_window: int = ROLLING_VOL_DAYS,
) -> list[_PriorBar] | None:
    """Return ``window_days`` prior bars ending strictly before ``as_of``.

    Returns None when the series lacks enough history. Each bar carries
    rolling 5-day log-return std and cumulative simple return over the
    20-day window (anchored at the first bar in the window).
    """

    last_idx = series.index_strictly_before(as_of)
    if last_idx < 0:
        return None
    start_idx = last_idx - window_days + 1
    # Need vol_window+1 prior closes to compute rolling 5d std for the first
    # bar in the window, and window_days closes to compute cum_return.
    # cum_return baseline is the bar immediately preceding the window.
    if start_idx - 1 < 0:
        return None
    if start_idx - vol_window < 0:
        return None

    log_rets_all = _log_returns(series.close[: last_idx + 1])
    # log_rets_all[i] corresponds to close[i+1]. Rolling std at close-index j
    # uses log_rets_all[j - vol_window .. j-1].
    bars: list[_PriorBar] = []
    base_close = series.close[start_idx - 1]
    for offset in range(window_days):
        i = start_idx + offset
        # rolling 5d log-return std anchored on bar i
        if i - vol_window < 0:
            return None
        chunk = log_rets_all[i - vol_window : i]
        if len(chunk) < vol_window:
            return None
        mean = sum(chunk) / vol_window
        var = sum((x - mean) ** 2 for x in chunk) / (vol_window - 1)
        vol = var**0.5
        cum_return = (series.close[i] - base_close) / base_close if base_close > 0 else 0.0
        bars.append(
            _PriorBar(
                date=series.dates[i],
                close=series.close[i],
                volume=series.volume[i],
                vol_5d=vol,
                cum_return_20d=cum_return,
            )
        )
    # Enforce no look-ahead: last prior bar must be < as_of
    assert bars[-1].date < as_of, (
        f"prior-window contract violated: last bar {bars[-1].date} not < as_of {as_of}"
    )
    return bars


def _fit_market_model(
    asset_returns: Sequence[float],
    bench_returns: Sequence[float],
) -> tuple[float, float]:
    """OLS regression ``asset = alpha + beta * bench``.

    Returns ``(alpha, beta)``. Falls back to ``(0.0, 1.0)`` when input is
    degenerate (empty, single point, or zero benchmark variance).
    """

    n = min(len(asset_returns), len(bench_returns))
    if n < 2:
        return (0.0, 1.0)
    a = list(asset_returns[-n:])
    b = list(bench_returns[-n:])
    mean_a = sum(a) / n
    mean_b = sum(b) / n
    cov = sum((ai - mean_a) * (bi - mean_b) for ai, bi in zip(a, b)) / n
    var_b = sum((bi - mean_b) ** 2 for bi in b) / n
    if var_b <= 1e-18:
        return (0.0, 1.0)
    beta = cov / var_b
    alpha = mean_a - beta * mean_b
    return (alpha, beta)


def _market_model_for_event(
    asset_series: _CloseSeries,
    bench_series: _CloseSeries,
    as_of: _dt.date,
    *,
    window_days: int = MARKET_MODEL_WINDOW_DAYS,
) -> tuple[float, float] | None:
    """Fit alpha, beta on the trailing ``window_days`` daily returns ending
    strictly before ``as_of``. Returns None when either series lacks
    enough history."""

    a_last = asset_series.index_strictly_before(as_of)
    b_last = bench_series.index_strictly_before(as_of)
    if a_last < window_days or b_last < window_days:
        return None

    # Daily returns aligned on date. Build a join on dates within the window.
    a_dates = asset_series.dates[a_last - window_days : a_last + 1]
    a_closes = asset_series.close[a_last - window_days : a_last + 1]
    b_date_to_close = dict(
        zip(
            bench_series.dates[: b_last + 1],
            bench_series.close[: b_last + 1],
        )
    )
    paired_a: list[float] = []
    paired_b: list[float] = []
    for i in range(1, len(a_dates)):
        d_prev, d_cur = a_dates[i - 1], a_dates[i]
        if d_prev not in b_date_to_close or d_cur not in b_date_to_close:
            continue
        a_ret = (a_closes[i] - a_closes[i - 1]) / a_closes[i - 1] if a_closes[i - 1] > 0 else 0.0
        bp, bc = b_date_to_close[d_prev], b_date_to_close[d_cur]
        b_ret = (bc - bp) / bp if bp > 0 else 0.0
        paired_a.append(a_ret)
        paired_b.append(b_ret)
    if len(paired_a) < window_days // 4:
        return None
    return _fit_market_model(paired_a, paired_b)


def _realized_returns(
    series: _CloseSeries,
    as_of: _dt.date,
    horizons: Sequence[int],
) -> dict[int, tuple[float, _dt.date | None]] | None:
    """Compute ``(close_{t-1} -> close_{t-1+h})`` returns for each horizon.

    The base is the last close strictly before ``as_of``. The target close
    is the ``h``-th trading day on-or-after ``as_of`` (i.e. event-day close
    counted as h=1 when event_date is itself a trading day; otherwise the
    next trading day).
    """

    base_idx = series.index_strictly_before(as_of)
    if base_idx < 0:
        return None
    base = series.close[base_idx]
    if base <= 0:
        return None
    on_or_after = series.index_on_or_after(as_of)
    if on_or_after >= len(series):
        return {}
    # Trading-day offset 1 == first trading day with date >= as_of.date()
    out: dict[int, tuple[float, _dt.date | None]] = {}
    for h in horizons:
        idx = on_or_after + (h - 1)
        if idx < 0 or idx >= len(series):
            continue
        close_h = series.close[idx]
        if close_h <= 0:
            continue
        out[h] = ((close_h - base) / base, series.dates[idx])
    return out


def _volatility_shift(
    series: _CloseSeries,
    as_of: _dt.date,
    *,
    window: int = VOL_WINDOW_DAYS,
) -> float | None:
    base_idx = series.index_strictly_before(as_of)
    if base_idx < window:
        return None
    on_or_after = series.index_on_or_after(as_of)
    if on_or_after + window > len(series):
        return None
    pre_closes = series.close[base_idx - window : base_idx + 1]
    pre_rets = _log_returns(pre_closes)
    post_closes = series.close[on_or_after - 1 : on_or_after + window]
    # post_closes spans (t-1, t, t+1, ..., t+window-1) -> window returns
    post_rets = _log_returns(post_closes)
    if len(pre_rets) < 2 or len(post_rets) < 2:
        return None

    def _std(values: list[float]) -> float:
        n = len(values)
        mean = sum(values) / n
        return (sum((v - mean) ** 2 for v in values) / (n - 1)) ** 0.5

    return _std(post_rets) - _std(pre_rets)


# ---------------------------------------------------------------------------
# Concurrent macro release calendar (real BLS / ISM / FRED-derived dates)
# ---------------------------------------------------------------------------


def _resolve_macro_calendar(
    csv_path: Path | str | None,
) -> MacroReleaseCalendar:
    """Pick the real-release calendar when available, fall back to heuristic.

    Resolution order:

    1. ``csv_path`` argument (parquet or CSV). Used by tests.
    2. ``data/external/macro_releases.csv`` -- a curated CSV shipped in the
       repo. Loaded once per build call.
    3. Pure-rule heuristic (legacy behaviour). Used when neither file is
       present, so the builder still runs on a fresh checkout without
       network calls.
    """

    if csv_path is not None:
        return load_macro_release_calendar(Path(csv_path))
    default_path = DEFAULT_MACRO_RELEASES_CSV
    if default_path.exists():
        return load_macro_release_calendar(default_path)
    return build_heuristic_calendar()


def _has_concurrent_macro_release(
    event_date: _dt.date,
    series: _CloseSeries,
    *,
    radius: int = CONCURRENT_MACRO_TRADING_DAY_RADIUS,
    calendar: MacroReleaseCalendar | None = None,
) -> bool:
    """Return True iff a major US macro release sits within ``radius``
    trading days of ``event_date``.

    The ``calendar`` argument carries the real release dates (BLS / ISM /
    FRED-derived). When omitted we fall back to the rule-based heuristic
    (legacy behaviour) so callers that never had a calendar continue to
    work.
    """

    if calendar is None:
        calendar = build_heuristic_calendar()
    macro_dates = calendar.dates_in_year(event_date.year)
    macro_dates |= calendar.dates_in_year(event_date.year - 1)
    macro_dates |= calendar.dates_in_year(event_date.year + 1)
    base_idx = series.index_on_or_after(event_date)
    if base_idx >= len(series):
        return False
    lo_idx = max(0, base_idx - radius)
    hi_idx = min(len(series) - 1, base_idx + radius)
    window = set(series.dates[lo_idx : hi_idx + 1])
    return any(d in macro_dates for d in window)


# ---------------------------------------------------------------------------
# Per-event row construction
# ---------------------------------------------------------------------------


def _document_id(source: str, event_date: str, event_kind: str) -> str:
    h = hashlib.sha256(f"{source}|{event_date}|{event_kind}".encode("utf-8"))
    return h.hexdigest()[:16]


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _prior_window_sha(bars: Sequence[_PriorBar]) -> str:
    parts: list[str] = []
    for b in bars:
        parts.append(
            "|".join(
                [
                    b.date.isoformat(),
                    f"{b.close:.10f}",
                    f"{b.volume:.4f}",
                    f"{b.vol_5d:.10f}",
                    f"{b.cum_return_20d:.10f}",
                ]
            )
        )
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def _bars_to_json(bars: Sequence[_PriorBar]) -> str:
    payload = [
        {
            "date": b.date.isoformat(),
            "close": round(b.close, 10),
            "volume": round(b.volume, 4),
            "vol_5d": round(b.vol_5d, 10),
            "cum_return_20d": round(b.cum_return_20d, 10),
        }
        for b in bars
    ]
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


def _as_of_for_event(event_date: str, event_kind: str) -> str:
    """Apply the placeholder announcement time.

    FOMC kinds (statement, minutes, press_conference, testimony) use the
    2pm ET / 19:00 UTC placeholder; speech kinds use 14:00 UTC. Documented
    in the module docstring.
    """

    if event_kind in _SPEECH_KINDS:
        return f"{event_date}{SPEECH_AS_OF_TIME}"
    return f"{event_date}{FOMC_AS_OF_TIME}"


def _assert_no_lookahead(as_of: _dt.date, bars: Sequence[_PriorBar]) -> None:
    if not bars:
        return
    last = bars[-1].date
    if not (last < as_of):
        raise ValueError(
            f"prior-window contract violated: last bar {last} is not strictly "
            f"before as_of {as_of}"
        )


def _build_event_rows(
    doc: _EventDoc,
    *,
    asset: str,
    benchmark: str,
    asset_series: _CloseSeries,
    bench_series: _CloseSeries,
    horizons: Sequence[int],
    credibility_kwargs: dict[str, Any],
    macro_release_calendar: MacroReleaseCalendar,
) -> list[dict[str, Any]]:
    event_date = _date(doc.event_date)
    as_of_ts = _as_of_for_event(doc.event_date, doc.event_kind)
    as_of_date = event_date  # placeholder time is same-day; window cuts on date

    prior_bars = _build_prior_window(asset_series, as_of_date)
    if prior_bars is None:
        return []
    _assert_no_lookahead(as_of_date, prior_bars)

    targets = _realized_returns(asset_series, as_of_date, horizons)
    if not targets:
        return []

    # t+1d direction
    t1d = targets.get(1)
    if t1d is None:
        direction_t1d = 0
    else:
        ret, _ = t1d
        direction_t1d = (1 if ret > 0 else (-1 if ret < 0 else 0))

    if asset == benchmark:
        alpha, beta = 0.0, 1.0
    else:
        fit = _market_model_for_event(asset_series, bench_series, as_of_date)
        if fit is None:
            return []
        alpha, beta = fit

    # Benchmark realized returns at same horizons for abnormal-return calc
    if asset == benchmark:
        bench_targets: dict[int, tuple[float, _dt.date | None]] = targets
    else:
        bench_targets = _realized_returns(bench_series, as_of_date, horizons) or {}

    vol_shift = _volatility_shift(asset_series, as_of_date)
    concurrent_macro = _has_concurrent_macro_release(
        event_date, asset_series, calendar=macro_release_calendar
    )

    # Credibility vector (degrades to zeros when inputs are absent)
    cred = _safe_credibility(as_of_ts, credibility_kwargs)

    prior_sha = _prior_window_sha(prior_bars)
    prior_json = _bars_to_json(prior_bars)
    text_hash = _text_hash(doc.text)
    document_id = _document_id(doc.source, doc.event_date, doc.event_kind)
    token_count = len(doc.text.split())

    rows: list[dict[str, Any]] = []
    for h in horizons:
        tgt = targets.get(h)
        if tgt is None:
            continue
        realized_return, realized_date = tgt
        if asset == benchmark:
            # No market-model adjustment: the benchmark IS the asset.
            # By contract (alpha=0, beta=1) and abnormal = raw return.
            abnormal = realized_return
        else:
            entry = bench_targets.get(h)
            bench_ret = entry[0] if entry else 0.0
            abnormal = realized_return - (alpha + beta * bench_ret)
        rows.append(
            {
                "event_date": doc.event_date,
                "event_kind": doc.event_kind,
                "document_id": document_id,
                "text_hash": text_hash,
                "source": doc.source,
                "source_record_id": doc.source_record_id,
                "as_of_ts": as_of_ts,
                "text": doc.text,
                "token_count": token_count,
                "axis_stance": doc.multi_axis.get("stance"),
                "axis_time": doc.multi_axis.get("time"),
                "axis_certainty": doc.multi_axis.get("certainty"),
                "axis_factor": doc.multi_axis.get("factor"),
                "axis_topic": doc.multi_axis.get("topic"),
                "credibility_drift_score": float(cred.drift_score),
                "credibility_realized_vs_stated_gap": float(cred.realized_vs_stated_gap),
                "credibility_market_implied_gap": float(cred.market_implied_gap),
                "credibility_months_since_reversal": int(cred.months_since_reversal),
                "prior_window_sha256": prior_sha,
                "prior_bars_json": prior_json,
                "asset_symbol": asset,
                "horizon": int(h),
                "realized_return": float(realized_return),
                "abnormal_return": float(abnormal),
                "alpha": float(alpha),
                "beta": float(beta),
                "direction_t1d": int(direction_t1d),
                "volatility_shift": float(vol_shift) if vol_shift is not None else None,
                "concurrent_macro_release": bool(concurrent_macro),
                "realized_date": realized_date.isoformat() if realized_date else None,
            }
        )
    return rows


def _safe_credibility(as_of_ts: str, kwargs: dict[str, Any]) -> CredibilityVector:
    """Wrap ``load_credibility_for_run`` so missing inputs degrade to zeros.

    The loader already handles "no embeddings / no FRED cache" gracefully,
    but we still catch ValueError so a malformed ``as_of_ts`` placeholder
    never aborts the whole build. Documented behaviour: zero vector means
    "credibility unknown", not "credibility zero".
    """

    try:
        return load_credibility_for_run(as_of_ts=as_of_ts, **kwargs)
    except (ValueError, FileNotFoundError):
        return CredibilityVector(0.0, 0.0, 0.0, 0)


# ---------------------------------------------------------------------------
# Top-level builder
# ---------------------------------------------------------------------------


COLUMN_ORDER = (
    "event_date",
    "event_kind",
    "document_id",
    "text_hash",
    "source",
    "source_record_id",
    "as_of_ts",
    "text",
    "token_count",
    "axis_stance",
    "axis_time",
    "axis_certainty",
    "axis_factor",
    "axis_topic",
    "credibility_drift_score",
    "credibility_realized_vs_stated_gap",
    "credibility_market_implied_gap",
    "credibility_months_since_reversal",
    "prior_window_sha256",
    "prior_bars_json",
    "asset_symbol",
    "horizon",
    "realized_return",
    "abnormal_return",
    "alpha",
    "beta",
    "direction_t1d",
    "volatility_shift",
    "concurrent_macro_release",
    "realized_date",
)


def build_event_rows(
    *,
    package_dir: Path,
    asset: str = DEFAULT_ASSET,
    benchmark: str = DEFAULT_BENCHMARK,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    asset_series: _CloseSeries | None = None,
    bench_series: _CloseSeries | None = None,
    market_cache_dir: Path | None = None,
    embedding_path: Path | str | None = None,
    fred_cache_dir: Path | None = None,
    fred_series_id: str = "DFF",
    stance_by_date: Sequence[tuple[str, float]] = (),
    summary: _BuildSummary | None = None,
    keep_all_sources: bool = False,
    macro_release_calendar: MacroReleaseCalendar | None = None,
    macro_release_csv_path: Path | str | None = None,
) -> pd.DataFrame:
    """Build the events DataFrame for one training package.

    Tests inject ``asset_series`` and ``bench_series`` directly. The smoke
    run leaves them ``None`` and we fetch (and cache) live yfinance bars.

    When ``keep_all_sources=False`` (default) the frame collapses to one
    row per ``(event_date, event_kind, asset_symbol, horizon)`` via the
    ``_SOURCE_PREFERENCE`` order. When ``keep_all_sources=True`` every
    source survives, producing one row per
    ``(event_date, event_kind, source, asset_symbol, horizon)``.

    The ``concurrent_macro_release`` flag consults ``macro_release_calendar``
    (a real BLS/ISM/FRED-derived calendar) when one is provided; otherwise it
    falls back to the deterministic heuristic. Pass ``macro_release_csv_path``
    to load a pre-built calendar parquet/CSV from disk.
    """

    if summary is None:
        summary = _BuildSummary()

    if macro_release_calendar is None:
        macro_release_calendar = _resolve_macro_calendar(macro_release_csv_path)

    registry_rows = _load_registry_rows(package_dir)
    docs_all = _aggregate_events(registry_rows)
    docs = list(docs_all) if keep_all_sources else _choose_preferred(docs_all)
    if not docs:
        return _empty_frame()

    # Decide the window we need to fetch.
    event_dates = sorted({_date(d.event_date) for d in docs})
    earliest = event_dates[0] - _dt.timedelta(days=MARKET_MODEL_WINDOW_DAYS * 2 + 60)
    latest = event_dates[-1] + _dt.timedelta(days=max(horizons) * 2 + 60)

    if asset_series is None:
        asset_series = _fetch_close_series(
            asset, start=earliest, end=latest, cache_dir=market_cache_dir
        )
    if bench_series is None:
        if benchmark == asset:
            bench_series = asset_series
        else:
            bench_series = _fetch_close_series(
                benchmark, start=earliest, end=latest, cache_dir=market_cache_dir
            )

    credibility_kwargs = {
        "embedding_path": embedding_path,
        "stance_by_date": tuple(stance_by_date),
        "fred_cache_dir": fred_cache_dir,
        "fred_series_id": fred_series_id,
    }

    out_rows: list[dict[str, Any]] = []
    for doc in docs:
        rows = _build_event_rows(
            doc,
            asset=asset,
            benchmark=benchmark,
            asset_series=asset_series,
            bench_series=bench_series,
            horizons=horizons,
            credibility_kwargs=credibility_kwargs,
            macro_release_calendar=macro_release_calendar,
        )
        if not rows:
            summary.dropped_no_prior_window += 1
            continue
        summary.events_emitted += 1
        for r in rows:
            summary.per_source_rows[r["source"]] = summary.per_source_rows.get(r["source"], 0) + 1
            summary.per_kind_rows[r["event_kind"]] = summary.per_kind_rows.get(r["event_kind"], 0) + 1
            if r["concurrent_macro_release"]:
                summary.concurrent_macro_release_rows += 1
        out_rows.extend(rows)

    summary.rows_written = len(out_rows)
    if not out_rows:
        return _empty_frame()

    df = pd.DataFrame(out_rows)
    # Deterministic ordering. The full view sorts on source + source_record_id
    # so the parquet bytes don't shift when sources interleave.
    sort_cols = ["event_date", "event_kind", "source", "source_record_id", "asset_symbol", "horizon"]
    df = df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    df = df[list(COLUMN_ORDER)]
    return df


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in COLUMN_ORDER})


def write_events_parquet(df: pd.DataFrame, output_path: Path) -> None:
    """Write the events frame deterministically.

    Uses pandas + pyarrow with snappy compression (no creation-time
    metadata) and stable column order. Same input dataframe -> same bytes.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, engine="pyarrow", index=False, compression="snappy")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the event-row parquet for Phase 8 event-study forecasting."
    )
    parser.add_argument("--training-package-id", required=True)
    parser.add_argument("--asset", default=DEFAULT_ASSET)
    parser.add_argument(
        "--benchmark",
        default=DEFAULT_BENCHMARK,
        help="Market-model benchmark. Defaults to ^GSPC (forces beta=1, alpha=0 when asset==benchmark).",
    )
    parser.add_argument(
        "--horizons",
        default=",".join(str(h) for h in DEFAULT_HORIZONS),
        help="Comma-separated trading-day horizons (default: 1,5,10,30).",
    )
    parser.add_argument(
        "--output",
        default="events.parquet",
        help=(
            "Name of the collapsed parquet (one row per event_date x "
            "event_kind x asset_symbol x horizon). The full-source view is "
            "always written alongside as 'events_full.parquet'."
        ),
    )
    parser.add_argument(
        "--full-output",
        default="events_full.parquet",
        help=(
            "Name of the full parquet (keeps every source/source_record_id). "
            "Set to '' to skip the full view."
        ),
    )
    parser.add_argument(
        "--market-cache-dir",
        default=None,
        help="Where to cache the yfinance pull (default: <package_dir>/_market_cache).",
    )
    parser.add_argument(
        "--embedding-path",
        default=None,
        help="Optional per-encoder embedding parquet for credibility drift.",
    )
    parser.add_argument(
        "--fred-cache-dir",
        default=str(DEFAULT_DATA_DIR / "external" / "fred"),
        help="FRED cache directory for credibility realized-vs-stated gap.",
    )
    parser.add_argument(
        "--macro-release-csv",
        default=None,
        help=(
            "Override the macro-release calendar source (CSV or parquet). "
            "Defaults to data/external/macro_releases.csv when present, "
            "else the legacy rule-based heuristic."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    package_dir = DEFAULT_DATA_DIR / "processed" / args.training_package_id
    if not package_dir.exists():
        raise SystemExit(f"Training package not found: {package_dir}")

    horizons = tuple(int(h) for h in args.horizons.split(",") if h.strip())
    market_cache_dir = (
        Path(args.market_cache_dir)
        if args.market_cache_dir
        else (package_dir / "_market_cache")
    )

    macro_csv = Path(args.macro_release_csv) if args.macro_release_csv else None
    macro_calendar = _resolve_macro_calendar(macro_csv)

    # Collapsed view
    summary = _BuildSummary()
    df = build_event_rows(
        package_dir=package_dir,
        asset=args.asset,
        benchmark=args.benchmark,
        horizons=horizons,
        market_cache_dir=market_cache_dir,
        embedding_path=Path(args.embedding_path) if args.embedding_path else None,
        fred_cache_dir=Path(args.fred_cache_dir) if args.fred_cache_dir else None,
        summary=summary,
        macro_release_calendar=macro_calendar,
    )
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = package_dir / output_path
    write_events_parquet(df, output_path)

    print(f"[event-rows] collapsed view: {summary.rows_written} rows -> {output_path}")
    print(f"[event-rows] unique events: {summary.events_emitted}")
    print(f"[event-rows] dropped (no prior window or no targets): {summary.dropped_no_prior_window}")
    print(
        f"[event-rows] concurrent_macro_release rows: {summary.concurrent_macro_release_rows} "
        f"({_pct(summary.concurrent_macro_release_rows, summary.rows_written)})"
    )
    print(f"[event-rows] macro calendar source: {macro_calendar.source_label}")
    print("[event-rows] per-source breakdown (collapsed):")
    for src, count in sorted(summary.per_source_rows.items(), key=lambda x: -x[1]):
        print(f"  {src}: {count}")
    print("[event-rows] per-kind breakdown (collapsed):")
    for kind, count in sorted(summary.per_kind_rows.items(), key=lambda x: -x[1]):
        print(f"  {kind}: {count}")

    # Full view -- emit only when --full-output is set to a non-empty value.
    full_output_arg = (args.full_output or "").strip()
    if full_output_arg:
        full_summary = _BuildSummary()
        df_full = build_event_rows(
            package_dir=package_dir,
            asset=args.asset,
            benchmark=args.benchmark,
            horizons=horizons,
            market_cache_dir=market_cache_dir,
            embedding_path=Path(args.embedding_path) if args.embedding_path else None,
            fred_cache_dir=Path(args.fred_cache_dir) if args.fred_cache_dir else None,
            summary=full_summary,
            macro_release_calendar=macro_calendar,
            keep_all_sources=True,
        )
        full_output_path = Path(full_output_arg)
        if not full_output_path.is_absolute():
            full_output_path = package_dir / full_output_path
        write_events_parquet(df_full, full_output_path)
        print(
            f"[event-rows] full view: {full_summary.rows_written} rows -> {full_output_path}"
        )
        print(
            f"[event-rows] full unique (date x kind x source) docs: "
            f"{full_summary.events_emitted}"
        )
        print("[event-rows] per-source breakdown (full):")
        for src, count in sorted(full_summary.per_source_rows.items(), key=lambda x: -x[1]):
            print(f"  {src}: {count}")

    # Column / dtype summary so the smoke run is self-describing
    print("[event-rows] column dtypes (collapsed):")
    for col, dtype in df.dtypes.items():
        print(f"  {col}: {dtype}")
    return 0


def _pct(num: int, denom: int) -> str:
    if denom <= 0:
        return "0.00%"
    return f"{(num / denom) * 100:.2f}%"


if __name__ == "__main__":
    sys.exit(main())
