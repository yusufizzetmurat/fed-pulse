"""Assemble :class:`CredibilityVector` inputs from on-disk caches.

The four credibility axes are computed from heterogeneous sources:

- ``drift_score``           — cosine distance between the as-of embedding and
  the mean of the four prior FOMC document embeddings; the embeddings come
  from the per-encoder cache built by :mod:`app.data.embedding_cache`.
- ``realized_vs_stated_gap``— Pearson correlation between a per-date stance
  score and the realized effective fed funds change pulled from FRED DFF.
- ``market_implied_gap``    — SEP committee long-run median minus the
  market-implied long-run proxy (5-year Treasury yield, DGS5), scaled by
  1/4 and clipped to ``[-1, 1]``. SEP comes from
  ``data/external/fred/sep_projections.parquet``; the DGS5 proxy stands in
  for a clean OIS forward where FRED does not publish one. Both lookups are
  strict ``< as_of`` so the same-day FOMC release never leaks into its own
  credibility feature.
- ``months_since_reversal`` — counts whole months since the last sign flip in
  the stance history.

The loader is intentionally additive: every axis degrades to 0.0 when its
input is unavailable, so it is safe to call on a training package that has
no vtasca rows or no FRED cache.
"""

from __future__ import annotations

import datetime
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import pandas as pd

from app.features.credibility import (
    CredibilityVector,
    compute_credibility_vector,
)
from app.services.fred_client import (
    FredObservation,
    FredSeriesResponse,
    fetch_fred_series,
)

DEFAULT_FRED_SERIES = "DFF"
DEFAULT_SEP_FILENAME = "sep_projections.parquet"
# DGS5 (5-year Treasury yield) stands in as the market-implied long-run
# fed-funds proxy. FRED does not publish a clean long-run OIS forward, and
# the 5-year nominal yield is the standard published series used by Fed
# researchers to anchor "market expects long-run policy to settle near X".
DEFAULT_OIS_FILENAME = "DGS5.json"


@dataclass(frozen=True)
class CredibilityInputs:
    """All inputs to ``compute_credibility_vector`` for a single as-of date."""

    current_embedding: list[float]
    prior_embeddings: list[list[float]]
    stance_history: list[float]
    stated_path: list[float]
    realized_path: list[float]
    sep_terminal: float | None = None
    ois_terminal: float | None = None


def _parse_iso(value: str) -> datetime.date | None:
    try:
        return datetime.date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


@lru_cache(maxsize=4)
def _load_embedding_frame(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "event_date" in df.columns:
        df["event_date"] = df["event_date"].astype(str)
    return df


def _select_document_embedding(
    embedding_df: pd.DataFrame,
    *,
    as_of: datetime.date,
    on_or_before: bool = True,
) -> list[float] | None:
    if embedding_df.empty:
        return None
    parsed = embedding_df["event_date"].map(_parse_iso)
    mask = parsed.notna()
    if on_or_before:
        mask &= parsed.map(lambda d: bool(d) and d <= as_of)
    if not mask.any():
        return None
    candidates = embedding_df.loc[mask].copy()
    candidates["_parsed_date"] = parsed[mask]
    candidates = candidates.sort_values(["_parsed_date"]).tail(1)
    embedding = candidates.iloc[-1]["embedding"]
    if embedding is None:
        return None
    return [float(x) for x in embedding]


def _select_prior_embeddings(
    embedding_df: pd.DataFrame,
    *,
    as_of: datetime.date,
    window: int = 4,
) -> list[list[float]]:
    if embedding_df.empty:
        return []
    parsed = embedding_df["event_date"].map(_parse_iso)
    mask = parsed.notna() & parsed.map(lambda d: bool(d) and d < as_of)
    if not mask.any():
        return []
    candidates = embedding_df.loc[mask].copy()
    candidates["_parsed_date"] = parsed[mask]
    candidates = candidates.sort_values(["_parsed_date"]).tail(window)
    out: list[list[float]] = []
    for embedding in candidates["embedding"].tolist():
        if embedding is None:
            continue
        out.append([float(x) for x in embedding])
    return out


def _stance_history_series(
    stance_by_date: Sequence[tuple[str, float]],
    *,
    as_of: datetime.date,
) -> list[float]:
    """Return a chronologically-ordered stance series up to ``as_of``."""

    series: list[tuple[datetime.date, float]] = []
    for date_str, value in stance_by_date:
        parsed = _parse_iso(date_str)
        if parsed is None or parsed > as_of:
            continue
        series.append((parsed, float(value)))
    series.sort(key=lambda item: item[0])
    return [v for _, v in series]


def _realized_series_from_fred(
    response: FredSeriesResponse | None,
    *,
    as_of: datetime.date,
    window_days: int = 90,
) -> list[float]:
    """Return realized FRED daily values strictly before ``as_of``.

    The strict upper bound matches the rest of the credibility loader: an
    FOMC release dated 2026-03-19 reads at most the 2026-03-18 close.
    The daily FRED rate is the calendar-day close, so including the
    same-day observation would feed the model the market's reaction to the
    very announcement being scored — a same-day leak the
    :mod:`app.services.credibility` features must not have.
    """

    if response is None:
        return []
    lower = as_of - datetime.timedelta(days=window_days)
    out: list[float] = []
    for obs in response.observations:
        parsed = _parse_iso(obs.date)
        if parsed is None or parsed < lower or parsed >= as_of:
            continue
        if obs.value is None:
            continue
        out.append(float(obs.value))
    return out


@lru_cache(maxsize=4)
def _sep_table(sep_path: str) -> pd.DataFrame:
    return pd.read_parquet(sep_path)


def _sep_terminal_at(as_of: datetime.date, *, sep_path: Path) -> float | None:
    """Return the SEP committee long-run median in force on ``as_of``.

    Reads ``sep_projections.parquet`` and returns the most recent
    ``ffr_median_longer_run`` whose ``meeting_date`` is strictly less than
    ``as_of``. Strict inequality keeps an SEP release issued at the same
    FOMC meeting being scored from leaking into its own credibility
    feature. Returns ``None`` when no eligible SEP row exists or the file
    is missing.
    """

    try:
        table = _sep_table(str(sep_path))
    except (FileNotFoundError, OSError):
        return None
    median = table["ffr_median_longer_run"].astype(float)
    parsed = table["meeting_date"].map(_parse_iso)
    mask = parsed.notna() & parsed.map(lambda d: bool(d) and d < as_of) & median.notna()
    if not mask.any():
        return None
    idx = parsed[mask].astype("string").sort_values().index[-1]
    # ``idx`` is the row label returned by ``.index[-1]``, so use
    # label-based ``loc`` rather than positional ``iloc``. Equivalent on
    # the default RangeIndex but defensive against any future reindex
    # that yields a non-positional label.
    value = median.loc[idx]
    if value is None or pd.isna(value):
        return None
    return float(value)


@lru_cache(maxsize=4)
def _fred_observations_from_json(path: str) -> list[tuple[datetime.date, float]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    out: list[tuple[datetime.date, float]] = []
    for obs in raw.get("observations", []):
        parsed = _parse_iso(obs.get("date") or "")
        if parsed is None:
            continue
        value = obs.get("value")
        if value is None:
            continue
        try:
            v = float(value)
        except (TypeError, ValueError):
            # FRED encodes missing as the literal "." string.
            continue
        out.append((parsed, v))
    out.sort(key=lambda item: item[0])
    return out


def _ois_terminal_at(as_of: datetime.date, *, ois_path: Path) -> float | None:
    """Return the most recent market-implied long-run rate proxy ``< as_of``.

    The proxy series is DGS5 (5-year Treasury yield, percent). Strict
    inequality matches ``_sep_terminal_at`` so an FOMC release timestamped on
    the same day cannot ingest its own post-announcement market reaction.
    Returns ``None`` when no eligible observation exists.
    """

    try:
        series = _fred_observations_from_json(str(ois_path))
    except (FileNotFoundError, OSError):
        return None
    last: float | None = None
    for parsed, value in series:
        if parsed >= as_of:
            break
        last = value
    return last


def assemble_inputs(
    *,
    as_of_ts: str,
    embedding_path: Path | str | None,
    stance_by_date: Sequence[tuple[str, float]] = (),
    fred_response: FredSeriesResponse | None = None,
    drift_window: int = 4,
    realized_window_days: int = 90,
    sep_terminal: float | None = None,
    ois_terminal: float | None = None,
) -> CredibilityInputs:
    """Pack disparate sources into a :class:`CredibilityInputs` for one date."""

    as_of = _parse_iso(as_of_ts)
    if as_of is None:
        raise ValueError(f"as_of_ts is not ISO-parseable: {as_of_ts!r}")

    if embedding_path is not None:
        path = Path(embedding_path)
        if path.exists():
            embedding_df = _load_embedding_frame(str(path))
        else:
            embedding_df = pd.DataFrame(columns=["event_date", "embedding"])
    else:
        embedding_df = pd.DataFrame(columns=["event_date", "embedding"])

    current = _select_document_embedding(embedding_df, as_of=as_of) or []
    prior = _select_prior_embeddings(embedding_df, as_of=as_of, window=drift_window)
    stance = _stance_history_series(stance_by_date, as_of=as_of)
    realized = _realized_series_from_fred(
        fred_response, as_of=as_of, window_days=realized_window_days
    )
    # The "stated" path mirrors the stance series clipped to the same window so
    # the correlation is computed over a matched horizon.
    stated_path = stance[-len(realized) :] if realized else []
    return CredibilityInputs(
        current_embedding=current,
        prior_embeddings=prior,
        stance_history=stance,
        stated_path=stated_path,
        realized_path=realized,
        sep_terminal=sep_terminal,
        ois_terminal=ois_terminal,
    )


def load_credibility_for_run(
    *,
    as_of_ts: str,
    embedding_path: Path | str | None,
    stance_by_date: Sequence[tuple[str, float]] = (),
    fred_response: FredSeriesResponse | None = None,
    fred_series_id: str = DEFAULT_FRED_SERIES,
    fred_cache_dir: Path | None = None,
    drift_window: int = 4,
    realized_window_days: int = 90,
    sep_projections_path: Path | str | None = None,
    ois_series_path: Path | str | None = None,
) -> CredibilityVector:
    """Single entry point used by training and the ``/analyze`` path.

    When ``fred_response`` is None and ``fred_cache_dir`` is provided, the
    loader tries to load a cached FRED series; if missing it falls back to
    an empty realized path (the gap returns 0.0).

    ``sep_projections_path`` and ``ois_series_path`` default to the standard
    locations under ``fred_cache_dir`` (``sep_projections.parquet`` and
    ``DGS5.json``). When either file is missing the corresponding terminal
    rate stays ``None`` and the market-implied gap degrades to 0.0 the same
    way the other axes do.
    """

    if fred_response is None and fred_cache_dir is not None:
        try:
            fred_response = fetch_fred_series(fred_series_id, cache_dir=Path(fred_cache_dir))
        except (FileNotFoundError, RuntimeError):
            fred_response = None

    as_of = _parse_iso(as_of_ts)
    sep_terminal: float | None = None
    ois_terminal: float | None = None
    if as_of is not None and fred_cache_dir is not None:
        sep_path = (
            Path(sep_projections_path)
            if sep_projections_path
            else Path(fred_cache_dir) / DEFAULT_SEP_FILENAME
        )
        if sep_path.exists():
            sep_terminal = _sep_terminal_at(as_of, sep_path=sep_path)
        ois_path = (
            Path(ois_series_path)
            if ois_series_path
            else Path(fred_cache_dir) / DEFAULT_OIS_FILENAME
        )
        if ois_path.exists():
            ois_terminal = _ois_terminal_at(as_of, ois_path=ois_path)

    inputs = assemble_inputs(
        as_of_ts=as_of_ts,
        embedding_path=embedding_path,
        stance_by_date=stance_by_date,
        fred_response=fred_response,
        drift_window=drift_window,
        realized_window_days=realized_window_days,
        sep_terminal=sep_terminal,
        ois_terminal=ois_terminal,
    )
    return compute_credibility_vector(
        current_embedding=inputs.current_embedding,
        prior_embeddings=inputs.prior_embeddings,
        stance_history=inputs.stance_history,
        stated_path=inputs.stated_path,
        realized_path=inputs.realized_path,
        sep_terminal=inputs.sep_terminal,
        ois_terminal=inputs.ois_terminal,
    )


__all__ = [
    "CredibilityInputs",
    "FredObservation",
    "assemble_inputs",
    "load_credibility_for_run",
]
