"""Assemble :class:`CredibilityVector` inputs from on-disk caches.

The four credibility axes are computed from heterogeneous sources:

- ``drift_score``           — cosine distance between the as-of embedding and
  the mean of the four prior FOMC document embeddings; the embeddings come
  from the per-encoder cache built by :mod:`app.data.embedding_cache`.
- ``realized_vs_stated_gap``— Pearson correlation between a per-date stance
  score and the realized effective fed funds change pulled from FRED DFF.
- ``market_implied_gap``    — currently a placeholder (returns 0.0). Will be
  filled in once the SEP / Eurodollar curve scraper lands.
- ``months_since_reversal`` — counts whole months since the last sign flip in
  the stance history.

The loader is intentionally additive: every axis degrades to 0.0 when its
input is unavailable, so it is safe to call on a training package that has
no vtasca rows or no FRED cache.
"""

from __future__ import annotations

import datetime
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


@dataclass(frozen=True)
class CredibilityInputs:
    """All inputs to ``compute_credibility_vector`` for a single as-of date."""

    current_embedding: list[float]
    prior_embeddings: list[list[float]]
    stance_history: list[float]
    stated_path: list[float]
    realized_path: list[float]


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
    if response is None:
        return []
    lower = as_of - datetime.timedelta(days=window_days)
    out: list[float] = []
    for obs in response.observations:
        parsed = _parse_iso(obs.date)
        if parsed is None or parsed < lower or parsed > as_of:
            continue
        if obs.value is None:
            continue
        out.append(float(obs.value))
    return out


def assemble_inputs(
    *,
    as_of_ts: str,
    embedding_path: Path | str | None,
    stance_by_date: Sequence[tuple[str, float]] = (),
    fred_response: FredSeriesResponse | None = None,
    drift_window: int = 4,
    realized_window_days: int = 90,
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
) -> CredibilityVector:
    """Single entry point used by training and the ``/analyze`` path.

    When ``fred_response`` is None and ``fred_cache_dir`` is provided, the
    loader tries to load a cached FRED series; if missing it falls back to
    an empty realized path (the gap returns 0.0).
    """

    if fred_response is None and fred_cache_dir is not None:
        try:
            fred_response = fetch_fred_series(
                fred_series_id, cache_dir=Path(fred_cache_dir)
            )
        except (FileNotFoundError, RuntimeError):
            fred_response = None

    inputs = assemble_inputs(
        as_of_ts=as_of_ts,
        embedding_path=embedding_path,
        stance_by_date=stance_by_date,
        fred_response=fred_response,
        drift_window=drift_window,
        realized_window_days=realized_window_days,
    )
    return compute_credibility_vector(
        current_embedding=inputs.current_embedding,
        prior_embeddings=inputs.prior_embeddings,
        stance_history=inputs.stance_history,
        stated_path=inputs.stated_path,
        realized_path=inputs.realized_path,
        sep_terminal=None,
        ois_terminal=None,
    )


__all__ = [
    "CredibilityInputs",
    "FredObservation",
    "assemble_inputs",
    "load_credibility_for_run",
]
