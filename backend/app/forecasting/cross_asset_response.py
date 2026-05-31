"""Predict the cross-section of asset abnormal returns after an FOMC event (closes #148).

Companion to :mod:`app.forecasting.next_fomc_decision`. Where the
next-FOMC head asks "what will the Fed do next?", this module asks
"how does the basket move when the Fed speaks?". Each FOMC event
fans out into a vector of abnormal-return responses across an asset
universe (equities, rates, dollar, gold, crude, sector ETFs) at
multiple trading-day horizons; the joint pattern of those responses
is what disambiguates a tightening shock ("DGS2 up, SPX down, DXY
up") from a dovish reset ("DGS2 down, SPX up, gold up") from a
risk-off panic ("everything down, VIX up").

Target
------

For each row in :file:`events.parquet` we already carry
``abnormal_return`` -- the market-model residual at horizon ``h``
relative to the trailing 252-day window. The cross-asset response
head reshapes that column into one supervised row per
``(event_date, asset, horizon)``: features known at meeting ``N``
predict the post-event abnormal return ``h`` trading days later.

Asset universe
--------------

Taken from the union of ``asset_symbol`` values present in
``events.parquet``. The schema already supports per-asset rebuilds.
For 2024-era packages this typically covers some subset of::

    ^GSPC ^IXIC ^DJI ^TNX DX-Y.NYB GC=F CL=F XLF XLK XLE

Whatever subset is actually in the parquet is what we model; missing
assets fall out of the metrics table rather than being faked. The
``asset_universe`` field on :class:`RunArtifacts` records the
actually-modelled symbols.

Horizons
--------

``1, 5, 10, 30`` trading days (already produced by the event-row
dataset builder). Each ``(asset, horizon)`` pair is one cell.

Models
------

Per-cell models (one regression per ``(asset, horizon)``):

- **ridge**     -- :class:`sklearn.linear_model.Ridge` (alpha=1.0).
  The headline. Linear in the joint feature matrix; deterministic;
  L2-regularised so the cell-level fit does not blow up on the ~150
  meetings we have.
- **hist_gbt**  -- :class:`sklearn.ensemble.HistGradientBoostingRegressor`
  seeded with ``random_state=11``. Non-linear comparator.

Optional pooled-panel exploration:

- **pooled_ridge** -- A single :class:`Ridge` on the stacked frame
  with per-asset and per-horizon dummies appended to the feature
  matrix. Marked exploratory because the panel structure assumes
  asset/horizon effects are additive over a shared coefficient
  vector, which they are not in practice; we report it so the
  reader sees how a pooled fit compares against per-cell fits with
  the same hyperparameter.

Baselines
---------

- **zero**     -- predict ``0`` abnormal return. The natural null
  for a residual that is supposed to be mean-zero by construction.
  Any per-cell model that fails to beat zero on RMSE is providing
  no lift.
- **ois_bp**   -- OIS-implied basis-point change at the 1-month
  tenor (``post_event_curve_1m - ff_target_after``) * 100. Same
  information cutoff as the model (PR #156's fix): both the model's
  features and the baseline read off meeting ``N``'s post-event
  curve, so the comparison is fair. The bp signal is converted to
  fractional return space (``bp / 10000``) before scoring so the
  units match ``abnormal_return`` (1 bp = 0.0001, a 1% move = 100
  bp). The OIS path is rate-direction, abnormal returns are price
  responses, so RMSE comparability depends on the asset --
  rate-sensitive cells (^TNX, DGS2) scale with bp moves;
  equity cells do not.

Caveat on baseline comparability: zero-prediction is a strict
RMSE-comparable null. ``ois_bp`` is a signed *direction* baseline
that may dominate on rate-sensitive cells (^TNX) but lose to zero on
sector-equity cells (XLE). We report both rather than picking one.

Walk-forward CV
---------------

Leave-one-meeting-out per cell, mirroring
``next_fomc_decision.walk_forward_predict``: for held-out meeting
``M`` the train set is every supervised row in that cell whose
``feature_event_date < M``. The fitter asserts the strict inequality
on every fold. Train folds with fewer rows than the feature
dimension fall back to baselines only.

Feature families
----------------

Same five families as :mod:`next_fomc_decision`:

- ``ois``        -- 10-dim OIS curve + level/path/info factors
- ``text``       -- 15-dim multi-axis one-hot
- ``linguistic`` -- 14-dim structured features (from #149)
- ``credibility``-- 4-dim credibility vector
- ``macro``      -- 6-dim FRED macro-state snapshot at feature time

Ablation subsets reported in ``feature_attribution.md``::

    ois_only / ois_text / ois_text_linguistic / ois_text_credibility
    ois_text_macro / full

Plus the two model-free rows (``zero_baseline``, ``ois_bp``).

Pandemic-era ablation
---------------------

The window ``2020-04-01..2021-06-30`` is excluded in the
ex-pandemic metrics block, same constant convention as
``next_fomc_decision``. The break is large enough that the
regime-conditioned numbers are worth seeing side-by-side.

Determinism
-----------

* All sklearn estimators seeded with ``random_state=11``.
* Rows are sorted by ``(event_date, asset_symbol, horizon)`` before
  any consumption.
* JSON dumps use ``sort_keys=True, indent=2``.
* Predictions are rounded to 8 decimal places before serialisation.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import math
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.config import DATA_DIR
from app.forecasting.next_fomc_decision import (
    PANDEMIC_END,
    PANDEMIC_START,
    _credibility_feature_names,
    _curve_value_at,
    _extract_credibility_features,
    _extract_linguistic_features,
    _extract_macro_features,
    _extract_ois_features,
    _extract_text_features,
    _linguistic_feature_names,
    _macro_feature_names,
    _ois_feature_names,
    _text_feature_names,
)

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FEATURE_FAMILIES: tuple[str, ...] = ("ois", "text", "linguistic", "credibility", "macro")

# Ablation subsets reported in feature_attribution.md. Order matters:
# each entry produces one row in the markdown table.
ABLATION_SUBSETS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("ois_only", ("ois",)),
    ("ois_text", ("ois", "text")),
    ("ois_text_linguistic", ("ois", "text", "linguistic")),
    ("ois_text_credibility", ("ois", "text", "credibility")),
    ("ois_text_macro", ("ois", "text", "macro")),
    ("full", FEATURE_FAMILIES),
)

# Asset symbols the issue (#148) lists as the canonical cross-section.
# The actual modelled universe is the intersection of this with whatever
# ``asset_symbol`` values are present in ``events.parquet``.
CANONICAL_ASSETS: tuple[str, ...] = (
    "^GSPC",
    "^IXIC",
    "^DJI",
    "^TNX",
    "DX-Y.NYB",
    "GC=F",
    "CL=F",
    "XLF",
    "XLK",
    "XLE",
)

# Canonical horizons (trading days) the event-row builder emits.
CANONICAL_HORIZONS: tuple[int, ...] = (1, 5, 10, 30)

# Headline cells highlighted in feature_attribution.md.
HEADLINE_CELLS: tuple[tuple[str, int], ...] = (("^GSPC", 1), ("^GSPC", 5))


# ---------------------------------------------------------------------------
# Supervised-row data class
# ---------------------------------------------------------------------------


@dataclass
class CrossAssetRow:
    """One supervised row.

    The row is keyed by ``(feature_event_date, asset_symbol, horizon)``.
    Features are extracted at meeting ``N`` (feature_event_date) and
    predict the abnormal return at horizon ``h`` trading days later.
    """

    feature_event_date: _dt.date
    asset_symbol: str
    horizon: int
    abnormal_return: float
    # Per-family feature vectors.
    ois: list[float]
    text: list[float]
    linguistic: list[float]
    credibility: list[float]
    macro: list[float]
    # OIS-implied basis-point signal for the bp baseline. Same
    # information cutoff as the model: read from meeting N's
    # post-event curve at 1m tenor minus meeting N's ff_target_after.
    ois_baseline_bp: float | None
    feature_names: dict[str, tuple[str, ...]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Dataset assembly
# ---------------------------------------------------------------------------


def _collapse_events_to_meeting_axis(events: pd.DataFrame) -> pd.DataFrame:
    """Pick one event_kind row per ``(event_date, asset_symbol, horizon)``.

    Multi-source duplicates are already collapsed upstream in
    ``events.parquet``; what remains is potentially multiple
    ``event_kind`` rows per meeting (statement + minutes + press
    conference). Same preference as :mod:`next_fomc_decision`:
    statement > press_conference > minutes > first available.
    """

    if events.empty:
        return events
    kind_pref = {"statement": 0, "press_conference": 1, "minutes": 2}
    e = events.copy()
    e["_kind_rank"] = e["event_kind"].map(lambda k: kind_pref.get(str(k), 99))
    e = e.sort_values(
        ["event_date", "asset_symbol", "horizon", "_kind_rank", "event_kind"],
        kind="mergesort",
    )
    e = e.drop_duplicates(
        subset=["event_date", "asset_symbol", "horizon"], keep="first"
    )
    return e.drop(columns=["_kind_rank"]).reset_index(drop=True)


def _safe_abnormal_return(value: Any) -> float | None:
    """Pull a float abnormal return; ``None`` if missing/NaN."""

    if value is None:
        return None
    try:
        fv = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(fv):
        return None
    return fv


def _ois_baseline_bp_from_mp_row(mp_row: pd.Series | None) -> float | None:
    """Compute the OIS-implied basis-point signal at meeting ``N``.

    Same information cutoff as the model: reads meeting ``N``'s
    post-event curve at the 1-month tenor and subtracts the
    ``ff_target_after`` set at meeting ``N``. The 1-month tenor is the
    closest free OIS-proxy tenor to the ~6-week inter-meeting window.
    Returns ``None`` when the curve or base rate is missing.
    """

    if mp_row is None:
        return None
    implied_rate = _curve_value_at(mp_row.get("post_event_curve"), 1)
    base = mp_row.get("ff_target_after")
    if isinstance(base, float) and math.isnan(base):
        base = None
    if implied_rate is None or base is None:
        return None
    return (float(implied_rate) - float(base)) * 100.0


def build_supervised_rows(
    *,
    events: pd.DataFrame,
    mp_surprises: pd.DataFrame,
    linguistic_features: pd.DataFrame,
    macro_state: pd.DataFrame,
    asset_universe: Sequence[str] | None = None,
    horizons: Sequence[int] | None = None,
) -> tuple[list[CrossAssetRow], dict[str, Any]]:
    """Join the four parquets into per-(meeting, asset, horizon) supervised rows.

    Returns ``(rows, summary)``. ``summary`` records dropped-row
    counts, the meeting span, and the realised asset/horizon
    universe. Rows are sorted by
    ``(feature_event_date, asset_symbol, horizon)``.
    """

    summary: dict[str, Any] = {
        "rows_in_events": int(len(events)),
        "rows_in_mp_surprises": int(len(mp_surprises)),
        "rows_in_linguistic": int(len(linguistic_features)),
        "rows_in_macro_state": int(len(macro_state)),
        "dropped_missing_target": 0,
        "dropped_unknown_asset": 0,
        "dropped_unknown_horizon": 0,
        "rows_emitted": 0,
        "asset_universe": [],
        "horizons": [],
    }

    if events.empty:
        return [], summary

    asset_filter: set[str] | None
    if asset_universe is None:
        asset_filter = None
    else:
        asset_filter = {str(a) for a in asset_universe}

    horizon_filter: set[int] | None
    if horizons is None:
        horizon_filter = None
    else:
        horizon_filter = {int(h) for h in horizons}

    events_per_cell = _collapse_events_to_meeting_axis(events)

    ling_cols = (
        _linguistic_feature_names(linguistic_features.columns)
        if not linguistic_features.empty
        else ()
    )

    # Index mp_surprises by event_date once.
    mp_by_date: dict[str, pd.Series] = {}
    if not mp_surprises.empty:
        for _, row in mp_surprises.iterrows():
            mp_by_date[str(row["event_date"])] = row

    feature_names = {
        "ois": _ois_feature_names(),
        "text": _text_feature_names(),
        "linguistic": ling_cols,
        "credibility": _credibility_feature_names(),
        "macro": _macro_feature_names(),
    }

    out_rows: list[CrossAssetRow] = []
    seen_assets: set[str] = set()
    seen_horizons: set[int] = set()

    for _, event_row in events_per_cell.iterrows():
        asset = str(event_row.get("asset_symbol", ""))
        if not asset:
            summary["dropped_unknown_asset"] += 1
            continue
        if asset_filter is not None and asset not in asset_filter:
            summary["dropped_unknown_asset"] += 1
            continue

        try:
            horizon = int(event_row["horizon"])
        except (KeyError, TypeError, ValueError):
            summary["dropped_unknown_horizon"] += 1
            continue
        if horizon_filter is not None and horizon not in horizon_filter:
            summary["dropped_unknown_horizon"] += 1
            continue

        abnormal = _safe_abnormal_return(event_row.get("abnormal_return"))
        if abnormal is None:
            summary["dropped_missing_target"] += 1
            continue

        try:
            feature_event_date = _dt.date.fromisoformat(str(event_row["event_date"]))
        except (KeyError, ValueError):
            summary["dropped_missing_target"] += 1
            continue

        mp_row = mp_by_date.get(feature_event_date.isoformat())
        # When mp_surprises is missing for this meeting, synthesize a
        # zero-valued series so the OIS feature vector has consistent
        # length. The baseline_bp will land at ``None``.
        if mp_row is None and not mp_surprises.empty:
            mp_row = pd.Series(dict.fromkeys(mp_surprises.columns))

        ois_vec = _extract_ois_features(mp_row) if mp_row is not None else [0.0] * len(_ois_feature_names())
        text_vec = _extract_text_features(event_row)
        cred_vec = _extract_credibility_features(event_row)
        macro_vec = _extract_macro_features(macro_state, feature_event_date)
        ling_vec = _extract_linguistic_features(
            event_row.get("text_hash"), linguistic_features, ling_cols
        )

        ois_bp = _ois_baseline_bp_from_mp_row(mp_row) if mp_row is not None else None

        out_rows.append(
            CrossAssetRow(
                feature_event_date=feature_event_date,
                asset_symbol=asset,
                horizon=horizon,
                abnormal_return=float(abnormal),
                ois=ois_vec,
                text=text_vec,
                linguistic=ling_vec,
                credibility=cred_vec,
                macro=macro_vec,
                ois_baseline_bp=ois_bp,
                feature_names=feature_names,
            )
        )
        seen_assets.add(asset)
        seen_horizons.add(horizon)

    out_rows.sort(
        key=lambda r: (r.feature_event_date, r.asset_symbol, r.horizon)
    )
    summary["rows_emitted"] = len(out_rows)
    summary["asset_universe"] = sorted(seen_assets)
    summary["horizons"] = sorted(seen_horizons)
    return out_rows, summary


# ---------------------------------------------------------------------------
# Feature matrix
# ---------------------------------------------------------------------------


def _build_feature_matrix(
    rows: Sequence[CrossAssetRow], families: Sequence[str]
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Stack the requested family vectors row-wise; returns ``(X, col_names)``."""

    if not rows:
        return np.zeros((0, 0), dtype=np.float64), ()
    col_names: list[str] = []
    for fam in families:
        col_names.extend(rows[0].feature_names.get(fam, ()))
    matrix = np.zeros((len(rows), len(col_names)), dtype=np.float64)
    for i, row in enumerate(rows):
        cursor = 0
        for fam in families:
            vec = getattr(row, fam)
            n = len(vec)
            matrix[i, cursor : cursor + n] = vec
            cursor += n
    return matrix, tuple(col_names)


def _standardise(
    X_train: np.ndarray, X_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Fit per-column ``(mean, std)`` on train; apply to both.

    Train-only fit honours the project's no-leakage contract.
    Zero-variance columns are passed through unchanged.
    """

    if X_train.size == 0:
        return X_train, X_test
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std_safe = np.where(std < 1e-8, 1.0, std)
    return (X_train - mean) / std_safe, (X_test - mean) / std_safe


# ---------------------------------------------------------------------------
# Per-cell walk-forward CV
# ---------------------------------------------------------------------------


@dataclass
class CellPrediction:
    """One held-out meeting in one ``(asset, horizon)`` cell."""

    feature_event_date: str
    asset_symbol: str
    horizon: int
    target: float
    n_train_rows: int
    predictions: dict[str, float] = field(default_factory=dict)


def _rows_for_cell(
    rows: Sequence[CrossAssetRow], *, asset: str, horizon: int
) -> list[CrossAssetRow]:
    """Subset by ``(asset, horizon)``; preserves time ordering."""

    return [
        r for r in rows if r.asset_symbol == asset and r.horizon == horizon
    ]


def walk_forward_predict_cell(
    cell_rows: Sequence[CrossAssetRow],
    *,
    families: Sequence[str] = FEATURE_FAMILIES,
    include_ridge: bool = True,
    include_gbt: bool = True,
    include_zero_baseline: bool = True,
    include_ois_bp_baseline: bool = True,
    ridge_alpha: float = 1.0,
    random_state: int = 11,
) -> list[CellPrediction]:
    """Leave-one-meeting-out walk-forward CV inside a single cell.

    Mirrors :func:`next_fomc_decision.walk_forward_predict` but the
    target is a float (abnormal return) so we use regressors and
    report regression metrics. Train folds with fewer rows than the
    feature dimension fall back to baselines only -- ridge with
    ``alpha=1`` can technically fit underdetermined systems but the
    answer is dominated by the prior, so we skip it.
    """

    X_all, _ = _build_feature_matrix(cell_rows, families)
    y_all = np.asarray([r.abnormal_return for r in cell_rows], dtype=np.float64)

    preds: list[CellPrediction] = []
    for i, row in enumerate(cell_rows):
        X_train = X_all[:i]
        y_train = y_all[:i]
        X_test = X_all[i : i + 1]

        # No-look-ahead assertion. Every train row's feature_event_date
        # must be strictly older than the held-out row's.
        for j in range(i):
            assert cell_rows[j].feature_event_date < row.feature_event_date, (
                "walk-forward leak: cell row "
                f"{j} feature_date={cell_rows[j].feature_event_date} "
                "not strictly before held-out feature_date="
                f"{row.feature_event_date}"
            )

        per_model: dict[str, float] = {}

        if include_zero_baseline:
            per_model["zero_baseline"] = 0.0
        if include_ois_bp_baseline:
            # bp signal -> fractional return space. abnormal_return is a
            # fractional close-to-close return (0.01 == 1%) so 1 bp =
            # 0.0001. We divide the raw bp by 10000. The signal is
            # rate-direction, not a calibrated return prediction;
            # docstring caveat below covers the asymmetry between
            # rate-sensitive cells (^TNX) and equity cells (XLE).
            if row.ois_baseline_bp is not None:
                per_model["ois_bp_baseline"] = float(row.ois_baseline_bp) / 10000.0
            else:
                per_model["ois_bp_baseline"] = 0.0

        feature_dim = X_train.shape[1] if X_train.size else 0
        if i > feature_dim and feature_dim > 0:
            X_tr_s, X_te_s = _standardise(X_train, X_test)
            if include_ridge:
                try:
                    from sklearn.linear_model import Ridge

                    model = Ridge(alpha=ridge_alpha, random_state=random_state)
                    model.fit(X_tr_s, y_train)
                    per_model["ridge"] = float(model.predict(X_te_s)[0])
                except Exception as exc:  # noqa: BLE001 -- log + fall through
                    LOGGER.warning("Ridge fit failed at row %d: %s", i, exc)

            if include_gbt:
                try:
                    from sklearn.ensemble import HistGradientBoostingRegressor

                    gbt = HistGradientBoostingRegressor(
                        random_state=random_state, max_iter=200
                    )
                    gbt.fit(X_tr_s, y_train)
                    per_model["hist_gbt"] = float(gbt.predict(X_te_s)[0])
                except Exception as exc:  # noqa: BLE001
                    LOGGER.warning(
                        "HistGradientBoostingRegressor failed at row %d: %s", i, exc
                    )

        per_model = {k: round(v, 8) for k, v in per_model.items()}

        preds.append(
            CellPrediction(
                feature_event_date=row.feature_event_date.isoformat(),
                asset_symbol=row.asset_symbol,
                horizon=row.horizon,
                target=round(float(row.abnormal_return), 8),
                n_train_rows=i,
                predictions=per_model,
            )
        )

    return preds


# ---------------------------------------------------------------------------
# Pooled-panel exploration
# ---------------------------------------------------------------------------


def walk_forward_predict_pooled(
    rows: Sequence[CrossAssetRow],
    *,
    families: Sequence[str] = FEATURE_FAMILIES,
    ridge_alpha: float = 1.0,
    random_state: int = 11,
) -> list[CellPrediction]:
    """Optional pooled-panel ridge with asset / horizon dummies.

    Exploratory comparator: one regression for the whole panel,
    asset and horizon enter as one-hot dummies. Walk-forward time
    boundary is the held-out row's ``feature_event_date``: train on
    every panel row whose date is strictly earlier. Marked
    exploratory because the additivity assumption (asset and horizon
    effects share the same coefficient vector across all features)
    is implausible -- DGS yields and equity sectors do not share a
    sentiment-to-return slope.

    Predictions land under model name ``pooled_ridge`` in the
    returned ``CellPrediction`` payloads.
    """

    if not rows:
        return []
    X_feat, _ = _build_feature_matrix(rows, families)
    assets = sorted({r.asset_symbol for r in rows})
    horizons = sorted({r.horizon for r in rows})
    asset_to_idx = {a: i for i, a in enumerate(assets)}
    horizon_to_idx = {h: i for i, h in enumerate(horizons)}

    n = len(rows)
    asset_dum = np.zeros((n, len(assets)), dtype=np.float64)
    horizon_dum = np.zeros((n, len(horizons)), dtype=np.float64)
    for i, row in enumerate(rows):
        asset_dum[i, asset_to_idx[row.asset_symbol]] = 1.0
        horizon_dum[i, horizon_to_idx[row.horizon]] = 1.0

    X_full = np.hstack([X_feat, asset_dum, horizon_dum])
    y_all = np.asarray([r.abnormal_return for r in rows], dtype=np.float64)

    # We need a row index ordered by date so the walk-forward time
    # boundary is well-defined across the panel.
    order = sorted(
        range(n),
        key=lambda i: (rows[i].feature_event_date, rows[i].asset_symbol, rows[i].horizon),
    )
    rows_sorted = [rows[i] for i in order]
    X_sorted = X_full[order]
    y_sorted = y_all[order]

    preds: list[CellPrediction] = []
    for i, row in enumerate(rows_sorted):
        # Train rows: every sorted row whose date is strictly earlier
        # than the held-out row's date. Equal-date rows in other
        # cells are excluded to avoid leaking same-event information
        # across assets.
        train_mask = np.asarray(
            [
                rows_sorted[j].feature_event_date < row.feature_event_date
                for j in range(n)
            ],
            dtype=bool,
        )
        X_train = X_sorted[train_mask]
        y_train = y_sorted[train_mask]
        X_test = X_sorted[i : i + 1]
        per_model: dict[str, float] = {}
        if X_train.shape[0] > X_full.shape[1]:
            try:
                from sklearn.linear_model import Ridge

                X_tr_s, X_te_s = _standardise(X_train, X_test)
                model = Ridge(alpha=ridge_alpha, random_state=random_state)
                model.fit(X_tr_s, y_train)
                per_model["pooled_ridge"] = round(
                    float(model.predict(X_te_s)[0]), 8
                )
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Pooled ridge fit failed at row %d: %s", i, exc)
        preds.append(
            CellPrediction(
                feature_event_date=row.feature_event_date.isoformat(),
                asset_symbol=row.asset_symbol,
                horizon=row.horizon,
                target=round(float(row.abnormal_return), 8),
                n_train_rows=int(train_mask.sum()),
                predictions=per_model,
            )
        )
    return preds


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_cell_metrics(
    preds: Sequence[CellPrediction], model_name: str
) -> dict[str, Any]:
    """RMSE / MAE / R^2 / directional-hit-rate for one model in one cell.

    The directional hit rate is ``mean(sign(pred) == sign(truth))``
    excluding rows where either side is exactly zero (those land in
    the "ties" bucket). When no rows have a prediction, all metrics
    return ``None``.
    """

    y_true: list[float] = []
    y_pred: list[float] = []
    for p in preds:
        v = p.predictions.get(model_name)
        if v is None:
            continue
        y_true.append(float(p.target))
        y_pred.append(float(v))
    n = len(y_true)
    if n == 0:
        return {
            "n": 0,
            "rmse": None,
            "mae": None,
            "r2": None,
            "directional_hit_rate": None,
            "directional_n": 0,
        }
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    err = yp - yt
    rmse = float(math.sqrt(float((err ** 2).mean())))
    mae = float(np.abs(err).mean())
    var_truth = float(((yt - yt.mean()) ** 2).sum())
    if var_truth <= 1e-18:
        r2: float | None = None
    else:
        ss_res = float((err ** 2).sum())
        r2 = float(1.0 - ss_res / var_truth)

    truth_sign = np.sign(yt)
    pred_sign = np.sign(yp)
    nonzero = (truth_sign != 0) & (pred_sign != 0)
    dir_n = int(nonzero.sum())
    if dir_n == 0:
        hit_rate: float | None = None
    else:
        hit_rate = float((truth_sign[nonzero] == pred_sign[nonzero]).mean())
    return {
        "n": int(n),
        "rmse": round(rmse, 8),
        "mae": round(mae, 8),
        "r2": None if r2 is None else round(r2, 8),
        "directional_hit_rate": None if hit_rate is None else round(hit_rate, 8),
        "directional_n": dir_n,
    }


def filter_predictions_excluding_window(
    preds: Sequence[CellPrediction], *, start: _dt.date, end: _dt.date
) -> list[CellPrediction]:
    """Drop predictions whose ``feature_event_date`` falls inside ``[start, end]``."""

    keep: list[CellPrediction] = []
    for pred in preds:
        try:
            d = _dt.date.fromisoformat(pred.feature_event_date)
        except ValueError:
            keep.append(pred)
            continue
        if start <= d <= end:
            continue
        keep.append(pred)
    return keep


# ---------------------------------------------------------------------------
# Top-level run
# ---------------------------------------------------------------------------


@dataclass
class RunArtifacts:
    """Returned by :func:`run`."""

    rows: list[CrossAssetRow]
    predictions: list[CellPrediction]
    metrics: dict[str, Any]
    summary: dict[str, Any]
    ablation_metrics: dict[str, dict[str, Any]]
    asset_universe: list[str]
    horizons: list[int]


def _per_cell_metrics(
    preds: Sequence[CellPrediction],
    *,
    assets: Sequence[str],
    horizons: Sequence[int],
    model_names: Sequence[str],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Nested dict: ``cell_id -> model -> metrics``."""

    by_cell: dict[tuple[str, int], list[CellPrediction]] = {
        (a, h): [] for a in assets for h in horizons
    }
    for p in preds:
        key = (p.asset_symbol, int(p.horizon))
        if key in by_cell:
            by_cell[key].append(p)
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for (asset, horizon), cell_preds in by_cell.items():
        cell_id = f"{asset}|h{horizon}"
        out[cell_id] = {
            m: compute_cell_metrics(cell_preds, m) for m in model_names
        }
    return out


def _aggregate_model_names(preds: Sequence[CellPrediction]) -> list[str]:
    names: set[str] = set()
    for p in preds:
        for k in p.predictions:
            names.add(k)
    return sorted(names)


def run(
    *,
    events: pd.DataFrame,
    mp_surprises: pd.DataFrame,
    linguistic_features: pd.DataFrame,
    macro_state: pd.DataFrame,
    output_dir: Path | None = None,
    asset_universe: Sequence[str] | None = None,
    horizons: Sequence[int] | None = None,
    include_pooled: bool = True,
    random_state: int = 11,
) -> RunArtifacts:
    """End-to-end run: assemble, per-cell walk-forward, score, optionally write."""

    rows, summary = build_supervised_rows(
        events=events,
        mp_surprises=mp_surprises,
        linguistic_features=linguistic_features,
        macro_state=macro_state,
        asset_universe=asset_universe,
        horizons=horizons,
    )

    realised_assets: list[str] = list(summary["asset_universe"])
    realised_horizons: list[int] = list(summary["horizons"])

    per_cell_preds: list[CellPrediction] = []
    for asset in realised_assets:
        for horizon in realised_horizons:
            cell_rows = _rows_for_cell(rows, asset=asset, horizon=horizon)
            if not cell_rows:
                continue
            preds = walk_forward_predict_cell(
                cell_rows,
                families=FEATURE_FAMILIES,
                random_state=random_state,
            )
            per_cell_preds.extend(preds)

    if include_pooled and rows:
        pooled_preds = walk_forward_predict_pooled(
            rows,
            families=FEATURE_FAMILIES,
            random_state=random_state,
        )
        # Merge ``pooled_ridge`` predictions onto matching per-cell
        # CellPrediction entries keyed by (date, asset, horizon).
        index = {
            (p.feature_event_date, p.asset_symbol, p.horizon): p for p in per_cell_preds
        }
        for pp in pooled_preds:
            key = (pp.feature_event_date, pp.asset_symbol, pp.horizon)
            target = index.get(key)
            if target is not None:
                target.predictions.update(pp.predictions)
            else:
                per_cell_preds.append(pp)
        per_cell_preds.sort(
            key=lambda p: (p.feature_event_date, p.asset_symbol, p.horizon)
        )

    model_names = _aggregate_model_names(per_cell_preds)

    metrics_full = _per_cell_metrics(
        per_cell_preds,
        assets=realised_assets,
        horizons=realised_horizons,
        model_names=model_names,
    )
    ex_pandemic_preds = filter_predictions_excluding_window(
        per_cell_preds, start=PANDEMIC_START, end=PANDEMIC_END
    )
    metrics_ex_pandemic = _per_cell_metrics(
        ex_pandemic_preds,
        assets=realised_assets,
        horizons=realised_horizons,
        model_names=model_names,
    )

    metrics: dict[str, Any] = {
        "full_window": metrics_full,
        "ex_pandemic_window": metrics_ex_pandemic,
        "pandemic_window": {
            "start": PANDEMIC_START.isoformat(),
            "end": PANDEMIC_END.isoformat(),
        },
        "model_names": model_names,
        "asset_universe": realised_assets,
        "horizons": realised_horizons,
        "n_predictions": len(per_cell_preds),
        "methodology_source": "cross_asset_event_response_v1",
    }

    ablation_metrics = _run_ablations(
        rows,
        random_state=random_state,
    )

    if output_dir is not None:
        write_artifacts(
            output_dir=Path(output_dir),
            predictions=per_cell_preds,
            metrics=metrics,
            summary=summary,
            ablations=ablation_metrics,
        )

    return RunArtifacts(
        rows=rows,
        predictions=per_cell_preds,
        metrics=metrics,
        summary=summary,
        ablation_metrics=ablation_metrics,
        asset_universe=realised_assets,
        horizons=realised_horizons,
    )


def _count_features(rows: Sequence[CrossAssetRow], families: Sequence[str]) -> int:
    if not rows:
        return 0
    return sum(len(rows[0].feature_names.get(fam, ())) for fam in families)


def _run_ablations(
    rows: Sequence[CrossAssetRow],
    *,
    random_state: int,
) -> dict[str, dict[str, Any]]:
    """Per-feature-family ablation on the ridge model only.

    Runs once per :data:`ABLATION_SUBSETS` entry, recording the metric
    bundle on each headline cell so the ``feature_attribution.md``
    table can compare what each family adds. The ridge model is the
    one we ablate; the gradient-boosted model is left at its full
    feature set to keep run time bounded.
    """

    out: dict[str, dict[str, Any]] = {}
    for name, fams in ABLATION_SUBSETS:
        cell_preds: list[CellPrediction] = []
        for asset, horizon in HEADLINE_CELLS:
            cell_rows = _rows_for_cell(rows, asset=asset, horizon=horizon)
            if not cell_rows:
                continue
            preds = walk_forward_predict_cell(
                cell_rows,
                families=fams,
                include_ridge=True,
                include_gbt=False,
                include_zero_baseline=False,
                include_ois_bp_baseline=False,
                random_state=random_state,
            )
            cell_preds.extend(preds)
        per_cell: dict[str, dict[str, Any]] = {}
        for asset, horizon in HEADLINE_CELLS:
            this_cell = [
                p for p in cell_preds if p.asset_symbol == asset and p.horizon == horizon
            ]
            per_cell[f"{asset}|h{horizon}"] = compute_cell_metrics(this_cell, "ridge")
        out[name] = {
            "families": list(fams),
            "n_features": _count_features(rows, fams),
            "headline_cells": per_cell,
        }

    # Two model-free reference rows. We run them once each with the
    # full feature pipeline (so the baselines have rows to report)
    # but only collect the baseline columns.
    base_preds: list[CellPrediction] = []
    for asset, horizon in HEADLINE_CELLS:
        cell_rows = _rows_for_cell(rows, asset=asset, horizon=horizon)
        if not cell_rows:
            continue
        preds = walk_forward_predict_cell(
            cell_rows,
            families=("ois",),
            include_ridge=False,
            include_gbt=False,
            include_zero_baseline=True,
            include_ois_bp_baseline=True,
            random_state=random_state,
        )
        base_preds.extend(preds)
    for model_name, label in (
        ("zero_baseline", "zero_baseline"),
        ("ois_bp_baseline", "ois_bp_baseline"),
    ):
        baseline_per_cell: dict[str, dict[str, Any]] = {}
        for asset, horizon in HEADLINE_CELLS:
            this_cell = [
                p for p in base_preds if p.asset_symbol == asset and p.horizon == horizon
            ]
            baseline_per_cell[f"{asset}|h{horizon}"] = compute_cell_metrics(
                this_cell, model_name
            )
        out[label] = {
            "families": [label],
            "n_features": 0,
            "headline_cells": baseline_per_cell,
        }
    return out


# ---------------------------------------------------------------------------
# Artifact writers
# ---------------------------------------------------------------------------


def write_artifacts(
    *,
    output_dir: Path,
    predictions: Sequence[CellPrediction],
    metrics: Mapping[str, Any],
    summary: Mapping[str, Any],
    ablations: Mapping[str, Any],
) -> None:
    """Persist predictions.json, metrics.json, feature_attribution.md."""

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "predictions.json").write_text(
        json.dumps(
            {
                "predictions": [
                    {
                        "feature_event_date": p.feature_event_date,
                        "asset_symbol": p.asset_symbol,
                        "horizon": p.horizon,
                        "target": p.target,
                        "n_train_rows": p.n_train_rows,
                        "predictions": p.predictions,
                    }
                    for p in predictions
                ],
                "summary": _stringify_summary(summary),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(dict(metrics), indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_dir / "feature_attribution.md").write_text(
        _format_attribution_md(ablations), encoding="utf-8"
    )


def _stringify_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in summary.items():
        if isinstance(v, list | tuple):
            out[k] = list(v)
        else:
            out[k] = v
    return out


def _format_attribution_md(ablations: Mapping[str, Any]) -> str:
    """Format the per-family ablation table for headline cells.

    Two tables -- one per headline cell -- so the reader can see how
    each subset moves the RMSE / MAE / directional-hit-rate.
    """

    lines = [
        "# Cross-asset response head -- feature-family attribution",
        "",
        "Each row is a per-cell leave-one-meeting-out walk-forward run on",
        "the ridge model trained on the listed feature subset. Two cells",
        "are highlighted: SPX (^GSPC) at h=1d and h=5d. Model-free rows",
        "(zero_baseline, ois_bp_baseline) sit at the bottom for reference.",
        "",
    ]
    for asset, horizon in HEADLINE_CELLS:
        cell_id = f"{asset}|h{horizon}"
        lines.append(f"## Headline cell: {cell_id}")
        lines.append("")
        lines.append(
            "| Subset | Families | #features | n | RMSE | MAE | R^2 | DirHitRate |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for name, payload in ablations.items():
            cells = payload.get("headline_cells", {})
            m = cells.get(cell_id, {})
            lines.append(
                "| {name} | {fams} | {nf} | {n} | {rmse} | {mae} | {r2} | {dh} |".format(
                    name=name,
                    fams=", ".join(payload.get("families", [])),
                    nf=payload.get("n_features", 0),
                    n=m.get("n"),
                    rmse=m.get("rmse"),
                    mae=m.get("mae"),
                    r2=m.get("r2"),
                    dh=m.get("directional_hit_rate"),
                )
            )
        lines.append("")
    lines.append(
        "Notes: ``zero_baseline`` is the mean-zero null (any model below it on "
        "RMSE is providing no lift); ``ois_bp_baseline`` is the OIS-implied "
        "bp signal at meeting N's post-event 1m tenor divided by 100, so its "
        "RMSE is comparable on rate-sensitive cells but inflated on equity "
        "cells -- read the module docstring caveat before drawing inference."
    )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _load_parquet_safely(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"required parquet missing: {path}")
    return pd.read_parquet(path)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Predict cross-asset abnormal-return responses to FOMC events "
            "using text + macro + OIS + credibility + linguistic features "
            "(Phase 8 #148)."
        ),
    )
    parser.add_argument("--training-package-id", required=True)
    parser.add_argument(
        "--data-dir",
        default=str(DATA_DIR),
        help="Root data dir (default: app.config.DATA_DIR).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Override for the artifact output dir. Default: "
            "data/artifacts/cross_asset/."
        ),
    )
    parser.add_argument(
        "--events-name",
        default="events.parquet",
        help="Filename of the events parquet under data/processed/<pkg>/.",
    )
    parser.add_argument(
        "--mp-surprises-name",
        default="mp_surprises.parquet",
        help="Filename of the MP surprises parquet under data/external/fred/.",
    )
    parser.add_argument(
        "--linguistic-name",
        default="linguistic_features.parquet",
        help="Filename of the linguistic features parquet under data/processed/<pkg>/.",
    )
    parser.add_argument(
        "--macro-state-name",
        default="macro_state.parquet",
        help="Filename of the macro-state parquet under data/external/fred/.",
    )
    parser.add_argument(
        "--asset",
        action="append",
        default=None,
        help=(
            "Restrict the asset universe. Pass multiple --asset flags to "
            "model a subset. Default: every ``asset_symbol`` in the events "
            "parquet."
        ),
    )
    parser.add_argument(
        "--horizon",
        action="append",
        type=int,
        default=None,
        help=(
            "Restrict the modelled horizons. Pass multiple --horizon flags. "
            "Default: every horizon in the events parquet."
        ),
    )
    parser.add_argument(
        "--no-pooled",
        action="store_true",
        help="Disable the exploratory pooled-panel ridge model.",
    )
    parser.add_argument("--seed", type=int, default=11)
    return parser.parse_args(argv)


def _resolve_iterable(
    flag: Iterable[Any] | None, default: tuple[Any, ...]
) -> tuple[Any, ...]:
    if not flag:
        return default
    return tuple(flag)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    data_dir = Path(args.data_dir)
    package_dir = data_dir / "processed" / args.training_package_id
    fred_dir = data_dir / "external" / "fred"

    events = _load_parquet_safely(package_dir / args.events_name)
    mp = _load_parquet_safely(fred_dir / args.mp_surprises_name)
    ling_path = package_dir / args.linguistic_name
    linguistic = (
        _load_parquet_safely(ling_path) if ling_path.exists() else pd.DataFrame()
    )
    macro_path = fred_dir / args.macro_state_name
    macro = (
        _load_parquet_safely(macro_path) if macro_path.exists() else pd.DataFrame()
    )

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else data_dir / "artifacts" / "cross_asset"
    )

    artifacts = run(
        events=events,
        mp_surprises=mp,
        linguistic_features=linguistic,
        macro_state=macro,
        output_dir=output_dir,
        asset_universe=args.asset,
        horizons=args.horizon,
        include_pooled=not args.no_pooled,
        random_state=args.seed,
    )

    print(f"[cross-asset] rows emitted: {artifacts.summary['rows_emitted']}")
    print(
        f"[cross-asset] asset universe: {', '.join(artifacts.asset_universe) or '<none>'}"
    )
    print(
        f"[cross-asset] horizons: {', '.join(str(h) for h in artifacts.horizons) or '<none>'}"
    )
    print(f"[cross-asset] models: {', '.join(artifacts.metrics['model_names'])}")
    print(f"[cross-asset] output dir: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
