"""Predict the FOMC's next decision (Phase 8 headline, closes #147).

Reframes fed-pulse from price-forecasting (random-walk null) to
central-bank-forecasting (OIS-implied baseline). At each scheduled FOMC
meeting ``N`` we predict the rate decision at meeting ``N+1`` as an
ordinal class:

    cut_50, cut_25, hold, hike_25, hike_50, hike_75

Out-of-set deltas (e.g. the March 2020 75 bp emergency cut sequence)
trigger a warning and the row is dropped from the supervised set --
intermeeting jumbos are reported in the run summary but are not part
of the next-scheduled-meeting forecast target. See #147 acceptance
criteria.

Feature families
----------------

* ``ois``        -- 5-tenor pre-event OIS-implied curve + level/path/info
                    factors from :file:`mp_surprises.parquet`.
* ``text``       -- multi-axis stance / time / certainty / factor
                    from :file:`events.parquet`. Multi-source duplicates
                    collapse to one row per meeting; the statement is the
                    preferred document kind when available.
* ``linguistic`` -- 14-dim structured features from
                    :file:`linguistic_features.parquet` joined on
                    ``text_hash``.
* ``credibility``-- 4-vector already on ``events.parquet`` as
                    ``credibility_*`` columns.
* ``macro``      -- last-published FRED indicators at decision-eve from
                    :func:`app.data.macro_state.build_macro_state`.

Baselines
---------

* ``OISBaseline`` -- per-class probability derived from
  ``pre_event_curve`` of meeting N+1 (the curve published the day before
  the next meeting). The 3-month tenor minus the prior fed-funds target
  is the OIS-implied next-meeting rate change in bp; we smooth it with a
  Gaussian (sigma = 12.5 bp, the default rationale documented below)
  into a probability vector over the class set.
* ``NaiveCarry`` -- ``P(hold) = 1`` regardless of features.

Sigma rationale: 12.5 bp is half the smallest non-zero class step (25
bp), so the Gaussian softly partitions the bp axis at class midpoints
without aliasing one class onto its neighbour. Tightening sigma toward
0 collapses to a hard-step assignment (a fine sanity benchmark) but
under-counts uncertainty at meetings where OIS prices a 50/50
hold-vs-cut split.

Ordinal lift
------------

Primary model: a NumPy-only **proportional-odds logit** (cumulative
ordered-logit) with L2 regularisation, fit via :func:`scipy.optimize.minimize`
(``L-BFGS-B``). When :mod:`statsmodels` is available the dispatcher will
prefer ``statsmodels.miscmodels.ordinal_model.OrderedModel`` because the
external-library implementation has been battle-tested in
peer-reviewed code; when :mod:`mord` is available the second-priority
fallback uses ``mord.LogisticIT``. Neither library is currently in
``backend/pyproject.toml`` so the NumPy-only path is the default. The
choice is logged on every run via :data:`OrdinalModel.backend`.

Secondary lift: :class:`sklearn.ensemble.HistGradientBoostingClassifier`
in multi-class mode. Acts as the stronger-non-linear comparator.

Walk-forward cross-validation
-----------------------------

Leave-one-meeting-out: for each held-out meeting ``M`` we fit each
model on all meetings strictly older than ``M`` (the test row's
``as_of_ts`` is the next-meeting ``as_of_ts``; train rows have target
``as_of_ts`` strictly less). Per-meeting predictions land in
``data/artifacts/next_fomc/results.json``; aggregate metrics in
``data/artifacts/next_fomc/metrics.json``; per-feature-family ablation
table in ``data/artifacts/next_fomc/feature_attribution.md``. We also
emit metrics for the pandemic-excluded window (cutting
``2020-04-01..2021-06-30``) so the reviewer can see whether the regime
break is doing the heavy lifting.

Determinism
-----------

* Same input parquets imply byte-identical artifacts. Predictions
  iterate over a sorted meeting list; probability vectors are rounded
  to 6 decimals before serialisation; JSON dumps use
  ``sort_keys=True, indent=2``.
* ``HistGradientBoostingClassifier`` is seeded with ``random_state=11``.
* L-BFGS-B has no random component; the loss function is convex.

No look-ahead
-------------

For each meeting ``M`` we predict the target at meeting ``M+1`` using
features dated strictly before ``M+1``'s ``as_of_ts``. Walk-forward
training uses *only* meetings ``< M+1.as_of_ts`` -- the constructor
asserts this on every fold.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import math
import sys
import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from app.config import DATA_DIR
from app.data.macro_state import COLUMN_ORDER as MACRO_STATE_COLUMNS

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Class set + delta -> class mapping
# ---------------------------------------------------------------------------

ORDINAL_CLASSES: tuple[str, ...] = (
    "cut_50",
    "cut_25",
    "hold",
    "hike_25",
    "hike_50",
    "hike_75",
)

# Class -> bp midpoint (used by the OIS baseline + report-time anchoring).
CLASS_BP: dict[str, float] = {
    "cut_50": -50.0,
    "cut_25": -25.0,
    "hold": 0.0,
    "hike_25": 25.0,
    "hike_50": 50.0,
    "hike_75": 75.0,
}

# Hard half-step boundaries used to assign an observed delta to a class.
# 12.5 bp is half a 25 bp step.
_BIN_TOL_BP: float = 12.5

# Documented sigma for the Gaussian OIS smoothing. See module docstring
# for the rationale. Tests assert this constant rather than reading a
# magic number.
OIS_BASELINE_SIGMA_BP: float = 12.5

# Pandemic-era window we ablate against (#147 acceptance criterion).
PANDEMIC_START: _dt.date = _dt.date(2020, 4, 1)
PANDEMIC_END: _dt.date = _dt.date(2021, 6, 30)


def delta_to_class(delta_bp: float | None, *, tol_bp: float = _BIN_TOL_BP) -> str | None:
    """Map a basis-point rate change to an ordinal class.

    Returns ``None`` and emits a :class:`UserWarning` if ``delta_bp``
    falls outside the supported class set (e.g. a 75 bp cut, a 100 bp
    move). This keeps 2008-era and 2020-emergency jumbo moves visible
    instead of silently coerced to the nearest in-set class.
    """

    if delta_bp is None or (isinstance(delta_bp, float) and math.isnan(delta_bp)):
        return None
    best: tuple[float, str] | None = None
    for cls, bp in CLASS_BP.items():
        d = abs(float(delta_bp) - bp)
        if d <= tol_bp:
            if best is None or d < best[0]:
                best = (d, cls)
    if best is None:
        warnings.warn(
            f"Rate delta {delta_bp:+.2f} bp outside supported class set "
            f"{ORDINAL_CLASSES}; row dropped from supervised set.",
            UserWarning,
            stacklevel=2,
        )
        return None
    return best[1]


# ---------------------------------------------------------------------------
# Feature families
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


# ---------------------------------------------------------------------------
# Proportional-odds (ordered logit) -- NumPy/SciPy fallback
# ---------------------------------------------------------------------------


@dataclass
class ProportionalOddsLogit:
    """Cumulative ordered-logit with L2 regularisation.

    The model is the standard McCullagh (1980) proportional-odds form:

        P(Y <= k | x) = sigmoid(theta_k - beta . x)
        P(Y = k)     = P(Y <= k) - P(Y <= k - 1)

    where ``theta_0 < theta_1 < ... < theta_{K-2}`` are class
    thresholds and ``beta`` is the shared feature-weight vector.
    Constraint is parameterised away by writing
    ``theta_k = theta_0 + sum_{j<=k} softplus(eta_j)`` so the optimiser
    sees an unconstrained vector. ``alpha`` is the L2 penalty applied
    only to ``beta`` (not to thresholds), default 1.0. Setting
    ``alpha=0`` reproduces unregularised MLE.

    Pure NumPy + ``scipy.optimize.minimize`` -- no statsmodels / mord
    dependency. Convex, deterministic.
    """

    alpha: float = 1.0
    n_classes_: int = 0
    n_features_: int = 0
    theta_: np.ndarray | None = None
    beta_: np.ndarray | None = None
    classes_: tuple[str, ...] = field(default_factory=tuple)

    def fit(self, X: np.ndarray, y: np.ndarray, classes: Sequence[str]) -> "ProportionalOddsLogit":
        from scipy import optimize  # local import keeps top-level deps lean

        X = np.asarray(X, dtype=np.float64)
        # ``y`` is integer-encoded over ``classes`` (already validated upstream).
        y = np.asarray(y, dtype=np.int64)
        n, p = X.shape
        K = len(classes)
        if K < 2:
            raise ValueError(f"Need at least 2 classes, got {K}")
        if n != y.shape[0]:
            raise ValueError(f"X has {n} rows, y has {y.shape[0]}")

        def unpack(params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            # First (K-1) params encode the threshold vector via softplus
            # gaps so order is enforced.
            theta_raw = params[: K - 1]
            beta = params[K - 1 :]
            theta = np.empty(K - 1, dtype=np.float64)
            theta[0] = theta_raw[0]
            # Softplus on the remaining (K-2) entries -> positive gaps.
            for k in range(1, K - 1):
                gap = math.log1p(math.exp(theta_raw[k])) if theta_raw[k] < 30 else theta_raw[k]
                theta[k] = theta[k - 1] + gap
            return theta, beta

        def neg_log_lik(params: np.ndarray) -> float:
            theta, beta = unpack(params)
            eta = X @ beta  # (n,)
            # Cumulative probabilities P(Y <= k) for k = 0..K-2.
            cum_logits = theta[np.newaxis, :] - eta[:, np.newaxis]  # (n, K-1)
            cum_prob = 1.0 / (1.0 + np.exp(-cum_logits))
            # Per-row class probabilities by differencing the cumulative.
            #   P(Y = 0)   = cum_prob[:, 0]
            #   P(Y = k)   = cum_prob[:, k] - cum_prob[:, k-1]   for 0 < k < K-1
            #   P(Y = K-1) = 1 - cum_prob[:, K-2]
            probs = np.zeros((n, K), dtype=np.float64)
            probs[:, 0] = cum_prob[:, 0]
            for k in range(1, K - 1):
                probs[:, k] = cum_prob[:, k] - cum_prob[:, k - 1]
            probs[:, K - 1] = 1.0 - cum_prob[:, K - 2]
            probs = np.clip(probs, 1e-12, 1.0)
            ll = float(np.log(probs[np.arange(n), y]).sum())
            reg = 0.5 * self.alpha * float(np.dot(beta, beta))
            return -ll + reg

        # Initial params: thresholds spread out, beta = 0.
        init_theta_raw = np.zeros(K - 1, dtype=np.float64)
        # Spread thresholds evenly across logits in [-2, 2] so the
        # optimiser doesn't start at a degenerate cumulative-prob ridge.
        init_theta_raw[0] = -2.0
        if K - 2 > 0:
            # Softplus(log(exp(gap)-1)) = gap; pick gap = 4/(K-1).
            gap = 4.0 / max(K - 1, 1)
            inv_softplus_gap = math.log(math.exp(gap) - 1) if gap > 0 else -10.0
            init_theta_raw[1:] = inv_softplus_gap
        init_beta = np.zeros(p, dtype=np.float64)
        init = np.concatenate([init_theta_raw, init_beta])

        result = optimize.minimize(
            neg_log_lik,
            init,
            method="L-BFGS-B",
            options={"maxiter": 500, "ftol": 1e-9},
        )
        theta, beta = unpack(result.x)
        self.theta_ = theta
        self.beta_ = beta
        self.n_classes_ = K
        self.n_features_ = p
        self.classes_ = tuple(classes)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.theta_ is None or self.beta_ is None:
            raise RuntimeError("Model not fit")
        X = np.asarray(X, dtype=np.float64)
        K = self.n_classes_
        eta = X @ self.beta_
        cum_logits = self.theta_[np.newaxis, :] - eta[:, np.newaxis]
        cum_prob = 1.0 / (1.0 + np.exp(-cum_logits))
        probs = np.zeros((X.shape[0], K), dtype=np.float64)
        probs[:, 0] = cum_prob[:, 0]
        for k in range(1, K - 1):
            probs[:, k] = cum_prob[:, k] - cum_prob[:, k - 1]
        probs[:, K - 1] = 1.0 - cum_prob[:, K - 2]
        probs = np.clip(probs, 1e-12, 1.0)
        # Re-normalise after the clip so rows sum to exactly 1.
        probs /= probs.sum(axis=1, keepdims=True)
        return probs


# ---------------------------------------------------------------------------
# Ordinal model dispatcher
# ---------------------------------------------------------------------------


@dataclass
class OrdinalModelHandle:
    """Backend-agnostic wrapper. ``predict_proba`` is the public surface."""

    backend: str
    fit_fn: Callable[[np.ndarray, np.ndarray], None]
    predict_proba_fn: Callable[[np.ndarray], np.ndarray]


def _dispatch_ordinal(
    *,
    classes: Sequence[str],
    alpha: float = 1.0,
    prefer: str | None = None,
) -> OrdinalModelHandle:
    """Pick the best available ordered-logit backend.

    Preference order (highest first): ``statsmodels`` ->  ``mord`` ->
    NumPy/SciPy fallback. Pass ``prefer="numpy"`` to force the
    fallback (used by tests so the assertion that the fallback path is
    exercised does not depend on which libraries happen to be in the
    test environment).
    """

    if prefer == "numpy":
        return _build_numpy_handle(classes=classes, alpha=alpha)
    if prefer == "statsmodels":
        try:
            return _build_statsmodels_handle(classes=classes, alpha=alpha)
        except ImportError:
            pass
    if prefer == "mord":
        try:
            return _build_mord_handle(classes=classes, alpha=alpha)
        except ImportError:
            pass

    # Auto-select.
    try:
        return _build_statsmodels_handle(classes=classes, alpha=alpha)
    except ImportError:
        pass
    try:
        return _build_mord_handle(classes=classes, alpha=alpha)
    except ImportError:
        pass
    return _build_numpy_handle(classes=classes, alpha=alpha)


def _build_numpy_handle(
    *, classes: Sequence[str], alpha: float
) -> OrdinalModelHandle:
    model = ProportionalOddsLogit(alpha=alpha)

    def _fit(X: np.ndarray, y: np.ndarray) -> None:
        model.fit(X, y, classes)

    def _predict_proba(X: np.ndarray) -> np.ndarray:
        return model.predict_proba(X)

    return OrdinalModelHandle(backend="numpy_proportional_odds", fit_fn=_fit, predict_proba_fn=_predict_proba)


def _build_statsmodels_handle(
    *, classes: Sequence[str], alpha: float
) -> OrdinalModelHandle:
    # Imported lazily; statsmodels is not in the backend's dependencies.
    from statsmodels.miscmodels.ordinal_model import OrderedModel  # type: ignore[import-not-found]

    state: dict[str, Any] = {"model": None, "result": None}

    def _fit(X: np.ndarray, y: np.ndarray) -> None:
        # ``y`` is integer-encoded; convert to a 1-d series. statsmodels
        # accepts any orderable sequence.
        mod = OrderedModel(y, X, distr="logit")
        state["result"] = mod.fit(method="lbfgs", disp=False)
        state["model"] = mod

    def _predict_proba(X: np.ndarray) -> np.ndarray:
        result = state["result"]
        if result is None:
            raise RuntimeError("statsmodels handle not fit")
        # ``predict`` on OrderedModel returns per-class probabilities.
        return np.asarray(result.predict(X), dtype=np.float64)

    return OrdinalModelHandle(backend="statsmodels_ordered_model", fit_fn=_fit, predict_proba_fn=_predict_proba)


def _build_mord_handle(
    *, classes: Sequence[str], alpha: float
) -> OrdinalModelHandle:
    # mord is an optional extra; not declared in pyproject.toml dependencies.
    from mord import LogisticIT  # type: ignore[import-not-found]

    model = LogisticIT(alpha=alpha)

    def _fit(X: np.ndarray, y: np.ndarray) -> None:
        model.fit(X, y)

    def _predict_proba(X: np.ndarray) -> np.ndarray:
        # mord exposes ``predict_proba`` only on some subclasses; fall
        # back to ``decision_function`` -> softmax otherwise.
        if hasattr(model, "predict_proba"):
            return np.asarray(model.predict_proba(X), dtype=np.float64)
        scores = model.decision_function(X)
        exp = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        normalised: np.ndarray = exp / np.sum(exp, axis=1, keepdims=True)
        return normalised

    return OrdinalModelHandle(backend="mord_logistic_it", fit_fn=_fit, predict_proba_fn=_predict_proba)


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------


def ois_baseline_probability(
    *,
    implied_rate: float | None,
    base_rate: float | None,
    sigma_bp: float = OIS_BASELINE_SIGMA_BP,
) -> dict[str, float]:
    """Per-class probability from the OIS-implied curve.

    Computes the implied next-meeting rate change in basis points
    (``(implied_rate - base_rate) * 100``) and smooths it with a
    Gaussian centred on each class midpoint::

        weight_k = exp(-0.5 * ((bp_signal - CLASS_BP[k]) / sigma_bp) ** 2)
        P(k)     = weight_k / sum_j weight_j

    Both inputs are in percent. ``implied_rate`` is the market-implied
    rate at the chosen tenor (the call site picks the curve and tenor);
    ``base_rate`` is the reference rate the deviation is measured
    against. Returns a uniform probability vector when either input is
    missing.
    """

    if implied_rate is None or base_rate is None:
        return _uniform()
    bp_signal = (float(implied_rate) - float(base_rate)) * 100.0
    weights = {
        cls: math.exp(-0.5 * ((bp_signal - bp) / sigma_bp) ** 2)
        for cls, bp in CLASS_BP.items()
    }
    total = sum(weights.values())
    if total <= 1e-18:
        return _uniform()
    return {cls: weights[cls] / total for cls in ORDINAL_CLASSES}


def naive_carry_probability() -> dict[str, float]:
    """``P(hold) = 1`` baseline."""

    return {cls: (1.0 if cls == "hold" else 0.0) for cls in ORDINAL_CLASSES}


def _uniform() -> dict[str, float]:
    p = 1.0 / len(ORDINAL_CLASSES)
    return dict.fromkeys(ORDINAL_CLASSES, p)


# ---------------------------------------------------------------------------
# Curve extraction
# ---------------------------------------------------------------------------


def _curve_value_at(curve_json: Any, months_ahead: int) -> float | None:
    """Pull the implied rate at ``months_ahead`` out of a JSON curve.

    ``pre_event_curve`` is stored as a JSON string with one entry per
    tenor (see :mod:`app.data.mp_surprise`). Returns ``None`` when the
    tenor is missing or the entry is NaN.
    """

    if curve_json is None or (isinstance(curve_json, float) and math.isnan(curve_json)):
        return None
    if isinstance(curve_json, str):
        try:
            curve = json.loads(curve_json)
        except json.JSONDecodeError:
            return None
    else:
        curve = curve_json
    for point in curve:
        try:
            tenor = int(point["months_ahead"])
            value = point["implied_rate"]
        except (KeyError, TypeError, ValueError):
            continue
        if tenor != months_ahead:
            continue
        if value is None:
            return None
        try:
            fv = float(value)
        except (TypeError, ValueError):
            return None
        if math.isnan(fv):
            return None
        return fv
    return None


# ---------------------------------------------------------------------------
# Dataset assembly
# ---------------------------------------------------------------------------


@dataclass
class MeetingRow:
    """One supervised row: features at meeting N predict target at meeting N+1.

    The supervisor identifies a row by ``(target_event_date,
    target_as_of_ts)`` -- the next meeting's announcement timestamp.
    Features carry ``feature_event_date`` (meeting N) for audit.
    """

    target_event_date: _dt.date
    target_as_of_ts: _dt.datetime
    target_class: str
    feature_event_date: _dt.date
    # Per-family feature vectors. Each is a flat float-only list.
    ois: list[float]
    text: list[float]
    linguistic: list[float]
    credibility: list[float]
    macro: list[float]
    # Inputs for the OIS baseline (separate from the feature vector so
    # the baseline does not double-count its inputs against the model).
    # The implied rate is read from meeting N's post-event curve at the
    # 1-month tenor (closest free tenor to the ~6-week inter-meeting
    # window), and the base rate is the ff target rate that meeting N
    # just set. Same information cutoff as the model -- both consume
    # what is known the moment meeting N's decision is public.
    ois_baseline_implied_rate: float | None
    ois_baseline_base_rate: float | None
    # Per-family column names, populated once and shared.
    feature_names: dict[str, tuple[str, ...]] = field(default_factory=dict)


def _events_per_meeting(events: pd.DataFrame) -> pd.DataFrame:
    """Collapse the events frame to one row per ``(event_date, asset, horizon)``.

    Preferred event kind: ``statement`` -> ``minutes`` -> ``press_conference``
    -> first available. Multi-source rows are already collapsed upstream
    when the parquet is built with the default flags.
    """

    if events.empty:
        return events
    # Stable order by preference.
    kind_pref = {"statement": 0, "press_conference": 1, "minutes": 2}
    events = events.copy()
    events["_kind_rank"] = events["event_kind"].map(lambda k: kind_pref.get(str(k), 99))
    # Pick the smallest horizon row for each meeting so the same row
    # feeds the supervised model regardless of how many horizons the
    # builder emitted. Horizons differ only in target columns we don't
    # consume here.
    events = events.sort_values(
        ["event_date", "_kind_rank", "horizon", "event_kind"],
        kind="mergesort",
    )
    events = events.drop_duplicates(subset=["event_date"], keep="first")
    return events.drop(columns=["_kind_rank"]).reset_index(drop=True)


def _axis_to_float(value: Any) -> tuple[float, float, float]:
    """One-hot a stance axis: returns ``(dovish, neutral, hawkish)``."""

    if value is None or (isinstance(value, float) and math.isnan(value)):
        return (0.0, 0.0, 0.0)
    s = str(value).lower()
    if s in {"dovish", "easy", "easing"}:
        return (1.0, 0.0, 0.0)
    if s in {"hawkish", "tight", "tightening"}:
        return (0.0, 0.0, 1.0)
    if s in {"neutral"}:
        return (0.0, 1.0, 0.0)
    return (0.0, 0.0, 0.0)


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        fv = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(fv):
        return default
    return fv


def _ois_feature_names() -> tuple[str, ...]:
    return (
        "pre_curve_1m",
        "pre_curve_3m",
        "pre_curve_6m",
        "pre_curve_12m",
        "pre_curve_24m",
        "mp_surprise_level",
        "mp_surprise_path_factor",
        "fed_info_factor",
        "ff_target_prior",
        "ff_target_after",
    )


def _text_feature_names() -> tuple[str, ...]:
    # 3 one-hot dims per axis * 4 axes = 12 dims. The topic axis was
    # retired in ADR 0044 (no upstream source ships topic labels).
    return tuple(
        f"axis_{axis}_{label}"
        for axis in ("stance", "time", "certainty", "factor")
        for label in ("dovish", "neutral", "hawkish")
    )


def _credibility_feature_names() -> tuple[str, ...]:
    return (
        "credibility_drift_score",
        "credibility_realized_vs_stated_gap",
        "credibility_market_implied_gap",
        "credibility_months_since_reversal",
    )


def _macro_feature_names() -> tuple[str, ...]:
    return (
        "unrate",
        "cpi_yoy",
        "core_pce_yoy",
        "ism_proxy",
        "payems_mom",
        "rsafs_mom",
    )


def _linguistic_feature_names(linguistic_columns: Iterable[str]) -> tuple[str, ...]:
    return tuple(c for c in linguistic_columns if c != "text_hash")


def _extract_ois_features(row: pd.Series) -> list[float]:
    pre_curve = row.get("pre_event_curve")
    return [
        _safe_float(_curve_value_at(pre_curve, 1)),
        _safe_float(_curve_value_at(pre_curve, 3)),
        _safe_float(_curve_value_at(pre_curve, 6)),
        _safe_float(_curve_value_at(pre_curve, 12)),
        _safe_float(_curve_value_at(pre_curve, 24)),
        _safe_float(row.get("mp_surprise_level")),
        _safe_float(row.get("mp_surprise_path_factor")),
        _safe_float(row.get("fed_info_factor")),
        _safe_float(row.get("ff_target_prior")),
        _safe_float(row.get("ff_target_after")),
    ]


def _extract_text_features(row: pd.Series) -> list[float]:
    out: list[float] = []
    for axis_col in ("axis_stance", "axis_time", "axis_certainty", "axis_factor"):
        d, n, h = _axis_to_float(row.get(axis_col))
        out.extend([d, n, h])
    return out


def _extract_credibility_features(row: pd.Series) -> list[float]:
    return [
        _safe_float(row.get("credibility_drift_score")),
        _safe_float(row.get("credibility_realized_vs_stated_gap")),
        _safe_float(row.get("credibility_market_implied_gap")),
        _safe_float(row.get("credibility_months_since_reversal")),
    ]


def _extract_macro_features(
    macro: pd.DataFrame, as_of: _dt.date
) -> list[float]:
    if macro.empty:
        return [0.0] * len(_macro_feature_names())
    iso = as_of.isoformat()
    sub = macro[macro["as_of_date"] < iso]
    if sub.empty:
        return [0.0] * len(_macro_feature_names())
    last = sub.iloc[-1]
    return [
        _safe_float(last.get("unrate")),
        _safe_float(last.get("cpi_yoy")),
        _safe_float(last.get("core_pce_yoy")),
        _safe_float(last.get("ism_proxy")),
        _safe_float(last.get("payems_mom")),
        _safe_float(last.get("rsafs_mom")),
    ]


def _extract_linguistic_features(
    text_hash: Any, ling: pd.DataFrame, columns: tuple[str, ...]
) -> list[float]:
    if ling.empty or text_hash is None:
        return [0.0] * len(columns)
    rows = ling[ling["text_hash"] == text_hash]
    if rows.empty:
        return [0.0] * len(columns)
    last = rows.iloc[0]
    return [_safe_float(last.get(c)) for c in columns]


def _parse_as_of(value: Any) -> _dt.datetime | None:
    if value is None:
        return None
    if isinstance(value, _dt.datetime):
        return value
    s = str(value)
    if not s:
        return None
    # Tolerate both ``2024-09-18T19:00:00Z`` and tz-naive ISO strings.
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        return _dt.datetime.fromisoformat(s)
    except ValueError:
        return None


def build_supervised_rows(
    *,
    events: pd.DataFrame,
    mp_surprises: pd.DataFrame,
    linguistic_features: pd.DataFrame,
    macro_state: pd.DataFrame,
) -> tuple[list[MeetingRow], dict[str, Any]]:
    """Join the four parquets into per-meeting supervised rows.

    Returns ``(rows, summary)``. ``summary`` records dropped-row counts
    (target out of class set, missing features), the meeting span, and
    the linguistic / macro coverage. The rows themselves are sorted by
    ``target_event_date``.
    """

    summary: dict[str, Any] = {
        "rows_in_events": int(len(events)),
        "rows_in_mp_surprises": int(len(mp_surprises)),
        "rows_in_linguistic": int(len(linguistic_features)),
        "rows_in_macro_state": int(len(macro_state)),
        "dropped_no_next_meeting": 0,
        "dropped_target_out_of_class": 0,
        "dropped_target_missing": 0,
        "rows_emitted": 0,
        "intermeeting_meetings_skipped_as_target": 0,
    }

    events_per_meeting = _events_per_meeting(events)
    if events_per_meeting.empty or mp_surprises.empty:
        return [], summary

    ling_cols = _linguistic_feature_names(linguistic_features.columns) if not linguistic_features.empty else ()

    # Index mp_surprises by event_date.
    mp_by_date: dict[str, pd.Series] = {}
    for _, row in mp_surprises.iterrows():
        ed = str(row["event_date"])
        mp_by_date[ed] = row

    # Build the sorted scheduled-meeting list out of mp_surprises so we
    # can identify "next scheduled meeting" -- intermeeting actions can
    # still appear as feature rows but never as a target row.
    mp_sorted = mp_surprises.sort_values("event_date", kind="mergesort").reset_index(drop=True)
    scheduled_dates: list[str] = [
        str(row["event_date"])
        for _, row in mp_sorted.iterrows()
        if not bool(row.get("is_intermeeting"))
    ]
    next_scheduled_after: dict[str, str | None] = {}
    for _, row in mp_sorted.iterrows():
        ed = str(row["event_date"])
        idx = -1
        # First scheduled meeting whose date is strictly > ed.
        for sd in scheduled_dates:
            if sd > ed:
                idx = scheduled_dates.index(sd)
                break
        next_scheduled_after[ed] = scheduled_dates[idx] if idx >= 0 else None

    feature_names = {
        "ois": _ois_feature_names(),
        "text": _text_feature_names(),
        "linguistic": ling_cols,
        "credibility": _credibility_feature_names(),
        "macro": _macro_feature_names(),
    }

    out_rows: list[MeetingRow] = []
    for _, event_row in events_per_meeting.iterrows():
        feature_event_date = _dt.date.fromisoformat(str(event_row["event_date"]))
        ed_iso = feature_event_date.isoformat()
        nxt_iso = next_scheduled_after.get(ed_iso)
        if nxt_iso is None:
            summary["dropped_no_next_meeting"] += 1
            continue
        next_row = mp_by_date.get(nxt_iso)
        if next_row is None:
            summary["dropped_no_next_meeting"] += 1
            continue
        next_event_date = _dt.date.fromisoformat(nxt_iso)
        # Reconstruct target.
        prior = next_row.get("ff_target_prior")
        after = next_row.get("ff_target_after")
        if prior is None or after is None or (
            isinstance(prior, float) and math.isnan(prior)
        ) or (isinstance(after, float) and math.isnan(after)):
            summary["dropped_target_missing"] += 1
            continue
        delta_bp = (float(after) - float(prior)) * 100.0
        target_class = delta_to_class(delta_bp)
        if target_class is None:
            summary["dropped_target_out_of_class"] += 1
            continue

        # The target is the next meeting's announcement; we anchor
        # ``target_as_of`` to the next meeting's date at 19:00 UTC
        # (the same placeholder convention the events builder uses for
        # FOMC kinds).
        target_as_of = _dt.datetime.combine(
            next_event_date, _dt.time(19, 0), _dt.timezone.utc
        )

        # Feature vectors at meeting N (current event_row + current
        # mp_surprise row -- not the next meeting's).
        mp_row = mp_by_date.get(ed_iso)
        if mp_row is None:
            # Synthesize a zero-valued row so the schema is preserved.
            mp_row = pd.Series(dict.fromkeys(mp_surprises.columns))
        ois_vec = _extract_ois_features(mp_row)
        text_vec = _extract_text_features(event_row)
        cred_vec = _extract_credibility_features(event_row)
        macro_vec = _extract_macro_features(macro_state, feature_event_date)
        ling_vec = _extract_linguistic_features(event_row.get("text_hash"), linguistic_features, ling_cols)

        # OIS baseline inputs use the *post-event* curve of meeting N
        # at the 1-month tenor (the closest free tenor to the ~6-week
        # inter-meeting window). The implied rate is what the market
        # prices for the next meeting given everything just announced
        # at N. The base rate is the target rate meeting N just set
        # (``ff_target_after``). Both are known the moment meeting N's
        # decision is public -- same information cutoff as the model.
        implied_rate = _curve_value_at(mp_row.get("post_event_curve"), 1)
        ff_after_current = mp_row.get("ff_target_after")
        if isinstance(ff_after_current, float) and math.isnan(ff_after_current):
            ff_after_current = None

        out_rows.append(
            MeetingRow(
                target_event_date=next_event_date,
                target_as_of_ts=target_as_of,
                target_class=target_class,
                feature_event_date=feature_event_date,
                ois=ois_vec,
                text=text_vec,
                linguistic=ling_vec,
                credibility=cred_vec,
                macro=macro_vec,
                ois_baseline_implied_rate=implied_rate,
                ois_baseline_base_rate=(float(ff_after_current) if ff_after_current is not None else None),
                feature_names=feature_names,
            )
        )

    out_rows.sort(key=lambda r: r.target_event_date)
    summary["rows_emitted"] = len(out_rows)
    return out_rows, summary


# ---------------------------------------------------------------------------
# Walk-forward cross-validation
# ---------------------------------------------------------------------------


def _build_feature_matrix(
    rows: Sequence[MeetingRow], families: Sequence[str]
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Stack the requested family vectors row-wise; returns ``(X, col_names)``."""

    col_names: list[str] = []
    for fam in families:
        # Use the first row's feature names; every row uses the same vector lengths.
        if not rows:
            continue
        col_names.extend(rows[0].feature_names.get(fam, ()))
    if not rows:
        return np.zeros((0, 0), dtype=np.float64), tuple(col_names)
    matrix = np.zeros((len(rows), len(col_names)), dtype=np.float64)
    for i, row in enumerate(rows):
        cursor = 0
        for fam in families:
            vec = getattr(row, fam)
            n = len(vec)
            matrix[i, cursor : cursor + n] = vec
            cursor += n
    return matrix, tuple(col_names)


def _encode_targets(rows: Sequence[MeetingRow]) -> np.ndarray:
    cls_to_idx = {cls: i for i, cls in enumerate(ORDINAL_CLASSES)}
    return np.asarray([cls_to_idx[r.target_class] for r in rows], dtype=np.int64)


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


@dataclass
class FoldPrediction:
    """One held-out meeting's per-model probability vector."""

    target_event_date: str
    target_as_of_ts: str
    target_class: str
    n_train_rows: int
    probabilities: dict[str, dict[str, float]] = field(default_factory=dict)


def walk_forward_predict(
    rows: Sequence[MeetingRow],
    *,
    families: Sequence[str] = FEATURE_FAMILIES,
    ordinal_backend: str | None = None,
    include_gbt: bool = True,
    include_ois_baseline: bool = True,
    include_naive: bool = True,
    random_state: int = 11,
) -> list[FoldPrediction]:
    """Leave-one-meeting-out walk-forward CV.

    For each row ``i`` in ``rows`` (sorted by target_event_date), the
    train set is ``rows[: i]`` (strictly earlier meetings). When the
    train set has fewer than the number of classes -- which can happen
    early in the series -- we fall back to the OIS / naive baselines
    only and skip the parametric model.
    """

    X_all, col_names = _build_feature_matrix(rows, families)
    y_all = _encode_targets(rows)

    preds: list[FoldPrediction] = []
    for i, row in enumerate(rows):
        X_train = X_all[:i]
        y_train = y_all[:i]
        X_test = X_all[i : i + 1]
        # No-look-ahead assertion: every train row's target_event_date is
        # strictly older than the held-out meeting's.
        for j in range(i):
            assert rows[j].target_event_date < row.target_event_date, (
                f"walk-forward leak: row {j} target={rows[j].target_event_date} "
                f"not strictly before held-out target={row.target_event_date}"
            )

        per_model: dict[str, dict[str, float]] = {}

        if include_ois_baseline:
            per_model["ois_baseline"] = ois_baseline_probability(
                implied_rate=row.ois_baseline_implied_rate,
                base_rate=row.ois_baseline_base_rate,
            )
        if include_naive:
            per_model["naive_carry"] = naive_carry_probability()

        if i >= len(ORDINAL_CLASSES) and X_train.shape[1] > 0:
            X_tr_s, X_te_s = _standardise(X_train, X_test)
            try:
                handle = _dispatch_ordinal(
                    classes=ORDINAL_CLASSES, prefer=ordinal_backend, alpha=1.0
                )
                handle.fit_fn(X_tr_s, y_train)
                proba = handle.predict_proba_fn(X_te_s)
                proba = _align_proba_to_classes(proba, y_train, ORDINAL_CLASSES)
                per_model["ordinal_logit"] = _vec_to_class_dict(proba[0])
            except Exception as exc:  # noqa: BLE001 -- log + fall through
                LOGGER.warning("ordinal fit failed at row %d: %s", i, exc)

            if include_gbt:
                try:
                    from sklearn.ensemble import HistGradientBoostingClassifier

                    gbt = HistGradientBoostingClassifier(
                        random_state=random_state, max_iter=200
                    )
                    gbt.fit(X_tr_s, y_train)
                    proba = gbt.predict_proba(X_te_s)
                    proba = _align_proba_to_classes(proba, y_train, ORDINAL_CLASSES, classes_attr=gbt.classes_)
                    per_model["hist_gbt"] = _vec_to_class_dict(proba[0])
                except Exception as exc:  # noqa: BLE001
                    LOGGER.warning("HistGradientBoostingClassifier failed at row %d: %s", i, exc)

        preds.append(
            FoldPrediction(
                target_event_date=row.target_event_date.isoformat(),
                target_as_of_ts=row.target_as_of_ts.isoformat(),
                target_class=row.target_class,
                n_train_rows=i,
                probabilities=per_model,
            )
        )

    return preds


def _align_proba_to_classes(
    proba: np.ndarray,
    y_train: np.ndarray,
    full_classes: Sequence[str],
    *,
    classes_attr: np.ndarray | None = None,
) -> np.ndarray:
    """Pad probability columns to match :data:`ORDINAL_CLASSES`.

    A train fold may be missing some classes entirely (e.g. no
    ``hike_75`` ever in pre-2022 data). The model's columns then cover
    only the observed subset; we widen the matrix and fill the missing
    columns with zeros, then renormalise.
    """

    K = len(full_classes)
    if proba.shape[1] == K and classes_attr is None:
        return proba
    if classes_attr is None:
        observed = sorted({int(c) for c in y_train.tolist()})
    else:
        observed = [int(c) for c in classes_attr]
    out = np.zeros((proba.shape[0], K), dtype=np.float64)
    for col_idx, cls_idx in enumerate(observed):
        if 0 <= int(cls_idx) < K:
            out[:, int(cls_idx)] = proba[:, col_idx]
    # Renormalise; if a row has zero mass (degenerate fit), fall back to uniform.
    row_sum = out.sum(axis=1, keepdims=True)
    safe = np.where(row_sum < 1e-12, 1.0, row_sum)
    out = out / safe
    zero_rows = (row_sum < 1e-12).flatten()
    if zero_rows.any():
        out[zero_rows] = 1.0 / K
    return out


def _vec_to_class_dict(vec: np.ndarray) -> dict[str, float]:
    return {cls: round(float(vec[i]), 6) for i, cls in enumerate(ORDINAL_CLASSES)}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _truth_one_hot(target_class: str) -> np.ndarray:
    out = np.zeros(len(ORDINAL_CLASSES), dtype=np.float64)
    out[ORDINAL_CLASSES.index(target_class)] = 1.0
    return out


def _model_predictions(
    preds: Sequence[FoldPrediction], model_name: str
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Stack ``(y_true_onehot, y_pred_proba)`` for one model.

    Returns ``(y_true_onehot, y_pred_proba, truth_class_names)``. Rows
    without a prediction for that model are skipped.
    """

    truths: list[np.ndarray] = []
    probs: list[np.ndarray] = []
    truth_class_names: list[str] = []
    for pred in preds:
        proba = pred.probabilities.get(model_name)
        if proba is None:
            continue
        truths.append(_truth_one_hot(pred.target_class))
        probs.append(np.asarray([proba[cls] for cls in ORDINAL_CLASSES], dtype=np.float64))
        truth_class_names.append(pred.target_class)
    if not truths:
        return np.zeros((0, len(ORDINAL_CLASSES))), np.zeros((0, len(ORDINAL_CLASSES))), []
    return np.vstack(truths), np.vstack(probs), truth_class_names


def compute_metrics(
    preds: Sequence[FoldPrediction], model_name: str
) -> dict[str, Any]:
    """Brier (multi-class), log-loss, top-1 accuracy, macro-F1, confusion matrix."""

    truth_onehot, proba, truth_classes = _model_predictions(preds, model_name)
    n = truth_onehot.shape[0]
    if n == 0:
        return {
            "n": 0,
            "brier": None,
            "log_loss": None,
            "top1_accuracy": None,
            "macro_f1": None,
            "confusion_matrix": {},
        }
    brier = float(((proba - truth_onehot) ** 2).sum(axis=1).mean())
    eps = 1e-15
    log_loss = float(-(truth_onehot * np.log(np.clip(proba, eps, 1.0))).sum(axis=1).mean())
    pred_class_idx = np.argmax(proba, axis=1)
    truth_idx = np.argmax(truth_onehot, axis=1)
    top1 = float((pred_class_idx == truth_idx).mean())
    macro_f1 = _macro_f1(truth_idx, pred_class_idx, n_classes=len(ORDINAL_CLASSES))
    cm = _confusion_matrix(truth_classes, [ORDINAL_CLASSES[i] for i in pred_class_idx])
    return {
        "n": int(n),
        "brier": round(brier, 6),
        "log_loss": round(log_loss, 6),
        "top1_accuracy": round(top1, 6),
        "macro_f1": round(macro_f1, 6),
        "confusion_matrix": cm,
    }


def _macro_f1(truth: np.ndarray, pred: np.ndarray, *, n_classes: int) -> float:
    f1s: list[float] = []
    for k in range(n_classes):
        tp = float(((pred == k) & (truth == k)).sum())
        fp = float(((pred == k) & (truth != k)).sum())
        fn = float(((pred != k) & (truth == k)).sum())
        if tp + fp == 0 or tp + fn == 0:
            continue
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        if precision + recall == 0:
            f1s.append(0.0)
        else:
            f1s.append(2 * precision * recall / (precision + recall))
    if not f1s:
        return 0.0
    return float(sum(f1s) / len(f1s))


def _confusion_matrix(
    truth_classes: Sequence[str], pred_classes: Sequence[str]
) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {
        t: dict.fromkeys(ORDINAL_CLASSES, 0) for t in ORDINAL_CLASSES
    }
    for t, p in zip(truth_classes, pred_classes):
        if t not in out or p not in ORDINAL_CLASSES:
            continue
        out[t][p] += 1
    return out


def filter_predictions_excluding_window(
    preds: Sequence[FoldPrediction], *, start: _dt.date, end: _dt.date
) -> list[FoldPrediction]:
    """Drop predictions whose target falls inside ``[start, end]``."""

    keep: list[FoldPrediction] = []
    for pred in preds:
        try:
            d = _dt.date.fromisoformat(pred.target_event_date)
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

    rows: list[MeetingRow]
    predictions: list[FoldPrediction]
    metrics: dict[str, Any]
    summary: dict[str, Any]
    ablation_metrics: dict[str, dict[str, Any]]


def run(
    *,
    events: pd.DataFrame,
    mp_surprises: pd.DataFrame,
    linguistic_features: pd.DataFrame,
    macro_state: pd.DataFrame,
    output_dir: Path | None = None,
    ordinal_backend: str | None = None,
    random_state: int = 11,
) -> RunArtifacts:
    """End-to-end run: assemble, walk-forward predict, score, optionally write."""

    rows, summary = build_supervised_rows(
        events=events,
        mp_surprises=mp_surprises,
        linguistic_features=linguistic_features,
        macro_state=macro_state,
    )

    preds = walk_forward_predict(
        rows,
        families=FEATURE_FAMILIES,
        ordinal_backend=ordinal_backend,
        random_state=random_state,
    )
    model_names = sorted({m for p in preds for m in p.probabilities})
    metrics_full: dict[str, Any] = {
        model: compute_metrics(preds, model) for model in model_names
    }
    pandemic_excl = filter_predictions_excluding_window(
        preds, start=PANDEMIC_START, end=PANDEMIC_END
    )
    metrics_ex_pandemic: dict[str, Any] = {
        model: compute_metrics(pandemic_excl, model) for model in model_names
    }

    metrics: dict[str, Any] = {
        "full_window": metrics_full,
        "ex_pandemic_window": metrics_ex_pandemic,
        "pandemic_window": {
            "start": PANDEMIC_START.isoformat(),
            "end": PANDEMIC_END.isoformat(),
        },
        "ordinal_backend": _detect_backend(ordinal_backend),
        "ois_baseline_sigma_bp": OIS_BASELINE_SIGMA_BP,
        "model_names": model_names,
        "n_predictions": len(preds),
    }

    # Per-feature-family ablations on the parametric ordinal_logit model only.
    ablation_metrics = _run_ablations(
        rows,
        ordinal_backend=ordinal_backend,
        random_state=random_state,
    )

    if output_dir is not None:
        write_artifacts(
            output_dir=Path(output_dir),
            predictions=preds,
            metrics=metrics,
            summary=summary,
            ablations=ablation_metrics,
        )

    return RunArtifacts(
        rows=rows,
        predictions=preds,
        metrics=metrics,
        summary=summary,
        ablation_metrics=ablation_metrics,
    )


def _detect_backend(prefer: str | None) -> str:
    try:
        handle = _dispatch_ordinal(classes=ORDINAL_CLASSES, prefer=prefer)
        return handle.backend
    except Exception:  # noqa: BLE001
        return "unknown"


def _run_ablations(
    rows: Sequence[MeetingRow],
    *,
    ordinal_backend: str | None,
    random_state: int,
) -> dict[str, dict[str, Any]]:
    """Per-feature-family ablation table on the ordinal_logit model.

    Each subset re-runs walk-forward CV with only the families in the
    subset. Reports the same metric bundle as ``compute_metrics``.
    Returned keyed by subset name.
    """

    out: dict[str, dict[str, Any]] = {}
    for name, fams in ABLATION_SUBSETS:
        preds = walk_forward_predict(
            rows,
            families=fams,
            ordinal_backend=ordinal_backend,
            include_gbt=False,
            random_state=random_state,
        )
        out[name] = {
            "families": list(fams),
            "n_features": _count_features(rows, fams),
            "metrics": compute_metrics(preds, "ordinal_logit"),
        }
    # Always include the OIS-baseline metric (model-free) so the table
    # has a fixed reference column.
    base_preds = walk_forward_predict(
        rows,
        families=("ois",),
        include_gbt=False,
        random_state=random_state,
    )
    out["ois_baseline_only"] = {
        "families": ["ois_baseline"],
        "n_features": 0,
        "metrics": compute_metrics(base_preds, "ois_baseline"),
    }
    out["naive_carry_only"] = {
        "families": ["naive_carry"],
        "n_features": 0,
        "metrics": compute_metrics(base_preds, "naive_carry"),
    }
    return out


def _count_features(rows: Sequence[MeetingRow], families: Sequence[str]) -> int:
    if not rows:
        return 0
    return sum(len(rows[0].feature_names.get(fam, ())) for fam in families)


# ---------------------------------------------------------------------------
# Artifact writers
# ---------------------------------------------------------------------------


def write_artifacts(
    *,
    output_dir: Path,
    predictions: Sequence[FoldPrediction],
    metrics: Mapping[str, Any],
    summary: Mapping[str, Any],
    ablations: Mapping[str, Any],
) -> None:
    """Persist results.json, metrics.json, feature_attribution.md."""

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(
        json.dumps(
            {
                "predictions": [
                    {
                        "target_event_date": p.target_event_date,
                        "target_as_of_ts": p.target_as_of_ts,
                        "target_class": p.target_class,
                        "n_train_rows": p.n_train_rows,
                        "probabilities": p.probabilities,
                    }
                    for p in predictions
                ],
                "summary": dict(summary),
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


def _format_attribution_md(ablations: Mapping[str, Any]) -> str:
    lines = [
        "# Next-FOMC decision -- feature-family attribution",
        "",
        "Each row reports leave-one-meeting-out walk-forward metrics on the",
        "ordinal_logit model trained on the listed feature subset. The",
        "baselines (no learned model) sit at the bottom for reference.",
        "",
        "| Subset | Families | #features | n | Brier | LogLoss | Top1Acc | MacroF1 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, payload in ablations.items():
        m = payload.get("metrics", {})
        lines.append(
            "| {name} | {fams} | {nf} | {n} | {brier} | {ll} | {top1} | {f1} |".format(
                name=name,
                fams=", ".join(payload.get("families", [])),
                nf=payload.get("n_features", 0),
                n=m.get("n"),
                brier=m.get("brier"),
                ll=m.get("log_loss"),
                top1=m.get("top1_accuracy"),
                f1=m.get("macro_f1"),
            )
        )
    lines.append("")
    lines.append(
        "Note: the OIS-only and naive-carry rows are model-free baselines."
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
            "Predict the next FOMC rate decision using text + macro + OIS + "
            "credibility + linguistic features (Phase 8 #147)."
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
        help="Override for the artifact output dir. Default: data/artifacts/next_fomc/.",
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
        "--ordinal-backend",
        default=None,
        choices=(None, "statsmodels", "mord", "numpy"),
        help="Force a specific ordinal backend. Default: auto-detect.",
    )
    parser.add_argument("--seed", type=int, default=11)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    data_dir = Path(args.data_dir)
    package_dir = data_dir / "processed" / args.training_package_id
    fred_dir = data_dir / "external" / "fred"

    events = _load_parquet_safely(package_dir / args.events_name)
    mp = _load_parquet_safely(fred_dir / args.mp_surprises_name)
    ling_path = package_dir / args.linguistic_name
    linguistic = _load_parquet_safely(ling_path) if ling_path.exists() else pd.DataFrame()
    macro_path = fred_dir / args.macro_state_name
    macro = _load_parquet_safely(macro_path) if macro_path.exists() else pd.DataFrame()

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else data_dir / "artifacts" / "next_fomc"
    )

    artifacts = run(
        events=events,
        mp_surprises=mp,
        linguistic_features=linguistic,
        macro_state=macro,
        output_dir=output_dir,
        ordinal_backend=args.ordinal_backend,
        random_state=args.seed,
    )

    print(f"[next-fomc] rows emitted: {artifacts.summary['rows_emitted']}")
    print(f"[next-fomc] ordinal backend: {artifacts.metrics['ordinal_backend']}")
    print(f"[next-fomc] models: {', '.join(artifacts.metrics['model_names'])}")
    print(f"[next-fomc] output dir: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
