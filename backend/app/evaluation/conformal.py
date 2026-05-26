from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

# Default nominal coverage 0.80 (alpha=0.20) matches the regression-head
# default. Picked so the prediction set is informative on a 3-class
# target — at alpha=0.05 the set frequently covers all three classes
# on borderline rows, which is correct but uninformative for a decision
# support surface.
DEFAULT_CLASSIFICATION_ALPHA = 0.2


@dataclass(frozen=True)
class ConformalManifest:
    """Split-conformal quantiles for a single forecaster checkpoint.

    `residual_quantile_close` is the (1 - alpha) quantile of `|y - y_hat|`
    measured on the calibration fold for the close head; the volatility head
    uses its own quantile. `nominal_coverage` is `1 - alpha`. Apply at
    inference time as `[y_hat - q, y_hat + q]` for symmetric two-sided bands.

    For classification-mode checkpoints, ``softmax_quantile`` carries the
    APS threshold (Romano et al. 2020) fitted on the same calibration
    partition using ``1 - softmax[y_true]`` as the non-conformity score.
    Pre-#216 manifests without the field load with ``softmax_quantile=None``
    and the inference path falls back to uncalibrated max-softmax confidence.

    #292 extension: ``rates_residual_quantiles`` maps the per-head short
    name (``2y`` / ``5y`` / ``terminal``) to the (1 - alpha) absolute-
    residual quantile in **raw bps** -- the inference path applies
    ``[y_hat - q, y_hat + q]`` as the conformal bps band. Per-head aux
    classification surfaces use ``rates_softmax_quantiles`` which maps
    the same short name to the APS threshold for the per-head
    (easing / neutral / tightening) classifier. Both default to empty
    dicts so pre-#292 manifests round-trip clean.
    """

    alpha: float
    nominal_coverage: float
    residual_quantile_close: float
    residual_quantile_volatility: float
    calibration_n: int
    notes: str | None = None
    softmax_quantile: float | None = None
    rates_residual_quantiles: dict[str, float] | None = None
    rates_softmax_quantiles: dict[str, float] | None = None
    # #326 conditional-coverage diagnostics. Both fields are computed on
    # the same calibration partition the ``softmax_quantile`` was fitted
    # on so they stay paired with the manifest's nominal coverage claim.
    # ``class_conditional_coverage`` maps the class label (string,
    # matches the active checkpoint's ``stance_classes`` / regime label
    # tuple) to empirical coverage = fraction of rows in that class
    # whose true label is inside the APS prediction set. A class whose
    # row count is zero on the calibration fold maps to ``nan``.
    # ``set_size_distribution`` maps the integer set size (``1``, ``2``,
    # ``3`` for the 3-class regime target) to the fraction of rows
    # emitting that set size, summing to ``1.0`` within finite-sample
    # rounding. Pre-#326 manifests round-trip with both fields ``None``.
    # On regression-canonical checkpoints the same fields carry the
    # bucketed-regression interpretation -- the calibrator bins the
    # predicted log_rv via ``bucket_log_rv`` and reports the same
    # diagnostics under the regression band's coverage surface; the
    # manifest's ``notes`` field surfaces the dual interpretation so a
    # downstream reader can tell which calibration produced the
    # numbers.
    class_conditional_coverage: dict[str, float] | None = None
    set_size_distribution: dict[int, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "alpha": self.alpha,
            "nominal_coverage": self.nominal_coverage,
            "residual_quantile_close": self.residual_quantile_close,
            "residual_quantile_volatility": self.residual_quantile_volatility,
            "calibration_n": self.calibration_n,
            "notes": self.notes,
            "softmax_quantile": self.softmax_quantile,
            "rates_residual_quantiles": (
                dict(self.rates_residual_quantiles)
                if self.rates_residual_quantiles
                else None
            ),
            "rates_softmax_quantiles": (
                dict(self.rates_softmax_quantiles)
                if self.rates_softmax_quantiles
                else None
            ),
            "class_conditional_coverage": (
                {str(k): float(v) for k, v in self.class_conditional_coverage.items()}
                if self.class_conditional_coverage
                else None
            ),
            "set_size_distribution": (
                # JSON object keys must be strings; the loader reverses
                # this to int on read.
                {str(int(k)): float(v) for k, v in self.set_size_distribution.items()}
                if self.set_size_distribution
                else None
            ),
        }


def split_conformal_quantile(residuals: Sequence[float], alpha: float) -> float:
    """Empirical (1 - alpha) quantile with the finite-sample correction.

    Source: Lei & Wasserman (2014), "Distribution-Free Prediction Bands". The
    correction multiplies (1 - alpha) by (n + 1) / n so the band carries the
    desired coverage on hold-out points. Residuals must be non-negative
    absolute errors.
    """

    cleaned = sorted(float(abs(r)) for r in residuals if math.isfinite(float(r)))
    n = len(cleaned)
    if n == 0:
        raise ValueError("Calibration residual set is empty.")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must lie in (0, 1); got {alpha!r}.")
    rank = math.ceil((1.0 - alpha) * (n + 1))
    rank = max(1, min(n, rank))
    return cleaned[rank - 1]


def calibrate_split_conformal(
    *,
    close_predictions: Sequence[float],
    close_actuals: Sequence[float],
    volatility_predictions: Sequence[float],
    volatility_actuals: Sequence[float],
    alpha: float = 0.2,
    notes: str | None = None,
) -> ConformalManifest:
    if len(close_predictions) != len(close_actuals):
        raise ValueError("close_predictions and close_actuals must align in length.")
    if len(volatility_predictions) != len(volatility_actuals):
        raise ValueError("volatility_predictions and volatility_actuals must align in length.")
    close_resid = [
        actual - pred for pred, actual in zip(close_predictions, close_actuals)
    ]
    vol_resid = [
        actual - pred for pred, actual in zip(volatility_predictions, volatility_actuals)
    ]
    return ConformalManifest(
        alpha=float(alpha),
        nominal_coverage=1.0 - float(alpha),
        residual_quantile_close=split_conformal_quantile(close_resid, alpha),
        residual_quantile_volatility=split_conformal_quantile(vol_resid, alpha),
        calibration_n=len(close_predictions),
        notes=notes,
    )


def apply_conformal_bands(
    *,
    close_predictions: Sequence[float],
    volatility_predictions: Sequence[float],
    manifest: ConformalManifest,
    horizon_scale: bool = True,
) -> tuple[list[float], list[float], list[float], list[float]]:
    """Return (close_lower, close_upper, vol_lower, vol_upper) using the
    manifest's residual quantiles. `horizon_scale=True` widens the band by
    sqrt(step) so multi-step forecasts inherit the usual variance scaling.

    Note: the marginal (1 - alpha) coverage guarantee from split-conformal
    holds only for step 1. With ``horizon_scale=True`` the multi-step bands
    are a random-walk heuristic, not a calibrated conformal interval — treat
    `manifest.nominal_coverage` as a single-step quantity. Pass
    ``horizon_scale=False`` if you need uniform width across the horizon.
    """

    close_lower: list[float] = []
    close_upper: list[float] = []
    vol_lower: list[float] = []
    vol_upper: list[float] = []
    for step_idx, (pred_close, pred_vol) in enumerate(
        zip(close_predictions, volatility_predictions), start=1
    ):
        scale = math.sqrt(step_idx) if horizon_scale else 1.0
        close_w = manifest.residual_quantile_close * scale
        vol_w = manifest.residual_quantile_volatility * scale
        close_lower.append(min(max(0.0, pred_close - close_w), pred_close))
        close_upper.append(pred_close + close_w)
        vol_lower.append(min(max(0.0, pred_vol - vol_w), pred_vol))
        vol_upper.append(pred_vol + vol_w)
    return close_lower, close_upper, vol_lower, vol_upper


def calibrate_rates_regression_conformal(
    *,
    predictions_bps: Sequence[float],
    actuals_bps: Sequence[float],
    alpha: float = 0.2,
) -> float:
    """Fit the (1 - alpha) absolute-residual quantile for one rates head.

    Inputs are paired predictions / observations in **raw bps**; the
    helper returns the conformal band half-width the inference path
    applies symmetrically as ``[y_hat - q, y_hat + q]``. Same Lei-
    Wasserman correction the close / volatility regression helper uses;
    rows with non-finite values are dropped silently after the length
    check.
    """

    if len(predictions_bps) != len(actuals_bps):
        raise ValueError(
            f"predictions_bps ({len(predictions_bps)}) and actuals_bps "
            f"({len(actuals_bps)}) must align in length."
        )
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must lie in (0, 1); got {alpha!r}.")
    residuals: list[float] = []
    for pred, actual in zip(predictions_bps, actuals_bps):
        if pred is None or actual is None:
            continue
        try:
            pf = float(pred)
            af = float(actual)
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(pf) and math.isfinite(af)):
            continue
        residuals.append(af - pf)
    if not residuals:
        raise ValueError("Calibration residual set is empty after filtering.")
    return split_conformal_quantile(residuals, alpha)


def calibrate_classification_conformal(
    *,
    softmax_scores: Sequence[Sequence[float]],
    true_classes: Sequence[int],
    alpha: float = DEFAULT_CLASSIFICATION_ALPHA,
) -> float:
    """Fit the APS threshold (Romano et al. 2020) on a calibration partition.

    The non-conformity score for row i is ``1 - softmax[i, true_classes[i]]``
    — high score when the model is uncertain about the truth, low when it
    is confident on the right class. The threshold is the (1 - alpha)
    finite-sample-corrected quantile of those scores via the same Lei-
    Wasserman rank formula the regression helper uses.

    Inputs must align: ``len(softmax_scores) == len(true_classes)``. Rows
    whose softmax does not include the true class index (e.g. truncated /
    malformed) are dropped silently after a length sanity check.
    """

    if len(softmax_scores) != len(true_classes):
        raise ValueError(
            f"softmax_scores ({len(softmax_scores)}) and true_classes "
            f"({len(true_classes)}) must align in length."
        )
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must lie in (0, 1); got {alpha!r}.")
    nonconformity: list[float] = []
    for row, true_idx in zip(softmax_scores, true_classes):
        idx = int(true_idx)
        if idx < 0 or idx >= len(row):
            continue
        prob = float(row[idx])
        if not math.isfinite(prob):
            continue
        nonconformity.append(1.0 - prob)
    if not nonconformity:
        raise ValueError("Calibration softmax set is empty after filtering.")
    return split_conformal_quantile(nonconformity, alpha)


def predict_conformal_set(
    softmax_probs: Sequence[float],
    threshold: float,
) -> list[int]:
    """Build the APS prediction set for one row's softmax distribution.

    Includes every class ``j`` whose ``1 - softmax[j] <= threshold``,
    i.e. ``softmax[j] >= 1 - threshold``. When no class clears the
    threshold (pathological row), falls back to ``[argmax]`` rather
    than emitting an empty set — the empty-set case is mathematically
    valid under APS but useless as a decision-support surface, and
    the fallback keeps the marginal coverage guarantee asymptotically
    valid because the row contributes a singleton instead of zero.
    """

    if not softmax_probs:
        return []
    keep = float(1.0 - threshold)
    included = [i for i, p in enumerate(softmax_probs) if float(p) >= keep]
    if included:
        return included
    argmax_idx = max(range(len(softmax_probs)), key=lambda i: float(softmax_probs[i]))
    return [argmax_idx]


def empirical_classification_coverage(
    predicted_sets: Sequence[Sequence[int]],
    true_classes: Sequence[int],
) -> float:
    """Fraction of rows where ``true_classes[i] in predicted_sets[i]``."""

    if len(predicted_sets) != len(true_classes):
        raise ValueError(
            f"predicted_sets ({len(predicted_sets)}) and true_classes "
            f"({len(true_classes)}) must align in length."
        )
    if not predicted_sets:
        return float("nan")
    inside = sum(
        1 for s, y in zip(predicted_sets, true_classes) if int(y) in {int(x) for x in s}
    )
    return inside / len(predicted_sets)


def compute_class_conditional_coverage(
    predicted_sets: Sequence[Sequence[int]],
    true_classes: Sequence[int],
    class_names: Sequence[str],
) -> dict[str, float]:
    """Per-class empirical coverage on the calibration partition (#326).

    APS marginally covers ``1 - alpha`` of rows under exchangeability;
    that guarantee is silent on the per-class slice. A model whose
    softmax is systematically miscalibrated on one class (the #326
    canonical example: ``normal`` at ~7 % recall on the 3-class
    vol-regime head) can emit prediction sets that exclude that class
    almost every row while marginal coverage still reads at nominal.
    The helper measures the per-class slice directly: for each class
    label, divide ``#{rows whose true class is in the set}`` by
    ``#{rows whose true class is that label}``.

    ``class_names`` is the active checkpoint's label tuple (regime
    label tuple on the vol-regime head; stance label tuple on the
    stance head). Classes with zero rows on the calibration partition
    map to ``float('nan')`` so the downstream gap-flag helper can
    distinguish "empty slice" from "degenerate coverage". Returns a
    fresh dict keyed by the string label so the manifest round-trip
    stays serialisable.
    """

    if len(predicted_sets) != len(true_classes):
        raise ValueError(
            f"predicted_sets ({len(predicted_sets)}) and true_classes "
            f"({len(true_classes)}) must align in length."
        )
    if not class_names:
        raise ValueError("class_names must carry at least one label.")
    coverage: dict[str, float] = {}
    for class_idx, label in enumerate(class_names):
        row_total = 0
        inside = 0
        for predicted_set, true_class in zip(predicted_sets, true_classes):
            if int(true_class) != class_idx:
                continue
            row_total += 1
            if int(true_class) in {int(x) for x in predicted_set}:
                inside += 1
        coverage[str(label)] = (
            float(inside) / float(row_total) if row_total > 0 else float("nan")
        )
    return coverage


def compute_set_size_distribution(
    predicted_sets: Sequence[Sequence[int]],
    *,
    n_classes: int = 3,
) -> dict[int, float]:
    """Fraction of rows emitting each prediction-set size (#326).

    The 3-class APS surface admits sizes ``{1, 2, 3}``; the helper
    returns a dict keyed by every size in ``[1, n_classes]`` so the
    distribution sums to 1.0 even when one or more sizes never appear
    (their entry is ``0.0`` rather than missing). Pathological empty
    sets (which ``predict_conformal_set`` already rewrites to the
    argmax singleton) would land in the ``0`` bucket, which is
    intentionally absent from the contract -- if a future surface
    emits true empty sets, this helper will raise rather than silently
    hide them.

    Returns a fresh dict keyed by ``int``. ``n_classes`` defaults to
    3 to match the active vol-regime / stance heads; callers on a
    different cardinality target pass the right number.
    """

    if n_classes < 1:
        raise ValueError(f"n_classes must be >= 1; got {n_classes!r}.")
    total = len(predicted_sets)
    if total == 0:
        # NaN per bucket so an empty partition does not fabricate a
        # 0.0 mass for every size -- the caller knows the calibration
        # partition was empty and needs to surface that on the
        # manifest.
        return {k: float("nan") for k in range(1, n_classes + 1)}
    counts: Counter[int] = Counter()
    for predicted_set in predicted_sets:
        size = len(predicted_set)
        if size <= 0:
            raise ValueError(
                "predicted_sets must not carry empty sets; APS contract "
                "rewrites empties to {argmax} (see predict_conformal_set)."
            )
        if size > n_classes:
            raise ValueError(
                f"predicted set size {size} exceeds n_classes={n_classes}."
            )
        counts[size] += 1
    return {k: float(counts.get(k, 0)) / float(total) for k in range(1, n_classes + 1)}


def class_conditional_gap_flag(
    coverage_dict: Mapping[str, float],
    *,
    nominal: float = 0.80,
    tolerance: float = 0.10,
) -> list[str]:
    """Return the class names whose conditional coverage falls > tolerance below nominal.

    A class with ``coverage < (nominal - tolerance)`` lands on the
    flag list. NaN coverage (empty class slice on the calibration
    partition) is skipped silently -- absence of evidence is not
    evidence of a degenerate coverage gap, and the gap diagnostic
    should not fire purely on a small-sample fold.

    Defaults track the #326 issue contract: nominal 0.80, tolerance
    0.10 (so 0.70 is the gap threshold). Class names returned in the
    order ``coverage_dict`` iterates so a deterministic dict ordering
    upstream produces a deterministic flag list.
    """

    if not (0.0 <= nominal <= 1.0):
        raise ValueError(f"nominal must lie in [0, 1]; got {nominal!r}.")
    if tolerance < 0.0:
        raise ValueError(f"tolerance must be >= 0; got {tolerance!r}.")
    # Float epsilon: ``0.80 - 0.10`` lands at ``0.7000000000000001`` so
    # a class with coverage exactly ``0.70`` would otherwise flag under
    # ``cov < threshold``. The issue contract is "> tolerance below"
    # nominal — a class whose gap is exactly ``tolerance`` is on the
    # boundary and must NOT flag. Compare the gap directly instead of
    # the subtracted threshold so the boundary case is clean.
    flagged: list[str] = []
    for label, coverage in coverage_dict.items():
        cov = float(coverage)
        if not math.isfinite(cov):
            continue
        gap = float(nominal) - cov
        if gap > tolerance + 1e-12:
            flagged.append(str(label))
    return flagged


def compute_regression_band_class_coverage(  # noqa: C901 — single-pass validation + bucketing, intentionally branchy
    *,
    log_rv_predictions: Sequence[float],
    log_rv_actuals: Sequence[float],
    residual_quantile: float,
    raw_vol_cutoffs: tuple[float, ...],
    class_names: Sequence[str] = ("calm", "normal", "high"),
) -> dict[str, float]:
    """Per-class coverage of a regression-canonical conformal band (#326).

    On the regression-canonical surface (ADR 0015 / #322) there is no
    softmax to threshold; the calibrated object is a band
    ``[y_hat - q, y_hat + q]`` in log-vol space. This helper carries
    the dual interpretation issue #326 asks for: bucket the true
    log_rv via the active checkpoint's tertile cutoffs and report,
    per class, the fraction of rows whose true value sits inside the
    regression band.

    A class with zero rows on the calibration partition maps to
    ``float('nan')`` so ``class_conditional_gap_flag`` can skip it.
    The helper is intentionally pure -- it imports nothing from the
    bucketing service to keep the unit tests trivially mockable; the
    caller is responsible for passing the live ``raw_vol_cutoffs``
    tuple off the active ``ModelConfig.vol_regime_quantiles``.
    """

    if len(log_rv_predictions) != len(log_rv_actuals):
        raise ValueError(
            f"log_rv_predictions ({len(log_rv_predictions)}) and "
            f"log_rv_actuals ({len(log_rv_actuals)}) must align in length."
        )
    if len(raw_vol_cutoffs) != 2:
        raise ValueError(
            f"raw_vol_cutoffs must carry exactly two cutoffs; "
            f"got {len(raw_vol_cutoffs)}."
        )
    cutoff_low, cutoff_high = (float(c) for c in raw_vol_cutoffs)
    if cutoff_low <= 0.0 or cutoff_high <= 0.0 or cutoff_low > cutoff_high:
        raise ValueError(
            f"raw_vol_cutoffs must be positive and ordered; got {raw_vol_cutoffs!r}."
        )
    q = float(residual_quantile)
    log_cutoff_low = math.log(cutoff_low)
    log_cutoff_high = math.log(cutoff_high)

    def _bucket(raw_value: float) -> str | None:
        if not math.isfinite(raw_value):
            return None
        if raw_value < log_cutoff_low:
            return class_names[0] if len(class_names) >= 1 else None
        if raw_value < log_cutoff_high:
            return class_names[1] if len(class_names) >= 2 else None
        return class_names[2] if len(class_names) >= 3 else None

    rows_per_class: Counter[str] = Counter()
    inside_per_class: Counter[str] = Counter()
    for pred, actual in zip(log_rv_predictions, log_rv_actuals):
        try:
            p = float(pred)
            a = float(actual)
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(p) and math.isfinite(a)):
            continue
        bucket = _bucket(a)
        if bucket is None:
            continue
        rows_per_class[bucket] += 1
        if (p - q) <= a <= (p + q):
            inside_per_class[bucket] += 1
    coverage: dict[str, float] = {}
    for label in class_names:
        total = rows_per_class.get(str(label), 0)
        if total == 0:
            coverage[str(label)] = float("nan")
        else:
            coverage[str(label)] = float(inside_per_class.get(str(label), 0)) / float(total)
    return coverage


def format_class_set_label(
    predicted_set: Sequence[int],
    class_names: Sequence[str],
) -> str:
    """Emit ``"{normal, high}"``-style label for the UI card.

    Renders class indices through ``class_names`` and wraps in braces.
    Empty input → ``"{}"``. Unknown indices fall through as ``"?"`` so
    a stale manifest still produces a readable string rather than
    raising in the response serializer.
    """

    labels = [
        str(class_names[i]) if 0 <= int(i) < len(class_names) else "?"
        for i in predicted_set
    ]
    return "{" + ", ".join(labels) + "}"


def empirical_coverage(
    *,
    predictions: Sequence[float],
    actuals: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
) -> float:
    if not (len(predictions) == len(actuals) == len(lower) == len(upper)):
        raise ValueError("predictions, actuals, lower, upper must align in length.")
    if not predictions:
        return float("nan")
    inside = sum(
        1
        for actual, lo, hi in zip(actuals, lower, upper)
        if math.isfinite(actual) and lo <= actual <= hi
    )
    return inside / len(predictions)


def load_manifest(path: Path | str) -> ConformalManifest:  # noqa: C901 — JSON deserialiser with many optional back-compat fields
    """Read a JSON manifest. Residual quantile fields default to 0.0
    when absent (classification-only manifests written by
    ``save_manifest`` drop them); the inference loader treats a 0.0
    residual quantile as "no regression bands available" and falls
    back to gaussian-z.
    """

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Conformal manifest not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Conformal manifest must be a JSON object: {path}")
    softmax_quantile_raw = payload.get("softmax_quantile")
    rates_residuals_raw = payload.get("rates_residual_quantiles")
    rates_softmax_raw = payload.get("rates_softmax_quantiles")
    rates_residuals: dict[str, float] | None = None
    rates_softmax: dict[str, float] | None = None
    if isinstance(rates_residuals_raw, Mapping):
        rates_residuals = {
            str(k): float(v)
            for k, v in rates_residuals_raw.items()
            if v is not None
        }
        if not rates_residuals:
            rates_residuals = None
    if isinstance(rates_softmax_raw, Mapping):
        rates_softmax = {
            str(k): float(v)
            for k, v in rates_softmax_raw.items()
            if v is not None
        }
        if not rates_softmax:
            rates_softmax = None
    # #326 conditional diagnostics. Pre-#326 manifests on disk simply
    # lack both keys; the loader resolves them to ``None`` so the
    # ConformalManifest constructor keeps its existing default and the
    # back-compat contract holds.
    class_cond_raw = payload.get("class_conditional_coverage")
    set_size_raw = payload.get("set_size_distribution")
    class_cond: dict[str, float] | None = None
    set_size: dict[int, float] | None = None
    if isinstance(class_cond_raw, Mapping):
        class_cond = {
            str(k): float(v)
            for k, v in class_cond_raw.items()
            if v is not None
        }
        if not class_cond:
            class_cond = None
    if isinstance(set_size_raw, Mapping):
        # JSON object keys are strings on disk; cast back to int. A
        # malformed key (non-integer) is dropped silently so a stale
        # manifest does not crash the inference loader.
        parsed: dict[int, float] = {}
        for k, v in set_size_raw.items():
            if v is None:
                continue
            try:
                parsed[int(k)] = float(v)
            except (TypeError, ValueError):
                continue
        if parsed:
            set_size = parsed
    return ConformalManifest(
        alpha=float(payload["alpha"]),
        nominal_coverage=float(payload["nominal_coverage"]),
        residual_quantile_close=float(payload.get("residual_quantile_close", 0.0)),
        residual_quantile_volatility=float(
            payload.get("residual_quantile_volatility", 0.0)
        ),
        calibration_n=int(payload["calibration_n"]),
        notes=payload.get("notes"),
        softmax_quantile=(
            float(softmax_quantile_raw) if softmax_quantile_raw is not None else None
        ),
        rates_residual_quantiles=rates_residuals,
        rates_softmax_quantiles=rates_softmax,
        class_conditional_coverage=class_cond,
        set_size_distribution=set_size,
    )


def save_manifest(manifest: ConformalManifest, path: Path | str) -> Path:
    """Persist a manifest atomically via temp file + ``Path.replace``.

    The temp-and-rename pattern means a mid-write process crash leaves
    the original sidecar intact rather than producing a half-written
    JSON the inference loader would later fail on. Same destination
    path on success; the temp file is unlinked even on failure.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = manifest.to_dict()
    # Drop residual_quantile_* fields entirely when both are zero so a
    # classification-only manifest is not mistaken for a regression
    # band manifest at inference time (the inference loader treats
    # any non-None manifest as conformal, so leaving the zeros in
    # would produce zero-width prediction bands).
    if (
        payload.get("residual_quantile_close") == 0.0
        and payload.get("residual_quantile_volatility") == 0.0
    ):
        payload.pop("residual_quantile_close", None)
        payload.pop("residual_quantile_volatility", None)
    payload = {k: v for k, v in payload.items() if v is not None}
    tmp_path = path.with_name(path.name + ".tmp")
    try:
        tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp_path.replace(path)
    except Exception:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise
    return path


def to_jsonable(manifest: ConformalManifest) -> dict[str, float | int | str | None]:
    return asdict(manifest)


def bootstrap_ci_columns(
    rows: Iterable[Mapping[str, Any]],
    *,
    sample_key: str = "samples",
    block_size: int = 6,
    n_resamples: int = 1000,
    coverage: float = 0.95,
    seed: int = 11,
) -> list[dict[str, float | int | str | None]]:
    """Augment aggregator rows with `ci_lo` / `ci_hi` columns derived from a
    moving-block bootstrap.  Each row must carry the raw `samples` list so the
    bootstrap can resample; rows without samples fall through with `None` CIs.
    """

    from app.evaluation.bootstrap import block_bootstrap_ci

    out: list[dict[str, float | int | str | None]] = []
    for row in rows:
        result: dict[str, float | int | str | None] = {k: v for k, v in row.items() if k != sample_key}
        samples = row.get(sample_key)
        if isinstance(samples, Sequence) and len(samples) > 1:
            ci = block_bootstrap_ci(
                list(samples),
                block_size=block_size,
                n_resamples=n_resamples,
                coverage=coverage,
                seed=seed,
            )
            result["ci_lo"] = float(ci.lo)
            result["ci_hi"] = float(ci.hi)
        else:
            result["ci_lo"] = None
            result["ci_hi"] = None
        out.append(result)
    return out
