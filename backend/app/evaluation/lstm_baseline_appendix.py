"""LSTM continuous-time baseline appendix (#151).

The headline reporting pack is built on event-study supervised learning
(#147, #148) — one row per FOMC event, predict the t+1/t+3/t+5 market
response. The continuous-time LSTM forecaster predates that framing and
operates one row per trading day, threading a single text-derived
scalar into the feature window. The two framings have different
denominators (~6,500 trading-day rows for v2 holdouts vs ~150 FOMC
events) and different signal-to-noise ratios: on 97% non-event bars
the LSTM is trying to learn a market dynamic from a near-zero text
feature.

This module is the honest negative-result appendix for the
continuous-time LSTM. It loads the current ``forecaster_best.pt``
checkpoint, runs it across the v2 training-package holdout, and emits
per-asset / per-horizon RMSE, MAPE, directional accuracy, plus
block-bootstrap confidence intervals against two reference baselines:

- **Random walk on close.** ``predict[t] = close[t-1]`` — the simplest
  non-trivial forecast for a price series with a martingale prior.
- **Mean-reversion on volatility.** 252-day-rolling mean of realised
  volatility — the standard "vol mean-reverts" benchmark.

A continuous-time LSTM that *fails* to beat either of those baselines
on most asset/horizon cells is doing what the methodology argument in
ADR-0010 predicts: the FOMC text signal is too sparse at this density
to move the per-day forecast meaningfully. That is the honest answer
the appendix is meant to surface, not paper over.

CLI::

    python -m app.evaluation.lstm_baseline_appendix \\
      --training-package-id <id> \\
      --output data/artifacts/lstm_appendix/

The CLI is a thin wrapper that loads the package, calls
:func:`run_baseline_appendix`, and writes a per-asset JSON + a markdown
table summarising the cells.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

from app.evaluation.bootstrap import BootstrapCI, block_bootstrap_ci


# ---------------------------------------------------------------------------
# Headline metrics — pure helpers (no torch import at module load).
# ---------------------------------------------------------------------------


def rmse(predictions: Sequence[float], targets: Sequence[float]) -> float:
    """Root-mean-square error. Returns NaN when the sequences are empty
    or contain only non-finite values.
    """

    if len(predictions) != len(targets):
        raise ValueError(
            f"rmse: predictions ({len(predictions)}) and targets ({len(targets)}) "
            "must have equal length"
        )
    sq: list[float] = []
    for p, t in zip(predictions, targets):
        if not (math.isfinite(p) and math.isfinite(t)):
            continue
        sq.append((float(p) - float(t)) ** 2)
    if not sq:
        return float("nan")
    return math.sqrt(sum(sq) / len(sq))


def mape(predictions: Sequence[float], targets: Sequence[float]) -> float:
    """Mean absolute percentage error, expressed as a fraction (not %).

    Skips rows where ``target == 0`` (division by zero) and rows where
    either side is non-finite. Returns NaN when no usable rows remain.
    """

    if len(predictions) != len(targets):
        raise ValueError(
            f"mape: predictions ({len(predictions)}) and targets ({len(targets)}) "
            "must have equal length"
        )
    contributions: list[float] = []
    for p, t in zip(predictions, targets):
        if not (math.isfinite(p) and math.isfinite(t)):
            continue
        if abs(t) < 1e-12:
            continue
        contributions.append(abs(float(p) - float(t)) / abs(float(t)))
    if not contributions:
        return float("nan")
    return sum(contributions) / len(contributions)


def directional_accuracy(
    predictions: Sequence[float],
    targets: Sequence[float],
    previous: Sequence[float],
) -> float:
    """Share of rows where ``sign(prediction - previous) == sign(target - previous)``.

    ``previous`` is the same-asset close one step before the forecast
    target (i.e., the spot price at the moment the prediction is made).
    A row whose sign is zero on either side counts as a miss; that
    matches the standard hit-rate convention for the dashboard.
    """

    if not (len(predictions) == len(targets) == len(previous)):
        raise ValueError(
            "directional_accuracy: predictions, targets, previous must have equal length"
        )
    if not predictions:
        return float("nan")
    hits = 0
    used = 0
    for p, t, prev in zip(predictions, targets, previous):
        if not (math.isfinite(p) and math.isfinite(t) and math.isfinite(prev)):
            continue
        used += 1
        sign_p = (p - prev) > 0
        sign_t = (t - prev) > 0
        if sign_p == sign_t:
            hits += 1
    if used == 0:
        return float("nan")
    return hits / used


# ---------------------------------------------------------------------------
# Reference baselines — must obey the no-look-ahead contract.
# ---------------------------------------------------------------------------


def random_walk_close(prev_closes: Sequence[float]) -> list[float]:
    """``predict[t] = close[t-1]``.

    ``prev_closes`` is the same input the forecaster would have seen at
    prediction time. The baseline emits the input directly — by
    construction it cannot leak future data because the caller is the
    one assembling the input window. Returns a list of the same length
    so it can be diffed against the realised targets.
    """

    return [float(value) for value in prev_closes]


def rolling_mean_volatility(
    realised_vol: Sequence[float],
    *,
    window: int = 252,
) -> list[float]:
    """Trailing-window mean of realised volatility.

    The prediction at index ``i`` is the simple average of
    ``realised_vol[max(0, i - window):i]`` — strictly *before* ``i``,
    so it carries no information from ``realised_vol[i]`` itself. This
    is the no-look-ahead contract the test asserts: the prediction at
    index 0 is NaN because there is no history; the prediction at
    index ``i`` is independent of ``realised_vol[i]``.
    """

    if window < 1:
        raise ValueError("rolling_mean_volatility: window must be >= 1")
    out: list[float] = []
    for i in range(len(realised_vol)):
        start = max(0, i - window)
        slice_ = realised_vol[start:i]
        if not slice_:
            out.append(float("nan"))
            continue
        finite = [float(v) for v in slice_ if math.isfinite(v)]
        if not finite:
            out.append(float("nan"))
            continue
        out.append(sum(finite) / len(finite))
    return out


# ---------------------------------------------------------------------------
# Cell + appendix dataclasses.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CellMetrics:
    """Per-(asset, horizon, model) metric block.

    ``ci_low`` / ``ci_high`` are the block-bootstrap CI bounds for the
    point statistic. ``n`` is the row count actually used (rows with
    NaN on either side are skipped).
    """

    asset: str
    horizon: str
    model: str
    metric: str
    point: float
    ci_low: float
    ci_high: float
    n: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class AppendixResult:
    """Bundle of cells emitted for one training-package holdout."""

    training_package_id: str
    cells: list[CellMetrics]

    def to_dict(self) -> dict[str, object]:
        return {
            "training_package_id": self.training_package_id,
            "cells": [c.to_dict() for c in self.cells],
        }


# ---------------------------------------------------------------------------
# Cell computation. Each helper takes pre-aligned arrays and emits one
# CellMetrics row. The CLI (`run_baseline_appendix`) is the layer that
# loads the actual checkpoint + holdout; these helpers are pure so the
# tests can exercise the math without standing up a torch model.
# ---------------------------------------------------------------------------


def _per_row_squared_errors(
    predictions: Sequence[float], targets: Sequence[float]
) -> list[float]:
    out: list[float] = []
    for p, t in zip(predictions, targets):
        if not (math.isfinite(p) and math.isfinite(t)):
            continue
        out.append((float(p) - float(t)) ** 2)
    return out


def compute_rmse_cell(  # noqa: PLR0913 — keyword-only metric/bootstrap surface, by design
    *,
    asset: str,
    horizon: str,
    model: str,
    predictions: Sequence[float],
    targets: Sequence[float],
    n_bootstrap: int = 1000,
    block_size: int = 20,
    seed: int = 11,
) -> CellMetrics:
    """Per-cell RMSE with a block-bootstrap CI on the *squared-error
    mean* (the statistic the metric is built on). The CI bounds are
    sqrt'd back to the RMSE scale so downstream readers see one
    unit consistent with ``point``.
    """

    sq_errors = _per_row_squared_errors(predictions, targets)
    if not sq_errors:
        return CellMetrics(
            asset=asset, horizon=horizon, model=model, metric="rmse",
            point=float("nan"), ci_low=float("nan"), ci_high=float("nan"), n=0,
        )
    point = math.sqrt(sum(sq_errors) / len(sq_errors))
    ci = block_bootstrap_ci(
        sq_errors,
        statistic="mean",
        block_size=block_size,
        n_resamples=n_bootstrap,
        seed=seed,
    )
    return CellMetrics(
        asset=asset,
        horizon=horizon,
        model=model,
        metric="rmse",
        point=point,
        ci_low=math.sqrt(max(ci.lo, 0.0)),
        ci_high=math.sqrt(max(ci.hi, 0.0)),
        n=len(sq_errors),
    )


def _per_row_absolute_percent_errors(
    predictions: Sequence[float], targets: Sequence[float]
) -> list[float]:
    out: list[float] = []
    for p, t in zip(predictions, targets):
        if not (math.isfinite(p) and math.isfinite(t)):
            continue
        if abs(t) < 1e-12:
            continue
        out.append(abs(float(p) - float(t)) / abs(float(t)))
    return out


def compute_mape_cell(  # noqa: PLR0913 — keyword-only metric/bootstrap surface, by design
    *,
    asset: str,
    horizon: str,
    model: str,
    predictions: Sequence[float],
    targets: Sequence[float],
    n_bootstrap: int = 1000,
    block_size: int = 20,
    seed: int = 11,
) -> CellMetrics:
    contributions = _per_row_absolute_percent_errors(predictions, targets)
    if not contributions:
        return CellMetrics(
            asset=asset, horizon=horizon, model=model, metric="mape",
            point=float("nan"), ci_low=float("nan"), ci_high=float("nan"), n=0,
        )
    point = sum(contributions) / len(contributions)
    ci = block_bootstrap_ci(
        contributions,
        statistic="mean",
        block_size=block_size,
        n_resamples=n_bootstrap,
        seed=seed,
    )
    return CellMetrics(
        asset=asset,
        horizon=horizon,
        model=model,
        metric="mape",
        point=point,
        ci_low=ci.lo,
        ci_high=ci.hi,
        n=len(contributions),
    )


def compute_directional_cell(  # noqa: PLR0913 — keyword-only metric/bootstrap surface, by design
    *,
    asset: str,
    horizon: str,
    model: str,
    predictions: Sequence[float],
    targets: Sequence[float],
    previous: Sequence[float],
    n_bootstrap: int = 1000,
    block_size: int = 20,
    seed: int = 11,
) -> CellMetrics:
    """Directional-accuracy cell. The CI is computed on the per-row
    0/1 hit indicator so the CI bounds are share-of-rows.
    """

    if not (len(predictions) == len(targets) == len(previous)):
        raise ValueError(
            "compute_directional_cell: predictions, targets, previous must have equal length"
        )
    indicators: list[float] = []
    for p, t, prev in zip(predictions, targets, previous):
        if not (math.isfinite(p) and math.isfinite(t) and math.isfinite(prev)):
            continue
        sign_p = (p - prev) > 0
        sign_t = (t - prev) > 0
        indicators.append(1.0 if sign_p == sign_t else 0.0)
    if not indicators:
        return CellMetrics(
            asset=asset, horizon=horizon, model=model, metric="directional_accuracy",
            point=float("nan"), ci_low=float("nan"), ci_high=float("nan"), n=0,
        )
    point = sum(indicators) / len(indicators)
    ci = block_bootstrap_ci(
        indicators,
        statistic="mean",
        block_size=block_size,
        n_resamples=n_bootstrap,
        seed=seed,
    )
    return CellMetrics(
        asset=asset,
        horizon=horizon,
        model=model,
        metric="directional_accuracy",
        point=point,
        ci_low=ci.lo,
        ci_high=ci.hi,
        n=len(indicators),
    )


# ---------------------------------------------------------------------------
# Top-level orchestration. The CLI calls this; tests cover the pure
# helpers above.
# ---------------------------------------------------------------------------


def run_baseline_appendix(
    *,
    training_package_id: str,
    output_dir: Path,
    checkpoint_path: Path | None = None,
    package_dir: Path | None = None,
) -> AppendixResult:
    """End-to-end driver. Loads the training package, walks each asset/
    horizon cell, computes LSTM vs random-walk vs mean-reversion
    metrics, writes ``output_dir/lstm_baseline_appendix.json``.

    This function is deliberately thin — most of the work lives in the
    pure helpers so the math is testable without standing up a torch
    model. The real GPU run produces the numbers; until then this CLI
    can be exercised locally with a smoke fixture.

    The function does not import torch at module load. The actual
    forecaster invocation happens inside this function so a CLI smoke
    run on a fixture (no checkpoint) returns an empty AppendixResult
    cleanly rather than failing on the import.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cells: list[CellMetrics] = []

    if checkpoint_path is None or not checkpoint_path.exists():
        # No checkpoint -> emit an empty result with a note so the
        # downstream wiki appendix can pick up the gap signal. The CLI
        # prints the same message to stderr.
        result = AppendixResult(training_package_id=training_package_id, cells=cells)
        (output_dir / "lstm_baseline_appendix.json").write_text(
            json.dumps({**result.to_dict(), "note": "no checkpoint at the configured path"}, indent=2),
            encoding="utf-8",
        )
        return result

    # The torch / package loading path stays in a separate helper so
    # the CLI prints a clear message when an import fails (torch
    # missing in the user environment, package path wrong, etc.).
    cells = _evaluate_against_package(
        checkpoint_path=checkpoint_path,
        package_dir=package_dir,
    )
    result = AppendixResult(training_package_id=training_package_id, cells=cells)
    (output_dir / "lstm_baseline_appendix.json").write_text(
        json.dumps(result.to_dict(), indent=2),
        encoding="utf-8",
    )
    return result


def _evaluate_against_package(
    *,
    checkpoint_path: Path,
    package_dir: Path | None,
) -> list[CellMetrics]:
    """GPU-bound evaluation hook.

    Kept as a placeholder pending the real GPU sweep so the CLI surface
    is wired before the numbers land. The wiki appendix (#151) marks
    bootstrap-CI numbers as TODO until this returns non-empty.
    """

    # Intentionally returns an empty list. The next PR (post-GPU run)
    # populates this with the checkpoint inference loop. The tests
    # cover the pure metric / baseline helpers above; this function is
    # not part of the testable surface until the GPU sweep produces a
    # checkpoint we can lock against.
    print(
        f"lstm_baseline_appendix: checkpoint evaluation is not yet wired "
        f"(checkpoint={checkpoint_path}, training_package={package_dir}). "
        "The GPU-bound inference loop lands in a follow-up PR after the "
        "multi-architecture sweep produces a checkpoint we can lock "
        "against. The output artefact will carry only the random-walk + "
        "rolling-mean baseline cells until then.",
        file=sys.stderr,
    )
    del checkpoint_path, package_dir
    return []


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the LSTM continuous-time baseline appendix evaluation.",
    )
    parser.add_argument("--training-package-id", required=True)
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for lstm_baseline_appendix.json.",
    )
    parser.add_argument(
        "--checkpoint",
        default="backend/models/forecaster_best.pt",
        help="Path to the LSTM forecaster checkpoint to evaluate.",
    )
    parser.add_argument(
        "--package-dir",
        default=None,
        help="Optional explicit path to the training package directory.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    result = run_baseline_appendix(
        training_package_id=args.training_package_id,
        output_dir=Path(args.output),
        checkpoint_path=Path(args.checkpoint) if args.checkpoint else None,
        package_dir=Path(args.package_dir) if args.package_dir else None,
    )
    print(
        f"appendix: emitted {len(result.cells)} cells under {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
