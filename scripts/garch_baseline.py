"""GARCH(1,1) classical-finance reference baseline.

Fits a GARCH(1,1) on SPX daily log-returns up to each walk-forward
fold's ``train_end`` and forecasts 10-day forward conditional volatility
for the val + test events. The forecast is binned to the same 3-class
regime grid the LSTM forecaster reports against — using quantile cutoffs
fit on the train slice's realised ``forward_realized_vol_10d``. The
output is the pooled-fold macro-F1 with a moving-block bootstrap CI,
matching the regime_pooled_aggregator headline so the GARCH row drops
straight into §6.6's comparison table.

GARCH(1,1) is the canonical no-text non-DL baseline cited in Hansen-Lunde
(2005) and Engle-Rangel (2008); the row is required for the thesis
writeup to anchor "what the classical signal achieves before deep
learning is layered on top".
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from app.config import DATA_DIR
from app.evaluation.bootstrap import BootstrapCI
from app.evaluation.classification_breakdown import (
    ClassificationBreakdown,
    compute_classification_breakdown,
)


DEFAULT_SPX_PATH = DATA_DIR / "external" / "fred" / "_spx_gspc.parquet"
DEFAULT_OUTPUT_ROOT = DATA_DIR / "artifacts" / "garch_baseline"
TARGET_COLUMN = "forward_realized_vol_10d"
HORIZON_BARS = 10


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GARCH(1,1) regime-classification baseline.")
    parser.add_argument("--training-package-id", required=True)
    parser.add_argument("--spx-path", default=str(DEFAULT_SPX_PATH))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--min-event-date",
        default="2010-01-01",
        help=(
            "Drop events whose date predates this cutoff. The SPX history "
            "parquet starts at 2009-12-18, so anything earlier has no daily "
            "returns to fit GARCH against."
        ),
    )
    parser.add_argument("--block-size", type=int, default=20)
    parser.add_argument("--n-resamples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=11)
    return parser.parse_args()


def _load_events(package_dir: Path, *, min_event_date: str) -> pd.DataFrame:
    events = pd.read_parquet(package_dir / "events.parquet")
    if TARGET_COLUMN not in events.columns:
        raise ValueError(
            f"events.parquet at {package_dir} is missing the {TARGET_COLUMN!r} column"
        )
    events = events[events["event_date"] >= min_event_date].copy()
    events = events.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)
    return events


def _load_returns(spx_path: Path) -> pd.Series:
    bars = pd.read_parquet(spx_path).sort_values("date").reset_index(drop=True)
    bars["date"] = pd.to_datetime(bars["date"]).dt.strftime("%Y-%m-%d")
    log_returns = np.log(bars["close"]).diff().dropna()
    log_returns.index = bars["date"].iloc[1:].tolist()
    return log_returns


def _load_folds(package_dir: Path) -> list[dict]:
    manifest_path = package_dir / "fold_manifest_expanding_walk_forward.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    folds = payload.get("folds") if isinstance(payload, dict) else None
    if not isinstance(folds, list) or not folds:
        raise ValueError(f"no folds found in {manifest_path}")
    return folds


def _quantile_cutoffs(values: Iterable[float]) -> tuple[float, float]:
    arr = np.asarray(list(values), dtype=float)
    q33, q67 = float(np.quantile(arr, 1.0 / 3.0)), float(np.quantile(arr, 2.0 / 3.0))
    return q33, q67


def _bin_to_regime(value: float, *, q33: float, q67: float) -> int:
    if value <= q33:
        return 0
    if value <= q67:
        return 1
    return 2


def _forecast_one(model_result, *, horizon: int) -> float:
    """Mean 1-day variance over the next ``horizon`` steps, square-rooted to
    daily-equivalent vol so the result is on the same scale as
    ``forward_realized_vol_10d`` (which is std of daily log-returns over the
    forward window)."""

    forecast = model_result.forecast(horizon=horizon, reindex=False)
    variance_row = forecast.variance.iloc[-1].to_numpy()
    if variance_row.size == 0:
        return float("nan")
    return float(math.sqrt(float(np.mean(variance_row))))


def _bootstrap_macro_f1(
    pooled_preds: list[int],
    pooled_targets: list[int],
    *,
    block_size: int,
    n_resamples: int,
    seed: int,
) -> BootstrapCI:
    n = len(pooled_preds)
    if n == 0:
        return BootstrapCI(
            point=float("nan"), lo=float("nan"), hi=float("nan"),
            coverage=0.95, n_resamples=n_resamples, block_size=block_size,
        )
    point = compute_classification_breakdown(
        predictions=pooled_preds, targets=pooled_targets, n_classes=3
    ).macro_f1
    rng = random.Random(seed)
    samples: list[float] = []
    n_blocks = max(1, (n + block_size - 1) // block_size)
    for _ in range(n_resamples):
        idx: list[int] = []
        for _ in range(n_blocks):
            start = rng.randint(0, max(0, n - block_size))
            idx.extend(range(start, min(n, start + block_size)))
        idx = idx[:n]
        b_preds = [pooled_preds[i] for i in idx]
        b_targets = [pooled_targets[i] for i in idx]
        samples.append(
            compute_classification_breakdown(
                predictions=b_preds, targets=b_targets, n_classes=3
            ).macro_f1
        )
    samples.sort()
    lo_idx = max(0, int(0.025 * n_resamples))
    hi_idx = min(n_resamples - 1, int(0.975 * n_resamples) - 1)
    return BootstrapCI(
        point=float(point),
        lo=float(samples[lo_idx]),
        hi=float(samples[hi_idx]),
        coverage=0.95,
        n_resamples=n_resamples,
        block_size=block_size,
    )


def _run_one_fold(
    *,
    fold: dict,
    events: pd.DataFrame,
    log_returns: pd.Series,
) -> tuple[list[int], list[int], dict]:
    from arch import arch_model  # local import keeps the cli importable without arch

    train_end = str(fold["train_end"])
    val_start = str(fold["val_start"])
    val_end = str(fold["val_end"])
    test_start = str(fold["test_start"])
    test_end = str(fold["test_end"])

    train_events = events[events["event_date"] <= train_end]
    val_events = events[(events["event_date"] >= val_start) & (events["event_date"] <= val_end)]
    test_events = events[(events["event_date"] >= test_start) & (events["event_date"] <= test_end)]

    if train_events.empty:
        raise ValueError(f"fold {fold['fold_id']!r} has zero train events")

    q33, q67 = _quantile_cutoffs(train_events[TARGET_COLUMN])

    fit_returns = log_returns[log_returns.index <= train_end]
    if fit_returns.empty:
        raise ValueError(
            f"fold {fold['fold_id']!r} has zero SPX returns up to train_end={train_end}"
        )
    # ``arch`` works on percent returns to keep the optimiser well-scaled.
    fit_returns_pct = fit_returns.to_numpy() * 100.0
    model = arch_model(fit_returns_pct, mean="Zero", vol="Garch", p=1, q=1).fit(disp="off")
    forecast_vol = _forecast_one(model, horizon=HORIZON_BARS) / 100.0

    holdout = pd.concat([val_events, test_events], ignore_index=True)
    pred_labels = [
        _bin_to_regime(forecast_vol, q33=q33, q67=q67) for _ in range(len(holdout))
    ]
    true_labels = [
        _bin_to_regime(float(v), q33=q33, q67=q67)
        for v in holdout[TARGET_COLUMN].to_list()
    ]
    fold_meta = {
        "fold_id": fold["fold_id"],
        "train_end": train_end,
        "train_size_events": int(len(train_events)),
        "holdout_size_events": int(len(holdout)),
        "fit_returns_n": int(len(fit_returns)),
        "q33": q33,
        "q67": q67,
        "forecast_vol_daily_equiv": forecast_vol,
    }
    return pred_labels, true_labels, fold_meta


def main() -> int:
    args = _parse_args()
    package_dir = DATA_DIR / "processed" / args.training_package_id
    if not package_dir.exists():
        raise SystemExit(f"training package not found at {package_dir}")

    events = _load_events(package_dir, min_event_date=args.min_event_date)
    log_returns = _load_returns(Path(args.spx_path))
    folds = _load_folds(package_dir)

    pooled_preds: list[int] = []
    pooled_targets: list[int] = []
    per_fold: list[dict] = []
    for fold in folds:
        preds, targets, meta = _run_one_fold(
            fold=fold, events=events, log_returns=log_returns
        )
        pooled_preds.extend(preds)
        pooled_targets.extend(targets)
        fold_breakdown = compute_classification_breakdown(
            predictions=preds, targets=targets, n_classes=3
        )
        meta["macro_f1"] = float(fold_breakdown.macro_f1)
        meta["n_pooled"] = int(len(preds))
        per_fold.append(meta)
        print(
            f"[garch] {meta['fold_id']}: macro_f1={meta['macro_f1']:.4f} "
            f"n={meta['n_pooled']} q33={meta['q33']:.4f} q67={meta['q67']:.4f}"
        )

    macro_ci = _bootstrap_macro_f1(
        pooled_preds,
        pooled_targets,
        block_size=int(args.block_size),
        n_resamples=int(args.n_resamples),
        seed=int(args.bootstrap_seed),
    )
    overall = compute_classification_breakdown(
        predictions=pooled_preds, targets=pooled_targets, n_classes=3
    )

    output_dir = Path(args.output_root) / args.training_package_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output = {
        "training_package_id": args.training_package_id,
        "horizon_bars": HORIZON_BARS,
        "min_event_date": args.min_event_date,
        "n_pooled": int(len(pooled_preds)),
        "macro_f1": float(overall.macro_f1),
        "macro_f1_ci": {
            "point": macro_ci.point,
            "lo": macro_ci.lo,
            "hi": macro_ci.hi,
            "coverage": macro_ci.coverage,
            "n_resamples": macro_ci.n_resamples,
            "block_size": macro_ci.block_size,
        },
        "per_fold": per_fold,
        "breakdown": overall.to_dict(),
    }
    out_path = output_dir / "garch_pooled_test_macro_f1.json"
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\n[garch] pooled macro_f1={overall.macro_f1:.4f} "
          f"[{macro_ci.lo:.3f}, {macro_ci.hi:.3f}] n={len(pooled_preds)}")
    print(f"[garch] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
