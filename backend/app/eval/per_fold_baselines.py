"""Per-fold class-distribution table with dual baselines (#500).

Reads a training package and computes two baselines per fold's test slice:
  majority_baseline_f1  -- predict train-partition mode for every test row
  stratified_random_f1  -- sample from train prior; mean over 1 000 seeds

Encoder macro-F1 is pulled from a sweep JSON when available.

CLI::

    python -m app.eval.per_fold_baselines \\
        --training-package-id canonical \\
        --sweep-artefact backend/artifacts/experiments/dual_head_comparison_canonical.json \\
        --output backend/artifacts/experiments/per_fold_baselines.json
"""

from __future__ import annotations

import argparse
import json
import random
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from app.config import BACKEND_ROOT, DATA_DIR as _DEFAULT_DATA_DIR
from app.evaluation.classification_breakdown import compute_classification_breakdown
from app.training.loaders import fit_vol_regime_quantiles, vol_regime_class_for

_PROCESSED_ROOT = _DEFAULT_DATA_DIR / "processed"
_N_CLASSES = 3
CLASS_NAMES: tuple[str, str, str] = ("calm", "normal", "high")
_DEFAULT_SWEEP = BACKEND_ROOT / "artifacts" / "experiments" / "dual_head_comparison_canonical.json"
_DEFAULT_OUTPUT = BACKEND_ROOT / "artifacts" / "experiments" / "per_fold_baselines.json"


def _pkg_dir(training_package_id: str, processed_root: Path) -> Path:
    if training_package_id == "canonical":
        return processed_root / "canonical"
    return processed_root / training_package_id


def load_fold_manifest(package_dir: Path) -> dict[str, Any]:
    path = package_dir / "fold_manifest_expanding_walk_forward.json"
    if not path.exists():
        raise FileNotFoundError(f"fold manifest not found: {path}")
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return payload


def load_events(package_dir: Path) -> list[dict[str, Any]]:
    import pandas as pd

    parquet_path = package_dir / "events.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"events.parquet not found: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    rows: list[dict[str, Any]] = df.to_dict("records")
    return rows


def _rows_in_range(rows: list[dict[str, Any]], start: str, end: str) -> list[dict[str, Any]]:
    return [r for r in rows if start <= str(r.get("event_date", "")) <= end]


def label_rows(
    rows: list[dict[str, Any]],
    quantiles: Sequence[float],
) -> list[int]:
    """Assign vol-regime class index to each row with a valid forward vol."""
    out: list[int] = []
    for r in rows:
        raw = r.get("forward_realized_vol_10d")
        if raw is None or raw != raw:  # noqa: PLR0124
            continue
        cls = vol_regime_class_for(float(raw), quantiles)
        if 0 <= cls < _N_CLASSES:
            out.append(cls)
    return out


def majority_baseline_f1(train_labels: list[int], test_labels: list[int]) -> float:
    """Macro-F1 under constant majority-class prediction."""
    if not train_labels or not test_labels:
        return 0.0
    counts = Counter(train_labels)
    majority = max(counts, key=lambda c: counts[c])
    bd = compute_classification_breakdown(
        predictions=[majority] * len(test_labels),
        targets=test_labels,
        n_classes=_N_CLASSES,
    )
    return float(bd.macro_f1)


def stratified_random_f1(
    train_labels: list[int],
    test_labels: list[int],
    *,
    n_seeds: int = 1000,
    base_seed: int = 0,
) -> float:
    """Mean macro-F1 of a random classifier matched to the train prior."""
    if not train_labels or not test_labels:
        return 0.0
    total = len(train_labels)
    counts = Counter(train_labels)
    priors = [counts.get(c, 0) / total for c in range(_N_CLASSES)]
    n_test = len(test_labels)
    f1_sum = 0.0
    for s in range(n_seeds):
        rng = random.Random(base_seed + s)
        preds = rng.choices(range(_N_CLASSES), weights=priors, k=n_test)
        bd = compute_classification_breakdown(
            predictions=preds, targets=test_labels, n_classes=_N_CLASSES
        )
        f1_sum += float(bd.macro_f1)
    return f1_sum / n_seeds


def _encoder_f1_from_sweep(
    sweep_path: Path,
    fold_id: str,
    head_mode: str,
) -> float | None:
    if not sweep_path.exists():
        return None
    try:
        payload = json.loads(sweep_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    mode_trials: list[Any] = (payload.get("trials") or {}).get(head_mode) or []
    vals: list[float] = []
    for trial in mode_trials:
        for fold in trial.get("folds") or []:
            if fold.get("fold_id") == fold_id:
                m = fold.get("metrics") or {}
                v = m.get("regime_f1_macro")
                if v is not None:
                    vals.append(float(v))
    return sum(vals) / len(vals) if vals else None


def compute_per_fold_baselines(
    training_package_id: str,
    *,
    processed_root: Path | None = None,
    sweep_artefact: Path | None = None,
    head_mode: str = "classification",
    n_stratified_seeds: int = 1000,
) -> dict[str, Any]:
    """Return per-fold distribution, majority-class, and stratified-random baselines.

    Each ``folds`` entry carries class_distribution (train/val/test),
    quantile_edges fitted on the train slice, majority_baseline_f1,
    stratified_random_f1, and encoder_f1 from the sweep artefact (None
    when unavailable).
    """
    root = processed_root or _PROCESSED_ROOT
    pkg_dir = _pkg_dir(training_package_id, root)
    manifest = load_fold_manifest(pkg_dir)
    events = load_events(pkg_dir)
    sweep = sweep_artefact or _DEFAULT_SWEEP

    folds_out: list[dict[str, Any]] = []
    for fold in manifest.get("folds") or []:
        fold_id = str(fold["fold_id"])
        train_rows = _rows_in_range(events, fold["train_start"], fold["train_end"])
        val_rows = _rows_in_range(events, fold["val_start"], fold["val_end"])
        test_rows = _rows_in_range(events, fold["test_start"], fold["test_end"])

        train_vols: list[float] = [
            float(r["forward_realized_vol_10d"])
            for r in train_rows
            if r.get("forward_realized_vol_10d") is not None
            and r["forward_realized_vol_10d"] == r["forward_realized_vol_10d"]
        ]
        quantiles = fit_vol_regime_quantiles(train_vols)
        if not quantiles:
            warnings.warn(
                f"[per_fold_baselines] {fold_id}: no vol data in train slice; skipping",
                stacklevel=2,
            )
            continue

        train_labels = label_rows(train_rows, quantiles)
        val_labels = label_rows(val_rows, quantiles)
        test_labels = label_rows(test_rows, quantiles)

        def _dist(labels: list[int]) -> dict[str, int]:
            c = Counter(labels)
            return {CLASS_NAMES[i]: c.get(i, 0) for i in range(_N_CLASSES)}

        folds_out.append(
            {
                "fold_id": fold_id,
                "class_distribution": {
                    "train": _dist(train_labels),
                    "val": _dist(val_labels),
                    "test": _dist(test_labels),
                },
                "quantile_edges": list(quantiles),
                "majority_baseline_f1": majority_baseline_f1(train_labels, test_labels),
                "stratified_random_f1": stratified_random_f1(
                    train_labels, test_labels, n_seeds=n_stratified_seeds
                ),
                "encoder_f1": _encoder_f1_from_sweep(sweep, fold_id, head_mode),
                "head_mode": head_mode,
            }
        )

    return {
        "training_package_id": training_package_id,
        "sweep_artefact": str(sweep),
        "head_mode": head_mode,
        "folds": folds_out,
    }


def print_summary(result: dict[str, Any]) -> None:
    folds = result.get("folds") or []
    if not folds:
        print("No fold results.")
        return
    hdr = f"{'Fold':<14} {'Majority':>10} {'Stratified':>12} {'Encoder':>10}"
    print(hdr)
    print("-" * len(hdr))
    for f in folds:
        enc = f["encoder_f1"]
        enc_s = f"{enc:.4f}" if enc is not None else "       n/a"
        print(
            f"{f['fold_id']:<14}"
            f"{f['majority_baseline_f1']:>10.4f}"
            f"{f['stratified_random_f1']:>12.4f}"
            f"{enc_s:>10}"
        )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Per-fold class-distribution + dual baselines table.")
    p.add_argument("--training-package-id", default="canonical")
    p.add_argument("--processed-root", default=str(_PROCESSED_ROOT))
    p.add_argument("--sweep-artefact", default=str(_DEFAULT_SWEEP))
    p.add_argument("--head-mode", default="classification")
    p.add_argument("--n-stratified-seeds", type=int, default=1000)
    p.add_argument("--output", default=str(_DEFAULT_OUTPUT))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    result = compute_per_fold_baselines(
        training_package_id=args.training_package_id,
        processed_root=Path(args.processed_root),
        sweep_artefact=Path(args.sweep_artefact),
        head_mode=args.head_mode,
        n_stratified_seeds=args.n_stratified_seeds,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    print_summary(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
