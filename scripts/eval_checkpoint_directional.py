"""Post-hoc directional evaluation of a saved forecaster checkpoint.

Loads one ``forecaster_best.pt`` and runs inference across every fold's
test partition, deriving the directional view from the continuous
predictions. Reports per-fold ``direction_accuracy``, ``f1_macro``,
``direction_auc`` against the dataset's majority-class baseline.

Single-checkpoint scope is intentional: the sweep harness persists
only the best trial across the entire sweep, so per-(architecture,
seed, fold) breakdown is not available without retraining. This
script answers "does the BEST sweep model carry directional signal
on the held-out test partitions?" -- the empirical gate that decides
whether the Phase 9 classification head + 4-layer NLP investment is
warranted.

Usage::

    python -m scripts.eval_checkpoint_directional \\
        --training-package-id tp_v2_sprint1_2026_05_15_sentiment_market_core_v1.0_epv1_v1.0 \\
        --checkpoint /app/models/forecaster_best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from app.config import DATA_DIR
from app.evaluation.directional_metrics import compute_directional_metrics
from app.models.config import (
    FEATURE_SIZE,
    RICH_FEATURE_SIZE,
    RichFeatureScalerParams,
)
from app.models.factory import build_forecaster
from app.training.checkpoint import _coerce_payload_config
from app.training.loaders import (
    apply_rich_feature_scaler_tensor,
    load_walk_forward_split,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--training-package-id",
        required=True,
        help="Training-package id under ``data/processed/<id>``.",
    )
    parser.add_argument(
        "--checkpoint",
        default="/app/models/forecaster_best.pt",
        help="Path to the saved forecaster checkpoint (.pt).",
    )
    parser.add_argument(
        "--folds",
        nargs="+",
        default=("wf_fold_1", "wf_fold_2", "wf_fold_3", "wf_fold_4"),
        help="Fold ids to evaluate against (one per --folds value).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Explicit device override (e.g. cuda or cpu). Auto-detects when omitted.",
    )
    parser.add_argument(
        "--baseline-accuracy",
        type=float,
        default=0.537,
        help=(
            "Majority-class baseline for the directional target on the "
            "current package. Default 0.537 = 2204 / 4099 +1 rows in "
            "events.parquet after excluding 4 zero-direction rows."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON output path. Prints to stdout regardless.",
    )
    return parser.parse_args()


def _resolve_device(arg: str | None) -> torch.device:
    if arg is not None:
        return torch.device(arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"checkpoint not found: {path}")
    payload = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(payload, dict):
        raise SystemExit(f"checkpoint at {path} is not a dict payload")
    required = ("model_state_dict", "model_config")
    missing = [k for k in required if k not in payload]
    if missing:
        raise SystemExit(f"checkpoint missing required keys: {missing}")
    return payload


def _build_model_from_payload(
    payload: dict[str, Any],
    device: torch.device,
) -> torch.nn.Module:
    config = _coerce_payload_config(payload)
    model = build_forecaster(config).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model


def _build_partition_tensor(
    sequences: Any,
    *,
    rich_payload_expected: bool,
    close_scale: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert a list-of-windows partition into (x, y, prev_close) tensors.

    ``y`` and ``prev_close`` are emitted in the SAME scaled-close
    units the model was trained against (``close / close_scale``).
    ``close_scale`` comes from the checkpoint payload; the eval would
    otherwise compute ``sign(unscaled_true_close - scaled_prev_close)``
    which is positive on every row and silently produces a
    100%-accuracy artefact instead of a real directional read.

    Returns tensors on the target device, ready for one forward pass.
    """

    if not sequences:
        return (
            torch.empty(0, dtype=torch.float32, device=device),
            torch.empty(0, dtype=torch.float32, device=device),
            torch.empty(0, dtype=torch.float32, device=device),
        )

    rows_x: list[list[list[float]]] = []
    rows_y: list[tuple[float, float]] = []
    rows_prev: list[float] = []

    for window in sequences:
        if len(window) < 2:
            continue
        target_vector = window[-1]
        history = window[:-1]
        row_x = [
            v.as_rich_list(close_scale=close_scale)
            if rich_payload_expected
            else v.as_list(close_scale=close_scale)
            for v in history
        ]
        rows_x.append(row_x)
        # Apply the same scaling to the target close so the directional
        # comparison stays in a single unit system.
        rows_y.append(
            (
                float(target_vector.market_close) / float(close_scale),
                float(target_vector.market_volatility),
            )
        )
        # Last bar of the input sequence's close, already scaled by
        # ``as_list`` / ``as_rich_list``. Position [1] is the close
        # slot in both layouts.
        last_bar_close_scaled = row_x[-1][1]
        rows_prev.append(last_bar_close_scaled)

    if not rows_x:
        return (
            torch.empty(0, dtype=torch.float32, device=device),
            torch.empty(0, dtype=torch.float32, device=device),
            torch.empty(0, dtype=torch.float32, device=device),
        )

    x = torch.tensor(rows_x, dtype=torch.float32, device=device)
    y = torch.tensor(rows_y, dtype=torch.float32, device=device)
    prev = torch.tensor(rows_prev, dtype=torch.float32, device=device)
    return x, y, prev


def _run_inference(
    model: torch.nn.Module,
    x: torch.Tensor,
    rich_scaler: RichFeatureScalerParams | None,
) -> torch.Tensor:
    x_in = apply_rich_feature_scaler_tensor(x, rich_scaler) if rich_scaler else x
    with torch.no_grad():
        predictions = model(x_in)
    return predictions


def _evaluate_fold(
    *,
    fold_id: str,
    package_id: str,
    rich_payload_expected: bool,
    close_scale: float,
    model: torch.nn.Module,
    rich_scaler: RichFeatureScalerParams | None,
    device: torch.device,
) -> dict[str, Any]:
    split = load_walk_forward_split(
        package_id,
        fold_id=fold_id,
        rich_features=rich_payload_expected,
    )
    test_sequences = split.test if split.test else []

    x, y, prev = _build_partition_tensor(
        test_sequences,
        rich_payload_expected=rich_payload_expected,
        close_scale=close_scale,
        device=device,
    )

    if x.numel() == 0:
        return {
            "fold_id": fold_id,
            "n_events": 0,
            "skipped": "empty test partition",
        }

    predictions = _run_inference(model, x, rich_scaler)
    pred_close = predictions[:, 0]
    true_close = y[:, 0]
    metrics = compute_directional_metrics(pred_close, true_close, prev)
    return {
        "fold_id": fold_id,
        "n_events": int(x.shape[0]),
        **metrics,
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args()
    device = _resolve_device(args.device)
    checkpoint_path = Path(args.checkpoint)
    payload = _load_checkpoint(checkpoint_path, device)
    model = _build_model_from_payload(payload, device)

    rich_payload_expected = int(
        payload["model_config"].get("input_size", FEATURE_SIZE)
    ) == RICH_FEATURE_SIZE

    rich_scaler = RichFeatureScalerParams.from_dict(
        payload.get("rich_feature_scaler")
    )
    close_scale = float(payload.get("close_scale", 10000.0))

    print("==== checkpoint context ====")
    print(f"  path:           {checkpoint_path}")
    print(f"  device:         {device}")
    print(f"  input_size:     {payload['model_config'].get('input_size')}")
    print(f"  architecture:   {payload['model_config'].get('architecture')}")
    print(f"  rich_features:  {rich_payload_expected}")
    print(f"  close_scale:    {close_scale}")
    print(f"  has scaler:     {rich_scaler is not None}")
    print(f"  baseline acc:   {args.baseline_accuracy:.3f} (majority class)")
    print()

    per_fold: list[dict[str, Any]] = []
    for fold_id in args.folds:
        result = _evaluate_fold(
            fold_id=fold_id,
            package_id=args.training_package_id,
            rich_payload_expected=rich_payload_expected,
            close_scale=close_scale,
            model=model,
            rich_scaler=rich_scaler,
            device=device,
        )
        per_fold.append(result)

    print(
        f"{'fold':<14}{'n':>6}{'accuracy':>12}{'f1_macro':>12}{'auc':>10}{'vs baseline':>14}"
    )
    print("-" * 70)
    accs: list[float] = []
    for row in per_fold:
        if "skipped" in row:
            print(f"  {row['fold_id']:<12}{row['n_events']:>6}  {row['skipped']}")
            continue
        acc = row.get("direction_accuracy")
        f1 = row.get("f1_macro")
        auc = row.get("direction_auc")
        acc_str = f"{acc:.3f}" if isinstance(acc, float) else "None"
        f1_str = f"{f1:.3f}" if isinstance(f1, float) else "None"
        auc_str = f"{auc:.3f}" if isinstance(auc, float) else "None"
        delta = (
            f"{(acc - args.baseline_accuracy):+.3f}"
            if isinstance(acc, float)
            else "None"
        )
        print(
            f"  {row['fold_id']:<12}"
            f"{row['n_events']:>6}"
            f"{acc_str:>12}"
            f"{f1_str:>12}"
            f"{auc_str:>10}"
            f"{delta:>14}"
        )
        if isinstance(acc, float):
            accs.append(acc)

    if accs:
        mean_acc = float(np.mean(accs))
        median_acc = float(np.median(accs))
        print()
        print(
            f"  aggregate    mean acc = {mean_acc:.3f}   "
            f"median acc = {median_acc:.3f}   "
            f"baseline = {args.baseline_accuracy:.3f}"
        )

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(per_fold, indent=2))
        print(f"\n  json written to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
