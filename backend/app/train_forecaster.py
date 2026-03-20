from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path
from typing import Any, Sequence

import torch

from app.services.forecaster import (
    BEST_MODEL_PATH,
    DEFAULT_BATCH_SIZE,
    DEFAULT_DATA_DIR,
    DEFAULT_DROPOUT,
    DEFAULT_EARLY_STOPPING_PATIENCE,
    DEFAULT_EPOCHS,
    DEFAULT_HEAD_HIDDEN_SIZE,
    DEFAULT_HIDDEN_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_NUM_LAYERS,
    DEFAULT_VALIDATION_SPLIT,
    ModelConfig,
    TrainingRunSummary,
    SEQUENCE_LENGTH,
    inspect_training_data_sources,
    train_model,
)


DEFAULT_REPORT_PATH = BEST_MODEL_PATH.parent / "forecaster_sweep_results.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the quantitative forecaster from prepared local datasets."
    )
    parser.add_argument(
        "--data-dir",
        default=str(DEFAULT_DATA_DIR),
        help="Directory containing JSON, JSONL, or CSV training datasets.",
    )
    parser.add_argument(
        "--checkpoint-path",
        default=str(BEST_MODEL_PATH),
        help="Where to save the best-performing checkpoint.",
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Maximum training epochs.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Mini-batch size for optimizer steps.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help="Optimizer learning rate.",
    )
    parser.add_argument(
        "--validation-split",
        type=float,
        default=DEFAULT_VALIDATION_SPLIT,
        help="Fraction of windows reserved for validation.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=DEFAULT_EARLY_STOPPING_PATIENCE,
        help="Number of non-improving epochs allowed before early stop.",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=DEFAULT_HIDDEN_SIZE,
        help="LSTM hidden size for a single training run.",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=DEFAULT_NUM_LAYERS,
        help="Number of stacked LSTM layers for a single training run.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=DEFAULT_DROPOUT,
        help="Dropout applied to the LSTM stack and forecast head.",
    )
    parser.add_argument(
        "--head-hidden-size",
        type=int,
        default=DEFAULT_HEAD_HIDDEN_SIZE,
        help="Hidden layer width for the projection head.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Explicit device override, e.g. 'cuda', 'cuda:0', or 'cpu'.",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run a grid search and select the best checkpoint by validation RMSE.",
    )
    parser.add_argument(
        "--hidden-sizes",
        nargs="+",
        type=int,
        help="Grid-search values for hidden size.",
    )
    parser.add_argument(
        "--num-layers-grid",
        nargs="+",
        type=int,
        help="Grid-search values for number of LSTM layers.",
    )
    parser.add_argument(
        "--dropouts",
        nargs="+",
        type=float,
        help="Grid-search values for dropout.",
    )
    parser.add_argument(
        "--learning-rates",
        nargs="+",
        type=float,
        help="Grid-search values for optimizer learning rate.",
    )
    parser.add_argument(
        "--epochs-grid",
        nargs="+",
        type=int,
        help="Grid-search values for epochs.",
    )
    parser.add_argument(
        "--report-path",
        default=str(DEFAULT_REPORT_PATH),
        help="Where to write the hyperparameter search report JSON.",
    )
    parser.add_argument(
        "--list-data",
        action="store_true",
        help="Only inspect discovered datasets and exit without training.",
    )
    return parser.parse_args()


def _build_model_config(args: argparse.Namespace) -> ModelConfig:
    return ModelConfig(
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        head_hidden_size=args.head_hidden_size,
    )


def _print_data_inventory(
    data_dir: Path,
    checkpoint_path: Path,
    device: torch.device,
    summaries: Sequence[Any],
    *,
    sequence_count: int,
    observation_count: int,
    window_count: int,
) -> None:
    print(f"Device: {device}")
    print(f"Data directory: {data_dir}")
    print(f"Checkpoint path: {checkpoint_path}")
    print(f"Existing checkpoint: {'yes' if checkpoint_path.exists() else 'no'}")
    print(f"Sequence groups discovered: {sequence_count}")
    print(f"Observations discovered: {observation_count}")
    print(f"Training windows available: {window_count}")
    if summaries:
        print("Discovered data sources:")
        for summary in summaries:
            relative_path = summary.path.relative_to(data_dir) if summary.path.is_relative_to(data_dir) else summary.path
            print(
                f"  - {relative_path} [{summary.status}] "
                f"groups={summary.record_groups}, records={summary.records}, "
                f"vectors={summary.vectors}, usable={summary.usable_sequences} :: {summary.message}"
            )
    else:
        print("Discovered data sources: none")


def _metrics_rank(summary: TrainingRunSummary) -> tuple[float, float]:
    if summary.metrics is None:
        return float("inf"), float("inf")
    return summary.metrics.combined_rmse, summary.metrics.loss


def select_best_summary(summaries: Sequence[TrainingRunSummary]) -> TrainingRunSummary | None:
    ranked = [summary for summary in summaries if summary.metrics is not None]
    if not ranked:
        return None
    return min(ranked, key=_metrics_rank)


def build_sweep_candidates(args: argparse.Namespace) -> list[dict[str, Any]]:
    hidden_sizes = args.hidden_sizes or [args.hidden_size]
    num_layers_options = args.num_layers_grid or [args.num_layers]
    dropouts = args.dropouts or [args.dropout]
    learning_rates = args.learning_rates or [args.learning_rate]
    epochs_options = args.epochs_grid or [args.epochs]

    candidates: list[dict[str, Any]] = []
    for hidden_size, num_layers, dropout, learning_rate, epochs in itertools.product(
        hidden_sizes,
        num_layers_options,
        dropouts,
        learning_rates,
        epochs_options,
    ):
        candidates.append(
            {
                "model_config": ModelConfig(
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout,
                    head_hidden_size=args.head_hidden_size,
                ),
                "learning_rate": float(learning_rate),
                "epochs": int(epochs),
            }
        )
    return candidates


def _flatten_trial_record(record: dict[str, Any]) -> dict[str, Any]:
    summary = record["summary"]
    metrics = summary.get("metrics") or {}
    model_config = summary.get("model_config") or {}
    return {
        "trial_index": record["trial_index"],
        "selected": record.get("selected", False),
        "hidden_size": model_config.get("hidden_size"),
        "num_layers": model_config.get("num_layers"),
        "dropout": model_config.get("dropout"),
        "head_hidden_size": model_config.get("head_hidden_size"),
        "epochs_requested": summary.get("epochs_requested"),
        "epochs_completed": summary.get("epochs_completed"),
        "learning_rate": summary.get("learning_rate"),
        "batch_size": summary.get("batch_size"),
        "validation_split": summary.get("validation_split"),
        "best_epoch": summary.get("best_epoch"),
        "combined_rmse": metrics.get("combined_rmse"),
        "loss": metrics.get("loss"),
        "close_rmse": metrics.get("close_rmse"),
        "volatility_rmse": metrics.get("volatility_rmse"),
        "checkpoint_saved": summary.get("checkpoint_saved"),
        "checkpoint_path": summary.get("checkpoint_path"),
    }


def _write_sweep_report(report_path: Path, payload: dict[str, Any]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    csv_path = report_path.with_suffix(".csv")
    trial_rows = [_flatten_trial_record(trial) for trial in payload.get("trials", [])]
    if not trial_rows:
        return
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(trial_rows[0].keys()))
        writer.writeheader()
        writer.writerows(trial_rows)


def _run_single_training(
    *,
    data_dir: Path,
    checkpoint_path: Path,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    validation_split: float,
    early_stopping_patience: int,
    model_config: ModelConfig,
    save_checkpoint: bool,
) -> TrainingRunSummary:
    result = train_model(
        data_dir=data_dir,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        validation_split=validation_split,
        early_stopping_patience=early_stopping_patience,
        checkpoint_path=checkpoint_path,
        save_checkpoint=save_checkpoint,
        device=device,
        model_config=model_config,
    )
    return result.summary


def _run_sweep(
    *,
    args: argparse.Namespace,
    data_dir: Path,
    checkpoint_path: Path,
    report_path: Path,
    device: torch.device,
) -> int:
    candidates = build_sweep_candidates(args)
    if not candidates:
        print("No sweep candidates generated.")
        return 1

    print(f"Starting hyperparameter sweep with {len(candidates)} trial(s)...")
    trial_records: list[dict[str, Any]] = []
    summaries: list[TrainingRunSummary] = []
    for index, candidate in enumerate(candidates, start=1):
        model_config = candidate["model_config"]
        learning_rate = candidate["learning_rate"]
        epochs = candidate["epochs"]
        summary = _run_single_training(
            data_dir=data_dir,
            checkpoint_path=checkpoint_path,
            device=device,
            epochs=epochs,
            batch_size=args.batch_size,
            learning_rate=learning_rate,
            validation_split=args.validation_split,
            early_stopping_patience=args.early_stopping_patience,
            model_config=model_config,
            save_checkpoint=False,
        )
        summaries.append(summary)
        trial_records.append({"trial_index": index, "summary": summary.to_dict()})
        metrics = summary.metrics
        metrics_label = (
            f"combined_rmse={metrics.combined_rmse:.6f}, loss={metrics.loss:.6f}"
            if metrics is not None
            else "no-metrics"
        )
        print(
            f"[trial {index}/{len(candidates)}] "
            f"hidden={model_config.hidden_size}, layers={model_config.num_layers}, "
            f"dropout={model_config.dropout:.3f}, lr={learning_rate:.6g}, epochs={epochs} -> {metrics_label}"
        )

    best_summary = select_best_summary(summaries)
    if best_summary is None or best_summary.metrics is None:
        print("Sweep completed, but no valid validation metrics were produced.")
        return 1

    best_trial_index = next(
        index
        for index, summary in enumerate(summaries, start=1)
        if summary == best_summary
    )
    best_model_config = best_summary.model_config
    print(
        "Re-training best configuration for final checkpoint: "
        f"hidden={best_model_config.hidden_size}, layers={best_model_config.num_layers}, "
        f"dropout={best_model_config.dropout:.3f}, lr={best_summary.learning_rate:.6g}, "
        f"epochs={best_summary.epochs_requested}"
    )
    final_summary = _run_single_training(
        data_dir=data_dir,
        checkpoint_path=checkpoint_path,
        device=device,
        epochs=best_summary.epochs_requested,
        batch_size=best_summary.batch_size,
        learning_rate=best_summary.learning_rate,
        validation_split=best_summary.validation_split,
        early_stopping_patience=best_summary.early_stopping_patience,
        model_config=best_model_config,
        save_checkpoint=True,
    )
    for trial in trial_records:
        trial["selected"] = trial["trial_index"] == best_trial_index

    report_payload = {
        "mode": "sweep",
        "selection_metric": "combined_rmse",
        "device": str(device),
        "data_dir": str(data_dir),
        "checkpoint_path": str(checkpoint_path),
        "trial_count": len(trial_records),
        "best_trial_index": best_trial_index,
        "best_trial": trial_records[best_trial_index - 1],
        "selected_checkpoint": final_summary.to_dict(),
        "trials": trial_records,
    }
    _write_sweep_report(report_path, report_payload)
    print(
        "Sweep complete. "
        f"Best combined RMSE={best_summary.metrics.combined_rmse:.6f}. "
        f"Final checkpoint saved to {checkpoint_path}"
    )
    print(f"Sweep report written to {report_path} and {report_path.with_suffix('.csv')}")
    return 0


def main() -> int:
    args = _parse_args()
    data_dir = Path(args.data_dir)
    checkpoint_path = Path(args.checkpoint_path)
    report_path = Path(args.report_path)
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sequences, summaries = inspect_training_data_sources(data_dir)
    sequence_count = len(sequences)
    observation_count = sum(len(sequence) for sequence in sequences)
    window_count = sum(max(0, len(sequence) - SEQUENCE_LENGTH) for sequence in sequences)

    _print_data_inventory(
        data_dir,
        checkpoint_path,
        device,
        summaries,
        sequence_count=sequence_count,
        observation_count=observation_count,
        window_count=window_count,
    )

    if args.list_data:
        return 0 if summaries else 1

    if not sequence_count or not window_count:
        print(
            "No sufficient training data found. Add prepared market series files under the data directory "
            "with fields like date, close, volatility_5d, and optional sentiment_score."
        )
        return 1

    if args.sweep or any(
        option is not None
        for option in (args.hidden_sizes, args.num_layers_grid, args.dropouts, args.learning_rates, args.epochs_grid)
    ):
        return _run_sweep(
            args=args,
            data_dir=data_dir,
            checkpoint_path=checkpoint_path,
            report_path=report_path,
            device=device,
        )

    print("Starting professional forecaster training...")
    summary = _run_single_training(
        data_dir=data_dir,
        checkpoint_path=checkpoint_path,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        validation_split=args.validation_split,
        early_stopping_patience=args.early_stopping_patience,
        model_config=_build_model_config(args),
        save_checkpoint=True,
    )
    metrics = summary.metrics
    if metrics is not None:
        print(
            "Validation metrics: "
            f"loss={metrics.loss:.6f}, combined_rmse={metrics.combined_rmse:.6f}, "
            f"close_rmse={metrics.close_rmse:.6f}, volatility_rmse={metrics.volatility_rmse:.6f}"
        )
    print(f"Training complete. Best checkpoint saved to {checkpoint_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
