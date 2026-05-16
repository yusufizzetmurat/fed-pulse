from __future__ import annotations

import argparse
import sys
import warnings
import csv
import itertools
import json
from pathlib import Path
from typing import Any, Sequence

import torch

from app.models import FORECASTER_ARCHITECTURES
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
    FeatureVector,
    ModelConfig,
    TrainingRunSummary,
    SEQUENCE_LENGTH,
    inspect_training_data_sources,
    train_model,
)
from app.training.loaders import load_training_sequences_from_package

# Official seed set for the multi-architecture sweep (mirrors the NLP bake-off
# protocol in ``docs/benchmark-policy.md``).
DEFAULT_SWEEP_SEEDS: tuple[int, ...] = (11, 29, 47, 71, 97)

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
        "--training-package-id",
        default=None,
        help=(
            "Phase 8 training-package id under data/processed/. When set, the "
            "trainer consumes events.parquet's prior_bars_json column instead "
            "of scanning --data-dir for raw market-record JSON/JSONL/CSV files. "
            "Takes precedence over --data-dir when both are supplied."
        ),
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
    # The walk-forward validation slice is a *chronological* prefix of the
    # windows we hold out from training, not a sklearn-style random split.
    # The canonical flag is now ``--validation-fraction``; ``--validation-split``
    # stays available as a deprecated alias so existing run scripts keep
    # working until the next major version of the CLI lands.
    parser.add_argument(
        "--validation-fraction",
        "--validation-split",
        dest="validation_fraction",
        type=float,
        default=DEFAULT_VALIDATION_SPLIT,
        help=(
            "Fraction of windows reserved for the walk-forward validation "
            "slice (chronological prefix, not a shuffle). The deprecated "
            "--validation-split alias is accepted for backwards compatibility."
        ),
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
        "--architecture",
        choices=list(FORECASTER_ARCHITECTURES),
        default="lstm",
        help="Forecaster architecture for a single training run.",
    )
    parser.add_argument(
        "--architectures",
        nargs="+",
        choices=list(FORECASTER_ARCHITECTURES),
        help="Sweep over the listed architectures (combine with --sweep).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Deterministic seed for a single training run.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        help="Sweep over the listed seeds. Defaults to the official seed set when "
        "combined with --architectures.",
    )
    parser.add_argument(
        "--credibility-features",
        action="store_true",
        help="Enable the 4-axis credibility feature path on the forecaster. Default off "
        "preserves the byte-identical regression-test contract for architecture=lstm.",
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
        "--target-mode",
        choices=("event_study", "realized_return"),
        default="event_study",
        help=(
            "Target-frame derivation for the training-package loader. "
            "'event_study' (default) uses abnormal_return + volatility_shift "
            "so the synthesised target removes the broad-market component "
            "from the close and reconstructs the post-event 10d realised "
            "vol from the prior vol + shift. 'realized_return' reproduces "
            "the pre-fix behaviour (close * (1 + realized_return) and a "
            "literal copy of prior vol_5d) and is preserved for back-compat "
            "smoke tests only. Ignored when --training-package-id is unset."
        ),
    )
    parser.add_argument(
        "--list-data",
        action="store_true",
        help="Only inspect discovered datasets and exit without training.",
    )
    args = parser.parse_args()
    # Emit DeprecationWarning when the legacy --validation-split alias is
    # used. argparse silently maps it onto validation_fraction, so callers
    # would otherwise have no signal that the name is on its way out.
    if any(arg == "--validation-split" or arg.startswith("--validation-split=") for arg in sys.argv[1:]):
        warnings.warn(
            "--validation-split is deprecated; use --validation-fraction. "
            "The walk-forward validation slice is a chronological prefix, "
            "not a sklearn-style random split, and the new flag name reflects that.",
            DeprecationWarning,
            stacklevel=2,
        )
    return args


def _build_model_config(args: argparse.Namespace) -> ModelConfig:
    return ModelConfig(
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        head_hidden_size=args.head_hidden_size,
        architecture=args.architecture,
        credibility_features=bool(args.credibility_features),
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
    architectures = args.architectures or [args.architecture]
    # When the caller asks for an architecture sweep but doesn't list seeds we
    # fall back to the official five-seed set; this matches the bake-off
    # aggregator and avoids accidentally publishing a single-seed table.
    if args.seeds:
        seeds: list[int | None] = [int(s) for s in args.seeds]
    elif args.architectures:
        seeds = list(DEFAULT_SWEEP_SEEDS)
    else:
        seeds = [args.seed]

    candidates: list[dict[str, Any]] = []
    for architecture, hidden_size, num_layers, dropout, learning_rate, epochs, seed in itertools.product(
        architectures,
        hidden_sizes,
        num_layers_options,
        dropouts,
        learning_rates,
        epochs_options,
        seeds,
    ):
        candidates.append(
            {
                "model_config": ModelConfig(
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout,
                    head_hidden_size=args.head_hidden_size,
                    architecture=str(architecture),
                    credibility_features=bool(args.credibility_features),
                ),
                "learning_rate": float(learning_rate),
                "epochs": int(epochs),
                "seed": int(seed) if seed is not None else None,
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
        "architecture": model_config.get("architecture") or record.get("architecture"),
        "seed": record.get("seed"),
        "credibility_features": model_config.get("credibility_features"),
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
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

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
    validation_fraction: float,
    early_stopping_patience: int,
    model_config: ModelConfig,
    save_checkpoint: bool,
    seed: int | None = None,
    sequence_groups: Sequence[Sequence[FeatureVector]] | None = None,
) -> TrainingRunSummary:
    # ``validation_fraction`` is the new idiomatic kwarg name. The
    # underlying ``train_model`` keeps ``validation_split`` for backwards
    # compatibility with checkpoints and tests; we relay by name here.
    # When ``sequence_groups`` is provided (training-package path), the
    # legacy ``data_dir`` scan is bypassed and the groups are appended as
    # standalone ``vectors=`` payloads, one call per group. Doing it this
    # way preserves the back-compat surface of ``train_model`` and
    # avoids touching the regression-test contract.
    if sequence_groups:
        result = _train_model_with_groups(
            sequence_groups=sequence_groups,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            validation_fraction=validation_fraction,
            early_stopping_patience=early_stopping_patience,
            checkpoint_path=checkpoint_path,
            save_checkpoint=save_checkpoint,
            device=device,
            model_config=model_config,
            seed=seed,
        )
    else:
        result = train_model(
            data_dir=data_dir,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            validation_split=validation_fraction,
            early_stopping_patience=early_stopping_patience,
            checkpoint_path=checkpoint_path,
            save_checkpoint=save_checkpoint,
            device=device,
            model_config=model_config,
            seed=seed,
        )
    return result.summary


def _train_model_with_groups(
    *,
    sequence_groups: Sequence[Sequence[FeatureVector]],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    validation_fraction: float,
    early_stopping_patience: int,
    checkpoint_path: Path,
    save_checkpoint: bool,
    device: torch.device,
    model_config: ModelConfig,
    seed: int | None,
) -> Any:
    """Invoke ``train_model`` against pre-loaded sequence groups.

    Routes through the ``sequence_groups`` kwarg on ``train_model`` so
    the legacy ``data_dir`` scan is bypassed entirely. Each inner list
    becomes one group consumed by ``_build_training_tensors`` -- the
    slicer treats each group independently, so prior bars from one
    FOMC event never leak into another event's training window.
    """

    materialised: list[list[FeatureVector]] = [list(group) for group in sequence_groups]
    return train_model(
        sequence_groups=materialised,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        validation_split=validation_fraction,
        early_stopping_patience=early_stopping_patience,
        checkpoint_path=checkpoint_path,
        save_checkpoint=save_checkpoint,
        device=device,
        model_config=model_config,
        seed=seed,
    )


def _run_sweep(
    *,
    args: argparse.Namespace,
    data_dir: Path,
    checkpoint_path: Path,
    report_path: Path,
    device: torch.device,
    sequence_groups: Sequence[Sequence[FeatureVector]] | None = None,
    training_package_id: str | None = None,
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
        seed = candidate.get("seed")
        summary = _run_single_training(
            data_dir=data_dir,
            checkpoint_path=checkpoint_path,
            device=device,
            epochs=epochs,
            batch_size=args.batch_size,
            learning_rate=learning_rate,
            validation_fraction=args.validation_fraction,
            early_stopping_patience=args.early_stopping_patience,
            model_config=model_config,
            save_checkpoint=False,
            seed=seed,
            sequence_groups=sequence_groups,
        )
        summaries.append(summary)
        trial_records.append(
            {
                "trial_index": index,
                "architecture": model_config.architecture,
                "seed": seed,
                "summary": summary.to_dict(),
            }
        )
        metrics = summary.metrics
        metrics_label = (
            f"combined_rmse={metrics.combined_rmse:.6f}, loss={metrics.loss:.6f}"
            if metrics is not None
            else "no-metrics"
        )
        print(
            f"[trial {index}/{len(candidates)}] "
            f"arch={model_config.architecture}, seed={seed}, "
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
    best_seed = candidates[best_trial_index - 1].get("seed")
    print(
        "Re-training best configuration for final checkpoint: "
        f"arch={best_model_config.architecture}, seed={best_seed}, "
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
        # ``best_summary.validation_split`` is the persisted summary
        # field (frozen for back-compat with the existing
        # TrainingRunSummary dataclass); we pass it under the new
        # idiomatic ``validation_fraction`` kwarg.
        validation_fraction=best_summary.validation_split,
        early_stopping_patience=best_summary.early_stopping_patience,
        model_config=best_model_config,
        save_checkpoint=True,
        seed=best_seed,
        sequence_groups=sequence_groups,
    )
    for trial in trial_records:
        trial["selected"] = trial["trial_index"] == best_trial_index

    report_payload = {
        "mode": "sweep",
        "selection_metric": "combined_rmse",
        "device": str(device),
        "data_dir": str(data_dir),
        "training_package_id": training_package_id,
        "checkpoint_path": str(checkpoint_path),
        "credibility_features": bool(args.credibility_features),
        "architectures": sorted({trial["architecture"] for trial in trial_records}),
        "seeds": sorted({trial["seed"] for trial in trial_records if trial["seed"] is not None}),
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

    # ``--training-package-id`` takes precedence over ``--data-dir`` so an
    # exported DATA_DIR override never silently falls back to the legacy
    # JSON/JSONL scan. The override path is logged so precedence is
    # auditable from the run log.
    use_package_path = bool(args.training_package_id)
    if use_package_path and args.data_dir != str(DEFAULT_DATA_DIR):
        print(
            f"--training-package-id is set; ignoring --data-dir={args.data_dir} "
            "in favour of the Phase 8 events.parquet path."
        )

    package_sequences: list[list[FeatureVector]] | None = None
    if use_package_path:
        print(f"Training-package id: {args.training_package_id}")
        print(f"Target mode: {args.target_mode}")
        package_sequences = load_training_sequences_from_package(
            args.training_package_id,
            target_mode=args.target_mode,
        )
        sequence_count = len(package_sequences)
        observation_count = sum(len(sequence) for sequence in package_sequences)
        window_count = sum(max(0, len(sequence) - SEQUENCE_LENGTH) for sequence in package_sequences)
        print(f"Device: {device}")
        print(f"Checkpoint path: {checkpoint_path}")
        print(f"Existing checkpoint: {'yes' if checkpoint_path.exists() else 'no'}")
        print(f"Sequence groups discovered: {sequence_count}")
        print(f"Observations discovered: {observation_count}")
        print(f"Training windows available: {window_count}")
        if args.list_data:
            return 0 if sequence_count else 1
        if not sequence_count or not window_count:
            print(
                "No sufficient training data found in training package "
                f"{args.training_package_id}. Verify events.parquet carries "
                "prior_bars_json with the full 20-bar prior window."
            )
            return 1
    else:
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

    sweep_mode = args.sweep or any(
        option is not None
        for option in (
            args.hidden_sizes,
            args.num_layers_grid,
            args.dropouts,
            args.learning_rates,
            args.epochs_grid,
            args.architectures,
            args.seeds,
        )
    )
    if sweep_mode:
        return _run_sweep(
            args=args,
            data_dir=data_dir,
            checkpoint_path=checkpoint_path,
            report_path=report_path,
            device=device,
            sequence_groups=package_sequences,
            training_package_id=args.training_package_id,
        )

    print(
        "Starting professional forecaster training "
        f"(architecture={args.architecture}, credibility_features={bool(args.credibility_features)})..."
    )
    summary = _run_single_training(
        data_dir=data_dir,
        checkpoint_path=checkpoint_path,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        validation_fraction=args.validation_fraction,
        early_stopping_patience=args.early_stopping_patience,
        model_config=_build_model_config(args),
        save_checkpoint=True,
        seed=args.seed,
        sequence_groups=package_sequences,
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
