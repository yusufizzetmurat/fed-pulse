from __future__ import annotations

import argparse
from pathlib import Path

import torch

from app.services.forecaster import (
    BEST_MODEL_PATH,
    DEFAULT_BATCH_SIZE,
    DEFAULT_DATA_DIR,
    DEFAULT_EARLY_STOPPING_PATIENCE,
    DEFAULT_EPOCHS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_VALIDATION_SPLIT,
    SEQUENCE_LENGTH,
    load_training_sequences_from_data,
    train_model,
)


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
        "--device",
        default=None,
        help="Explicit device override, e.g. 'cuda', 'cuda:0', or 'cpu'.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    data_dir = Path(args.data_dir)
    checkpoint_path = Path(args.checkpoint_path)
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sequences = load_training_sequences_from_data(data_dir)
    sequence_count = len(sequences)
    observation_count = sum(len(sequence) for sequence in sequences)
    window_count = sum(max(0, len(sequence) - SEQUENCE_LENGTH) for sequence in sequences)

    print(f"Device: {device}")
    print(f"Data directory: {data_dir}")
    print(f"Sequence groups discovered: {sequence_count}")
    print(f"Observations discovered: {observation_count}")
    print(f"Training windows available: {window_count}")

    if not sequence_count or not window_count:
        print("No sufficient training data found. Add prepared series files under the data directory.")
        return 1

    print("Starting professional forecaster training...")
    train_model(
        data_dir=data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        validation_split=args.validation_split,
        early_stopping_patience=args.early_stopping_patience,
        checkpoint_path=checkpoint_path,
        save_checkpoint=True,
        device=device,
    )
    print(f"Training complete. Best checkpoint saved to {checkpoint_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
