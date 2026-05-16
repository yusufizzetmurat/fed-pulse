from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from app.models.config import ModelConfig


@dataclass(frozen=True)
class EvaluationMetrics:
    loss: float
    close_rmse: float
    volatility_rmse: float
    combined_rmse: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class TrainingRunSummary:
    model_config: ModelConfig
    device: str
    epochs_requested: int
    epochs_completed: int
    batch_size: int
    learning_rate: float
    validation_split: float
    early_stopping_patience: int
    sequence_groups: int
    total_windows: int
    train_windows: int
    validation_windows: int
    checkpoint_path: str | None
    checkpoint_saved: bool
    best_epoch: int | None = None
    metrics: EvaluationMetrics | None = None
    train_metrics: EvaluationMetrics | None = None
    # Explicit walk-forward partition metrics. ``val_metrics`` is the
    # best early-stopping checkpoint's validation RMSE; ``test_metrics``
    # is the held-out test-partition RMSE -- the headline number the
    # aggregator emits as ``test-RMSE``. On the legacy 80/20 internal
    # split path both fields collapse to the val partition (no real
    # held-out test set exists), so the back-compat regression contract
    # is preserved while the walk-forward path carries a real test
    # eval. ``protocol`` distinguishes the two paths in the per-trial
    # record.
    val_metrics: EvaluationMetrics | None = None
    test_metrics: EvaluationMetrics | None = None
    fold_id: str | None = None
    protocol: str = "legacy-80-20"
    weight_decay: float = 1e-4
    target_mode: str = "real"
    text_encoder: str | None = None
    text_adapter_dim: int = 0
    text_pool_lambda_inv_days: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrainingResult:
    model: "Any"
    summary: TrainingRunSummary


@dataclass
class TrainingDataSourceSummary:
    path: Path
    format: str
    record_groups: int
    records: int
    vectors: int
    usable_sequences: int
    status: str
    message: str
