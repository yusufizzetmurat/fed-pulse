from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from app.determinism import enable_deterministic_mode, make_generator, seed_worker
from app.evaluation.metrics import EvaluationMetrics, TrainingResult, TrainingRunSummary
from app.models.config import (
    BEST_MODEL_PATH,
    DEFAULT_BATCH_SIZE,
    DEFAULT_DROPOUT,
    DEFAULT_EARLY_STOPPING_PATIENCE,
    DEFAULT_EPOCHS,
    DEFAULT_HEAD_HIDDEN_SIZE,
    DEFAULT_HIDDEN_SIZE,
    DEFAULT_INITIAL_DECAY_RATE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_NUM_LAYERS,
    DEFAULT_VALIDATION_SPLIT,
    FEATURE_SIZE,
    FeatureVector,
    ModelConfig,
)
from app.models.lstm import ForecasterModel
from app.training.loaders import (
    _build_training_tensors,
    _split_train_validation,
    load_training_sequences_from_data,
)


def _resolve_device(device: str | torch.device | None = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _coerce_model_config(model_config: ModelConfig | dict[str, Any] | None = None) -> ModelConfig:
    if isinstance(model_config, ModelConfig):
        return model_config
    if isinstance(model_config, dict):
        return ModelConfig(
            input_size=int(model_config.get("input_size", FEATURE_SIZE)),
            hidden_size=int(model_config.get("hidden_size", DEFAULT_HIDDEN_SIZE)),
            num_layers=int(model_config.get("num_layers", DEFAULT_NUM_LAYERS)),
            dropout=float(model_config.get("dropout", DEFAULT_DROPOUT)),
            head_hidden_size=int(model_config.get("head_hidden_size", DEFAULT_HEAD_HIDDEN_SIZE)),
            initial_decay_rate=float(model_config.get("initial_decay_rate", DEFAULT_INITIAL_DECAY_RATE)),
            text_channel=str(model_config.get("text_channel", "scalar")),
            embedding_adapter_dim=int(model_config.get("embedding_adapter_dim", 128)),
            credibility_features=bool(model_config.get("credibility_features", False)),
        )
    return ModelConfig()


def _build_model(
    model_config: ModelConfig | dict[str, Any] | None = None,
    *,
    device: torch.device | None = None,
) -> ForecasterModel:
    resolved_config = _coerce_model_config(model_config)
    model = ForecasterModel(**resolved_config.to_dict())
    if device is not None:
        model = model.to(device)
    return model


def _zero_credibility(model: ForecasterModel, batch_size: int, device: torch.device) -> torch.Tensor | None:
    """Return a zero credibility tensor for the batch when the model expects one.

    Models trained with ``credibility_features=True`` must always receive a
    ``credibility`` tensor on the forward path. Until the per-row vtasca + FRED
    loader is wired into the data pipeline, this hook supplies the neutral
    "all axes unknown" reading so training stays runnable.
    """

    if not getattr(model, "credibility_features", False):
        return None
    dim = int(getattr(model, "credibility_dim", 4))
    return torch.zeros((batch_size, dim), dtype=torch.float32, device=device)


def _evaluate_model(
    model: ForecasterModel,
    loader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
) -> EvaluationMetrics:
    model.eval()
    total_loss = 0.0
    total_items = 0
    close_squared_error = 0.0
    volatility_squared_error = 0.0
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device, non_blocking=device.type == "cuda")
            batch_y = batch_y.to(device, non_blocking=device.type == "cuda")
            credibility = _zero_credibility(model, batch_x.size(0), device)
            kwargs = {"credibility": credibility} if credibility is not None else {}
            predictions = model(batch_x, **kwargs)
            loss = loss_fn(predictions, batch_y)
            batch_size = batch_x.size(0)
            total_loss += float(loss.item()) * batch_size
            total_items += batch_size
            delta = predictions - batch_y
            close_squared_error += float(torch.square(delta[:, 0]).sum().item())
            volatility_squared_error += float(torch.square(delta[:, 1]).sum().item())
    if total_items <= 0:
        return EvaluationMetrics(
            loss=float("inf"),
            close_rmse=float("inf"),
            volatility_rmse=float("inf"),
            combined_rmse=float("inf"),
        )

    combined_squared_error = close_squared_error + volatility_squared_error
    return EvaluationMetrics(
        loss=total_loss / total_items,
        close_rmse=math.sqrt(close_squared_error / total_items),
        volatility_rmse=math.sqrt(volatility_squared_error / total_items),
        combined_rmse=math.sqrt(combined_squared_error / (total_items * 2)),
    )


def train_model(
    *,
    base_model: ForecasterModel | None = None,
    model_config: ModelConfig | dict[str, Any] | None = None,
    vectors: list[FeatureVector] | None = None,
    data_dir: str | Path | None = None,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    validation_split: float = DEFAULT_VALIDATION_SPLIT,
    early_stopping_patience: int = DEFAULT_EARLY_STOPPING_PATIENCE,
    checkpoint_path: str | Path | None = None,
    save_checkpoint: bool = True,
    device: str | torch.device | None = None,
    seed: int | None = None,
) -> TrainingResult:
    if seed is not None:
        enable_deterministic_mode(seed)
    device_obj = _resolve_device(device)
    active_model_config = ModelConfig.from_model(base_model) if base_model is not None else _coerce_model_config(model_config)
    sequence_groups = load_training_sequences_from_data(data_dir)
    if vectors:
        sequence_groups.append(list(vectors))

    x, y = _build_training_tensors(sequence_groups)
    if x is None or y is None:
        model = copy.deepcopy(base_model).to(device_obj) if base_model is not None else _build_model(active_model_config, device=device_obj)
        model.eval()
        return TrainingResult(
            model=model,
            summary=TrainingRunSummary(
                model_config=ModelConfig.from_model(model),
                device=str(device_obj),
                epochs_requested=epochs,
                epochs_completed=0,
                batch_size=batch_size,
                learning_rate=learning_rate,
                validation_split=validation_split,
                early_stopping_patience=early_stopping_patience,
                sequence_groups=len(sequence_groups),
                total_windows=0,
                train_windows=0,
                validation_windows=0,
                checkpoint_path=str(checkpoint_path) if checkpoint_path is not None else str(BEST_MODEL_PATH),
                checkpoint_saved=False,
                best_epoch=None,
                metrics=None,
            ),
        )

    train_x, train_y, val_x, val_y = _split_train_validation(x, y, validation_split)
    # The current Torch build emits deprecation warnings from DataLoader pinning internals.
    # For this dataset size, disabling pinning keeps training clean without a meaningful throughput hit.
    pin_memory = False
    loader_generator = make_generator(seed) if seed is not None else None
    train_loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=min(batch_size, len(train_x)),
        shuffle=True,
        pin_memory=pin_memory,
        generator=loader_generator,
        worker_init_fn=seed_worker if seed is not None else None,
    )
    val_loader = DataLoader(
        TensorDataset(val_x, val_y),
        batch_size=min(batch_size, len(val_x)),
        shuffle=False,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker if seed is not None else None,
    )

    work_model = (
        copy.deepcopy(base_model).to(device_obj)
        if base_model is not None
        else _build_model(active_model_config, device=device_obj)
    )
    work_model.train()
    optimizer = torch.optim.AdamW(work_model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    loss_fn = nn.SmoothL1Loss(beta=0.02)

    best_metrics: EvaluationMetrics | None = None
    best_state = copy.deepcopy(work_model.state_dict())
    best_epoch: int | None = None
    completed_epochs = 0
    stale_epochs = 0
    checkpoint_target = Path(checkpoint_path) if checkpoint_path is not None else BEST_MODEL_PATH

    for epoch_index in range(epochs):
        work_model.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device_obj, non_blocking=device_obj.type == "cuda")
            batch_y = batch_y.to(device_obj, non_blocking=device_obj.type == "cuda")
            optimizer.zero_grad(set_to_none=True)
            credibility = _zero_credibility(work_model, batch_x.size(0), device_obj)
            kwargs = {"credibility": credibility} if credibility is not None else {}
            predictions = work_model(batch_x, **kwargs)
            loss = loss_fn(predictions, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(work_model.parameters(), max_norm=1.0)
            optimizer.step()

        completed_epochs = epoch_index + 1
        eval_metrics = _evaluate_model(work_model, val_loader, device_obj, loss_fn)
        scheduler.step(eval_metrics.loss)

        if best_metrics is None or eval_metrics.loss + 1e-6 < best_metrics.loss:
            best_metrics = eval_metrics
            best_state = copy.deepcopy(work_model.state_dict())
            best_epoch = completed_epochs
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= early_stopping_patience:
                break

    work_model.load_state_dict(best_state)
    work_model.eval()
    if best_metrics is None:
        best_metrics = _evaluate_model(work_model, val_loader, device_obj, loss_fn)

    summary = TrainingRunSummary(
        model_config=ModelConfig.from_model(work_model),
        device=str(device_obj),
        epochs_requested=epochs,
        epochs_completed=completed_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        validation_split=validation_split,
        early_stopping_patience=early_stopping_patience,
        sequence_groups=len(sequence_groups),
        total_windows=len(x),
        train_windows=len(train_x),
        validation_windows=len(val_x),
        checkpoint_path=str(checkpoint_target),
        checkpoint_saved=save_checkpoint,
        best_epoch=best_epoch,
        metrics=best_metrics,
    )

    if save_checkpoint:
        from app.training.checkpoint import _save_model_checkpoint

        _save_model_checkpoint(work_model, checkpoint_target, summary)
        # Lazy-import the services facade so subsequent inference picks up the
        # freshly trained weights without a process restart. Doing this at the
        # module top would create a circular (services.forecaster -> training.loop -> services.forecaster).
        try:
            from app.services.forecaster import _set_singleton_after_train

            _set_singleton_after_train(work_model, checkpoint_target, device_obj)
        except ImportError:  # pragma: no cover — facade unavailable, e.g. tests
            pass

    return TrainingResult(model=work_model, summary=summary)


def bootstrap_checkpoint(
    *,
    vectors: list[FeatureVector],
    epochs: int = 80,
    batch_size: int = 64,
    learning_rate: float = 3e-4,
    validation_split: float = 0.2,
    early_stopping_patience: int = 10,
    checkpoint_path: str | Path = BEST_MODEL_PATH,
) -> TrainingResult:
    return train_model(
        vectors=vectors,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        validation_split=validation_split,
        early_stopping_patience=early_stopping_patience,
        checkpoint_path=checkpoint_path,
        save_checkpoint=True,
    )
