from __future__ import annotations

import copy
import dataclasses
import logging
import math
from pathlib import Path
from typing import Any, Sequence

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
    _build_text_embedding_tensors,
    _build_training_tensors,
    _split_train_validation,
    load_training_sequences_from_data,
)

_logger = logging.getLogger(__name__)


# Architectures whose forward path uses control-flow that ``torch.compile``
# cannot trace cleanly under the small-batch regime, or that overflow in
# fp16 autocast on the recurrent core. Compile + autocast are skipped for
# anything in this table; the eager + fp32 path runs unchanged so the
# byte-identity regression contract stays green.
_COMPILE_INCOMPATIBLE_ARCHITECTURES: frozenset[str] = frozenset({"informer", "tft"})
_AMP_INCOMPATIBLE_ARCHITECTURES: frozenset[str] = frozenset({"informer", "tft"})


def _resolve_device(device: str | torch.device | None = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _move_to_device(
    tensor: torch.Tensor | None, device: torch.device
) -> torch.Tensor | None:
    """Move ``tensor`` to ``device`` once. Returns ``None`` unchanged."""

    if tensor is None:
        return None
    if tensor.device == device:
        return tensor
    return tensor.to(device, non_blocking=device.type == "cuda")


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
            architecture=str(model_config.get("architecture", "lstm")),
            text_embedding_dim=int(model_config.get("text_embedding_dim", 0) or 0),
            text_adapter_dim=int(model_config.get("text_adapter_dim", 0) or 0),
        )
    return ModelConfig()


def _build_model(
    model_config: ModelConfig | dict[str, Any] | None = None,
    *,
    device: torch.device | None = None,
) -> ForecasterModel:
    # Local import keeps ``app.models.factory`` cold until training fires,
    # which keeps the FastAPI singleton import path narrow.
    from app.models.factory import build_forecaster

    resolved_config = _coerce_model_config(model_config)
    model = build_forecaster(resolved_config)
    if device is not None:
        model = model.to(device)
    return model


def _zero_credibility(model: nn.Module, batch_size: int, device: torch.device) -> torch.Tensor | None:
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


def _allocate_credibility_buffer(
    model: nn.Module, max_batch_size: int, device: torch.device
) -> torch.Tensor | None:
    """Pre-allocate the largest credibility buffer the epoch loop needs.

    Models trained with ``credibility_features=True`` accept a per-batch
    credibility tensor on every forward; the old hook allocated a fresh
    zero tensor inside the train loop, which dominated kernel-launch
    overhead at small batch sizes. The buffer is sized to ``max_batch_size``
    and sliced down for each batch, so the residual launch is a single
    narrow + no allocation.
    """

    if not getattr(model, "credibility_features", False):
        return None
    dim = int(getattr(model, "credibility_dim", 4))
    return torch.zeros((max_batch_size, dim), dtype=torch.float32, device=device)


def _slice_credibility_buffer(
    buffer: torch.Tensor | None, batch_size: int
) -> torch.Tensor | None:
    """Slice a pre-allocated credibility buffer to the active batch size."""

    if buffer is None:
        return None
    if batch_size == buffer.shape[0]:
        return buffer
    return buffer.narrow(0, 0, batch_size)


def _copy_state_inplace(
    target_state: dict[str, torch.Tensor], source_state: dict[str, torch.Tensor]
) -> None:
    """Overwrite ``target_state`` tensors with ``source_state`` values in place.

    Replaces the previous ``copy.deepcopy(model.state_dict())`` pattern
    on every val improvement: the deepcopy ran a GPU->CPU sync per
    tensor (250k+ params on a hidden=128 layers=3 LSTM), which the eval
    loop was paying for once per epoch. The in-place form keeps the
    best-state buffers on the same device the model lives on and copies
    only the tensor data; no Python-side cloning, no host sync.
    """

    with torch.no_grad():
        for key, value in source_state.items():
            target = target_state.get(key)
            if target is None:
                target_state[key] = value.detach().clone()
                continue
            target.copy_(value, non_blocking=True)


def _snapshot_state(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return a detached clone of the model's state-dict tensors."""

    with torch.no_grad():
        return {key: value.detach().clone() for key, value in model.state_dict().items()}


def _resolve_compile_amp_flags(
    model: nn.Module,
    architecture: str,
    device: torch.device,
    *,
    use_compile: bool,
    use_amp: bool,
) -> tuple[bool, bool]:
    """Return the effective ``(use_compile, use_amp)`` after compatibility checks.

    Compile and AMP only fire on CUDA; on CPU both reduce to no-ops and
    the helper returns ``(False, False)`` so the training loop's hot
    path stays branch-free. The per-arch table flags architectures whose
    forward paths trip on either feature; a logger warning records every
    downgrade so a sweep run can audit the per-cell decisions in the
    container log.
    """

    if device.type != "cuda":
        return False, False
    effective_compile = use_compile and architecture not in _COMPILE_INCOMPATIBLE_ARCHITECTURES
    if use_compile and not effective_compile:
        _logger.warning(
            "torch.compile disabled for architecture=%r (incompatible); "
            "running eager forward instead",
            architecture,
        )
    effective_amp = use_amp and architecture not in _AMP_INCOMPATIBLE_ARCHITECTURES
    if use_amp and not effective_amp:
        _logger.warning(
            "autocast disabled for architecture=%r (incompatible); "
            "running fp32 forward instead",
            architecture,
        )
    return effective_compile, effective_amp


def _maybe_compile_model(
    model: nn.Module, *, use_compile: bool
) -> nn.Module:
    """Wrap ``model`` in ``torch.compile`` when ``use_compile`` is on.

    ``mode="reduce-overhead"`` is the documented setting for the
    small-model + tight-loop pattern this forecaster lives in. The
    compile call is wrapped in a broad try/except so a torch build
    without the inductor backend falls back to eager with a single
    warning instead of bringing the sweep down.
    """

    if not use_compile:
        return model
    try:
        compiled = torch.compile(model, mode="reduce-overhead")
        return compiled  # type: ignore[return-value]
    except Exception as exc:  # pragma: no cover - depends on torch build
        _logger.warning("torch.compile failed (%s); falling back to eager", exc)
        return model


def _unpack_batch(batch: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Decode a DataLoader batch into (x, y, text_embedding, text_missing).

    Two batch shapes are tolerated:

    - ``(batch_x, batch_y)`` -- legacy two-tensor batch, used on the
      pre-PR-#176 path. ``text_embedding`` and ``text_embedding_missing``
      are ``None``.
    - ``(batch_x, batch_y, batch_text_embedding, batch_text_missing)``
      -- four-tensor batch emitted when the text-embedding path is
      active. The model forward picks the extras up by name.
    """

    if len(batch) == 4:
        batch_x, batch_y, batch_text, batch_text_missing = batch
        return batch_x, batch_y, batch_text, batch_text_missing
    if len(batch) == 2:
        batch_x, batch_y = batch
        return batch_x, batch_y, None, None
    raise ValueError(
        f"unexpected batch arity from DataLoader: {len(batch)} (want 2 or 4)"
    )


def _evaluate_model(
    model: nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
    loss_fn: nn.Module,
    credibility_buffer: torch.Tensor | None = None,
) -> EvaluationMetrics:
    """Evaluate ``model`` on ``loader`` and return aggregate metrics.

    The previous implementation called ``.item()`` three times per batch
    (loss, close-squared-error, volatility-squared-error) which forced a
    GPU->CPU sync every iteration. With ~10 val batches across 40 epochs
    the loop paid ~1200 forced syncs per cell. The new path accumulates
    the running sums as GPU tensors and calls ``.item()`` exactly three
    times at the end of the loader, regardless of batch count.

    ``credibility_buffer`` is the pre-allocated zero tensor the training
    loop hoists out of the epoch; the helper slices it down to the
    active batch size on each iteration. When ``None`` the legacy
    per-batch allocation kicks in via :func:`_zero_credibility` so
    callers outside the training loop (smoke checks, regression tests)
    keep working.
    """

    model.eval()
    total_loss_sum = torch.zeros((), dtype=torch.float64, device=device)
    total_items = torch.zeros((), dtype=torch.int64, device=device)
    close_squared_error = torch.zeros((), dtype=torch.float64, device=device)
    volatility_squared_error = torch.zeros((), dtype=torch.float64, device=device)
    use_text_path = bool(getattr(model, "_text_path_active", False))
    non_blocking = device.type == "cuda"
    with torch.no_grad():
        for batch in loader:
            batch_x, batch_y, batch_text, batch_text_missing = _unpack_batch(batch)
            if batch_x.device != device:
                batch_x = batch_x.to(device, non_blocking=non_blocking)
            if batch_y.device != device:
                batch_y = batch_y.to(device, non_blocking=non_blocking)
            batch_size = batch_x.size(0)
            if credibility_buffer is not None:
                credibility = _slice_credibility_buffer(credibility_buffer, batch_size)
            else:
                credibility = _zero_credibility(model, batch_size, device)
            kwargs: dict[str, torch.Tensor] = {}
            if credibility is not None:
                kwargs["credibility"] = credibility
            if batch_text is not None and use_text_path:
                if batch_text.device != device:
                    batch_text = batch_text.to(device, non_blocking=non_blocking)
                kwargs["text_embedding"] = batch_text
                if batch_text_missing is not None:
                    if batch_text_missing.device != device:
                        batch_text_missing = batch_text_missing.to(
                            device, non_blocking=non_blocking
                        )
                    kwargs["text_embedding_missing"] = batch_text_missing
            predictions = model(batch_x, **kwargs)
            loss = loss_fn(predictions, batch_y)
            total_loss_sum += loss.detach().to(torch.float64) * batch_size
            total_items += batch_size
            delta = predictions - batch_y
            close_squared_error += torch.square(delta[:, 0]).sum().to(torch.float64)
            volatility_squared_error += torch.square(delta[:, 1]).sum().to(torch.float64)
    total_items_int = int(total_items.item())
    if total_items_int <= 0:
        return EvaluationMetrics(
            loss=float("inf"),
            close_rmse=float("inf"),
            volatility_rmse=float("inf"),
            combined_rmse=float("inf"),
        )

    total_loss_value = float(total_loss_sum.item())
    close_value = float(close_squared_error.item())
    volatility_value = float(volatility_squared_error.item())
    combined_squared_error = close_value + volatility_value
    return EvaluationMetrics(
        loss=total_loss_value / total_items_int,
        close_rmse=math.sqrt(close_value / total_items_int),
        volatility_rmse=math.sqrt(volatility_value / total_items_int),
        combined_rmse=math.sqrt(combined_squared_error / (total_items_int * 2)),
    )


def _build_partition_tensors(
    sequence_groups: Sequence[Sequence[FeatureVector]],
    *,
    fallback_text_in_dim: int,
    close_scale: float | None = None,
) -> tuple[
    torch.Tensor | None,
    torch.Tensor | None,
    float,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """Tensorise one partition into (x, y, close_scale, text_emb, text_missing).

    The text-embedding tensor is materialised against the same
    ``fallback_text_in_dim`` the legacy single-partition path uses, so a
    partition whose every sequence is missing a pooled embedding still
    materialises a zero-payload tensor of the right width when the
    model's adapter is configured for the text channel.
    """

    x, y, scale = _build_training_tensors(sequence_groups, close_scale=close_scale)
    text_emb, text_missing, _ = _build_text_embedding_tensors(
        sequence_groups, fallback_in_dim=fallback_text_in_dim
    )
    return x, y, scale, text_emb, text_missing


@dataclasses.dataclass(frozen=True)
class BackCompatTrainingSplit:
    """Marker for callers that need the legacy 80/20 internal split.

    The walk-forward training-package path supplies pre-split
    ``train_sequence_groups`` / ``val_sequence_groups`` /
    ``test_sequence_groups`` and the loop honours those partitions.
    Pre-walk-forward callers (the ``--data-dir`` scan path and the
    determinism regression-test fixture) pass a single flat list and
    the loop falls back to the documented-but-deprecated 80/20
    chronological split on the tensorised windows so the byte-identity
    contract on those legacy paths stays green.

    Constructing this object on a call signals the legacy path; the
    field is read only as a marker.
    """

    validation_fraction: float = DEFAULT_VALIDATION_SPLIT


def train_model(
    *,
    base_model: ForecasterModel | None = None,
    model_config: ModelConfig | dict[str, Any] | None = None,
    vectors: list[FeatureVector] | None = None,
    sequence_groups: list[list[FeatureVector]] | None = None,
    train_sequence_groups: Sequence[Sequence[FeatureVector]] | None = None,
    val_sequence_groups: Sequence[Sequence[FeatureVector]] | None = None,
    test_sequence_groups: Sequence[Sequence[FeatureVector]] | None = None,
    fold_id: str | None = None,
    protocol: str | None = None,
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
    weight_decay: float = 1e-4,
    shuffle_targets_control: bool = False,
    text_encoder: str | None = None,
    text_pool_lambda_inv_days: float = 0.0,
    grad_clip_norm: float = 0.0,
    use_compile: bool = True,
    use_amp: bool = True,
) -> TrainingResult:
    if seed is not None:
        enable_deterministic_mode(seed)
    device_obj = _resolve_device(device)
    active_model_config = ModelConfig.from_model(base_model) if base_model is not None else _coerce_model_config(model_config)

    # Two split protocols are honoured:
    #
    # - Walk-forward (preferred): the caller supplies pre-split
    #   train_sequence_groups / val_sequence_groups /
    #   test_sequence_groups lists. ``_split_train_validation`` is NOT
    #   called; the partitions are tensorised independently and the
    #   final reported ``test_metrics`` is the real held-out RMSE.
    # - Legacy 80/20 (back-compat): the caller supplies a single flat
    #   sequence-groups list (``vectors`` / ``sequence_groups`` /
    #   ``data_dir`` scan). The loop falls back to the documented
    #   chronological 80/20 split on the tensorised windows so the
    #   pre-walk-forward regression contract stays green.
    walk_forward_path = (
        train_sequence_groups is not None
        and val_sequence_groups is not None
        and test_sequence_groups is not None
    )
    active_protocol = protocol or ("walk-forward" if walk_forward_path else "legacy-80-20")
    fallback_text_in_dim = int(getattr(active_model_config, "text_embedding_dim", 0) or 0)

    if walk_forward_path:
        train_groups: list[list[FeatureVector]] = [list(group) for group in train_sequence_groups or []]
        val_groups: list[list[FeatureVector]] = [list(group) for group in val_sequence_groups or []]
        test_groups: list[list[FeatureVector]] = [list(group) for group in test_sequence_groups or []]
        # Fit the close-scale on the training partition only; never on
        # the val or test rows. The walk-forward protocol forbids
        # fitting any scaler over held-out events.
        train_x, train_y, close_scale, train_text_emb, train_text_missing = _build_partition_tensors(
            train_groups,
            fallback_text_in_dim=fallback_text_in_dim,
            close_scale=None,
        )
        val_x, val_y, _val_scale, val_text_emb, val_text_missing = _build_partition_tensors(
            val_groups,
            fallback_text_in_dim=fallback_text_in_dim,
            close_scale=close_scale,
        )
        test_x, test_y, _test_scale, test_text_emb, test_text_missing = _build_partition_tensors(
            test_groups,
            fallback_text_in_dim=fallback_text_in_dim,
            close_scale=close_scale,
        )
        sequence_groups_for_summary = train_groups + val_groups + test_groups
    else:
        if sequence_groups is not None:
            active_sequence_groups: list[list[FeatureVector]] = [list(group) for group in sequence_groups]
        else:
            active_sequence_groups = load_training_sequences_from_data(data_dir)
        if vectors:
            active_sequence_groups.append(list(vectors))
        sequence_groups_for_summary = active_sequence_groups

        x, y, close_scale = _build_training_tensors(active_sequence_groups)
        text_emb_tensor, text_missing_tensor, _text_emb_dim = _build_text_embedding_tensors(
            active_sequence_groups,
            fallback_in_dim=fallback_text_in_dim,
        )
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
                    sequence_groups=len(active_sequence_groups),
                    total_windows=0,
                    train_windows=0,
                    validation_windows=0,
                    checkpoint_path=str(checkpoint_path) if checkpoint_path is not None else str(BEST_MODEL_PATH),
                    checkpoint_saved=False,
                    best_epoch=None,
                    metrics=None,
                    fold_id=fold_id,
                    protocol=active_protocol,
                    weight_decay=float(weight_decay),
                    target_mode="shuffled" if shuffle_targets_control else "real",
                    text_encoder=text_encoder,
                    text_adapter_dim=int(getattr(model, "text_adapter_dim", 0) or 0),
                    text_pool_lambda_inv_days=float(text_pool_lambda_inv_days),
                ),
            )

        # Shuffled-targets control runs only on the legacy single-tensor
        # path; the walk-forward branch applies the same permutation
        # per partition below.
        if shuffle_targets_control:
            if seed is None:
                shuffle_seed = 11
            else:
                shuffle_seed = int(seed)
            shuffle_generator = torch.Generator()
            shuffle_generator.manual_seed(shuffle_seed)
            perm = torch.randperm(y.shape[0], generator=shuffle_generator)
            y = y[perm].clone()

        train_x, train_y, val_x, val_y = _split_train_validation(x, y, validation_split)
        if text_emb_tensor is not None and text_missing_tensor is not None:
            train_text_emb = text_emb_tensor[: len(train_x)]
            val_text_emb = text_emb_tensor[len(train_x) :]
            train_text_missing = text_missing_tensor[: len(train_x)]
            val_text_missing = text_missing_tensor[len(train_x) :]
        else:
            train_text_emb = val_text_emb = None
            train_text_missing = val_text_missing = None
        # Legacy path has no real held-out test partition; the val
        # tensors serve as both early-stopping and final-report eval.
        test_x = val_x
        test_y = val_y
        test_text_emb = val_text_emb
        test_text_missing = val_text_missing

    # Empty-tensor guard for the walk-forward branch. The legacy branch
    # already short-circuits above on (x, y) == (None, None).
    if walk_forward_path and (train_x is None or train_y is None):
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
                sequence_groups=len(sequence_groups_for_summary),
                total_windows=0,
                train_windows=0,
                validation_windows=0,
                checkpoint_path=str(checkpoint_path) if checkpoint_path is not None else str(BEST_MODEL_PATH),
                checkpoint_saved=False,
                best_epoch=None,
                metrics=None,
                fold_id=fold_id,
                protocol=active_protocol,
                weight_decay=float(weight_decay),
                target_mode="shuffled" if shuffle_targets_control else "real",
                text_encoder=text_encoder,
                text_adapter_dim=int(getattr(model, "text_adapter_dim", 0) or 0),
                text_pool_lambda_inv_days=float(text_pool_lambda_inv_days),
            ),
        )

    if walk_forward_path and shuffle_targets_control and train_y is not None:
        # Permute the train partition only -- the val / test partitions
        # keep their real targets so the memorisation control's
        # held-out RMSE still measures what the model learns.
        if seed is None:
            shuffle_seed = 11
        else:
            shuffle_seed = int(seed)
        shuffle_generator = torch.Generator()
        shuffle_generator.manual_seed(shuffle_seed)
        perm = torch.randperm(train_y.shape[0], generator=shuffle_generator)
        train_y = train_y[perm].clone()

    # Pre-move every partition tensor onto the target device once, so
    # the per-batch loop is left with index-and-forward only and the
    # per-iteration ``.to(device)`` calls disappear from the kernel-launch
    # budget. ``pin_memory`` stays off because the tensors live on the
    # GPU after this point. On CPU device the helper is a no-op (the
    # tensors already live there), which keeps the byte-identity
    # regression contract on the determinism test green.
    train_x = _move_to_device(train_x, device_obj)
    train_y = _move_to_device(train_y, device_obj)
    train_text_emb = _move_to_device(train_text_emb, device_obj)
    train_text_missing = _move_to_device(train_text_missing, device_obj)
    val_x = _move_to_device(val_x, device_obj)
    val_y = _move_to_device(val_y, device_obj)
    val_text_emb = _move_to_device(val_text_emb, device_obj)
    val_text_missing = _move_to_device(val_text_missing, device_obj)
    test_x = _move_to_device(test_x, device_obj)
    test_y = _move_to_device(test_y, device_obj)
    test_text_emb = _move_to_device(test_text_emb, device_obj)
    test_text_missing = _move_to_device(test_text_missing, device_obj)
    # Tensors now live on the target device, so DataLoader pinning is
    # neither needed nor supported (PyTorch raises on pinning a CUDA
    # tensor). The original pin-memory comment about deprecation
    # warnings still applies on CPU device.
    pin_memory = False
    loader_generator = make_generator(seed) if seed is not None else None
    if train_text_emb is not None and train_text_missing is not None:
        train_dataset = TensorDataset(train_x, train_y, train_text_emb, train_text_missing)
    else:
        train_dataset = TensorDataset(train_x, train_y)

    # Early-stopping val loader: when the walk-forward branch supplied
    # an empty val partition (rare, edge-case folds), reuse the train
    # tensors as a tracker so the loop still has a stopping signal.
    # ``val_metrics`` then collapses to the training-set value and
    # ``test_metrics`` stays the headline number.
    if val_x is None or val_y is None or len(val_x) == 0:
        val_x_used = train_x
        val_y_used = train_y
        val_text_emb_used = train_text_emb
        val_text_missing_used = train_text_missing
    else:
        val_x_used = val_x
        val_y_used = val_y
        val_text_emb_used = val_text_emb
        val_text_missing_used = val_text_missing

    if val_text_emb_used is not None and val_text_missing_used is not None:
        val_dataset = TensorDataset(val_x_used, val_y_used, val_text_emb_used, val_text_missing_used)
    else:
        val_dataset = TensorDataset(val_x_used, val_y_used)

    if test_x is not None and test_y is not None and len(test_x) > 0:
        if test_text_emb is not None and test_text_missing is not None:
            test_dataset = TensorDataset(test_x, test_y, test_text_emb, test_text_missing)
        else:
            test_dataset = TensorDataset(test_x, test_y)
    else:
        test_dataset = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=min(batch_size, len(train_x)),
        shuffle=True,
        pin_memory=pin_memory,
        generator=loader_generator,
        worker_init_fn=seed_worker if seed is not None else None,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=min(batch_size, len(val_x_used)),
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
    optimizer = torch.optim.AdamW(work_model.parameters(), lr=learning_rate, weight_decay=float(weight_decay))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    loss_fn = nn.SmoothL1Loss(beta=0.02)

    active_arch = str(getattr(active_model_config, "architecture", "lstm") or "lstm")
    effective_compile, effective_amp = _resolve_compile_amp_flags(
        work_model,
        active_arch,
        device_obj,
        use_compile=use_compile,
        use_amp=use_amp,
    )
    # Pre-allocate the credibility zero buffer at the maximum batch size
    # so the train + eval loops don't pay an allocation per batch.
    cred_max_batch = min(batch_size, len(train_x))
    train_credibility_buffer = _allocate_credibility_buffer(
        work_model, cred_max_batch, device_obj
    )
    val_credibility_buffer = _allocate_credibility_buffer(
        work_model, min(batch_size, len(val_x_used)), device_obj
    )
    scaler: "torch.cuda.amp.GradScaler | None" = None
    if effective_amp:
        scaler = torch.cuda.amp.GradScaler()
    forward_model: nn.Module = _maybe_compile_model(
        work_model, use_compile=effective_compile
    )

    # Grad clipping is opt-in: ``grad_clip_norm > 0.0`` enables the
    # per-step clip with that norm. The legacy ``max_norm=1.0`` clip
    # forced a host sync inside ``clip_grad_norm_`` on every training
    # step; defaulting it off recovers that overhead. The CLI exposes
    # ``--grad-clip-norm`` for callers that need the old behaviour.
    clip_norm_value = float(grad_clip_norm)
    apply_grad_clip = clip_norm_value > 0.0

    best_val_metrics: EvaluationMetrics | None = None
    best_state = _snapshot_state(work_model)
    best_epoch: int | None = None
    completed_epochs = 0
    stale_epochs = 0
    checkpoint_target = Path(checkpoint_path) if checkpoint_path is not None else BEST_MODEL_PATH

    use_text_path = bool(getattr(work_model, "_text_path_active", False))

    for epoch_index in range(epochs):
        work_model.train()
        for batch in train_loader:
            batch_x, batch_y, batch_text, batch_text_missing = _unpack_batch(batch)
            # Tensors are already on the target device; the .to() calls
            # below were the hot kernel-launch source the perf rewrite
            # eliminates.
            optimizer.zero_grad(set_to_none=True)
            credibility = _slice_credibility_buffer(
                train_credibility_buffer, batch_x.size(0)
            )
            kwargs: dict[str, torch.Tensor] = {}
            if credibility is not None:
                kwargs["credibility"] = credibility
            if batch_text is not None and use_text_path:
                kwargs["text_embedding"] = batch_text
                if batch_text_missing is not None:
                    kwargs["text_embedding_missing"] = batch_text_missing
            if effective_amp:
                with torch.cuda.amp.autocast():
                    predictions = forward_model(batch_x, **kwargs)
                    loss = loss_fn(predictions, batch_y)
                assert scaler is not None
                scaler.scale(loss).backward()
                if apply_grad_clip:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(work_model.parameters(), max_norm=clip_norm_value)
                scaler.step(optimizer)
                scaler.update()
            else:
                predictions = forward_model(batch_x, **kwargs)
                loss = loss_fn(predictions, batch_y)
                loss.backward()
                if apply_grad_clip:
                    nn.utils.clip_grad_norm_(work_model.parameters(), max_norm=clip_norm_value)
                optimizer.step()

        completed_epochs = epoch_index + 1
        eval_metrics = _evaluate_model(
            forward_model,
            val_loader,
            device_obj,
            loss_fn,
            credibility_buffer=val_credibility_buffer,
        )
        scheduler.step(eval_metrics.loss)

        if best_val_metrics is None or eval_metrics.loss + 1e-6 < best_val_metrics.loss:
            best_val_metrics = eval_metrics
            _copy_state_inplace(best_state, work_model.state_dict())
            best_epoch = completed_epochs
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= early_stopping_patience:
                break

    work_model.load_state_dict(best_state)
    work_model.eval()
    if best_val_metrics is None:
        best_val_metrics = _evaluate_model(
            forward_model,
            val_loader,
            device_obj,
            loss_fn,
            credibility_buffer=val_credibility_buffer,
        )
    # Final-state training-set evaluation so the aggregator can emit
    # ``test_train_gap = (test_rmse - train_rmse) / train_rmse``. The
    # training set is re-evaluated through the same eval path (no
    # dropout / no grad / fixed batch ordering) so the number is
    # comparable to the held-out RMSE.
    train_eval_loader = DataLoader(
        train_dataset,
        batch_size=min(batch_size, len(train_x)),
        shuffle=False,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker if seed is not None else None,
    )
    train_metrics = _evaluate_model(
        forward_model,
        train_eval_loader,
        device_obj,
        loss_fn,
        credibility_buffer=train_credibility_buffer,
    )

    # Final-state held-out test evaluation. On the walk-forward path
    # this is the real test partition the manifest pins; on the legacy
    # 80/20 path no real held-out exists so the test loader is the val
    # loader's tensors -- the per-trial record's ``test_rmse`` then
    # equals the ``val_rmse``, which is exactly the pre-PR behaviour
    # the legacy regression contract pins.
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=min(batch_size, len(test_x)),
            shuffle=False,
            pin_memory=pin_memory,
            worker_init_fn=seed_worker if seed is not None else None,
        )
        test_credibility_buffer = _allocate_credibility_buffer(
            work_model, min(batch_size, len(test_x)), device_obj
        )
        test_metrics = _evaluate_model(
            forward_model,
            test_loader,
            device_obj,
            loss_fn,
            credibility_buffer=test_credibility_buffer,
        )
    else:
        test_metrics = best_val_metrics

    # ``metrics`` keeps the pre-PR semantics: the headline number the
    # downstream best-selection ranks by. On the walk-forward path
    # this is the held-out ``test_metrics``; on the legacy 80/20 path
    # this is ``best_val_metrics`` so the byte-identity regression on
    # ``test_forecaster_determinism.py`` stays green.
    headline_metrics = test_metrics if walk_forward_path else best_val_metrics

    summary = TrainingRunSummary(
        model_config=ModelConfig.from_model(work_model),
        device=str(device_obj),
        epochs_requested=epochs,
        epochs_completed=completed_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        validation_split=validation_split,
        early_stopping_patience=early_stopping_patience,
        sequence_groups=len(sequence_groups_for_summary),
        total_windows=len(train_x) + (len(val_x) if val_x is not None else 0) + (len(test_x) if test_x is not None else 0),
        train_windows=len(train_x),
        validation_windows=len(val_x) if val_x is not None else 0,
        checkpoint_path=str(checkpoint_target),
        checkpoint_saved=save_checkpoint,
        best_epoch=best_epoch,
        metrics=headline_metrics,
        train_metrics=train_metrics,
        val_metrics=best_val_metrics,
        test_metrics=test_metrics,
        fold_id=fold_id,
        protocol=active_protocol,
        weight_decay=float(weight_decay),
        target_mode="shuffled" if shuffle_targets_control else "real",
        text_encoder=text_encoder,
        text_adapter_dim=int(getattr(work_model, "text_adapter_dim", 0) or 0),
        text_pool_lambda_inv_days=float(text_pool_lambda_inv_days),
    )

    if save_checkpoint:
        from app.training.checkpoint import _save_model_checkpoint

        # `close_scale` was fitted on this fold's training rows in
        # `_build_training_tensors`; persisting it on the checkpoint is what
        # lets inference (`services.forecaster._predict_next_point`) recover
        # the original price magnitude. The two values must agree byte-for-
        # byte across save/load — see the determinism regression test.
        _save_model_checkpoint(
            work_model, checkpoint_target, summary, close_scale=close_scale
        )
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
