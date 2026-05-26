from __future__ import annotations

import contextlib
import copy
import dataclasses
import logging
import math
import warnings
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn
from torch.nn import functional as F
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
    SEQUENCE_LENGTH,
    FeatureVector,
    ModelConfig,
)
from app.models.lstm import ForecasterModel
from app.models.multimodal_forecaster import MultiModalForecasterModel
from app.training.loaders import (
    _build_text_embedding_tensors,
    _build_training_tensors,
    _split_train_validation,
    apply_rich_feature_scaler_tensor,
    collect_forward_vols,
    fit_class_weights,
    fit_rich_feature_scaler_tensor,
    fit_vol_regime_quantiles,
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


def _resolve_validation_fraction(
    validation_fraction: float | None,
    validation_split: float | None,
) -> float:
    """Coalesce the canonical and legacy validation-fraction kwargs.

    The internal codepath has historically called the val/train split
    knob ``validation_split``; the CLI has used ``--validation-fraction``
    since the v2 refactor. Issue #181 collapses the two onto the
    canonical name everywhere. Callers passing ``validation_split=``
    keep working but get a single ``DeprecationWarning`` per call so
    the rename can finish in a future PR without breaking downstream
    smoke tests in-flight.
    """

    if validation_fraction is not None and validation_split is not None:
        raise TypeError(
            "received both ``validation_fraction`` and ``validation_split``. "
            "Pass only ``validation_fraction``; ``validation_split`` is the "
            "deprecated alias."
        )
    if validation_split is not None:
        warnings.warn(
            "``validation_split`` is deprecated; use ``validation_fraction``. "
            "The val partition is a chronological prefix of the training "
            "slice, not a sklearn-style random split, and the new name "
            "reflects that.",
            DeprecationWarning,
            stacklevel=3,
        )
        return float(validation_split)
    if validation_fraction is None:
        return float(DEFAULT_VALIDATION_SPLIT)
    return float(validation_fraction)


from typing import overload


@overload
def _move_to_device(tensor: None, device: torch.device) -> None: ...
@overload
def _move_to_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor: ...


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
    # The factory may return MultiModalForecasterModel under
    # ``fusion_mode=gated_infonce`` (#235). Downstream consumers
    # (``_save_model_checkpoint``, ``_set_singleton_after_train``,
    # ``app.services.forecaster._load_state_dict_loose``) only touch
    # ``nn.Module`` APIs that both classes share, so the runtime
    # contract holds even though the static return type narrows
    # to ``ForecasterModel`` here. The narrower annotation keeps
    # those callers' signatures unchanged.
    return model  # type: ignore[return-value]


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


_MULTI_TASK_AUX_KEYS: tuple[str, ...] = (
    "factor",
    "factor_mask",
    "certainty",
    "certainty_mask",
    "topic",
    "topic_mask",
)


def _make_partition_dataset(
    x: torch.Tensor,
    y: torch.Tensor,
    text_emb: torch.Tensor | None,
    text_missing: torch.Tensor | None,
    mt_aux: dict[str, torch.Tensor] | None,
    log_rv: torch.Tensor | None = None,
) -> TensorDataset:
    """Pack one partition's tensors into a TensorDataset using a fixed contract.

    Supported arities, in order:

    - 2: ``(x, y)``
    - 3: ``(x, y, log_rv)`` -- #304 dual-head log(RV) target only
    - 4: ``(x, y, text_emb, text_missing)``
    - 5: text + log_rv combined
    - 8: ``(x, y, factor, factor_mask, certainty, certainty_mask, topic, topic_mask)``
    - 9: mt_aux + log_rv combined
    - 10: text + multi-task combined
    - 11: text + multi-task + log_rv combined

    The multi-task aux ordering is fixed by :data:`_MULTI_TASK_AUX_KEYS` so
    :func:`_unpack_batch` can recover the tensors positionally. The
    ``stance`` axis (a.k.a. the primary vol-regime target) is not packed
    here -- it lives in ``y`` and the train step rebuilds the
    ``stance_mask`` (all True) at the batch boundary. This drops the
    text-side ``stance`` field from ``_build_multi_task_target_tensors``
    because the model's stance head is already booked for the
    vol-regime classification target.

    The optional ``log_rv`` tensor (#304) carries the
    ``log(forward_realized_vol_10d)`` scalar target the regression head
    is trained against under ``head_mode`` in ``{regression, dual}``.
    The tensor sits at the end of the tuple regardless of whether the
    multi-task aux block precedes it so the contract is composable.
    """

    tensors: list[torch.Tensor] = [x, y]
    if text_emb is not None and text_missing is not None:
        tensors.extend([text_emb, text_missing])
    if mt_aux is not None:
        for key in _MULTI_TASK_AUX_KEYS:
            if key not in mt_aux:
                raise ValueError(
                    f"multi-task aux dict is missing required key {key!r}; "
                    f"got keys: {sorted(mt_aux)}"
                )
            tensors.append(mt_aux[key])
    if log_rv is not None:
        tensors.append(log_rv)
    return TensorDataset(*tensors)


def _unpack_batch(
    batch: Any,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    dict[str, torch.Tensor] | None,
    torch.Tensor | None,
]:
    """Decode a DataLoader batch into ``(x, y, text, text_missing, mt_aux, log_rv)``.

    Eight batch shapes are tolerated; see :func:`_make_partition_dataset`
    for the arity-to-contents map. ``mt_aux`` is a 6-key dict (factor,
    factor_mask, certainty, certainty_mask, topic, topic_mask) when the
    multi-task path is active and ``None`` otherwise. ``log_rv`` is the
    optional 1-D dual-head regression target tensor (#304); ``None``
    on classification-only runs.
    """

    arity = len(batch)
    if arity == 2:
        batch_x, batch_y = batch
        return batch_x, batch_y, None, None, None, None
    if arity == 3:
        batch_x, batch_y, batch_log_rv = batch
        return batch_x, batch_y, None, None, None, batch_log_rv
    if arity == 4:
        batch_x, batch_y, batch_text, batch_text_missing = batch
        return batch_x, batch_y, batch_text, batch_text_missing, None, None
    if arity == 5:
        batch_x, batch_y, batch_text, batch_text_missing, batch_log_rv = batch
        return batch_x, batch_y, batch_text, batch_text_missing, None, batch_log_rv
    if arity == 8:
        batch_x = batch[0]
        batch_y = batch[1]
        mt_aux = {key: batch[2 + idx] for idx, key in enumerate(_MULTI_TASK_AUX_KEYS)}
        return batch_x, batch_y, None, None, mt_aux, None
    if arity == 9:
        batch_x = batch[0]
        batch_y = batch[1]
        mt_aux = {key: batch[2 + idx] for idx, key in enumerate(_MULTI_TASK_AUX_KEYS)}
        return batch_x, batch_y, None, None, mt_aux, batch[8]
    if arity == 10:
        batch_x = batch[0]
        batch_y = batch[1]
        batch_text = batch[2]
        batch_text_missing = batch[3]
        mt_aux = {key: batch[4 + idx] for idx, key in enumerate(_MULTI_TASK_AUX_KEYS)}
        return batch_x, batch_y, batch_text, batch_text_missing, mt_aux, None
    if arity == 11:
        batch_x = batch[0]
        batch_y = batch[1]
        batch_text = batch[2]
        batch_text_missing = batch[3]
        mt_aux = {key: batch[4 + idx] for idx, key in enumerate(_MULTI_TASK_AUX_KEYS)}
        return batch_x, batch_y, batch_text, batch_text_missing, mt_aux, batch[10]
    raise ValueError(
        f"unexpected batch arity from DataLoader: {arity} "
        "(want 2, 3, 4, 5, 8, 9, 10, or 11)"
    )


def _maybe_write_classification_conformal_manifest(
    best_val_metrics: "EvaluationMetrics | None",
    checkpoint_target: Path,
) -> None:
    """Fit the APS threshold + persist a conformal sidecar on classification runs.

    Reads ``class_scores`` (per-row softmax) and ``targets`` (true class
    indices) off ``best_val_metrics`` — both are already collected by
    ``_evaluate_model`` on classification mode and ride on the
    EvaluationMetrics dataclass. When either is missing (regression-only
    runs, or a val partition that did not record row-level predictions)
    the helper is a no-op so the legacy regression path stays byte-
    identical.

    When a prior regression-side manifest already exists at the same
    sibling path, the function merges the new ``softmax_quantile`` into
    that file instead of overwriting it — a future joint regression +
    classification checkpoint can carry both quantiles in one manifest.
    """

    if best_val_metrics is None:
        return
    class_scores = getattr(best_val_metrics, "class_scores", None)
    targets = getattr(best_val_metrics, "targets", None)
    if class_scores is None or targets is None or not class_scores or not targets:
        return
    if len(class_scores) != len(targets):
        return

    from app.evaluation.conformal import (
        DEFAULT_CLASSIFICATION_ALPHA,
        ConformalManifest,
        calibrate_classification_conformal,
        load_manifest,
        save_manifest,
    )

    try:
        softmax_q = calibrate_classification_conformal(
            softmax_scores=class_scores,
            true_classes=targets,
            alpha=DEFAULT_CLASSIFICATION_ALPHA,
        )
    except ValueError as exc:
        print(f"[conformal] classification calibration skipped: {exc}", flush=True)
        return

    sidecar = Path(str(checkpoint_target.with_suffix("")) + ".conformal.json")
    if sidecar.exists():
        try:
            existing = load_manifest(sidecar)
            # Overwrite alpha + nominal_coverage with the classification
            # calibration values rather than keeping the existing
            # regression-side alpha (which can differ from
            # DEFAULT_CLASSIFICATION_ALPHA on a sidecar calibrated at a
            # different coverage target). The softmax_quantile fitted
            # in this run is paired with DEFAULT_CLASSIFICATION_ALPHA,
            # so the manifest as a whole must report that pair to keep
            # the inference path's coverage claim consistent with the
            # calibrated threshold.
            manifest = ConformalManifest(
                alpha=DEFAULT_CLASSIFICATION_ALPHA,
                nominal_coverage=1.0 - DEFAULT_CLASSIFICATION_ALPHA,
                residual_quantile_close=existing.residual_quantile_close,
                residual_quantile_volatility=existing.residual_quantile_volatility,
                calibration_n=existing.calibration_n,
                notes=existing.notes,
                softmax_quantile=softmax_q,
            )
        except Exception:
            # Stale / unreadable sidecar — overwrite with a classification-only
            # manifest. The regression bands fall back to gaussian_z on the
            # inference path when the residual_quantile fields are zero.
            manifest = ConformalManifest(
                alpha=DEFAULT_CLASSIFICATION_ALPHA,
                nominal_coverage=1.0 - DEFAULT_CLASSIFICATION_ALPHA,
                residual_quantile_close=0.0,
                residual_quantile_volatility=0.0,
                calibration_n=len(class_scores),
                softmax_quantile=softmax_q,
            )
    else:
        manifest = ConformalManifest(
            alpha=DEFAULT_CLASSIFICATION_ALPHA,
            nominal_coverage=1.0 - DEFAULT_CLASSIFICATION_ALPHA,
            residual_quantile_close=0.0,
            residual_quantile_volatility=0.0,
            calibration_n=len(class_scores),
            softmax_quantile=softmax_q,
        )
    save_manifest(manifest, sidecar)
    print(
        f"[conformal] calibrated softmax_quantile={softmax_q:.4f} "
        f"on n={len(class_scores)} val rows -> {sidecar.name}",
        flush=True,
    )


def _summarise_gate(
    gate_chunks: list[torch.Tensor],
    true_classes: torch.Tensor,
    n_classes: int,
) -> dict[str, Any] | None:
    """Reduce per-batch gate tensors into a per-fold diagnostic dict.

    Returns ``None`` when the eval pass collected no gate values
    (legacy single-modal path). The summary carries the scalar mean
    (>0.5 leans market, <0.5 leans text), per-class means so the
    thesis appendix can say whether the gate shifts modality
    reliance per regime, and the partition row count the summary
    was averaged over.
    """

    if not gate_chunks:
        return None
    gate = torch.cat(gate_chunks, dim=0)  # (N, latent_dim)
    n_rows = int(gate.size(0))
    if n_rows == 0:
        return None
    overall_mean = float(gate.mean().item())
    per_dim_mean: list[float] = [float(v) for v in gate.mean(dim=0).tolist()]
    per_class_mean: list[float | None] = [None] * n_classes
    if true_classes.numel() == n_rows:
        for class_idx in range(n_classes):
            mask = true_classes == class_idx
            count = int(mask.sum().item())
            if count > 0:
                per_class_mean[class_idx] = float(gate[mask].mean().item())
    return {
        "mean": overall_mean,
        "mean_per_class": per_class_mean,
        "mean_per_dim": per_dim_mean,
        "n_rows": n_rows,
    }


def _build_partition_multi_task_tensors(
    sequence_groups: "Sequence[Sequence[FeatureVector]]",
    *,
    vol_regime_quantiles: "Sequence[float]",
) -> dict[str, torch.Tensor] | None:
    """Materialise per-partition multi-task target + mask tensors (#273).

    Thin wrapper around
    :func:`app.training.loaders._build_multi_task_target_tensors` that
    matches the partition-tensor naming convention in this module.
    Used by the train loop to attach the 8 multi-task tensors
    (4 targets + 4 masks) to each partition's TensorDataset when
    ``multi_task_loss=True`` is set on the active ModelConfig.

    Returns ``None`` when the partition has no usable supervised rows.
    The underlying helper drops rows whose ``forward_realized_vol_10d``
    is missing — same row-filter the classification-mode
    ``_build_training_tensors`` applies, so the multi-task tensors
    stay row-aligned with ``y``.
    """

    from app.training.loaders import _build_multi_task_target_tensors

    return _build_multi_task_target_tensors(
        sequence_groups, vol_regime_quantiles=vol_regime_quantiles
    )


def _fit_axis_class_weights_from_mask(
    target_idx: torch.Tensor,
    mask: torch.Tensor,
    n_classes: int,
    *,
    smoothing: float = 1.0,
) -> torch.Tensor:
    """Inverse-frequency class weights fit on masked rows of one axis (#273).

    Mirrors :func:`app.training.loaders.fit_class_weights` for the
    multi-task path: each axis (stance, certainty, topic) computes its
    own class weights using only the rows where the axis mask is True.
    Smoothing keeps an empty class from blowing the weight up; the
    weights are normalised so they sum to ``n_classes``.

    Returns a length-``n_classes`` tensor (uniform 1.0 fallback when
    no rows are masked) so :class:`MultiTaskLoss` can construct a
    well-defined :class:`torch.nn.CrossEntropyLoss` for that axis.
    """

    if mask.numel() == 0 or not bool(mask.any().item()):
        return torch.ones(n_classes, dtype=torch.float32)
    masked = target_idx[mask].detach().to("cpu").long()
    counts = [0] * n_classes
    for v in masked.tolist():
        idx = int(v)
        if 0 <= idx < n_classes:
            counts[idx] += 1
    if sum(counts) == 0:
        return torch.ones(n_classes, dtype=torch.float32)
    raw = [1.0 / (c + smoothing) for c in counts]
    total = sum(raw)
    return torch.tensor(
        [(w / total) * n_classes for w in raw], dtype=torch.float32
    )


def _run_train_forward_multi_task(
    forward_model: nn.Module,
    batch_x: torch.Tensor,
    kwargs: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Run ``forward_multi_task`` on the underlying model.

    Unwraps DDP (``.module``) and ``torch.compile`` (``._orig_mod``)
    wrappers so the call lands on the real
    :meth:`ForecasterModel.forward_multi_task` (returns a
    gradient-tracked dict of per-axis logits). Compiled wrappers do
    not expose ``module`` — without the ``_orig_mod`` fallback the
    multi-task forward would silently run in eager mode even when
    ``--use-compile`` is on.
    """

    if hasattr(forward_model, "module"):
        underlying = forward_model.module
    else:
        underlying = getattr(forward_model, "_orig_mod", forward_model)
    forward_multi = getattr(underlying, "forward_multi_task", None)
    if forward_multi is None:
        raise RuntimeError(
            "multi_task_loss=True requires a model exposing "
            "forward_multi_task (built with output_mode='classification' "
            "+ MultiTaskHead); check the factory dispatch."
        )
    out: dict[str, torch.Tensor] = forward_multi(batch_x, **kwargs)
    return out


def _run_train_forward_and_align(
    forward_model: nn.Module,
    batch_x: torch.Tensor,
    kwargs: dict[str, torch.Tensor],
    *,
    info_nce_loss_fn: nn.Module | None,
    multimodal_active: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Forward + (optional) InfoNCE alignment in one helper.

    Returns ``(logits, align_loss_or_none)``. When ``multimodal_active``
    is True, calls ``forward_with_modality_outputs`` on the wrapped
    model and computes the InfoNCE term on ``(r_t, t_t)``; otherwise
    runs the legacy single-output forward and returns ``None`` for
    the alignment term so the caller can skip the addition.
    """

    if multimodal_active and info_nce_loss_fn is not None:
        underlying = forward_model.module if hasattr(forward_model, "module") else forward_model
        forward_with_modality = getattr(underlying, "forward_with_modality_outputs", None)
        if forward_with_modality is None:
            raise RuntimeError(
                "multimodal_active=True but the wrapped model does not expose "
                "forward_with_modality_outputs; check the factory dispatch."
            )
        out = forward_with_modality(batch_x, **kwargs)
        align_loss = info_nce_loss_fn(out["r_t"], out["t_t"])
        return out["logits"], align_loss
    return forward_model(batch_x, **kwargs), None


def _combine_dual_head_loss(
    *,
    ce_loss: torch.Tensor,
    logits_dict: dict[str, torch.Tensor],
    batch_log_rv: torch.Tensor | None,
    head_mode: str,
    regression_alpha: float,
) -> torch.Tensor:
    """Combine the classification CE with the #304 log(RV) MSE term.

    ``head_mode == "dual"`` returns ``(1 - alpha) * ce + alpha * mse``.
    Boundary alpha values short-circuit so the head whose weight is 0
    contributes neither loss nor gradient:

    - ``alpha == 0.0`` returns the CE unchanged (regression head is
      neither summed in nor required in ``logits_dict``).
    - ``alpha == 1.0`` returns the MSE alone -- equivalent to
      ``head_mode == "regression"`` at the loss level.

    ``head_mode == "regression"`` returns the MSE only -- the classifier
    head still runs forward so the checkpoint shape is unchanged, but
    its loss contribution is dropped so the experiment isolates
    regression-only learning. Any other ``head_mode`` (including the
    default ``classification``) returns the CE unchanged -- this branch
    is only reached when the train step explicitly opted in.
    """

    if head_mode not in {"regression", "dual"}:
        return ce_loss
    alpha = float(regression_alpha)
    # Dual + alpha=0 collapses to pure CE; do not require the log_rv
    # branch on the dict so the no-op boundary works byte-identically
    # to head_mode='classification'.
    if head_mode == "dual" and alpha <= 0.0:
        return ce_loss
    if "log_rv" not in logits_dict:
        raise RuntimeError(
            "head_mode requires the regression head but logits_dict has "
            "no 'log_rv' key; the model was built without "
            "head_mode in {regression, dual}."
        )
    if batch_log_rv is None:
        raise RuntimeError(
            "head_mode requires a log_rv target tensor but the batch "
            "carries None; the partition builder did not emit the "
            "dual-head target."
        )
    log_rv_pred = logits_dict["log_rv"]
    mse_loss = F.mse_loss(log_rv_pred, batch_log_rv.to(log_rv_pred.dtype))
    _assert_dual_head_scales_balanced(ce_loss, mse_loss)
    if head_mode == "regression":
        return mse_loss
    if alpha >= 1.0:
        return mse_loss
    return (1.0 - alpha) * ce_loss + alpha * mse_loss


def _maybe_add_dual_head_loss(
    loss: torch.Tensor,
    *,
    logits_dict: dict[str, torch.Tensor],
    batch_log_rv: torch.Tensor | None,
    head_mode: str,
    regression_alpha: float,
) -> torch.Tensor:
    """Augment an existing multi-task loss with the dual-head MSE.

    The multi-task path already pays the per-axis losses (stance CE +
    factor SmoothL1 + certainty CE + topic CE) inside
    :class:`MultiTaskLoss`; this helper preserves the full multi-task
    objective and adds the dual-head log(RV) MSE on top so the four
    axis classifiers keep learning under every head_mode.

    Per-mode behaviour:

    - ``head_mode == "classification"`` -- return ``loss`` unchanged.
    - ``head_mode == "dual"`` -- return ``(1 - alpha) * loss + alpha * mse``.
      ``alpha == 0`` collapses to ``loss`` so the dual + alpha=0
      boundary is byte-identical to classification at the loss level.
    - ``head_mode == "regression"`` -- return ``loss + mse`` so the
      multi-task auxiliary objectives keep training even when the
      stance head's classification view is the secondary surface. The
      regression head is the primary learning signal here; the
      per-axis losses still contribute their (smaller) gradient so the
      certainty / topic / factor heads continue to learn. The previous
      implementation discarded ``loss`` entirely, which silently
      stopped the four axis classifiers under multi-task + regression.

    ``logits_dict`` missing ``log_rv`` or ``batch_log_rv == None``
    short-circuits to ``loss`` to keep degenerate fixtures from
    raising; the partition builder is responsible for never letting
    that happen on real runs.
    """

    if head_mode not in {"regression", "dual"}:
        return loss
    if "log_rv" not in logits_dict or batch_log_rv is None:
        return loss
    log_rv_pred = logits_dict["log_rv"]
    mse_loss = F.mse_loss(log_rv_pred, batch_log_rv.to(log_rv_pred.dtype))
    _assert_dual_head_scales_balanced(loss, mse_loss)
    if head_mode == "regression":
        return loss + mse_loss
    alpha = float(regression_alpha)
    if alpha <= 0.0:
        return loss
    if alpha >= 1.0:
        return mse_loss
    return (1.0 - alpha) * loss + alpha * mse_loss


# One-shot diagnostic: warn (once per process) when the CE side and the
# MSE side of the dual-head loss are more than an order of magnitude
# apart. Standardising the log_rv target on the train slice (mean=0,
# std=1) should keep MSE in roughly the same range as CE (~log(n_classes)),
# so a large gap signals the standardiser was bypassed or the target
# tensor was built with non-finite values. The warning fires once and
# stays out of the hot loop.
_DUAL_HEAD_SCALE_WARNED: bool = False


def _assert_dual_head_scales_balanced(
    ce_loss: torch.Tensor, mse_loss: torch.Tensor
) -> None:
    """Log a one-shot warning when CE / MSE scales drift apart."""

    global _DUAL_HEAD_SCALE_WARNED
    if _DUAL_HEAD_SCALE_WARNED:
        return
    try:
        ce_value = float(ce_loss.detach().item())
        mse_value = float(mse_loss.detach().item())
    except RuntimeError:
        return
    if not (math.isfinite(ce_value) and math.isfinite(mse_value)):
        return
    if ce_value <= 0.0 or mse_value <= 0.0:
        return
    ratio = max(ce_value, mse_value) / max(min(ce_value, mse_value), 1e-12)
    if ratio > 10.0:
        _logger.warning(
            "[dual-head] CE / MSE scales out of balance on first batch: "
            "ce=%.4g mse=%.4g ratio=%.2fx. Verify the log_rv target was "
            "standardised on the train slice (mean=0, std=1).",
            ce_value,
            mse_value,
            ratio,
        )
    _DUAL_HEAD_SCALE_WARNED = True


# #309 derived-text-feature slices on the rich-feature tensor. Each
# slice covers one family of "text-derived" inputs the forecaster head
# would otherwise read. The ablation zeros every one of these slots so
# the only text-derived signal that survives is the document-level
# pooled encoder embedding (which lives off the rich-feature tensor and
# enters the model through ``text_adapter``).
#
# - [0]      : per-bar ``sentiment_score`` aggregate (ProsusAI).
# - [10:25]  : linguistic block (8 LDA topic shares + 6 hand-crafted
#              densities + pivot_distance).
# - [25:29]  : MP-surprise block (mp_surprise_level / mp_surprise_path_factor
#              / fed_info_factor / mp_is_intermeeting). The level and
#              path-factor components are derived off the FOMC text via
#              the gtfintechlab classifier upstream, so they belong to
#              the derived-text family for this ablation's purpose.
# - [29:35]  : multi-axis 6-slot (stance_hawk / stance_dove /
#              stance_neutral / time_label_forward /
#              certain_label_certain / stance_missing).
# - [45:80]  : 35-dim B1 LLM-extracted one-hot block.
_DERIVED_TEXT_SLICES: tuple[tuple[int, int], ...] = (
    (0, 1),
    (10, 25),
    (25, 29),
    (29, 35),
    (45, 80),
)


def _zero_derived_text_features(
    x: torch.Tensor | None,
    mt_aux: dict[str, torch.Tensor] | None,
) -> tuple[torch.Tensor | None, dict[str, torch.Tensor] | None]:
    """Zero the #309 derived-text-feature slots on a partition's tensors.

    Touches every slot the forecaster's rich-feature input carries that
    is downstream of the text (per-sentence sentiment aggregate, the
    linguistic / MP-surprise / multi-axis blocks, and the B1
    LLM-extracted one-hot block). The exact byte ranges are defined in
    :data:`_DERIVED_TEXT_SLICES`; per-bar position 0 (``sentiment_score``)
    is always zeroed regardless of whether the rich-feature payload was
    attached, so a legacy 6-feature tensor still loses the
    per-sentence sentiment input. Slices that fall beyond the tensor's
    last dim short-circuit (a 6-feature tensor only zeros [0]; a 35-dim
    rich tensor zeros up to [29:35]; an 80-dim rich tensor zeros every
    slice including the LLM block).

    The multi-task aux ``factor`` / ``certainty`` / ``topic`` masks are
    set to all-False so the auxiliary loss contribution from those axes
    drops to zero, matching the "derived features off" semantics on the
    multi-task supervision arm.

    The helper is called BEFORE the per-fold rich-feature RobustScaler
    is fit (see ``train_model``'s walk-forward branch) so the scaler's
    median for every zeroed slot lands at 0 and post-scaling the slot
    stays a literal 0. Val + test partitions are zeroed BEFORE the
    scaler is applied, then the same scaler is applied, so they too
    retain the literal-zero contract.
    """

    if x is not None and x.dim() == 3 and x.shape[-1] >= 1:
        x = x.clone()
        last_dim = x.shape[-1]
        for start, stop in _DERIVED_TEXT_SLICES:
            if start >= last_dim:
                continue
            effective_stop = min(stop, last_dim)
            if effective_stop <= start:
                continue
            x[..., start:effective_stop] = 0.0
    if mt_aux is not None:
        new_aux = dict(mt_aux)
        for axis in ("factor", "certainty", "topic"):
            mask_key = f"{axis}_mask"
            if mask_key in new_aux:
                new_aux[mask_key] = torch.zeros_like(new_aux[mask_key])
        mt_aux = new_aux
    return x, mt_aux


def _is_supervised_target_row(
    target_row: FeatureVector,
    quantiles: Sequence[float],
) -> bool:
    """Shared row-level filter for the partition-tensor builders.

    Returns ``True`` iff the row's ``forward_realized_vol_10d`` survives
    the same gate ``_build_training_tensors`` and
    ``_build_multi_task_target_tensors`` apply: present, NaN-free, and
    mappable to a regime class index under the fitted quantiles. Used
    by :func:`_build_partition_log_rv_target` so its row count tracks
    ``y`` exactly, and by the dual-head finite guard so a row with a
    pathological forward-vol value never enters the regression target.
    """

    from app.training.loaders import vol_regime_class_for

    forward_vol = getattr(target_row, "forward_realized_vol_10d", None)
    return vol_regime_class_for(forward_vol, quantiles) >= 0


def _is_finite_positive_forward_vol(value: float | None) -> bool:
    """Guard against inf / NaN / non-positive forward-vol values.

    A ``forward_realized_vol_10d`` row of ``inf`` passes the NaN gate
    (``inf != inf`` is False), and ``math.log(inf)`` blows the
    regression target's MSE up. Zero and negative values produce
    extreme outliers under ``log(...)`` that dominate the gradient on
    the first step (a single zero-vol row maps to ``log(eps) ≈ -18``
    when ``eps = 1e-8``). The dual-head builder rejects every such row
    so the partition's regression target stays well-behaved.
    """

    if value is None:
        return False
    if not isinstance(value, (int, float)):
        return False
    return math.isfinite(value) and value > 0.0


def _build_partition_log_rv_target(
    sequence_groups: "Sequence[Sequence[FeatureVector]]",
    *,
    vol_regime_quantiles: "Sequence[float]",
    log_rv_scaler: "tuple[float, float] | None" = None,
) -> tuple[torch.Tensor | None, "tuple[float, float] | None"]:
    """Materialise per-partition log(forward_realized_vol_10d) targets (#304).

    Returns ``(tensor, scaler)`` where ``tensor`` is a 1-D
    ``torch.float32`` carrying the standardised
    ``(log(forward_realized_vol_10d) - mean) / std`` row-aligned with
    the classification ``y`` tensor :func:`_build_partition_tensors`
    emits. ``scaler`` is ``(mean, std)`` when this call fitted the
    standardiser (train slice) or echoed back the caller's
    ``log_rv_scaler`` argument unchanged (val / test slices). Returns
    ``(None, None)`` when no rows survive the filter.

    Row alignment is enforced by walking the SAME filter
    :func:`_build_partition_tensors` does: the group-level pre-filter
    (drop a group whose leading target's ``forward_realized_vol_10d``
    is missing) PLUS the per-row filter
    :func:`_is_supervised_target_row` applies. Rows whose value is
    non-finite / non-positive are additionally rejected so the
    regression target never carries an inf-MSE outlier.

    Standardisation is fitted on the train slice only (no look-ahead).
    The CE / MSE scale-imbalance bug the joint-loss path used to hit
    (raw log_rv targets clustered around -4 with std ~0.5, giving
    initial MSE ~16 while CE ~log(3); alpha=0.5 left MSE owning ~93%
    of the gradient) drops to MSE ~1 once the targets sit in unit
    variance, so the alpha knob behaves like a true mixing weight.
    """

    from app.training.loaders import vol_regime_class_for

    values: list[float] = []
    rejected_non_finite = 0
    for sequence_group in sequence_groups:
        if len(sequence_group) < SEQUENCE_LENGTH + 1:
            continue
        # Group-level pre-filter mirrors ``_build_partition_tensors`` so
        # the row counts agree. The classification branch in
        # ``_build_partition_tensors`` drops a whole group when its
        # leading target (idx == SEQUENCE_LENGTH) has a null
        # forward_realized_vol_10d; iterating per-row without that
        # group-level gate would emit log_rv values for downstream
        # rows whose x / y counterparts were dropped, breaking the
        # TensorDataset row-count invariant.
        leading_target = sequence_group[SEQUENCE_LENGTH]
        leading_vol = getattr(leading_target, "forward_realized_vol_10d", None)
        if leading_vol is None or (
            isinstance(leading_vol, float) and leading_vol != leading_vol
        ):
            continue
        for idx in range(SEQUENCE_LENGTH, len(sequence_group)):
            target_row = sequence_group[idx]
            cls_idx = vol_regime_class_for(
                getattr(target_row, "forward_realized_vol_10d", None),
                vol_regime_quantiles,
            )
            if cls_idx < 0:
                continue
            forward_vol = getattr(target_row, "forward_realized_vol_10d", None)
            if not _is_finite_positive_forward_vol(forward_vol):
                rejected_non_finite += 1
                continue
            # ``_is_finite_positive_forward_vol`` narrowed the type
            # at runtime; mypy does not propagate the guard so the
            # explicit ``float(...)`` cast lands here.
            values.append(math.log(float(forward_vol)))  # type: ignore[arg-type]
    if not values:
        return None, log_rv_scaler

    raw_tensor = torch.tensor(values, dtype=torch.float32)
    if log_rv_scaler is None:
        # Train slice: fit the standardiser. Single-value partitions
        # would emit std=0 and a division by zero; floor std at 1e-6 to
        # keep the transform well-defined on degenerate fixtures.
        mean_val = float(raw_tensor.mean().item())
        std_val = float(raw_tensor.std(unbiased=False).item())
        if std_val < 1e-6:
            std_val = 1.0
        scaler_out = (mean_val, std_val)
    else:
        scaler_out = (float(log_rv_scaler[0]), float(log_rv_scaler[1]))
    mean_tensor = float(scaler_out[0])
    std_tensor = float(scaler_out[1])
    standardised = (raw_tensor - mean_tensor) / std_tensor
    if rejected_non_finite:
        _logger.warning(
            "[dual-head] rejected %d row(s) with non-finite or non-positive "
            "forward_realized_vol_10d from the log_rv regression target",
            rejected_non_finite,
        )
    return standardised, scaler_out


def _evaluate_model(
    model: nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
    loss_fn: nn.Module,
    credibility_buffer: torch.Tensor | None = None,
    *,
    record_row_predictions: bool = False,
    encoder_lora_bundle: Any = None,
    multi_task_loss_fn: nn.Module | None = None,
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

    ``multi_task_loss_fn`` (#273 follow-up): when provided, the eval
    dispatches to ``forward_multi_task`` on the model and computes the
    val loss via :class:`MultiTaskLoss` using the per-axis targets +
    masks carried on each batch's mt-aux block. The headline
    ``EvaluationMetrics`` surface (loss, regime_accuracy,
    regime_f1_macro, classification_breakdown) is computed off the
    ``stance`` logits so the dataclass shape stays identical to the
    single-task path. Pass ``None`` to keep the legacy single-task
    behaviour byte-identical.
    """

    model.eval()
    if encoder_lora_bundle is not None:
        encoder_lora_bundle.encoder.eval()
    is_classification = str(getattr(model, "output_mode", "regression")) == "classification"
    multi_task_active = multi_task_loss_fn is not None
    # #304 dual-head eval. Detect the regression head off the wrapped
    # model (un-DDP, un-compile) so the partition can emit
    # log(RV) RMSE / MAE without forcing the train loop to plumb the
    # head_mode flag through. When the head is absent the running sums
    # stay at zero and the helper emits ``None`` for the regression
    # surface, preserving the byte-identical EvaluationMetrics shape on
    # every pre-#304 caller.
    _eval_underlying = model.module if hasattr(model, "module") else model
    _eval_underlying = getattr(_eval_underlying, "_orig_mod", _eval_underlying)
    has_regression_head = (
        is_classification
        and getattr(_eval_underlying, "regression_head", None) is not None
    )
    total_loss_sum = torch.zeros((), dtype=torch.float64, device=device)
    total_items = torch.zeros((), dtype=torch.int64, device=device)
    log_rv_squared_error_sum = torch.zeros((), dtype=torch.float64, device=device)
    log_rv_abs_error_sum = torch.zeros((), dtype=torch.float64, device=device)
    log_rv_items = torch.zeros((), dtype=torch.int64, device=device)
    # Per-axis loss bookkeeping for the multi-task eval path (#273
    # follow-up). Each axis accumulates ``loss * batch_size`` so the
    # final mean matches the per-batch mean ``MultiTaskLoss`` emits
    # weighted by partition row count. Empty / zero on the single-task
    # path so the per-axis breakdown surfaces only when the eval was
    # actually run against MultiTaskLoss.
    mt_axis_loss_sums: dict[str, torch.Tensor] = {
        "stance": torch.zeros((), dtype=torch.float64, device=device),
        "factor": torch.zeros((), dtype=torch.float64, device=device),
        "certainty": torch.zeros((), dtype=torch.float64, device=device),
        "topic": torch.zeros((), dtype=torch.float64, device=device),
    }
    # Weighted CE bookkeeping. When ``loss_fn`` is
    # ``CrossEntropyLoss(weight=w, reduction='mean')`` the per-batch
    # loss is ``sum_i(w[y_i] * l_i) / sum_i(w[y_i])`` -- NOT divided by
    # batch_size. Multiplying the batch-mean loss by batch_size to get
    # the running total (the legacy regression-path arithmetic) over-
    # or under-weighs the val loss whenever the in-batch class mix
    # diverges from the corpus mean. The fix: accumulate
    # ``loss * weight_sum_in_batch`` against ``weight_total`` and
    # divide at the end. Falls back to ``batch_size`` when ``loss_fn``
    # has no ``weight`` attribute or weights are uniform, so the
    # regression byte-identity regression contract stays green.
    #
    # The multi-task eval path bypasses the CE-weight reweighting
    # entirely: ``MultiTaskLoss`` already emits a per-batch mean over
    # masked rows on each axis, so the partition mean is the size-
    # weighted mean of the per-batch values, identical to how the train
    # step aggregates the loss across batches.
    ce_weight: torch.Tensor | None = None
    weight_attr = getattr(loss_fn, "weight", None)
    if (
        is_classification
        and not multi_task_active
        and isinstance(weight_attr, torch.Tensor)
        and weight_attr.numel() > 0
    ):
        ce_weight = weight_attr.to(device=device, dtype=torch.float64)
    total_weight_sum = torch.zeros((), dtype=torch.float64, device=device)
    close_squared_error = torch.zeros((), dtype=torch.float64, device=device)
    volatility_squared_error = torch.zeros((), dtype=torch.float64, device=device)
    # Per-event arrays for the directional view. Empty until Phase 9
    # wired this in; downstream helper short-circuits on a zero-length
    # input so the regression-only legacy regression contract stays
    # byte-identical when these stay empty.
    pred_close_chunks: list[torch.Tensor] = []
    true_close_chunks: list[torch.Tensor] = []
    prev_close_chunks: list[torch.Tensor] = []
    # Phase 9 V2 (#195) classification view. Holds per-batch argmax
    # predictions + class targets so the post-loop helper can produce
    # top-1 accuracy + macro-F1 over the whole partition. Empty on
    # regression runs.
    pred_class_chunks: list[torch.Tensor] = []
    true_class_chunks: list[torch.Tensor] = []
    class_score_chunks: list[torch.Tensor] = []
    # Gated InfoNCE fusion (#235) diagnostic. When the model exposes
    # ``forward_with_modality_outputs`` the eval pass also captures
    # the per-row gate tensor so the post-loop summariser can attach
    # the gate distribution to the EvaluationMetrics. Empty on every
    # legacy single-modal path.
    multimodal_underlying = (
        model.module if hasattr(model, "module") else model
    )
    multimodal_forward = getattr(
        multimodal_underlying, "forward_with_modality_outputs", None
    )
    gate_chunks: list[torch.Tensor] = []
    use_text_path = bool(getattr(model, "_text_path_active", False)) or (
        multimodal_forward is not None
    )
    non_blocking = device.type == "cuda"
    with torch.no_grad():
        for batch in loader:
            (
                batch_x,
                batch_y,
                batch_text,
                batch_text_missing,
                batch_mt_aux,
                batch_log_rv,
            ) = _unpack_batch(batch)
            if multi_task_active and batch_mt_aux is None:
                raise RuntimeError(
                    "multi_task_loss_fn is active but the DataLoader yielded "
                    "a batch without the aux tensors. This usually means the "
                    "TensorDataset for this partition was built without the "
                    "multi-task aux block -- check the partition build path."
                )
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
                if encoder_lora_bundle is not None:
                    # Round 5 (#244): batch_text + batch_text_missing
                    # carry (input_ids, attention_mask) in LoRA mode.
                    # Run the LoRA-wrapped encoder over the tokens to
                    # materialise the pooled embedding the downstream
                    # text-adapter projection consumes. ``no_grad`` is
                    # active in this eval helper; train-loop forward
                    # has the same logic without the grad guard.
                    from app.training.encoder_lora import encode_batch_pooled

                    if batch_text_missing is not None and batch_text_missing.device != device:
                        batch_text_missing = batch_text_missing.to(
                            device, non_blocking=non_blocking
                        )
                    pooled, lora_missing = encode_batch_pooled(
                        encoder_lora_bundle,
                        batch_text,
                        batch_text_missing if batch_text_missing is not None
                        else torch.ones_like(batch_text, dtype=torch.long),
                    )
                    kwargs["text_embedding"] = pooled
                    kwargs["text_embedding_missing"] = lora_missing
                else:
                    kwargs["text_embedding"] = batch_text
                    if batch_text_missing is not None:
                        if batch_text_missing.device != device:
                            batch_text_missing = batch_text_missing.to(
                                device, non_blocking=non_blocking
                            )
                        kwargs["text_embedding_missing"] = batch_text_missing
            if multi_task_active:
                # Multi-task eval (#273 follow-up): dispatch to
                # ``forward_multi_task`` so the val loss matches the
                # train-side objective. The stance logits drive the
                # surfaced accuracy / F1 / breakdown so the
                # EvaluationMetrics shape stays identical to the
                # single-task path.
                logits_dict = _run_train_forward_multi_task(
                    model, batch_x, kwargs
                )
                assert batch_mt_aux is not None  # narrowed by the guard above
                assert multi_task_loss_fn is not None  # narrowed by ``multi_task_active``
                stance_mask = torch.ones(
                    (batch_size,), dtype=torch.bool, device=batch_x.device
                )
                mt_targets = {
                    "stance": batch_y,
                    "factor": batch_mt_aux["factor"].to(device, non_blocking=non_blocking),
                    "certainty": batch_mt_aux["certainty"].to(device, non_blocking=non_blocking),
                    "topic": batch_mt_aux["topic"].to(device, non_blocking=non_blocking),
                }
                mt_masks = {
                    "stance_mask": stance_mask,
                    "factor_mask": batch_mt_aux["factor_mask"].to(device, non_blocking=non_blocking),
                    "certainty_mask": batch_mt_aux["certainty_mask"].to(device, non_blocking=non_blocking),
                    "topic_mask": batch_mt_aux["topic_mask"].to(device, non_blocking=non_blocking),
                }
                loss, axis_breakdown = multi_task_loss_fn(
                    logits_dict, mt_targets, mt_masks
                )
                predictions = logits_dict["stance"]
                total_loss_sum += loss.detach().to(torch.float64) * batch_size
                for axis_name in ("stance", "factor", "certainty", "topic"):
                    mt_axis_loss_sums[axis_name] += (
                        axis_breakdown[axis_name].detach().to(torch.float64) * batch_size
                    )
                # #304 dual-head eval. When the regression head is also
                # mounted under multi-task training, surface the
                # log(RV) MAE / RMSE alongside the classification
                # numbers.
                if has_regression_head and "log_rv" in logits_dict and batch_log_rv is not None:
                    log_rv_pred = logits_dict["log_rv"].detach().to(torch.float64)
                    log_rv_true = batch_log_rv.to(device, non_blocking=non_blocking).to(torch.float64)
                    diff = log_rv_pred - log_rv_true
                    log_rv_squared_error_sum += torch.square(diff).sum()
                    log_rv_abs_error_sum += torch.abs(diff).sum()
                    log_rv_items += int(diff.shape[0])
            elif has_regression_head:
                # #304 single-task dual-head eval. ``forward_multi_task``
                # is the only path that emits the ``log_rv`` head, so
                # route through it whenever the regression head is
                # mounted; the headline classification surface still
                # reads off the stance logits so the eval contract on
                # legacy callers stays compatible.
                logits_dict = _run_train_forward_multi_task(
                    model, batch_x, kwargs
                )
                predictions = logits_dict["stance"]
                loss = loss_fn(predictions, batch_y)
                if ce_weight is not None:
                    batch_weight_sum = ce_weight.index_select(
                        0, batch_y.detach().to(device=device, dtype=torch.long)
                    ).sum()
                    total_loss_sum += loss.detach().to(torch.float64) * batch_weight_sum
                    total_weight_sum += batch_weight_sum
                else:
                    total_loss_sum += loss.detach().to(torch.float64) * batch_size
                if "log_rv" in logits_dict and batch_log_rv is not None:
                    log_rv_pred = logits_dict["log_rv"].detach().to(torch.float64)
                    log_rv_true = batch_log_rv.to(device, non_blocking=non_blocking).to(torch.float64)
                    diff = log_rv_pred - log_rv_true
                    log_rv_squared_error_sum += torch.square(diff).sum()
                    log_rv_abs_error_sum += torch.abs(diff).sum()
                    log_rv_items += int(diff.shape[0])
            elif multimodal_forward is not None:
                modality_out = multimodal_forward(batch_x, **kwargs)
                predictions = modality_out["logits"]
                # Capture gate on CPU in float32 so the per-partition
                # accumulator stays bounded across long val/test splits.
                gate_chunks.append(modality_out["gate"].detach().to("cpu", torch.float32))
                loss = loss_fn(predictions, batch_y)
                if ce_weight is not None:
                    batch_weight_sum = ce_weight.index_select(
                        0, batch_y.detach().to(device=device, dtype=torch.long)
                    ).sum()
                    total_loss_sum += loss.detach().to(torch.float64) * batch_weight_sum
                    total_weight_sum += batch_weight_sum
                else:
                    total_loss_sum += loss.detach().to(torch.float64) * batch_size
            else:
                predictions = model(batch_x, **kwargs)
                loss = loss_fn(predictions, batch_y)
                if ce_weight is not None:
                    batch_weight_sum = ce_weight.index_select(
                        0, batch_y.detach().to(device=device, dtype=torch.long)
                    ).sum()
                    total_loss_sum += loss.detach().to(torch.float64) * batch_weight_sum
                    total_weight_sum += batch_weight_sum
                else:
                    total_loss_sum += loss.detach().to(torch.float64) * batch_size
            total_items += batch_size
            if is_classification:
                pred_class_chunks.append(
                    predictions.argmax(dim=1).detach().to("cpu", torch.long)
                )
                true_class_chunks.append(batch_y.detach().to("cpu", torch.long))
                # Per-class softmax probabilities ride alongside the
                # argmax so the breakdown helper can compute one-vs-rest
                # ROC-AUC + PR-AUC at the end of the loop. Keep on CPU
                # in float32 to bound memory on long partitions.
                class_score_chunks.append(
                    torch.softmax(predictions, dim=1).detach().to("cpu", torch.float32)
                )
            else:
                delta = predictions - batch_y
                close_squared_error += torch.square(delta[:, 0]).sum().to(torch.float64)
                volatility_squared_error += torch.square(delta[:, 1]).sum().to(torch.float64)
                # Collect the close-axis arrays for the directional view.
                # Eval partitions are small (val ~60, test ~60-70 windows
                # per fold), so the per-event copy is cheap. ``batch_x[:, -1, 1]``
                # is the prev-bar's close in scaled units; the model's
                # output and the target are in the same scaled units so
                # ``sign(pred - prev) == sign(true - prev)`` is the
                # directional ground truth.
                pred_close_chunks.append(predictions[:, 0].detach().to("cpu", torch.float32))
                true_close_chunks.append(batch_y[:, 0].detach().to("cpu", torch.float32))
                prev_close_chunks.append(batch_x[:, -1, 1].detach().to("cpu", torch.float32))
    total_items_int = int(total_items.item())
    if total_items_int <= 0:
        return EvaluationMetrics(
            loss=float("inf"),
            close_rmse=float("inf"),
            volatility_rmse=float("inf"),
            combined_rmse=float("inf"),
        )

    total_loss_value = float(total_loss_sum.item())
    # Weighted CE: the mean is ``sum_b (loss_b * weight_sum_b) / total_weight_sum``,
    # not ``sum_b (loss_b * batch_size) / total_batch_size``. Falls back
    # to the per-item count when no class weights were supplied.
    total_weight_value = float(total_weight_sum.item()) if ce_weight is not None else 0.0
    loss_divisor = total_weight_value if (ce_weight is not None and total_weight_value > 0.0) else float(total_items_int)
    # Per-axis multi-task breakdown (#273 follow-up). Computed only on
    # the multi-task eval path; the single-task path leaves the dict
    # empty so the existing classification_breakdown payload shape stays
    # unchanged on legacy runs.
    multi_task_axis_losses: dict[str, float] | None = None
    if multi_task_active and total_items_int > 0:
        multi_task_axis_losses = {
            axis: float(mt_axis_loss_sums[axis].item()) / float(total_items_int)
            for axis in ("stance", "factor", "certainty", "topic")
        }

    if is_classification:
        # Classification partition: surface accuracy + macro-F1 on the
        # regime axis, leave the regression columns at +inf so legacy
        # consumers that read them notice immediately rather than seeing
        # zeros that look like a perfect regression fit. The full
        # breakdown (confusion matrix + per-class P/R/F1 + one-vs-rest
        # AUC) lives on ``EvaluationMetrics.classification_breakdown``.
        from app.evaluation.classification_breakdown import (
            compute_classification_breakdown,
        )

        pred_classes = torch.cat(pred_class_chunks) if pred_class_chunks else torch.empty(0, dtype=torch.long)
        true_classes = torch.cat(true_class_chunks) if true_class_chunks else torch.empty(0, dtype=torch.long)
        class_scores_tensor = (
            torch.cat(class_score_chunks) if class_score_chunks else None
        )
        n_classes_eval = int(getattr(model, "n_classes", 3) or 3)
        regime_acc = (
            float((pred_classes == true_classes).float().mean().item())
            if pred_classes.numel()
            else 0.0
        )
        regime_loss = total_loss_value / loss_divisor

        class_scores_list: list[list[float]] | None = None
        if class_scores_tensor is not None and class_scores_tensor.numel():
            class_scores_list = class_scores_tensor.tolist()

        breakdown = compute_classification_breakdown(
            predictions=pred_classes.tolist(),
            targets=true_classes.tolist(),
            n_classes=n_classes_eval,
            class_scores=class_scores_list,
        )
        breakdown_payload = breakdown.to_dict()
        # Attach the per-axis multi-task loss breakdown so the per-trial
        # JSON / sweep aggregator can surface it; the headline
        # ``regime_loss`` keeps reporting the lambda-weighted total so
        # checkpoint selection by val_loss ranks against the same
        # objective the train side optimises.
        if multi_task_axis_losses is not None:
            breakdown_payload["multi_task_axis_losses"] = multi_task_axis_losses

        predictions_payload: list[int] | None = None
        targets_payload: list[int] | None = None
        scores_payload: list[list[float]] | None = None
        if record_row_predictions:
            predictions_payload = [int(x) for x in pred_classes.tolist()]
            targets_payload = [int(x) for x in true_classes.tolist()]
            if class_scores_list is not None:
                scores_payload = [
                    [float(p) for p in row] for row in class_scores_list
                ]

        gate_summary = _summarise_gate(gate_chunks, true_classes, n_classes_eval)

        # #304 dual-head eval surface. When the regression head ran on
        # this partition, surface the log(RV) RMSE / MAE so the §16
        # three-way comparison table can read them off the per-trial
        # JSON. ``None`` when the head was absent so the legacy
        # EvaluationMetrics shape is unchanged on every pre-#304 caller.
        log_rv_items_int = int(log_rv_items.item())
        regression_rmse_log_rv_value: float | None = None
        regression_mae_log_rv_value: float | None = None
        regression_loss_value: float | None = None
        if has_regression_head and log_rv_items_int > 0:
            regression_loss_value = float(
                log_rv_squared_error_sum.item() / log_rv_items_int
            )
            regression_rmse_log_rv_value = math.sqrt(regression_loss_value)
            regression_mae_log_rv_value = float(
                log_rv_abs_error_sum.item() / log_rv_items_int
            )

        return EvaluationMetrics(
            loss=regime_loss,
            close_rmse=float("inf"),
            volatility_rmse=float("inf"),
            combined_rmse=float("inf"),
            regime_accuracy=regime_acc,
            regime_f1_macro=float(breakdown.macro_f1),
            regime_loss=regime_loss,
            classification_breakdown=breakdown_payload,
            predictions=predictions_payload,
            targets=targets_payload,
            class_scores=scores_payload,
            gate_summary=gate_summary,
            regression_rmse_log_rv=regression_rmse_log_rv_value,
            regression_mae_log_rv=regression_mae_log_rv_value,
            regression_loss=regression_loss_value,
        )

    close_value = float(close_squared_error.item())
    volatility_value = float(volatility_squared_error.item())
    combined_squared_error = close_value + volatility_value

    # Directional view (Phase 9). The helper returns None on every
    # axis when no events were collected, so the dataclass field
    # stays None and the regression contract is unchanged for callers
    # that ignore the new metrics.
    directional: dict[str, float | None] = {
        "direction_accuracy": None,
        "f1_macro": None,
        "direction_auc": None,
    }
    if pred_close_chunks:
        from app.evaluation.directional_metrics import compute_directional_metrics

        directional = compute_directional_metrics(
            torch.cat(pred_close_chunks),
            torch.cat(true_close_chunks),
            torch.cat(prev_close_chunks),
        )

    return EvaluationMetrics(
        loss=total_loss_value / loss_divisor,
        close_rmse=math.sqrt(close_value / total_items_int),
        volatility_rmse=math.sqrt(volatility_value / total_items_int),
        combined_rmse=math.sqrt(combined_squared_error / (total_items_int * 2)),
        direction_accuracy=directional["direction_accuracy"],
        f1_macro=directional["f1_macro"],
        direction_auc=directional["direction_auc"],
    )


def _build_partition_tensors(
    sequence_groups: Sequence[Sequence[FeatureVector]],
    *,
    fallback_text_in_dim: int,
    close_scale: float | None = None,
    output_mode: str = "regression",
    vol_regime_quantiles: Sequence[float] = (),
    lora_bundle: Any = None,
    lora_max_tokens: int = 0,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    float,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """Tensorise one partition into (x, y, close_scale, text_emb, text_missing).

    Regression mode (``output_mode="regression"``) preserves the
    byte-identity contract: ``y`` is the (N, 2) float tensor of
    (close / close_scale, max(vol, 0)) and the text tensors align
    one-to-one with the x rows.

    Classification mode (``output_mode="classification"``) materialises
    ``y`` as a 1-D Long tensor of class indices computed via the
    per-fold ``vol_regime_quantiles`` cutoffs. Groups whose target row
    has a null ``forward_realized_vol_10d`` are dropped from BOTH the
    x/y tensors and the text-embedding tensors so the row alignment
    invariant holds.

    Round 5 (#244) LoRA branch: when ``lora_bundle`` is supplied (a
    :class:`app.training.encoder_lora.LoraEncoderBundle`), the last
    two return slots carry ``(input_ids, attention_mask)`` long
    tensors (shape ``(N, lora_max_tokens)``) instead of the
    pooled-embedding pair. The train step detects LoRA mode by
    inspecting the dtype + running the bundle's encoder over the
    tokens per batch to materialise gradient-tracked pooled vectors.

    Multi-task aux tensors (#273) are NOT returned by this function —
    the caller computes them separately via
    :func:`_build_partition_multi_task_tensors` on the same partition
    groups so the 5-tuple contract here stays stable for every
    existing caller (scripts/calibrate_regime_classifier.py,
    tests/unit/test_phase9_partition_tensors.py, etc.).
    """

    if output_mode == "classification":
        # Pre-filter groups whose target row has an unusable forward-vol
        # column. The training-tensor builder + text-embedding builder
        # then both operate on the same filtered list and emit
        # row-aligned tensors.
        filtered: list[list[FeatureVector]] = []
        for group in sequence_groups:
            if len(group) < SEQUENCE_LENGTH + 1:
                continue
            target_value = getattr(
                group[SEQUENCE_LENGTH], "forward_realized_vol_10d", None
            )
            if target_value is None:
                continue
            if target_value != target_value:  # NaN
                continue
            filtered.append(list(group))
        active_groups: Sequence[Sequence[FeatureVector]] = filtered
    else:
        active_groups = sequence_groups

    x, y, scale = _build_training_tensors(
        active_groups,
        close_scale=close_scale,
        output_mode=output_mode,
        vol_regime_quantiles=vol_regime_quantiles,
    )
    if x is None or y is None:
        # ``_build_training_tensors`` returns None tensors only when the
        # partition has zero usable windows. The walk-forward path
        # guarantees every partition has at least one event so the
        # caller can rely on a non-None return; this assertion-style
        # raise pushes the failure mode to the data-prep layer where
        # it belongs and unlocks the rest of the training loop from
        # ``Tensor | None`` typing.
        raise ValueError(
            "_build_partition_tensors produced an empty partition; "
            "every walk-forward fold must carry at least one event."
        )
    if lora_bundle is not None:
        # Round 5 (#244): tokenise each sequence's target-row text once
        # and return (input_ids, attention_mask) in the text slots.
        # The training step runs the LoRA-wrapped encoder over these
        # tokens per batch to compute a gradient-tracked pooled vector.
        from app.training.encoder_lora import tokenize_sequence_texts

        max_tokens = int(lora_max_tokens) if int(lora_max_tokens) > 0 else 256
        input_ids, attention_mask = tokenize_sequence_texts(
            active_groups,
            lora_bundle.tokenizer,
            max_tokens=max_tokens,
        )
        return x, y, scale, input_ids, attention_mask
    text_emb, text_missing, _ = _build_text_embedding_tensors(
        active_groups, fallback_in_dim=fallback_text_in_dim
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
    validation_fraction: float | None = None,
    validation_split: float | None = None,
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
    lr_schedule: str = "plateau",
    use_class_weights: bool = True,
) -> TrainingResult:
    # ``validation_split`` is the legacy kwarg name; ``validation_fraction``
    # is the canonical one across the CLI, the training loop, and the
    # downstream consumers. Both are accepted on this signature so the
    # public API does not break callers that still pass the old name;
    # the deprecation warning fires only when the legacy kwarg is the
    # one being used (positional ambiguity is impossible because both
    # are keyword-only). See issue #181 for the broader rename pass.
    validation_split = _resolve_validation_fraction(
        validation_fraction, validation_split
    )
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
    # Multi-task aux tensors (#273) — populated by the walk-forward branch
    # when ``multi_task_loss=True``; the legacy 80/20 branch leaves these
    # at None so the regression contract on the determinism test stays
    # byte-identical.
    train_mt_aux: dict[str, torch.Tensor] | None = None
    val_mt_aux: dict[str, torch.Tensor] | None = None
    test_mt_aux: dict[str, torch.Tensor] | None = None
    multi_task_loss_active = bool(
        getattr(active_model_config, "multi_task_loss", False)
    )
    # #304 dual-head methodology. The log(RV) target tensor lives on
    # each partition only when ``head_mode in {regression, dual}``;
    # default ``classification`` leaves the three slots ``None`` so
    # the dataset arity stays unchanged for every pre-#304 caller.
    train_log_rv: torch.Tensor | None = None
    val_log_rv: torch.Tensor | None = None
    test_log_rv: torch.Tensor | None = None
    # #304 dual-head: per-fold log_rv standardiser fitted on the train
    # slice only. Persisted onto the run summary so downstream consumers
    # can invert the standardised regression head output.
    log_rv_scaler: tuple[float, float] | None = None
    active_head_mode = str(
        getattr(active_model_config, "head_mode", "classification")
        or "classification"
    )
    dual_head_active = active_head_mode in {"regression", "dual"}

    if walk_forward_path:
        train_groups: list[list[FeatureVector]] = [list(group) for group in train_sequence_groups or []]
        val_groups: list[list[FeatureVector]] = [list(group) for group in val_sequence_groups or []]
        test_groups: list[list[FeatureVector]] = [list(group) for group in test_sequence_groups or []]
        # Phase 9 V2 (#195) per-fold quantile fit. In classification
        # mode we fit (n_classes-1) interior cutoffs on the train slice
        # only so val + test see the same boundaries the optimiser saw.
        # The cutoffs persist onto ``active_model_config`` (and from
        # there into the saved checkpoint) so inference + eval apply
        # the identical mapping. In regression mode the call is skipped
        # and the cutoff tuple stays empty.
        active_output_mode = str(
            getattr(active_model_config, "output_mode", "regression") or "regression"
        )
        if active_output_mode == "classification":
            n_classes_active = int(getattr(active_model_config, "n_classes", 3) or 3)
            train_forward_vols = collect_forward_vols(train_groups)
            fitted_quantiles = fit_vol_regime_quantiles(
                train_forward_vols, n_classes=n_classes_active
            )
            if not fitted_quantiles:
                raise ValueError(
                    "vol-regime classification requires >= n_classes valid "
                    "forward_realized_vol_10d targets on the train slice; "
                    f"got {len(train_forward_vols)} valid rows for "
                    f"n_classes={n_classes_active}."
                )
            active_model_config = dataclasses.replace(
                active_model_config, vol_regime_quantiles=fitted_quantiles
            )
            # A1 (#206) per-fold class weighting. Counts each class in
            # the train slice under the just-fitted quantile cutoffs,
            # then builds inverse-frequency weights so the loss path
            # of least resistance is no longer "predict the majority
            # prior". Train-only fit; val + test see the same weights
            # but only at loss computation, not in their own slices.
            # Round 2c (#234) ablation: ``use_class_weights=False``
            # skips the fit and leaves the weight tuple empty, so the
            # downstream ``CrossEntropyLoss(weight=None)`` path runs
            # for direct A1-on-vs-A1-off comparison.
            if use_class_weights:
                fitted_class_weights = fit_class_weights(
                    train_forward_vols,
                    fitted_quantiles,
                    n_classes=n_classes_active,
                    power=float(getattr(active_model_config, "class_weight_power", 1.0)),
                )
            else:
                fitted_class_weights = ()
        else:
            fitted_quantiles = ()
            fitted_class_weights = ()
        # Round 5 (#244) LoRA bundle. Built ONCE before any partition
        # tensor materialisation so the tokeniser is shared across
        # train / val / test. When ``encoder_lora`` is off (default)
        # the bundle stays ``None`` and the static-cache path runs.
        encoder_lora_active = bool(
            getattr(active_model_config, "encoder_lora", False)
        )
        encoder_lora_bundle: Any = None
        if encoder_lora_active:
            from app.training.encoder_lora import build_lora_encoder

            if not text_encoder or str(text_encoder) == "none":
                raise ValueError(
                    "encoder_lora=True requires text_encoder to be set to a "
                    "registered alias (got 'none' / empty)"
                )
            encoder_lora_bundle = build_lora_encoder(str(text_encoder))
            encoder_lora_bundle.encoder.to(device_obj)
            # Stdout breadcrumb so a grep of the run log can confirm
            # LoRA actually activated on this cell — the persisted
            # summary only records the post-train state and the model
            # factory previously dropped the flag (see PR fixing the
            # encoder_lora persistence bug).
            print(
                f"[train_model] encoder_lora active: alias={text_encoder} "
                f"out_dim={encoder_lora_bundle.out_dim}",
                flush=True,
            )
        # Fit the close-scale on the training partition only; never on
        # the val or test rows. The walk-forward protocol forbids
        # fitting any scaler over held-out events.
        if multi_task_loss_active and active_output_mode != "classification":
            raise ValueError(
                "multi_task_loss=True requires output_mode='classification'; "
                f"got output_mode={active_output_mode!r}"
            )
        train_x, train_y, close_scale, train_text_emb, train_text_missing = _build_partition_tensors(
            train_groups,
            fallback_text_in_dim=fallback_text_in_dim,
            close_scale=None,
            output_mode=active_output_mode,
            vol_regime_quantiles=fitted_quantiles,
            lora_bundle=encoder_lora_bundle,
        )
        # Multi-task aux tensors (#273) — sibling call so the partition
        # tensorisation contract on _build_partition_tensors stays a
        # stable 5-tuple for the calibrate script + the determinism test.
        # The aux builder filters rows by the same vol_regime_class_for
        # predicate _build_training_tensors applies in classification
        # mode, so the row order aligns with train_x / train_y.
        if multi_task_loss_active:
            train_mt_aux = _build_partition_multi_task_tensors(
                train_groups, vol_regime_quantiles=fitted_quantiles
            )
        if dual_head_active and active_output_mode == "classification":
            train_log_rv, log_rv_scaler = _build_partition_log_rv_target(
                train_groups, vol_regime_quantiles=fitted_quantiles
            )
        # #309 derived-text-features ablation runs BEFORE the per-fold
        # rich-feature RobustScaler so the scaler's median for every
        # zeroed slot lands at 0 and the post-scaling slot stays a
        # literal 0. Doing this AFTER the scaler would subtract the
        # populated-distribution median from 0 and leave a non-zero
        # entry in scaled units, defeating the ablation contract.
        # The legacy 80/20 branch below mirrors the same ordering.
        if not bool(getattr(active_model_config, "use_derived_text_features", True)):
            train_x, train_mt_aux = _zero_derived_text_features(train_x, train_mt_aux)  # type: ignore[assignment]
        # Fit the rich-feature RobustScaler on the TRAIN tensor only;
        # no-op for legacy 6-feature tensors so the regression contract
        # at tests/regression/test_forecaster_determinism.py stays
        # byte-identical. Apply to every partition with the same
        # parameters so val + test see the train-time normalisation.
        rich_feature_scaler = fit_rich_feature_scaler_tensor(train_x)
        train_x = apply_rich_feature_scaler_tensor(train_x, rich_feature_scaler)
        val_x, val_y, _val_scale, val_text_emb, val_text_missing = _build_partition_tensors(
            val_groups,
            fallback_text_in_dim=fallback_text_in_dim,
            close_scale=close_scale,
            output_mode=active_output_mode,
            vol_regime_quantiles=fitted_quantiles,
            lora_bundle=encoder_lora_bundle,
        )
        if multi_task_loss_active:
            val_mt_aux = _build_partition_multi_task_tensors(
                val_groups, vol_regime_quantiles=fitted_quantiles
            )
        if dual_head_active and active_output_mode == "classification":
            val_log_rv, _ = _build_partition_log_rv_target(
                val_groups,
                vol_regime_quantiles=fitted_quantiles,
                log_rv_scaler=log_rv_scaler,
            )
        if not bool(getattr(active_model_config, "use_derived_text_features", True)):
            val_x, val_mt_aux = _zero_derived_text_features(val_x, val_mt_aux)  # type: ignore[assignment]
        val_x = apply_rich_feature_scaler_tensor(val_x, rich_feature_scaler)
        test_x, test_y, _test_scale, test_text_emb, test_text_missing = _build_partition_tensors(
            test_groups,
            fallback_text_in_dim=fallback_text_in_dim,
            close_scale=close_scale,
            output_mode=active_output_mode,
            vol_regime_quantiles=fitted_quantiles,
            lora_bundle=encoder_lora_bundle,
        )
        if multi_task_loss_active:
            test_mt_aux = _build_partition_multi_task_tensors(
                test_groups, vol_regime_quantiles=fitted_quantiles
            )
        if dual_head_active and active_output_mode == "classification":
            test_log_rv, _ = _build_partition_log_rv_target(
                test_groups,
                vol_regime_quantiles=fitted_quantiles,
                log_rv_scaler=log_rv_scaler,
            )
        if not bool(getattr(active_model_config, "use_derived_text_features", True)):
            test_x, test_mt_aux = _zero_derived_text_features(test_x, test_mt_aux)  # type: ignore[assignment]
            print(
                "[train_model] derived-text-features OFF: zeroed slices "
                "[0], [10:25], [25:29], [29:35], [45:80] on x before "
                "scaler fit; masked factor/certainty/topic on mt_aux",
                flush=True,
            )
        test_x = apply_rich_feature_scaler_tensor(test_x, rich_feature_scaler)
        sequence_groups_for_summary = train_groups + val_groups + test_groups
    else:
        # Legacy single-list path: no LoRA support. encoder_lora must
        # be off in this branch -- the per-batch encoder forward needs
        # the walk-forward partition tensor builder to emit token
        # tensors, which the legacy path bypasses entirely.
        encoder_lora_bundle = None
        if bool(getattr(active_model_config, "encoder_lora", False)):
            raise ValueError(
                "encoder_lora=True is only supported on the walk-forward "
                "training-package path; the legacy single-list ``data_dir`` "
                "branch does not produce the token tensors LoRA needs"
            )
        if multi_task_loss_active:
            raise ValueError(
                "multi_task_loss=True is only supported on the walk-forward "
                "training-package path; the legacy single-list ``data_dir`` "
                "branch does not materialise the per-axis target tensors"
            )
        if sequence_groups is not None:
            active_sequence_groups: list[list[FeatureVector]] = [list(group) for group in sequence_groups]
        else:
            active_sequence_groups = load_training_sequences_from_data(data_dir)
        if vectors:
            active_sequence_groups.append(list(vectors))
        sequence_groups_for_summary = active_sequence_groups

        # Legacy 80/20 path: when the caller has switched to
        # ``output_mode='classification'``, fit the per-fold vol-regime
        # quantiles on the active single-list so the classification
        # ``y`` builder + the #304 dual-head log_rv builder both see
        # the same cutoffs. The pre-#304 regression path leaves the
        # tuple empty and ``_build_training_tensors`` ignores it.
        active_output_mode = str(
            getattr(active_model_config, "output_mode", "regression") or "regression"
        )
        legacy_fitted_quantiles: tuple[float, ...] = ()
        if active_output_mode == "classification":
            n_classes_active = int(getattr(active_model_config, "n_classes", 3) or 3)
            legacy_forward_vols = collect_forward_vols(active_sequence_groups)
            legacy_fitted_quantiles = fit_vol_regime_quantiles(
                legacy_forward_vols, n_classes=n_classes_active
            )
            if not legacy_fitted_quantiles:
                raise ValueError(
                    "vol-regime classification requires >= n_classes valid "
                    "forward_realized_vol_10d targets on the legacy single-list "
                    f"path; got {len(legacy_forward_vols)} valid rows for "
                    f"n_classes={n_classes_active}."
                )
            active_model_config = dataclasses.replace(
                active_model_config, vol_regime_quantiles=legacy_fitted_quantiles
            )
        x, y, close_scale = _build_training_tensors(
            active_sequence_groups,
            output_mode=active_output_mode,
            vol_regime_quantiles=legacy_fitted_quantiles,
        )
        # #304 dual-head on the legacy path. Build the log_rv target
        # tensor over the full active_sequence_groups list, then split
        # below alongside (x, y) so the row alignment invariant holds.
        legacy_full_log_rv: torch.Tensor | None = None
        if dual_head_active and active_output_mode == "classification":
            legacy_full_log_rv, log_rv_scaler = _build_partition_log_rv_target(
                active_sequence_groups,
                vol_regime_quantiles=legacy_fitted_quantiles,
            )
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
        # #309 derived-text-features ablation -- mirror the walk-forward
        # branch on the legacy 80/20 path. ``True`` (default) is
        # byte-identical to the pre-#309 path; ``False`` zeros the
        # FeatureVector slots on the assembled X tensors BEFORE the
        # rich-feature scaler fits, so the post-scaler slots stay a
        # literal 0. The legacy path never carries mt_aux, so the
        # helper's mask-zeroing branch short-circuits cleanly.
        if not bool(getattr(active_model_config, "use_derived_text_features", True)):
            train_x, _unused = _zero_derived_text_features(train_x, None)  # type: ignore[assignment]
            val_x, _unused = _zero_derived_text_features(val_x, None)  # type: ignore[assignment]
        # Fit rich-feature scaler on the train slice; legacy 6-feature
        # tensors short-circuit to no-op.
        rich_feature_scaler = fit_rich_feature_scaler_tensor(train_x)
        train_x = apply_rich_feature_scaler_tensor(train_x, rich_feature_scaler)
        val_x = apply_rich_feature_scaler_tensor(val_x, rich_feature_scaler)
        if text_emb_tensor is not None and text_missing_tensor is not None:
            train_text_emb = text_emb_tensor[: len(train_x)]
            val_text_emb = text_emb_tensor[len(train_x) :]
            train_text_missing = text_missing_tensor[: len(train_x)]
            val_text_missing = text_missing_tensor[len(train_x) :]
        else:
            train_text_emb = val_text_emb = None
            train_text_missing = val_text_missing = None
        # Split the legacy log_rv tensor at the same boundary as x / y
        # so the dataset row count invariant holds on the dual-head
        # legacy path. test partition reuses the val tensors per the
        # legacy contract below; the same goes for log_rv.
        if legacy_full_log_rv is not None:
            train_log_rv = legacy_full_log_rv[: len(train_x)]
            val_log_rv = legacy_full_log_rv[len(train_x) :]
        # Legacy path has no real held-out test partition; the val
        # tensors serve as both early-stopping and final-report eval.
        test_x = val_x
        test_y = val_y
        test_text_emb = val_text_emb
        test_text_missing = val_text_missing
        if legacy_full_log_rv is not None:
            test_log_rv = val_log_rv

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
    # Move the multi-task aux tensors to device so the per-batch step
    # can index them with the same shuffled order the DataLoader yields.
    if train_mt_aux is not None:
        train_mt_aux = {
            key: _move_to_device(tensor, device_obj) for key, tensor in train_mt_aux.items()
        }
    if val_mt_aux is not None:
        val_mt_aux = {
            key: _move_to_device(tensor, device_obj) for key, tensor in val_mt_aux.items()
        }
    if test_mt_aux is not None:
        test_mt_aux = {
            key: _move_to_device(tensor, device_obj) for key, tensor in test_mt_aux.items()
        }
    # #304 dual-head -- move the log(RV) target tensors to the active
    # device so the DataLoader shuffler can index them with the same
    # in-batch order it indexes ``x`` / ``y``. ``None`` when the head
    # mode is the default ``classification``.
    train_log_rv = _move_to_device(train_log_rv, device_obj)
    val_log_rv = _move_to_device(val_log_rv, device_obj)
    test_log_rv = _move_to_device(test_log_rv, device_obj)
    # Tensors now live on the target device, so DataLoader pinning is
    # neither needed nor supported (PyTorch raises on pinning a CUDA
    # tensor). The original pin-memory comment about deprecation
    # warnings still applies on CPU device.
    pin_memory = False
    loader_generator = make_generator(seed) if seed is not None else None
    train_dataset = _make_partition_dataset(
        train_x, train_y, train_text_emb, train_text_missing, train_mt_aux, train_log_rv
    )

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
        val_mt_aux_used = train_mt_aux
        val_log_rv_used = train_log_rv
    else:
        val_x_used = val_x
        val_y_used = val_y
        val_text_emb_used = val_text_emb
        val_text_missing_used = val_text_missing
        val_mt_aux_used = val_mt_aux
        val_log_rv_used = val_log_rv

    val_dataset = _make_partition_dataset(
        val_x_used,
        val_y_used,
        val_text_emb_used,
        val_text_missing_used,
        val_mt_aux_used,
        val_log_rv_used,
    )

    if test_x is not None and test_y is not None and len(test_x) > 0:
        test_dataset = _make_partition_dataset(
            test_x, test_y, test_text_emb, test_text_missing, test_mt_aux, test_log_rv
        )
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
    # AdamW best-practice param-group split (BERT-era convention).
    # Weight decay applies only to ``weight`` tensors; biases,
    # LayerNorm parameters, positional encodings, and 1-D normalisation
    # weights are exempted. Applying WD to LayerNorm cripples the
    # model's ability to shift distributions across regime shifts, and
    # WD on biases is mathematically meaningless. Falls back to a
    # single group when ``weight_decay=0`` so the param-group plumbing
    # does not perturb the byte-identity regression contract on the
    # legacy regression path.
    wd_value = float(weight_decay)
    # Round 5 (#244): build a unified param iterator that yields both
    # the forecaster's parameters AND (when LoRA is on) the encoder
    # adapter parameters. The base encoder is frozen by
    # ``build_lora_encoder``, so the ``requires_grad`` filter below
    # picks up exactly the adapter layers without the rest of the
    # encoder leaking in.
    def _trainable_named_parameters() -> Any:
        seen: set[int] = set()
        for name, param in work_model.named_parameters():
            if id(param) in seen:
                continue
            seen.add(id(param))
            yield f"forecaster.{name}", param
        if encoder_lora_bundle is not None:
            for name, param in encoder_lora_bundle.encoder.named_parameters():
                if id(param) in seen:
                    continue
                seen.add(id(param))
                yield f"encoder_lora.{name}", param

    if wd_value > 0.0:
        decay_params: list[torch.nn.Parameter] = []
        no_decay_params: list[torch.nn.Parameter] = []
        for name, param in _trainable_named_parameters():
            if not param.requires_grad:
                continue
            # Skip biases, layer-norm / batch-norm weights+biases, and
            # positional-encoding lookup tables. ``param.ndim <= 1``
            # is the standard "is this a vector parameter" gate that
            # catches biases + LN.weight + LN.bias + embedding norms
            # without enumerating module types.
            if name.endswith(".bias") or param.ndim <= 1 or "norm" in name.lower() or "pos" in name.lower():
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        optimizer = torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": wd_value},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=learning_rate,
        )
    else:
        optimizer = torch.optim.AdamW(
            [param for _name, param in _trainable_named_parameters() if param.requires_grad],
            lr=learning_rate,
            weight_decay=0.0,
        )
    # Phase B (#227) LR-schedule selector. ``plateau`` keeps the legacy
    # ReduceLROnPlateau path locked by ``tests/regression/test_forecaster_determinism.py``.
    # ``cosine_warmup`` swaps in OneCycleLR (warmup -> cosine -> tail)
    # over the configured epoch budget. ``schedule_steps_per_epoch``
    # holds the per-iter step count for the OneCycleLR branch so the
    # scheduler advances once per optimizer step.
    schedule_choice = str(lr_schedule).lower()
    schedule_steps_per_epoch: int | None = None
    scheduler: Any
    if schedule_choice == "cosine_warmup":
        steps_per_epoch = max(1, len(train_loader))
        schedule_steps_per_epoch = steps_per_epoch
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=learning_rate,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.1,
            anneal_strategy="cos",
            div_factor=10.0,
            final_div_factor=100.0,
        )
    elif schedule_choice == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3
        )
    else:
        raise ValueError(
            f"unsupported lr_schedule={lr_schedule!r}; choose plateau or cosine_warmup"
        )
    # Phase 9 V2 (#195) loss dispatch. ``output_mode=="classification"``
    # swaps the regression-side SmoothL1 (close, vol) loss for the
    # CrossEntropy loss the vol-regime classifier needs. The model's
    # forward path emits raw logits in classification mode so
    # CrossEntropyLoss can apply log_softmax internally.
    _active_output_mode = str(getattr(work_model, "output_mode", "regression"))
    loss_fn: nn.Module
    if _active_output_mode == "classification":
        # A1 (#206) -- pass the per-fold class weights when available.
        # ``walk_forward_path`` is the only branch that computes
        # ``fitted_class_weights``; the legacy 80/20 path leaves it
        # absent and the loss falls back to uniform weighting.
        weights_tuple = locals().get("fitted_class_weights", ())
        class_weight_tensor: torch.Tensor | None = None
        if weights_tuple:
            class_weight_tensor = torch.tensor(
                list(weights_tuple), dtype=torch.float32, device=device_obj
            )
        loss_fn = nn.CrossEntropyLoss(weight=class_weight_tensor)
    else:
        loss_fn = nn.SmoothL1Loss(beta=0.02)

    # Multi-task auxiliary loss (#273). Constructed once before the
    # epoch loop when ``multi_task_loss=True``: fits per-axis class
    # weights from the train partition's mask-aware label distribution
    # so each axis is weighted independently, then wraps everything in
    # the canonical :class:`MultiTaskLoss` module with the configured
    # lambdas. The stance branch reuses ``class_weight_tensor`` (the
    # vol-regime class weights) so the primary head's gradient stays
    # identical to the single-task path on rows where only stance is
    # supervised.
    multi_task_loss_fn: nn.Module | None = None
    if multi_task_loss_active and _active_output_mode == "classification":
        from app.models.config import (
            MULTI_TASK_CERTAINTY_CLASSES,
            MULTI_TASK_TOPIC_CLASSES,
        )
        from app.training.loss import MultiTaskLoss

        # Per-axis class counts pinned in app.models.config; the head
        # uses these exact constants so the fitted class-weight tensors
        # match the logit shape.
        n_certainty_classes = int(MULTI_TASK_CERTAINTY_CLASSES)
        n_topic_classes = int(MULTI_TASK_TOPIC_CLASSES)
        if train_mt_aux is None:
            raise RuntimeError(
                "multi_task_loss_active=True but train_mt_aux is None; "
                "the partition builder did not materialise the aux tensors."
            )
        certainty_weight = _fit_axis_class_weights_from_mask(
            train_mt_aux["certainty"],
            train_mt_aux["certainty_mask"],
            n_certainty_classes,
        ).to(device_obj)
        topic_weight = _fit_axis_class_weights_from_mask(
            train_mt_aux["topic"],
            train_mt_aux["topic_mask"],
            n_topic_classes,
        ).to(device_obj)
        multi_task_loss_fn = MultiTaskLoss(
            stance_weight=class_weight_tensor,  # vol-regime weights
            certainty_weight=certainty_weight,
            topic_weight=topic_weight,
            lambda_stance=float(getattr(active_model_config, "multi_task_lambda_stance", 1.0)),
            lambda_factor=float(getattr(active_model_config, "multi_task_lambda_factor", 0.3)),
            lambda_certainty=float(getattr(active_model_config, "multi_task_lambda_certainty", 0.3)),
            lambda_topic=float(getattr(active_model_config, "multi_task_lambda_topic", 0.3)),
        ).to(device_obj)
        print(
            "[train_model] multi_task_loss active: "
            f"lambda_stance={multi_task_loss_fn.lambda_stance} "
            f"lambda_factor={multi_task_loss_fn.lambda_factor} "
            f"lambda_certainty={multi_task_loss_fn.lambda_certainty} "
            f"lambda_topic={multi_task_loss_fn.lambda_topic}",
            flush=True,
        )

    # #304 dual-head methodology. ``regression_alpha`` is the weight on
    # the log(RV) MSE term in the joint loss ``(1 - alpha) * CE +
    # alpha * MSE``. ``head_mode="classification"`` (the default)
    # leaves the value unused -- the partition build emitted no log_rv
    # tensor and the train step's dual-head branch never fires. The
    # value is read from the active ModelConfig so a resumed checkpoint
    # reuses the same alpha the original run trained under.
    regression_alpha = float(
        getattr(active_model_config, "regression_alpha", 0.5)
    )
    if dual_head_active and _active_output_mode != "classification":
        raise ValueError(
            "head_mode in {regression, dual} requires "
            "output_mode='classification' (the regression head reads "
            "log(forward_realized_vol_10d) which only exists on the "
            "classification branch); got output_mode="
            f"{_active_output_mode!r}"
        )
    if dual_head_active:
        print(
            f"[train_model] dual-head active: head_mode={active_head_mode} "
            f"regression_alpha={regression_alpha}",
            flush=True,
        )
        # #304 alpha-boundary byte-identity. When the dual path is set
        # to alpha=0 the regression head's loss contribution drops to
        # zero; gating the head's forward computation as well keeps
        # the autograd graph identical to head_mode='classification'
        # so the parity test holds.
        if active_head_mode == "dual" and regression_alpha <= 0.0:
            work_model._skip_regression_head = True  # type: ignore[assignment]

    # InfoNCE alignment loss for the gated_infonce fusion mode (#235).
    # The training step calls ``forward_with_modality_outputs`` on the
    # multi-modal model to recover the per-modality projections, then
    # adds ``lambda * info_nce(r_t, t_t)`` on top of the classification
    # loss. The single-modality path leaves both ``info_nce_loss`` and
    # ``infonce_lambda`` unset and skips the alignment term entirely.
    info_nce_loss_fn: nn.Module | None = None
    infonce_lambda = 0.0
    multimodal_active = (
        str(getattr(active_model_config, "fusion_mode", "concat") or "concat")
        == "gated_infonce"
    )
    if multimodal_active and multi_task_loss_active:
        raise ValueError(
            "multi_task_loss + gated_infonce in the same cell is not yet "
            "supported (#273 follow-up). Disable one to proceed."
        )
    if multimodal_active:
        from app.training.info_nce_loss import InfoNCELoss

        temperature = float(getattr(active_model_config, "infonce_temperature", 0.07))
        infonce_lambda = float(getattr(active_model_config, "infonce_lambda", 0.1))
        info_nce_loss_fn = InfoNCELoss(temperature=temperature).to(device_obj)
        print(
            f"[train_model] gated_infonce active: lambda={infonce_lambda} "
            f"temperature={temperature}",
            flush=True,
        )

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

    # Bundle B LoRA freeze curriculum. When the config carries a
    # non-None ``lora_curriculum_freeze_epoch``, the LoRA adapter trains
    # for the first ``freeze_epoch`` epochs and then gets frozen at the
    # start of epoch ``freeze_epoch`` (0-indexed). Stage 2 only updates
    # the classification head while the encoder representation stays
    # fixed. The boundary is logged once so post-hoc analysis can find
    # the transition in the run log.
    lora_freeze_epoch_cfg = getattr(
        active_model_config, "lora_curriculum_freeze_epoch", None
    )
    lora_freeze_epoch: int | None = (
        int(lora_freeze_epoch_cfg) if lora_freeze_epoch_cfg is not None else None
    )
    lora_adapter_frozen = False

    for epoch_index in range(epochs):
        if encoder_lora_bundle is not None:
            from app.training.encoder_lora import (
                freeze_adapter,
                should_freeze_lora_at_epoch,
            )

            if should_freeze_lora_at_epoch(
                lora_freeze_epoch,
                epoch_index,
                already_frozen=lora_adapter_frozen,
            ):
                frozen_count = freeze_adapter(encoder_lora_bundle.encoder)
                print(
                    "INFO lora_curriculum_freeze "
                    f"epoch={epoch_index} path=encoder_lora "
                    f"frozen_params={frozen_count} "
                    f"alias={encoder_lora_bundle.encoder_alias}",
                    flush=True,
                )
                lora_adapter_frozen = True
        work_model.train()
        if encoder_lora_bundle is not None:
            encoder_lora_bundle.encoder.train()
        for batch in train_loader:
            (
                batch_x,
                batch_y,
                batch_text,
                batch_text_missing,
                batch_mt_aux,
                batch_log_rv,
            ) = _unpack_batch(batch)
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
                if encoder_lora_bundle is not None:
                    # Round 5 (#244): convert batch tokens to pooled
                    # embedding via the LoRA-wrapped encoder so the
                    # regime loss backpropagates into the adapter.
                    from app.training.encoder_lora import encode_batch_pooled

                    attention_mask_tensor = (
                        batch_text_missing
                        if batch_text_missing is not None
                        else torch.ones_like(batch_text, dtype=torch.long)
                    )
                    pooled, lora_missing = encode_batch_pooled(
                        encoder_lora_bundle, batch_text, attention_mask_tensor
                    )
                    kwargs["text_embedding"] = pooled
                    kwargs["text_embedding_missing"] = lora_missing
                else:
                    kwargs["text_embedding"] = batch_text
                    if batch_text_missing is not None:
                        kwargs["text_embedding_missing"] = batch_text_missing

            # Single computation path. The AMP gate wraps only the
            # forward + loss; backward + clip + step run outside autocast
            # (the gradient scaler handles the dtype transitions). Multi-
            # task loss (#273) builds the per-axis logits dict from
            # ``forward_multi_task`` and uses :class:`MultiTaskLoss`;
            # the single-task path keeps the legacy CE / SmoothL1 + the
            # optional InfoNCE alignment term.
            amp_ctx: Any = (
                torch.cuda.amp.autocast() if effective_amp else contextlib.nullcontext()
            )
            if multi_task_loss_fn is not None and batch_mt_aux is None:
                raise RuntimeError(
                    "multi_task_loss_fn is active but the DataLoader yielded "
                    "a batch without the aux tensors. This usually means the "
                    "TensorDataset for this partition was built without the "
                    "multi-task aux block — check the partition build path."
                )
            with amp_ctx:
                if multi_task_loss_fn is not None:
                    assert batch_mt_aux is not None  # the guard above narrows this
                    logits_dict = _run_train_forward_multi_task(
                        forward_model, batch_x, kwargs
                    )
                    stance_mask = torch.ones(
                        (batch_x.size(0),), dtype=torch.bool, device=batch_x.device
                    )
                    mt_targets = {
                        "stance": batch_y,
                        "factor": batch_mt_aux["factor"],
                        "certainty": batch_mt_aux["certainty"],
                        "topic": batch_mt_aux["topic"],
                    }
                    mt_masks = {
                        "stance_mask": stance_mask,
                        "factor_mask": batch_mt_aux["factor_mask"],
                        "certainty_mask": batch_mt_aux["certainty_mask"],
                        "topic_mask": batch_mt_aux["topic_mask"],
                    }
                    loss, _ = multi_task_loss_fn(logits_dict, mt_targets, mt_masks)
                    loss = _maybe_add_dual_head_loss(
                        loss,
                        logits_dict=logits_dict,
                        batch_log_rv=batch_log_rv,
                        head_mode=active_head_mode,
                        regression_alpha=regression_alpha,
                    )
                elif dual_head_active:
                    # #304 dual-head fast path. Single-task classification
                    # already drives stance via ``loss_fn(predictions,
                    # batch_y)``; the dual-head retrofit needs the
                    # ``log_rv`` head's MSE too, which requires the
                    # multi-task dict (since the regression head lives
                    # alongside the MultiTaskHead). Run
                    # ``forward_multi_task`` so both heads share the
                    # same backbone activations in this batch.
                    logits_dict = _run_train_forward_multi_task(
                        forward_model, batch_x, kwargs
                    )
                    stance_logits = logits_dict["stance"]
                    ce_loss = loss_fn(stance_logits, batch_y)
                    loss = _combine_dual_head_loss(
                        ce_loss=ce_loss,
                        logits_dict=logits_dict,
                        batch_log_rv=batch_log_rv,
                        head_mode=active_head_mode,
                        regression_alpha=regression_alpha,
                    )
                else:
                    predictions, align_loss = _run_train_forward_and_align(
                        forward_model,
                        batch_x,
                        kwargs,
                        info_nce_loss_fn=info_nce_loss_fn,
                        multimodal_active=multimodal_active,
                    )
                    loss = loss_fn(predictions, batch_y)
                    if align_loss is not None and infonce_lambda > 0.0:
                        loss = loss + infonce_lambda * align_loss
            if effective_amp:
                assert scaler is not None
                scaler.scale(loss).backward()
                if apply_grad_clip:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(work_model.parameters(), max_norm=clip_norm_value)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if apply_grad_clip:
                    nn.utils.clip_grad_norm_(work_model.parameters(), max_norm=clip_norm_value)
                optimizer.step()
            # OneCycleLR advances per-iter; ReduceLROnPlateau advances
            # once per epoch on the val metric (block after the eval).
            if schedule_steps_per_epoch is not None:
                scheduler.step()

        completed_epochs = epoch_index + 1
        eval_metrics = _evaluate_model(
            forward_model,
            val_loader,
            device_obj,
            loss_fn,
            credibility_buffer=val_credibility_buffer,
            encoder_lora_bundle=encoder_lora_bundle,
            multi_task_loss_fn=multi_task_loss_fn,
        )
        if schedule_steps_per_epoch is None:
            scheduler.step(eval_metrics.loss)

        # Early-stop signal. Classification mode tracks macro-F1
        # (higher = better) because CE loss can spike from logit
        # over-confidence while macro-F1 keeps improving on noisy
        # targets like forward 10-day realised vol. Regression mode
        # keeps the legacy combined-RMSE / loss path so the
        # tests/regression/test_forecaster_determinism.py byte-identity
        # lock at +/-1e-4 stays green.
        #
        # #304 dual-head adjustment: under ``head_mode='regression'``
        # the classifier head receives no gradient (the CE branch is
        # dropped from the joint loss), so ``regime_f1_macro`` would be
        # driven by noise around random init and the best-epoch
        # selection would be meaningless. Route the early-stop signal
        # through the regression head's val RMSE (lower-is-better) in
        # that mode. ``head_mode='dual'`` keeps the F1 selection since
        # the classifier head is still trained.
        if _active_output_mode == "classification" and active_head_mode == "regression":
            current_rmse = float(eval_metrics.regression_rmse_log_rv or float("inf"))
            best_rmse = (
                float(
                    getattr(best_val_metrics, "regression_rmse_log_rv", float("inf"))
                    or float("inf")
                )
                if best_val_metrics is not None
                else float("inf")
            )
            improved = (
                best_val_metrics is None
                or current_rmse + 1e-6 < best_rmse
            )
        elif _active_output_mode == "classification":
            current_macro_f1 = float(eval_metrics.regime_f1_macro or 0.0)
            best_macro_f1 = float(
                getattr(best_val_metrics, "regime_f1_macro", 0.0) or 0.0
            ) if best_val_metrics is not None else -1.0
            improved = (
                best_val_metrics is None
                or current_macro_f1 > best_macro_f1 + 1e-6
            )
        else:
            improved = (
                best_val_metrics is None
                or eval_metrics.loss + 1e-6 < best_val_metrics.loss
            )
        if improved:
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
            multi_task_loss_fn=multi_task_loss_fn,
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
        encoder_lora_bundle=encoder_lora_bundle,
        multi_task_loss_fn=multi_task_loss_fn,
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
            record_row_predictions=True,
            encoder_lora_bundle=encoder_lora_bundle,
            multi_task_loss_fn=multi_task_loss_fn,
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
        log_rv_scaler=(
            {"mean": float(log_rv_scaler[0]), "std": float(log_rv_scaler[1])}
            if log_rv_scaler is not None
            else None
        ),
    )

    if save_checkpoint:
        from app.training.checkpoint import _save_model_checkpoint

        # `close_scale` was fitted on this fold's training rows in
        # `_build_training_tensors`; persisting it on the checkpoint is what
        # lets inference (`services.forecaster._predict_next_point`) recover
        # the original price magnitude. The two values must agree byte-for-
        # byte across save/load — see the determinism regression test.
        _save_model_checkpoint(
            work_model,
            checkpoint_target,
            summary,
            close_scale=close_scale,
            rich_feature_scaler=rich_feature_scaler,
        )
        # Conformal calibration sidecar (#216). Classification-mode runs
        # write a manifest with the APS softmax_quantile fitted on the
        # held-out val partition's per-row softmax scores at the best
        # epoch. The /analyze inference path reads the manifest via
        # ``app.services.forecaster._conformal_manifest_for`` to build
        # calibrated prediction sets.
        _maybe_write_classification_conformal_manifest(
            best_val_metrics, checkpoint_target
        )
        if encoder_lora_bundle is not None:
            # Round 5 (#244) sidecar: write only the LoRA adapter state
            # (not the base encoder weights) so a future audit / resume
            # path can rebuild the wrapped encoder by re-loading the
            # registry-pinned base + the adapter delta. ``peft`` exposes
            # ``get_peft_model_state_dict`` for this exact use; the
            # lazy import keeps the helper module agnostic when peft
            # is absent (which only happens on a ``encoder_lora=False``
            # run, so the branch never fires).
            get_peft_model_state_dict: Any = None
            try:
                from peft import get_peft_model_state_dict as _peft_state_dict
                get_peft_model_state_dict = _peft_state_dict
            except ImportError:  # pragma: no cover - defensive
                get_peft_model_state_dict = None
            if get_peft_model_state_dict is not None:
                adapter_path = Path(str(checkpoint_target) + ".lora_adapter.pt")
                adapter_state = get_peft_model_state_dict(encoder_lora_bundle.encoder)
                torch.save(
                    {
                        "encoder_alias": encoder_lora_bundle.encoder_alias,
                        "out_dim": encoder_lora_bundle.out_dim,
                        "adapter_state": adapter_state,
                    },
                    adapter_path,
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
    validation_fraction: float | None = None,
    validation_split: float | None = None,
    early_stopping_patience: int = 10,
    checkpoint_path: str | Path = BEST_MODEL_PATH,
) -> TrainingResult:
    resolved_fraction = _resolve_validation_fraction(
        validation_fraction, validation_split
    )
    return train_model(
        vectors=vectors,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        validation_fraction=resolved_fraction,
        early_stopping_patience=early_stopping_patience,
        checkpoint_path=checkpoint_path,
        save_checkpoint=True,
    )
