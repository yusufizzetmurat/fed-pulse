"""Facade for the forecaster module.

The actual implementations live under ``app.models``, ``app.training``, and
``app.evaluation``. This module hosts the FastAPI-singleton state plus the
inference path that depends on it (``forecast_quantitative_series``,
``_get_model``, ``_predict_next_point``, ``get_model_artifact_metadata``,
``_build_confidence_bands``) and re-exports every public name that callers
across the codebase import from here.
"""
from __future__ import annotations

import copy
import math
import threading
from pathlib import Path
from collections.abc import Iterable
from typing import Any

import torch

from app.evaluation.metrics import (
    EvaluationMetrics,
    TrainingDataSourceSummary,
    TrainingResult,
    TrainingRunSummary,
)
from app.models.attention import ChunkAttentionPooler, TimeDecayAttention
from app.models.config import (
    BEST_MODEL_PATH,
    CONFIDENCE_Z_SCORE,
    DEFAULT_BATCH_SIZE,
    DEFAULT_CHUNK_DECAY_RATE,
    DEFAULT_CHUNK_EMBEDDING_SIZE,
    DEFAULT_CHUNK_PROJECTION_DIM,
    DEFAULT_CLOSE_SCALE,
    DEFAULT_DATA_DIR,
    DEFAULT_DROPOUT,
    DEFAULT_EARLY_STOPPING_PATIENCE,
    DEFAULT_EPOCHS,
    DEFAULT_HEAD_HIDDEN_SIZE,
    DEFAULT_HIDDEN_SIZE,
    DEFAULT_INITIAL_DECAY_RATE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_NUM_LAYERS,
    DEFAULT_VALIDATION_SPLIT,
    ELAPSED_TIME_FEATURE_INDEX,
    FEATURE_SIZE,
    FORECAST_CONFIDENCE_LEVEL,
    RICH_FEATURE_SIZE,
    RichFeatureScalerParams,
    MODELS_DIR,
    SENTIMENT_FEATURE_INDEX,
    SEQUENCE_LENGTH,
    FeatureVector,
    ModelConfig,
    build_lookback_sequence,
)
from app.models.lstm import ForecasterModel
from app.models.tcn import TemporalConvNet
from app.models.transformer import SmallTransformer, _SinusoidalPositionalEncoding
from app.training.checkpoint import (
    _capture_rng_state,
    _checkpoint_metadata,
    _checkpoint_payload,
    _load_model_checkpoint,
    _load_state_dict_loose,
    _read_checkpoint_payload,
    _restore_rng_state,
    _save_model_checkpoint,
    checkpoint_exists,
)
from app.training.loaders import (
    _build_training_tensors,
    _extract_required_float,
    _extract_record_groups,
    _is_record_mapping_list,
    _load_csv_records,
    _load_json_records,
    _load_jsonl_records,
    _load_record_groups,
    _split_train_validation,
    build_feature_vectors,
    inspect_training_data_sources,
    load_training_sequences_from_data,
    load_training_sequences_from_package,
)
from app.training.loop import (
    _build_model,
    _coerce_model_config,
    _evaluate_model,
    _resolve_device,
    bootstrap_checkpoint,
    train_model,
)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


_model: ForecasterModel | None = None
_model_artifact_metadata: dict[str, Any] | None = None
_model_lock = threading.Lock()


def _get_model() -> ForecasterModel:
    global _model, _model_artifact_metadata
    if _model is not None:
        return _model

    with _model_lock:
        if _model is None:
            device = _resolve_device()
            payload = _read_checkpoint_payload(BEST_MODEL_PATH, device)
            model = _build_model(
                payload.get("model_config") if isinstance(payload, dict) else None,
                device=device,
            )
            if payload is not None:
                _load_state_dict_loose(model, payload["model_state_dict"], str(BEST_MODEL_PATH))
            model.eval()
            _model = model
            _model_artifact_metadata = _checkpoint_metadata(payload, BEST_MODEL_PATH, model=model)
    return _model


def _set_singleton_after_train(
    work_model: ForecasterModel,
    checkpoint_target: Path,
    device_obj: torch.device,
) -> None:
    """Refresh the in-process singleton + metadata after training writes a checkpoint."""
    global _model, _model_artifact_metadata
    with _model_lock:
        _model = copy.deepcopy(work_model).to(device_obj)
        _model.eval()
        _model_artifact_metadata = _checkpoint_metadata(
            _read_checkpoint_payload(checkpoint_target, device_obj),
            checkpoint_target,
            model=_model,
        )


def _build_inference_tensor(
    sequence: list[FeatureVector],
    model: ForecasterModel,
    device: torch.device,
) -> torch.Tensor:
    """Build the per-event input tensor for one forward pass.

    Dispatches on the loaded model's ``input_size``: rich-features
    models (input_size == RICH_FEATURE_SIZE = 35) use
    ``as_rich_list`` and apply the persisted RobustScaler from the
    checkpoint metadata so inference matches training-time
    normalisation. Legacy 6-feature models keep the byte-identical
    ``as_list`` path so the existing /analyze contract is unchanged.
    """

    if int(getattr(model, "input_size", FEATURE_SIZE)) == RICH_FEATURE_SIZE:
        rows = [item.as_rich_list() for item in sequence]
        x = torch.tensor([rows], dtype=torch.float32, device=device)
        scaler = (_model_artifact_metadata or {}).get("rich_feature_scaler")
        if scaler is not None:
            from app.training.loaders import apply_rich_feature_scaler_tensor

            x = apply_rich_feature_scaler_tensor(x, scaler)
        return x
    rows = [item.as_list() for item in sequence]
    return torch.tensor([rows], dtype=torch.float32, device=device)


def _predict_next_point(model: ForecasterModel, sequence: list[FeatureVector]) -> tuple[float, float]:
    device = next(model.parameters()).device
    x = _build_inference_tensor(sequence, model, device)
    kwargs: dict[str, torch.Tensor] = {}
    if getattr(model, "credibility_features", False):
        # Inference-side credibility uses a zero vector by default; the live
        # vtasca + FRED loader (services.credibility_loader) populates real
        # numbers in the training loop and at /analyze when the caller wires it
        # in. Zero is the neutral "all axes unknown" reading from
        # CredibilityVector — safe for forecast inference.
        kwargs["credibility"] = torch.zeros(
            (1, int(getattr(model, "credibility_dim", 4))),
            dtype=torch.float32,
            device=device,
        )
    with torch.no_grad():
        out = model(x, **kwargs).squeeze(0)
    close_scale = float((_model_artifact_metadata or {}).get("close_scale", DEFAULT_CLOSE_SCALE))
    pred_close = float(out[0].item()) * close_scale
    pred_vol = float(out[1].item())
    return pred_close, pred_vol


def _parse_horizon_steps(horizon: str) -> int:
    if horizon.endswith("d") and horizon[:-1].isdigit():
        return max(1, int(horizon[:-1]))
    return 3


def parse_horizon_steps(horizon: str) -> int:
    return _parse_horizon_steps(horizon)


def _sample_std(values: Iterable[float]) -> float:
    items = [float(value) for value in values]
    if len(items) < 2:
        return 0.0
    mean = sum(items) / len(items)
    variance = sum((value - mean) ** 2 for value in items) / (len(items) - 1)
    return math.sqrt(max(variance, 0.0))


def _conformal_manifest_for(checkpoint_path: Path | None) -> Any:
    if checkpoint_path is None:
        return None
    # `with_suffix(".conformal.json")` rejects multi-dot suffixes on Python < 3.12.
    # `with_name` constructs the sibling path explicitly so behaviour is identical
    # on 3.11 and 3.12+.
    manifest_path = checkpoint_path.with_name(checkpoint_path.stem + ".conformal.json")
    if not manifest_path.exists():
        return None
    try:
        from app.evaluation.conformal import load_manifest

        return load_manifest(manifest_path)
    except Exception:
        return None


def _build_confidence_bands(
    history_close: list[float],
    history_vol: list[float],
    forecast_close: list[float],
    forecast_vol: list[float],
    *,
    conformal_manifest: Any = None,
) -> tuple[list[float], list[float], list[float], list[float]]:
    if conformal_manifest is not None:
        from app.evaluation.conformal import apply_conformal_bands

        return apply_conformal_bands(
            close_predictions=forecast_close,
            volatility_predictions=forecast_vol,
            manifest=conformal_manifest,
        )

    close_returns = [
        (curr - prev) / prev
        for prev, curr in zip(history_close, history_close[1:])
        if abs(prev) > 1e-12
    ]
    vol_changes = [curr - prev for prev, curr in zip(history_vol, history_vol[1:])]

    close_sigma = max(_sample_std(close_returns), 0.0025)
    latest_vol = max(history_vol[-1] if history_vol else 0.0, forecast_vol[0] if forecast_vol else 0.0)
    vol_sigma = max(_sample_std(vol_changes), latest_vol * 0.08, 0.00015)

    forecast_close_lower: list[float] = []
    forecast_close_upper: list[float] = []
    forecast_vol_lower: list[float] = []
    forecast_vol_upper: list[float] = []

    for step_idx, (pred_close, pred_vol) in enumerate(zip(forecast_close, forecast_vol), start=1):
        horizon_scale = math.sqrt(step_idx)
        close_width = max(pred_close, 1.0) * close_sigma * CONFIDENCE_Z_SCORE * horizon_scale
        vol_width = vol_sigma * CONFIDENCE_Z_SCORE * horizon_scale

        forecast_close_lower.append(min(max(0.0, pred_close - close_width), pred_close))
        forecast_close_upper.append(pred_close + close_width)
        forecast_vol_lower.append(min(max(0.0, pred_vol - vol_width), pred_vol))
        forecast_vol_upper.append(pred_vol + vol_width)

    return (
        forecast_close_lower,
        forecast_close_upper,
        forecast_vol_lower,
        forecast_vol_upper,
    )


def get_model_artifact_metadata(
    *,
    runtime_mode: str = "fast",
    model: ForecasterModel | None = None,
    adaptation_summary: TrainingRunSummary | None = None,
) -> dict[str, Any]:
    base_metadata = dict(
        _model_artifact_metadata
        or _checkpoint_metadata(
            None,
            BEST_MODEL_PATH,
            runtime_mode=runtime_mode,
            model=model,
            adaptation_summary=adaptation_summary,
        )
    )
    base_metadata["runtime_mode"] = runtime_mode
    if model is not None:
        config = ModelConfig.from_model(model)
        base_metadata.update(
            {
                "hidden_size": config.hidden_size,
                "num_layers": config.num_layers,
                "dropout": config.dropout,
                "head_hidden_size": config.head_hidden_size,
            }
        )
    if adaptation_summary is not None:
        base_metadata.update(
            {
                "adaptation_epochs_completed": adaptation_summary.epochs_completed,
                "adaptation_best_epoch": adaptation_summary.best_epoch,
                "adaptation_loss": adaptation_summary.metrics.loss if adaptation_summary.metrics else None,
                "adaptation_combined_rmse": (
                    adaptation_summary.metrics.combined_rmse if adaptation_summary.metrics else None
                ),
            }
        )
    return base_metadata


def forecast_quantitative_series(
    vectors: list[FeatureVector],
    forecast_mode: str = "fast",
    horizon: str = "3d",
    forecast_dates: list[str] | None = None,
) -> dict[str, object]:
    if not vectors:
        vectors = [FeatureVector(date="", sentiment_score=0.0, market_close=0.0, market_volatility=0.0)]

    base_model = _get_model()
    training_result = (
        train_model(
            base_model=base_model,
            vectors=vectors,
            epochs=18,
            batch_size=32,
            learning_rate=5e-4,
            validation_split=0.25,
            early_stopping_patience=4,
            save_checkpoint=False,
        )
        if forecast_mode == "quick_train"
        else None
    )
    model = training_result.model if training_result is not None else base_model

    history_vectors = vectors[-30:]
    history_timestamps = [item.date for item in history_vectors]
    history_close = [float(item.market_close) for item in history_vectors]
    history_vol = [float(item.market_volatility) for item in history_vectors]

    steps = _parse_horizon_steps(horizon)
    rolling = history_vectors[-SEQUENCE_LENGTH:]
    forecast_close: list[float] = []
    forecast_vol: list[float] = []
    forecast_timestamps: list[str] = []

    last_date = history_timestamps[-1] if history_timestamps else ""
    for step in range(steps):
        fixed_sequence = build_lookback_sequence(rolling)
        next_close, next_vol = _predict_next_point(model, fixed_sequence)
        last_vector = fixed_sequence[-1]
        if forecast_dates and step < len(forecast_dates):
            next_date_label = str(forecast_dates[step])
        else:
            next_date_label = f"{last_date}+{step + 1}" if last_date else f"t+{step + 1}"
        next_vector = FeatureVector.from_market_state(
            date=next_date_label,
            sentiment_score=float(last_vector.sentiment_score),
            market_close=next_close,
            market_volatility=next_vol,
            previous_close=float(last_vector.market_close),
            previous_volatility=float(last_vector.market_volatility),
        )
        rolling = (rolling + [next_vector])[-SEQUENCE_LENGTH:]

        forecast_timestamps.append(next_date_label)
        forecast_close.append(next_close)
        forecast_vol.append(next_vol)

    conformal_manifest = _conformal_manifest_for(BEST_MODEL_PATH)
    (
        forecast_close_lower,
        forecast_close_upper,
        forecast_vol_lower,
        forecast_vol_upper,
    ) = _build_confidence_bands(
        history_close,
        history_vol,
        forecast_close,
        forecast_vol,
        conformal_manifest=conformal_manifest,
    )

    vol_values = [*history_vol, *forecast_vol, *forecast_vol_lower, *forecast_vol_upper]
    if vol_values:
        vol_min = min(vol_values)
        vol_max = max(vol_values)
        spread = max(vol_max - vol_min, 1e-6)
        vol_scale = {
            "suggested_ymin": max(0.0, vol_min - spread * 0.15),
            "suggested_ymax": vol_max + spread * 0.15,
        }
    else:
        vol_scale = {"suggested_ymin": 0.0, "suggested_ymax": 1.0}

    return {
        "prediction": {
            "close": float(forecast_close[-1]),
            "volatility": float(forecast_vol[-1]),
            "horizon": horizon,
        },
        "model": get_model_artifact_metadata(
            runtime_mode=forecast_mode,
            model=model,
            adaptation_summary=training_result.summary if training_result is not None else None,
        ),
        "series": {
            "timestamps": history_timestamps,
            "history_close": history_close,
            "history_volatility": history_vol,
            "forecast_timestamps": forecast_timestamps,
            "forecast_close": forecast_close,
            "forecast_close_lower": forecast_close_lower,
            "forecast_close_upper": forecast_close_upper,
            "forecast_volatility": forecast_vol,
            "forecast_volatility_lower": forecast_vol_lower,
            "forecast_volatility_upper": forecast_vol_upper,
            "forecast_confidence_level": (
                float(conformal_manifest.nominal_coverage)
                if conformal_manifest is not None
                else FORECAST_CONFIDENCE_LEVEL
            ),
            "volatility_scale": vol_scale,
            "forecast_band_source": (
                "conformal" if conformal_manifest is not None else "gaussian_z"
            ),
            "conformal_coverage": (
                float(conformal_manifest.nominal_coverage)
                if conformal_manifest is not None
                else None
            ),
        },
    }
