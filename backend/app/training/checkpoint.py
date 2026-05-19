from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch

from app.evaluation.metrics import EvaluationMetrics, TrainingRunSummary
from app.models.config import (
    BEST_MODEL_PATH,
    DEFAULT_CLOSE_SCALE,
    FEATURE_SIZE,
    SEQUENCE_LENGTH,
    ModelConfig,
    RichFeatureScalerParams,
)
from app.models.lstm import ForecasterModel


def _checkpoint_input_size(payload: dict[str, Any]) -> int | None:
    model_config = payload.get("model_config")
    if isinstance(model_config, dict) and "input_size" in model_config:
        try:
            return int(model_config["input_size"])
        except (TypeError, ValueError):
            return None
    if "input_size" in payload:
        try:
            return int(payload["input_size"])
        except (TypeError, ValueError):
            return None
    return None


def _read_checkpoint_payload(checkpoint_path: Path, device: torch.device) -> dict[str, Any] | None:
    if not checkpoint_path.exists():
        return None

    # `weights_only=False` is intentional: our checkpoints are written by this
    # process and carry trusted python objects (TrainingRunSummary,
    # ModelConfig, RNG state). PyTorch 2.6+ defaults `weights_only=True` which
    # rejects those payloads.
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not (isinstance(payload, dict) and "model_state_dict" in payload):
        payload = {"model_state_dict": payload}

    # Audit Tier 0.2: previous behaviour silently dropped any checkpoint
    # whose ``input_size`` did not match the legacy ``FEATURE_SIZE = 6``
    # constant. That rejected every rich-feature (``input_size=35``)
    # checkpoint and bootstrap-trained from scratch in the API path
    # without surfacing the swap. The model factory honours the saved
    # ``input_size`` via ``ModelConfig.from_dict`` so the caller can
    # build the correctly-sized model from the payload itself; the
    # legacy-constant gate was the wrong sentinel in the wrong place.
    return payload


def _metrics_from_payload(payload: dict[str, Any] | None) -> EvaluationMetrics | None:
    if not isinstance(payload, dict):
        return None
    metrics = payload.get("metrics")
    if isinstance(metrics, dict):
        try:
            return EvaluationMetrics(
                loss=float(metrics["loss"]),
                close_rmse=float(metrics["close_rmse"]),
                volatility_rmse=float(metrics["volatility_rmse"]),
                combined_rmse=float(metrics["combined_rmse"]),
            )
        except (KeyError, TypeError, ValueError):
            return None
    return None


def _coerce_payload_config(payload: dict[str, Any] | None) -> ModelConfig:
    if not isinstance(payload, dict):
        return ModelConfig()
    raw = payload.get("model_config")
    if isinstance(raw, ModelConfig):
        return raw
    if isinstance(raw, dict):
        return ModelConfig(
            input_size=int(raw.get("input_size", FEATURE_SIZE)),
            hidden_size=int(raw.get("hidden_size", ModelConfig().hidden_size)),
            num_layers=int(raw.get("num_layers", ModelConfig().num_layers)),
            dropout=float(raw.get("dropout", ModelConfig().dropout)),
            head_hidden_size=int(raw.get("head_hidden_size", ModelConfig().head_hidden_size)),
            initial_decay_rate=float(raw.get("initial_decay_rate", ModelConfig().initial_decay_rate)),
            text_channel=str(raw.get("text_channel", "scalar")),
            embedding_adapter_dim=int(raw.get("embedding_adapter_dim", 128)),
            credibility_features=bool(raw.get("credibility_features", False)),
            architecture=str(raw.get("architecture", "lstm")),
            # Phase 9 V2 (#195) classification-mode fields. Defaults
            # match the regression path so pre-Phase-9 checkpoints
            # rehydrate byte-identical.
            output_mode=str(raw.get("output_mode", "regression")),
            n_classes=int(raw.get("n_classes", 3)),
            vol_regime_quantiles=tuple(
                float(v) for v in (raw.get("vol_regime_quantiles") or ())
            ),
            vol_regime_target=str(
                raw.get("vol_regime_target", "forward_realized_vol_10d")
            ),
        )
    return ModelConfig()


def _checkpoint_metadata(
    payload: dict[str, Any] | None,
    checkpoint_path: Path,
    *,
    runtime_mode: str = "fast",
    model: ForecasterModel | None = None,
    adaptation_summary: TrainingRunSummary | None = None,
) -> dict[str, Any]:
    model_config = (
        ModelConfig.from_model(model)
        if model is not None
        else _coerce_payload_config(payload)
    )
    payload_metrics = _metrics_from_payload(payload)
    metrics = adaptation_summary.metrics if adaptation_summary and adaptation_summary.metrics else payload_metrics
    decay_rate: float | None = None
    if model is not None and hasattr(model, "time_decay"):
        decay_rate = float(model.time_decay.decay_rate.detach().cpu().item())
    metadata: dict[str, Any] = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_exists": checkpoint_path.exists(),
        "checkpoint_loaded": bool(payload),
        "runtime_mode": runtime_mode,
        "hidden_size": model_config.hidden_size,
        "num_layers": model_config.num_layers,
        "dropout": model_config.dropout,
        "head_hidden_size": model_config.head_hidden_size,
        "close_scale": (
            float(payload.get("close_scale", DEFAULT_CLOSE_SCALE))
            if isinstance(payload, dict)
            else float(DEFAULT_CLOSE_SCALE)
        ),
        "rich_feature_scaler": (
            RichFeatureScalerParams.from_dict(payload.get("rich_feature_scaler"))
            if isinstance(payload, dict)
            else None
        ),
        "sequence_length": (
            int(payload.get("sequence_length", SEQUENCE_LENGTH))
            if isinstance(payload, dict)
            else int(SEQUENCE_LENGTH)
        ),
        "best_loss": payload_metrics.loss if payload_metrics else payload.get("best_loss") if isinstance(payload, dict) else None,
        "combined_rmse": payload_metrics.combined_rmse if payload_metrics else None,
        "adaptation_epochs_completed": adaptation_summary.epochs_completed if adaptation_summary else None,
        "adaptation_best_epoch": adaptation_summary.best_epoch if adaptation_summary else None,
        "adaptation_loss": metrics.loss if adaptation_summary and metrics else None,
        "adaptation_combined_rmse": metrics.combined_rmse if adaptation_summary and metrics else None,
        "decay_rate": decay_rate,
        "chunk_attention": None,
    }
    return metadata


def _capture_rng_state() -> dict[str, Any]:
    import random as _stdrandom

    rng: dict[str, Any] = {
        "torch_cpu": torch.get_rng_state().tolist(),
        "python": _stdrandom.getstate(),
    }
    try:
        import numpy as _np

        state = _np.random.get_state()
        # `state` is a tuple; first element is the name, second the state array.
        rng["numpy"] = list(state) if isinstance(state, tuple) else state
    except Exception:
        rng["numpy"] = None
    if torch.cuda.is_available():
        try:
            rng["torch_cuda"] = [s.tolist() for s in torch.cuda.get_rng_state_all()]
        except Exception:
            rng["torch_cuda"] = None
    return rng


def _restore_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    import random as _stdrandom

    cpu_state = state.get("torch_cpu")
    if cpu_state is not None:
        try:
            torch.set_rng_state(torch.tensor(cpu_state, dtype=torch.uint8))
        except Exception:
            pass
    py_state = state.get("python")
    if py_state is not None:
        try:
            _stdrandom.setstate(tuple(py_state) if isinstance(py_state, list) else py_state)
        except Exception:
            pass
    np_state = state.get("numpy")
    if np_state is not None:
        try:
            import numpy as _np

            if isinstance(np_state, list):
                np_state = tuple(np_state)
            _np.random.set_state(np_state)
        except Exception:
            pass


def _checkpoint_payload(
    model: ForecasterModel,
    summary: TrainingRunSummary,
    *,
    close_scale: float | None = None,
    rich_feature_scaler: RichFeatureScalerParams | None = None,
) -> dict[str, Any]:
    """Build the dict torch.save writes for one trained model.

    ``close_scale`` is the per-fold normaliser fitted in
    ``app.training.loaders.fit_close_scale``. Passing ``None`` falls back
    to the legacy ``DEFAULT_CLOSE_SCALE`` constant — older paths that did
    not thread the fitted scale through (e.g. bare manifest exports) keep
    working unchanged. New callers always pass the fitted value so resume-
    from-checkpoint matches the training-time normalisation.

    ``rich_feature_scaler`` is the RobustScaler (median + IQR) fitted in
    ``app.training.loaders.fit_rich_feature_scaler_tensor`` over the
    train slice of the rich-feature block [FEATURE_SIZE:RICH_FEATURE_SIZE].
    ``None`` is the legacy 6-feature path -- it serialises to a literal
    None key so the rehydration side returns no scaler and inference
    falls back to the identity transform.
    """

    return {
        "model_state_dict": model.state_dict(),
        "best_loss": float(summary.metrics.loss) if summary.metrics else None,
        "metrics": summary.metrics.to_dict() if summary.metrics else None,
        "model_config": ModelConfig.from_model(model).to_dict(),
        "training_summary": summary.to_dict(),
        "input_size": FEATURE_SIZE,
        "sequence_length": SEQUENCE_LENGTH,
        "close_scale": float(close_scale) if close_scale is not None else float(DEFAULT_CLOSE_SCALE),
        "rich_feature_scaler": (
            rich_feature_scaler.to_dict()
            if rich_feature_scaler is not None
            else None
        ),
        "rng_state": _capture_rng_state(),
    }


def _save_model_checkpoint(
    model: ForecasterModel,
    checkpoint_path: Path,
    summary: TrainingRunSummary,
    *,
    close_scale: float | None = None,
    rich_feature_scaler: RichFeatureScalerParams | None = None,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        _checkpoint_payload(
            model,
            summary,
            close_scale=close_scale,
            rich_feature_scaler=rich_feature_scaler,
        ),
        checkpoint_path,
    )
    try:
        from app.audit import append_audit_entry, hash_file
        from app.logging import current_run_id

        append_audit_entry(
            "checkpoint_saved",
            run_id=current_run_id(),
            metadata={
                "path": str(checkpoint_path),
                "sha256": hash_file(checkpoint_path),
                "best_loss": float(summary.metrics.loss) if summary.metrics else None,
            },
        )
    except Exception:  # pragma: no cover — never let audit break training
        pass


def _load_state_dict_loose(model: ForecasterModel, state_dict: dict[str, Any], source: str) -> None:
    """Load a checkpoint non-strictly and surface any missing/unexpected keys."""
    result = model.load_state_dict(state_dict, strict=False)
    missing = list(getattr(result, "missing_keys", []) or [])
    unexpected = list(getattr(result, "unexpected_keys", []) or [])
    if missing or unexpected:
        print(
            f"[forecaster] checkpoint {source}: missing={missing} unexpected={unexpected}",
            file=sys.stderr,
        )


def _load_model_checkpoint(
    model: ForecasterModel,
    checkpoint_path: Path,
    device: torch.device,
) -> dict[str, Any] | None:
    payload = _read_checkpoint_payload(checkpoint_path, device)
    if payload is None:
        return None
    _load_state_dict_loose(model, payload["model_state_dict"], str(checkpoint_path))
    return payload


def checkpoint_exists(checkpoint_path: str | Path = BEST_MODEL_PATH) -> bool:
    return Path(checkpoint_path).exists()
