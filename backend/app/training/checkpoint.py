from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)

from app.evaluation.metrics import EvaluationMetrics, TrainingRunSummary
from app.models.config import (
    BEST_MODEL_PATH,
    DEFAULT_CLOSE_SCALE,
    FEATURE_SIZE,
    SEQUENCE_LENGTH,
    ModelConfig,
    RichFeatureScalerParams,
)

# Post-#336 the checkpoint helpers accept both research and serving
# forecasters (they only touch ``nn.Module`` APIs and the shared
# attribute surface from :class:`ForecasterBase`). Annotating against
# the base widens the contract without breaking the legacy
# ``ForecasterModel`` callers, which now alias to the research class.
from app.models.forecaster_base import ForecasterBase
from app.models.lstm import ForecasterModel  # noqa: F401 -- back-compat re-export


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
        # Phase 9 V2 (#195, #199) classification fields are nullable
        # and default to None on EvaluationMetrics; passing missing
        # keys explicitly keeps the regression-mode contract
        # byte-identical while letting a classification-mode checkpoint
        # round-trip its regime metrics + classification_breakdown.
        def _opt_float(key: str) -> float | None:
            value = metrics.get(key)
            if value is None:
                return None
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        breakdown_raw = metrics.get("classification_breakdown")
        breakdown = breakdown_raw if isinstance(breakdown_raw, dict) else None
        # #317 finding #7: forward the per-head rates_metrics block so a
        # checkpoint with rates heads round-trips through the loader.
        rates_metrics_raw = metrics.get("rates_metrics")
        rates_metrics = rates_metrics_raw if isinstance(rates_metrics_raw, dict) else None

        try:
            return EvaluationMetrics(
                loss=float(metrics["loss"]),
                close_rmse=float(metrics["close_rmse"]),
                volatility_rmse=float(metrics["volatility_rmse"]),
                combined_rmse=float(metrics["combined_rmse"]),
                direction_accuracy=_opt_float("direction_accuracy"),
                f1_macro=_opt_float("f1_macro"),
                direction_auc=_opt_float("direction_auc"),
                regime_accuracy=_opt_float("regime_accuracy"),
                regime_f1_macro=_opt_float("regime_f1_macro"),
                regime_loss=_opt_float("regime_loss"),
                classification_breakdown=breakdown,
                rates_metrics=rates_metrics,
                # #304 dual-head regression surface. Pre-#304
                # checkpoints leave these absent; the defaults give
                # back ``None`` so the legacy contract holds.
                regression_rmse_log_rv=_opt_float("regression_rmse_log_rv"),
                regression_mae_log_rv=_opt_float("regression_mae_log_rv"),
                regression_loss=_opt_float("regression_loss"),
                regression_r2_log_rv=_opt_float("regression_r2_log_rv"),
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
    # Walk the payload top-level for an ``input_size`` shim so a
    # checkpoint that stored the dim at the payload root rehydrates
    # against the right feature width instead of silently falling back
    # to the legacy ``FEATURE_SIZE`` constant.
    payload_input_size = _checkpoint_input_size(payload)
    if isinstance(raw, dict):
        return ModelConfig(
            input_size=int(
                raw.get(
                    "input_size",
                    payload_input_size if payload_input_size is not None else FEATURE_SIZE,
                )
            ),
            hidden_size=int(raw.get("hidden_size", ModelConfig().hidden_size)),
            num_layers=int(raw.get("num_layers", ModelConfig().num_layers)),
            dropout=float(raw.get("dropout", ModelConfig().dropout)),
            head_hidden_size=int(raw.get("head_hidden_size", ModelConfig().head_hidden_size)),
            initial_decay_rate=float(
                raw.get("initial_decay_rate", ModelConfig().initial_decay_rate)
            ),
            text_channel=str(raw.get("text_channel", "scalar")),
            embedding_adapter_dim=int(raw.get("embedding_adapter_dim", 128)),
            credibility_features=bool(raw.get("credibility_features", False)),
            architecture=str(raw.get("architecture", "lstm")),
            # Phase 9 V2 (#195) classification-mode fields. Defaults
            # match the regression path so pre-Phase-9 checkpoints
            # rehydrate byte-identical.
            output_mode=str(raw.get("output_mode", "regression")),
            n_classes=int(raw.get("n_classes", 3)),
            vol_regime_quantiles=tuple(float(v) for v in (raw.get("vol_regime_quantiles") or ())),
            vol_regime_target=str(raw.get("vol_regime_target", "forward_realized_vol_10d")),
            # #317 finding #4 mirror: forward post-#292 rates fields
            # so a checkpoint with rates heads rehydrates the same
            # rates config the run trained against. Pre-#292
            # checkpoints leave these absent and the defaults give back
            # the empty-tuple no-op.
            rates_heads=tuple(str(v).lower() for v in (raw.get("rates_heads") or ())),
            rates_head_mode=str(raw.get("rates_head_mode", "regression") or "regression"),
            rates_aux_classification=bool(raw.get("rates_aux_classification", False)),
            rates_alpha=float(raw.get("rates_alpha", 0.5)),
            # #435: forward the new vol-target-mode so a checkpoint trained
            # under --vol-target-mode=garch_residual rehydrates with the
            # right target column on eval / calibration paths. Pre-#435
            # checkpoints leave the key absent and the default collapses
            # to the raw column.
            vol_target_mode=str(raw.get("vol_target_mode", "raw") or "raw"),
            # Round-trip the supervised forward-vol horizon so a checkpoint
            # trained under ``--target-horizon`` rehydrates against the
            # same events-parquet column on eval / calibration paths.
            vol_target_horizon=int(raw.get("vol_target_horizon", 10) or 10),
            # #304 dual-head: forward head_mode + regression_alpha so a
            # dual-head checkpoint rehydrates the same head config the
            # run trained against. Pre-#304 checkpoints leave the keys
            # absent and the defaults collapse to the canonical config
            # the ModelConfig dataclass ships with.
            head_mode=str(raw.get("head_mode", "dual") or "dual"),
            regression_alpha=float(raw.get("regression_alpha", 0.5)),
            # #423: mirror the #292 rates-fields landing for the #273
            # multi-task loss knob + the four per-axis lambda fields.
            # Pre-#273 checkpoints leave these absent and the defaults
            # collapse to the single-task CE path. Without this,
            # eval_checkpoint_directional / calibrate_regime_classifier
            # silently rebuild a multi_task_loss=False config from a
            # --multi-task-loss=on checkpoint.
            multi_task_loss=bool(raw.get("multi_task_loss", False)),
            multi_task_lambda_stance=float(raw.get("multi_task_lambda_stance", 1.0)),
            multi_task_lambda_certainty=float(raw.get("multi_task_lambda_certainty", 0.3)),
            multi_task_lambda_time=float(raw.get("multi_task_lambda_time", 0.3)),
            # #214: round-trip press-conf opt-in.
            use_press_conf=bool(raw.get("use_press_conf", False)),
            # #443/#444: forward the two new opt-in flags so a checkpoint
            # trained under ``--use-statement-delta`` / ``--use-vote-features``
            # rehydrates with the same per-bar input width on the eval /
            # calibration paths. Pre-#443 checkpoints leave the keys
            # absent and the defaults collapse to the byte-identical
            # legacy path.
            use_statement_delta=bool(raw.get("use_statement_delta", False)),
            use_vote_features=bool(raw.get("use_vote_features", False)),
            # #480 symbol-conditioned regime head. Default 0 keeps the
            # pre-#480 path byte-identical (no embedding module mounted,
            # no head-input widening).
            symbol_embedding_dim=int(raw.get("symbol_embedding_dim", 0) or 0),
        )
    return ModelConfig()


def _checkpoint_metadata(
    payload: dict[str, Any] | None,
    checkpoint_path: Path,
    *,
    runtime_mode: str = "fast",
    model: ForecasterBase | None = None,
    adaptation_summary: TrainingRunSummary | None = None,
) -> dict[str, Any]:
    model_config = (
        ModelConfig.from_model(model) if model is not None else _coerce_payload_config(payload)
    )
    payload_metrics = _metrics_from_payload(payload)
    metrics = (
        adaptation_summary.metrics
        if adaptation_summary and adaptation_summary.metrics
        else payload_metrics
    )
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
        "best_loss": payload_metrics.loss
        if payload_metrics
        else payload.get("best_loss")
        if isinstance(payload, dict)
        else None,
        "combined_rmse": payload_metrics.combined_rmse if payload_metrics else None,
        "adaptation_epochs_completed": adaptation_summary.epochs_completed
        if adaptation_summary
        else None,
        "adaptation_best_epoch": adaptation_summary.best_epoch if adaptation_summary else None,
        "adaptation_loss": metrics.loss if adaptation_summary and metrics else None,
        "adaptation_combined_rmse": metrics.combined_rmse
        if adaptation_summary and metrics
        else None,
        "decay_rate": decay_rate,
        "chunk_attention": None,
    }
    # #292 -- rates scalers + tertile edges live on training_summary so
    # the inference path can invert the per-head standardiser. None when
    # the checkpoint was trained without rates heads.
    if isinstance(payload, dict):
        ts = payload.get("training_summary")
        if isinstance(ts, dict):
            rates_scalers = ts.get("rates_scalers")
            rates_edges = ts.get("rates_quantile_edges")
            if rates_scalers:
                metadata["rates_scalers"] = rates_scalers
            if rates_edges:
                metadata["rates_quantile_edges"] = rates_edges
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
    model: ForecasterBase,
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
        "close_scale": float(close_scale)
        if close_scale is not None
        else float(DEFAULT_CLOSE_SCALE),
        "rich_feature_scaler": (
            rich_feature_scaler.to_dict() if rich_feature_scaler is not None else None
        ),
        "rng_state": _capture_rng_state(),
    }


def _save_model_checkpoint(
    model: ForecasterBase,
    checkpoint_path: Path,
    summary: TrainingRunSummary,
    *,
    close_scale: float | None = None,
    rich_feature_scaler: RichFeatureScalerParams | None = None,
    encoder_alias: str | None = None,
    inference_features: tuple[str, ...] = (),
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
    # #341: per-checkpoint inference contract sidecar. Lives next to the
    # ``.pt`` file so a downstream serving loader can refuse to bind a
    # checkpoint whose required kwargs the live serving signature does
    # not satisfy. The sidecar write is a soft step -- a failure here
    # logs + degrades so the training run still succeeds, but the
    # default is to emit one on every save so the deployed model and
    # the published model stay in lockstep.
    try:
        from app.training.inference_contract import (
            derive_contract,
            write_sidecar,
        )

        contract = derive_contract(
            model,
            encoder_alias=encoder_alias,
            inference_features=tuple(inference_features),
        )
        write_sidecar(contract, checkpoint_path)
    except Exception:  # pragma: no cover -- never let sidecar break training
        import logging as _logging

        _logging.getLogger(__name__).warning(
            "inference_contract_sidecar_write_failed path=%s",
            checkpoint_path,
            exc_info=True,
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


def _load_state_dict_loose(model: ForecasterBase, state_dict: dict[str, Any], source: str) -> None:
    """Load a checkpoint non-strictly and surface any missing/unexpected keys."""
    result = model.load_state_dict(state_dict, strict=False)
    missing = list(getattr(result, "missing_keys", []) or [])
    unexpected = list(getattr(result, "unexpected_keys", []) or [])
    if missing or unexpected:
        logger.warning(
            "checkpoint_load_mismatch source=%s missing=%s unexpected=%s",
            source,
            missing,
            unexpected,
        )


def _load_model_checkpoint(
    model: ForecasterBase,
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
