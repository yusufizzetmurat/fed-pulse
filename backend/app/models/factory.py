"""Forecaster architecture factory.

The eight architectures (``lstm``, ``lstm_attn``, ``gru``, ``tcn``,
``transformer``, ``dlinear``, ``informer``, ``tft``) are all wired through
the shared :class:`app.models.forecaster_base.ForecasterBase` backbone via
the ``model_type`` constructor argument. ``role="research"`` returns a
:class:`ForecasterResearchModel` (all knobs, training entrypoint);
``role="serving"`` returns a :class:`ForecasterServingModel` (frozen
surface, /analyze entrypoint, regression-canonical default per ADR 0015).
Issue #336 split the legacy 712-line ``ForecasterModel`` into the two
classes and pulled the input-prep onto :func:`prepare_recurrent_input`.

The default ``architecture="lstm"`` research path is byte-identical to the
pre-#336 ``ForecasterModel()`` construction so the determinism regression
at ``tests/regression/test_forecaster_determinism.py`` stays green.

When ``config.fusion_mode == "gated_infonce"`` (#235), the factory
dispatches to :class:`MultiModalForecasterModel` instead -- the two
modalities stay separate until the gated fusion stage so the InfoNCE
alignment loss has separable representations to pull together. Gated-
InfoNCE is research-only and rejects ``role="serving"``.
"""

from __future__ import annotations

import warnings
from typing import Any, Literal

from app.models.config import (
    FORECASTER_ARCHITECTURES,
    TFT_EXCLUSION_REASON,
    ModelConfig,
)
from app.models.flat_mlp import ForecasterFlatMLP
from app.models.multimodal_forecaster import MultiModalForecasterModel
from app.models.research_model import ForecasterResearchModel
from app.models.serving_model import ForecasterServingModel

# Back-compat alias. Pre-#336 the factory return type was
# ``ForecasterModel | MultiModalForecasterModel``; post-split the alias
# resolves to the research class so existing call sites keep type-checking.
ForecasterModel = ForecasterResearchModel

Role = Literal["research", "serving"]


def build_forecaster(
    config: ModelConfig | dict[str, Any],
    *,
    role: Role = "research",
) -> (
    ForecasterResearchModel
    | ForecasterServingModel
    | MultiModalForecasterModel
    | ForecasterFlatMLP
):
    """Build a forecaster module for ``config.architecture``.

    All architectures share the same input contract ``(batch, seq_len, 6)``
    and output contract ``(batch, 2)`` (close, volatility).  The wrapper
    dispatches the recurrent core internally via ``model_type``; this
    function exists to translate the public ``architecture`` field to the
    wrapper's internal ``model_type`` argument and to raise a clear error
    when the architecture string is unknown.
    """

    resolved = config if isinstance(config, ModelConfig) else ModelConfig(**config)
    architecture = resolved.architecture
    if architecture not in FORECASTER_ARCHITECTURES:
        raise ValueError(
            f"Unknown architecture: {architecture!r}. "
            f"Allowed: {sorted(FORECASTER_ARCHITECTURES)}"
        )
    # TFT stays importable + checkpoint-loadable for back-compat, but is
    # excluded from canonical sweep targets. Surface a DeprecationWarning
    # whenever the factory is asked to build one so a future sweep that
    # mis-includes TFT does not silently regress the comparison.
    if architecture == "tft":
        warnings.warn(
            TFT_EXCLUSION_REASON,
            DeprecationWarning,
            stacklevel=2,
        )

    if role not in {"research", "serving"}:
        raise ValueError(
            f"Unknown role: {role!r}. Allowed: research, serving"
        )

    if resolved.fusion_mode == "gated_infonce":
        if role == "serving":
            raise ValueError(
                "fusion_mode='gated_infonce' is research-only; promote a checkpoint "
                "to the serving class via scripts.promote_checkpoint instead."
            )
        if resolved.output_mode != "classification":
            raise ValueError(
                "fusion_mode='gated_infonce' requires output_mode='classification' "
                f"(got {resolved.output_mode!r}); InfoNCE-paired regression is out of scope."
            )
        text_dim = int(resolved.text_embedding_dim or 0)
        if text_dim <= 0:
            raise ValueError(
                "fusion_mode='gated_infonce' requires text_embedding_dim > 0; "
                "set --text-encoder so the loader emits pooled embeddings."
            )
        multimodal = MultiModalForecasterModel(
            market_input_size=int(resolved.input_size),
            text_embedding_dim=text_dim,
            latent_dim=int(resolved.infonce_latent_dim),
            hidden_size=int(resolved.hidden_size),
            num_layers=int(resolved.num_layers),
            dropout=float(resolved.dropout),
            head_hidden_size=int(resolved.head_hidden_size),
            architecture=architecture,
            n_classes=int(resolved.n_classes),
        )
        # Stash the InfoNCE hyperparameters on the module so
        # ``ModelConfig.from_model`` round-trips them and the training
        # loop can read them without an extra config plumbing pass.
        multimodal.fusion_mode = "gated_infonce"  # type: ignore[assignment]
        multimodal.infonce_lambda = float(resolved.infonce_lambda)  # type: ignore[assignment]
        multimodal.infonce_temperature = float(resolved.infonce_temperature)  # type: ignore[assignment]
        multimodal.infonce_latent_dim = int(resolved.infonce_latent_dim)  # type: ignore[assignment]
        return multimodal

    # #327 Arm B. ``flat_mlp`` is the no-sequence-wrap comparator on the
    # text path. It is research-only (the broadcast-static methodology
    # question is not a serving concern); a serving dispatch on it
    # would skip the recurrent core which the /analyze contract still
    # exposes through the close/vol time series.
    if architecture == "flat_mlp":
        if role == "serving":
            raise ValueError(
                "architecture='flat_mlp' is research-only (issue #327 Arm B); "
                "the comparator runs against the canonical recurrent forecaster."
            )
        flat_kwargs = resolved.to_dict()
        flat_kwargs.pop("architecture", None)
        # Drop fields the flat_mlp ctor does not consume so the kwargs
        # dispatch stays a strict shape contract.
        for drop in (
            "fusion_mode",
            "infonce_lambda",
            "infonce_temperature",
            "infonce_latent_dim",
            "lr_schedule",
            "sequence_length",
            "encoder_lora",
            "lora_curriculum_freeze_epoch",
            "multi_task_loss",
            "multi_task_lambda_stance",
            "multi_task_lambda_factor",
            "multi_task_lambda_certainty",
            "multi_task_lambda_topic",
            "class_weight_power",
            "regression_alpha",
            "use_derived_text_features",
            "rates_head_mode",
            "rates_alpha",
        ):
            flat_kwargs.pop(drop, None)
        flat_rates_heads = tuple(
            str(v) for v in flat_kwargs.pop("rates_heads", ()) or ()
        )
        if flat_rates_heads and resolved.output_mode == "regression":
            raise ValueError(
                "rates_heads can only be mounted alongside "
                "output_mode='classification' (current: 'regression'). "
                "Pass --output-mode classification or --rates-heads none."
            )
        flat = ForecasterFlatMLP(
            model_type=architecture,
            rates_heads=flat_rates_heads,
            **flat_kwargs,
        )
        # Stash the loss-side / training-loop flags on the module so
        # ``ModelConfig.from_model`` round-trips them onto the persisted
        # run summary the same way the recurrent path does.
        flat.encoder_lora = bool(resolved.encoder_lora)  # type: ignore[assignment]
        flat.lora_curriculum_freeze_epoch = resolved.lora_curriculum_freeze_epoch  # type: ignore[assignment]
        flat.multi_task_loss = bool(resolved.multi_task_loss)  # type: ignore[assignment]
        flat.multi_task_lambda_stance = float(resolved.multi_task_lambda_stance)  # type: ignore[assignment]
        flat.multi_task_lambda_factor = float(resolved.multi_task_lambda_factor)  # type: ignore[assignment]
        flat.multi_task_lambda_certainty = float(resolved.multi_task_lambda_certainty)  # type: ignore[assignment]
        flat.multi_task_lambda_topic = float(resolved.multi_task_lambda_topic)  # type: ignore[assignment]
        flat.class_weight_power = float(resolved.class_weight_power)  # type: ignore[assignment]
        flat.regression_alpha = float(resolved.regression_alpha)  # type: ignore[assignment]
        flat.use_derived_text_features = bool(resolved.use_derived_text_features)  # type: ignore[assignment]
        flat.rates_heads = flat_rates_heads  # type: ignore[assignment]
        flat.rates_head_mode = str(resolved.rates_head_mode or "regression")  # type: ignore[assignment]
        flat.rates_alpha = float(resolved.rates_alpha)  # type: ignore[assignment]
        return flat

    kwargs = resolved.to_dict()
    kwargs.pop("architecture", None)
    # Multi-modal-only fields stay on the ModelConfig for round-tripping
    # but the legacy ForecasterModel constructor does not consume them.
    kwargs.pop("fusion_mode", None)
    kwargs.pop("infonce_lambda", None)
    kwargs.pop("infonce_temperature", None)
    kwargs.pop("infonce_latent_dim", None)
    # Multi-task loss (#273) is a training-loop concern; the existing
    # MultiTaskHead is already wired on the ForecasterModel (since #272)
    # in classification mode. These knobs stash on the built module so
    # ``ModelConfig.from_model`` recovers them when the checkpoint is
    # loaded for resume / inference.
    multi_task_loss_flag = bool(kwargs.pop("multi_task_loss", False))
    multi_task_lambda_stance = float(kwargs.pop("multi_task_lambda_stance", 1.0))
    multi_task_lambda_factor = float(kwargs.pop("multi_task_lambda_factor", 0.3))
    multi_task_lambda_certainty = float(kwargs.pop("multi_task_lambda_certainty", 0.3))
    multi_task_lambda_topic = float(kwargs.pop("multi_task_lambda_topic", 0.3))
    class_weight_power = float(kwargs.pop("class_weight_power", 1.0))
    # #304 dual-head methodology. ``regression_alpha`` is a loss-side
    # knob (the training loop reads it from the model attribute);
    # ``head_mode`` is forwarded to the ForecasterModel constructor so
    # the regression_head mounts correctly. ``use_derived_text_features``
    # (#309) is a loader-side flag, only stashed on the module for
    # round-tripping through ModelConfig.from_model.
    regression_alpha_value = float(kwargs.pop("regression_alpha", 0.5))
    use_derived_text_features_flag = bool(
        kwargs.pop("use_derived_text_features", True)
    )
    # #292 rates heads. ``rates_heads`` is a tuple of head short-names
    # forwarded to the ForecasterModel constructor so the rates regression
    # + aux classification heads mount on the shared encoder. The mode +
    # alpha knobs are loss-side concerns (the training loop reads them
    # off the stashed module attribute); stash them so
    # ``ModelConfig.from_model`` round-trips them onto the persisted
    # summary regardless of whether the run also wires the loss.
    rates_heads_tuple = tuple(
        str(v) for v in kwargs.pop("rates_heads", ()) or ()
    )
    rates_head_mode_value = str(
        kwargs.pop("rates_head_mode", "regression") or "regression"
    )
    rates_alpha_value = float(kwargs.pop("rates_alpha", 0.5))
    # #317 finding #8: fail fast at the factory rather than silently
    # zeroing rates_heads when output_mode='regression'. The operator
    # gets a clear error message instead of a checkpoint that
    # advertises rates_heads in its config but mounts no rates heads.
    if rates_heads_tuple and resolved.output_mode == "regression":
        raise ValueError(
            "rates_heads can only be mounted alongside "
            "output_mode='classification' (current: 'regression'). "
            "Pass --output-mode classification or --rates-heads none."
        )
    # Phase 9 V2 (#195) fields all forwarded: ``output_mode`` /
    # ``n_classes`` drive the head shape; ``vol_regime_quantiles`` /
    # ``vol_regime_target`` ride on the module so the checkpoint
    # round-trips the per-fold boundaries via ``ModelConfig.from_model``.
    # Phase B (#227) fields ride on the ``ModelConfig`` so the checkpoint
    # carries the schedule + sequence-length choice into resume, but the
    # underlying ``ForecasterModel`` does not consume them on
    # construction (the training loop reads them from the config).
    kwargs.pop("lr_schedule", None)
    kwargs.pop("sequence_length", None)
    # Round 5 (#244) LoRA toggle is a training-loop concern; the
    # ``ForecasterModel`` consumes its text input from a per-batch
    # tensor regardless of whether the encoder ran statically (parquet
    # cache) or per-batch (LoRA-wrapped tower). The constructor does
    # not need to know about it, but we stash the value as a plain
    # attribute on the built module so ``ModelConfig.from_model``
    # (called when the run summary is serialised) reflects what
    # actually trained — without it, every persisted summary shows
    # ``encoder_lora=False`` even on an active LoRA cell.
    encoder_lora_flag = bool(kwargs.pop("encoder_lora", False))
    # Bundle B LoRA freeze curriculum is a training-loop concern too;
    # the recurrent ``ForecasterModel`` constructor never sees it. The
    # value is stashed back on the built module so ``ModelConfig.from_model``
    # round-trips it onto the persisted run summary the same way
    # ``encoder_lora`` does.
    lora_curriculum_freeze_epoch_val = kwargs.pop(
        "lora_curriculum_freeze_epoch", None
    )
    model: ForecasterResearchModel | ForecasterServingModel
    if role == "serving":
        # Serving construction trims the loss-side / sweep-side knobs the
        # narrow class does not consume. The state_dict shape is identical
        # to the research class for shared backbone + adapter keys; the
        # narrower forward path is what makes the class an inference-only
        # surface.
        model = ForecasterServingModel(
            model_type=architecture,
            rates_heads=rates_heads_tuple,
            **kwargs,
        )
    else:
        model = ForecasterResearchModel(
            model_type=architecture,
            rates_heads=rates_heads_tuple,
            **kwargs,
        )
    # mypy reads ``nn.Module`` attribute writes as ``Tensor | Module``;
    # the LoRA flag is a plain bool stashed for ``from_model`` to read
    # back, so suppress the noise rather than register a fake buffer.
    model.encoder_lora = encoder_lora_flag  # type: ignore[assignment]
    model.lora_curriculum_freeze_epoch = lora_curriculum_freeze_epoch_val
    model.multi_task_loss = multi_task_loss_flag  # type: ignore[assignment]
    model.multi_task_lambda_stance = multi_task_lambda_stance  # type: ignore[assignment]
    model.multi_task_lambda_factor = multi_task_lambda_factor  # type: ignore[assignment]
    model.multi_task_lambda_certainty = multi_task_lambda_certainty  # type: ignore[assignment]
    model.multi_task_lambda_topic = multi_task_lambda_topic  # type: ignore[assignment]
    model.class_weight_power = class_weight_power  # type: ignore[assignment]
    # #304 / #309 -- stash the loss + loader flags so
    # ``ModelConfig.from_model`` round-trips them onto the persisted
    # checkpoint payload. ``head_mode`` itself was already passed to
    # the ForecasterModel constructor above; the alpha + derived-text
    # flag are training-loop / loader concerns and stay as attributes.
    model.regression_alpha = regression_alpha_value  # type: ignore[assignment]
    model.use_derived_text_features = use_derived_text_features_flag  # type: ignore[assignment]
    # #292 stash the loss-side knobs so ``ModelConfig.from_model`` round-
    # trips them onto the persisted run summary.
    model.rates_heads = rates_heads_tuple  # type: ignore[assignment]
    model.rates_head_mode = rates_head_mode_value  # type: ignore[assignment]
    model.rates_alpha = rates_alpha_value  # type: ignore[assignment]
    return model


def build_research_forecaster(
    config: ModelConfig | dict[str, Any],
) -> ForecasterResearchModel | MultiModalForecasterModel | ForecasterFlatMLP:
    """Convenience entrypoint for the training / sweep / research callers."""
    model = build_forecaster(config, role="research")
    assert not isinstance(model, ForecasterServingModel)
    return model


def build_serving_forecaster(
    config: ModelConfig | dict[str, Any],
) -> ForecasterServingModel:
    """Convenience entrypoint for /analyze cold-start + checkpoint load."""
    model = build_forecaster(config, role="serving")
    assert isinstance(model, ForecasterServingModel)
    return model


__all__ = [
    "ForecasterModel",
    "build_forecaster",
    "build_research_forecaster",
    "build_serving_forecaster",
]
