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
            "regime_loss_mode",
            "class_weight_power",
            # #502 focal + class_balanced loss-side hyperparameters.
            # The flat_mlp ctor does not consume them; the trainer reads
            # them off the stashed module attribute when constructing
            # the loss kernel.
            "focal_gamma",
            "class_balanced_beta",
            "regression_alpha",
            "use_derived_text_features",
            "rates_head_mode",
            "rates_aux_classification",
            "rates_alpha",
            "rates_target_mode",
            "vol_target_mode",
            "vol_target_horizon",
            # #472 vol-regime labelling mode + absolute thresholds are
            # loop / loader-side knobs; the flat_mlp ctor does not
            # consume them.
            "vol_regime_label_mode",
            "absolute_vol_thresholds",
            "use_regime_conditioning",
            "use_sep",
            "use_press_conf",
            "use_statement_delta",
            "use_vote_features",
            "use_vix_features",
            # #543 doc_length is a per-event scalar broadcast in the
            # recurrent path only; the flat_mlp ctor never widens its
            # input vector with it.
            "use_doc_length",
            # #480 symbol-conditioned regime head is research-only on the
            # recurrent class. The flat_mlp ctor does not consume it.
            "symbol_embedding_dim",
            # #471 multi-horizon aux regression heads. The flat_mlp ctor
            # does not mount the recurrent log-RV head's architecture,
            # so the aux heads are not wired into it either. Drop both
            # the head-list and the loss-side alpha so the kwargs
            # dispatch stays a strict shape contract.
            "aux_horizons",
            "aux_horizon_alpha",
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
        flat.regime_loss_mode = str(resolved.regime_loss_mode or "ce")  # type: ignore[assignment]
        flat.class_weight_power = float(resolved.class_weight_power)  # type: ignore[assignment]
        flat.focal_gamma = float(resolved.focal_gamma)  # type: ignore[assignment]
        flat.class_balanced_beta = float(resolved.class_balanced_beta)  # type: ignore[assignment]
        flat.regression_alpha = float(resolved.regression_alpha)  # type: ignore[assignment]
        flat.use_derived_text_features = bool(resolved.use_derived_text_features)  # type: ignore[assignment]
        flat.rates_heads = flat_rates_heads  # type: ignore[assignment]
        flat.rates_head_mode = str(resolved.rates_head_mode or "regression")  # type: ignore[assignment]
        flat.rates_aux_classification = bool(resolved.rates_aux_classification)  # type: ignore[assignment]
        flat.rates_alpha = float(resolved.rates_alpha)  # type: ignore[assignment]
        # #305 round-trip the rates target derivation onto the persisted
        # run summary so the checkpoint records which target the heads
        # were trained against.
        flat.rates_target_mode = str(
            getattr(resolved, "rates_target_mode", "raw") or "raw"
        )  # type: ignore[assignment]
        # #435 round-trip the forward-vol target derivation onto the
        # persisted run summary so the checkpoint records which target
        # the regression / dual head trained against.
        flat.vol_target_mode = str(
            getattr(resolved, "vol_target_mode", "raw") or "raw"
        )  # type: ignore[assignment]
        # Round-trip the supervised forward-vol horizon so
        # ``ModelConfig.from_model`` recovers it on resume.
        flat.vol_target_horizon = int(
            getattr(resolved, "vol_target_horizon", 10) or 10
        )  # type: ignore[assignment]
        # #472 round-trip the vol-regime labelling knobs onto the flat_mlp
        # module so the persisted run summary records which contract the
        # classification target trained under.
        from app.models.config import (
            DEFAULT_ABSOLUTE_VOL_THRESHOLDS,
            DEFAULT_VOL_REGIME_LABEL_MODE,
        )

        flat.vol_regime_label_mode = str(
            getattr(resolved, "vol_regime_label_mode", DEFAULT_VOL_REGIME_LABEL_MODE)
            or DEFAULT_VOL_REGIME_LABEL_MODE
        )  # type: ignore[assignment]
        _flat_thresholds_raw = getattr(
            resolved, "absolute_vol_thresholds", DEFAULT_ABSOLUTE_VOL_THRESHOLDS
        )
        if _flat_thresholds_raw is None:
            flat.absolute_vol_thresholds = DEFAULT_ABSOLUTE_VOL_THRESHOLDS  # type: ignore[assignment]
        else:
            _flat_seq = tuple(_flat_thresholds_raw)
            if len(_flat_seq) != 2:
                flat.absolute_vol_thresholds = DEFAULT_ABSOLUTE_VOL_THRESHOLDS  # type: ignore[assignment]
            else:
                flat.absolute_vol_thresholds = (  # type: ignore[assignment]
                    float(_flat_seq[0]),
                    float(_flat_seq[1]),
                )
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
    # #470 regime-loss mode is loss-side: the trainer reads it off the
    # stashed module attribute when constructing the CE / MultiTaskLoss
    # instance. Pop here so the ForecasterModel ctor does not see the
    # unrecognised kwarg.
    regime_loss_mode_value = str(kwargs.pop("regime_loss_mode", "ce") or "ce")
    class_weight_power = float(kwargs.pop("class_weight_power", 1.0))
    # #502 focal + class_balanced regime-loss hyperparameters. Pure
    # loss-side knobs the trainer reads off the stashed module
    # attribute when constructing the loss kernel; pop here so the
    # ForecasterModel ctor does not see the unrecognised kwargs.
    focal_gamma_value = float(kwargs.pop("focal_gamma", 2.0))
    class_balanced_beta_value = float(kwargs.pop("class_balanced_beta", 0.999))
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
    rates_aux_classification_flag = bool(
        kwargs.pop("rates_aux_classification", False)
    )
    rates_alpha_value = float(kwargs.pop("rates_alpha", 0.5))
    rates_target_mode_value = str(
        kwargs.pop("rates_target_mode", "raw") or "raw"
    )
    # #435 forward-vol target derivation. Loop-side knob; the trainer
    # reads it off the stashed module attribute when materialising the
    # regression target tensor. Pop here so the ForecasterModel ctor
    # does not see the unrecognised kwarg.
    vol_target_mode_value = str(
        kwargs.pop("vol_target_mode", "raw") or "raw"
    )
    # #472 vol-regime labelling mode + absolute thresholds. Both are
    # loop / loader-side knobs (the trainer selects between
    # ``fit_vol_regime_quantiles`` and the fixed thresholds; the loader
    # uses the resulting cutoffs in ``vol_regime_class_for``). Pop here
    # so the ForecasterModel ctor never sees the unrecognised kwargs.
    # Stash back on the built module so ``ModelConfig.from_model``
    # round-trips both onto the persisted run summary regardless of
    # whether the run also wires the absolute branch.
    from app.models.config import (
        DEFAULT_ABSOLUTE_VOL_THRESHOLDS,
        DEFAULT_VOL_REGIME_LABEL_MODE,
    )

    vol_regime_label_mode_value = str(
        kwargs.pop("vol_regime_label_mode", DEFAULT_VOL_REGIME_LABEL_MODE)
        or DEFAULT_VOL_REGIME_LABEL_MODE
    )
    _absolute_thresholds_raw = kwargs.pop(
        "absolute_vol_thresholds", DEFAULT_ABSOLUTE_VOL_THRESHOLDS
    )
    if _absolute_thresholds_raw is None:
        absolute_vol_thresholds_value: tuple[float, float] = DEFAULT_ABSOLUTE_VOL_THRESHOLDS
    else:
        _seq = tuple(_absolute_thresholds_raw)
        if len(_seq) != 2:
            absolute_vol_thresholds_value = DEFAULT_ABSOLUTE_VOL_THRESHOLDS
        else:
            absolute_vol_thresholds_value = (float(_seq[0]), float(_seq[1]))
    # Supervised forward-vol horizon. Loader-side knob (the loader
    # routes the per-row ``forward_realized_vol_10d`` slot to the
    # chosen column); the model ctor does not consume it. Stash it back
    # on the built module so ``ModelConfig.from_model`` round-trips it
    # onto the persisted run summary.
    vol_target_horizon_value = int(
        kwargs.pop("vol_target_horizon", 10) or 10
    )
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
    # The CE term in the rates loss needs an aux classifier to land on.
    # Reject the joint-loss configuration where the classifier was
    # explicitly disabled but the loss mode still expects CE; surface
    # the misconfiguration early instead of silently dropping the term.
    if (
        rates_heads_tuple
        and rates_head_mode_value in {"classification", "dual"}
        and not rates_aux_classification_flag
    ):
        raise ValueError(
            f"rates_head_mode={rates_head_mode_value!r} requires "
            "--rates-classification-heads (the CE term has no aux head "
            "to land on otherwise). Either pass "
            "--rates-classification-heads or set "
            "--rates-head-mode=regression."
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
    # #307 macro-regime conditioning toggle. Forwarded to the model
    # constructor (research and serving classes both accept it via the
    # shared ``ForecasterBase`` super().__init__) so the gating layer
    # mounts at build time when the flag is on. Default ``False`` keeps
    # the no-gate forward byte-identical for every existing checkpoint.
    use_regime_conditioning_flag = bool(kwargs.pop("use_regime_conditioning", False))
    # #215 SEP dot-plot toggle. Forwarded to the model constructor (both
    # research and serving classes accept it via the shared ``ForecasterBase``
    # super().__init__) so the recurrent core widens its input projection
    # at build time when the flag is on. Default ``False`` keeps every
    # existing checkpoint byte-identical.
    use_sep_flag = bool(kwargs.pop("use_sep", False))
    # #214 press-conf opt-in.
    use_press_conf_flag = bool(kwargs.pop("use_press_conf", False))
    # #443/#444 statement-delta + vote-features opt-in flags.
    use_statement_delta_flag = bool(kwargs.pop("use_statement_delta", False))
    use_vote_features_flag = bool(kwargs.pop("use_vote_features", False))
    # #478 VIX term-structure + VRP opt-in flag.
    use_vix_features_flag = bool(kwargs.pop("use_vix_features", False))
    # #543 doc_length per-event scalar opt-in flag.
    use_doc_length_flag = bool(kwargs.pop("use_doc_length", False))
    # #480 symbol-conditioned regime head. Pop here so the serving
    # constructor (which does not accept the kwarg in v1) does not
    # receive it. The research class consumes the kwarg directly to
    # mount the embedding and widen the regime / log-RV head input.
    # Serving wiring is deferred to a follow-up alongside the
    # response-surface picker. Default 0 keeps the legacy path
    # byte-identical (no embedding module, no widening).
    symbol_embedding_dim_value = int(kwargs.pop("symbol_embedding_dim", 0) or 0)
    # #471 multi-horizon aux regression heads. Pop here so the serving
    # constructor (which does not accept the kwarg) does not receive
    # it. The research class consumes ``aux_horizons`` directly to
    # mount one parallel log-RV head per horizon; ``aux_horizon_alpha``
    # is a loss-side knob the training loop reads off the stashed
    # module attribute, so the model ctor never sees it.
    aux_horizons_value: tuple[int, ...] = tuple(
        int(v) for v in kwargs.pop("aux_horizons", ()) or ()
    )
    aux_horizon_alpha_value = float(kwargs.pop("aux_horizon_alpha", 0.3))
    # Validate aux horizons are in the supported set and don't include
    # the canonical primary (10). Reject early instead of mounting heads
    # that the loss path cannot supervise. The empty-tuple default
    # short-circuits cleanly.
    if aux_horizons_value:
        from app.models.config import SUPPORTED_VOL_TARGET_HORIZONS
        invalid = [
            h for h in aux_horizons_value
            if h not in SUPPORTED_VOL_TARGET_HORIZONS or h == 10
        ]
        if invalid:
            raise ValueError(
                f"aux_horizons={aux_horizons_value} contains unsupported entries "
                f"{invalid}. Allowed: any non-empty subset of "
                f"{tuple(h for h in SUPPORTED_VOL_TARGET_HORIZONS if h != 10)}."
            )
        # Aux heads share the architecture of the primary log-RV head
        # (head_mode in {regression, dual}). Reject classification mode
        # so the misconfiguration surfaces before training fires.
        head_mode_value = str(kwargs.get("head_mode", "dual") or "dual")
        if head_mode_value == "classification":
            raise ValueError(
                f"aux_horizons={aux_horizons_value} requires head_mode in "
                "{'regression', 'dual'} (the aux heads share the architecture "
                f"of the primary log-RV head). Got head_mode={head_mode_value!r}."
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
            rates_aux_classification=rates_aux_classification_flag,
            use_regime_conditioning=use_regime_conditioning_flag,
            use_sep=use_sep_flag,
            use_press_conf=use_press_conf_flag,
            use_statement_delta=use_statement_delta_flag,
            use_vote_features=use_vote_features_flag,
            use_vix_features=use_vix_features_flag,
            use_doc_length=use_doc_length_flag,
            **kwargs,
        )
    else:
        model = ForecasterResearchModel(
            model_type=architecture,
            rates_heads=rates_heads_tuple,
            rates_aux_classification=rates_aux_classification_flag,
            use_regime_conditioning=use_regime_conditioning_flag,
            use_sep=use_sep_flag,
            use_press_conf=use_press_conf_flag,
            use_statement_delta=use_statement_delta_flag,
            use_vote_features=use_vote_features_flag,
            use_vix_features=use_vix_features_flag,
            use_doc_length=use_doc_length_flag,
            symbol_embedding_dim=symbol_embedding_dim_value,
            aux_horizons=aux_horizons_value,
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
    model.regime_loss_mode = regime_loss_mode_value  # type: ignore[assignment]
    model.class_weight_power = class_weight_power  # type: ignore[assignment]
    model.focal_gamma = focal_gamma_value  # type: ignore[assignment]
    model.class_balanced_beta = class_balanced_beta_value  # type: ignore[assignment]
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
    model.rates_aux_classification = rates_aux_classification_flag
    model.rates_alpha = rates_alpha_value  # type: ignore[assignment]
    # #305 round-trip the rates target derivation onto the built module
    # so ``ModelConfig.from_model`` recovers it on resume / inference.
    model.rates_target_mode = rates_target_mode_value  # type: ignore[assignment]
    # #435 round-trip the forward-vol target derivation onto the built
    # module so ``ModelConfig.from_model`` recovers it on resume.
    model.vol_target_mode = vol_target_mode_value  # type: ignore[assignment]
    model.vol_target_horizon = vol_target_horizon_value  # type: ignore[assignment]
    # #472 round-trip the vol-regime labelling knobs so a resumed
    # checkpoint reuses the same calm / normal / high contract the
    # original run trained under.
    model.vol_regime_label_mode = vol_regime_label_mode_value  # type: ignore[assignment]
    model.absolute_vol_thresholds = absolute_vol_thresholds_value  # type: ignore[assignment]
    # #443/#444 round-trip the two new opt-in flags. Default-off path
    # behaves byte-identically; flag-on a future sweep that resumes off
    # this checkpoint rebuilds with the same loader-tail widths. The
    # ctor wires these as ``ForecasterBase`` attrs already; the explicit
    # assignment here covers the flat_mlp path (which doesn't go through
    # the recurrent base) and serves as the round-trip source the
    # persisted run summary reads via ``ModelConfig.from_model``.
    model.use_statement_delta = use_statement_delta_flag
    model.use_vote_features = use_vote_features_flag
    model.use_vix_features = use_vix_features_flag
    model.use_doc_length = use_doc_length_flag
    # #480 round-trip the symbol-embedding dim so
    # ``ModelConfig.from_model`` recovers it on resume. The research
    # class set this in its ctor; stash it on the serving instance too
    # so the persisted checkpoint payload carries the dim regardless of
    # which role built the module.
    if not hasattr(model, "symbol_embedding_dim"):
        model.symbol_embedding_dim = symbol_embedding_dim_value
    # #471 round-trip the aux-horizons tuple + alpha so
    # ``ModelConfig.from_model`` recovers them on resume. The research
    # class set ``aux_horizons`` in its ctor; the serving class never
    # mounts the heads, so the tuple stashes empty there. ``aux_horizon_alpha``
    # is loss-side only -- the model ctor never sees it.
    if not hasattr(model, "aux_horizons"):
        model.aux_horizons = aux_horizons_value
    model.aux_horizon_alpha = aux_horizon_alpha_value  # type: ignore[assignment]
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
