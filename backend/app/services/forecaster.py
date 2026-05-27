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
import logging
import math
import threading
from datetime import date as _date_cls
from pathlib import Path
from collections.abc import Iterable
from typing import Any

import torch

logger = logging.getLogger(__name__)

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
from app.models.lstm import ForecasterModel  # noqa: F401 -- back-compat re-export
from app.models.research_model import ForecasterResearchModel  # noqa: F401
from app.models.serving_model import ForecasterServingModel
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
from app.services.regime_bucketing import bucket_log_rv, derive_distribution

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


_model: ForecasterServingModel | None = None
_model_artifact_metadata: dict[str, Any] | None = None
_model_lock = threading.Lock()

# #341: structured surface for the loader's contract validation. The
# /health endpoint reads this to surface "checkpoint incompatible with
# serving signature" without 5xx'ing the request path. Populated by
# ``_get_model`` on the cold-load path; stays at the optimistic default
# until the first /analyze (or until a /health probe fires
# ``ensure_serving_contract_validated``).
_serving_contract_status: dict[str, Any] = {
    "status": "uninitialised",
    "checkpoint_path": None,
    "missing_kwargs": [],
    "extra_kwargs": [],
    "message": "",
}

# #341: in-process counters so an operator can grep the structured
# error surface without standing up Prometheus. Kept module-level so
# the FastAPI test client and the /health endpoint can read the same
# numbers a long-running production process accumulates. Reset on
# process restart by design -- the structured logs alongside the
# increment are the durable record.
_contract_counters: dict[str, int] = {
    "regime_classification_inference_kwarg_missing": 0,
    "regime_classification_unexpected_exception": 0,
    "market_reaction_inference_kwarg_missing": 0,
    "market_reaction_unexpected_exception": 0,
}


def get_serving_contract_status() -> dict[str, Any]:
    """Return the cached contract-validation surface for /health.

    The cold-load path in ``_get_model`` writes into the module-level
    ``_serving_contract_status`` dict; this helper returns a shallow
    copy so callers cannot mutate the cache. A status of ``ok`` means
    the checkpoint's declared kwargs are a subset of the serving
    signature; ``serving_signature_missing_kwargs`` /
    ``registry_inference_features_mismatch`` means the loader refused
    to bind and ``_model`` is still ``None``.
    """

    return dict(_serving_contract_status)


def get_contract_counters() -> dict[str, int]:
    """Return a snapshot of the structured-error increment counters."""

    return dict(_contract_counters)


def reset_singleton_for_revalidation() -> None:
    """#342: drop the cached singleton so the next ``_get_model`` cold-loads.

    Used by ``_bootstrap_cold_start`` after the bootstrap-write path:
    ``_set_singleton_after_train`` bypasses ``_validate_serving_contract``,
    so the cold-start needs to round-trip through the canonical loader
    to actually validate the freshly written sidecar. Lives here (not
    in main.py) so the private singleton attributes stay encapsulated;
    a future refactor of the storage shape only needs to update this
    helper.
    """

    global _model, _model_artifact_metadata
    with _model_lock:
        _model = None
        _model_artifact_metadata = None


def _extract_missing_kwarg_from_typeerror(exc: TypeError) -> str | None:
    """Parse the kwarg name out of a python ``TypeError`` message.

    Mirrors :func:`app.main._extract_missing_kwarg_from_typeerror`; lives
    here so the forecaster service has a local helper for its own
    structured surface and the two functions stay in lockstep.
    """

    import re

    message = str(exc)
    match = re.search(
        r"keyword[- ]?(?:only )?argument[s]?:?\s*['\"]([^'\"]+)['\"]", message
    )
    if match:
        return match.group(1)
    match = re.search(
        r"unexpected keyword argument\s*['\"]([^'\"]+)['\"]", message
    )
    if match:
        return match.group(1)
    return None


def _record_contract_status(
    *,
    status: str,
    checkpoint_path: Any,
    missing_kwargs: tuple[str, ...] = (),
    extra_kwargs: tuple[str, ...] = (),
    message: str = "",
) -> None:
    _serving_contract_status.update(
        {
            "status": str(status),
            "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
            "missing_kwargs": list(missing_kwargs),
            "extra_kwargs": list(extra_kwargs),
            "message": str(message),
        }
    )


def _validate_serving_contract(
    checkpoint_path: Path,
) -> tuple[bool, str]:
    """Cross-check the sidecar against the serving signature + registry.

    Returns ``(ok, status)``. ``ok=False`` means the serving loader
    must refuse to bind the checkpoint. A missing sidecar degrades to
    ``ok=True`` with status ``"sidecar_absent"`` so pre-#341
    checkpoints continue to load -- the contract is a soft surface for
    legacy artefacts and a hard surface for any checkpoint written
    under the #341 contract. (#374: the prior ``payload`` parameter
    was never read; the sidecar is what we validate, not the .pt
    payload.)
    """

    from app.training.inference_contract import (
        read_sidecar,
        validate_against_serving,
    )

    contract = read_sidecar(checkpoint_path)
    if contract is None:
        _record_contract_status(
            status="sidecar_absent",
            checkpoint_path=checkpoint_path,
            message=(
                "no inference_contract.json sidecar next to checkpoint; "
                "treating as pre-#341 legacy artefact"
            ),
        )
        return True, "sidecar_absent"

    registry_features: tuple[str, ...] | None = None
    if contract.encoder_alias:
        try:
            from app.models.registry import encoder_ref

            ref = encoder_ref(contract.encoder_alias)
            if ref is not None:
                registry_features = tuple(ref.inference_features)
        except Exception:  # noqa: BLE001 -- defensive
            registry_features = None

    validation = validate_against_serving(
        contract,
        serving_model_cls=ForecasterServingModel,
        registry_inference_features=registry_features,
    )
    if not validation.ok:
        _record_contract_status(
            status=validation.status,
            checkpoint_path=checkpoint_path,
            missing_kwargs=validation.missing_kwargs,
            extra_kwargs=validation.extra_kwargs,
            message=validation.message,
        )
        logger.error(
            "checkpoint_inference_contract_mismatch path=%s status=%s missing=%s extra=%s",
            checkpoint_path,
            validation.status,
            list(validation.missing_kwargs),
            list(validation.extra_kwargs),
        )
        return False, validation.status

    _record_contract_status(
        status="ok",
        checkpoint_path=checkpoint_path,
        message="inference contract matches serving signature",
    )
    return True, "ok"


def _get_model() -> ForecasterServingModel:
    """Resolve the cached serving singleton; cold-load from disk if absent.

    Issue #336 split the model classes. The /analyze inference path
    holds a :class:`ForecasterServingModel`; the state_dict on disk is
    written by the research class but the two share the same key
    layout (the backbone + adapter weights live on
    :class:`ForecasterBase`, and the heads use the same names), so a
    research checkpoint loads cleanly into a serving model under
    ``load_state_dict(strict=False)``. The promotion contract in
    :mod:`scripts.promote_checkpoint` should still run on every
    research artefact before it is served -- this loose-load is the
    safety net for legacy checkpoints written before the promotion
    contract existed.
    """
    global _model, _model_artifact_metadata
    if _model is not None:
        return _model

    with _model_lock:
        if _model is None:
            from app.models.factory import build_serving_forecaster
            from app.training.loop import _coerce_model_config

            device = _resolve_device()
            payload = _read_checkpoint_payload(BEST_MODEL_PATH, device)
            # #341: contract validation runs BEFORE state-dict load so a
            # mismatched sidecar refuses to bind instead of allowing a
            # partial bind + a later RuntimeError on /analyze. Pre-#341
            # checkpoints with no sidecar degrade to "sidecar_absent"
            # + ok=True so the legacy serving fleet keeps working.
            ok, _status = _validate_serving_contract(BEST_MODEL_PATH)
            if not ok:
                raise RuntimeError(
                    "checkpoint inference contract incompatible with serving "
                    f"signature: {_status}"
                )
            raw_config = (
                payload.get("model_config") if isinstance(payload, dict) else None
            )
            resolved = _coerce_model_config(raw_config)
            model = build_serving_forecaster(resolved).to(device)
            if payload is not None:
                _load_state_dict_loose(
                    model, payload["model_state_dict"], str(BEST_MODEL_PATH)
                )
            model.eval()  # set inference mode
            _model = model
            _model_artifact_metadata = _checkpoint_metadata(
                payload, BEST_MODEL_PATH, model=model
            )
    return _model


def _set_singleton_after_train(
    work_model: Any,
    checkpoint_target: Path,
    device_obj: torch.device,
) -> None:
    """Refresh the in-process singleton + metadata after training writes a checkpoint.

    The training loop hands over a research-side module
    (:class:`ForecasterResearchModel` or :class:`MultiModalForecasterModel`);
    issue #336 routes /analyze through :class:`ForecasterServingModel`,
    so we build a serving instance from the persisted ``model_config``
    and load the freshly trained state into it. The state_dict keys are
    shared verbatim through :class:`ForecasterBase`, so this is the
    in-memory equivalent of the
    :mod:`scripts.promote_checkpoint` promotion contract.
    """
    global _model, _model_artifact_metadata
    from app.models.factory import build_serving_forecaster
    from app.models.multimodal_forecaster import MultiModalForecasterModel

    with _model_lock:
        payload = _read_checkpoint_payload(checkpoint_target, device_obj)
        raw_config = (
            payload.get("model_config") if isinstance(payload, dict) else None
        )
        # Multi-modal (gated-InfoNCE) research-side checkpoints are
        # research-only by construction: the serving class does not
        # mount the InfoNCE alignment head. Fall back to a deep-copy of
        # the research module so the singleton at least stays a
        # working forward path until a serving-shaped artefact lands.
        if isinstance(work_model, MultiModalForecasterModel):
            # The static type of ``_model`` is ``ForecasterServingModel``;
            # the multimodal fallback writes a research-side module so
            # the in-process /analyze handler keeps a working forward
            # pass on an InfoNCE-trained run. Promoted artefacts come
            # back through the standard build_serving_forecaster path.
            _model = copy.deepcopy(work_model).to(device_obj)  # type: ignore[assignment]
        else:
            resolved = _coerce_model_config(raw_config)
            serving = build_serving_forecaster(resolved).to(device_obj)
            _load_state_dict_loose(
                serving, work_model.state_dict(), str(checkpoint_target)
            )
            _model = serving
        assert _model is not None
        _model.eval()
        _model_artifact_metadata = _checkpoint_metadata(
            payload,
            checkpoint_target,
            model=_model,
        )


def _build_inference_tensor(
    sequence: list[FeatureVector],
    model: ForecasterServingModel,
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


def _resolve_inference_credibility(
    model: ForecasterServingModel,
    *,
    sequence: list[FeatureVector],
    device: torch.device,
) -> torch.Tensor:
    """Build the credibility kwarg for one /analyze forward pass.

    Issue #339 finding #4: training-time loaders feed real per-row credibility
    vectors via ``app.services.credibility_loader.load_credibility_for_run``,
    but ``/analyze`` was passing ``torch.zeros(...)``. This helper pulls the
    as-of date off the last lookback bar and consults the live loader, falling
    back to zeros only when the loader raises (e.g. missing FRED cache on a
    fresh checkout) so the inference path never 5xx's on a missing artefact.
    """

    dim = int(getattr(model, "credibility_dim", 4))
    zero_vec = torch.zeros((1, dim), dtype=torch.float32, device=device)
    if not sequence:
        return zero_vec
    last = sequence[-1]
    as_of = str(getattr(last, "date", "")).strip()
    if not as_of:
        return zero_vec
    try:
        vec = compute_credibility_for_inference(as_of)
    except Exception:  # pragma: no cover -- defensive
        logger.warning("credibility_inference_loader_failed as_of=%s", as_of, exc_info=True)
        return zero_vec
    if vec is None:
        return zero_vec
    if vec.dim() == 1:
        vec = vec.unsqueeze(0)
    if vec.shape[-1] != dim:
        logger.warning(
            "credibility_inference_dim_mismatch expected=%d got=%d as_of=%s",
            dim,
            int(vec.shape[-1]),
            as_of,
        )
        return zero_vec
    return vec.to(dtype=torch.float32, device=device)


def compute_credibility_for_inference(
    event_date: _date_cls | str,
) -> torch.Tensor | None:
    """Live credibility loader wrapper for inference paths.

    Returns a ``(1, 4)`` float tensor matching the
    :class:`CredibilityVector` axis ordering, or ``None`` when the
    loader's inputs are absent. Mirrors the training-time call in
    ``app.data.event_dataset_builder._safe_credibility`` so the
    forecaster sees the same axis layout at /analyze as it did during
    training.

    The encoder embedding cache + FRED cache paths are resolved off the
    canonical encoder alias from the registry; both degrade silently to
    the loader's zero-default when missing.
    """

    from app.services.credibility_loader import load_credibility_for_run
    from app.services.fred_client import DEFAULT_CACHE_DIR as _FRED_CACHE_DIR
    from app.config import DATA_DIR

    if isinstance(event_date, _date_cls):
        as_of_ts = event_date.isoformat()
    else:
        as_of_ts = str(event_date)[:10]
    if not as_of_ts:
        return None

    # Best-effort embedding cache lookup; if the file is absent the
    # drift axis degrades to 0.0 inside the loader.
    embedding_path: Path | None = None
    try:
        from app.models.registry import encoder_ref

        ref = encoder_ref("finbert_fed_adjacent_xbank_dapt")
        if ref is not None and ref.revision:
            candidate = (
                DATA_DIR
                / "raw"
                / "embeddings"
                / f"finbert_fed_adjacent_xbank_dapt_{ref.revision[:14]}.parquet"
            )
            if candidate.exists():
                embedding_path = candidate
    except Exception:  # pragma: no cover -- defensive
        embedding_path = None

    fred_cache_dir = _FRED_CACHE_DIR if Path(_FRED_CACHE_DIR).exists() else None

    try:
        vector = load_credibility_for_run(
            as_of_ts=as_of_ts,
            embedding_path=embedding_path,
            stance_by_date=(),
            fred_response=None,
            fred_cache_dir=fred_cache_dir,
        )
    except (ValueError, FileNotFoundError):
        return None
    return torch.tensor([vector.as_list()], dtype=torch.float32)


def _resolve_inference_text_embedding(
    model: ForecasterServingModel,
    *,
    sequence: list[FeatureVector],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the ``text_embedding`` + ``text_embedding_missing`` kwargs.

    Issue #339 finding #1: training fed pooled prior-N statement
    embeddings via the time-decay adapter, but ``/analyze`` never threaded
    them through ``forward_multi_task``. Without inputs the forward path
    raised ``ValueError("text_embedding when text_adapter_dim > 0")`` and
    the regime card silently fell to ``None`` on the canonical
    checkpoint.

    Pull the pooled vector + missing flag off the last
    :class:`FeatureVector`'s ``text_embedding_pooled`` / ``text_embedding_missing``
    fields (the loader anchors this on the target-row bar; mirror the
    training-time contract at inference). Wrong shape or empty payload
    -> zero tensor + ``missing=1`` so the adapter's keep-mask zeros the
    slot and the model sees an unambiguous "no text" signal rather than
    raising.
    """

    dim = int(getattr(model, "text_embedding_dim", 0))
    if dim <= 0:
        return (
            torch.zeros((1, 0), dtype=torch.float32, device=device),
            torch.ones((1, 1), dtype=torch.float32, device=device),
        )
    pooled: list[float] = []
    missing_flag = 1.0
    if sequence:
        last = sequence[-1]
        raw = getattr(last, "text_embedding_pooled", None) or []
        if isinstance(raw, (list, tuple)) and len(raw) == dim:  # noqa: UP038 — runtime tuple check; X|Y form breaks on isinstance for some older Python builds in the deploy image
            pooled = [float(v) for v in raw]
            missing_flag = float(getattr(last, "text_embedding_missing", 1.0))
    if not pooled:
        return (
            torch.zeros((1, dim), dtype=torch.float32, device=device),
            torch.ones((1, 1), dtype=torch.float32, device=device),
        )
    text_embedding = torch.tensor([pooled], dtype=torch.float32, device=device)
    text_embedding_missing = torch.tensor(
        [[missing_flag]], dtype=torch.float32, device=device
    )
    return text_embedding, text_embedding_missing


def _log_serving_forward_kwargs(model: ForecasterServingModel) -> None:
    """#342: structured INFO line describing the per-request kwarg set.

    Mirrors the kwarg gates in ``_predict_next_point`` so the log line
    is greppable evidence of what the serving forward was called with
    on a given request. Format:

        ``analyze_serving_forward kwargs=<a,b,c> checkpoint=<stem> mode=<output_mode>``

    The kwargs list is empty (``kwargs=``) for the legacy 6-feature
    regression-only path; checkpoints with text + credibility + chunks
    paths active emit the full list. The checkpoint stem is taken from
    ``BEST_MODEL_PATH`` so the operator can correlate against the
    settings inventory + the inference contract sidecar. The ``mode``
    field carries the active output_mode so a grep can distinguish
    "kwargs declared but forward short-circuited" (classification-mode
    request: ``_predict_next_point`` echoes the last bar without
    calling forward) from "kwargs declared and forward invoked"
    (regression mode). One log line per /analyze, NOT one per kwarg.
    """

    populated: list[str] = []
    if bool(getattr(model, "credibility_features", False)):
        populated.append("credibility")
    if bool(getattr(model, "_text_path_active", False)):
        populated.append("text_embedding")
        populated.append("text_embedding_missing")
    if bool(getattr(model, "use_chunk_attention", False)) or bool(
        getattr(model, "use_llm_embeddings", False)
    ):
        populated.append("chunks")
        populated.append("elapsed_days")

    try:
        checkpoint_stem = Path(BEST_MODEL_PATH).stem
    except Exception:  # pragma: no cover -- defensive
        checkpoint_stem = ""

    output_mode = str(getattr(model, "output_mode", "regression") or "regression")

    logger.info(
        "analyze_serving_forward kwargs=%s checkpoint=%s mode=%s",
        ",".join(populated),
        checkpoint_stem,
        output_mode,
    )


def _predict_next_point(model: ForecasterServingModel, sequence: list[FeatureVector]) -> tuple[float, float]:
    # Classification-mode checkpoints emit ``(B, n_classes)`` logits
    # from ``forward()`` (the stance branch under the MultiTaskHead);
    # reading ``out[0]`` and ``out[1]`` as ``(close, vol)`` would
    # surface logits in the response. Until /analyze splits the
    # regression series from the regime classification card (filed as
    # a follow-up to #216), echo the most recent bar's market state as
    # a degenerate forecast so the response stays valid. The new
    # ``RegimeClassificationCard`` is the correct surface for
    # classification checkpoints.
    if str(getattr(model, "output_mode", "regression")) == "classification":
        last = sequence[-1] if sequence else None
        last_close = float(getattr(last, "market_close", 0.0)) if last else 0.0
        last_vol = float(getattr(last, "market_volatility", 0.0)) if last else 0.0
        return last_close, last_vol
    device = next(model.parameters()).device
    x = _build_inference_tensor(sequence, model, device)
    kwargs: dict[str, torch.Tensor] = {}
    if getattr(model, "credibility_features", False):
        # #339 finding #4: pull the live four-axis credibility vector
        # via the FRED + drift loader. Zero-fallback only when the
        # loader raises (missing FRED cache on a fresh checkout).
        kwargs["credibility"] = _resolve_inference_credibility(
            model, sequence=sequence, device=device
        )
    if getattr(model, "_text_path_active", False):
        # #339 finding #1: mirror the training-time text-embedding
        # contract -- pooled prior-N statement vector + missing flag --
        # so the regression-only forward path consumes the same input
        # shape it was trained against.
        text_embedding, text_embedding_missing = _resolve_inference_text_embedding(
            model, sequence=sequence, device=device
        )
        kwargs["text_embedding"] = text_embedding
        kwargs["text_embedding_missing"] = text_embedding_missing
    with torch.no_grad():
        out = model(x, **kwargs).squeeze(0)
    close_scale = float((_model_artifact_metadata or {}).get("close_scale", DEFAULT_CLOSE_SCALE))
    pred_close = float(out[0].item()) * close_scale
    pred_vol = float(out[1].item())
    return pred_close, pred_vol


@torch.no_grad()
def build_market_reaction_panel(
    sequence: list[FeatureVector],
    *,
    text_embedding: "torch.Tensor | None" = None,
    chunks: "torch.Tensor | None" = None,
    elapsed_days: "torch.Tensor | None" = None,
) -> dict[str, Any] | None:
    """Build the four-card market-reaction panel (#293).

    Returns ``None`` only when the active checkpoint mounts NEITHER the
    rates heads (any of ``2y`` / ``5y`` / ``terminal``) NOR the
    vol-regime classifier (a regression-only forecaster). When the
    checkpoint mounts rates heads under ``output_mode='regression'``
    the rates cards still render -- the gate that drops the function
    on ``output_mode != 'classification'`` was removed (#317 finding
    #11). The vol-regime card stays conditional on classification mode
    via :func:`build_regime_classification_card`.

    Per-rates-head card semantics:

    - point_bps + symmetric conformal band (when the sidecar carries
      ``rates_residual_quantiles[name]``);
    - directional bucket + per-bucket softmax probabilities (when the
      aux classifier is mounted). Regression-only checkpoints set
      both to ``None`` so the frontend renders a "no model evidence"
      badge rather than a fake 'easing' grounded in uniform probs
      (#317 finding #10);
    - calibrated APS predicted_set when the sidecar carries
      ``rates_softmax_quantiles[name]``.

    The optional ``text_embedding`` / ``chunks`` / ``elapsed_days``
    are forwarded into ``forward_multi_task`` when the checkpoint has
    the corresponding path active so the call does not raise on
    text-mounted models (#317 finding #17).
    """

    model = _get_model()
    active_rates = tuple(
        str(name).lower()
        for name in getattr(model, "rates_heads_active", ()) or ()
    )
    output_mode = str(getattr(model, "output_mode", "regression"))
    # #317 finding #11: rates cards render under either output_mode as
    # long as the rates heads are mounted; the vol-regime card remains
    # gated on classification via build_regime_classification_card.
    if output_mode != "classification" and not active_rates:
        # #341: structured "not classification mode" surface so the
        # operator can distinguish "model deliberately mute" from
        # "model crashed silently". The frontend treats any payload
        # without a "rates" key as an absent panel.
        return {"status": "not_classification_mode"}
    device = next(model.parameters()).device
    window = build_lookback_sequence(sequence)
    x = _build_inference_tensor(window, model, device)
    kwargs: dict[str, torch.Tensor] = {}
    if getattr(model, "credibility_features", False):
        # #339 finding #4: pull live credibility (zero-fallback only on
        # loader failure) instead of zero-defaulting on every call.
        kwargs["credibility"] = _resolve_inference_credibility(
            model, sequence=window, device=device
        )
    # #317 finding #17 + #339 finding #1: forward the text path inputs
    # through to forward_multi_task on text-mounted checkpoints. Use
    # the explicit ``text_embedding`` argument when the caller threads
    # one in; otherwise resolve from the last lookback bar's pooled
    # vector + missing flag so the canonical checkpoint's regime card
    # stops silently rendering None.
    if getattr(model, "_text_path_active", False):
        if text_embedding is not None:
            kwargs["text_embedding"] = text_embedding.to(device)
        else:
            inferred_text, inferred_missing = _resolve_inference_text_embedding(
                model, sequence=window, device=device
            )
            kwargs["text_embedding"] = inferred_text
            kwargs["text_embedding_missing"] = inferred_missing
    if (
        getattr(model, "use_chunk_attention", False)
        or getattr(model, "use_llm_embeddings", False)
    ) and chunks is not None and elapsed_days is not None:
        kwargs["chunks"] = chunks.to(device)
        kwargs["elapsed_days"] = elapsed_days.to(device)
    forward_multi = getattr(model, "forward_multi_task", None)
    if forward_multi is None:
        return {
            "status": "not_classification_mode",
            "detail": "model has no forward_multi_task method",
        }
    try:
        out_dict = forward_multi(x, **kwargs)
    except TypeError as exc:
        # #341 structured surface (b): the call site populated kwargs
        # the checkpoint did not declare (or omitted a required one).
        # Increment the counter, log at WARNING, return a structured
        # payload instead of bare None so the operator can grep for
        # the bug.
        _contract_counters["market_reaction_inference_kwarg_missing"] += 1
        missing = _extract_missing_kwarg_from_typeerror(exc)
        logger.warning(
            "market_reaction_inference_kwarg_missing kwarg=%s detail=%s",
            missing,
            str(exc),
        )
        return {
            "status": "inference_kwarg_missing",
            "missing_kwarg": missing,
        }
    except RuntimeError as exc:
        # #341 structured surface (c-bis): RuntimeError is the
        # text/chunks-path mounted-but-not-threaded shape from
        # ``prepare_recurrent_input``. Surface it as a structured
        # unexpected exception (NOT bare None) so the symptom is
        # visible on /analyze + greppable in logs. The raw exception
        # message stays in the WARNING log only -- it can carry
        # tensor-shape detail / file paths that should not ship to
        # the client.
        _contract_counters["market_reaction_unexpected_exception"] += 1
        logger.warning(
            "market_reaction_runtime_error detail=%s",
            str(exc),
            exc_info=True,
        )
        return {
            "status": "unexpected_exception",
            "exception_class": "RuntimeError",
        }
    except Exception as exc:  # noqa: BLE001 -- structured surface for c
        _contract_counters["market_reaction_unexpected_exception"] += 1
        logger.warning(
            "market_reaction_unexpected_exception exception_class=%s detail=%s",
            type(exc).__name__,
            str(exc),
            exc_info=True,
        )
        return {
            "status": "unexpected_exception",
            "exception_class": type(exc).__name__,
        }

    metadata = _model_artifact_metadata or {}
    rates_scalers_payload = metadata.get("rates_scalers") or {}
    manifest = _conformal_manifest_for(BEST_MODEL_PATH)
    rates_residuals = (
        getattr(manifest, "rates_residual_quantiles", None) if manifest else None
    ) or {}
    rates_softmax_quantiles = (
        getattr(manifest, "rates_softmax_quantiles", None) if manifest else None
    ) or {}
    coverage = (
        float(getattr(manifest, "nominal_coverage", 0.0))
        if manifest is not None
        else None
    )

    # Rates cards.
    from app.evaluation.conformal import predict_conformal_set
    from app.models.rates_heads import RATES_HEAD_LABEL_NAMES, RATES_HEAD_NAMES
    from app.training.rates_targets import RatesHeadScaler, inverse_standardise_bps

    rates_cards: list[dict[str, Any]] = []
    for name in active_rates:
        if name not in RATES_HEAD_NAMES:
            continue
        pred_key = f"rates_{name}_bps"
        cls_key = f"rates_{name}_cls_logits"
        if pred_key not in out_dict:
            continue
        pred_std_tensor = out_dict[pred_key]
        pred_std = float(pred_std_tensor.squeeze().item())
        scaler_payload = rates_scalers_payload.get(name) if isinstance(rates_scalers_payload, dict) else None
        if isinstance(scaler_payload, dict):
            scaler = RatesHeadScaler(
                mean=float(scaler_payload.get("mean", 0.0)),
                std=float(scaler_payload.get("std", 1.0)),
            )
        else:
            scaler = RatesHeadScaler(mean=0.0, std=1.0)
        point_bps = inverse_standardise_bps(pred_std, scaler)
        band_q: float | None = None
        if isinstance(rates_residuals, dict) and name in rates_residuals:
            band_q = float(rates_residuals[name])
        lower_bps = point_bps - band_q if band_q is not None else None
        upper_bps = point_bps + band_q if band_q is not None else None
        labels = RATES_HEAD_LABEL_NAMES
        # #317 finding #10: only render directional_bucket /
        # bucket_probabilities when the aux classifier is mounted. A
        # missing cls_logits key (regression-only head) leaves both
        # None so the frontend shows "not available" instead of a fake
        # 'easing' on uniform probs.
        bucket: str | None = None
        bucket_probs: dict[str, float] | None = None
        predicted_set: list[str] | None = None
        if cls_key in out_dict:
            cls_logits = out_dict[cls_key].squeeze(0)
            cls_probs_list = torch.softmax(cls_logits, dim=-1).tolist()
            argmax_idx = max(range(len(cls_probs_list)), key=lambda i: cls_probs_list[i])
            bucket = labels[argmax_idx] if argmax_idx < len(labels) else "neutral"
            bucket_probs = {
                labels[i]: float(cls_probs_list[i])
                for i in range(min(len(cls_probs_list), len(labels)))
            }
            # #317 finding #3: calibrated APS prediction set per head.
            cls_threshold = (
                float(rates_softmax_quantiles[name])
                if isinstance(rates_softmax_quantiles, dict)
                and name in rates_softmax_quantiles
                else None
            )
            if cls_threshold is not None:
                set_indices = predict_conformal_set(cls_probs_list, cls_threshold)
                predicted_set = [
                    labels[i] if i < len(labels) else f"class_{i}"
                    for i in set_indices
                ]
        rates_cards.append(
            {
                "head": name,
                "point_bps": float(point_bps),
                "lower_bps": float(lower_bps) if lower_bps is not None else None,
                "upper_bps": float(upper_bps) if upper_bps is not None else None,
                "coverage": coverage if band_q is not None else None,
                "directional_bucket": bucket,
                "bucket_probabilities": bucket_probs,
                "predicted_set": predicted_set,
            }
        )

    # Vol-regime card: reuse the build_regime_classification_card surface
    # but also lift the dual-head log_rv prediction off the same forward.
    vol_regime_card: dict[str, Any] | None = None
    if output_mode == "classification":
        regime_payload = build_regime_classification_card(sequence)
        if regime_payload is not None:
            log_rv_point: float | None = None
            log_rv_payload = out_dict.get("log_rv")
            if log_rv_payload is not None:
                log_rv_point = float(log_rv_payload.squeeze().item())
            vol_regime_card = {
                "log_rv_point": log_rv_point,
                "log_rv_lower": None,
                "log_rv_upper": None,
                "regime_label": str(regime_payload.get("argmax_class") or "normal"),
                "regime_probabilities": {
                    str(k): float(v) for k, v in regime_payload.get("distribution", {}).items()
                },
                "predicted_set": list(regime_payload.get("predicted_set") or []),
                "coverage": float(regime_payload.get("coverage") or 0.0) or None,
            }

    if not rates_cards and vol_regime_card is None:
        # #341: structured-status payload instead of bare ``None`` so
        # the contract surface is symmetric with the regime-card path.
        # ``no_active_heads`` fires when the forward succeeded but
        # nothing surfaced (no rates heads mounted + classification
        # head off). The route handler collapses any status payload
        # to an empty MarketReactionPanel; no schema impact.
        return {"status": "no_active_heads"}
    return {
        "rates": rates_cards,
        "vol_regime": vol_regime_card,
        "encoder_alias": metadata.get("encoder_key"),
        "checkpoint_path": str(BEST_MODEL_PATH),
    }


@torch.no_grad()
def build_regime_classification_card(
    sequence: list[FeatureVector],
) -> dict[str, Any] | None:
    """Run the regime head on the inference window and emit the card.

    Under ADR 0015 / #322 the regime card carries one of two surfaces:

    * **Regression-canonical** (``head_mode in {"regression", "dual"}``):
      the regression head's ``log_rv`` point is the source of truth.
      ``argmax_class`` + ``distribution`` are recovered UI-side by
      bucketing the point against the active checkpoint's
      ``vol_regime_quantiles`` cutoffs (see
      :mod:`app.services.regime_bucketing`); ``log_rv_lower`` /
      ``log_rv_upper`` come from the 80% conformal residual on the
      regression head when a manifest is on disk, falling back to a
      fixed log-vol std otherwise. ``bucket_source = "regression"``.
    * **Classification-only legacy** (``head_mode == "classification"``):
      keep the pre-#322 path: softmax of the stance logits + APS
      prediction set via the conformal ``softmax_quantile`` manifest.
      ``log_rv_*`` are ``None`` and ``bucket_source = "classification"``.

    Returns ``None`` whenever the active checkpoint is not in
    classification ``output_mode`` (regression-output checkpoints emit
    close/vol only, no regime axis), or when neither path can produce a
    populated surface (e.g. classification ``head_mode`` with no
    conformal sidecar on disk). The /analyze handler then leaves
    ``regime_classification`` at ``None`` on the response.
    """

    model = _get_model()
    if str(getattr(model, "output_mode", "regression")) != "classification":
        return None
    manifest = _conformal_manifest_for(BEST_MODEL_PATH)

    from app.evaluation.conformal import (
        format_class_set_label,
        predict_conformal_set,
    )

    device = next(model.parameters()).device
    window = build_lookback_sequence(sequence)
    x = _build_inference_tensor(window, model, device)
    kwargs: dict[str, torch.Tensor] = {}
    if getattr(model, "credibility_features", False):
        # #339 finding #4: live credibility kwarg (zero fallback on
        # loader failure only).
        kwargs["credibility"] = _resolve_inference_credibility(
            model, sequence=window, device=device
        )
    if getattr(model, "_text_path_active", False):
        # #339 finding #1: thread the pooled text embedding +
        # missing-flag pair through ``forward_multi_task`` so the
        # canonical checkpoint's regime card renders instead of
        # silently falling to ``None`` via the try/except in
        # ``_safe_regime_classification``.
        text_embedding, text_embedding_missing = _resolve_inference_text_embedding(
            model, sequence=window, device=device
        )
        kwargs["text_embedding"] = text_embedding
        kwargs["text_embedding_missing"] = text_embedding_missing

    head_mode = str(getattr(model, "head_mode", "classification") or "classification")

    # Resolve the regression branch: only meaningful when the model
    # carries a mounted ``regression_head`` AND the operator opted into
    # regression / dual head_mode. The forward dispatch goes through
    # ``forward_multi_task`` so we can read both the stance logits and
    # the log_rv scalar off one pass; mirrors the
    # ``build_market_reaction_panel`` pattern.
    use_regression_path = (
        head_mode in {"regression", "dual"}
        and getattr(model, "regression_head", None) is not None
    )

    out_dict: dict[str, torch.Tensor] | None = None
    if use_regression_path:
        forward_multi = getattr(model, "forward_multi_task", None)
        if forward_multi is not None:
            try:
                out_dict = forward_multi(x, **kwargs)
            except RuntimeError:
                # Text / chunks path is mounted but inputs not threaded
                # in for this call; fall back to the classification
                # surface so the card still serialises.
                out_dict = None

    log_rv_point: float | None = None
    log_rv_lower: float | None = None
    log_rv_upper: float | None = None
    if out_dict is not None:
        log_rv_payload = out_dict.get("log_rv")
        if log_rv_payload is not None:
            log_rv_point = float(log_rv_payload.squeeze().item())
            # 80% conformal residual on the regression head. The
            # manifest re-uses the existing ``residual_quantile_volatility``
            # slot under the regression-canonical objective (same
            # symmetric ± band convention as the close/vol series).
            band_q: float | None = None
            if manifest is not None:
                raw_q = float(getattr(manifest, "residual_quantile_volatility", 0.0))
                if raw_q > 0.0:
                    band_q = raw_q
            if band_q is not None:
                log_rv_lower = log_rv_point - band_q
                log_rv_upper = log_rv_point + band_q

    cutoffs = get_vol_regime_quantiles()

    # Regression-canonical card.
    if log_rv_point is not None:
        # Per-fold residual std on the log-vol regression head. Prefer
        # the conformal-derived value (80%-band half-width divided by
        # ``CONFIDENCE_Z_SCORE``); fall back to 0.3 when no manifest is
        # on disk -- educated guess on log-vol residual std so the UI
        # distribution still renders. Replaced by the conformal-derived
        # value as soon as a manifest lands.
        log_rv_std = 0.3
        if manifest is not None:
            raw_q = float(getattr(manifest, "residual_quantile_volatility", 0.0))
            if raw_q > 0.0:
                log_rv_std = raw_q / CONFIDENCE_Z_SCORE
        bucket = bucket_log_rv(log_rv_point, cutoffs)
        distribution = derive_distribution(log_rv_point, log_rv_std, cutoffs)
        if bucket is not None and distribution is not None:
            # Try to layer the existing conformal APS set on top of the
            # regression-derived bucket. When the classifier sidecar is
            # available we run the softmax path to recover the calibrated
            # prediction set / coverage; otherwise collapse to a single-
            # class set around the regression bucket so the field stays
            # serialisable.
            predicted_set: list[str]
            set_label: str
            set_size: int
            coverage_val: float
            if (
                manifest is not None
                and getattr(manifest, "softmax_quantile", None) is not None
            ):
                logits = model(x, **kwargs)
                if logits.dim() == 2:
                    probs_tensor = torch.softmax(logits, dim=-1).squeeze(0)
                    probs = [float(p) for p in probs_tensor.tolist()]
                    n_classes = len(probs)
                    labels_local: tuple[str, ...] = VOL_REGIME_CLASS_LABELS
                    if n_classes != len(labels_local):
                        labels_local = tuple(
                            labels_local[i] if i < len(labels_local) else f"class_{i}"
                            for i in range(n_classes)
                        )
                    threshold = float(manifest.softmax_quantile)
                    set_indices = predict_conformal_set(probs, threshold)
                    predicted_set = [labels_local[i] for i in set_indices]
                    set_label = format_class_set_label(set_indices, labels_local)
                    set_size = len(set_indices)
                    coverage_val = float(manifest.nominal_coverage)
                else:
                    predicted_set = [bucket]
                    set_label = "{" + bucket + "}"
                    set_size = 1
                    coverage_val = float(getattr(manifest, "nominal_coverage", 0.0))
            else:
                predicted_set = [bucket]
                set_label = "{" + bucket + "}"
                set_size = 1
                coverage_val = float(
                    getattr(manifest, "nominal_coverage", 0.0)
                ) if manifest is not None else 0.0
            return {
                "predicted_set": predicted_set,
                "set_label": set_label,
                "set_size": set_size,
                "coverage": coverage_val,
                "distribution": distribution,
                "argmax_class": bucket,
                "log_rv_point": log_rv_point,
                "log_rv_lower": log_rv_lower,
                "log_rv_upper": log_rv_upper,
                "bucket_source": "regression",
            }
        # Fall through to the classification path when the regression
        # bucket / distribution cannot be derived (e.g. cutoffs missing
        # on a fresh cold-start checkpoint).

    # Classification-only legacy path. Requires a softmax_quantile
    # manifest to emit a calibrated card; otherwise return None so the
    # /analyze handler leaves the field unset (matches pre-#322 contract).
    if manifest is None or getattr(manifest, "softmax_quantile", None) is None:
        return None
    logits = model(x, **kwargs)
    if logits.dim() != 2:
        return None
    probs_tensor = torch.softmax(logits, dim=-1).squeeze(0)
    probs = [float(p) for p in probs_tensor.tolist()]
    n_classes = len(probs)
    labels: tuple[str, ...] = VOL_REGIME_CLASS_LABELS
    if n_classes != len(labels):
        # Defensive: a 5-class quantile run would emit 5 probs but our
        # label tuple is 3-wide. Pad with f"class_{i}" so the response
        # still serialises rather than indexing past the tuple.
        labels = tuple(
            labels[i] if i < len(labels) else f"class_{i}" for i in range(n_classes)
        )
    threshold = float(manifest.softmax_quantile)
    set_indices = predict_conformal_set(probs, threshold)
    set_labels = [labels[i] for i in set_indices]
    argmax_idx = max(range(n_classes), key=lambda i: probs[i])
    return {
        "predicted_set": set_labels,
        "set_label": format_class_set_label(set_indices, labels),
        "set_size": len(set_indices),
        "coverage": float(manifest.nominal_coverage),
        "distribution": {labels[i]: probs[i] for i in range(n_classes)},
        "argmax_class": labels[argmax_idx],
        "log_rv_point": None,
        "log_rv_lower": None,
        "log_rv_upper": None,
        "bucket_source": "classification",
    }


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


# Canonical vol-regime class labels (#216). Index ordering matches the
# per-fold quantile bins: 0 = lowest tertile (calm), 1 = middle (normal),
# 2 = highest (high). Used by the conformal prediction-set card on
# /analyze to render the set as ``"{normal, high}"`` instead of indices.
VOL_REGIME_CLASS_LABELS: tuple[str, ...] = ("calm", "normal", "high")


def get_vol_regime_quantiles() -> tuple[float, ...]:
    """Expose the active checkpoint's regime quantile cutoffs.

    Returns () when the loaded model is regression-only or when the
    cutoffs were never fit (e.g. cold-start bootstrap).
    """

    try:
        model = _get_model()
    except Exception:
        return ()
    return tuple(getattr(model, "vol_regime_quantiles", ()) or ())


def bucket_realized_regime(realized_vol: float | None) -> str | None:
    """Map a realised forward-vol value to a calm / normal / high label.

    Uses the loaded checkpoint's train-only quantile cutoffs (see
    :func:`app.training.loaders.fit_vol_regime_quantiles`). Returns None
    when the input is missing or when the active model carries no
    cutoffs (regression-only or fresh cold-start). Boundaries match the
    training-time definition: ``v < q[0]`` -> first class, ``v >= q[-1]``
    -> last class, intermediate v -> middle classes in order.
    """

    if realized_vol is None:
        return None
    try:
        value = float(realized_vol)
    except (TypeError, ValueError):
        return None
    if value != value:  # NaN guard
        return None
    cutoffs = get_vol_regime_quantiles()
    if not cutoffs or len(cutoffs) + 1 != len(VOL_REGIME_CLASS_LABELS):
        return None
    for idx, cutoff in enumerate(cutoffs):
        if value < cutoff:
            return VOL_REGIME_CLASS_LABELS[idx]
    return VOL_REGIME_CLASS_LABELS[-1]


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
    # A non-None manifest with both residual quantiles at 0 is the
    # marker for a classification-only sidecar (saved by the training
    # loop when only the APS softmax_quantile is fit on the val
    # partition). Treat that case as "no regression bands available"
    # and fall through to the gaussian-z heuristic so the close/vol
    # series does not emit zero-width bands.
    has_residual_bands = conformal_manifest is not None and (
        float(getattr(conformal_manifest, "residual_quantile_close", 0.0)) > 0.0
        or float(getattr(conformal_manifest, "residual_quantile_volatility", 0.0)) > 0.0
    )
    if has_residual_bands:
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
    model: ForecasterServingModel | None = None,
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
    base_metadata.setdefault("encoder_key", _resolve_encoder_key())
    return base_metadata


def _resolve_encoder_key() -> str | None:
    """Best-effort fetch of the multi-axis classifier encoder alias.

    Inner import so a missing optional dependency in the classifier
    service cannot break the forecaster's diagnostics path.
    """

    try:
        from app.services.multi_axis_classifier import get_loaded_encoder_alias
    except Exception:  # pragma: no cover — defensive
        return None
    try:
        return get_loaded_encoder_alias()
    except Exception:  # pragma: no cover — defensive
        return None


def forecast_quantitative_series(
    vectors: list[FeatureVector],
    forecast_mode: str = "fast",
    horizon: str = "3d",
    forecast_dates: list[str] | None = None,
) -> dict[str, object]:
    if not vectors:
        vectors = [FeatureVector(date="", sentiment_score=0.0, market_close=0.0, market_volatility=0.0)]

    # ``forecast_mode`` is kept as a parameter (default ``"fast"``) for
    # back-compat with persisted history rows; the runtime path is
    # always the cached checkpoint now. The quick_train adaptation
    # branch was retired in #265 along with the rest of the runtime
    # adaptation surface.
    model = _get_model()
    training_result = None

    # #342: emit one structured INFO line per request listing the kwargs
    # the serving forward will be called with on this request. Operators
    # can ``grep analyze_serving_forward | awk`` for drift between the
    # sidecar-declared required kwargs and what the call site actually
    # populates. The list is computed by mirroring the kwarg gates in
    # ``_predict_next_point`` (the canonical serving forward call site)
    # so the log line is the request-level intent, not a per-step
    # repetition.
    _log_serving_forward_kwargs(model)

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
                if (
                    conformal_manifest is not None
                    and float(
                        getattr(conformal_manifest, "residual_quantile_close", 0.0)
                    )
                    > 0.0
                )
                else FORECAST_CONFIDENCE_LEVEL
            ),
            "volatility_scale": vol_scale,
            "forecast_band_source": (
                "conformal"
                if (
                    conformal_manifest is not None
                    and float(
                        getattr(conformal_manifest, "residual_quantile_close", 0.0)
                    )
                    > 0.0
                )
                else "gaussian_z"
            ),
            "conformal_coverage": (
                float(conformal_manifest.nominal_coverage)
                if (
                    conformal_manifest is not None
                    and float(
                        getattr(conformal_manifest, "residual_quantile_close", 0.0)
                    )
                    > 0.0
                )
                else None
            ),
        },
    }
