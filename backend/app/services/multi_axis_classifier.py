"""Multi-axis text classifier inference service (#78 follow-up).

Wraps the trained ``TextMultiAxisClassifier`` checkpoint behind a
thread-safe singleton so the FastAPI handler at ``/analyze`` can
emit per-axis predictions without paying the cold-start cost on
every request. Mirrors the pattern in
``app.services.text_encoder.get_classifier``.

The classifier is treated as optional: if no checkpoint exists at
the configured path the service returns ``None`` for every prediction
and the /analyze handler falls back to populating only the stance
card from the legacy sentiment classifier. Cold-start training is
NOT triggered automatically — the classifier consumes a fixed
supervised corpus (events.parquet) and is trained out-of-band via
``python -m app.data.train_text_multi_axis_classifier``.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from app.config import MODEL_CHECKPOINT_DIR
from app.models.config import (
    MULTI_TASK_CERTAINTY_LABELS,
    MULTI_TASK_STANCE_LABELS,
)
from app.models.text_multi_axis_classifier import TextMultiAxisClassifier

DEFAULT_CHECKPOINT_PATH = MODEL_CHECKPOINT_DIR / "text_multi_axis_best.pt"
DEFAULT_MAX_LENGTH = 256

# #393: structured surface for the inference-contract validation. The
# /health endpoint reads this so an operator can grep "checkpoint
# incompatible with serving signature" without parsing logs. Mirrors
# the forecaster service's ``_serving_contract_status`` shape.
_contract_status: dict[str, Any] = {
    "status": "uninitialised",
    "checkpoint_path": None,
    "missing_kwargs": [],
    "extra_kwargs": [],
    "message": "",
}
# Below this factor-axis label coverage on the training pool, the
# factor branch's tanh-bounded regression head has trained almost
# exclusively on the masked-out path and emits effectively random
# values at inference. The /analyze response then omits the factor
# card (the MultiAxisBlock.factor field is None) so the frontend
# renders honest absence instead of noise dressed as a prediction.
# Issue #328 picks 0.01 (1 %) as the gate: a coverage tag of 0.0
# (the canonical training package today) trips it, while a real
# gss_factor backfill of even a few hundred rows across ~3 k FOMC
# rows would clear it. The threshold is conservative on purpose —
# we would rather under-emit a real-but-marginal prediction than
# render a low-coverage one that the user reads as load-bearing.
DEFAULT_FACTOR_COVERAGE_GATE = 0.01

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _LoadFailure:
    """Sticky sentinel cached when the checkpoint cannot be loaded.

    Distinct from the ``None`` ``get_classifier`` returns so the cache
    can tell "never tried" (the initial ``_UNSET`` state) from "tried
    and failed". Per #410 / #454: once a failure is cached, subsequent
    ``get_classifier`` calls return ``None`` without re-attempting the
    load and without re-emitting the warning, so a broken checkpoint
    on a sweep does not flood logs with one stack trace per /analyze
    request. The reason string is what feeds the structured warning
    emitted on the first failure.
    """

    reason: str


_UNSET: Any = object()
# ``Any`` so mypy --strict accepts the three-state union (unset / failure /
# state) without insisting on a TypeAlias the runtime ``isinstance`` checks
# do not need. Matches the analogs.py pattern (#410).
_state: Any = _UNSET
_state_lock = threading.Lock()


@dataclass(frozen=True)
class _ClassifierState:
    model: TextMultiAxisClassifier
    tokenizer: Any
    device: torch.device
    max_length: int
    encoder_alias: str
    # Fraction of the training pool that carried a populated
    # axis_factor label. None when the persisted checkpoint pre-dates
    # the #328 coverage stamp (treated as "unknown" → factor stays
    # gated off to match the new default behaviour).
    factor_coverage: float | None
    factor_coverage_threshold: float


def _resolve_checkpoint_path() -> Path:
    override = (os.environ.get("FED_PULSE_TEXT_MULTI_AXIS_CHECKPOINT") or "").strip()
    if override:
        return Path(override)
    return DEFAULT_CHECKPOINT_PATH


def checkpoint_exists() -> bool:
    """Best-effort probe used by /analyze to decide whether to invoke the classifier."""

    return _resolve_checkpoint_path().exists()


def _record_contract_status(
    *,
    status: str,
    checkpoint_path: Any,
    missing_kwargs: tuple[str, ...] = (),
    extra_kwargs: tuple[str, ...] = (),
    message: str = "",
) -> None:
    _contract_status.update(
        {
            "status": str(status),
            "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
            "missing_kwargs": list(missing_kwargs),
            "extra_kwargs": list(extra_kwargs),
            "message": str(message),
        }
    )


def get_serving_contract_status() -> dict[str, Any]:
    """Return the cached contract-validation surface (#393).

    A status of ``ok`` means the checkpoint's declared kwargs are a
    subset of the multi-axis classifier's forward signature; a
    ``serving_signature_missing_kwargs`` / ``sidecar_absent`` status
    surfaces the legacy-vs-mismatch dispatch the forecaster sidecar
    documents. Callers receive a shallow copy so the cache cannot be
    mutated.
    """

    return dict(_contract_status)


def _validate_contract(checkpoint_path: Path) -> tuple[bool, str]:
    """Cross-check the multi-axis sidecar against the forward signature.

    Returns ``(ok, status)``. ``ok=False`` means the loader must
    refuse to bind. A missing sidecar degrades to ``ok=True`` with
    status ``sidecar_absent`` so pre-#393 multi-axis checkpoints keep
    binding (matches the forecaster soft-legacy behaviour).
    """

    from app.training.inference_contract import (
        collect_serving_forward_kwargs,
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
                "treating as pre-#393 legacy artefact"
            ),
        )
        return True, "sidecar_absent"

    serving_kwargs = collect_serving_forward_kwargs(TextMultiAxisClassifier)
    validation = validate_against_serving(
        contract,
        serving_kwargs=serving_kwargs,
    )
    if not validation.ok:
        _record_contract_status(
            status=validation.status,
            checkpoint_path=checkpoint_path,
            missing_kwargs=validation.missing_kwargs,
            extra_kwargs=validation.extra_kwargs,
            message=validation.message,
        )
        _logger.error(
            "multi_axis_inference_contract_mismatch path=%s status=%s missing=%s extra=%s",
            checkpoint_path,
            validation.status,
            list(validation.missing_kwargs),
            list(validation.extra_kwargs),
        )
        return False, validation.status

    _record_contract_status(
        status="ok",
        checkpoint_path=checkpoint_path,
        message="inference contract matches multi-axis classifier signature",
    )
    _logger.info(
        "multi_axis_inference_contract_ok path=%s",
        checkpoint_path,
    )
    return True, "ok"


def _load_state() -> "_ClassifierState | _LoadFailure":
    """Build the singleton from the checkpoint payload.

    Returns a :class:`_LoadFailure` sentinel when the checkpoint is
    missing or any load step fails — the caller (``get_classifier``)
    converts that to a ``None`` return for the public surface and
    caches the sentinel so subsequent calls skip the broken load.
    Raises only on the inference-contract incompatibility path (#393);
    every other failure is captured as a structured sticky sentinel.
    """

    path = _resolve_checkpoint_path()
    if not path.exists():
        return _LoadFailure(reason=f"checkpoint_missing path={path}")
    # #393: validate the inference-contract sidecar before doing any
    # state-dict load. A sidecar declaring kwargs the forward
    # signature does not accept is a hard refusal -- raise so the
    # serving layer surfaces a structured error rather than silently
    # binding a mismatched checkpoint. A missing sidecar (pre-#393
    # artefact) degrades to the legacy "load anyway" path.
    ok, status = _validate_contract(path)
    if not ok:
        raise RuntimeError(
            "multi_axis checkpoint inference contract incompatible with "
            f"serving signature: {status}"
        )
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        # No log here -- get_classifier emits exactly one structured
        # warning when it caches the _LoadFailure sentinel. Mirrors the
        # analogs.py contract (#410): _load_state stays silent so the
        # double-log surfaced in the #551 review pass does not recur.
        return _LoadFailure(
            reason=f"torch_load_failed path={path} error={type(exc).__name__}: {exc}"
        )

    metadata = payload.get("metadata") or {}
    encoder_alias = str(metadata.get("encoder_alias") or "finbert_fed_adjacent")
    head_hidden_size = int(metadata.get("head_hidden_size") or 128)
    dropout = float(metadata.get("dropout") or 0.1)

    try:
        from transformers import AutoTokenizer

        from app.models.registry import encoder_ref

        ref = encoder_ref(encoder_alias)
        if ref is None or not ref.revision:
            raise ValueError(
                f"Encoder alias {encoder_alias!r} is unpinned in registry.yaml"
            )
        tokenizer = AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
            ref.repo,
            revision=ref.revision,
            trust_remote_code=bool(getattr(ref, "trust_remote_code", False)),
        )
        model = TextMultiAxisClassifier.from_encoder_alias(
            encoder_alias=encoder_alias,
            head_hidden_size=head_hidden_size,
            dropout=dropout,
        )
        state_dict = payload.get("model_state_dict") or {}
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            _logger.warning(
                "multi_axis_checkpoint_partial_load missing=%d unexpected=%d",
                len(missing),
                len(unexpected),
            )
    except Exception as exc:
        # Silent here (the single warning fires in get_classifier when
        # the sentinel is first cached). The partial_load warning above
        # is informational rather than a failure so it stays put.
        return _LoadFailure(
            reason=f"model_build_failed encoder={encoder_alias} error={type(exc).__name__}: {exc}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    max_length = int(
        (payload.get("training_args") or {}).get("max_length") or DEFAULT_MAX_LENGTH
    )
    factor_coverage = _coerce_factor_coverage(payload)
    threshold = _resolve_factor_coverage_threshold()
    return _ClassifierState(
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_length=max_length,
        encoder_alias=encoder_alias,
        factor_coverage=factor_coverage,
        factor_coverage_threshold=threshold,
    )


def _coerce_factor_coverage(payload: dict[str, Any]) -> float | None:
    """Read the persisted factor-axis label coverage off the payload.

    The trainer stamps the fraction of train rows that carried a
    populated ``axis_factor`` label under ``training_args.factor_coverage``.
    Pre-#328 checkpoints predate the stamp; ``None`` flags the unknown
    case so the inference path treats them like a 0 %-coverage run
    (factor card stays absent rather than rendering effectively-random
    predictions). The ``metadata`` fallback path mirrors the same key
    so a future caller can write it on the metadata side without
    breaking the loader.
    """

    for bucket_key in ("training_args", "metadata"):
        bucket = payload.get(bucket_key)
        if not isinstance(bucket, dict):
            continue
        raw = bucket.get("factor_coverage")
        if raw is None:
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if value != value:  # NaN
            continue
        return max(0.0, min(1.0, value))
    return None


def _resolve_factor_coverage_threshold() -> float:
    """Env-override hook for the factor-axis coverage gate.

    Defaults to ``DEFAULT_FACTOR_COVERAGE_GATE`` (0.01). The env knob
    is intentional — a future training run that backfills factor labels
    from the gss_factor source can lower the gate without a code change
    if the team decides 0.01 is too strict.
    """

    raw = (os.environ.get("FED_PULSE_TEXT_MULTI_AXIS_FACTOR_GATE") or "").strip()
    if not raw:
        return DEFAULT_FACTOR_COVERAGE_GATE
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return DEFAULT_FACTOR_COVERAGE_GATE
    if value != value or value < 0.0:
        return DEFAULT_FACTOR_COVERAGE_GATE
    return value


def get_loaded_encoder_alias() -> str | None:
    """Encoder alias backing the loaded classifier, or None when absent.

    Used by /analyze diagnostics to surface ``model.encoder_key`` so the
    workspace status bar and pipeline trace can show which encoder is
    live without poking at the file system. The lock matches the
    consistency model used by ``get_classifier`` and ``reset_classifier``
    so a concurrent reload window cannot read a stale state.
    """

    with _state_lock:
        state = _state
    if state is _UNSET or isinstance(state, _LoadFailure):
        return None
    return str(state.encoder_alias)


def get_classifier() -> "_ClassifierState | None":
    """Return the lazily-loaded classifier singleton (or None when absent).

    Thread-safe: callers that race on first use see one load. Subsequent
    callers read the cached state without acquiring the lock. Per #454,
    a load failure is cached as a sticky :class:`_LoadFailure` sentinel
    so subsequent calls return ``None`` without re-attempting the load
    and without re-emitting the warning — call :func:`reset_classifier`
    (or restart the worker) to clear the sticky failure once the
    underlying breakage is fixed.
    """

    global _state
    cached = _state
    if cached is not _UNSET:
        return None if isinstance(cached, _LoadFailure) else cached
    with _state_lock:
        cached = _state
        if cached is not _UNSET:
            return None if isinstance(cached, _LoadFailure) else cached
        loaded = _load_state()
        _state = loaded
        if isinstance(loaded, _LoadFailure):
            _logger.warning("multi_axis_classifier_load_failed reason=%s", loaded.reason)
            return None
        return loaded


def reset_classifier() -> None:
    """Drop the singleton so the next call rebuilds (test hook + post-train refresh).

    Also clears a sticky :class:`_LoadFailure` cached by #454, so an
    operator who fixed the underlying breakage can recover without a
    process restart.
    """

    global _state
    with _state_lock:
        _state = _UNSET
    _record_contract_status(
        status="uninitialised",
        checkpoint_path=None,
        message="",
    )


@torch.no_grad()
def score_text(text: str) -> dict[str, Any] | None:
    """Run the classifier on ``text`` and return the per-axis prediction block.

    Returns ``None`` when no checkpoint is loaded. The output shape
    matches the ``MultiAxisBlock`` Pydantic schema in
    ``app.schemas`` — keys for stance / factor / certainty each
    carrying ``label``, ``confidence``, and (where applicable) a
    per-class distribution. The topic axis was retired in ADR 0044
    (no upstream source ships topic labels).

    The factor card is gated on the persisted factor-axis label
    coverage (#328): when the active checkpoint's training pool
    carried < ``DEFAULT_FACTOR_COVERAGE_GATE`` (1 %) populated
    ``axis_factor`` rows, the regression head trained almost
    exclusively on the masked-out path and its outputs are noise. The
    response then carries ``factor=None`` so the frontend renders an
    absent card instead of rendering noise dressed as a prediction.
    """

    state = get_classifier()
    if state is None:
        return None
    text = (text or "").strip()
    if not text:
        return None
    encoded = state.tokenizer(
        text,
        max_length=state.max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(state.device)
    attention_mask = encoded["attention_mask"].to(state.device)
    logits = state.model(input_ids=input_ids, attention_mask=attention_mask)

    stance_probs = torch.softmax(logits["stance"], dim=-1)[0]
    stance_idx = int(stance_probs.argmax().item())
    stance_label = MULTI_TASK_STANCE_LABELS[stance_idx]
    stance_dist = {
        MULTI_TASK_STANCE_LABELS[i]: float(stance_probs[i].item())
        for i in range(len(MULTI_TASK_STANCE_LABELS))
    }

    certainty_probs = torch.softmax(logits["certainty"], dim=-1)[0]
    certainty_idx = int(certainty_probs.argmax().item())
    certainty_label = MULTI_TASK_CERTAINTY_LABELS[certainty_idx]
    certainty_dist = {
        MULTI_TASK_CERTAINTY_LABELS[i]: float(certainty_probs[i].item())
        for i in range(len(MULTI_TASK_CERTAINTY_LABELS))
    }

    factor_card = _build_factor_card(state, logits)

    return {
        "stance": {
            "label": stance_label,
            "confidence": float(stance_probs[stance_idx].item()),
            "distribution": stance_dist,
        },
        "factor": factor_card,
        "certainty": {
            "label": certainty_label,
            "confidence": float(certainty_probs[certainty_idx].item()),
            "distribution": certainty_dist,
        },
    }


def _build_factor_card(
    state: _ClassifierState, logits: dict[str, torch.Tensor]
) -> dict[str, float] | None:
    """Return the factor card or ``None`` per the coverage gate (#328).

    Coverage below the gate (or ``None`` on a pre-#328 checkpoint that
    did not stamp the field) drops the card entirely so the
    ``MultiAxisBlock.factor`` field surfaces as ``None``. Above the
    gate, the head's tanh output is clipped to [-1, 1] and surfaced
    as the card value with the legacy abs-as-confidence proxy.
    """

    coverage = state.factor_coverage
    threshold = state.factor_coverage_threshold
    if coverage is None or coverage < threshold:
        _logger.debug(
            "multi_axis_factor_card_absent coverage=%s threshold=%s",
            coverage,
            threshold,
        )
        return None
    factor_value = float(logits["factor"][0].item())
    return {
        "value": max(-1.0, min(1.0, factor_value)),
        # Factor regression confidence is not a probability; we emit
        # the absolute value as a proxy ("how far from neutral") and
        # the frontend renders it as the bar magnitude. Calibration
        # is a follow-up.
        "confidence": min(1.0, abs(factor_value)),
    }
