"""Per-checkpoint inference contract sidecar (#341).

Every checkpoint write path emits a ``<stem>.inference_contract.json``
sidecar next to the ``.pt`` file. The sidecar declares the kwargs the
serving forward will consume and the inference-feature aliases the
encoders + artefacts contribute. The serving loader reads the sidecar
at startup and refuses to bind the checkpoint when the declared
kwargs are not a subset of the serving call sites' supplied kwargs --
or when the registry declares a different feature set for the
encoder slot the checkpoint was trained against.

The contract is a soft surface: a missing sidecar is treated as a
pre-#341 legacy checkpoint and degrades to "skip validation" so the
inference path keeps working on the existing serving fleet. A
present-but-mismatched sidecar is a hard error -- the loader refuses
to bind and ``/health`` exposes the structured reason.
"""

from __future__ import annotations

import inspect
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.models.forecaster_base import ForecasterBase

logger = logging.getLogger(__name__)

# Sidecar suffix. Lives next to the ``.pt`` file the contract describes.
SIDECAR_SUFFIX = ".inference_contract.json"
# Schema version. Bump only on a breaking sidecar shape change so older
# inference fleets can still parse newer sidecars when the fields are
# additive.
SIDECAR_SCHEMA_VERSION = 1

# Kwargs the serving ``forward`` / ``forward_multi_task`` accept besides
# the positional ``x`` tensor. The serving loader's kwarg-superset check
# is "every required kwarg the contract declares must be in this set".
SERVING_FORWARD_KWARGS: frozenset[str] = frozenset(
    {
        "chunks",
        "elapsed_days",
        "chunk_mask",
        "credibility",
        "text_embedding",
        "text_embedding_missing",
        "text_embedding_per_bar",
    }
)


@dataclass(frozen=True)
class InferenceContract:
    """Persisted shape of a checkpoint's inference contract.

    ``required_kwargs`` is the strict set the loader binds against.
    ``optional_kwargs`` is informational -- the loader does not refuse
    to bind on missing optional kwargs. ``inference_features`` is the
    encoder-side feature alias list pulled off ``registry.yaml`` so
    the loader can cross-reference the encoder slot the checkpoint was
    trained against; when the registry's declared aliases for the
    pinned encoder diverge from the contract's, the loader refuses
    to bind.
    """

    schema_version: int
    model_class: str
    required_kwargs: tuple[str, ...]
    optional_kwargs: tuple[str, ...] = ()
    inference_features: tuple[str, ...] = ()
    encoder_alias: str | None = None
    notes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "model_class": str(self.model_class),
            "required_kwargs": list(self.required_kwargs),
            "optional_kwargs": list(self.optional_kwargs),
            "inference_features": list(self.inference_features),
            "encoder_alias": self.encoder_alias,
            "notes": dict(self.notes),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> InferenceContract:
        return cls(
            schema_version=int(payload.get("schema_version", SIDECAR_SCHEMA_VERSION)),
            model_class=str(payload.get("model_class", "")),
            required_kwargs=tuple(str(k) for k in payload.get("required_kwargs") or ()),
            optional_kwargs=tuple(str(k) for k in payload.get("optional_kwargs") or ()),
            inference_features=tuple(
                str(k) for k in payload.get("inference_features") or ()
            ),
            encoder_alias=(
                str(payload["encoder_alias"])
                if payload.get("encoder_alias") is not None
                else None
            ),
            notes=dict(payload.get("notes") or {}),
        )


def sidecar_path_for(checkpoint_path: Path) -> Path:
    """Resolve the sidecar path that sits next to ``checkpoint_path``."""

    return checkpoint_path.with_suffix(checkpoint_path.suffix + SIDECAR_SUFFIX)


def derive_contract(
    model: ForecasterBase,
    *,
    encoder_alias: str | None = None,
    inference_features: tuple[str, ...] = (),
) -> InferenceContract:
    """Compute the contract a freshly trained ``model`` requires.

    The required-kwarg set is derived from the model's runtime gates:
    ``text_embedding`` / ``text_embedding_missing`` ride on
    ``_text_path_active``, ``credibility`` rides on
    ``credibility_features``, ``chunks`` / ``elapsed_days`` ride on
    ``use_chunk_attention`` or ``use_llm_embeddings``. The set is the
    minimal contract a caller must satisfy for the canonical
    ``forward_multi_task`` call to succeed without a ``RuntimeError``
    fallback.

    ``encoder_alias`` and ``inference_features`` are threaded in by the
    caller (the training loop has the registry context the model
    object lacks). They land in the persisted sidecar so the loader can
    cross-check against ``registry.yaml`` at boot.
    """

    required: list[str] = []
    optional: list[str] = []

    if bool(getattr(model, "credibility_features", False)):
        required.append("credibility")

    if bool(getattr(model, "_text_path_active", False)) or int(
        getattr(model, "text_embedding_dim", 0) or 0
    ) > 0:
        required.append("text_embedding")
        required.append("text_embedding_missing")

    if bool(getattr(model, "use_chunk_attention", False)) or bool(
        getattr(model, "use_llm_embeddings", False)
    ):
        required.append("chunks")
        required.append("elapsed_days")
        optional.append("chunk_mask")

    optional.append("text_embedding_per_bar")

    return InferenceContract(
        schema_version=SIDECAR_SCHEMA_VERSION,
        model_class=type(model).__name__,
        required_kwargs=tuple(required),
        optional_kwargs=tuple(optional),
        inference_features=tuple(inference_features),
        encoder_alias=encoder_alias,
    )


def write_sidecar(contract: InferenceContract, checkpoint_path: Path) -> Path:
    """Persist ``contract`` to ``<checkpoint_path>.inference_contract.json``.

    Returns the sidecar path. Caller owns concurrency: the sidecar is
    written non-atomically with a plain ``write_text`` -- the checkpoint
    itself is the synchronisation primitive, and a half-written sidecar
    is recoverable by re-running the trainer.
    """

    sidecar = sidecar_path_for(checkpoint_path)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(
        json.dumps(contract.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return sidecar


def read_sidecar(checkpoint_path: Path) -> InferenceContract | None:
    """Load the contract sidecar, or ``None`` when absent / malformed.

    Malformed sidecars degrade to ``None`` with a warning log -- the
    loader treats the checkpoint as legacy rather than refusing to
    bind. A present-but-malformed sidecar is a data-quality issue worth
    surfacing, not a serving outage.
    """

    sidecar = sidecar_path_for(checkpoint_path)
    if not sidecar.exists():
        return None
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 -- malformed sidecar, log + degrade
        logger.warning("inference_contract_sidecar_malformed path=%s", sidecar)
        return None
    if not isinstance(payload, dict):
        logger.warning("inference_contract_sidecar_not_object path=%s", sidecar)
        return None
    try:
        return InferenceContract.from_dict(payload)
    except Exception:  # noqa: BLE001 -- malformed sidecar, log + degrade
        logger.warning(
            "inference_contract_sidecar_unparseable path=%s", sidecar, exc_info=True
        )
        return None


def collect_serving_forward_kwargs(model_cls: type[Any]) -> frozenset[str]:
    """Return the kwarg names the model's serving ``forward`` accepts.

    Used by the loader's kwarg-superset check. Inspects both
    ``forward`` and ``forward_multi_task`` so a checkpoint that only
    exercises one path still validates against the union.
    """

    accepted: set[str] = set()
    for fname in ("forward", "forward_multi_task"):
        fn = getattr(model_cls, fname, None)
        if fn is None:
            continue
        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError):  # pragma: no cover -- defensive
            continue
        for name, param in sig.parameters.items():
            if name in {"self", "x"}:
                continue
            if param.kind in {
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            }:
                continue
            accepted.add(name)
    return frozenset(accepted)


@dataclass(frozen=True)
class ContractValidation:
    """Outcome of validating a sidecar against the serving signature."""

    ok: bool
    status: str
    missing_kwargs: tuple[str, ...] = ()
    extra_kwargs: tuple[str, ...] = ()
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": bool(self.ok),
            "status": str(self.status),
            "missing_kwargs": list(self.missing_kwargs),
            "extra_kwargs": list(self.extra_kwargs),
            "message": str(self.message),
        }


def validate_against_serving(
    contract: InferenceContract,
    *,
    serving_kwargs: frozenset[str] | None = None,
    serving_model_cls: type[Any] | None = None,
    registry_inference_features: tuple[str, ...] | None = None,
) -> ContractValidation:
    """Cross-check ``contract`` against the live serving signature.

    Two checks run:

    1. Every kwarg in ``contract.required_kwargs`` must be accepted by
       the serving model's ``forward`` / ``forward_multi_task`` (the
       call sites in ``app.services.forecaster`` populate from this
       superset).
    2. When ``registry_inference_features`` is provided (the registry
       was consulted for the checkpoint's encoder alias), the contract's
       ``inference_features`` must be a subset. A registry that declares
       fewer features than the contract is a wiring mismatch -- the
       checkpoint was trained on encoder data the serving registry no
       longer pins.
    """

    if serving_kwargs is None:
        if serving_model_cls is not None:
            serving_kwargs = collect_serving_forward_kwargs(serving_model_cls)
        else:
            serving_kwargs = SERVING_FORWARD_KWARGS

    required = set(contract.required_kwargs)
    missing = sorted(required - set(serving_kwargs))
    if missing:
        return ContractValidation(
            ok=False,
            status="serving_signature_missing_kwargs",
            missing_kwargs=tuple(missing),
            message=(
                "checkpoint inference contract requires kwargs the serving "
                f"forward does not accept: {missing}"
            ),
        )

    if registry_inference_features is not None:
        contract_features = set(contract.inference_features)
        registry_features = set(registry_inference_features)
        extra = sorted(contract_features - registry_features)
        if extra:
            return ContractValidation(
                ok=False,
                status="registry_inference_features_mismatch",
                extra_kwargs=tuple(extra),
                message=(
                    "checkpoint declares inference_features the registry does "
                    f"not pin for this encoder alias: {extra}"
                ),
            )

    return ContractValidation(ok=True, status="ok")


__all__ = [
    "ContractValidation",
    "InferenceContract",
    "SERVING_FORWARD_KWARGS",
    "SIDECAR_SCHEMA_VERSION",
    "SIDECAR_SUFFIX",
    "collect_serving_forward_kwargs",
    "derive_contract",
    "read_sidecar",
    "sidecar_path_for",
    "validate_against_serving",
    "write_sidecar",
]
