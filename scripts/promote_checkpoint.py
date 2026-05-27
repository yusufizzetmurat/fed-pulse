"""Promote a research-side checkpoint to a serving-shape artefact (issue #336).

The research class (:class:`ForecasterResearchModel`) and the serving
class (:class:`ForecasterServingModel`) share their backbone +
adapter state_dict keys through :class:`ForecasterBase`. Promotion is
therefore a copy of the persisted state_dict into a serving-shape
checkpoint payload, with the ``model_version`` field on the payload
bumped to flag that the artefact has been promoted. The
``model_state_dict`` keys themselves are unchanged so subsequent
``services.forecaster._get_model()`` loads via the loose-load path.

Usage:

    python scripts/promote_checkpoint.py \\
        artifacts/.../research.pt \\
        backend/models/serving.pt

The script does NOT re-train anything; it is a metadata-only +
state_dict-copy step. The serving side picks up the new artefact on
the next /analyze call once it is moved to ``backend/models/forecaster_best.pt``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

from app.models.factory import build_serving_forecaster
from app.training.checkpoint import _load_state_dict_loose
from app.training.loop import _coerce_model_config


def _bump_model_version(current: str | None) -> str:
    """Append ``+serving`` to the research-side ``model_version`` tag.

    The serving artefact must be distinguishable from the research
    artefact in the run registry. We keep the original tag prefix so the
    provenance is still readable; appending ``+serving`` is a one-way
    marker so a downstream tool can recognise the promotion without
    re-parsing the full pedigree.
    """
    base = (current or "").strip()
    if base and base.endswith("+serving"):
        return base
    return f"{base}+serving" if base else "promoted+serving"


def promote_research_checkpoint_to_serving(
    research_path: Path,
    serving_path: Path | None = None,
) -> Path:
    """Promote a research checkpoint to a serving-shape artefact.

    Parameters
    ----------
    research_path
        Path to the research-side checkpoint (``.pt``).
    serving_path
        Destination path. Defaults to ``<research_path>.serving.pt`` next
        to the source so the promotion is non-destructive.

    Returns
    -------
    Path to the serving-shape artefact.
    """
    research_path = Path(research_path)
    if not research_path.exists():
        raise FileNotFoundError(
            f"research checkpoint not found: {research_path}"
        )
    if serving_path is None:
        serving_path = research_path.with_name(
            research_path.stem + ".serving.pt"
        )
    serving_path = Path(serving_path)

    payload: dict[str, Any] = torch.load(research_path, map_location="cpu")
    if "model_state_dict" not in payload:
        raise ValueError(
            f"checkpoint {research_path} has no 'model_state_dict' key; "
            "is it a real training artefact?"
        )
    raw_config = payload.get("model_config")
    resolved = _coerce_model_config(raw_config)

    # Rebuild a serving model from the same config so we exercise the
    # narrow ctor surface. The loose-load discards any research-only
    # tensors the serving class does not allocate.
    serving = build_serving_forecaster(resolved)
    _load_state_dict_loose(
        serving, payload["model_state_dict"], str(research_path)
    )

    promoted_payload: dict[str, Any] = dict(payload)
    promoted_payload["model_state_dict"] = serving.state_dict()
    promoted_payload["model_version"] = _bump_model_version(
        payload.get("model_version") if isinstance(payload, dict) else None
    )
    promoted_payload["promoted_from"] = str(research_path)
    promoted_payload["serving_class"] = "ForecasterServingModel"

    serving_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(promoted_payload, serving_path)
    # #341: every checkpoint write path -- including promotion -- emits
    # the inference contract sidecar. Promoted artefacts inherit the
    # required-kwarg set from the freshly built serving instance (which
    # mirrors the research model's runtime gates). A source-side sidecar
    # next to ``research_path`` is preferred when present (it carries
    # the encoder_alias / inference_features the trainer wired in);
    # otherwise the promoted-side derivation is the floor.
    try:
        from app.training.inference_contract import (
            derive_contract,
            read_sidecar,
            write_sidecar,
        )

        source_contract = read_sidecar(research_path)
        if source_contract is not None:
            promoted_contract = source_contract
        else:
            promoted_contract = derive_contract(serving)
        write_sidecar(promoted_contract, serving_path)
    except Exception:  # pragma: no cover -- never let sidecar break promotion
        import logging as _logging

        _logging.getLogger(__name__).warning(
            "inference_contract_sidecar_write_failed_in_promotion path=%s",
            serving_path,
            exc_info=True,
        )
    return serving_path


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Promote a research checkpoint to a serving artefact."
    )
    parser.add_argument(
        "research_path",
        type=Path,
        help="Source research checkpoint (.pt).",
    )
    parser.add_argument(
        "serving_path",
        type=Path,
        nargs="?",
        default=None,
        help="Destination serving checkpoint (default: <source>.serving.pt).",
    )
    args = parser.parse_args(argv)

    out = promote_research_checkpoint_to_serving(
        args.research_path, args.serving_path
    )
    print(out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main(sys.argv[1:]))
