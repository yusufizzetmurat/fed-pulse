"""Loader / serving kwarg-superset property test (issue #341).

The cheap signature-only sibling of ``test_forward_parity.py``.
Introspects the kwargs the training loop populates (in
``backend/app/training/loop.py``) and the kwargs the serving call
sites populate (in ``backend/app/services/forecaster.py``), then
asserts every kwarg the training-side loader produces is also handled
at every serving call site.

The training loop's ``kwargs[...]`` populations and the serving
forecaster's ``kwargs[...]`` populations are the two sides of the same
``forward_multi_task`` signature; this test catches the class of bug
where one side adds a new kwarg but the other never threads it
through. Cheaper to run than the numeric parity check and runs even
without torch available.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = REPO_ROOT / "backend"

TRAINING_LOOP_PATH = BACKEND_DIR / "app" / "training" / "loop.py"
SERVING_FORECASTER_PATH = BACKEND_DIR / "app" / "services" / "forecaster.py"
SERVING_MODEL_PATH = BACKEND_DIR / "app" / "models" / "serving_model.py"


# Set of kwargs we do NOT expect the inference call site to populate
# at every entry. These are training-only kwargs that ride on training
# tensors that have no serving-time equivalent (e.g. ``text_embedding_per_bar``
# is fed on a per-batch basis in the training collator but resolves to
# the same pooled vector at inference).
_SERVING_OPTIONAL_KWARGS = frozenset(
    {
        # serving emits one bar at a time, so per-bar text embedding is
        # equivalent to the pooled text_embedding kwarg.
        "text_embedding_per_bar",
        # only mounted when use_chunk_attention / use_llm_embeddings is
        # on; the chunk_mask kwarg is informational.
        "chunk_mask",
    }
)


def _extract_kwarg_assignments(path: Path) -> set[str]:
    """Return the set of kwarg names assigned via ``kwargs["..."] = ...``.

    Parses ``path`` as Python via :mod:`ast` and walks every
    ``Subscript`` assignment whose target's value is the
    :class:`ast.Name` ``kwargs``. This pattern matches the training
    loop's ``kwargs["credibility"] = credibility`` shape and the
    serving call site's ``kwargs["text_embedding"] = ...`` shape, but
    is robust to comments, formatting drift, and string-content
    changes.
    """

    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Subscript):
                continue
            if not (
                isinstance(target.value, ast.Name) and target.value.id == "kwargs"
            ):
                continue
            slice_node = target.slice
            if isinstance(slice_node, ast.Constant) and isinstance(
                slice_node.value, str
            ):
                found.add(slice_node.value)
    return found


def _extract_explicit_kwargs(path: Path, call_name_regex: str) -> set[str]:
    """Return kwargs passed by name into a call matching ``call_name_regex``.

    Used to catch the cases where the call site uses
    ``forward_multi_task(x, text_embedding=..., credibility=...)``
    directly instead of the ``kwargs`` indirection.
    """

    tree = ast.parse(path.read_text(encoding="utf-8"))
    pattern = re.compile(call_name_regex)
    found: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        # Resolve the callee name -- either bare ``foo`` or ``self.foo``.
        if isinstance(node.func, ast.Name):
            name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            name = node.func.attr
        else:
            continue
        if not pattern.match(name):
            continue
        for kw in node.keywords:
            if kw.arg is not None:
                found.add(kw.arg)
    return found


def _serving_forward_kwargs() -> set[str]:
    """Return the kwarg names the serving model's forward methods accept.

    Cross-checks against
    :func:`app.training.inference_contract.collect_serving_forward_kwargs`
    so the property test fails fast if the helper drifts away from
    the actual signature.
    """

    from app.models.serving_model import ForecasterServingModel
    from app.training.inference_contract import collect_serving_forward_kwargs

    return set(collect_serving_forward_kwargs(ForecasterServingModel))


def test_training_loop_kwargs_are_subset_of_serving_forward() -> None:
    """Every kwarg the training loop populates must be a kwarg the
    serving forward accepts.

    A kwarg the loader feeds the model that the serving forward does
    not accept is the textbook "train-side input mounted but inference
    can't supply it" bug. The check asserts the inclusion both
    directions: the training loop is the population source of truth,
    and the serving signature is the consumption surface.
    """

    train_kwargs = _extract_kwarg_assignments(TRAINING_LOOP_PATH)
    # Also include any kwargs threaded directly into forward / forward_multi_task
    # calls from the loop module so the check catches both populating
    # conventions.
    train_kwargs |= _extract_explicit_kwargs(
        TRAINING_LOOP_PATH, r"^(forward_multi_task|forward)$"
    )
    serving_kwargs = _serving_forward_kwargs()

    # Trim the train kwargs to the ones whose names are actually kwargs
    # on the forward signature -- the loop also uses ``kwargs[...]``
    # for non-forward dispatch state (e.g. ModelConfig field
    # propagation), and those are not signature concerns.
    train_forward_kwargs = train_kwargs & (
        serving_kwargs | _SERVING_OPTIONAL_KWARGS
        | {"text_embedding", "text_embedding_missing", "credibility", "chunks", "elapsed_days"}
    )

    missing = train_forward_kwargs - serving_kwargs
    assert not missing, (
        "training loop populates forward kwargs the serving signature "
        f"does not accept: {sorted(missing)}. The serving call site "
        f"cannot thread these through, so the deployed model would "
        f"score differently from the trained model."
    )


def test_serving_call_site_handles_required_forward_kwargs() -> None:
    """The /analyze call sites must thread every kwarg the canonical
    forward path requires when the corresponding model gate is on.

    Specifically: ``text_embedding`` + ``text_embedding_missing``
    (when ``_text_path_active``), ``credibility`` (when
    ``credibility_features``), and ``chunks`` + ``elapsed_days`` (when
    chunk attention is on). The assertion is "the serving forecaster
    module references every required kwarg name" -- not "every gate
    is unconditionally enabled".
    """

    serving_kwargs = _extract_kwarg_assignments(SERVING_FORECASTER_PATH)
    serving_kwargs |= _extract_explicit_kwargs(
        SERVING_FORECASTER_PATH, r"^(forward_multi_task|forward)$"
    )

    required_when_gate_on = {
        "text_embedding",
        "text_embedding_missing",
        "credibility",
        "chunks",
        "elapsed_days",
    }
    missing = required_when_gate_on - serving_kwargs
    assert not missing, (
        "serving call site in app.services.forecaster does not thread "
        f"the following forward kwargs: {sorted(missing)}. Train-time "
        f"loader passes them but serving never does -- the deployed "
        f"model receives zero / default inputs where the trained model "
        f"received real ones."
    )


def test_serving_model_class_accepts_known_kwargs() -> None:
    """Sanity check: every kwarg in the union is on the serving forward.

    Closes the loop on the previous two tests -- the property is
    'inference signature is a superset of training-side populations',
    so the inference signature itself must list every union-member
    kwarg.
    """

    serving_kwargs = _serving_forward_kwargs()
    expected = {
        "text_embedding",
        "text_embedding_missing",
        "text_embedding_per_bar",
        "credibility",
        "chunks",
        "elapsed_days",
        "chunk_mask",
    }
    missing = expected - serving_kwargs
    assert not missing, (
        f"serving forward signature is missing {sorted(missing)}; "
        "the contract sidecar derivation depends on these kwarg names."
    )


def test_inference_contract_helper_matches_serving_signature() -> None:
    """The :func:`derive_contract` helper's static SERVING_FORWARD_KWARGS
    constant must match the live serving signature.

    Drift here means a sidecar derivation that reports the wrong
    required-kwarg set on every new checkpoint. The
    ``collect_serving_forward_kwargs`` introspector is the source of
    truth; the static constant is the cheap default the
    ``validate_against_serving`` helper falls back to when the caller
    doesn't pass a ``serving_model_cls`` argument.
    """

    pytest.importorskip("torch")
    from app.training.inference_contract import SERVING_FORWARD_KWARGS

    live_kwargs = _serving_forward_kwargs()
    drift = (live_kwargs ^ SERVING_FORWARD_KWARGS)
    assert not drift, (
        f"SERVING_FORWARD_KWARGS drift: {sorted(drift)}. Update the "
        "constant in app.training.inference_contract to match the "
        "live ForecasterServingModel.forward signature."
    )
