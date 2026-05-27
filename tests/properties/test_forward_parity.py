"""Forward-parity property test (issue #341).

For a fixed ``(checkpoint, FeatureVector, prior-N cache)`` triple, the
train-loop forward (research model) and the serving forward
(:class:`ForecasterServingModel`) MUST produce identical output,
modulo dtype / device. This is the load-bearing safety net behind the
"deployed model is the published model" invariant: a checkpoint that
passes the property test is guaranteed to score identically on both
code paths, so the contract sidecar (#341) + the promote step (#336)
cannot disagree on what the checkpoint actually does at forward
time.

The fixture builds a small but real serving + research forecaster
pair off ``backend/models/forecaster_best.pt`` when one is on disk
(canonical CI / production path), and falls back to a deterministic
in-memory toy checkpoint when no artefact is present (CI on a fresh
clone). Both paths exercise the SAME state_dict load on both classes;
the toy fallback is not a stub of the serving model, it IS the
serving model -- the only thing that changes is which weights the
common state_dict carries.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from app.models.config import (
    BEST_MODEL_PATH,
    DEFAULT_DROPOUT,
    DEFAULT_HEAD_HIDDEN_SIZE,
    DEFAULT_HIDDEN_SIZE,
    DEFAULT_NUM_LAYERS,
    FEATURE_SIZE,
    ModelConfig,
)
from app.models.factory import build_research_forecaster, build_serving_forecaster
from app.training.checkpoint import _load_state_dict_loose, _read_checkpoint_payload
from app.training.loop import _coerce_model_config


def _resolve_canonical_payload() -> tuple[ModelConfig, dict | None]:
    """Return ``(config, state_dict_payload)`` for the parity check.

    Prefers the canonical ``backend/models/forecaster_best.pt`` when
    present; otherwise emits a deterministic toy config + a freshly
    initialised state_dict so the test runs end-to-end on a fresh
    clone. Either way the SAME state_dict feeds both classes.
    """

    if BEST_MODEL_PATH.exists():
        payload = _read_checkpoint_payload(BEST_MODEL_PATH, torch.device("cpu"))
        if isinstance(payload, dict) and "model_state_dict" in payload:
            raw_config = payload.get("model_config")
            return _coerce_model_config(raw_config), payload

    # Fallback: deterministic toy config + a freshly initialised serving
    # model whose state_dict feeds both classes. The toy config matches
    # the legacy 6-feature contract so ``_build_inference_tensor`` does
    # not require the rich-feature scaler from a sidecar.
    torch.manual_seed(0)
    toy_config = ModelConfig(
        input_size=FEATURE_SIZE,
        hidden_size=DEFAULT_HIDDEN_SIZE,
        num_layers=DEFAULT_NUM_LAYERS,
        dropout=DEFAULT_DROPOUT,
        head_hidden_size=DEFAULT_HEAD_HIDDEN_SIZE,
        architecture="lstm",
    )
    return toy_config, None


def _set_inference_mode(module: torch.nn.Module) -> None:
    """Put ``module`` in inference mode (eq. of ``module.eval()``)."""

    module.train(False)


@pytest.fixture(scope="module")
def parity_pair() -> tuple[torch.nn.Module, torch.nn.Module, ModelConfig]:
    """Build the research + serving forecasters with a shared state_dict.

    The fixture is module-scoped so the canonical checkpoint load runs
    once. The two classes share the backbone + adapter weights through
    :class:`ForecasterBase`; ``load_state_dict(strict=False)`` discards
    research-only tensors the serving class does not allocate, so the
    parity check exercises exactly the overlap of the two surfaces.
    """

    config, payload = _resolve_canonical_payload()

    research = build_research_forecaster(config)
    serving = build_serving_forecaster(config)

    if payload is not None:
        state_dict = payload["model_state_dict"]
    else:
        # Use the freshly initialised serving state as the shared
        # state_dict; the research class loads what it has parameters
        # for and ignores the rest.
        state_dict = copy.deepcopy(serving.state_dict())

    _load_state_dict_loose(research, state_dict, "parity-fixture")
    _load_state_dict_loose(serving, state_dict, "parity-fixture")

    _set_inference_mode(research)
    _set_inference_mode(serving)
    return research, serving, config


def _build_input_tensor(config: ModelConfig) -> torch.Tensor:
    """Build a deterministic (1, seq, feature) input for the forward pair."""

    torch.manual_seed(42)
    return torch.randn(1, 30, int(config.input_size), dtype=torch.float32)


@torch.no_grad()
def test_forward_parity_research_vs_serving(parity_pair) -> None:
    """Research forward output matches serving forward output element-wise.

    The acceptance: ``torch.allclose(research(x), serving(x), atol=1e-6)``.
    Higher tolerances would mask a real divergence -- e.g. an extra
    detach + tensor reshape on one path that quietly changes the
    numeric output. The two classes share their backbone via
    :class:`ForecasterBase`, so the equality is exact up to float32
    rounding noise of ~1e-7.
    """

    research, serving, config = parity_pair
    x = _build_input_tensor(config)

    research_out = research(x)
    serving_out = serving(x)

    assert research_out.shape == serving_out.shape, (
        f"shape mismatch: research={research_out.shape} "
        f"serving={serving_out.shape}"
    )
    assert torch.allclose(research_out, serving_out, atol=1e-6, rtol=0.0), (
        "research and serving forwards diverged numerically -- "
        "the train-loop forward and the serving forward are no longer "
        "the same function, which means the deployed model is no "
        "longer the published model. See ADR 0023."
    )


@torch.no_grad()
def test_forward_parity_multi_task(parity_pair) -> None:
    """When the canonical checkpoint mounts classification mode, the
    multi-task dispatch matches across classes too.

    ``forward_multi_task`` is the dispatch the regime card /
    market-reaction panel ride on. Skipped on regression-output
    checkpoints (the toy fallback config) where the method raises by
    contract.
    """

    research, serving, config = parity_pair
    if str(getattr(serving, "output_mode", "regression")) != "classification":
        pytest.skip("forward_multi_task is classification-mode only")
    x = _build_input_tensor(config)

    research_mt = research.forward_multi_task(x)
    serving_mt = serving.forward_multi_task(x)

    assert set(research_mt.keys()) == set(serving_mt.keys()), (
        f"forward_multi_task key mismatch: research={set(research_mt)} "
        f"serving={set(serving_mt)}"
    )
    for key in research_mt:
        assert torch.allclose(
            research_mt[key], serving_mt[key], atol=1e-6, rtol=0.0
        ), f"forward_multi_task[{key!r}] diverged across research / serving"


@torch.no_grad()
def test_forward_parity_with_prior_n_cache(parity_pair) -> None:
    """Forward parity holds when the prior-4 lookback cache is non-trivial.

    Mirrors the runtime path where ``build_lookback_sequence`` hands a
    30-bar window into ``_build_inference_tensor``. The point is to
    exercise the same code under a real-looking input distribution
    (zero-mean, unit-std) rather than the all-zero canonical test
    vector -- a divergent code path that only fires on non-zero inputs
    is exactly the class of bug this test exists to catch.
    """

    research, serving, config = parity_pair
    torch.manual_seed(123)
    cache = torch.randn(1, 30, int(config.input_size), dtype=torch.float32)
    cache *= 2.0
    cache += 0.5

    research_out = research(cache)
    serving_out = serving(cache)

    assert torch.allclose(
        research_out, serving_out, atol=1e-6, rtol=0.0
    ), "prior-N cache forward parity failed"


def test_canonical_checkpoint_path_constant_exists() -> None:
    """Guard the constant the loader path relies on.

    ``BEST_MODEL_PATH`` is the singleton anchor for the /analyze
    cold-load + this test's canonical-checkpoint preference. If the
    constant drifts, the fixture above silently degrades to the toy
    fallback on a production-shaped box.
    """

    assert isinstance(BEST_MODEL_PATH, Path)
    assert BEST_MODEL_PATH.name.endswith(".pt")
