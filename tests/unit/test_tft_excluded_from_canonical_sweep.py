"""Guard against TFT silently re-entering canonical sweep defaults (#331).

ADR 0020 excludes TFT from the canonical architecture comparison.
The exclusion bites at three surfaces:

1. ``app.models.config.CANONICAL_SWEEP_ARCHITECTURES`` -- the tuple new
   sweep code iterates -- must omit ``"tft"`` while
   ``FORECASTER_ARCHITECTURES`` retains it (the encoder module +
   existing checkpoints rely on the identifier remaining importable).
2. ``scripts.run_regime_architecture_sweep._DEFAULT_ARCHITECTURES`` --
   the default architecture list the canonical sweep runner uses when
   the CLI flag is not passed -- must omit ``"tft"``.
3. ``app.models.factory.build_forecaster`` must raise a
   ``DeprecationWarning`` (carrying ``TFT_EXCLUSION_REASON``) when
   asked to build a TFT instance, so an opt-in re-run surfaces the
   exclusion in the trainer logs.

Re-adding TFT to the canonical surface requires updating both this
test and ADR 0020 -- the failure mode is a loud test, not a silent
sweep regression.
"""

from __future__ import annotations

import warnings

import pytest

from app.models import (
    CANONICAL_SWEEP_ARCHITECTURES,
    FORECASTER_ARCHITECTURES,
    TFT_EXCLUSION_REASON,
)


def test_canonical_sweep_architectures_excludes_tft() -> None:
    """``CANONICAL_SWEEP_ARCHITECTURES`` omits ``"tft"`` -- per ADR 0020."""

    assert "tft" not in CANONICAL_SWEEP_ARCHITECTURES, (
        "TFT must stay out of the canonical sweep targets per ADR 0020. "
        "Re-adding it requires a faithful native-quantile-head implementation "
        "(STRETCH); update ADR 0020 + this test in lockstep."
    )


def test_forecaster_architectures_retains_tft_for_backcompat() -> None:
    """``FORECASTER_ARCHITECTURES`` keeps ``"tft"`` so existing checkpoints load.

    The back-compat surface is intentional -- dropping the identifier
    would break checkpoint round-tripping through ``ModelConfig.from_model``
    on any artefact that recorded ``architecture="tft"``.
    """

    assert "tft" in FORECASTER_ARCHITECTURES, (
        "TFT must stay in FORECASTER_ARCHITECTURES so existing checkpoints "
        "and the TFTEncoder module remain back-compat-loadable. The "
        "exclusion lives in CANONICAL_SWEEP_ARCHITECTURES, not here."
    )


def test_canonical_sweep_is_subset_of_forecaster_architectures() -> None:
    """The canonical tuple is a strict subset of the full registry."""

    assert set(CANONICAL_SWEEP_ARCHITECTURES).issubset(
        set(FORECASTER_ARCHITECTURES)
    )
    assert set(CANONICAL_SWEEP_ARCHITECTURES) < set(FORECASTER_ARCHITECTURES)


def test_tft_exclusion_reason_cites_adr_quantile_head() -> None:
    """``TFT_EXCLUSION_REASON`` documents the ADR rationale."""

    assert "ADR 0020" in TFT_EXCLUSION_REASON
    assert "quantile" in TFT_EXCLUSION_REASON.lower()
    assert "vsn" in TFT_EXCLUSION_REASON.lower() or "variable selection" in TFT_EXCLUSION_REASON.lower()


def test_regime_arch_sweep_default_architectures_excludes_tft() -> None:
    """``scripts/run_regime_architecture_sweep.py`` default list omits TFT.

    This is the runner the ``make regime-arch-sweep`` target invokes.
    A canonical sweep dispatched without ``--architectures`` must not
    include TFT in its per-architecture report.

    The script lives under ``scripts/`` which is not a Python package
    (no ``__init__.py``); we load the module directly off its file
    path so the test runs from any pytest working directory.
    """

    import importlib.util
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "run_regime_architecture_sweep.py"
    assert script_path.is_file(), f"runner missing at {script_path}"

    spec = importlib.util.spec_from_file_location(
        "_test_regime_arch_sweep_module", script_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    default_architectures = module._DEFAULT_ARCHITECTURES
    assert "tft" not in default_architectures, (
        "regime_arch_sweep _DEFAULT_ARCHITECTURES must exclude TFT per "
        "ADR 0020; pass --architectures tft explicitly to opt back in."
    )
    # The canonical four-architecture comparison: gru / tcn / transformer
    # / lstm_attn (the §6.7 post-correction headline set). Order is not
    # asserted; membership is.
    assert {"gru", "tcn", "transformer", "lstm_attn"}.issubset(
        set(default_architectures)
    )


def test_build_forecaster_emits_deprecation_warning_for_tft() -> None:
    """``build_forecaster`` warns when asked to build a TFT instance.

    The factory still builds the module (back-compat: checkpoints that
    recorded ``architecture="tft"`` must continue to load), but the
    deprecation warning surfaces ADR 0020 in the trainer logs so a
    canonical sweep that mis-includes TFT cannot regress silently.
    """

    pytest.importorskip("torch")
    from app.models.config import ModelConfig
    from app.models.factory import build_forecaster

    config = ModelConfig(architecture="tft")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        build_forecaster(config, role="research")
    deprecation = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert deprecation, (
        "Building a TFT forecaster must emit a DeprecationWarning citing "
        "ADR 0020 (see TFT_EXCLUSION_REASON)."
    )
    messages = " ".join(str(w.message) for w in deprecation)
    assert "ADR 0020" in messages
