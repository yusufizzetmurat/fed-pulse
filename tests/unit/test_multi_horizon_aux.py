"""Multi-horizon auxiliary regression heads (#471).

The regime classifier trains primarily against
``forward_realized_vol_10d``. ``aux_horizons`` mounts parallel
regression heads against other forward-vol horizons so the encoder
sees more horizon-signal without changing the canonical headline.

These tests pin the wiring at four layers:

- ``ModelConfig.aux_horizons`` defaults empty and round-trips through
  ``from_model`` when set.
- ``build_forecaster`` mounts the correct number of aux regression heads
  alongside the primary log-RV head.
- The joint loss equals the hand-computed sum of the primary MSE +
  ``alpha * MSE`` per aux horizon.
- A model built with ``aux_horizons=()`` (default) has identical
  state_dict shape to the pre-#471 dual-head construction.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from app.models.config import ModelConfig
from app.models.factory import build_forecaster
from app.training.loop import _maybe_aux_horizon_mse


# ---------------------------------------------------------------------------
# ModelConfig round-trip
# ---------------------------------------------------------------------------


def test_aux_horizons_default_empty() -> None:
    """No aux horizons mounted by default keeps the pre-#471 path byte-identical."""

    config = ModelConfig()
    assert config.aux_horizons == ()
    assert config.aux_horizon_alpha == pytest.approx(0.3)


def test_aux_horizons_opt_in_roundtrip_via_from_model() -> None:
    """``aux_horizons`` survives the ModelConfig.from_model round-trip."""

    config = ModelConfig(
        output_mode="classification",
        head_mode="dual",
        n_classes=3,
        hidden_size=16,
        head_hidden_size=8,
        aux_horizons=(5, 20),
        aux_horizon_alpha=0.4,
    )
    model = build_forecaster(config)
    rebuilt = ModelConfig.from_model(model)
    assert rebuilt.aux_horizons == (5, 20)
    assert rebuilt.aux_horizon_alpha == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# Factory + head mounting
# ---------------------------------------------------------------------------


def _build_dual_head(
    aux_horizons: tuple[int, ...] = (), aux_horizon_alpha: float = 0.3
) -> "torch.nn.Module":
    config = ModelConfig(
        output_mode="classification",
        head_mode="dual",
        n_classes=3,
        hidden_size=16,
        head_hidden_size=8,
        aux_horizons=aux_horizons,
        aux_horizon_alpha=aux_horizon_alpha,
    )
    return build_forecaster(config)


def test_factory_mounts_one_aux_head_per_horizon() -> None:
    """``aux_horizons=(5, 20)`` mounts two aux regression heads."""

    model = _build_dual_head(aux_horizons=(5, 20))
    assert hasattr(model, "aux_regression_heads")
    aux_heads = model.aux_regression_heads
    # ModuleDict keyed by ``h<H>`` so the state_dict carries
    # ``aux_regression_heads.h5`` / ``.h20`` paths.
    assert set(aux_heads.keys()) == {"h5", "h20"}


def test_factory_rejects_aux_horizons_under_classification_only() -> None:
    """Aux heads need a primary regression branch to compose with."""

    with pytest.raises(ValueError, match="head_mode"):
        build_forecaster(
            ModelConfig(
                output_mode="classification",
                head_mode="classification",
                n_classes=3,
                hidden_size=16,
                head_hidden_size=8,
                aux_horizons=(5,),
            )
        )


def test_factory_rejects_unsupported_aux_horizon() -> None:
    with pytest.raises(ValueError, match="aux_horizons"):
        build_forecaster(
            ModelConfig(
                output_mode="classification",
                head_mode="dual",
                n_classes=3,
                hidden_size=16,
                head_hidden_size=8,
                aux_horizons=(7,),
            )
        )


def test_factory_rejects_primary_horizon_in_aux() -> None:
    """10d is the primary and cannot appear in the aux tuple."""

    with pytest.raises(ValueError, match="aux_horizons"):
        build_forecaster(
            ModelConfig(
                output_mode="classification",
                head_mode="dual",
                n_classes=3,
                hidden_size=16,
                head_hidden_size=8,
                aux_horizons=(10,),
            )
        )


# ---------------------------------------------------------------------------
# Default-off byte-identity: aux heads do not perturb the state_dict shape
# ---------------------------------------------------------------------------


def test_aux_horizons_off_byte_identical_state_dict_shape() -> None:
    """``aux_horizons=()`` must mount no aux head and leave the legacy
    state_dict shape intact. A future regression here would surface as a
    deserialisation failure on every pre-#471 checkpoint."""

    legacy = _build_dual_head(aux_horizons=())
    legacy_keys = set(legacy.state_dict().keys())
    # No aux key may appear when aux_horizons=().
    assert not any(key.startswith("aux_regression_heads.") for key in legacy_keys)
    # Active path mounts at least one new key per aux head.
    active = _build_dual_head(aux_horizons=(5,))
    active_keys = set(active.state_dict().keys())
    aux_keys = {k for k in active_keys if k.startswith("aux_regression_heads.")}
    assert aux_keys  # at least one parameter per aux head
    # Removing aux keys recovers exactly the legacy key set.
    assert (active_keys - aux_keys) == legacy_keys


# ---------------------------------------------------------------------------
# Loss math
# ---------------------------------------------------------------------------


def test_aux_horizon_mse_hand_computed_three_rows() -> None:
    """Verify the joint aux MSE equals the hand-computed sum for 3 rows.

    Primary log_rv predictions: [0.0, 0.0, 0.0]
    Aux 5d predictions:        [1.0, 1.0, 1.0]
    Aux 20d predictions:       [2.0, 2.0, 2.0]
    Targets aux 5d:            [0.0, 0.0, 0.0] -> MSE = 1.0
    Targets aux 20d:           [0.0, 0.0, 0.0] -> MSE = 4.0
    alpha = 0.3
    Aux contribution is alpha * mean(per-horizon MSE):
    Expected: 0.3 * (1.0 + 4.0) / 2 = 0.75
    """

    logits = {
        "log_rv": torch.zeros(3),
        "aux_log_rv_5d": torch.ones(3),
        "aux_log_rv_20d": torch.full((3,), 2.0),
    }
    aux_target = torch.zeros((3, 2))
    out = _maybe_aux_horizon_mse(
        logits_dict=logits,
        batch_aux_log_rv=aux_target,
        aux_horizons=(5, 20),
        aux_horizon_alpha=0.3,
    )
    expected = 0.3 * (1.0 + 4.0) / 2.0
    assert out.item() == pytest.approx(expected, rel=1e-6)


def test_aux_horizon_mse_empty_horizons_returns_zero() -> None:
    """No aux heads mounted -> zero contribution to the joint loss."""

    logits = {"log_rv": torch.zeros(3)}
    out = _maybe_aux_horizon_mse(
        logits_dict=logits,
        batch_aux_log_rv=None,
        aux_horizons=(),
        aux_horizon_alpha=0.3,
    )
    assert out.item() == pytest.approx(0.0)


def test_aux_horizon_mse_alpha_zero_zeroes_contribution() -> None:
    """alpha=0 must collapse the aux contribution to a graph-attached zero."""

    logits = {
        "log_rv": torch.zeros(3),
        "aux_log_rv_5d": torch.ones(3),
    }
    target = torch.zeros((3, 1))
    out = _maybe_aux_horizon_mse(
        logits_dict=logits,
        batch_aux_log_rv=target,
        aux_horizons=(5,),
        aux_horizon_alpha=0.0,
    )
    assert out.item() == pytest.approx(0.0)
