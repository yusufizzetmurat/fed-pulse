"""Unit tests for the per-panel integrated-gradients attribution (#297).

Covers the IG kernel against a known-good fixture (zeroed input ⇒ zero
attribution; non-zero input ⇒ non-zero attribution on the responsible
feature family) and the structured-degrade payload when a panel cannot
be explained.
"""

from __future__ import annotations

import torch

from app.services.xai_attribution import (
    DEFAULT_N_STEPS,
    MAX_N_STEPS,
    FeatureFamilyAttribution,
    PanelAttribution,
    aggregate_feature_families,
    attribute_rates_panel,
    attribute_regime_panel,
    attribute_trajectory_panel,
    integrated_gradients,
    resolve_n_steps,
)


class _LinearModel(torch.nn.Module):
    """Minimal model the IG tests can backprop against.

    Mirrors the rates head shape: per-bar features ``(B, T, F)`` →
    scalar bps prediction. The linear weights pick out a single feature
    so we can read the attribution back analytically.
    """

    def __init__(self, feature_size: int, target_idx: int):
        super().__init__()
        self.feature_size = feature_size
        self.target_idx = target_idx
        weight = torch.zeros(feature_size)
        weight[target_idx] = 1.0
        self.register_buffer("weight", weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Sum across time so the gradient flows back to every bar.
        return (x * self.weight).sum(dim=(1, 2))


def test_resolve_n_steps_default():
    assert resolve_n_steps() == DEFAULT_N_STEPS


def test_resolve_n_steps_clamps_high():
    assert resolve_n_steps(override=10_000) == MAX_N_STEPS


def test_resolve_n_steps_clamps_low():
    # Single-step IG collapses to a finite difference; clamp to 2.
    assert resolve_n_steps(override=0) == 2
    assert resolve_n_steps(override=1) == 2


def test_resolve_n_steps_env_var(monkeypatch):
    monkeypatch.setenv("FED_PULSE_IG_N_STEPS", "8")
    assert resolve_n_steps() == 8
    monkeypatch.setenv("FED_PULSE_IG_N_STEPS", "not-a-number")
    assert resolve_n_steps() == DEFAULT_N_STEPS


def test_integrated_gradients_zero_input_yields_zero_attribution():
    # IG_i = (x_i - 0) * gradient_avg. For a zero input the (x - x')
    # factor is zero so attribution must be zero element-wise. Holds
    # regardless of the model's response to the baseline.
    model = _LinearModel(feature_size=6, target_idx=2)
    x = torch.zeros((1, 5, 6))
    attribution = integrated_gradients(lambda t: model(t), x, n_steps=10)
    assert torch.allclose(attribution, torch.zeros_like(x))


def test_integrated_gradients_picks_responsible_feature():
    # The linear model only reads feature index 2; non-zero attribution
    # must land on that index and only that index.
    feature_size = 6
    target_idx = 2
    model = _LinearModel(feature_size=feature_size, target_idx=target_idx)
    x = torch.randn((1, 4, feature_size))
    attribution = integrated_gradients(lambda t: model(t), x, n_steps=20)
    # Target feature attribution equals the input value at that slot
    # (gradient = 1, baseline = 0). Other features have zero gradient.
    expected_target = x[..., target_idx]
    actual_target = attribution[..., target_idx]
    assert torch.allclose(actual_target, expected_target, atol=1e-5)
    # Off-target features attribute to ~zero.
    off_target = attribution[..., [i for i in range(feature_size) if i != target_idx]]
    assert torch.allclose(off_target, torch.zeros_like(off_target), atol=1e-5)


def test_aggregate_feature_families_buckets_by_slice():
    # Build a hand-crafted attribution where only the "credibility"
    # slice (indices 6..10 in the rich-feature layout) carries mass.
    from app.models.config import RICH_FEATURE_SIZE

    attribution = torch.zeros((1, 3, RICH_FEATURE_SIZE))
    attribution[..., 6:10] = 0.5  # all credibility features pushed up
    families = aggregate_feature_families(attribution, feature_size=RICH_FEATURE_SIZE)
    by_name = {item.family: item for item in families}
    assert by_name["credibility"].magnitude > 0
    assert by_name["credibility"].signed > 0
    # Every other family must collapse to zero.
    for name, item in by_name.items():
        if name == "credibility":
            continue
        assert item.magnitude == 0.0
        assert item.signed == 0.0


def test_aggregate_feature_families_handles_legacy_feature_size():
    # A 6-feature legacy checkpoint emits attribution of shape
    # ``(1, T, 6)``; slices beyond FEATURE_SIZE collapse to zero.
    attribution = torch.ones((1, 5, 6))
    families = aggregate_feature_families(attribution, feature_size=6)
    by_name = {item.family: item for item in families}
    # Market slice spans [0, 6) → fully populated.
    assert by_name["market"].magnitude == 30.0  # 5 bars * 6 features * 1
    # Everything else is past the feature_size and reads zero.
    for name in ("credibility", "linguistic", "mp_surprise", "multi_axis"):
        assert by_name[name].magnitude == 0.0


def test_attribute_regime_panel_degrades_on_regression_mode():
    # A bare object with output_mode='regression' must surface the
    # structured "not_classification_mode" payload rather than raise.
    class _Stub:
        output_mode = "regression"

    x = torch.zeros((1, 2, 6))
    result = attribute_regime_panel(_Stub(), x)
    assert isinstance(result, PanelAttribution)
    assert result.unavailable is True
    assert result.reason == "not_classification_mode"
    assert result.families == []


def test_attribute_regime_panel_degrades_when_no_multi_task_forward():
    class _Stub:
        output_mode = "classification"

    x = torch.zeros((1, 2, 6))
    result = attribute_regime_panel(_Stub(), x)
    assert result.unavailable is True
    assert result.reason == "no_multi_task_forward"


def test_attribute_rates_panel_degrades_when_head_not_mounted():
    class _Stub:
        rates_heads_active = ("2y",)

        def forward_multi_task(self, *args, **kwargs):  # pragma: no cover -- never called
            raise AssertionError("should not be reached")

    x = torch.zeros((1, 2, 6))
    result = attribute_rates_panel(_Stub(), x, head_name="terminal")
    assert result.unavailable is True
    assert result.reason == "head_not_mounted"


def test_attribute_trajectory_panel_degrades_on_missing_bundle():
    inputs = torch.zeros((1, 3, 8))
    result = attribute_trajectory_panel(None, inputs)
    assert result.unavailable is True
    assert result.reason == "bundle_not_loaded"


def test_attribute_trajectory_panel_runs_on_small_model():
    feature_size = 4

    class _TrajStub(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(feature_size, 3)

        def forward(self, inputs: torch.Tensor, mask=None):  # noqa: ARG002
            pooled = inputs.mean(dim=1)
            logits = self.linear(pooled)
            return logits, pooled

    model = _TrajStub()
    inputs = torch.randn((1, 5, feature_size))
    result = attribute_trajectory_panel(model, inputs)
    assert result.unavailable is False
    assert len(result.families) == 1
    assert result.families[0].family == "trajectory_input"
    # Non-zero input should give non-zero magnitude.
    assert result.families[0].magnitude > 0.0


def test_panel_attribution_to_dict_round_trip():
    item = PanelAttribution(
        panel="rates_2y",
        target="rates_2y_bps",
        families=[FeatureFamilyAttribution(family="market", magnitude=1.5, signed=-0.5)],
        n_steps=20,
    )
    payload = item.to_dict()
    assert payload["panel"] == "rates_2y"
    assert payload["target"] == "rates_2y_bps"
    assert payload["n_steps"] == 20
    assert payload["unavailable"] is False
    assert payload["reason"] is None
    assert payload["families"][0]["family"] == "market"
    assert payload["families"][0]["magnitude"] == 1.5
    assert payload["families"][0]["signed"] == -0.5
