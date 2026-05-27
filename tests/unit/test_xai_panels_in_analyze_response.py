"""Smoke test for the /analyze panel-attribution wire-up (#297, #385).

The full /analyze route is exercised under tests/integration; here we
just verify that ``build_panel_attributions`` produces a list-of-dicts
shape the response builder can serialise into
:class:`app.schemas.XaiPanelAttribution` items without further coercion.

The function is also expected to never raise — it must surface
structured-degrade payloads for any panel that cannot be explained on
the active checkpoint. This test forces that contract by stubbing
``_get_model`` to a minimal classification-mode model.

The #385 coverage stubs the trajectory singleton with a known-good
bundle and asserts the trajectory panel surfaces alongside regime +
rates on the returned panel list.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import torch

from app.services import forecaster as forecaster_service
from app.services import trajectory as trajectory_service
from app.services.xai_attribution import PanelAttribution


class _StubModel(torch.nn.Module):
    """Minimal classification-mode model with one mounted rates head."""

    def __init__(self) -> None:
        super().__init__()
        self.output_mode = "classification"
        self.input_size = 6
        self.rates_heads_active = ("2y",)
        self.credibility_features = False
        self._text_path_active = False
        # Tiny linear stance head so the IG run can backprop.
        self.stance_w = torch.nn.Linear(6, 3)
        self.rates_w = torch.nn.Linear(6, 1)

    def parameters(self):  # noqa: D401 -- required for device probe
        return super().parameters()

    def forward_multi_task(self, x: torch.Tensor, **_kwargs):
        pooled = x.mean(dim=1)
        stance = self.stance_w(pooled)
        bps = self.rates_w(pooled).squeeze(-1)
        return {"stance": stance, "rates_2y_bps": bps}


def test_build_panel_attributions_emits_serialisable_dicts(monkeypatch):
    stub = _StubModel()
    monkeypatch.setattr(forecaster_service, "_get_model", lambda: stub)

    # Bypass the rich-feature scaler / lookback builder by stubbing the
    # tensor builder + lookback to keep the test self-contained.
    monkeypatch.setattr(
        forecaster_service,
        "_build_inference_tensor",
        lambda sequence, model, device: torch.randn((1, 4, 6)),
    )
    monkeypatch.setattr(
        forecaster_service, "build_lookback_sequence", lambda sequence: sequence
    )
    # No trajectory bundle installed; the trajectory panel should still
    # surface but as a structured ``unavailable`` payload.
    monkeypatch.setattr(trajectory_service, "get_state", lambda: None)

    # The sequence arg is opaque here -- the stubbed builders ignore it.
    panels = forecaster_service.build_panel_attributions([], n_steps=4)
    assert isinstance(panels, list)
    assert len(panels) == 3
    by_panel = {item["panel"]: item for item in panels}
    assert "regime" in by_panel
    assert "rates_2y" in by_panel
    assert "trajectory" in by_panel
    # Each entry has the schema-mandated keys.
    for payload in panels:
        assert {"panel", "target", "families", "n_steps", "unavailable", "reason"} <= payload.keys()
        assert isinstance(payload["families"], list)
        # Each family carries the magnitude + signed pair.
        for fam in payload["families"]:
            assert {"family", "magnitude", "signed"} <= fam.keys()
    # Regime + rates ran against the stub model; trajectory degrades.
    assert by_panel["regime"]["unavailable"] is False
    assert by_panel["rates_2y"]["unavailable"] is False
    assert by_panel["trajectory"]["unavailable"] is True
    assert by_panel["trajectory"]["reason"] == "bundle_not_loaded"


def test_build_panel_attributions_degrades_when_model_get_fails(monkeypatch):
    """If ``_get_model`` raises (cold start, missing checkpoint), the
    helper must return an empty list -- never propagate the exception
    into the /analyze response.
    """

    def _raise():
        raise RuntimeError("no checkpoint on disk")

    monkeypatch.setattr(forecaster_service, "_get_model", _raise)
    assert forecaster_service.build_panel_attributions([]) == []


def test_build_panel_attributions_skips_regime_on_regression_checkpoint(monkeypatch):
    """A regression-output checkpoint must not surface a regime panel.

    The rates-head loop still runs because rates heads can be mounted
    in regression-output mode (#317 finding #11), but the regime branch
    is skipped entirely so the response carries only rates panels.
    """

    class _RegressionStub(_StubModel):
        def __init__(self) -> None:
            super().__init__()
            self.output_mode = "regression"

    monkeypatch.setattr(forecaster_service, "_get_model", lambda: _RegressionStub())
    monkeypatch.setattr(
        forecaster_service,
        "_build_inference_tensor",
        lambda sequence, model, device: torch.randn((1, 4, 6)),
    )
    monkeypatch.setattr(
        forecaster_service, "build_lookback_sequence", lambda sequence: sequence
    )
    monkeypatch.setattr(trajectory_service, "get_state", lambda: None)
    panels = forecaster_service.build_panel_attributions([], n_steps=4)
    by_panel = {item["panel"]: item for item in panels}
    assert "regime" not in by_panel
    assert "rates_2y" in by_panel


class _TrajStubModel(torch.nn.Module):
    """Minimal trajectory model that matches the (inputs, mask) forward."""

    def __init__(self, feature_dim: int, n_classes: int = 3) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(feature_dim, n_classes)

    def forward(self, inputs: torch.Tensor, mask=None):  # noqa: ARG002
        pooled = inputs.mean(dim=1)
        logits = self.linear(pooled)
        return logits, pooled


def _install_known_good_trajectory_state(monkeypatch):
    """Install a known-good trajectory bundle on the singleton.

    Builds a 3-meeting history with non-zero embeddings + market blocks
    so the IG kernel sees a non-zero integration path, then wires the
    state via ``trajectory_service.get_state``. Returns the stubbed
    model so callers can assert against it if needed.
    """

    from app.trajectory.model import TrajectoryConfig

    embedding_dim = 4
    market_dim = 3  # MARKET_FEATURE_DIM
    feature_dim = embedding_dim + market_dim
    history_length = 3

    config = TrajectoryConfig(
        architecture="lstm",
        embedding_dim=embedding_dim,
        history_length=history_length,
        market_feature_dim=market_dim,
    )
    model = _TrajStubModel(feature_dim=feature_dim, n_classes=3)
    embeddings = np.array(
        [
            [0.3, -0.1, 0.2, 0.4],
            [0.1, 0.2, -0.3, 0.5],
            [0.4, 0.0, 0.1, -0.2],
        ],
        dtype=np.float32,
    )
    metadata = pd.DataFrame(
        {
            "event_date": ["2024-01-31", "2024-03-20", "2024-05-01"],
            "axis_stance": ["dovish", "neutral", "hawkish"],
            "embedding_2d_x": [0.0, 0.1, 0.2],
            "embedding_2d_y": [0.0, 0.1, 0.2],
            "text_hash": ["h0", "h1", "h2"],
            "pre_meeting_trailing_2y_yield_change_5d_bps": [1.0, -2.0, 0.5],
            "vix_close": [18.0, 22.0, 20.0],
        }
    )
    state = trajectory_service.build_state_for_tests(
        model=model,
        config=config,
        embeddings=embeddings,
        metadata=metadata,
        train_end="2024-12-31",
    )
    monkeypatch.setattr(trajectory_service, "get_state", lambda: state)
    return state


def test_build_panel_attributions_dispatches_trajectory_panel(monkeypatch):
    """#385: with a known-good trajectory bundle on the singleton, the
    helper must dispatch through ``attribute_trajectory_panel`` and
    surface the trajectory entry on the returned panel list with the
    coarse ``trajectory_input`` family bar."""

    stub = _StubModel()
    monkeypatch.setattr(forecaster_service, "_get_model", lambda: stub)
    monkeypatch.setattr(
        forecaster_service,
        "_build_inference_tensor",
        lambda sequence, model, device: torch.randn((1, 4, 6)),
    )
    monkeypatch.setattr(
        forecaster_service, "build_lookback_sequence", lambda sequence: sequence
    )
    _install_known_good_trajectory_state(monkeypatch)

    panels = forecaster_service.build_panel_attributions(
        [], n_steps=4, as_of_date=date(2024, 6, 15)
    )
    by_panel = {item["panel"]: item for item in panels}
    assert "trajectory" in by_panel
    trajectory_panel = by_panel["trajectory"]
    assert trajectory_panel["unavailable"] is False
    assert trajectory_panel["reason"] is None
    # The trajectory IG runner aggregates per-bar magnitudes into a
    # single coarse family bar per ADR 0026 lines 95-101.
    assert len(trajectory_panel["families"]) == 1
    family = trajectory_panel["families"][0]
    assert family["family"] == "trajectory_input"
    assert family["magnitude"] > 0.0
    # The dict must serialise into the pydantic schema without coercion.
    from app.schemas import XaiPanelAttribution

    parsed = XaiPanelAttribution.model_validate(trajectory_panel)
    assert parsed.panel == "trajectory"


def test_build_panel_attributions_trajectory_history_empty(monkeypatch):
    """When the strict-backward window selects no meetings (as_of_date
    falls before every metadata row), the trajectory panel still
    surfaces but with a structured ``trajectory_history_empty`` reason."""

    stub = _StubModel()
    monkeypatch.setattr(forecaster_service, "_get_model", lambda: stub)
    monkeypatch.setattr(
        forecaster_service,
        "_build_inference_tensor",
        lambda sequence, model, device: torch.randn((1, 4, 6)),
    )
    monkeypatch.setattr(
        forecaster_service, "build_lookback_sequence", lambda sequence: sequence
    )
    _install_known_good_trajectory_state(monkeypatch)

    panels = forecaster_service.build_panel_attributions(
        [], n_steps=4, as_of_date=date(2000, 1, 1)
    )
    by_panel = {item["panel"]: item for item in panels}
    assert by_panel["trajectory"]["unavailable"] is True
    assert by_panel["trajectory"]["reason"] == "trajectory_history_empty"


def test_panel_attribution_serialises_into_pydantic_schema():
    """Direct schema round-trip: a PanelAttribution dict must parse
    cleanly into :class:`app.schemas.XaiPanelAttribution` so the
    /analyze response handler doesn't need a coercion layer."""

    from app.schemas import XaiPanelAttribution

    item = PanelAttribution(
        panel="rates_2y", target="rates_2y_bps", families=[], n_steps=20
    )
    parsed = XaiPanelAttribution.model_validate(item.to_dict())
    assert parsed.panel == "rates_2y"
    assert parsed.unavailable is False
    assert parsed.reason is None
