"""Smoke test for the /analyze panel-attribution wire-up (#297).

The full /analyze route is exercised under tests/integration; here we
just verify that ``build_panel_attributions`` produces a list-of-dicts
shape the response builder can serialise into
:class:`app.schemas.XaiPanelAttribution` items without further coercion.

The function is also expected to never raise — it must surface
structured-degrade payloads for any panel that cannot be explained on
the active checkpoint. This test forces that contract by stubbing
``_get_model`` to a minimal classification-mode model.
"""

from __future__ import annotations

import torch

from app.services import forecaster as forecaster_service
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

    # The sequence arg is opaque here -- the stubbed builders ignore it.
    panels = forecaster_service.build_panel_attributions([], n_steps=4)
    assert isinstance(panels, list)
    assert len(panels) == 2
    by_panel = {item["panel"]: item for item in panels}
    assert "regime" in by_panel
    assert "rates_2y" in by_panel
    # Each entry has the schema-mandated keys.
    for payload in panels:
        assert {"panel", "target", "families", "n_steps", "unavailable", "reason"} <= payload.keys()
        assert isinstance(payload["families"], list)
        assert payload["unavailable"] is False
        # Each family carries the magnitude + signed pair.
        for fam in payload["families"]:
            assert {"family", "magnitude", "signed"} <= fam.keys()


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
    panels = forecaster_service.build_panel_attributions([], n_steps=4)
    by_panel = {item["panel"]: item for item in panels}
    assert "regime" not in by_panel
    assert "rates_2y" in by_panel


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
