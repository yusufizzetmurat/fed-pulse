"""Cover the /analyze regime_classification response branch (#216).

When the active checkpoint is classification mode AND a sibling
``.conformal.json`` manifest with ``softmax_quantile`` exists, the
``/analyze`` response carries a populated ``RegimeClassificationCard``.
Without either, the field stays ``None`` and the legacy stance card
in ``sentiment`` remains the only confidence surface.
"""

from __future__ import annotations


def test_safe_regime_classification_swallows_exceptions(monkeypatch, caplog) -> None:
    """``_safe_regime_classification`` must never raise. #341 promoted
    the previous bare-None swallow to a structured payload carrying
    ``status="unexpected_exception"`` + the exception class so the
    operator can grep the response for the failure mode; /analyze
    stays 200 even with a broken classifier checkpoint. The raw
    exception message is kept in the WARNING log only (never in the
    client-facing payload) so internal detail does not leak through
    the API."""

    import logging

    from app.main import _safe_regime_classification
    import app.main as main_module

    def _boom(_vectors):
        raise RuntimeError("simulated broken checkpoint")

    monkeypatch.setattr(main_module, "build_regime_classification_card", _boom)
    with caplog.at_level(logging.WARNING, logger="app.main"):
        out = _safe_regime_classification([])
    assert isinstance(out, dict)
    assert out["status"] == "unexpected_exception"
    assert out["exception_class"] == "RuntimeError"
    assert "detail" not in out
    assert any(
        "simulated broken checkpoint" in record.getMessage()
        for record in caplog.records
    ), "raw exception detail must reach the WARNING log even when stripped from the response"


def test_safe_regime_classification_typeerror_surfaces_kwarg(monkeypatch) -> None:
    """#341: a ``TypeError`` raised by the forward path surfaces the
    missing kwarg name through the structured-state surface."""

    from app.main import _safe_regime_classification
    import app.main as main_module

    def _missing(_vectors):
        raise TypeError(
            "forward_multi_task() missing 1 required keyword-only argument: 'text_embedding'"
        )

    monkeypatch.setattr(main_module, "build_regime_classification_card", _missing)
    out = _safe_regime_classification([])
    assert isinstance(out, dict)
    assert out["status"] == "inference_kwarg_missing"
    assert out["missing_kwarg"] == "text_embedding"


def test_safe_regime_classification_passes_through_card(monkeypatch) -> None:
    """When the classifier service returns a card dict, the wrapper
    passes it through verbatim — the card lands on the response
    without further mangling."""

    from app.main import _safe_regime_classification
    import app.main as main_module

    card = {
        "predicted_set": ["normal", "high"],
        "set_label": "{normal, high}",
        "set_size": 2,
        "coverage": 0.8,
        "distribution": {"calm": 0.18, "normal": 0.52, "high": 0.30},
        "argmax_class": "normal",
    }
    monkeypatch.setattr(
        main_module, "build_regime_classification_card", lambda _v: card
    )
    out = _safe_regime_classification([])
    assert out == card


def test_safe_regime_classification_passes_through_none(monkeypatch) -> None:
    """When the classifier service returns ``None`` (regression-only
    checkpoint or no manifest), #341 promoted the legacy bare-None to
    a structured ``status="not_classification_mode"`` payload so the
    operator can tell ``model deliberately mute`` apart from ``model
    crashed silently``. The /analyze route handler splits this into
    the sibling ``regime_classification_status`` field; the legacy
    ``regime_classification`` slot lands as None."""

    from app.main import _safe_regime_classification
    import app.main as main_module

    monkeypatch.setattr(
        main_module, "build_regime_classification_card", lambda _v: None
    )
    out = _safe_regime_classification([])
    assert isinstance(out, dict)
    assert out["status"] == "not_classification_mode"
