"""Cover the /analyze regime_classification response branch (#216).

When the active checkpoint is classification mode AND a sibling
``.conformal.json`` manifest with ``softmax_quantile`` exists, the
``/analyze`` response carries a populated ``RegimeClassificationCard``.
Without either, the field stays ``None`` and the legacy stance card
in ``sentiment`` remains the only confidence surface.
"""

from __future__ import annotations


def test_safe_regime_classification_swallows_exceptions(monkeypatch) -> None:
    """``_safe_regime_classification`` must never raise — any failure
    inside the inference + calibrated-set path degrades to ``None``
    so /analyze stays 200 even with a broken classifier checkpoint."""

    from app.main import _safe_regime_classification
    import app.main as main_module

    def _boom(_vectors):
        raise RuntimeError("simulated broken checkpoint")

    monkeypatch.setattr(main_module, "build_regime_classification_card", _boom)
    assert _safe_regime_classification([]) is None


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
    checkpoint or no manifest), the wrapper also returns ``None``
    so the response field stays absent."""

    from app.main import _safe_regime_classification
    import app.main as main_module

    monkeypatch.setattr(
        main_module, "build_regime_classification_card", lambda _v: None
    )
    assert _safe_regime_classification([]) is None
