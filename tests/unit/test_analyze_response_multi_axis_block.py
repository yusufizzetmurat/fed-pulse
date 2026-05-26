"""Multi-axis block on the /analyze response (#78).

Two paths the builder routes between:

- ``TextMultiAxisClassifier`` checkpoint present → all four cards
  populated from the classifier service.
- No checkpoint → stance card sourced from the existing sentiment
  classifier output; factor / certainty / topic stay None.

The frontend renders None cards as absent (no placeholder), so the
contract is "stance always present; the others fill when the
classifier ships a checkpoint".
"""

from __future__ import annotations


def test_multi_axis_block_falls_back_to_sentiment_when_classifier_absent(
    monkeypatch,
) -> None:
    """Without a trained classifier checkpoint, stance comes from the
    existing sentiment output and the other three axes default to None."""

    from app.main import _build_multi_axis_block
    from app.services import multi_axis_classifier as svc

    monkeypatch.setattr(svc, "score_text", lambda _text: None)
    sentiment = {
        "label": "hawkish",
        "score": 0.82,
        "raw": [
            {"label": "hawkish", "score": 0.82},
            {"label": "neutral", "score": 0.12},
            {"label": "dovish", "score": 0.06},
        ],
    }
    block = _build_multi_axis_block("text body", sentiment)
    assert block is not None
    assert block["stance"]["label"] == "hawkish"
    assert 0.0 <= block["stance"]["confidence"] <= 1.0
    distribution = block["stance"]["distribution"]
    assert set(distribution.keys()) == {"hawkish", "dovish", "neutral"}
    assert abs(distribution["hawkish"] - 0.82) < 1e-6
    assert block["factor"] is None
    assert block["certainty"] is None
    assert block["topic"] is None


def test_multi_axis_block_uses_classifier_when_checkpoint_loaded(monkeypatch) -> None:
    """When the classifier returns a populated block, the /analyze
    response surfaces it verbatim (all four cards with real values)."""

    from app.main import _build_multi_axis_block
    from app.services import multi_axis_classifier as svc

    classifier_block = {
        "stance": {
            "label": "dovish",
            "confidence": 0.71,
            "distribution": {"hawkish": 0.12, "dovish": 0.71, "neutral": 0.17},
        },
        "factor": {"value": -0.42, "confidence": 0.42},
        "certainty": {
            "label": "uncertain",
            "confidence": 0.65,
            "distribution": {"certain": 0.18, "uncertain": 0.65, "neutral": 0.17},
        },
        "topic": {
            "label": "forward_guidance",
            "confidence": 0.58,
            "distribution": {
                "macro": 0.20,
                "forward_guidance": 0.58,
                "market_reaction": 0.12,
                "other": 0.10,
            },
        },
    }
    monkeypatch.setattr(svc, "score_text", lambda _text: classifier_block)
    sentiment = {"label": "hawkish", "score": 0.5, "raw": []}
    block = _build_multi_axis_block("text body", sentiment)
    assert block == classifier_block


def test_multi_axis_block_coerces_unknown_label_to_neutral(monkeypatch) -> None:
    """If the upstream sentiment classifier returns a label outside the
    canonical set (case mismatch, stale checkpoint, fallback model),
    the fallback path collapses to neutral with the available distribution."""

    from app.main import _build_multi_axis_block
    from app.services import multi_axis_classifier as svc

    monkeypatch.setattr(svc, "score_text", lambda _text: None)
    sentiment = {
        "label": "UNKNOWN",
        "score": 0.0,
        "raw": [],
    }
    block = _build_multi_axis_block("text body", sentiment)
    assert block is not None
    assert block["stance"]["label"] == "neutral"
    assert block["stance"]["confidence"] == 0.0
