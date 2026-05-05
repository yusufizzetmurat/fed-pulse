from __future__ import annotations

import pytest

from app.services import gemini_client


class _StubModel:
    def __init__(self, response_text: str):
        self._response_text = response_text

    def generate_content(self, prompt, **kwargs):
        class _R:
            def __init__(self, text):
                self.text = text

        return _R(self._response_text)


def test_score_passage_parses_label_and_confidence_from_clean_response() -> None:
    stub = _StubModel('{"label": "hawkish", "confidence": 0.91}')
    result = gemini_client.score_passage(
        "We see persistent inflation pressures",
        model=stub,
    )
    assert result["label"] == "hawkish"
    assert result["confidence"] == pytest.approx(0.91)
    assert "raw" in result


def test_score_passage_normalizes_label_case_and_whitespace() -> None:
    stub = _StubModel('{"label": "  Dovish  ", "confidence": 0.6}')
    result = gemini_client.score_passage("text", model=stub)
    assert result["label"] == "dovish"


def test_score_passage_falls_back_to_neutral_when_response_unparseable() -> None:
    stub = _StubModel("I cannot decide.")
    result = gemini_client.score_passage("text", model=stub)
    assert result["label"] == "neutral"
    assert result["confidence"] == 0.0
    assert result["raw"] == "I cannot decide."


def test_score_passage_falls_back_to_neutral_on_unknown_label() -> None:
    stub = _StubModel('{"label": "uncertain", "confidence": 0.3}')
    result = gemini_client.score_passage("text", model=stub)
    assert result["label"] == "neutral"
    assert result["confidence"] == 0.0


def test_score_passage_strips_markdown_code_fences() -> None:
    stub = _StubModel('```json\n{"label": "neutral", "confidence": 0.5}\n```')
    result = gemini_client.score_passage("text", model=stub)
    assert result["label"] == "neutral"
    assert result["confidence"] == pytest.approx(0.5)
