from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from app.data.llm_zero_shot_execution import _build_prompt, _parse_label  # noqa: E402


class _StubTokenizerNoChatTemplate:
    pass


def test_parse_label_picks_first_match():
    assert _parse_label("hawkish") == "hawkish"
    assert _parse_label("Dovish.") == "dovish"
    assert _parse_label("The answer is NEUTRAL.") == "neutral"


def test_parse_label_handles_explanations():
    assert _parse_label("This is hawkish because the Fed signaled tightening.") == "hawkish"
    assert _parse_label("Definitely dovish given the rate cut hint.") == "dovish"


def test_parse_label_falls_back_to_neutral_on_no_match():
    assert _parse_label("indeterminate") == "neutral"
    assert _parse_label("") == "neutral"


def test_build_prompt_falls_back_when_chat_template_unsupported():
    text = "The Committee maintained the target range."
    rendered = _build_prompt(text, _StubTokenizerNoChatTemplate())
    assert text in rendered
    assert "hawkish" in rendered.lower()
    assert "dovish" in rendered.lower()
    assert "neutral" in rendered.lower()
