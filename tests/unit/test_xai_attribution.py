from __future__ import annotations

from app.evaluation.xai import (
    attribute_sentence,
    attribute_text,
    split_sentences,
    to_response,
)


def test_split_sentences_handles_basic_punctuation():
    text = "Recent indicators expanded. Inflation remains elevated! The Committee is committed."
    sentences = split_sentences(text)
    assert len(sentences) == 3
    assert sentences[0].startswith("Recent")
    assert sentences[-1].endswith(".")


def test_attribute_sentence_picks_hawkish_keywords():
    attr = attribute_sentence("The Committee is committed to tightening policy decisively.")
    assert attr.score > 0
    tokens = {token.token for token in attr.top_tokens}
    assert "tightening" in tokens
    assert "committed" in tokens
    assert "decisively" in tokens


def test_attribute_sentence_picks_dovish_keywords():
    attr = attribute_sentence("Easing patient stance with cuts and accommodative stimulus.")
    assert attr.score < 0
    tokens = {token.token for token in attr.top_tokens}
    assert "easing" in tokens
    assert "cuts" in tokens
    assert "accommodative" in tokens


def test_attribute_sentence_neutral_text_yields_zero_score():
    attr = attribute_sentence("The Committee reviewed available data and discussed alternatives.")
    assert attr.score == 0
    assert attr.top_tokens == []


def test_attribute_sentence_score_is_bounded():
    # 20 hawkish hits should still produce a score strictly less than 1.0.
    text = " ".join(["tighten"] * 20)
    attr = attribute_sentence(text)
    assert 0 < attr.score < 1.0


def test_attribute_text_returns_one_entry_per_sentence():
    text = "Inflation is elevated. The Committee is patient about easing."
    attrs = attribute_text(text)
    assert len(attrs) == 2
    assert attrs[0].score > 0
    assert attrs[1].score < 0


def test_to_response_matches_frontend_contract():
    attrs = attribute_text("Inflation remains elevated.")
    payload = to_response(attrs)
    assert payload["method"] == "keyword_salience_v1"
    assert isinstance(payload["sentences"], list)
    assert payload["sentences"][0]["text"].startswith("Inflation")
    assert "topTokens" in payload["sentences"][0]
    if payload["sentences"][0]["topTokens"]:
        first_token = payload["sentences"][0]["topTokens"][0]
        assert {"token", "weight"} <= set(first_token.keys())
