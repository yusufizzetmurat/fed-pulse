from __future__ import annotations

import sys
import types

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


def test_score_passage_handles_empty_text_response_as_neutral() -> None:
    class _NoneTextModel:
        def generate_content(self, prompt, **kwargs):
            return type("R", (), {"text": None})()

    result = gemini_client.score_passage("text", model=_NoneTextModel())
    assert result["label"] == "neutral"
    assert result["confidence"] == 0.0
    assert result["raw"] == ""


def test_score_passage_defaults_confidence_to_zero_when_not_numeric() -> None:
    stub = _StubModel('{"label": "hawkish", "confidence": "strong"}')
    result = gemini_client.score_passage("text", model=stub)
    assert result["label"] == "hawkish"
    assert result["confidence"] == 0.0


def test_score_passage_falls_back_when_response_json_is_not_an_object() -> None:
    stub = _StubModel('["neutral", 0.5]')
    result = gemini_client.score_passage("text", model=stub)
    assert result["label"] == "neutral"
    assert result["confidence"] == 0.0


@pytest.mark.parametrize(
    ("raw_confidence", "expected"),
    [
        (4.2, 1.0),
        (-2.0, 0.0),
    ],
)
def test_score_passage_clamps_confidence_into_unit_interval(raw_confidence, expected) -> None:
    stub = _StubModel(f'{{"label": "neutral", "confidence": {raw_confidence}}}')
    result = gemini_client.score_passage("text", model=stub)
    assert result["label"] == "neutral"
    assert result["confidence"] == pytest.approx(expected)


def test_load_model_requires_api_key(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY is not set"):
        gemini_client.load_model()


def test_model_adapter_calls_genai_client_with_deterministic_config(monkeypatch) -> None:
    class _FakeModels:
        def __init__(self):
            self.calls = []

        def generate_content(self, **kwargs):
            self.calls.append(kwargs)
            return type("Resp", (), {"text": '{"label":"dovish","confidence":0.8}'})()

    class _FakeClient:
        def __init__(self, api_key):
            self.api_key = api_key
            self.models = _FakeModels()

    created_clients = []

    def _client_factory(*, api_key):
        client = _FakeClient(api_key)
        created_clients.append(client)
        return client

    fake_genai = types.ModuleType("google.genai")
    fake_genai.Client = _client_factory
    fake_genai.types = types.SimpleNamespace(
        GenerateContentConfig=lambda **kwargs: {"temperature": kwargs.get("temperature")}
    )
    fake_google = types.ModuleType("google")
    fake_google.genai = fake_genai

    monkeypatch.setenv("GEMINI_API_KEY", "unit-test-key")
    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)

    adapter = gemini_client.load_model("gemini-test-model")
    response = adapter.generate_content("Monetary policy statement")
    assert response.text == '{"label":"dovish","confidence":0.8}'
    assert created_clients[0].api_key == "unit-test-key"
    assert created_clients[0].models.calls == [
        {
            "model": "gemini-test-model",
            "contents": "Monetary policy statement",
            "config": {"temperature": 0.0},
        }
    ]


class _StubEmbeddingModel:
    def __init__(self, embeddings):
        self._embeddings = list(embeddings)

    def embed_content(self, content, **kwargs):
        v = self._embeddings.pop(0)
        wrapper = type("R", (), {"embedding": type("E", (), {"values": v})()})
        return wrapper()


def test_embed_text_returns_list_of_floats() -> None:
    stub = _StubEmbeddingModel([[0.1, 0.2, 0.3, 0.4]])
    result = gemini_client.embed_text("hello world", model=stub)
    assert isinstance(result, list)
    assert len(result) == 4
    assert result == [0.1, 0.2, 0.3, 0.4]


def test_embed_text_handles_empty_string_returns_zero_vector() -> None:
    """Empty input should not raise; return a zero vector of the configured
    default embedding dim (768)."""

    result = gemini_client.embed_text("", model=None)
    assert isinstance(result, list)
    assert len(result) == 768
    assert all(v == 0.0 for v in result)


def test_load_embedding_model_requires_api_key(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY is not set"):
        gemini_client.load_embedding_model()


def test_embedding_adapter_wraps_new_genai_embedding_shape(monkeypatch) -> None:
    class _FakeModels:
        def __init__(self):
            self.calls = []

        def embed_content(self, **kwargs):
            self.calls.append(kwargs)
            embedding = type("Embedding", (), {"values": [0.5, 1.0, 2.5]})()
            return type("Resp", (), {"embeddings": [embedding]})()

    class _FakeClient:
        def __init__(self, api_key):
            self.api_key = api_key
            self.models = _FakeModels()

    created_clients = []

    def _client_factory(*, api_key):
        client = _FakeClient(api_key)
        created_clients.append(client)
        return client

    fake_genai = types.ModuleType("google.genai")
    fake_genai.Client = _client_factory
    fake_google = types.ModuleType("google")
    fake_google.genai = fake_genai

    monkeypatch.setenv("GEMINI_API_KEY", "embedding-test-key")
    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)

    adapter = gemini_client.load_embedding_model("embedding-model-x")
    wrapped = adapter.embed_content("fomc")
    assert wrapped.embedding.values == [0.5, 1.0, 2.5]
    assert created_clients[0].api_key == "embedding-test-key"
    assert created_clients[0].models.calls == [
        {
            "model": "embedding-model-x",
            "contents": "fomc",
        }
    ]
