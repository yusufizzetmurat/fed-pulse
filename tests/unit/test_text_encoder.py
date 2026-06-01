from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
from transformers import BatchEncoding  # noqa: E402

from app.services.text_encoder import (  # noqa: E402
    ChunkEncoding,
    aggregate_label,
    encode_chunks,
    split_into_chunks,
)


class _FakeTokenizer:
    def __init__(self, tokens_per_char: float = 1.0) -> None:
        self.tokens_per_char = tokens_per_char

    def encode(self, text: str, add_special_tokens: bool = False, truncation: bool = False) -> list[int]:
        return list(range(max(1, int(len(text) * self.tokens_per_char))))

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return f"chunk[{ids[0]}:{ids[-1] + 1}]"

    def __call__(
        self,
        text: str | list[str],
        truncation: bool = False,
        max_length: int = 512,
        padding: bool | str = False,
        return_tensors: str | None = None,
    ) -> BatchEncoding:
        batch = [text] if isinstance(text, str) else list(text)
        seq_len = 4
        return BatchEncoding(
            {
                "input_ids": torch.zeros((len(batch), seq_len), dtype=torch.long),
                "attention_mask": torch.ones((len(batch), seq_len), dtype=torch.long),
            }
        )


@dataclass
class _FakeModelOutput:
    hidden_states: tuple[torch.Tensor, ...]


class _FakeModel:
    def __init__(self, hidden_size: int = 6) -> None:
        self.hidden_size = hidden_size
        self._param = torch.nn.Parameter(torch.zeros(1))

    def parameters(self):
        yield self._param

    def __call__(self, **inputs: Any) -> _FakeModelOutput:
        input_ids = inputs["input_ids"]
        batch, seq_len = input_ids.shape
        hidden = torch.full((batch, seq_len, self.hidden_size), 0.5)
        return _FakeModelOutput(hidden_states=(hidden, hidden))


class _FakeClassifier:
    def __init__(self, tokens_per_char: float = 1.0, hidden_size: int = 6) -> None:
        self.tokenizer = _FakeTokenizer(tokens_per_char=tokens_per_char)
        self.model = _FakeModel(hidden_size=hidden_size)

    def __call__(self, text: str, truncation: bool = False, max_length: int = 512):
        return [
            [
                {"label": "hawkish", "score": 0.6},
                {"label": "dovish", "score": 0.3},
                {"label": "neutral", "score": 0.1},
            ]
        ]


def test_split_into_chunks_returns_single_chunk_for_short_text():
    classifier = _FakeClassifier(tokens_per_char=1.0)
    chunks = split_into_chunks("short text", classifier=classifier, max_tokens=480, stride=400)
    assert chunks == ["short text"]


def test_split_into_chunks_splits_long_text_with_stride():
    classifier = _FakeClassifier(tokens_per_char=1.0)
    long_text = "a" * 1000
    chunks = split_into_chunks(long_text, classifier=classifier, max_tokens=400, stride=300)
    assert len(chunks) >= 3
    assert all(chunk.startswith("chunk[") for chunk in chunks)


def test_split_into_chunks_handles_empty_text():
    classifier = _FakeClassifier()
    assert split_into_chunks("", classifier=classifier) == []


def test_encode_chunks_returns_one_encoding_per_chunk():
    classifier = _FakeClassifier(tokens_per_char=1.0, hidden_size=8)
    encodings = encode_chunks("hello world", classifier=classifier)
    assert len(encodings) == 1
    encoding = encodings[0]
    assert isinstance(encoding, ChunkEncoding)
    assert encoding.text == "hello world"
    assert len(encoding.embedding) == 8
    assert encoding.scores[0]["label"] == "hawkish"


def test_encode_chunks_emits_per_chunk_embeddings_for_long_text():
    classifier = _FakeClassifier(tokens_per_char=1.0, hidden_size=4)
    long_text = "x" * 1500
    encodings = encode_chunks(long_text, classifier=classifier)
    assert len(encodings) >= 2
    for encoding in encodings:
        assert len(encoding.embedding) == 4
        assert encoding.scores


def test_aggregate_label_averages_scores_across_chunks():
    encodings = [
        ChunkEncoding(
            text="a",
            embedding=[],
            scores=[{"label": "hawkish", "score": 0.8}, {"label": "dovish", "score": 0.2}],
        ),
        ChunkEncoding(
            text="b",
            embedding=[],
            scores=[{"label": "hawkish", "score": 0.4}, {"label": "dovish", "score": 0.6}],
        ),
    ]
    result = aggregate_label(encodings)
    assert result["label"] == "hawkish"
    assert result["score"] == pytest.approx(0.6)
    raw_by_label = {item["label"]: item["score"] for item in result["raw"]}
    assert raw_by_label["dovish"] == pytest.approx(0.4)


def test_aggregate_label_handles_empty_input():
    result = aggregate_label([])
    assert result == {"label": "UNKNOWN", "score": 0.0, "raw": []}


def test_assert_primary_model_loaded_raises_on_fallback(monkeypatch):
    # The silent-fallback guard: if the active model is the fallback (e.g. the
    # primary HF repo 404'd -> distilbert), embedding builds must fail loudly.
    import app.services.text_encoder as te

    monkeypatch.setattr(te, "get_classifier", lambda: None)
    monkeypatch.setattr(te, "MODEL_ID", "primary/stance-model")
    monkeypatch.setattr(te, "_loaded_model_id", te.FALLBACK_MODEL_ID)
    with pytest.raises(RuntimeError, match="refusing to build embeddings"):
        te.assert_primary_model_loaded()


def test_assert_primary_model_loaded_passes_when_primary(monkeypatch):
    import app.services.text_encoder as te

    monkeypatch.setattr(te, "get_classifier", lambda: None)
    monkeypatch.setattr(te, "MODEL_ID", "primary/stance-model")
    monkeypatch.setattr(te, "_loaded_model_id", "primary/stance-model")
    te.assert_primary_model_loaded()  # must not raise


def test_get_loaded_model_id_accessor(monkeypatch):
    import app.services.text_encoder as te

    monkeypatch.setattr(te, "_loaded_model_id", "some/model")
    assert te.get_loaded_model_id() == "some/model"


class _StanceCfg:
    pass


class _StanceModel:
    def __init__(self):
        self.config = _StanceCfg()


class _StanceClf:
    def __init__(self):
        self.model = _StanceModel()


def test_fomc_roberta_generic_labels_remapped_to_stance(monkeypatch):
    # FOMC-RoBERTa ships LABEL_0/1/2; _build_pipeline must remap to stance names.
    import app.services.text_encoder as te

    monkeypatch.setattr(te, "pipeline", lambda *a, **k: _StanceClf())
    monkeypatch.setattr(te, "revision_for", lambda _m: None)
    clf = te._build_pipeline(te.PRIMARY_HF_MODEL_ID, -1)
    assert clf.model.config.id2label == {0: "dovish", 1: "hawkish", 2: "neutral"}
    assert clf.model.config.label2id["hawkish"] == 1


def test_other_model_labels_not_remapped(monkeypatch):
    # A non-FOMC-RoBERTa model id must NOT get the stance remap.
    import app.services.text_encoder as te

    monkeypatch.setattr(te, "pipeline", lambda *a, **k: _StanceClf())
    monkeypatch.setattr(te, "revision_for", lambda _m: None)
    clf = te._build_pipeline("some/other-classifier", -1)
    assert not hasattr(clf.model.config, "id2label")
