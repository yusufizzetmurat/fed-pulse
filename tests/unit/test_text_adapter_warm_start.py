"""#327 text-adapter warm-start coverage.

The warm-start pipeline must:

- fit a non-zero adapter weight matrix on the proxy stance task
- persist the state_dict in a round-trippable layout
- load cleanly into the forecaster's ``text_adapter`` submodule
- reject in_dim / adapter_dim mismatch with a clear error
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from app.models.config import RICH_FEATURE_SIZE, ModelConfig  # noqa: E402
from app.models.factory import build_forecaster  # noqa: E402
from app.models.text_adapter_warm_start import (  # noqa: E402
    STANCE_LABEL_MAP,
    load_warm_start_into_adapter,
    pretrain_text_adapter,
)
from app.models.text_embedding_adapter import TextEmbeddingAdapter  # noqa: E402

TEXT_IN_DIM = 16
ADAPTER_DIM = 8


def _write_jsonl_corpus(path: Path, *, n_per_class: int = 30) -> int:
    rows: list[dict[str, object]] = []
    torch.manual_seed(13)
    for label, idx in STANCE_LABEL_MAP.items():
        centre = torch.randn(TEXT_IN_DIM) * 0.5 + idx * 0.5
        for _ in range(n_per_class):
            vec = (centre + torch.randn(TEXT_IN_DIM) * 0.1).tolist()
            rows.append(
                {
                    "text_embedding_pooled": vec,
                    "stance_label": label,
                }
            )
    path.write_text("\n".join(json.dumps(r) for r in rows))
    return len(rows)


def test_pretrain_writes_round_trippable_checkpoint(tmp_path: Path):
    corpus = tmp_path / "warm.jsonl"
    n = _write_jsonl_corpus(corpus)
    checkpoint = tmp_path / "adapter.pt"
    metadata = pretrain_text_adapter(
        corpus_path=corpus,
        output_path=checkpoint,
        adapter_dim=ADAPTER_DIM,
        epochs=3,
        batch_size=16,
        learning_rate=1e-3,
        seed=11,
    )
    assert checkpoint.exists()
    assert metadata["in_dim"] == TEXT_IN_DIM
    assert metadata["adapter_dim"] == ADAPTER_DIM
    assert metadata["n_samples"] == n
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert "state_dict" in payload and "metadata" in payload
    keys = list(payload["state_dict"].keys())
    assert any(k.startswith("text_adapter.") for k in keys)


def test_load_warm_start_into_adapter_overwrites_zero_init(tmp_path: Path):
    corpus = tmp_path / "warm.jsonl"
    _write_jsonl_corpus(corpus)
    checkpoint = tmp_path / "adapter.pt"
    pretrain_text_adapter(
        corpus_path=corpus,
        output_path=checkpoint,
        adapter_dim=ADAPTER_DIM,
        epochs=3,
        batch_size=16,
        learning_rate=1e-3,
        seed=11,
    )
    adapter = TextEmbeddingAdapter(in_dim=TEXT_IN_DIM, out_dim=ADAPTER_DIM, zero_init=True)
    # Sanity: zero-init starting point.
    assert torch.allclose(adapter.linear.weight, torch.zeros_like(adapter.linear.weight))
    metadata = load_warm_start_into_adapter(adapter, checkpoint)
    assert metadata["in_dim"] == TEXT_IN_DIM
    # After loading, the linear must NOT be all zeros.
    assert not torch.allclose(adapter.linear.weight, torch.zeros_like(adapter.linear.weight))


def test_load_rejects_dim_mismatch(tmp_path: Path):
    corpus = tmp_path / "warm.jsonl"
    _write_jsonl_corpus(corpus)
    checkpoint = tmp_path / "adapter.pt"
    pretrain_text_adapter(
        corpus_path=corpus,
        output_path=checkpoint,
        adapter_dim=ADAPTER_DIM,
        epochs=1,
        batch_size=16,
        seed=11,
    )
    wrong_in = TextEmbeddingAdapter(in_dim=TEXT_IN_DIM + 4, out_dim=ADAPTER_DIM)
    with pytest.raises(ValueError, match="in_dim mismatch"):
        load_warm_start_into_adapter(wrong_in, checkpoint)
    wrong_out = TextEmbeddingAdapter(in_dim=TEXT_IN_DIM, out_dim=ADAPTER_DIM + 2)
    with pytest.raises(ValueError, match="adapter_dim mismatch"):
        load_warm_start_into_adapter(wrong_out, checkpoint)


def test_warm_start_loads_into_forecaster_via_training_loop_hook(tmp_path: Path):
    """The training loop's ``_build_model`` honours ``text_adapter_warm_start``."""

    from app.training.loop import _build_model

    corpus = tmp_path / "warm.jsonl"
    _write_jsonl_corpus(corpus)
    checkpoint = tmp_path / "adapter.pt"
    pretrain_text_adapter(
        corpus_path=corpus,
        output_path=checkpoint,
        adapter_dim=ADAPTER_DIM,
        epochs=3,
        batch_size=16,
        seed=11,
    )
    config = ModelConfig(
        input_size=RICH_FEATURE_SIZE,
        hidden_size=16,
        num_layers=1,
        head_hidden_size=16,
        n_classes=3,
        text_embedding_dim=TEXT_IN_DIM,
        text_adapter_dim=ADAPTER_DIM,
    )
    model = _build_model(
        model_config=config, text_adapter_warm_start=str(checkpoint)
    )
    # The adapter is no longer at the zero-init starting point.
    assert not torch.allclose(
        model.text_adapter.linear.weight,
        torch.zeros_like(model.text_adapter.linear.weight),
    )
    # Metadata stashed on the model for downstream logging.
    assert getattr(model, "text_adapter_warm_start", None) is not None


def test_corpus_loader_handles_json_list(tmp_path: Path):
    """The corpus loader also accepts a JSON list of dicts."""

    rows = [
        {"text_embedding_pooled": [0.1] * TEXT_IN_DIM, "stance_label": "hawkish"},
        {"text_embedding_pooled": [0.2] * TEXT_IN_DIM, "stance_label": "dovish"},
        {"text_embedding_pooled": [0.3] * TEXT_IN_DIM, "stance_label": "neutral"},
        {"text_embedding_pooled": [0.4] * TEXT_IN_DIM, "stance_label": "hawkish"},
        {"text_embedding_pooled": [0.5] * TEXT_IN_DIM, "stance_label": "dovish"},
        {"text_embedding_pooled": [0.6] * TEXT_IN_DIM, "stance_label": "neutral"},
    ]
    corpus = tmp_path / "warm.json"
    corpus.write_text(json.dumps(rows))
    checkpoint = tmp_path / "adapter.pt"
    metadata = pretrain_text_adapter(
        corpus_path=corpus,
        output_path=checkpoint,
        adapter_dim=ADAPTER_DIM,
        epochs=1,
        batch_size=2,
        seed=11,
    )
    assert metadata["n_samples"] == len(rows)
