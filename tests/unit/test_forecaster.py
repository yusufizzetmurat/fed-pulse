from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from app.services.forecaster import (  # noqa: E402
    ChunkAttentionPooler,
    ELAPSED_TIME_FEATURE_INDEX,
    FEATURE_SIZE,
    FeatureVector,
    ForecasterModel,
    ModelConfig,
    SENTIMENT_FEATURE_INDEX,
    TimeDecayAttention,
    build_feature_vectors,
    build_last5_sequence,
    forecast_quantitative_series,
    get_model_artifact_metadata,
    inspect_training_data_sources,
    parse_horizon_steps,
    train_model,
)


def _sample_vectors(n: int = 8) -> list[FeatureVector]:
    return [
        FeatureVector(
            date=f"2026-01-{idx + 1:02d}",
            sentiment_score=0.6,
            market_close=5000 + idx * 10,
            market_volatility=0.01 + idx * 0.0002,
        )
        for idx in range(n)
    ]


def test_parse_horizon_steps():
    assert parse_horizon_steps("1d") == 1
    assert parse_horizon_steps("10d") == 10
    assert parse_horizon_steps("invalid") == 3


def test_build_last5_sequence_padding():
    seq = build_last5_sequence(_sample_vectors(2), length=5)
    assert len(seq) == 5
    assert seq[0].date == "2026-01-01"


def test_build_feature_vectors_derives_change_signals():
    vectors = build_feature_vectors(
        [
            {"date": "2026-01-01", "close": 5000.0, "volatility_5d": 0.0100, "sentiment_score": 0.4},
            {"date": "2026-01-02", "close": 5100.0, "volatility_5d": 0.0115, "sentiment_score": 0.5},
        ]
    )
    assert len(vectors) == 2
    assert vectors[0].close_change_pct == 0.0
    assert vectors[0].volatility_change == 0.0
    assert vectors[1].close_change_pct == pytest.approx(0.02)
    assert vectors[1].volatility_change == pytest.approx(0.0015)


def test_feature_vector_text_embedding_defaults_to_none_and_aslist_unchanged():
    vector = FeatureVector(
        date="2026-01-01",
        sentiment_score=0.5,
        market_close=5000.0,
        market_volatility=0.01,
    )
    assert vector.text_embedding is None
    assert len(vector.as_list()) == FEATURE_SIZE


def test_build_feature_vectors_threads_text_embedding_to_each_row():
    embedding = [0.1, 0.2, 0.3]
    vectors = build_feature_vectors(
        [
            {"date": "2026-01-01", "close": 5000.0, "volatility_5d": 0.0100},
            {"date": "2026-01-02", "close": 5100.0, "volatility_5d": 0.0115},
        ],
        sentiment_score=0.6,
        text_embedding=embedding,
    )
    assert all(v.text_embedding == embedding for v in vectors)
    # Per-record override wins over the default kwarg.
    vectors_with_override = build_feature_vectors(
        [
            {"date": "2026-01-01", "close": 5000.0, "volatility_5d": 0.0100, "text_embedding": [9.0]},
            {"date": "2026-01-02", "close": 5100.0, "volatility_5d": 0.0115},
        ],
        text_embedding=embedding,
    )
    assert vectors_with_override[0].text_embedding == [9.0]
    assert vectors_with_override[1].text_embedding == embedding


def test_inspect_training_data_sources_reports_usable_and_insufficient_files(tmp_path):
    usable = {
        "records": [
            {"date": f"2026-01-{idx + 1:02d}", "close": 5000 + idx * 10, "volatility_5d": 0.01 + idx * 0.0001}
            for idx in range(7)
        ]
    }
    insufficient = {
        "records": [
            {"date": "2026-02-01", "close": 5200, "volatility_5d": 0.011},
            {"date": "2026-02-02", "close": 5210, "volatility_5d": 0.0112},
        ]
    }
    (tmp_path / "usable.json").write_text(json.dumps(usable), encoding="utf-8")
    (tmp_path / "insufficient.json").write_text(json.dumps(insufficient), encoding="utf-8")

    sequences, summaries = inspect_training_data_sources(tmp_path)

    assert len(sequences) == 1
    assert len(summaries) == 2
    statuses = {summary.path.name: summary.status for summary in summaries}
    assert statuses["usable.json"] == "usable"
    assert statuses["insufficient.json"] == "insufficient"


def test_forecast_quantitative_series_fast_shape():
    out = forecast_quantitative_series(_sample_vectors(10), forecast_mode="fast", horizon="3d")
    assert "prediction" in out and "series" in out and "model" in out
    assert out["prediction"]["horizon"] == "3d"
    assert out["model"]["runtime_mode"] == "fast"
    assert "hidden_size" in out["model"]
    assert len(out["series"]["forecast_close"]) == 3
    assert len(out["series"]["forecast_close_lower"]) == 3
    assert len(out["series"]["forecast_close_upper"]) == 3
    assert len(out["series"]["forecast_volatility"]) == 3
    assert len(out["series"]["forecast_volatility_lower"]) == 3
    assert len(out["series"]["forecast_volatility_upper"]) == 3
    assert len(out["series"]["timestamps"]) == 10
    assert out["series"]["forecast_confidence_level"] == 0.8
    assert all(
        lower <= point <= upper
        for lower, point, upper in zip(
            out["series"]["forecast_close_lower"],
            out["series"]["forecast_close"],
            out["series"]["forecast_close_upper"],
        )
    )
    assert "volatility_scale" in out["series"]


def test_forecast_quantitative_series_quick_train_shape():
    out = forecast_quantitative_series(_sample_vectors(12), forecast_mode="quick_train", horizon="5d")
    assert out["prediction"]["horizon"] == "5d"
    assert out["model"]["runtime_mode"] == "quick_train"
    assert out["model"]["adaptation_epochs_completed"] is not None
    assert len(out["series"]["forecast_close"]) == 5
    assert len(out["series"]["forecast_volatility"]) == 5
    assert len(out["series"]["forecast_close_lower"]) == 5
    assert len(out["series"]["forecast_close_upper"]) == 5
    assert len(out["series"]["forecast_volatility_lower"]) == 5
    assert len(out["series"]["forecast_volatility_upper"]) == 5


def test_train_model_reports_model_config_and_metrics():
    result = train_model(
        vectors=_sample_vectors(10),
        model_config=ModelConfig(hidden_size=24, num_layers=1, dropout=0.05, head_hidden_size=12),
        epochs=3,
        batch_size=4,
        save_checkpoint=False,
        device="cpu",
    )

    assert result.model.lstm.hidden_size == 24
    assert result.summary.model_config.hidden_size == 24
    assert result.summary.model_config.num_layers == 1
    assert result.summary.metrics is not None
    assert result.summary.metrics.loss >= 0.0
    assert result.summary.metrics.combined_rmse >= 0.0


def test_train_model_checkpoint_contains_training_metadata(tmp_path):
    checkpoint_path = tmp_path / "forecaster.pt"
    result = train_model(
        vectors=_sample_vectors(10),
        model_config=ModelConfig(hidden_size=20, num_layers=2, dropout=0.10, head_hidden_size=10),
        epochs=2,
        batch_size=4,
        save_checkpoint=True,
        checkpoint_path=checkpoint_path,
        device="cpu",
    )

    payload = torch.load(checkpoint_path, map_location="cpu")

    assert checkpoint_path.exists()
    assert payload["model_config"]["hidden_size"] == 20
    assert payload["training_summary"]["model_config"]["num_layers"] == 2
    assert payload["training_summary"]["metrics"]["combined_rmse"] == pytest.approx(
        result.summary.metrics.combined_rmse
    )


def _time_decay_input(batch: int, seq_len: int, elapsed: float) -> torch.Tensor:
    x = torch.randn(batch, seq_len, FEATURE_SIZE)
    x[..., ELAPSED_TIME_FEATURE_INDEX] = elapsed
    return x


def test_time_decay_attention_preserves_shape():
    layer = TimeDecayAttention()
    x = _time_decay_input(batch=2, seq_len=5, elapsed=0.5)
    out = layer(x)
    assert out.shape == x.shape
    assert out.dtype == x.dtype


def test_time_decay_attention_identity_when_elapsed_is_zero():
    layer = TimeDecayAttention()
    x = _time_decay_input(batch=3, seq_len=5, elapsed=0.0)
    out = layer(x)
    assert torch.allclose(out, x, atol=1e-6)


def test_time_decay_attention_dampens_sentiment_for_positive_elapsed():
    layer = TimeDecayAttention(initial_decay_rate=0.5)
    x = torch.ones(2, 5, FEATURE_SIZE)
    x[..., SENTIMENT_FEATURE_INDEX] = 1.0
    x[..., ELAPSED_TIME_FEATURE_INDEX] = torch.tensor([0.0, 0.2, 0.4, 0.6, 0.8])
    out = layer(x)
    sentiment_out = out[..., SENTIMENT_FEATURE_INDEX]
    # elapsed=0 leaves sentiment untouched; later timesteps decay monotonically.
    assert sentiment_out[0, 0].item() == pytest.approx(1.0, abs=1e-6)
    for b in range(sentiment_out.shape[0]):
        diffs = sentiment_out[b, 1:] - sentiment_out[b, :-1]
        assert torch.all(diffs < 0)
    # Non-sentiment columns must be untouched.
    for idx in range(FEATURE_SIZE):
        if idx == SENTIMENT_FEATURE_INDEX:
            continue
        assert torch.allclose(out[..., idx], x[..., idx])


def test_time_decay_attention_dampens_sentiment_for_negative_elapsed():
    layer = TimeDecayAttention(initial_decay_rate=0.5)
    x = torch.ones(2, 5, FEATURE_SIZE)
    x[..., SENTIMENT_FEATURE_INDEX] = 1.0
    x[..., ELAPSED_TIME_FEATURE_INDEX] = torch.tensor([0.0, -0.2, -0.4, -0.6, -0.8])
    out = layer(x)
    sentiment_out = out[..., SENTIMENT_FEATURE_INDEX]
    # Negative (past) elapsed must damp, not amplify — decay is symmetric in time.
    assert sentiment_out[0, 0].item() == pytest.approx(1.0, abs=1e-6)
    assert torch.all(sentiment_out <= 1.0 + 1e-6)
    for b in range(sentiment_out.shape[0]):
        diffs = sentiment_out[b, 1:] - sentiment_out[b, :-1]
        assert torch.all(diffs < 0)


def test_time_decay_attention_gradient_flows_to_raw_lambda():
    layer = TimeDecayAttention()
    x = _time_decay_input(batch=2, seq_len=5, elapsed=0.5)
    out = layer(x)
    out.sum().backward()
    assert layer.raw_lambda.grad is not None
    assert layer.raw_lambda.grad.abs().item() > 0.0


def test_forecaster_model_exposes_time_decay_layer():
    model = ForecasterModel()
    assert isinstance(model.time_decay, TimeDecayAttention)
    x = torch.randn(1, 5, FEATURE_SIZE)
    x[..., ELAPSED_TIME_FEATURE_INDEX] = 0.1
    out = model(x)
    assert out.shape == (1, 2)
    # Volatility head output is non-negative by construction.
    assert out[0, 1].item() >= 0.0


def test_chunk_attention_pooler_unbatched_shape():
    pooler = ChunkAttentionPooler(embedding_size=8)
    embeddings = torch.randn(4, 8)
    elapsed = torch.tensor([0.0, 5.0, 12.0, 30.0])
    pooled, weights, decay = pooler(embeddings, elapsed)
    assert pooled.shape == (8,)
    assert weights.shape == (4,)
    assert decay.shape == (4,)


def test_chunk_attention_pooler_batched_shape():
    pooler = ChunkAttentionPooler(embedding_size=4)
    embeddings = torch.randn(2, 3, 4)
    elapsed = torch.tensor([[0.0, 1.0, 2.0], [0.0, 5.0, 10.0]])
    pooled, weights, decay = pooler(embeddings, elapsed)
    assert pooled.shape == (2, 4)
    assert weights.shape == (2, 3)
    assert decay.shape == (2, 3)


def test_chunk_attention_pooler_weights_sum_to_one():
    pooler = ChunkAttentionPooler(embedding_size=8)
    embeddings = torch.randn(5, 8)
    elapsed = torch.tensor([0.0, 1.0, 7.0, 14.0, 30.0])
    _, weights, _ = pooler(embeddings, elapsed)
    assert weights.sum().item() == pytest.approx(1.0, abs=1e-5)


def test_chunk_attention_pooler_decay_at_zero_is_one():
    pooler = ChunkAttentionPooler(embedding_size=4)
    embeddings = torch.randn(3, 4)
    elapsed = torch.zeros(3)
    _, _, decay = pooler(embeddings, elapsed)
    assert torch.allclose(decay, torch.ones(3))


def test_chunk_attention_pooler_decay_decreases_with_elapsed():
    pooler = ChunkAttentionPooler(embedding_size=4, initial_decay_rate=0.5)
    embeddings = torch.randn(4, 4)
    elapsed = torch.tensor([0.0, 1.0, 5.0, 20.0])
    _, _, decay = pooler(embeddings, elapsed)
    diffs = decay[1:] - decay[:-1]
    assert torch.all(diffs < 0)
    # Decay symmetric in time — magnitude alone matters.
    elapsed_neg = torch.tensor([0.0, -1.0, -5.0, -20.0])
    _, _, decay_neg = pooler(embeddings, elapsed_neg)
    assert torch.allclose(decay, decay_neg, atol=1e-6)


def test_chunk_attention_pooler_gradient_flows_to_lambda_and_projections():
    pooler = ChunkAttentionPooler(embedding_size=6)
    embeddings = torch.randn(4, 6, requires_grad=True)
    elapsed = torch.tensor([0.0, 2.0, 5.0, 10.0])
    pooled, _, _ = pooler(embeddings, elapsed)
    pooled.sum().backward()
    assert pooler.raw_lambda.grad is not None
    assert pooler.raw_lambda.grad.abs().item() > 0.0
    assert pooler.q_proj.weight.grad is not None
    assert pooler.q_proj.weight.grad.abs().sum().item() > 0.0
    assert pooler.v_proj.weight.grad.abs().sum().item() > 0.0


def test_chunk_attention_pooler_rejects_dim_mismatch():
    pooler = ChunkAttentionPooler(embedding_size=8)
    embeddings = torch.randn(3, 4)
    elapsed = torch.zeros(3)
    with pytest.raises(ValueError):
        pooler(embeddings, elapsed)


def test_chunk_attention_pooler_mask_zeros_padding_weights():
    pooler = ChunkAttentionPooler(embedding_size=4)
    embeddings = torch.randn(5, 4)
    elapsed = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])
    mask = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0])
    _, weights, decay = pooler(embeddings, elapsed, mask=mask)
    # Weights at masked positions are 0; valid positions sum to 1.
    assert weights[3].item() == pytest.approx(0.0, abs=1e-6)
    assert weights[4].item() == pytest.approx(0.0, abs=1e-6)
    assert weights[:3].sum().item() == pytest.approx(1.0, abs=1e-5)
    # Decay is also zeroed at masked positions.
    assert decay[3].item() == 0.0
    assert decay[4].item() == 0.0


def test_chunk_attention_pooler_handles_fully_masked_row():
    pooler = ChunkAttentionPooler(embedding_size=4)
    embeddings = torch.zeros(2, 3, 4)
    elapsed = torch.zeros(2, 3)
    mask = torch.tensor([[1.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    pooled, weights, _ = pooler(embeddings, elapsed, mask=mask)
    assert torch.all(torch.isfinite(pooled))
    assert torch.all(torch.isfinite(weights))
    # Fully masked row: all weights zero, pooled is zero.
    assert weights[1].sum().item() == pytest.approx(0.0, abs=1e-6)
    assert torch.allclose(pooled[1], torch.zeros(4))


def test_get_model_artifact_metadata_surfaces_decay_rate_from_model():
    model = ForecasterModel()
    metadata = get_model_artifact_metadata(runtime_mode="fast", model=model)
    assert "decay_rate" in metadata
    assert metadata["decay_rate"] is not None
    assert metadata["decay_rate"] == pytest.approx(
        float(model.time_decay.decay_rate.detach().cpu().item()),
        rel=1e-6,
    )
    assert metadata["chunk_attention"] is None


def test_forecaster_variant_a_off_skips_time_decay():
    model = ForecasterModel(use_time_decay=False)
    x = torch.ones(1, 5, FEATURE_SIZE)
    x[..., ELAPSED_TIME_FEATURE_INDEX] = 1.0
    out = model(x)
    assert out.shape == (1, 2)


def test_forecaster_variant_b_concats_chunk_projection_to_lstm_input():
    model = ForecasterModel(
        use_chunk_attention=True,
        chunk_embedding_size=12,
        chunk_projection_dim=4,
    )
    assert model.lstm.input_size == FEATURE_SIZE + 4
    x = torch.randn(2, 5, FEATURE_SIZE)
    chunks = torch.randn(2, 6, 12)
    elapsed = torch.zeros(2, 6)
    mask = torch.ones(2, 6)
    out = model(x, chunks=chunks, elapsed_days=elapsed, chunk_mask=mask)
    assert out.shape == (2, 2)
    assert out[:, 1].min().item() >= 0.0


def test_forecaster_variant_b_requires_chunks():
    model = ForecasterModel(use_chunk_attention=True, chunk_embedding_size=8, chunk_projection_dim=2)
    x = torch.randn(1, 5, FEATURE_SIZE)
    with pytest.raises(ValueError):
        model(x)


def test_forecaster_attention_diagnostics_returns_weights_when_enabled():
    model = ForecasterModel(use_chunk_attention=True, chunk_embedding_size=8, chunk_projection_dim=2)
    chunks = torch.randn(3, 8)
    elapsed = torch.tensor([0.0, 5.0, 30.0])
    diag = model.attention_diagnostics(chunks, elapsed)
    assert diag is not None
    assert diag["weights"].shape == (3,)
    assert diag["decay_coeffs"].shape == (3,)


def test_forecaster_attention_diagnostics_none_when_disabled():
    model = ForecasterModel()
    diag = model.attention_diagnostics(torch.randn(3, 8), torch.zeros(3))
    assert diag is None


def test_forecaster_variant_b_gradient_reaches_projection_on_first_step():
    # With zero-init chunk_projection, the first backward pass yields zero
    # gradient at pooled (projection weight is zero), so raw_lambda's gradient
    # is zero on step 1. The projection itself still receives gradient
    # (pooled^T · dL/d(proj_output)), which is what unblocks pooler training
    # on subsequent steps.
    model = ForecasterModel(use_chunk_attention=True, chunk_embedding_size=6, chunk_projection_dim=3)
    x = torch.randn(1, 5, FEATURE_SIZE)
    chunks = torch.randn(1, 4, 6)
    elapsed = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    mask = torch.ones(1, 4)
    out = model(x, chunks=chunks, elapsed_days=elapsed, chunk_mask=mask)
    out.sum().backward()
    assert model.chunk_projection.weight.grad is not None
    assert model.chunk_projection.weight.grad.abs().sum().item() > 0.0


def test_forecaster_variant_b_pooler_gradient_resumes_after_projection_departs_from_zero():
    model = ForecasterModel(use_chunk_attention=True, chunk_embedding_size=6, chunk_projection_dim=3)
    # Manually move projection off the zero subspace to simulate one optimizer step.
    with torch.no_grad():
        model.chunk_projection.weight.fill_(0.01)
    x = torch.randn(1, 5, FEATURE_SIZE)
    chunks = torch.randn(1, 4, 6)
    elapsed = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    mask = torch.ones(1, 4)
    out = model(x, chunks=chunks, elapsed_days=elapsed, chunk_mask=mask)
    out.sum().backward()
    assert model.chunk_pooler.raw_lambda.grad is not None
    assert model.chunk_pooler.raw_lambda.grad.abs().item() > 0.0
    assert model.chunk_projection.weight.grad.abs().sum().item() > 0.0


# ---------------------------------------------------------------------------
# Variant C (use_llm_embeddings) tests
# ---------------------------------------------------------------------------


def test_forecaster_variant_c_builds_without_error():
    """ForecasterModel with use_llm_embeddings=True constructs successfully."""
    model = ForecasterModel(
        use_llm_embeddings=True,
        chunk_embedding_size=10,
        chunk_projection_dim=4,
    )
    assert model.use_llm_embeddings is True
    assert model.use_chunk_attention is False
    assert model.chunk_pooler is not None
    assert model.chunk_projection is not None
    assert model.lstm.input_size == FEATURE_SIZE + 4


def test_forecaster_variant_c_forward_returns_correct_shape():
    """Variant C forward pass returns (batch, 2) and volatility >= 0."""
    model = ForecasterModel(
        use_llm_embeddings=True,
        chunk_embedding_size=12,
        chunk_projection_dim=4,
    )
    x = torch.randn(2, 5, FEATURE_SIZE)
    # LLM path: one doc per slot, same interface as chunk path.
    chunks = torch.randn(2, 6, 12)
    elapsed = torch.zeros(2, 6)
    mask = torch.ones(2, 6)
    out = model(x, chunks=chunks, elapsed_days=elapsed, chunk_mask=mask)
    assert out.shape == (2, 2)
    assert out[:, 1].min().item() >= 0.0


def test_forecaster_variant_c_requires_chunks():
    """Variant C raises ValueError when chunks/elapsed_days are not supplied."""
    model = ForecasterModel(use_llm_embeddings=True, chunk_embedding_size=8, chunk_projection_dim=2)
    x = torch.randn(1, 5, FEATURE_SIZE)
    with pytest.raises(ValueError):
        model(x)


def test_forecaster_variant_b_and_c_mutually_exclusive():
    """Setting both use_chunk_attention and use_llm_embeddings raises ValueError."""
    with pytest.raises(ValueError):
        ForecasterModel(use_chunk_attention=True, use_llm_embeddings=True)


def test_forecaster_variant_c_attention_diagnostics():
    """attention_diagnostics returns weights when Variant C is active."""
    model = ForecasterModel(use_llm_embeddings=True, chunk_embedding_size=8, chunk_projection_dim=2)
    chunks = torch.randn(3, 8)
    elapsed = torch.tensor([0.0, 5.0, 30.0])
    diag = model.attention_diagnostics(chunks, elapsed)
    assert diag is not None
    assert diag["weights"].shape == (3,)
    assert diag["decay_coeffs"].shape == (3,)


def test_forecaster_variant_c_gradient_reaches_projection():
    """Projection weight receives gradient after one backward pass (Variant C)."""
    model = ForecasterModel(use_llm_embeddings=True, chunk_embedding_size=6, chunk_projection_dim=3)
    with torch.no_grad():
        model.chunk_projection.weight.fill_(0.01)
    x = torch.randn(1, 5, FEATURE_SIZE)
    chunks = torch.randn(1, 4, 6)
    elapsed = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    mask = torch.ones(1, 4)
    out = model(x, chunks=chunks, elapsed_days=elapsed, chunk_mask=mask)
    out.sum().backward()
    assert model.chunk_projection.weight.grad is not None
    assert model.chunk_projection.weight.grad.abs().sum().item() > 0.0


def test_forecaster_variant_b_still_works_as_regression():
    """Existing Variant B path remains unbroken after Variant C additions."""
    model = ForecasterModel(
        use_chunk_attention=True,
        chunk_embedding_size=8,
        chunk_projection_dim=4,
    )
    x = torch.randn(2, 5, FEATURE_SIZE)
    chunks = torch.randn(2, 5, 8)
    elapsed = torch.zeros(2, 5)
    mask = torch.ones(2, 5)
    out = model(x, chunks=chunks, elapsed_days=elapsed, chunk_mask=mask)
    assert out.shape == (2, 2)
    assert out[:, 1].min().item() >= 0.0


# ---------------------------------------------------------------------------
# model_type parameter tests (Plan 7 / Task 1)
# ---------------------------------------------------------------------------


def test_forecaster_model_default_model_type_is_lstm() -> None:
    """Existing call sites must not break."""
    model = ForecasterModel(input_size=6, hidden_size=32)
    assert model.model_type == "lstm"


def test_forecaster_model_gru_variant_forward_returns_expected_shape() -> None:
    import torch

    model = ForecasterModel(input_size=6, hidden_size=32, model_type="gru")
    x = torch.zeros(2, 5, 6)
    out = model(x)
    lstm_model = ForecasterModel(input_size=6, hidden_size=32, model_type="lstm")
    lstm_out = lstm_model(x)
    if isinstance(out, tuple):
        for a, b in zip(out, lstm_out):
            assert a.shape == b.shape
    else:
        assert out.shape == lstm_out.shape


def test_forecaster_model_unknown_model_type_raises() -> None:
    with pytest.raises(ValueError):
        ForecasterModel(input_size=6, hidden_size=32, model_type="bogus")
