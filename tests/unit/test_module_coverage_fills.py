"""Targeted unit tests filling the highest-leverage coverage gaps on the
four modules the 2026-05-19 audit flagged at 84-85% line coverage:
``backend/app/training/loaders.py``, ``backend/app/services/text_encoder.py``,
``backend/app/models/lstm.py``, ``backend/app/services/forecaster.py``.

The tests favour small pure helpers and error paths because that is what
the coverage gap actually is — large training-loop branches sit at
~95% via the integration tests, but the input-coercion + dispatch
helpers underneath have been collateral damage from the focus on the
training surface.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.training.loaders import _parse_prior_bars, _stance_to_sentiment


# ---------------------------------------------------------------------------
# loaders.py: stance encoding + prior-bars JSON parsing
# ---------------------------------------------------------------------------


def test_stance_to_sentiment_known_labels() -> None:
    assert _stance_to_sentiment("hawkish") == pytest.approx(1.0)
    assert _stance_to_sentiment("dovish") == pytest.approx(-1.0)
    assert _stance_to_sentiment("neutral") == pytest.approx(0.0)


def test_stance_to_sentiment_normalises_case_and_whitespace() -> None:
    assert _stance_to_sentiment("  Hawkish  ") == pytest.approx(1.0)
    assert _stance_to_sentiment("DOVISH") == pytest.approx(-1.0)


def test_stance_to_sentiment_returns_zero_on_missing_or_nan() -> None:
    assert _stance_to_sentiment(None) == 0.0
    assert _stance_to_sentiment(float("nan")) == 0.0
    # Unknown label collapses to neutral; integers / floats also fall through.
    assert _stance_to_sentiment("uncertain") == 0.0
    assert _stance_to_sentiment(42) == 0.0


def test_parse_prior_bars_handles_none_and_empty() -> None:
    assert _parse_prior_bars(None) == []
    assert _parse_prior_bars("") == []
    assert _parse_prior_bars("   ") == []


def test_parse_prior_bars_decodes_json_and_passes_list_through() -> None:
    json_payload = '[{"date": "2024-01-01", "close": 100.0}, {"date": "2024-01-02", "close": 101.0}]'
    bars = _parse_prior_bars(json_payload)
    assert len(bars) == 2
    assert bars[0]["close"] == 100.0
    # Already-decoded lists round-trip unchanged.
    raw = [{"date": "2024-01-01", "close": 100.0}]
    assert _parse_prior_bars(raw) == raw


def test_parse_prior_bars_drops_non_dict_entries_and_non_list_payloads() -> None:
    # Strings inside the list are not bar dicts; they get dropped.
    bars = _parse_prior_bars(["not-a-dict", {"close": 1.0}, 42])
    assert bars == [{"close": 1.0}]
    # Top-level non-list payloads collapse to empty.
    assert _parse_prior_bars('{"close": 100.0}') == []


# ---------------------------------------------------------------------------
# services/text_encoder.py: score normalisation + OOD manifest lookup
# ---------------------------------------------------------------------------


def test_normalize_scores_unwraps_nested_list_and_skips_non_dicts() -> None:
    from app.services.text_encoder import _normalize_scores

    # HF pipeline often emits ``[[{...}, {...}]]`` for single inputs.
    nested = [[{"label": "POSITIVE", "score": 0.9}, {"label": "NEGATIVE", "score": 0.1}]]
    normalised = _normalize_scores(nested)
    assert normalised == [
        {"label": "POSITIVE", "score": 0.9},
        {"label": "NEGATIVE", "score": 0.1},
    ]

    # Non-dict entries inside the list are dropped without crashing.
    mixed = [{"label": "OK", "score": 0.5}, "garbage", None]
    normalised = _normalize_scores(mixed)
    assert normalised == [{"label": "OK", "score": 0.5}]

    # Non-list inputs collapse to empty.
    assert _normalize_scores("not a list") == []
    assert _normalize_scores(None) == []


def test_resolve_ood_manifest_path_returns_none_for_hub_model_id(monkeypatch) -> None:
    """When ``MODEL_ID`` is an HF hub identifier (no local directory)
    the helper short-circuits to ``None`` so the response code does
    not try to surface OOD fields it cannot compute."""

    from app.services import text_encoder

    monkeypatch.setattr(text_encoder, "MODEL_ID", "ProsusAI/finbert")
    assert text_encoder.resolve_ood_manifest_path() is None


def test_resolve_ood_manifest_path_returns_none_when_manifest_missing(
    monkeypatch, tmp_path: Path
) -> None:
    """Local checkpoint directory exists but no OOD manifest -> ``None``."""

    from app.services import text_encoder

    monkeypatch.setattr(text_encoder, "MODEL_ID", str(tmp_path))
    assert text_encoder.resolve_ood_manifest_path() is None


# ---------------------------------------------------------------------------
# models/lstm.py: recurrent-core dispatch + attention diagnostics
# ---------------------------------------------------------------------------


def test_build_recurrent_core_rejects_unknown_model_type() -> None:
    from app.models.lstm import ForecasterModel

    with pytest.raises(ValueError, match="Unknown model_type"):
        ForecasterModel._build_recurrent_core(
            model_type="not-a-real-arch",
            input_size=6,
            hidden_size=8,
            num_layers=1,
            dropout=0.0,
        )


def test_attention_diagnostics_returns_none_when_pooler_off() -> None:
    """The diagnostics helper short-circuits when neither chunk-attention
    nor LLM-embeddings pooling is active. Default ForecasterModel has
    both off, so the call returns ``None`` regardless of the input."""

    import torch
    from app.models.lstm import ForecasterModel

    model = ForecasterModel(input_size=6, hidden_size=8, num_layers=1, dropout=0.0)
    chunks = torch.zeros((1, 4, 8))
    elapsed = torch.zeros((1, 4))
    assert model.attention_diagnostics(chunks, elapsed) is None


# ---------------------------------------------------------------------------
# services/forecaster.py: horizon parsing, sample std, conformal manifest
# ---------------------------------------------------------------------------


def test_parse_horizon_steps_handles_numeric_d_suffix() -> None:
    from app.services.forecaster import _parse_horizon_steps

    assert _parse_horizon_steps("1d") == 1
    assert _parse_horizon_steps("10d") == 10
    # Clamp to >= 1 so a "0d" request never produces a zero-step forecast.
    assert _parse_horizon_steps("0d") == 1


def test_parse_horizon_steps_falls_back_to_three_on_bad_input() -> None:
    from app.services.forecaster import _parse_horizon_steps

    # The default three-step horizon is the documented fallback.
    assert _parse_horizon_steps("abc") == 3
    assert _parse_horizon_steps("3w") == 3
    assert _parse_horizon_steps("") == 3


def test_sample_std_returns_zero_for_short_lists() -> None:
    from app.services.forecaster import _sample_std

    assert _sample_std([]) == 0.0
    assert _sample_std([1.5]) == 0.0


def test_sample_std_computes_n_minus_one_variance() -> None:
    from app.services.forecaster import _sample_std

    # Mean = 2.0, variance over (n-1) = ((1-2)^2 + (2-2)^2 + (3-2)^2) / 2 = 1.0
    assert _sample_std([1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_conformal_manifest_for_returns_none_without_path() -> None:
    from app.services.forecaster import _conformal_manifest_for

    assert _conformal_manifest_for(None) is None


def test_conformal_manifest_for_returns_none_when_sidecar_missing(tmp_path: Path) -> None:
    from app.services.forecaster import _conformal_manifest_for

    # Checkpoint path exists but no ``.conformal.json`` sidecar next to it.
    checkpoint = tmp_path / "model_best.pt"
    checkpoint.write_bytes(b"")
    assert _conformal_manifest_for(checkpoint) is None
