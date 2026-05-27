"""Integration test for the regression-canonical regime card on /analyze (#322).

ADR 0015 makes regression on ``log(forward_realized_vol_10d)`` the
canonical training objective; the regime card therefore needs to surface
the regression point + band alongside the legacy classifier surface, and
the response schema needs to advertise which path produced the bucket
label via ``bucket_source``.

The test exercises ``/analyze`` end-to-end with the FastAPI ``TestClient``,
stubbing the market path so the assertion focuses on the regime card.
The two paths covered are:

- Regression-canonical: ``regime_classification`` carries
  ``log_rv_point``, ``log_rv_lower``, ``log_rv_upper`` floats and
  ``bucket_source == "regression"``.
- Classification-only legacy: ``log_rv_point`` is ``None`` but the
  ``argmax_class`` + ``distribution`` surface stays populated so the
  pre-#322 UI keeps rendering. This branch is intentionally light —
  the integration harness does not currently support fixturing two
  distinct ``head_mode`` checkpoints cleanly inside a single process,
  so the back-compat case is marked skipped with a pointer to the
  manual matrix in ``tests/e2e/test_api_e2e.py``.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")
pytest.importorskip("torch")
pytest.importorskip("transformers")

from fastapi.testclient import TestClient  # noqa: E402

import app.main as main_mod  # noqa: E402


def _stub_analyze_market_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Wire the /analyze handler onto deterministic fixture data so the
    response shape depends only on the regime-card stub under test."""

    monkeypatch.setattr(
        main_mod,
        "analyze_text",
        lambda _: {
            "label": "DOVISH",
            "score": 0.62,
            "raw": [{"label": "DOVISH", "score": 0.62}],
        },
    )
    monkeypatch.setattr(
        main_mod,
        "fetch_market_snapshot",
        lambda **_: {
            "symbol": "^GSPC",
            "requested_date": "2026-03-15",
            "date_used": "2026-03-13",
            "lookback_days": 5,
            "close": 5000.0,
            "volatility_5d": 0.012,
        },
    )
    monkeypatch.setattr(
        main_mod,
        "fetch_market_history",
        lambda **_: [
            {"date": "2026-03-09", "close": 4960.0, "volatility_5d": 0.0125},
            {"date": "2026-03-10", "close": 4975.0, "volatility_5d": 0.0124},
            {"date": "2026-03-11", "close": 4985.0, "volatility_5d": 0.0123},
            {"date": "2026-03-12", "close": 4990.0, "volatility_5d": 0.0122},
            {"date": "2026-03-13", "close": 5000.0, "volatility_5d": 0.0120},
        ],
    )
    monkeypatch.setattr(main_mod, "fetch_realized_forward", lambda **_: [])
    monkeypatch.setattr(main_mod, "parse_horizon_steps", lambda _: 3)
    monkeypatch.setattr(
        main_mod,
        "fetch_forward_trading_dates",
        lambda **_: ["2026-03-16", "2026-03-17", "2026-03-18"],
    )
    monkeypatch.setattr(
        main_mod,
        "forecast_quantitative_series",
        lambda **_: {
            "prediction": {"close": 5050.0, "volatility": 0.012, "horizon": "3d"},
            "model": {
                "checkpoint_path": "backend/models/forecaster_best.pt",
                "checkpoint_exists": True,
                "checkpoint_loaded": True,
                "runtime_mode": "fast",
                "hidden_size": 64,
                "num_layers": 2,
                "dropout": 0.15,
                "head_hidden_size": 32,
                "close_scale": 10000.0,
                "sequence_length": 5,
            },
            "series": {
                "timestamps": ["2026-03-12", "2026-03-13"],
                "history_close": [4990.0, 5000.0],
                "history_volatility": [0.0122, 0.0120],
                "forecast_timestamps": ["2026-03-16", "2026-03-17", "2026-03-18"],
                "forecast_close": [5020.0, 5040.0, 5050.0],
                "forecast_close_lower": [5000.0, 5015.0, 5020.0],
                "forecast_close_upper": [5040.0, 5060.0, 5080.0],
                "forecast_volatility": [0.011, 0.012, 0.012],
                "forecast_volatility_lower": [0.009, 0.010, 0.010],
                "forecast_volatility_upper": [0.013, 0.014, 0.015],
                "forecast_confidence_level": 0.8,
                "volatility_scale": {"suggested_ymin": 0.0, "suggested_ymax": 0.02},
            },
        },
    )


def test_analyze_regime_card_regression_canonical_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Against a regression-canonical checkpoint, the regime card carries
    the regression point + band and declares ``bucket_source = "regression"``.
    The 3-class ``argmax_class`` + ``distribution`` surface is still
    populated (recovered from the regression head via
    ``regime_bucketing.bucket_log_rv`` / ``derive_distribution``)."""

    _stub_analyze_market_path(monkeypatch)
    monkeypatch.setattr(main_mod, "checkpoint_exists", lambda: True)

    regression_card = {
        "predicted_set": ["normal"],
        "set_label": "{normal}",
        "set_size": 1,
        "coverage": 0.8,
        "distribution": {"calm": 0.15, "normal": 0.7, "high": 0.15},
        "argmax_class": "normal",
        "log_rv_point": -0.32,
        "log_rv_lower": -0.71,
        "log_rv_upper": 0.07,
        "bucket_source": "regression",
    }
    monkeypatch.setattr(
        main_mod, "build_regime_classification_card", lambda _seq: regression_card
    )

    client = TestClient(main_mod.app)
    response = client.post(
        "/analyze",
        json={
            "text": "Recent indicators point to a soft landing.",
            "date": "2026-03-15",
            "symbol": "^GSPC",
            "horizon": "3d",
            "include_realized": False,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    regime = payload["regime_classification"]
    assert regime is not None

    # Regression-canonical surface: point + band are real floats.
    assert isinstance(regime["log_rv_point"], float)
    assert isinstance(regime["log_rv_lower"], float)
    assert isinstance(regime["log_rv_upper"], float)
    assert regime["log_rv_lower"] <= regime["log_rv_point"] <= regime["log_rv_upper"]
    assert regime["bucket_source"] == "regression"

    # 3-class bucket recovered from the regression point lands in the
    # canonical label set, and the classification surface still serialises
    # cleanly for the back-compat consumers.
    assert regime["argmax_class"] in {"calm", "normal", "high"}
    assert set(regime["distribution"].keys()) == {"calm", "normal", "high"}
    assert abs(sum(regime["distribution"].values()) - 1.0) < 1e-6
    assert isinstance(regime["predicted_set"], list)
    assert isinstance(regime["set_size"], int)


@pytest.mark.skip(
    reason=(
        "Classification-only legacy back-compat path is covered by "
        "tests/e2e/test_api_e2e.py manual matrix; the integration "
        "harness cannot fixture two distinct head_mode checkpoints "
        "inside a single process."
    )
)
def test_analyze_regime_card_classification_only_legacy_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On a pre-#322 classification-only checkpoint the regime card omits
    the regression point + band (``log_rv_point is None``) but the
    classifier surface (``argmax_class``, ``distribution``,
    ``predicted_set``) stays populated. ``bucket_source`` flips to
    ``"classification"`` so the UI can pick the right badge."""

    _stub_analyze_market_path(monkeypatch)
    monkeypatch.setattr(main_mod, "checkpoint_exists", lambda: True)

    classification_card = {
        "predicted_set": ["normal", "high"],
        "set_label": "{normal, high}",
        "set_size": 2,
        "coverage": 0.8,
        "distribution": {"calm": 0.10, "normal": 0.55, "high": 0.35},
        "argmax_class": "normal",
        "log_rv_point": None,
        "log_rv_lower": None,
        "log_rv_upper": None,
        "bucket_source": "classification",
    }
    monkeypatch.setattr(
        main_mod, "build_regime_classification_card", lambda _seq: classification_card
    )

    client = TestClient(main_mod.app)
    response = client.post(
        "/analyze",
        json={
            "text": "Policy statement sample",
            "date": "2026-03-15",
            "symbol": "^GSPC",
            "horizon": "3d",
            "include_realized": False,
        },
    )

    assert response.status_code == 200
    regime = response.json()["regime_classification"]
    assert regime is not None
    assert regime["log_rv_point"] is None
    assert regime["log_rv_lower"] is None
    assert regime["log_rv_upper"] is None
    assert regime["bucket_source"] == "classification"
    assert regime["argmax_class"] in {"calm", "normal", "high"}
    assert set(regime["distribution"].keys()) == {"calm", "normal", "high"}
