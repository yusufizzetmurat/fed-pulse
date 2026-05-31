from __future__ import annotations

import pytest

pytest.importorskip("pydantic")

from pydantic import ValidationError

from app.schemas import AnalyzeRequest, SentimentResponse


def _valid_request_payload() -> dict:
    return {
        "text": "FOMC body",
        "date": "2024-09-18",
        "symbol": "^GSPC",
        "horizon": "3d",
        "include_realized": False,
    }


def test_unknown_field_is_rejected():
    payload = _valid_request_payload()
    payload["extra_field"] = "leak"
    with pytest.raises(ValidationError):
        AnalyzeRequest.model_validate(payload)


def test_wrong_typed_field_is_rejected_in_strict_mode():
    payload = _valid_request_payload()
    payload["include_realized"] = "true"  # would coerce in non-strict mode
    with pytest.raises(ValidationError):
        AnalyzeRequest.model_validate(payload)


def test_valid_request_round_trips():
    request = AnalyzeRequest.model_validate(_valid_request_payload())
    assert request.symbol == "^GSPC"


def test_response_model_is_frozen():
    response = SentimentResponse(label="HAWKISH", score=0.8, raw=[])
    with pytest.raises(ValidationError):
        response.label = "DOVISH"


def test_market_data_response_strict_rejects_float_for_int() -> None:
    """#99: strict_int refuses a bare ``float`` (incl. numpy.float64
    via subclass) in an ``int`` field."""

    from app.schemas import MarketDataResponse

    with pytest.raises(ValidationError):
        MarketDataResponse(
            symbol="^GSPC",
            requested_date="2024-09-18",
            date_used="2024-09-18",
            lookback_days=14.0,  # float, not int
            close=5400.0,
            volatility_5d=0.012,
        )


def test_prediction_response_strict_rejects_string_for_float() -> None:
    """#99: strict_float refuses string coercion into a float field."""

    from app.schemas import PredictionResponse

    ok = PredictionResponse(close=5400.0, volatility=0.012, horizon="3d")
    assert ok.close == 5400.0

    with pytest.raises(ValidationError):
        PredictionResponse(close="5400.0", volatility=0.012, horizon="3d")


def test_prediction_response_strict_rejects_bool_for_float() -> None:
    """#99: strict_float refuses ``bool`` even though ``bool`` is an
    ``int`` subclass — a True/False leak into close/volatility now
    fails loud at construction."""

    from app.schemas import PredictionResponse

    with pytest.raises(ValidationError):
        PredictionResponse(close=True, volatility=0.012, horizon="3d")
