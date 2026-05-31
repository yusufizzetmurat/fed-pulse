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


def test_market_data_response_strict_rejects_numpy_float() -> None:
    """#99 (first half): MarketDataResponse must reject numpy.float64.

    Pre-#99 the response model was frozen but not strict; numpy values
    silently coerced to Python floats which masked service-layer leaks
    that downstream consumers had to defend against. After this PR
    every numeric field on the response MUST receive a built-in float
    or int; if a service forgets the ``float(...)`` cast on a numpy
    value the response build fails loud.
    """

    pytest.importorskip("numpy")
    import numpy as np

    from app.schemas import MarketDataResponse

    ok = MarketDataResponse(
        symbol="^GSPC",
        requested_date="2024-09-18",
        date_used="2024-09-18",
        lookback_days=14,
        close=5400.0,
        volatility_5d=0.012,
    )
    assert ok.close == 5400.0

    with pytest.raises(ValidationError):
        MarketDataResponse(
            symbol="^GSPC",
            requested_date="2024-09-18",
            date_used="2024-09-18",
            lookback_days=14,
            close=np.float64(5400.0),
            volatility_5d=0.012,
        )


def test_prediction_response_strict_rejects_numpy_float() -> None:
    """#99 (first half): PredictionResponse must reject numpy.float64."""

    pytest.importorskip("numpy")
    import numpy as np

    from app.schemas import PredictionResponse

    ok = PredictionResponse(close=5400.0, volatility=0.012, horizon="3d")
    assert ok.close == 5400.0

    with pytest.raises(ValidationError):
        PredictionResponse(
            close=np.float64(5400.0), volatility=0.012, horizon="3d"
        )
