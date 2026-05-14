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
        "forecast_mode": "fast",
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
