from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("yfinance")
from fastapi.testclient import TestClient  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MINUTES_PATH_CANDIDATES = [ROOT / "data" / "fomc_minutes.json", Path("/data/fomc_minutes.json")]

backend_candidates = [ROOT / "backend", ROOT]
for candidate in backend_candidates:
    if (candidate / "app" / "main.py").exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))
        break

from app.main import app  # noqa: E402


def _load_minutes() -> list[dict]:
    for path in MINUTES_PATH_CANDIDATES:
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, list):
                raise RuntimeError(f"Expected list in {path}, got {type(payload).__name__}")
            return payload
    raise FileNotFoundError(
        "Could not find fomc_minutes.json in ./data or /data. Run scraper first."
    )


def _pick_dates(minutes: list[dict], limit: int = 5) -> list[str]:
    dates = sorted({str(item.get("date", "")).strip() for item in minutes if item.get("date")}, reverse=True)
    if len(dates) < limit:
        raise RuntimeError(f"Need at least {limit} dated minute records, found {len(dates)}")
    return dates[:limit]


def _assert_valid_result(response_data: dict, for_date: str) -> None:
    if "sentiment" not in response_data:
        raise AssertionError(f"Missing 'sentiment' key for date {for_date}")
    if "market" not in response_data:
        raise AssertionError(f"Missing 'market' key for date {for_date}")

    market = response_data["market"]
    volatility = market.get("volatility_5d")
    if not isinstance(volatility, (float, int)):
        raise AssertionError(f"volatility_5d must be numeric for date {for_date}, got {type(volatility)}")
    if float(volatility) <= 0:
        raise AssertionError(f"volatility_5d must be positive for date {for_date}, got {volatility}")


def main() -> None:
    minutes = _load_minutes()
    dates = _pick_dates(minutes, limit=5)
    sample_text = (
        "The Committee seeks to achieve maximum employment and inflation at the rate of 2 percent over "
        "the longer run."
    )

    client = TestClient(app)

    print(f"Running integration checks for dates: {dates}")
    for idx, document_date in enumerate(dates, start=1):
        response = client.post(
            "/analyze",
            json={
                "text": sample_text,
                "date": document_date,
                "symbol": "^GSPC",
                "horizon": "3d",
            },
        )
        if response.status_code == 500:
            raise AssertionError(f"/analyze returned 500 for date {document_date}: {response.text}")
        if response.status_code != 200:
            raise AssertionError(f"/analyze returned {response.status_code} for {document_date}: {response.text}")
        _assert_valid_result(response.json(), document_date)
        print(f"[{idx}/5] OK -> {document_date}")

    holiday_date = f"{dates[0][:4]}-01-01"
    holiday_response = client.post(
        "/analyze",
        json={
            "text": sample_text,
            "date": holiday_date,
            "symbol": "^GSPC",
            "horizon": "3d",
        },
    )
    if holiday_response.status_code == 500:
        raise AssertionError(
            f"/analyze returned 500 for holiday-like date {holiday_date}: {holiday_response.text}"
        )
    if holiday_response.status_code == 200:
        _assert_valid_result(holiday_response.json(), holiday_date)

    print(f"Holiday resilience OK -> {holiday_date} returned status {holiday_response.status_code}")
    print("Integration checks passed.")


if __name__ == "__main__":
    main()
