from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")
SCRIPT_DIR = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))


class _FakeTicker:
    def __init__(self, symbol: str) -> None:
        self.symbol = symbol

    def history(self, *, start: str, end: str, auto_adjust: bool):
        index = pd.date_range(start=start, end=end, freq="B")[:20]
        return pd.DataFrame({"Close": [100.0 + i for i in range(len(index))]}, index=index)


class _FakeYFinance:
    Ticker = _FakeTicker


def test_snapshot_writes_parquet_and_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import snapshot_market_data as script

    monkeypatch.setattr(script, "_fetch_close_series", lambda symbol, start, end: _FakeTicker(symbol).history(start=start, end=end, auto_adjust=True)["Close"])

    snapshot_dir = tmp_path / "market"
    lock_path = snapshot_dir / "SOURCES.lock"
    results = script.snapshot(
        symbols=["^GSPC"],
        start="2024-01-01",
        end="2024-02-01",
        snapshot_dir=snapshot_dir,
        lock_path=lock_path,
    )
    assert len(results) == 1
    entry = results[0]
    assert entry["symbol"] == "^GSPC"
    assert entry["rows"] == 20
    assert len(entry["sha256"]) == 64
    assert entry["source"] == "yfinance"
    assert entry["source_symbol"] == "^GSPC"

    parquet_path = snapshot_dir / "GSPC.parquet"
    assert parquet_path.exists()
    frame = pd.read_parquet(parquet_path)
    assert list(frame.columns) == ["symbol", "date", "close"]
    assert len(frame) == 20

    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    assert "^GSPC" in lock["entries"]
    assert lock["entries"]["^GSPC"]["sha256"] == entry["sha256"]
    assert lock["entries"]["^GSPC"]["rows"] == 20
    assert lock["entries"]["^GSPC"]["source"] == "yfinance"
    assert lock["entries"]["^GSPC"]["source_symbol"] == "^GSPC"


def test_route_source_prefers_fred_for_macro_tickers() -> None:
    import snapshot_market_data as script

    assert script._route_source("^VIX") == ("fred", "VIXCLS")
    assert script._route_source("^TNX") == ("fred", "DGS10")
    assert script._route_source("GC=F") == ("fred", "GOLDAMGBD228NLBM")
    assert script._route_source("^IXIC") == ("fred", "NASDAQCOM")


def test_route_source_falls_back_to_yfinance_for_unmapped_tickers() -> None:
    import snapshot_market_data as script

    assert script._route_source("^GSPC") == ("yfinance", "^GSPC")
    assert script._route_source("^DJI") == ("yfinance", "^DJI")
    assert script._route_source("AAPL") == ("yfinance", "AAPL")


def test_fetch_from_fred_parses_csv(monkeypatch: pytest.MonkeyPatch) -> None:
    import snapshot_market_data as script

    csv_text = "observation_date,VIXCLS\n2024-01-02,13.20\n2024-01-03,14.04\n2024-01-04,.\n"

    class _FakeResponse:
        text = csv_text

        def raise_for_status(self) -> None:
            return None

    def _fake_get(url, params=None, timeout=None):
        assert params["id"] == "VIXCLS"
        return _FakeResponse()

    requests_mod = pytest.importorskip("requests")
    monkeypatch.setattr(requests_mod, "get", _fake_get)
    series = script._fetch_from_fred("VIXCLS", "2024-01-01", "2024-01-10")
    assert len(series) == 2
    assert float(series.iloc[0]) == 13.20
    assert float(series.iloc[-1]) == 14.04
