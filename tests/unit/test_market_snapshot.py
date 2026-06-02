from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")


def _write_fixture_parquet(tmp_path: Path, symbol: str = "TEST", n: int = 60) -> Path:
    snapshot_dir = tmp_path / "market"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    base = pd.Timestamp("2024-01-02")
    rows = [
        {
            "symbol": symbol,
            "date": (base + pd.Timedelta(days=i)).date().isoformat(),
            "close": 4000.0 + i * 1.5,
        }
        for i in range(n)
    ]
    frame = pd.DataFrame(rows)
    out_path = snapshot_dir / f"{symbol}.parquet"
    frame.to_parquet(out_path, index=False)
    lock = {
        "format_version": 1,
        "entries": {
            symbol: {
                "parquet_path": str(out_path.relative_to(tmp_path)),
                "sha256": "deadbeef" * 8,
                "rows": n,
                "start": rows[0]["date"],
                "end": rows[-1]["date"],
                "fetched_at": "2026-05-12T00:00:00+00:00",
                "source": "yfinance",
            }
        },
    }
    (snapshot_dir / "SOURCES.lock").write_text(json.dumps(lock), encoding="utf-8")
    return snapshot_dir


@pytest.fixture
def snapshot_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    snapshot_dir = _write_fixture_parquet(tmp_path)
    monkeypatch.setenv("FED_PULSE_MARKET_SOURCE", "snapshot")
    monkeypatch.setenv("FED_PULSE_MARKET_SNAPSHOT_DIR", str(snapshot_dir))
    from app.services import market_data

    market_data._load_snapshot_series.cache_clear()
    market_data._download_close_series_in_window.cache_clear()
    return snapshot_dir


def test_snapshot_reader_loads_close_series(snapshot_env: Path) -> None:
    from app.services.market_data import _load_snapshot_series

    series = _load_snapshot_series("TEST")
    assert len(series) == 60
    assert series.iloc[0] == 4000.0
    assert series.iloc[-1] == pytest.approx(4088.5)


def test_snapshot_reader_window_filter(snapshot_env: Path) -> None:
    from app.services.market_data import _download_close_series_in_window

    start = date(2024, 1, 10)
    end = date(2024, 1, 20)
    series = _download_close_series_in_window("TEST", start, end)
    assert series.index.min().date() >= start
    assert series.index.max().date() < end


def test_snapshot_reader_raises_when_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("FED_PULSE_MARKET_SOURCE", "snapshot")
    monkeypatch.setenv("FED_PULSE_MARKET_SNAPSHOT_DIR", str(tmp_path))
    from app.services import market_data

    market_data._load_snapshot_series.cache_clear()
    with pytest.raises(FileNotFoundError, match="snapshot"):
        market_data._load_snapshot_series("DOES_NOT_EXIST")


def test_default_market_source_is_live(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FED_PULSE_MARKET_SOURCE", raising=False)
    from app.services import market_data

    assert market_data._market_source() == "live"


def test_safe_symbol_strips_special_chars() -> None:
    from app.services.market_data import _safe_symbol

    assert _safe_symbol("^GSPC") == "GSPC"
    assert _safe_symbol("DX-Y.NYB") == "DX-Y.NYB"
    assert _safe_symbol("GC=F") == "GC_F"
