"""Unit tests for the cross-bank snapshot service + /cross-bank/snapshot route.

Both layers are exercised against a stub classifier + stub market lookup so
CI does not need to load the multi-axis checkpoint or hit yfinance. The
production wiring (real checkpoint + real ``fetch_market_snapshot``) is
covered indirectly by the existing /analyze contract suite.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services import cross_bank_snapshot as svc


def _write_registry(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def _stub_score(text: str) -> dict[str, Any]:
    """Deterministic classifier stub.

    Sentence containing 'hike' -> hawkish-leaning, 'cut' -> dovish, else
    neutral. Returns the three-block multi-axis shape so the aggregator
    code paths are exercised end to end.
    """

    text_l = (text or "").lower()
    if "hike" in text_l:
        stance = {"hawkish": 0.7, "dovish": 0.1, "neutral": 0.2}
    elif "cut" in text_l:
        stance = {"hawkish": 0.1, "dovish": 0.75, "neutral": 0.15}
    else:
        stance = {"hawkish": 0.25, "dovish": 0.25, "neutral": 0.5}
    return {
        "stance": {
            "label": max(stance, key=lambda k: stance[k]),
            "confidence": max(stance.values()),
            "distribution": stance,
        },
        "certainty": {
            "label": "certain",
            "confidence": 0.8,
            "distribution": {"certain": 0.8, "uncertain": 0.1, "neutral": 0.1},
        },
        "time": {
            "label": "forward looking",
            "confidence": 0.65,
            "distribution": {"forward looking": 0.65, "not forward looking": 0.35},
        },
    }


def _stub_market(symbol: str) -> dict[str, Any]:
    return {
        "label": "calm" if symbol == "^GSPC" else "normal",
        "confidence": 0.78,
        "close": 100.0,
        "vol_5d_annualised": 0.08 if symbol == "^GSPC" else 0.17,
        "as_of": "2026-06-03",
        "status": "ok",
    }


@pytest.fixture(autouse=True)
def _reset_cache() -> None:
    svc.reset_cache()
    yield
    svc.reset_cache()


def test_build_bank_card_aggregates_stance_from_corpus(tmp_path: Path) -> None:
    registry = tmp_path / "source_registry.jsonl"
    _write_registry(
        registry,
        rows=[
            # Three sentences for ECB on the same event_date; one hawkish-tilt.
            {
                "source": "gtfintechlab_european_central_bank",
                "event_date": "2024-01-01",
                "text": "the council expects to hike rates further if inflation persists.",
            },
            {
                "source": "gtfintechlab_european_central_bank",
                "event_date": "2024-01-01",
                "text": "underlying inflation remains elevated.",
            },
            {
                "source": "gtfintechlab_european_central_bank",
                "event_date": "2024-01-01",
                "text": "growth has slowed.",
            },
            # An older row should not be picked up.
            {
                "source": "gtfintechlab_european_central_bank",
                "event_date": "2020-01-01",
                "text": "the council intends to cut rates to support recovery.",
            },
            # A foreign source must be ignored.
            {
                "source": "gtfintechlab_bank_of_japan",
                "event_date": "2024-01-01",
                "text": "rates will remain accommodative.",
            },
        ],
    )

    ecb_spec = next(spec for spec in svc.BANK_SPECS if spec.key == "ecb")
    card = svc.build_bank_card(
        ecb_spec,
        score_text=_stub_score,
        market_lookup=_stub_market,
        registry_path=registry,
    )

    assert card["status"] == "ok"
    assert card["bank"] == "ecb"
    assert card["short_code"] == "ECB"
    assert card["symbol"] == "^STOXX50E"
    assert card["latest_statement_date"] == "2024-01-01"
    assert card["sample_size"] == 3
    assert card["stance"] is not None
    # Hawkish should be the dominant axis since one of three sentences is hawkish
    # and the rest are neutral.
    assert card["stance_label"] == "hawkish"
    assert 0.0 <= card["stance_confidence"] <= 1.0
    assert pytest.approx(sum(card["stance"].values()), abs=1e-6) == 1.0
    assert card["certainty_label"] == "certain"
    assert card["time_axis"] == "forward looking"
    assert card["vol_regime_label"] == "normal"
    assert card["vol_regime_status"] == "ok"


def test_build_bank_card_marks_missing_corpus(tmp_path: Path) -> None:
    registry = tmp_path / "source_registry.jsonl"
    # Empty registry — no row for any bank.
    registry.write_text("")
    fed_spec = next(spec for spec in svc.BANK_SPECS if spec.key == "fed")
    card = svc.build_bank_card(
        fed_spec,
        score_text=_stub_score,
        market_lookup=_stub_market,
        registry_path=registry,
    )
    assert card["status"] == "corpus_missing"
    assert card["stance"] is None
    assert card["stance_label"] is None
    assert card["latest_statement_date"] is None
    # Market lookup is independent and should still populate.
    assert card["vol_regime_label"] == "calm"
    assert card["sample_size"] == 0


def test_build_snapshot_returns_all_six_banks(tmp_path: Path) -> None:
    registry = tmp_path / "source_registry.jsonl"
    rows = []
    for spec in svc.BANK_SPECS:
        rows.append(
            {
                "source": spec.source,
                "event_date": "2024-01-01",
                "text": f"{spec.short_code} statement: rates will hike further.",
            }
        )
    _write_registry(registry, rows)

    payload = svc.build_snapshot(
        score_text=_stub_score,
        market_lookup=_stub_market,
        registry_path=registry,
        use_cache=False,
    )

    keys = [card["bank"] for card in payload["banks"]]
    assert keys == ["fed", "ecb", "boe", "boc", "boj", "rba"]
    assert all(card["status"] == "ok" for card in payload["banks"])
    assert payload["cache_ttl_seconds"] == 3600
    assert payload["generated_at"]


def test_build_snapshot_cache_returns_same_payload(tmp_path: Path) -> None:
    registry = tmp_path / "source_registry.jsonl"
    _write_registry(
        registry,
        rows=[
            {
                "source": spec.source,
                "event_date": "2024-01-01",
                "text": "neutral pace of expansion continues.",
            }
            for spec in svc.BANK_SPECS
        ],
    )

    call_count = {"n": 0}

    def counting_score(text: str) -> dict[str, Any]:
        call_count["n"] += 1
        return _stub_score(text)

    first = svc.build_snapshot(
        score_text=counting_score,
        market_lookup=_stub_market,
        registry_path=registry,
        use_cache=True,
    )
    after_first = call_count["n"]
    second = svc.build_snapshot(
        score_text=counting_score,
        market_lookup=_stub_market,
        registry_path=registry,
        use_cache=True,
    )
    # Cache hit: classifier must not be called a second time.
    assert call_count["n"] == after_first
    assert second is first


def test_cross_bank_snapshot_endpoint(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    registry = tmp_path / "source_registry.jsonl"
    _write_registry(
        registry,
        rows=[
            {
                "source": spec.source,
                "event_date": "2024-01-01",
                "text": "policy will hike further to anchor inflation.",
            }
            for spec in svc.BANK_SPECS
        ],
    )

    svc.reset_cache()

    real_build = svc.build_snapshot

    def patched_build(**kwargs: Any) -> Any:
        kwargs.setdefault("score_text", _stub_score)
        kwargs.setdefault("market_lookup", _stub_market)
        kwargs.setdefault("registry_path", registry)
        return real_build(**kwargs)

    monkeypatch.setattr(svc, "build_snapshot", patched_build)

    client = TestClient(app)
    resp = client.get("/cross-bank/snapshot")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert {card["bank"] for card in body["banks"]} == {
        "fed",
        "ecb",
        "boe",
        "boc",
        "boj",
        "rba",
    }
    assert body["cache_ttl_seconds"] == 3600
    fed_card = next(c for c in body["banks"] if c["bank"] == "fed")
    assert fed_card["stance_label"] == "hawkish"
    assert fed_card["vol_regime_label"] == "calm"
    assert fed_card["symbol"] == "^GSPC"
    # Vol regime band confidence must lie in [0.55, 0.95].
    assert 0.55 <= fed_card["vol_regime_confidence"] <= 0.95
