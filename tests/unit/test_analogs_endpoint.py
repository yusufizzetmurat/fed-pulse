"""Endpoint tests for POST /analyze/analogs (#294)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("torch")
pytest.importorskip("transformers")

from fastapi.testclient import TestClient  # noqa: E402

import app.main as main_mod  # noqa: E402
from app.retrieval import index as ret_index  # noqa: E402
from app.services import analogs as analogs_service  # noqa: E402


@pytest.fixture
def client() -> TestClient:
    return TestClient(main_mod.app)


@pytest.fixture(autouse=True)
def _reset_analogs_singleton():
    """Make sure each test sees a clean singleton — no cross-test bleed."""

    analogs_service.reset_state()
    yield
    analogs_service.reset_state()


def _fixture_rows() -> list[dict]:
    rows = [
        {
            "event_date": "2008-12-16",
            "event_kind": "statement",
            "text": "The Committee will employ all available tools to promote recovery and price stability.",
            "axis_stance": "dovish",
            "forward_realized_vol_10d": 0.045,
            "horizon": 1,
        },
        {
            "event_date": "2022-09-21",
            "event_kind": "statement",
            "text": "Inflation remains elevated. The Committee anticipates ongoing increases in the target range.",
            "axis_stance": "hawkish",
            "forward_realized_vol_10d": 0.022,
            "horizon": 1,
        },
        {
            "event_date": "2015-12-16",
            "event_kind": "statement",
            "text": "Economic conditions warrant a gradual removal of policy accommodation.",
            "axis_stance": "neutral",
            "forward_realized_vol_10d": 0.011,
            "horizon": 1,
        },
    ]
    for row in rows:
        row["text_hash"] = hashlib.sha256(row["text"].encode("utf-8")).hexdigest()
    return rows


def _make_keyword_embedder(keywords: list[str]):
    lower = [k.lower() for k in keywords]

    def _embed(texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), len(keywords)), dtype=np.float32)
        for row_idx, text in enumerate(texts):
            text_lc = (text or "").lower()
            for col_idx, kw in enumerate(lower):
                out[row_idx, col_idx] = float(text_lc.count(kw))
        return out

    return _embed


def _install_fake_state(tmp_path: Path) -> int:
    """Build a tiny on-disk bundle and wire it into the analogs singleton.

    Returns the index size so tests can assert against it without
    re-counting the fixture rows.
    """

    rows = _fixture_rows()
    events_parquet = tmp_path / "events.parquet"
    pd.DataFrame(rows).to_parquet(events_parquet, index=False)
    embed = _make_keyword_embedder(["inflation", "recovery", "accommodation"])
    loaded = ret_index.build_index_from_events(
        events_parquet=events_parquet,
        encoder_alias="test_retrieval",
        encoder_revision="rev1234",
        embed_fn=embed,
        training_package_id="tp_test",
        out_dir=tmp_path / "bundle",
    )
    state = analogs_service.build_state_from_index(
        loaded, embed_fn=embed, encoder_alias="test_retrieval"
    )
    analogs_service.install_state(state)
    return loaded.size


def test_analyze_analogs_returns_empty_when_bundle_missing(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When no bundle exists on disk the endpoint shapes an empty result."""

    monkeypatch.setenv("FED_PULSE_RETRIEVAL_DIR", str(tmp_path / "does-not-exist"))
    response = client.post("/analyze/analogs", json={"text": "inflation outlook", "k": 3})
    assert response.status_code == 200
    body = response.json()
    assert body["analogs"] == []
    assert body["index_size"] == 0
    assert body["encoder_alias"] == "finbert_fed_adjacent_xbank_dapt_retrieval"


def test_analyze_analogs_returns_descending_similarity(
    client: TestClient, tmp_path: Path
) -> None:
    _install_fake_state(tmp_path)
    response = client.post(
        "/analyze/analogs",
        json={"text": "Inflation outlook remains elevated", "k": 3},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["index_size"] == 3
    assert body["encoder_alias"] == "test_retrieval"
    analogs = body["analogs"]
    assert len(analogs) == 3
    # Top-1 must be the 2022 hawkish row because "inflation" dominates.
    assert analogs[0]["event_date"] == "2022-09-21"
    assert analogs[0]["axis_stance"] == "hawkish"
    assert analogs[0]["forward_realized_vol_10d"] == pytest.approx(0.022, rel=1e-5)
    assert analogs[0]["similarity"] == pytest.approx(1.0, abs=1e-5)
    # Strictly descending by similarity.
    similarities = [card["similarity"] for card in analogs]
    assert similarities == sorted(similarities, reverse=True)


def test_analyze_analogs_excerpt_present_and_bounded(
    client: TestClient, tmp_path: Path
) -> None:
    _install_fake_state(tmp_path)
    response = client.post(
        "/analyze/analogs",
        json={"text": "The Committee will support recovery", "k": 1},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["analogs"][0]["event_date"] == "2008-12-16"
    excerpt = body["analogs"][0]["excerpt"]
    assert excerpt
    # Excerpt is bounded by the EXCERPT_CHARS index constant.
    assert len(excerpt) <= ret_index.EXCERPT_CHARS


def test_analyze_analogs_validates_k_lower_bound(client: TestClient) -> None:
    response = client.post(
        "/analyze/analogs", json={"text": "Inflation outlook", "k": 0}
    )
    assert response.status_code == 422


def test_analyze_analogs_validates_k_upper_bound(client: TestClient) -> None:
    response = client.post(
        "/analyze/analogs", json={"text": "Inflation outlook", "k": 999}
    )
    assert response.status_code == 422


def test_analyze_analogs_requires_text(client: TestClient) -> None:
    response = client.post("/analyze/analogs", json={"text": "", "k": 3})
    assert response.status_code == 422
