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


@pytest.fixture(autouse=True)
def _stub_realized_close_returns(monkeypatch: pytest.MonkeyPatch) -> None:
    """#299: the analogs renderer now augments each card with realized
    5d/20d S&P close-to-close returns. Stub the lookup to None so this
    file's existing tests do not depend on network access; a dedicated
    test file exercises the augmentation behaviour itself."""

    monkeypatch.setattr(
        analogs_service,
        "_subsequent_close_pct",
        lambda event_date, *, horizon: None,
    )


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
    # 0.022 > VOL_REGIME_BUCKET_EDGES[1] (0.020) -> "high"
    assert analogs[0]["subsequent_vol_regime"] == "high"
    assert "forward_realized_vol_10d" not in analogs[0]
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


def test_analyze_analogs_503_does_not_leak_exception_text(
    client: TestClient, tmp_path: Path
) -> None:
    """An unexpected embedder error responds 503 with a sanitized detail.

    The endpoint's exception arm must NOT echo internal exception text
    (file paths, library internals) back to the client.
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

    secret_path = "/internal/secret/checkpoint/path"

    def _exploding_embedder(texts: list[str]):
        raise RuntimeError(f"boom at {secret_path}")

    state = analogs_service.build_state_from_index(
        loaded, embed_fn=_exploding_embedder, encoder_alias="test_retrieval"
    )
    analogs_service.install_state(state)

    response = client.post(
        "/analyze/analogs", json={"text": "anything", "k": 3}
    )
    assert response.status_code == 503
    body = response.json()
    # Sanitised detail field — no internal exception text / paths.
    assert body.get("detail") == "Analog retrieval unavailable"
    assert secret_path not in response.text
    assert "RuntimeError" not in response.text
    assert "Traceback" not in response.text


def test_analyze_analogs_as_of_date_filters_future_rows(
    client: TestClient, tmp_path: Path
) -> None:
    """``as_of_date`` cuts the candidate pool to strict-backward rows."""

    _install_fake_state(tmp_path)
    # Fixture rows are 2008-12-16, 2022-09-21, 2015-12-16. Query as_of
    # in 2016 must drop the 2022 row even though it dominates on
    # "inflation".
    response = client.post(
        "/analyze/analogs",
        json={
            "text": "Inflation outlook remains elevated",
            "k": 5,
            "as_of_date": "2016-01-01",
        },
    )
    assert response.status_code == 200
    body = response.json()
    dates = {card["event_date"] for card in body["analogs"]}
    assert "2022-09-21" not in dates
    assert dates <= {"2008-12-16", "2015-12-16"}
