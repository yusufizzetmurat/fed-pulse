from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")

from fastapi.testclient import TestClient  # noqa: E402

import app.db as db_module  # noqa: E402
import app.main as main_mod  # noqa: E402


@pytest.fixture()
def client(tmp_path):
    db_module.reset_for_testing(f"sqlite:///{tmp_path / 'fed_pulse_history.db'}")
    return TestClient(main_mod.app)


def _seed(session, *, symbol="^GSPC", stance="HAWKISH", document_date="2024-09-18"):
    return db_module.persist_analysis_run(
        session,
        payload={
            "sentiment": {"label": stance, "score": 0.7, "raw": []},
            "prediction": {"close": 5050.0, "volatility": 0.012, "horizon": "3d"},
        },
        request={
            "text": "Recent indicators…",
            "date": document_date,
            "symbol": symbol,
            "horizon": "3d",
            "forecast_mode": "fast",
            "include_realized": False,
        },
        response={
            "sentiment": {"label": stance, "score": 0.7, "raw": []},
            "prediction": {"close": 5050.0, "volatility": 0.012, "horizon": "3d"},
            "market": {
                "symbol": symbol,
                "requested_date": document_date,
                "date_used": document_date,
                "lookback_days": 5,
                "close": 5000.0,
                "volatility_5d": 0.011,
            },
            "model": {},
            "series": {},
        },
    )


def test_history_endpoints_round_trip(client):
    session_iter = db_module.get_session()
    sess = next(session_iter)
    try:
        first = _seed(sess)
        _seed(sess, symbol="^NDX")
        _seed(sess, stance="DOVISH")
    finally:
        sess.close()

    response = client.get("/history")
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 3
    assert len(body["items"]) == 3

    filtered = client.get("/history", params={"symbol": "^NDX"})
    assert filtered.status_code == 200
    assert filtered.json()["total"] == 1

    detail = client.get(f"/history/{first.id}")
    assert detail.status_code == 200
    assert detail.json()["id"] == first.id
    assert "payload" in detail.json()

    deletion = client.delete(f"/history/{first.id}")
    assert deletion.status_code == 204

    missing = client.get(f"/history/{first.id}")
    assert missing.status_code == 404


def test_history_filter_by_stance_and_date(client):
    session_iter = db_module.get_session()
    sess = next(session_iter)
    try:
        _seed(sess, stance="HAWKISH", document_date="2024-09-18")
        _seed(sess, stance="DOVISH", document_date="2024-11-06")
    finally:
        sess.close()

    response = client.get("/history", params={"stance": "dovish"})
    assert response.status_code == 200
    assert response.json()["total"] == 1
    assert response.json()["items"][0]["stance"] == "dovish"

    by_date = client.get("/history", params={"document_date": "2024-09-18"})
    assert by_date.status_code == 200
    assert by_date.json()["total"] == 1


def test_history_realized_endpoint(client, monkeypatch):
    session_iter = db_module.get_session()
    sess = next(session_iter)
    try:
        run = _seed(sess, document_date="2024-09-18")
    finally:
        sess.close()

    monkeypatch.setattr(
        main_mod,
        "fetch_realized_forward",
        lambda **_: [
            {"date": "2024-09-19", "close": 5602.0, "volatility_5d": 0.0110},
            {"date": "2024-09-20", "close": 5615.0, "volatility_5d": 0.0112},
            {"date": "2024-09-23", "close": 5628.0, "volatility_5d": 0.0115},
        ],
    )

    response = client.get(f"/history/{run.id}/realized")
    assert response.status_code == 200
    body = response.json()
    assert body["run_id"] == run.id
    assert body["timestamps"] == ["2024-09-19", "2024-09-20", "2024-09-23"]
    assert body["close"] == [5602.0, 5615.0, 5628.0]
    assert body["volatility"] == [0.0110, 0.0112, 0.0115]


def test_history_realized_endpoint_404_for_missing_run(client):
    response = client.get("/history/does-not-exist/realized")
    assert response.status_code == 404


def test_history_realized_batch_returns_items_and_missing(client, monkeypatch):
    session_iter = db_module.get_session()
    sess = next(session_iter)
    try:
        run_a = _seed(sess, document_date="2024-09-18")
        run_b = _seed(sess, document_date="2024-11-06", symbol="^NDX")
    finally:
        sess.close()

    monkeypatch.setattr(
        main_mod,
        "fetch_realized_forward",
        lambda **_: [
            {"date": "2024-09-19", "close": 5602.0, "volatility_5d": 0.011},
        ],
    )

    response = client.get(
        "/history-realized",
        params={"ids": f"{run_a.id},{run_b.id},does-not-exist"},
    )
    assert response.status_code == 200
    body = response.json()
    assert set(body["items"].keys()) == {run_a.id, run_b.id}
    assert body["missing"] == ["does-not-exist"]
    assert body["items"][run_a.id]["timestamps"] == ["2024-09-19"]


def test_history_realized_batch_rejects_empty_ids(client):
    response = client.get("/history-realized", params={"ids": ""})
    assert response.status_code == 422


def test_history_realized_batch_rejects_oversize_id_list(client):
    too_many = ",".join(f"id-{n}" for n in range(51))
    response = client.get("/history-realized", params={"ids": too_many})
    assert response.status_code == 422


def _seed_with_regime(sess, *, predicted_set, document_date="2024-09-18", symbol="^GSPC"):
    return db_module.persist_analysis_run(
        sess,
        payload={
            "sentiment": {"label": "HAWKISH", "score": 0.7, "raw": []},
            "prediction": {"close": 5050.0, "volatility": 0.012, "horizon": "10d"},
            "regime_classification": {
                "predicted_set": predicted_set,
                "set_label": "|".join(predicted_set),
                "set_size": len(predicted_set),
                "coverage": 0.8,
                "distribution": {"calm": 0.2, "normal": 0.5, "high": 0.3},
                "argmax_class": "normal",
            },
            "series": {"forecast_confidence_level": 0.8},
        },
        request={
            "text": "Recent indicators…",
            "date": document_date,
            "symbol": symbol,
            "horizon": "10d",
            "forecast_mode": "fast",
            "include_realized": False,
        },
        response={
            "sentiment": {"label": "HAWKISH", "score": 0.7, "raw": []},
            "prediction": {"close": 5050.0, "volatility": 0.012, "horizon": "10d"},
            "regime_classification": {
                "predicted_set": predicted_set,
                "set_label": "|".join(predicted_set),
                "set_size": len(predicted_set),
                "coverage": 0.8,
                "distribution": {"calm": 0.2, "normal": 0.5, "high": 0.3},
                "argmax_class": "normal",
            },
            "series": {"forecast_confidence_level": 0.8},
            "market": {
                "symbol": symbol,
                "requested_date": document_date,
                "date_used": document_date,
                "lookback_days": 5,
                "close": 5000.0,
                "volatility_5d": 0.011,
            },
            "model": {},
        },
    )


def test_evaluation_coverage_aggregates_hits_and_misses(client, monkeypatch):
    main_mod._reset_coverage_cache()

    session_iter = db_module.get_session()
    sess = next(session_iter)
    try:
        _seed_with_regime(sess, predicted_set=["calm", "normal"], document_date="2024-09-18")
        _seed_with_regime(sess, predicted_set=["normal"], document_date="2024-11-06")
        _seed_with_regime(sess, predicted_set=["high"], document_date="2024-12-18")
    finally:
        sess.close()

    monkeypatch.setattr(
        main_mod, "bucket_realized_regime", lambda *_args, **_kwargs: "normal"
    )
    monkeypatch.setattr(
        main_mod,
        "fetch_realized_forward",
        lambda **_: [{"date": "x", "close": 1.0, "volatility_5d": 0.01}],
    )

    response = client.get("/evaluation/coverage", params={"lookback_runs": 10})
    assert response.status_code == 200
    body = response.json()
    # 3 runs, 2 contain "normal" in their predicted_set → empirical 2/3
    assert body["sample_size"] == 3
    assert body["runs_total"] == 3
    assert body["empirical"] == pytest.approx(2 / 3)
    assert body["nominal"] == pytest.approx(0.8)


def test_evaluation_coverage_returns_zero_sample_when_no_predicted_set(client):
    main_mod._reset_coverage_cache()
    response = client.get("/evaluation/coverage")
    assert response.status_code == 200
    body = response.json()
    assert body["sample_size"] == 0
    assert body["runs_total"] == 0
    assert body["empirical"] is None
    assert body["nominal"] is None


def _write_breakdown_artifact(root, *, package_id="tp_fixture"):
    import json
    from pathlib import Path

    artifact_dir = Path(root) / "artifacts" / "regime_baseline_tiers" / package_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "training_package_id": package_id,
        "checkpoint_path": f"/app/models/forecaster_{package_id}.pt",
        "best_trial": {
            "summary": {
                "metrics": {
                    "classification_breakdown": {
                        "confusion_matrix": [[10, 1, 2], [3, 12, 1], [1, 2, 8]],
                        "per_class": [
                            {
                                "class_id": 0,
                                "precision": 0.71,
                                "recall": 0.77,
                                "f1": 0.74,
                                "support": 13,
                                "roc_auc": 0.82,
                                "pr_auc": 0.78,
                            },
                            {
                                "class_id": 1,
                                "precision": 0.80,
                                "recall": 0.75,
                                "f1": 0.77,
                                "support": 16,
                                "roc_auc": 0.85,
                                "pr_auc": 0.80,
                            },
                            {
                                "class_id": 2,
                                "precision": 0.73,
                                "recall": 0.73,
                                "f1": 0.73,
                                "support": 11,
                                "roc_auc": 0.80,
                                "pr_auc": 0.75,
                            },
                        ],
                        "macro_f1": 0.75,
                        "macro_precision": 0.75,
                        "macro_recall": 0.75,
                        "macro_roc_auc": 0.82,
                        "macro_pr_auc": 0.78,
                        "weighted_f1": 0.75,
                        "n_classes": 3,
                    },
                },
            },
        },
    }
    path = artifact_dir / "forecaster_sweep_results.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_evaluation_classification_breakdown_reads_latest_artifact(client, tmp_path, monkeypatch):
    monkeypatch.setattr(main_mod, "DATA_DIR", tmp_path)
    older = _write_breakdown_artifact(tmp_path, package_id="tp_old")
    # Newer artifact wins on mtime. Avoid ``time.sleep`` for ordering --
    # filesystems with second-resolution mtimes (or fast CI runners)
    # will round both writes to the same tick. Anchor both files at
    # explicit timestamps so the ordering is deterministic regardless
    # of FS resolution.
    import os
    import time

    newer = _write_breakdown_artifact(tmp_path, package_id="tp_new")
    now = time.time()
    os.utime(older, (now, now))
    os.utime(newer, (now + 3, now + 3))

    response = client.get("/evaluation/classification-breakdown")
    assert response.status_code == 200
    body = response.json()
    assert body["available"] is True
    assert body["macro_f1"] == 0.75
    assert body["n_classes"] == 3
    assert len(body["per_class"]) == 3
    assert body["per_class"][0]["class_id"] == 0
    assert body["confusion_matrix"][0] == [10, 1, 2]
    assert body["source"]["training_package_id"] == "tp_new"
    assert body["source"]["relative_path"].endswith(".json")


def test_evaluation_classification_breakdown_returns_unavailable_when_missing(
    client, tmp_path, monkeypatch
):
    monkeypatch.setattr(main_mod, "DATA_DIR", tmp_path)
    response = client.get("/evaluation/classification-breakdown")
    assert response.status_code == 200
    body = response.json()
    assert body["available"] is False
    assert body["macro_f1"] is None
    assert body["per_class"] is None
    assert body["source"] is None


def test_fomc_calendar_endpoint_returns_past_and_upcoming(client):
    response = client.get("/fomc/calendar", params={"as_of": "2024-09-18"})
    assert response.status_code == 200
    body = response.json()
    assert body["upcoming"]
    assert body["past"]
    assert body["upcoming"][0]["meeting_date"] >= "2024-09-18"
    assert body["past"][0]["meeting_date"] < "2024-09-18"


def test_fomc_calendar_validates_as_of(client):
    response = client.get("/fomc/calendar", params={"as_of": "not-a-date"})
    assert response.status_code == 422
