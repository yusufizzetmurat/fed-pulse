"""Tests for /forecasts/next-fomc (multi-page expansion #150)."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("torch")
pytest.importorskip("transformers")

from fastapi.testclient import TestClient  # noqa: E402

import app.main as main_mod  # noqa: E402
from app.services import decision_forecast  # noqa: E402


@pytest.fixture
def client() -> TestClient:
    return TestClient(main_mod.app)


def test_next_fomc_endpoint_empty_state(client, monkeypatch, tmp_path):
    monkeypatch.setattr(main_mod, "DATA_DIR", tmp_path)
    # Disable the HF cold-start hydration so this exercises the genuine
    # no-artifact empty state rather than pulling the published fallback.
    monkeypatch.setattr(decision_forecast, "_NEXT_FOMC_HF_REPO", "")
    response = client.get("/forecasts/next-fomc")
    assert response.status_code == 200
    body = response.json()
    assert body["available"] is False
    assert body["headline"] is None
    assert body["history"] == []
    assert body["ordinal_classes"] == [
        "cut_50",
        "cut_25",
        "hold",
        "hike_25",
        "hike_50",
        "hike_75",
    ]
    # The upcoming meeting is still surfaced even with no artefacts.
    assert "upcoming_meeting" in body


def test_next_fomc_endpoint_with_artifacts(client, monkeypatch, tmp_path):
    artifacts_dir = tmp_path / "artifacts" / "next_fomc"
    artifacts_dir.mkdir(parents=True)
    results = {
        "predictions": [
            {
                "target_event_date": "2024-09-17",
                "target_as_of_ts": "2024-09-17T19:00:00+00:00",
                "target_class": "cut_25",
                "n_train_rows": 12,
                "probabilities": {
                    "ordinal_logit": {
                        "cut_50": 0.05,
                        "cut_25": 0.55,
                        "hold": 0.30,
                        "hike_25": 0.07,
                        "hike_50": 0.02,
                        "hike_75": 0.01,
                    },
                    "ois_baseline": {
                        "cut_50": 0.10,
                        "cut_25": 0.45,
                        "hold": 0.40,
                        "hike_25": 0.04,
                        "hike_50": 0.01,
                        "hike_75": 0.00,
                    },
                },
            },
            {
                "target_event_date": "2024-11-06",
                "target_as_of_ts": "2024-11-06T19:00:00+00:00",
                "target_class": "hold",
                "n_train_rows": 13,
                "probabilities": {
                    "ordinal_logit": {
                        "cut_50": 0.01,
                        "cut_25": 0.10,
                        "hold": 0.70,
                        "hike_25": 0.15,
                        "hike_50": 0.03,
                        "hike_75": 0.01,
                    },
                    "ois_baseline": {
                        "cut_50": 0.02,
                        "cut_25": 0.12,
                        "hold": 0.65,
                        "hike_25": 0.18,
                        "hike_50": 0.02,
                        "hike_75": 0.01,
                    },
                },
            },
        ],
        "summary": {"rows_emitted": 2, "rows_in_events": 2},
    }
    (artifacts_dir / "results.json").write_text(
        json.dumps(results, indent=2, sort_keys=True), encoding="utf-8"
    )
    metrics = {
        "model_names": ["ordinal_logit", "ois_baseline"],
        "full_window": {
            "ordinal_logit": {
                "n": 2,
                "brier": 0.32,
                "log_loss": 0.71,
                "top1_accuracy": 1.0,
                "macro_f1": 0.83,
                "confusion_matrix": {"hold": {"hold": 1}, "cut_25": {"cut_25": 1}},
            },
            "ois_baseline": {
                "n": 2,
                "brier": 0.45,
                "log_loss": 0.95,
                "top1_accuracy": 0.5,
                "macro_f1": 0.5,
                "confusion_matrix": {"hold": {"hold": 1}, "cut_25": {"hold": 1}},
            },
        },
        "ex_pandemic_window": {
            "ordinal_logit": {
                "n": 2,
                "brier": 0.32,
                "log_loss": 0.71,
                "top1_accuracy": 1.0,
                "macro_f1": 0.83,
                "confusion_matrix": {},
            },
        },
    }
    (artifacts_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
    )
    (artifacts_dir / "feature_attribution.md").write_text(
        (
            "# Next-FOMC decision -- feature-family attribution\n\n"
            "| Subset | Families | #features | n | Brier | LogLoss | Top1Acc | MacroF1 |\n"
            "| --- | --- | --- | --- | --- | --- | --- | --- |\n"
            "| ois_only | ois | 10 | 2 | 0.45 | 0.95 | 0.5 | 0.5 |\n"
            "| full | ois, text, linguistic, credibility, macro | 39 | 2 | 0.32 | 0.71 | 1.0 | 0.83 |\n"
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(main_mod, "DATA_DIR", tmp_path)
    response = client.get("/forecasts/next-fomc")
    assert response.status_code == 200
    body = response.json()
    assert body["available"] is True
    assert body["model_names"] == ["ordinal_logit", "ois_baseline"]
    assert len(body["history"]) == 2
    assert body["history"][0]["predicted_class"]["ordinal_logit"] == "cut_25"
    headline = body["headline"]
    assert headline is not None
    # Latest entry is the November meeting; matches the latest history row.
    assert headline["target_event_date"] in {"2024-09-17", "2024-11-06"}
    # Attribution: parsed two rows.
    subsets = [row["subset"] for row in body["feature_attribution"]]
    assert subsets == ["ois_only", "full"]
    assert body["feature_attribution"][1]["families"] == [
        "ois",
        "text",
        "linguistic",
        "credibility",
        "macro",
    ]
    assert body["metrics_full_window"]["ordinal_logit"]["macro_f1"] == 0.83
    assert body["summary"]["rows_emitted"] == 2


def test_decision_forecast_loader_argmax(tmp_path):
    artifacts_dir = tmp_path / "next_fomc"
    artifacts_dir.mkdir()
    (artifacts_dir / "results.json").write_text(
        json.dumps(
            {
                "predictions": [
                    {
                        "target_event_date": "2026-09-15",
                        "target_as_of_ts": "2026-09-15T19:00:00+00:00",
                        "target_class": None,
                        "n_train_rows": 28,
                        "probabilities": {
                            "ordinal_logit": {
                                "cut_50": 0.02,
                                "cut_25": 0.18,
                                "hold": 0.55,
                                "hike_25": 0.20,
                                "hike_50": 0.04,
                                "hike_75": 0.01,
                            }
                        },
                    }
                ],
                "summary": {},
            }
        ),
        encoding="utf-8",
    )
    payload = decision_forecast.load_next_fomc_artifacts(
        artifacts_dir, reference_date=date(2026, 9, 1)
    )
    assert payload["available"] is True
    assert payload["headline"]["predicted_class"]["ordinal_logit"] == "hold"
