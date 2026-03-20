from __future__ import annotations

import json

from app.prepare_training_data import prepare_training_dataset
from app.services.forecaster import inspect_training_data_sources


def test_prepare_training_dataset_creates_grouped_records(tmp_path, monkeypatch):
    documents = [
        {
            "date": f"2026-01-{idx + 1:02d}",
            "text": f"Document text {idx}",
            "title": f"Doc {idx}",
            "document_type": "Minutes",
        }
        for idx in range(7)
    ]
    (tmp_path / "fomc_minutes.json").write_text(json.dumps(documents), encoding="utf-8")
    (tmp_path / "fomc_statements.json").write_text(json.dumps([]), encoding="utf-8")

    monkeypatch.setattr(
        "app.prepare_training_data.analyze_text",
        lambda text: {"label": "POSITIVE", "score": 0.55, "raw": [{"label": "POSITIVE", "score": 0.55}]},
    )
    monkeypatch.setattr(
        "app.prepare_training_data.fetch_market_snapshot",
        lambda target_date, symbol: {
            "symbol": symbol,
            "requested_date": target_date,
            "date_used": target_date,
            "lookback_days": 7,
            "close": 5000.0 + len(symbol),
            "volatility_5d": 0.01,
        },
    )

    output_path = prepare_training_dataset(
        data_dir=tmp_path,
        symbols=["^GSPC", "^VIX"],
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert "groups" in payload
    assert len(payload["groups"]) == 2
    assert payload["metadata"]["document_count"] == 7
    assert payload["metadata"]["symbols"] == ["^GSPC", "^VIX"]
    assert all(group["records"] for group in payload["groups"])


def test_prepare_training_dataset_output_is_trainer_compatible(tmp_path, monkeypatch):
    documents = [
        {
            "date": f"2026-02-{idx + 1:02d}",
            "text": f"Statement text {idx}",
            "title": f"Statement {idx}",
            "document_type": "Statement",
        }
        for idx in range(7)
    ]
    (tmp_path / "fomc_statements.json").write_text(json.dumps(documents), encoding="utf-8")
    (tmp_path / "fomc_minutes.json").write_text(json.dumps([]), encoding="utf-8")

    monkeypatch.setattr(
        "app.prepare_training_data.analyze_text",
        lambda text: {"label": "NEGATIVE", "score": 0.42, "raw": [{"label": "NEGATIVE", "score": 0.42}]},
    )
    monkeypatch.setattr(
        "app.prepare_training_data.fetch_market_snapshot",
        lambda target_date, symbol: {
            "symbol": symbol,
            "requested_date": target_date,
            "date_used": target_date,
            "lookback_days": 7,
            "close": 4500.0,
            "volatility_5d": 0.012,
        },
    )

    prepare_training_dataset(data_dir=tmp_path, symbols=["^GSPC"])
    sequences, summaries = inspect_training_data_sources(tmp_path)

    assert len(sequences) == 1
    assert len(sequences[0]) == 7
    assert any(summary.path.name == "train_dataset.json" and summary.status == "usable" for summary in summaries)
