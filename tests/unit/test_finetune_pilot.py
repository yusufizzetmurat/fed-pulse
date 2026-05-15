from __future__ import annotations

import json
from pathlib import Path

from app.data.finetune_pilot import write_predictions_jsonl


def test_write_predictions_jsonl_emits_one_row_per_prediction(tmp_path: Path) -> None:
    rows = [
        {"record_id": "r1", "gold": "hawkish", "pred": "hawkish"},
        {"record_id": "r2", "gold": "dovish", "pred": "neutral"},
        {"record_id": "r3", "gold": "neutral", "pred": "neutral"},
    ]
    output = tmp_path / "predictions.jsonl"
    write_predictions_jsonl(
        record_ids=[r["record_id"] for r in rows],
        gold_labels=[r["gold"] for r in rows],
        predicted_labels=[r["pred"] for r in rows],
        output_path=output,
    )

    lines = output.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    payloads = [json.loads(line) for line in lines]
    assert payloads[0]["record_id"] == "r1"
    assert payloads[0]["mapped_label"] == "hawkish"
    assert payloads[0]["predicted_label"] == "hawkish"
    assert payloads[2]["mapped_label"] == "neutral"
    assert payloads[2]["predicted_label"] == "neutral"


def test_write_predictions_jsonl_raises_on_length_mismatch(tmp_path: Path) -> None:
    import pytest

    output = tmp_path / "predictions.jsonl"
    with pytest.raises(ValueError):
        write_predictions_jsonl(
            record_ids=["r1", "r2"],
            gold_labels=["hawkish"],
            predicted_labels=["hawkish", "dovish"],
            output_path=output,
        )
