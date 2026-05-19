from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from app.data.ingest_sources import _coerce_label_origin


# ---------------------------------------------------------------------------
# Tier 1.6 — empty label_origin coerces to "unlabeled", not "pseudo"
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("empty_label", ["", "   ", None])
def test_coerce_label_origin_returns_unlabeled_for_empty(empty_label: Any) -> None:
    assert _coerce_label_origin(empty_label or "") == "unlabeled"


@pytest.mark.parametrize("real_label", ["hawkish", "dovish", "neutral", "anything"])
def test_coerce_label_origin_returns_human_for_real_labels(real_label: str) -> None:
    assert _coerce_label_origin(real_label) == "human"


def test_coerce_label_origin_never_emits_pseudo() -> None:
    """The audit's concern: empty labels MUST NOT collide with the
    teacher-model pseudo-label channel."""
    for label in ("", "   ", "hawkish", "dovish", "neutral"):
        assert _coerce_label_origin(label) != "pseudo"


# ---------------------------------------------------------------------------
# Tier 1.8 — normalize_labels.axes pulls from multi_axis_extras
# ---------------------------------------------------------------------------


def _run_normalize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Run the label-normalisation main path on an in-memory rowset.

    Writes a tmp JSONL, invokes the module entry point, reads back the
    output JSONL. Mirrors how the real pipeline drives normalize_labels.
    """
    import tempfile
    import subprocess
    import sys as _sys

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        input_path = tmp_path / "in.jsonl"
        output_path = tmp_path / "out.jsonl"
        exceptions_path = tmp_path / "exc.jsonl"
        metadata_path = tmp_path / "meta.json"
        input_path.write_text(
            "\n".join(json.dumps(r) for r in rows), encoding="utf-8"
        )
        result = subprocess.run(
            [
                _sys.executable,
                "-m",
                "app.data.normalize_labels",
                "--input",
                str(input_path),
                "--output",
                str(output_path),
                "--exceptions-output",
                str(exceptions_path),
                "--metadata-output",
                str(metadata_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        return [
            json.loads(line)
            for line in output_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]


def test_normalize_labels_axes_carries_gtfintechlab_time_label() -> None:
    rows = [
        {
            "record_id": "rec-1",
            "source": "gtfintechlab_federal_reserve_system",
            "source_record_id": "src-1",
            "document_type": "statement",
            "event_date": "2024-09-18",
            "text": "FOMC statement body",
            "text_hash": "a1b2c3d4e5f60718293a4b5c6d7e8f90a1b2c3d4e5f60718293a4b5c6d7e8f90",
            "label": "hawkish",
            "label_origin": "human",
            "license_scope": "research",
            "citation_ref": "test",
            "ingested_at_utc": "2024-09-18T19:00:00Z",
            "multi_axis_extras": {
                "gtfintechlab_time_label": "forward looking",
                "gtfintechlab_certain_label": "certain",
            },
        }
    ]
    normalised = _run_normalize(rows)
    assert len(normalised) == 1
    axes = normalised[0]["axes"]
    assert axes["stance"] == "hawkish"
    assert axes["time"] == "forward looking"
    assert axes["certainty"] == "certain"


def test_normalize_labels_axes_carries_gss_target_factor() -> None:
    rows = [
        {
            "record_id": "rec-gss",
            "source": "gss_factor",
            "source_record_id": "gss-1",
            "document_type": "statement",
            "event_date": "2008-09-16",
            "text": "GSS factor body",
            "text_hash": "0102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f20",
            "label": "hawkish",
            "label_origin": "human",
            "license_scope": "research",
            "citation_ref": "GSS",
            "ingested_at_utc": "2008-09-16T19:00:00Z",
            "multi_axis_extras": {
                "gss_target_factor": 0.42,
            },
        }
    ]
    normalised = _run_normalize(rows)
    axes = normalised[0]["axes"]
    assert axes["stance"] == "hawkish"
    assert axes["factor"] == pytest.approx(0.42)


def test_normalize_labels_axes_clean_when_no_extras() -> None:
    """A row with no multi_axis_extras keeps the time/factor/certainty
    slots at None -- the upstream rows that never carried per-axis
    labels should not invent them."""
    rows = [
        {
            "record_id": "rec-plain",
            "source": "hf_fomc_communication",
            "source_record_id": "tdw-1",
            "document_type": "statement",
            "event_date": "2024-09-18",
            "text": "TDW sentence",
            "text_hash": "2122232425262728292a2b2c2d2e2f303132333435363738393a3b3c3d3e3f40",
            "label": "neutral",
            "label_origin": "human",
            "license_scope": "research",
            "citation_ref": "TDW",
            "ingested_at_utc": "2024-09-18T19:00:00Z",
        }
    ]
    normalised = _run_normalize(rows)
    axes = normalised[0]["axes"]
    assert axes["stance"] == "neutral"
    assert axes["time"] is None
    assert axes["factor"] is None
    assert axes["certainty"] is None
