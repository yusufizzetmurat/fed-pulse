"""Unit tests for the Phase C cross-bank supervision flag (#228).

The flag toggles whether ``finetune_pilot._load_registry_rows`` drops
rows tagged ``sample_weight==0`` (the cross-bank pool's provenance
gate). ``off`` (default) reproduces today's strictly-FOMC training pool
byte-identical. ``on`` admits cross-bank rows and forces their weight to
1.0 so the head treats them as full-weight training rows.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.data.finetune_pilot import _load_registry_rows


def _write_registry(
    package_dir: Path,
    rows: list[dict[str, object]],
) -> None:
    """Materialise a synthetic ``registry_normalized.jsonl`` so
    ``_load_registry_rows`` can ingest it."""

    package_dir.mkdir(parents=True, exist_ok=True)
    target = package_dir / "registry_normalized.jsonl"
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _fomc_row(record_id: str, label: str = "hawkish") -> dict[str, object]:
    return {
        "record_id": record_id,
        "text": "Committee notes inflation pressures remain elevated.",
        "mapped_label": label,
        "event_date": "2024-09-18",
        "provenance": "peer_reviewed",
        "sample_weight": 1.0,
    }


def _cross_bank_row(record_id: str, label: str = "dovish") -> dict[str, object]:
    return {
        "record_id": record_id,
        "text": "The Governing Council expects rates to remain accommodative.",
        "mapped_label": label,
        "event_date": "2024-09-12",
        "provenance": "peer_reviewed_cross_bank",
        "sample_weight": 0.0,
    }


def test_default_drops_zero_weight_rows(tmp_path: Path) -> None:
    """``include_zero_weight=False`` (the default) keeps only the FOMC
    rows. Reproduces the pre-Phase-C training-row count exactly."""

    _write_registry(
        tmp_path,
        [
            _fomc_row("fomc-1"),
            _fomc_row("fomc-2", label="dovish"),
            _cross_bank_row("ecb-1"),
            _cross_bank_row("boj-1", label="hawkish"),
        ],
    )
    rows = _load_registry_rows(tmp_path)
    record_ids = sorted(r.record_id for r in rows)
    assert record_ids == ["fomc-1", "fomc-2"]


def test_include_zero_weight_admits_cross_bank_rows(tmp_path: Path) -> None:
    """``include_zero_weight=True`` admits cross-bank rows. Phase C
    leaves the weight at the on-disk value; the head-side trainer is
    responsible for forcing it to 1.0 when ``--cross-bank-supervision
    on`` is set."""

    _write_registry(
        tmp_path,
        [
            _fomc_row("fomc-1"),
            _cross_bank_row("ecb-1"),
            _cross_bank_row("boj-1", label="hawkish"),
        ],
    )
    rows = _load_registry_rows(tmp_path, include_zero_weight=True)
    record_ids = sorted(r.record_id for r in rows)
    assert record_ids == ["boj-1", "ecb-1", "fomc-1"]
    # The cross-bank rows surface with their on-disk sample_weight; the
    # trainer call site is what flips them to 1.0 on the ``on`` path.
    weights = {r.record_id: r.sample_weight for r in rows}
    assert weights["fomc-1"] == 1.0
    assert weights["ecb-1"] == 0.0
    assert weights["boj-1"] == 0.0


def test_cross_bank_provenance_distinct_from_supervised_pool(tmp_path: Path) -> None:
    """The cross-bank rows preserve their ``provenance`` field through
    ingestion, so downstream consumers can filter on it after the
    inclusion gate flips."""

    _write_registry(
        tmp_path,
        [
            _fomc_row("fomc-1"),
            _cross_bank_row("ecb-1"),
        ],
    )
    rows = _load_registry_rows(tmp_path, include_zero_weight=True)
    by_id = {r.record_id: r for r in rows}
    assert by_id["fomc-1"].provenance == "peer_reviewed"
    assert by_id["ecb-1"].provenance == "peer_reviewed_cross_bank"
