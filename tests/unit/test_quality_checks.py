from __future__ import annotations

from app.data import quality_checks
from app.data.source_type import (
    SOURCE_TYPE_FOMC_MINUTES,
    SOURCE_TYPE_FOMC_STATEMENT,
)


def test_exact_dedup_collapses_same_text_hash_across_different_event_dates() -> None:
    # The same speech text can be linked to two FOMC events (e.g. when a
    # cross-source row mirrors a release date and the prior-day window).
    # QualityPassedRowSchema asserts text_hash uniqueness, so dedup must
    # keep one row per text_hash regardless of event_date.
    rows = [
        {"record_id": "r1", "text_hash": "h1", "event_date": "2024-01-30"},
        {"record_id": "r2", "text_hash": "h1", "event_date": "2024-03-20"},
        {"record_id": "r3", "text_hash": "h2", "event_date": "2024-01-30"},
    ]

    kept, report = quality_checks._exact_dedup(rows)

    kept_hashes = [row["text_hash"] for row in kept]
    assert kept_hashes == ["h1", "h2"]
    assert report["input_rows"] == 3
    assert report["kept_rows"] == 2
    assert report["dropped_rows"] == 1
    assert report["dropped"][0]["record_id"] == "r2"
    assert report["dropped"][0]["kept_record_id"] == "r1"
    assert report["dropped"][0]["reason"] == "exact_text_hash_duplicate"


def test_exact_dedup_still_collapses_same_text_hash_same_event_date() -> None:
    rows = [
        {"record_id": "r1", "text_hash": "h1", "event_date": "2024-01-30"},
        {"record_id": "r2", "text_hash": "h1", "event_date": "2024-01-30"},
    ]

    kept, report = quality_checks._exact_dedup(rows)

    assert [row["record_id"] for row in kept] == ["r1"]
    assert report["kept_rows"] == 1
    assert report["dropped_rows"] == 1


def test_distribution_report_carries_source_type_aggregations() -> None:
    rows = [
        {
            "source": "scraped_fed",
            "source_type": SOURCE_TYPE_FOMC_MINUTES,
            "mapped_label": "hawkish",
        },
        {
            "source": "scraped_fed",
            "source_type": SOURCE_TYPE_FOMC_MINUTES,
            "mapped_label": "dovish",
        },
        {
            "source": "kaggle_fed_statements_minutes",
            "source_type": SOURCE_TYPE_FOMC_STATEMENT,
            "mapped_label": "neutral",
        },
        {
            "source": "scraped_fed",
            "source_type": SOURCE_TYPE_FOMC_MINUTES,
            "mapped_label": "",
        },
    ]

    report = quality_checks._distribution_report(rows)

    assert report["source_type_counts"] == {
        SOURCE_TYPE_FOMC_MINUTES: 3,
        SOURCE_TYPE_FOMC_STATEMENT: 1,
    }
    assert report["source_type_label_counts"] == {
        SOURCE_TYPE_FOMC_MINUTES: {"hawkish": 1, "dovish": 1, "unlabeled": 1},
        SOURCE_TYPE_FOMC_STATEMENT: {"neutral": 1},
    }
    # existing keys must still be present
    assert "source_counts" in report
    assert "mapped_label_counts" in report
    assert "source_label_counts" in report
