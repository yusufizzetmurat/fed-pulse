from __future__ import annotations

from app.data import quality_checks
from app.data.source_type import (
    SOURCE_TYPE_FOMC_MINUTES,
    SOURCE_TYPE_FOMC_STATEMENT,
)


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
