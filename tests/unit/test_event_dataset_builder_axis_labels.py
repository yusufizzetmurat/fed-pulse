from __future__ import annotations

from app.data.event_dataset_builder import (
    _RegistryRow,
    _aggregate_events,
)


def _row(
    source: str,
    event_date: str,
    *,
    mapped_label: str | None = "hawkish",
    time_label: str | None = None,
    certain_label: str | None = None,
) -> _RegistryRow:
    extras: dict[str, object] = {}
    if time_label is not None:
        extras["gtfintechlab_time_label"] = time_label
    if certain_label is not None:
        extras["gtfintechlab_certain_label"] = certain_label
    return _RegistryRow(
        source=source,
        source_record_id=f"rec-{source}-{event_date}",
        document_type="statement",
        event_date=event_date,
        text="dummy text",
        mapped_label=mapped_label,
        multi_axis_extras=extras,
        axes={"stance": None, "time": None, "certainty": None, "factor": None, "topic": None},
    )


def test_aggregate_events_lifts_gtfintechlab_labels() -> None:
    rows = [
        _row(
            "gtfintechlab_federal_reserve_system",
            "2024-09-18",
            time_label="forward looking",
            certain_label="certain",
        )
    ]
    docs = _aggregate_events(rows)
    assert len(docs) == 1
    doc = docs[0]
    assert doc.time_label == "forward looking"
    assert doc.certain_label == "certain"


def test_aggregate_events_leaves_labels_none_for_non_gtfintechlab_rows() -> None:
    rows = [_row("hf_fomc_communication", "2024-09-18")]
    docs = _aggregate_events(rows)
    assert docs[0].time_label is None
    assert docs[0].certain_label is None


def test_label_state_does_not_leak_across_buckets() -> None:
    """A populated bucket must not leak its labels into the next iteration."""

    rows = [
        _row(
            "gtfintechlab_federal_reserve_system",
            "2024-09-18",
            time_label="not forward looking",
            certain_label="uncertain",
        ),
        # Different source + date — fresh bucket; labels must start at None.
        _row("hf_fomc_communication", "2024-11-07"),
    ]
    docs = sorted(_aggregate_events(rows), key=lambda d: d.event_date)
    assert docs[0].time_label == "not forward looking"
    assert docs[0].certain_label == "uncertain"
    assert docs[1].time_label is None
    assert docs[1].certain_label is None


def test_aggregate_events_lowercases_and_strips_label_strings() -> None:
    rows = [
        _row(
            "gtfintechlab_federal_reserve_system",
            "2024-09-18",
            time_label="  Forward Looking  ",
            certain_label="CERTAIN",
        )
    ]
    docs = _aggregate_events(rows)
    assert docs[0].time_label == "forward looking"
    assert docs[0].certain_label == "certain"
