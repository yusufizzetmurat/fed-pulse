"""Cover the event_dataset_builder hooks for the macro_release event kind.

The Variant A augmentation adds CPI / NFP release-date supervised
rows. Their document_type values (``macro_release_cpi`` /
``macro_release_nfp``) must map to a single canonical event_kind
(``macro_release``), and the announcement-time placeholder must use
the dedicated BLS 8:30 ET slot instead of the FOMC 2pm ET default.
"""

from __future__ import annotations


def test_event_kind_map_routes_macro_release_subtypes_to_one_kind() -> None:
    from app.data import event_dataset_builder as edb

    assert edb._EVENT_KIND_MAP["macro_release_cpi"] == "macro_release"
    assert edb._EVENT_KIND_MAP["macro_release_nfp"] == "macro_release"


def test_source_preference_includes_fred_macro_releases() -> None:
    """The dedup-collapse keeps the preferred source per (date, kind).
    Without ``fred_macro_releases`` in the preference list a future
    duplicate from another source could shadow the macro row."""

    from app.data import event_dataset_builder as edb

    assert "fred_macro_releases" in edb._SOURCE_PREFERENCE


def test_macro_release_as_of_ts_uses_bls_8_30_et_placeholder() -> None:
    """BLS releases land at 8:30 ET; the canonical timestamp on a
    macro_release event is the 13:00 UTC placeholder, not the 19:00
    UTC FOMC default or the 14:00 UTC speech default."""

    from app.data import event_dataset_builder as edb

    ts = edb._as_of_for_event("2024-01-12", "macro_release")
    assert ts == "2024-01-12T13:00:00Z"
    # And the FOMC slot is unchanged.
    fomc_ts = edb._as_of_for_event("2024-01-12", "statement")
    assert fomc_ts == "2024-01-12T19:00:00Z"


def test_allowed_event_kind_schema_admits_macro_release() -> None:
    """The pandera ``EventRowSchema`` rejects any unknown event_kind;
    augmented rows must pass the validator at parquet-write time."""

    from app.data.schemas import _ALLOWED_EVENT_KIND

    assert "macro_release" in _ALLOWED_EVENT_KIND
