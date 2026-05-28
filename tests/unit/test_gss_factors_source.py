"""Frozen-fixture round-trip tests for the GSS factor-decomposition adapter."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from textwrap import dedent

import pytest

pytest.importorskip("bs4")  # source registry init imports HTML scrapers

from app.data.sources import SOURCES  # noqa: E402
from app.data.sources.gss import GssFactorsScraper  # noqa: E402
from app.data.sources import gss as gss_module  # noqa: E402


FIXTURE_FACTORS_CSV = dedent(
    """\
    meeting_date,target_factor,path_factor,fomc_statement
    1994-08-16,10.7,-8.3,T
    2001-01-03,-32.3,22.8,T
    1990-02-08,0.3,5.8,
    """
)

FIXTURE_SURPRISES_CSV = dedent(
    """\
    meeting_date,surprise_30min_bp,surprise_1hour_bp,surprise_1day_bp,diff_wide_minus_tight,diff_daily_minus_tight
    2001-01-03,-39.3,-36.5,-38.2,1.1,2.8
    """
)


def test_gss_scraper_is_registered() -> None:
    assert "gss_factor_decomposition" in SOURCES
    scraper = SOURCES["gss_factor_decomposition"]
    assert isinstance(scraper, GssFactorsScraper)
    assert scraper.metadata.provenance.value == "peer_reviewed"
    assert "GSS" in scraper.metadata.name


def test_fetch_listing_returns_one_dict_per_meeting() -> None:
    scraper = GssFactorsScraper()
    listing = scraper.fetch_listing(FIXTURE_FACTORS_CSV)
    assert len(listing) == 3
    assert listing[0]["meeting_date"] == "1994-08-16"
    assert listing[2]["fomc_statement"] == ""


def test_fetch_listing_returns_empty_on_empty_string() -> None:
    assert GssFactorsScraper().fetch_listing("") == []


def test_parse_entry_round_trips_three_rows_with_surprises() -> None:
    scraper = GssFactorsScraper(surprises_csv_text=FIXTURE_SURPRISES_CSV)
    listing = scraper.fetch_listing(FIXTURE_FACTORS_CSV)
    parsed = [
        scraper.parse_entry(json.dumps(row), source_url="https://gss.test/factors.csv")
        for row in listing
    ]
    assert all(entry is not None for entry in parsed)
    by_date = {entry["event_date_hint"]: entry for entry in parsed}

    # Factor axis is continuous — no categorical label.
    assert all(entry["label"] == "" for entry in parsed)
    assert all(entry["license_scope"] == "research_only" for entry in parsed)
    assert all(
        entry["citation_ref"] == "gurkaynak_sack_swanson_2005_ijcb" for entry in parsed
    )

    enriched = by_date["2001-01-03"]
    extras = enriched["multi_axis_extras"]
    assert extras["gss_target_factor"] == pytest.approx(-32.3)
    assert extras["gss_path_factor"] == pytest.approx(22.8)
    assert extras["gss_fomc_statement"] is True
    assert extras["surprise_30min_bp"] == pytest.approx(-39.3)
    assert extras["diff_daily_minus_tight"] == pytest.approx(2.8)

    bare = by_date["1990-02-08"]
    assert bare["multi_axis_extras"]["gss_fomc_statement"] is False
    assert "surprise_30min_bp" not in bare["multi_axis_extras"]
    assert "GSS factor decomposition for 1990-02-08" in bare["text"]
    assert bare["source_url"] == "https://gss.test/factors.csv"


def test_parse_entry_without_surprises_table() -> None:
    scraper = GssFactorsScraper()
    listing = scraper.fetch_listing(FIXTURE_FACTORS_CSV)
    parsed = [scraper.parse_entry(json.dumps(row), source_url="x") for row in listing]
    assert all(entry is not None for entry in parsed)
    extras = parsed[0]["multi_axis_extras"]
    assert "surprise_30min_bp" not in extras
    assert extras["gss_target_factor"] == pytest.approx(10.7)


def test_parse_entry_drops_rows_missing_date_or_factors() -> None:
    scraper = GssFactorsScraper()
    assert scraper.parse_entry(json.dumps({}), source_url="x") is None
    # Date but no factor values.
    assert (
        scraper.parse_entry(
            json.dumps({"meeting_date": "2001-01-03", "target_factor": "", "path_factor": ""}),
            source_url="x",
        )
        is None
    )
    # Garbage input doesn't crash the scraper.
    assert scraper.parse_entry("not-valid-json", source_url="x") is None
    # Non-dict JSON.
    assert scraper.parse_entry("[1, 2, 3]", source_url="x") is None


def test_parse_entry_tolerates_unparseable_factor_values() -> None:
    scraper = GssFactorsScraper()
    row = {"meeting_date": "2001-01-03", "target_factor": "n/a", "path_factor": "5.0"}
    parsed = scraper.parse_entry(json.dumps(row), source_url="x")
    assert parsed is not None
    extras = parsed["multi_axis_extras"]
    assert extras["gss_target_factor"] is None
    assert extras["gss_path_factor"] == pytest.approx(5.0)


def test_write_serialises_parsed_to_jsonl(tmp_path: Path) -> None:
    scraper = GssFactorsScraper(surprises_csv_text=FIXTURE_SURPRISES_CSV)
    listing = scraper.fetch_listing(FIXTURE_FACTORS_CSV)
    parsed = [
        scraper.parse_entry(json.dumps(row), source_url="https://gss.test/factors.csv")
        for row in listing
    ]
    output = tmp_path / "gss_factors.jsonl"
    count = scraper.write(parsed, output)
    assert count == 3
    lines = output.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    by_date = {json.loads(line)["event_date_hint"]: json.loads(line) for line in lines}
    assert by_date["2001-01-03"]["multi_axis_extras"]["surprise_30min_bp"] == pytest.approx(-39.3)


def test_write_skips_none_entries(tmp_path: Path) -> None:
    scraper = GssFactorsScraper()
    output = tmp_path / "gss_factors.jsonl"
    count = scraper.write([None, {"x": 1}, None], output)
    assert count == 1
    assert output.read_text(encoding="utf-8").strip() == json.dumps({"x": 1})


# ----- log guards + collision guard (issue #433) ----------------------------


def test_fetch_listing_debug_logs_when_non_empty_input_yields_zero_rows(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Header-only CSV: csv.DictReader gives zero rows but the input is
    # non-empty. The previous adapter swallowed this silently.
    scraper = GssFactorsScraper()
    with caplog.at_level(logging.DEBUG, logger=gss_module.logger.name):
        result = scraper.fetch_listing("meeting_date,target_factor,path_factor\n")
    assert result == []
    assert any("factors CSV parsed to zero rows" in rec.message for rec in caplog.records)


def test_parse_surprises_csv_debug_logs_when_non_empty_yields_zero_rows(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Wrong date column name -> every row drops on the date-missing check.
    bad = dedent(
        """\
        wrong_date_column,surprise_30min_bp
        2001-01-03,-39.3
        """
    )
    with caplog.at_level(logging.DEBUG, logger=gss_module.logger.name):
        result = gss_module._parse_surprises_csv(bad)
    assert result == {}
    assert any("surprises CSV parsed to zero rows" in rec.message for rec in caplog.records)


def test_parse_surprises_csv_does_not_log_when_input_empty(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.DEBUG, logger=gss_module.logger.name):
        result = gss_module._parse_surprises_csv("")
    assert result == {}
    # Empty input is the no-op path, not a parse failure — must not log.
    assert not any(
        "surprises CSV parsed to zero rows" in rec.message for rec in caplog.records
    )


def test_side_table_merge_does_not_overwrite_factor_keys_on_collision(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Synthesise a colliding surprises table: a (hypothetical, future)
    # column named identically to a factor key. The merge must keep the
    # factor value and warn.
    scraper = GssFactorsScraper()
    scraper._surprises_by_date = {
        "2001-01-03": {"gss_target_factor": 999.0, "surprise_30min_bp": -1.0},
    }
    row = {"meeting_date": "2001-01-03", "target_factor": "-32.3", "path_factor": "22.8"}
    with caplog.at_level(logging.WARNING, logger=gss_module.logger.name):
        parsed = scraper.parse_entry(json.dumps(row), source_url="x")
    assert parsed is not None
    extras = parsed["multi_axis_extras"]
    # Factor value preserved, not overwritten by the colliding side-table key.
    assert extras["gss_target_factor"] == pytest.approx(-32.3)
    # Non-colliding side-table column still merged through.
    assert extras["surprise_30min_bp"] == pytest.approx(-1.0)
    assert any("collides with factor key" in rec.message for rec in caplog.records)
