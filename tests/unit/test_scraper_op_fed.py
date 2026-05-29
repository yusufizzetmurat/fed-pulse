"""Frozen-fixture round-trip tests for the Op-Fed external corpus adapter."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

import pytest

pytest.importorskip("bs4")  # source registry init imports HTML scrapers

from app.data.sources import SOURCES  # noqa: E402
from app.data.sources.op_fed import OpFedScraper  # noqa: E402


FIXTURE_CSV = dedent(
    """\
    unique_id,sentence,speaker,1_opinion,2_mp,3_mp_context,4_stance_nli,5_stance_nli_context
    19770315_alpha_001,"Inflation remains elevated and the committee will act to restore price stability.",Burns,no,yes,context_a,entailment,ctx
    19770315_alpha_002,"Conditions in financial markets argue for accommodation.",Burns,no,yes,context_b,contradiction,ctx2
    19770315_alpha_003,"The committee agreed to maintain the current stance.",Volcker,no,no,context_c,neutral,
    """
)


@pytest.fixture
def frozen_op_fed_csv(tmp_path: Path) -> str:
    return FIXTURE_CSV


def test_op_fed_scraper_is_registered() -> None:
    assert "fomc_meeting_transcript" in SOURCES
    scraper = SOURCES["fomc_meeting_transcript"]
    assert isinstance(scraper, OpFedScraper)
    assert scraper.metadata.provenance.value == "peer_reviewed"
    assert scraper.metadata.name.startswith("Op-Fed")


def test_fetch_listing_returns_one_dict_per_row(frozen_op_fed_csv: str) -> None:
    scraper = OpFedScraper()
    listing = scraper.fetch_listing(frozen_op_fed_csv)
    assert len(listing) == 3
    assert listing[0]["unique_id"] == "19770315_alpha_001"
    assert listing[2]["speaker"] == "Volcker"


def test_parse_entry_round_trips_three_rows(frozen_op_fed_csv: str) -> None:
    scraper = OpFedScraper()
    listing = scraper.fetch_listing(frozen_op_fed_csv)
    parsed = [
        scraper.parse_entry(json.dumps(row), source_url="https://op-fed.test/v1.csv")
        for row in listing
    ]
    assert all(entry is not None for entry in parsed)
    assert parsed[0]["label"] == "hawkish"  # entailment → hawkish
    assert parsed[1]["label"] == "dovish"   # contradiction → dovish
    assert parsed[2]["label"] == "neutral"
    assert parsed[0]["document_type"] == "meeting_transcript"
    assert parsed[0]["license_scope"] == "mit"
    assert parsed[0]["citation_ref"] == "keith_etal_2025_op_fed"
    assert parsed[0]["source_url"] == "https://op-fed.test/v1.csv"
    # multi-axis extras drop empty values
    assert "op_fed_stance_nli_context" not in parsed[2]["multi_axis_extras"]
    assert parsed[0]["multi_axis_extras"]["op_fed_stance_nli"] == "entailment"


def test_parse_entry_drops_rows_missing_text_or_id() -> None:
    scraper = OpFedScraper()
    assert scraper.parse_entry(json.dumps({}), source_url="x") is None
    assert (
        scraper.parse_entry(
            json.dumps({"unique_id": "20240101_x", "sentence": ""}),
            source_url="x",
        )
        is None
    )
    # Garbage input doesn't crash the scraper.
    assert scraper.parse_entry("not-valid-json", source_url="x") is None


def test_write_serialises_parsed_to_jsonl(
    frozen_op_fed_csv: str, tmp_path: Path
) -> None:
    scraper = OpFedScraper()
    listing = scraper.fetch_listing(frozen_op_fed_csv)
    parsed = [
        scraper.parse_entry(json.dumps(row), source_url="https://op-fed.test/v1.csv")
        for row in listing
    ]
    output = tmp_path / "op_fed.jsonl"
    count = scraper.write(parsed, output)
    assert count == 3
    lines = output.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    first = json.loads(lines[0])
    assert first["source_record_id"] == "19770315_alpha_001"
    assert first["label"] == "hawkish"
    assert first["text"].startswith("Inflation remains elevated")


def test_write_skips_none_entries(tmp_path: Path) -> None:
    scraper = OpFedScraper()
    output = tmp_path / "op_fed.jsonl"
    count = scraper.write([None, {"x": 1}, None], output)
    assert count == 1
    assert output.read_text(encoding="utf-8").strip() == json.dumps({"x": 1})


def test_fetch_listing_returns_empty_on_empty_string() -> None:
    scraper = OpFedScraper()
    assert scraper.fetch_listing("") == []


# ----- pull_op_fed_csv -----


from unittest.mock import patch  # noqa: E402

from app.data.sources.op_fed import OP_FED_UPSTREAM_URL, pull_op_fed_csv  # noqa: E402


class _FakeResponse:
    def __init__(self, body: bytes, status: int = 200) -> None:
        self._body = body
        self.status = status

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        return None


def test_pull_downloads_and_writes_csv(tmp_path: Path) -> None:
    target = tmp_path / "external" / "op_fed" / "opfed_v1.csv"
    body = FIXTURE_CSV.encode("utf-8")
    with patch(
        "app.data.sources.op_fed.urllib.request.urlopen",
        return_value=_FakeResponse(body),
    ) as opener:
        rows = pull_op_fed_csv(target)
    assert rows == 3
    assert target.exists()
    assert target.read_bytes() == body
    # URL passed through default
    opener.assert_called_once()
    assert opener.call_args.args[0] == OP_FED_UPSTREAM_URL


def test_pull_is_idempotent_when_cache_exists(tmp_path: Path) -> None:
    target = tmp_path / "opfed_v1.csv"
    target.write_text(FIXTURE_CSV, encoding="utf-8")
    with patch("app.data.sources.op_fed.urllib.request.urlopen") as opener:
        rows = pull_op_fed_csv(target)
    assert rows == 3
    opener.assert_not_called()


def test_pull_force_re_downloads_over_existing_cache(tmp_path: Path) -> None:
    target = tmp_path / "opfed_v1.csv"
    target.write_text("stale,placeholder\n1,2\n", encoding="utf-8")
    body = FIXTURE_CSV.encode("utf-8")
    with patch(
        "app.data.sources.op_fed.urllib.request.urlopen",
        return_value=_FakeResponse(body),
    ):
        rows = pull_op_fed_csv(target, force=True)
    assert rows == 3
    assert target.read_bytes() == body


def test_pull_raises_on_zero_row_download(tmp_path: Path) -> None:
    target = tmp_path / "opfed_v1.csv"
    # Header-only CSV → DictReader sees zero rows.
    body = b"unique_id,sentence\n"
    with patch(
        "app.data.sources.op_fed.urllib.request.urlopen",
        return_value=_FakeResponse(body),
    ):
        with pytest.raises(RuntimeError, match="zero rows"):
            pull_op_fed_csv(target)
    assert not target.exists()
    assert not target.with_suffix(target.suffix + ".tmp").exists()


def test_pull_raises_on_http_error(tmp_path: Path) -> None:
    target = tmp_path / "opfed_v1.csv"
    with patch(
        "app.data.sources.op_fed.urllib.request.urlopen",
        return_value=_FakeResponse(b"", status=404),
    ):
        with pytest.raises(RuntimeError, match="HTTP 404"):
            pull_op_fed_csv(target)
    assert not target.exists()
    assert not target.with_suffix(target.suffix + ".tmp").exists()
