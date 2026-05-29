from __future__ import annotations

import json
import re
from pathlib import Path
from unittest.mock import patch

import pytest
import requests

from app.services.scraper_press_conferences import (
    ARCHIVE_LISTING_URL,
    PressConferenceListingEntry,
    ParsedPressConference,
    build_qa_lookup,
    extract_press_conference_listing,
    parse_press_conference_page,
    pull_press_conferences_archive,
    split_prepared_remarks_and_qa,
    write_press_conferences_json,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def test_extract_press_conference_listing_returns_entries_from_calendar() -> None:
    html = (FIXTURES / "fed_fomc_calendar.html").read_text(encoding="utf-8")

    entries = extract_press_conference_listing(html)

    # Calendar typically lists 8 per year for the past 2-3 years; expect at least 5 total
    assert len(entries) >= 5
    for entry in entries:
        assert isinstance(entry, PressConferenceListingEntry)
        assert entry.url.startswith("https://www.federalreserve.gov/monetarypolicy/fomcpresconf")
        assert entry.url.endswith(".htm")
        assert entry.date  # ISO yyyy-mm-dd derived from URL


def test_extract_press_conference_listing_returns_empty_on_unrelated_html() -> None:
    assert extract_press_conference_listing("<html><body>nothing</body></html>") == []


def test_extract_press_conference_listing_deduplicates_repeated_urls() -> None:
    repeated = "/monetarypolicy/fomcpresconf20240131.htm"
    html = f'<html><body><a href="{repeated}">A</a><a href="{repeated}">B</a></body></html>'
    entries = extract_press_conference_listing(html)
    assert len(entries) == 1


def test_parse_press_conference_page_extracts_transcript_from_pdf(tmp_path: Path, monkeypatch) -> None:
    """The press conference HTML page is a video-only landing; the
    transcript lives in a sibling PDF. parse_press_conference_page
    must download and extract the PDF text."""

    sample_html = (FIXTURES / "fed_press_conference_sample.html").read_text(encoding="utf-8")
    pdf_bytes = (FIXTURES / "fed_press_conference_sample.pdf").read_bytes()

    calendar_html = (FIXTURES / "fed_fomc_calendar.html").read_text(encoding="utf-8")
    match = re.search(r'/monetarypolicy/fomcpresconf202[45][0-9]{4}\.htm', calendar_html)
    assert match
    source_url = "https://www.federalreserve.gov" + match.group(0)

    # Stub the PDF download
    class _StubResponse:
        def __init__(self, content):
            self.content = content
            self.status_code = 200

        def raise_for_status(self):
            pass

    def fake_get(url, *args, **kwargs):
        # Verify the PDF URL is constructed correctly
        assert "FOMCpresconf" in url
        assert url.endswith(".pdf")
        return _StubResponse(pdf_bytes)

    monkeypatch.setattr("app.services.scraper_press_conferences.requests.get", fake_get)

    parsed = parse_press_conference_page(sample_html, source_url=source_url)

    assert isinstance(parsed, ParsedPressConference)
    assert parsed.date.startswith("20")
    # Real transcript should be substantial (thousands of chars), not 600 chars of boilerplate
    assert len(parsed.text) > 5000
    assert parsed.url == source_url
    # Powell's prepared remarks should mention the FOMC at minimum
    assert "FOMC" in parsed.text or "Federal" in parsed.text


def test_parse_press_conference_page_falls_back_when_pdf_unavailable(monkeypatch) -> None:
    """If the PDF download fails (404, network), return empty text rather than raising."""

    sample_html = (FIXTURES / "fed_press_conference_sample.html").read_text(encoding="utf-8")
    source_url = "https://www.federalreserve.gov/monetarypolicy/fomcpresconf20240131.htm"

    def fake_get(url, *args, **kwargs):
        raise Exception("simulated network failure")

    monkeypatch.setattr("app.services.scraper_press_conferences.requests.get", fake_get)

    parsed = parse_press_conference_page(sample_html, source_url=source_url)
    assert parsed.text == ""
    assert parsed.date.startswith("2024")


def test_write_press_conferences_json_emits_one_row_per_parsed(tmp_path: Path) -> None:
    parsed = [
        ParsedPressConference(
            date="2024-03-20",
            title="FOMC Press Conference",
            text="Full transcript " * 50,
            url="https://www.federalreserve.gov/monetarypolicy/fomcpresconf20240320.htm",
            prepared_remarks_text="Opening remarks " * 10,
            qa_text="Q. and A. " * 30,
        ),
        ParsedPressConference(
            date="",
            title="missing date",
            text="something",
            url="https://www.federalreserve.gov/monetarypolicy/fomcpresconf20240501.htm",
        ),
    ]

    output = tmp_path / "press_conferences.json"
    written = write_press_conferences_json(parsed, output)

    assert written == 1
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert len(payload) == 1
    assert payload[0]["document_type"] == "press_conference"
    assert payload[0]["date"] == "2024-03-20"
    # #214: Q&A and prepared remarks are persisted alongside the legacy
    # full-transcript text so downstream lookups can address either slice.
    assert "qa_text" in payload[0]
    assert "prepared_remarks_text" in payload[0]
    assert payload[0]["qa_text"].startswith("Q. and A.")


def test_split_prepared_remarks_and_qa_handles_real_transcript() -> None:
    """The real sample PDF must split into a small remarks slice and a
    much larger Q&A slice (Q&A is the high-information portion). The
    boundary anchors on "I look forward to your questions"."""

    from pypdf import PdfReader

    pdf = PdfReader(FIXTURES / "fed_press_conference_sample.pdf")
    text = "\n".join(page.extract_text() or "" for page in pdf.pages)
    prepared, qa = split_prepared_remarks_and_qa(text)

    assert prepared, "prepared remarks must be non-empty on a valid transcript"
    assert qa, "Q&A must be non-empty on a valid transcript"
    # Sanity floor: prepared remarks usually run 5-10% of the transcript;
    # Q&A is the bulk of the text. A real Powell press conference has at
    # least 30k chars of Q&A.
    assert len(qa) > 30_000
    assert len(prepared) < len(qa) // 3
    # The hand-off phrase ends the prepared remarks.
    assert "look forward to your questions" in prepared.lower()
    # Q&A must contain at least one reporter speaker turn.
    assert "STEVE LIESMAN" in qa or "MICHELLE SMITH" in qa


def test_split_prepared_remarks_and_qa_empty_on_missing_text() -> None:
    assert split_prepared_remarks_and_qa("") == ("", "")


def test_split_prepared_remarks_and_qa_returns_remarks_on_missing_boundary() -> None:
    """When neither the hand-off phrase nor a moderator turn appears the
    whole text is returned as prepared remarks and Q&A is empty — the
    caller flips ``has_press_conf`` based on whether ``text`` is
    populated, not on whether Q&A landed."""

    text = "Some short statement. " * 50
    prepared, qa = split_prepared_remarks_and_qa(text)
    assert prepared.strip()
    assert qa == ""


def test_build_qa_lookup_keys_on_event_date() -> None:
    parsed = [
        ParsedPressConference(
            date="2024-03-20",
            title="FOMC Press Conference",
            text="full transcript",
            url="https://www.federalreserve.gov/monetarypolicy/fomcpresconf20240320.htm",
            prepared_remarks_text="opening",
            qa_text="reporter and chair exchange",
        ),
        ParsedPressConference(
            date="2024-05-01",
            title="FOMC Press Conference",
            text="",  # download failed; skipped
            url="https://www.federalreserve.gov/monetarypolicy/fomcpresconf20240501.htm",
        ),
    ]

    lookup = build_qa_lookup(parsed)
    assert set(lookup.keys()) == {"2024-03-20"}
    assert lookup["2024-03-20"]["qa_text"].startswith("reporter and chair")
    assert lookup["2024-03-20"]["has_press_conf"] == "1"


def test_parse_press_conference_page_caches_pdf_to_disk(tmp_path: Path, monkeypatch) -> None:
    """The cache_pdf_dir kwarg must persist the fetched PDF locally so
    a second call short-circuits the network fetch (#214)."""

    sample_html = (FIXTURES / "fed_press_conference_sample.html").read_text(encoding="utf-8")
    pdf_bytes = (FIXTURES / "fed_press_conference_sample.pdf").read_bytes()
    source_url = "https://www.federalreserve.gov/monetarypolicy/fomcpresconf20250129.htm"

    call_count = {"n": 0}

    class _StubResponse:
        def __init__(self, content):
            self.content = content
            self.status_code = 200

        def raise_for_status(self):
            pass

    def fake_get(url, *args, **kwargs):
        call_count["n"] += 1
        return _StubResponse(pdf_bytes)

    monkeypatch.setattr("app.services.scraper_press_conferences.requests.get", fake_get)

    cache_dir = tmp_path / "press_conf_cache"
    parsed_first = parse_press_conference_page(
        sample_html, source_url=source_url, cache_pdf_dir=cache_dir
    )
    parsed_second = parse_press_conference_page(
        sample_html, source_url=source_url, cache_pdf_dir=cache_dir
    )

    assert call_count["n"] == 1  # second call short-circuits to cache
    cached_pdf = cache_dir / "20250129.pdf"
    assert cached_pdf.exists()
    assert cached_pdf.read_bytes() == pdf_bytes
    assert parsed_first.text == parsed_second.text
    # Both calls must produce the same Q&A split.
    assert parsed_first.qa_text == parsed_second.qa_text
    assert parsed_first.qa_text
    assert parsed_first.prepared_remarks_text


# ----- pull_press_conferences_archive -----


class _FakeRequestsResponse:
    def __init__(
        self,
        *,
        text: str = "",
        content: bytes = b"",
        status_code: int = 200,
    ) -> None:
        self.text = text
        self.content = content
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            err = requests.HTTPError(f"HTTP {self.status_code}")
            err.response = self  # type: ignore[assignment]
            raise err


_LISTING_HTML = """
<html><body>
<a href="/monetarypolicy/fomcpresconf20240131.htm">January</a>
<a href="/monetarypolicy/fomcpresconf20240320.htm">March</a>
</body></html>
"""

_PRESS_HTML_TEMPLATE = """
<html><head><meta property="og:title" content="FOMC Press Conference - {label}"></head>
<body><p>See the PDF.</p></body></html>
"""


def _route_requests_get(url, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
    if url == ARCHIVE_LISTING_URL:
        return _FakeRequestsResponse(text=_LISTING_HTML)
    # Suppress historical-year listing fetches (they 404 in the
    # default historical window and are caught by the per-listing
    # try/except in the orchestrator).
    if "fomchistorical" in url:
        return _FakeRequestsResponse(status_code=404)
    if "fomcpresconf20240131.htm" in url:
        return _FakeRequestsResponse(text=_PRESS_HTML_TEMPLATE.format(label="January"))
    if "fomcpresconf20240320.htm" in url:
        return _FakeRequestsResponse(text=_PRESS_HTML_TEMPLATE.format(label="March"))
    raise AssertionError(f"unexpected URL fetched: {url}")


def _stub_pdf_text(_bytes: bytes) -> str:
    return "Prepared remarks. " + "Filler. " * 20


def test_pull_press_walks_listing_and_writes_json(tmp_path: Path) -> None:
    target = tmp_path / "press_conferences.json"
    with patch(
        "app.services.scraper_press_conferences.requests.get",
        side_effect=_route_requests_get,
    ), patch(
        "app.services.scraper_press_conferences._load_or_fetch_pdf_bytes",
        return_value=b"%PDF-1.4 stub",
    ), patch(
        "app.services.scraper_press_conferences._extract_pdf_text",
        side_effect=_stub_pdf_text,
    ):
        rows = pull_press_conferences_archive(
            target,
            historical_years=[],
            delay_seconds=0.0,
        )
    assert rows == 2
    payload = json.loads(target.read_text(encoding="utf-8"))
    # Sorted by date desc: March first, then January
    assert [r["date"] for r in payload] == ["2024-03-20", "2024-01-31"]
    for row in payload:
        assert row["document_type"] == "press_conference"
        assert row["text"]


def test_pull_press_is_idempotent_when_cache_has_rows(tmp_path: Path) -> None:
    target = tmp_path / "press_conferences.json"
    target.write_text(
        json.dumps(
            [{"date": "2024-01-31", "text": "x", "document_type": "press_conference"}]
        ),
        encoding="utf-8",
    )
    with patch(
        "app.services.scraper_press_conferences.requests.get"
    ) as get_mock:
        rows = pull_press_conferences_archive(
            target, historical_years=[], delay_seconds=0.0
        )
    assert rows == 1
    get_mock.assert_not_called()


def test_pull_press_force_re_walks_over_existing_cache(tmp_path: Path) -> None:
    target = tmp_path / "press_conferences.json"
    target.write_text(json.dumps([{"date": "stale", "text": "x"}]), encoding="utf-8")
    with patch(
        "app.services.scraper_press_conferences.requests.get",
        side_effect=_route_requests_get,
    ), patch(
        "app.services.scraper_press_conferences._load_or_fetch_pdf_bytes",
        return_value=b"%PDF-1.4 stub",
    ), patch(
        "app.services.scraper_press_conferences._extract_pdf_text",
        side_effect=_stub_pdf_text,
    ):
        rows = pull_press_conferences_archive(
            target,
            force=True,
            historical_years=[],
            delay_seconds=0.0,
        )
    assert rows == 2


def test_pull_press_continues_when_one_page_404s(tmp_path: Path) -> None:
    target = tmp_path / "press_conferences.json"

    def _route_one_404(url, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        if "fomcpresconf20240131.htm" in url:
            return _FakeRequestsResponse(status_code=404)
        return _route_requests_get(url, *args, **kwargs)

    with patch(
        "app.services.scraper_press_conferences.requests.get",
        side_effect=_route_one_404,
    ), patch(
        "app.services.scraper_press_conferences._load_or_fetch_pdf_bytes",
        return_value=b"%PDF-1.4 stub",
    ), patch(
        "app.services.scraper_press_conferences._extract_pdf_text",
        side_effect=_stub_pdf_text,
    ):
        with pytest.warns(UserWarning, match="Press conference fetch failed"):
            rows = pull_press_conferences_archive(
                target,
                historical_years=[],
                delay_seconds=0.0,
            )
    assert rows == 1  # only March survived


def test_pull_press_raises_on_calendar_listing_http_error(tmp_path: Path) -> None:
    target = tmp_path / "press_conferences.json"

    def _route_calendar_503(url, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        return _FakeRequestsResponse(status_code=503)

    with patch(
        "app.services.scraper_press_conferences.requests.get",
        side_effect=_route_calendar_503,
    ):
        with pytest.warns(UserWarning, match="Press conference listing fetch failed"):
            with pytest.raises(RuntimeError, match="zero rows"):
                pull_press_conferences_archive(
                    target,
                    historical_years=[],
                    delay_seconds=0.0,
                )
    assert not target.exists()
    assert not target.with_suffix(target.suffix + ".tmp").exists()


def test_pull_press_limit_caps_walk(tmp_path: Path) -> None:
    target = tmp_path / "press_conferences.json"
    with patch(
        "app.services.scraper_press_conferences.requests.get",
        side_effect=_route_requests_get,
    ), patch(
        "app.services.scraper_press_conferences._load_or_fetch_pdf_bytes",
        return_value=b"%PDF-1.4 stub",
    ), patch(
        "app.services.scraper_press_conferences._extract_pdf_text",
        side_effect=_stub_pdf_text,
    ):
        rows = pull_press_conferences_archive(
            target,
            historical_years=[],
            limit=1,
            delay_seconds=0.0,
        )
    assert rows == 1
    payload = json.loads(target.read_text(encoding="utf-8"))
    # Most recent first
    assert payload[0]["date"] == "2024-03-20"
