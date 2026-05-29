from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

import urllib.error
from unittest.mock import patch

from app.services.scraper_beige_book import (
    ARCHIVE_LISTING_URL,
    BeigeBookListingEntry,
    ParsedBeigeBook,
    extract_beige_book_listing,
    parse_beige_book_page,
    pull_beige_book_archive,
    write_beige_book_json,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def test_extract_beige_book_listing_returns_entries() -> None:
    """The listing has both base and summary URLs; adapter returns deduplicated entries."""

    html = (FIXTURES / "fed_beige_book_listing.html").read_text(encoding="utf-8")
    entries = extract_beige_book_listing(html)

    assert len(entries) >= 5
    for entry in entries:
        assert isinstance(entry, BeigeBookListingEntry)
        assert entry.url.startswith("https://www.federalreserve.gov/monetarypolicy/beigebook")
        assert entry.url.endswith(".htm")
        assert entry.date  # ISO yyyy-mm-dd derived from URL


def test_extract_beige_book_listing_returns_empty_on_unrelated_html() -> None:
    assert extract_beige_book_listing("<html><body>nothing</body></html>") == []


def test_extract_beige_book_listing_deduplicates_repeated_urls() -> None:
    repeated = "/monetarypolicy/beigebook202401-summary.htm"
    html = f'<html><body><a href="{repeated}">A</a><a href="{repeated}">B</a></body></html>'
    entries = extract_beige_book_listing(html)
    assert len(entries) == 1


def test_parse_beige_book_page_extracts_substantive_text() -> None:
    html = (FIXTURES / "fed_beige_book_sample.html").read_text(encoding="utf-8")
    # The sample fixture is the January 2026 National Summary
    source_url = "https://www.federalreserve.gov/monetarypolicy/beigebook202601-summary.htm"

    parsed = parse_beige_book_page(html, source_url=source_url)

    assert isinstance(parsed, ParsedBeigeBook)
    assert parsed.date.startswith("20")
    # Full Beige Book national summary is substantial — at least 5k chars of economic content
    assert len(parsed.text) > 5000
    assert parsed.url == source_url


def test_write_beige_book_json_emits_one_row_per_parsed(tmp_path: Path) -> None:
    parsed = [
        ParsedBeigeBook(
            date="2024-03-01",
            title="Beige Book - March 2024",
            text="Full body of the report " * 50,
            url="https://www.federalreserve.gov/monetarypolicy/beigebook202403-summary.htm",
        ),
        ParsedBeigeBook(
            date="",
            title="Empty date",
            text="something",
            url="https://www.federalreserve.gov/monetarypolicy/beigebook202404-summary.htm",
        ),
    ]

    output = tmp_path / "beige_book.json"
    written = write_beige_book_json(parsed, output)

    assert written == 1  # empty-date row is skipped
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert len(payload) == 1
    assert payload[0]["document_type"] == "beige_book"


# ----- pull_beige_book_archive -----


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


_LISTING_HTML = """
<html><body>
<a href="/monetarypolicy/beigebook202401-summary.htm">January 2024</a>
<a href="/monetarypolicy/beigebook202403-summary.htm">March 2024</a>
</body></html>
"""

_PAGE_HTML_TEMPLATE = """
<html><head><meta property="og:title" content="Beige Book {month}"></head>
<body><div id="article">
<p>{filler}</p>
<p>{filler2}</p>
<p>{filler3}</p>
</div></body></html>
"""


def _fake_page(month: str) -> str:
    filler = "Substantive regional summary " * 20
    filler2 = "Manufacturing activity expanded modestly across most districts " * 6
    filler3 = "Labor markets remained tight with wage pressures elevated " * 6
    return _PAGE_HTML_TEMPLATE.format(
        month=month, filler=filler, filler2=filler2, filler3=filler3
    )


def _route_urlopen(url, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
    url_str = url if isinstance(url, str) else url.full_url
    if url_str == ARCHIVE_LISTING_URL:
        return _FakeResponse(_LISTING_HTML.encode("utf-8"))
    if "202401" in url_str:
        return _FakeResponse(_fake_page("January 2024").encode("utf-8"))
    if "202403" in url_str:
        return _FakeResponse(_fake_page("March 2024").encode("utf-8"))
    raise AssertionError(f"unexpected URL fetched: {url_str}")


def test_pull_walks_listing_and_writes_json(tmp_path: Path) -> None:
    target = tmp_path / "beige_book.json"
    with patch(
        "app.services.scraper_beige_book.urllib.request.urlopen",
        side_effect=_route_urlopen,
    ) as opener:
        rows = pull_beige_book_archive(target, delay_seconds=0.0)
    assert rows == 2
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert [r["date"] for r in payload] == ["2024-01-01", "2024-03-01"]
    # 1 listing fetch + 2 page fetches
    assert opener.call_count == 3


def test_pull_is_idempotent_when_cache_has_rows(tmp_path: Path) -> None:
    target = tmp_path / "beige_book.json"
    target.write_text(
        json.dumps([{"date": "2024-01-01", "text": "x", "document_type": "beige_book"}]),
        encoding="utf-8",
    )
    with patch("app.services.scraper_beige_book.urllib.request.urlopen") as opener:
        rows = pull_beige_book_archive(target, delay_seconds=0.0)
    assert rows == 1
    opener.assert_not_called()


def test_pull_force_re_walks_over_existing_cache(tmp_path: Path) -> None:
    target = tmp_path / "beige_book.json"
    target.write_text(json.dumps([{"date": "stale", "text": "x"}]), encoding="utf-8")
    with patch(
        "app.services.scraper_beige_book.urllib.request.urlopen",
        side_effect=_route_urlopen,
    ):
        rows = pull_beige_book_archive(target, force=True, delay_seconds=0.0)
    assert rows == 2


def test_pull_repulls_when_cache_is_empty_list(tmp_path: Path) -> None:
    target = tmp_path / "beige_book.json"
    target.write_text("[]", encoding="utf-8")
    with patch(
        "app.services.scraper_beige_book.urllib.request.urlopen",
        side_effect=_route_urlopen,
    ) as opener:
        rows = pull_beige_book_archive(target, delay_seconds=0.0)
    assert rows == 2
    assert opener.call_count == 3


def test_pull_limit_caps_walk(tmp_path: Path) -> None:
    target = tmp_path / "beige_book.json"
    with patch(
        "app.services.scraper_beige_book.urllib.request.urlopen",
        side_effect=_route_urlopen,
    ) as opener:
        rows = pull_beige_book_archive(target, limit=1, delay_seconds=0.0)
    assert rows == 1
    # 1 listing + 1 page = 2 fetches
    assert opener.call_count == 2


def test_pull_continues_when_one_page_404s(tmp_path: Path) -> None:
    target = tmp_path / "beige_book.json"

    def _route_with_one_404(url, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        url_str = url if isinstance(url, str) else url.full_url
        if "202401" in url_str:
            raise urllib.error.HTTPError(url_str, 404, "Not Found", None, None)  # type: ignore[arg-type]
        return _route_urlopen(url, *args, **kwargs)

    with patch(
        "app.services.scraper_beige_book.urllib.request.urlopen",
        side_effect=_route_with_one_404,
    ):
        with pytest.warns(UserWarning, match="Beige Book fetch failed"):
            rows = pull_beige_book_archive(target, delay_seconds=0.0)
    assert rows == 1  # only March survived


def test_pull_raises_on_listing_http_error(tmp_path: Path) -> None:
    target = tmp_path / "beige_book.json"
    err = urllib.error.HTTPError(ARCHIVE_LISTING_URL, 503, "boom", None, None)  # type: ignore[arg-type]
    with patch(
        "app.services.scraper_beige_book.urllib.request.urlopen",
        side_effect=err,
    ):
        with pytest.raises(RuntimeError, match="HTTP 503"):
            pull_beige_book_archive(target, delay_seconds=0.0)
    assert not target.exists()
    assert not target.with_suffix(target.suffix + ".tmp").exists()


def test_pull_raises_when_every_page_fails(tmp_path: Path) -> None:
    target = tmp_path / "beige_book.json"

    def _route_all_fail(url, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        url_str = url if isinstance(url, str) else url.full_url
        if url_str == ARCHIVE_LISTING_URL:
            return _FakeResponse(_LISTING_HTML.encode("utf-8"))
        raise urllib.error.HTTPError(url_str, 500, "x", None, None)  # type: ignore[arg-type]

    with patch(
        "app.services.scraper_beige_book.urllib.request.urlopen",
        side_effect=_route_all_fail,
    ):
        with pytest.warns(UserWarning):
            with pytest.raises(RuntimeError, match="zero rows"):
                pull_beige_book_archive(target, delay_seconds=0.0)
    assert not target.exists()
    assert not target.with_suffix(target.suffix + ".tmp").exists()
