from __future__ import annotations

import json
import urllib.error
from pathlib import Path
from unittest.mock import patch

import pytest

from app.services.scraper_testimonies import (
    TestimonyListingEntry,
    ParsedTestimony,
    extract_testimony_listing,
    parse_testimony_page,
    pull_testimonies_archive,
    write_testimonies_json,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def test_extract_testimony_listing_returns_entries_from_real_archive_page() -> None:
    html = (FIXTURES / "fed_testimony_archive.html").read_text(encoding="utf-8")

    entries = extract_testimony_listing(html)

    # Archive should list at least a few testimonies
    assert len(entries) >= 3
    for entry in entries:
        assert isinstance(entry, TestimonyListingEntry)
        assert entry.url.startswith("https://www.federalreserve.gov/newsevents/testimony/")
        assert entry.url.endswith(".htm")
        assert entry.date  # ISO yyyy-mm-dd
        assert entry.title


def test_extract_testimony_listing_returns_empty_on_unrelated_html() -> None:
    assert extract_testimony_listing("<html><body>nothing here</body></html>") == []


def test_extract_testimony_listing_deduplicates_repeated_urls() -> None:
    repeated_url = "/newsevents/testimony/powell20240131a.htm"
    html = f'<html><body><a href="{repeated_url}">A</a><a href="{repeated_url}">B</a></body></html>'
    entries = extract_testimony_listing(html)
    assert len(entries) == 1
    assert entries[0].url.endswith(repeated_url)


def test_parse_testimony_page_extracts_speaker_date_and_body() -> None:
    html = (FIXTURES / "fed_testimony_sample.html").read_text(encoding="utf-8")
    # Use any testimony URL — the actual fixture's URL is what matters for date inference
    # Pick one that grep'd from the archive
    archive_html = (FIXTURES / "fed_testimony_archive.html").read_text(encoding="utf-8")
    import re
    match = re.search(r'/newsevents/testimony/[a-z]+[0-9]{8}[a-z]\.htm', archive_html)
    assert match, "No testimony URL found in archive fixture"
    source_url = "https://www.federalreserve.gov" + match.group(0)

    parsed = parse_testimony_page(html, source_url=source_url)

    assert isinstance(parsed, ParsedTestimony)
    assert parsed.speaker  # non-empty
    assert parsed.date.startswith("20")  # ISO date
    assert len(parsed.text) > 200
    assert parsed.title
    assert parsed.url == source_url


def test_write_testimonies_json_emits_one_row_per_parsed(tmp_path: Path) -> None:
    parsed = [
        ParsedTestimony(
            date="2024-03-06",
            speaker="Chair Powell",
            title="Semiannual Monetary Policy Report",
            text="Full body of the testimony " * 30,
            url="https://www.federalreserve.gov/newsevents/testimony/powell20240306a.htm",
        ),
        ParsedTestimony(
            date="",
            speaker="Governor Waller",
            title="Empty date",
            text="something",
            url="https://www.federalreserve.gov/newsevents/testimony/waller20240501a.htm",
        ),
    ]

    output = tmp_path / "congressional_testimonies.json"
    written = write_testimonies_json(parsed, output)

    assert written == 1  # the empty-date row is skipped
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert len(payload) == 1
    assert payload[0]["document_type"] == "congressional_testimony"
    assert payload[0]["title"] == "Semiannual Monetary Policy Report"


# ----- pull_testimonies_archive -----


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
<a href="/newsevents/testimony/powell20240306a.htm">Semiannual Monetary Policy Report</a>
<a href="/newsevents/testimony/waller20240501a.htm">Banking Conditions</a>
</body></html>
"""

_PAGE_HTML_TEMPLATE = """
<html><head><meta property="og:title" content="Testimony - {speaker} - {label}"></head>
<body><div id="article">
<p class="speaker">{speaker}</p>
<p>{filler}</p>
<p>{filler2}</p>
<p>{filler3}</p>
</div></body></html>
"""


def _fake_page(speaker: str, label: str) -> str:
    filler = "Substantive testimony paragraph on monetary policy and the economy " * 8
    filler2 = "Labor market conditions remain resilient and consistent with the dual mandate " * 6
    filler3 = "Inflation has eased materially over the past year toward the two percent objective " * 6
    return _PAGE_HTML_TEMPLATE.format(
        speaker=speaker, label=label, filler=filler, filler2=filler2, filler3=filler3
    )


_SINGLE_ARCHIVE_URL = "https://www.federalreserve.gov/newsevents/testimony/2024-testimony.htm"


def _route_urlopen(url, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
    url_str = url if isinstance(url, str) else url.full_url
    if url_str == _SINGLE_ARCHIVE_URL:
        return _FakeResponse(_LISTING_HTML.encode("utf-8"))
    if "powell20240306" in url_str:
        return _FakeResponse(_fake_page("Chair Powell", "March 2024").encode("utf-8"))
    if "waller20240501" in url_str:
        return _FakeResponse(_fake_page("Governor Waller", "May 2024").encode("utf-8"))
    raise AssertionError(f"unexpected URL fetched: {url_str}")


def test_pull_walks_listing_and_writes_json(tmp_path: Path) -> None:
    target = tmp_path / "congressional_testimonies.json"
    with patch(
        "app.services.scraper_testimonies.urllib.request.urlopen",
        side_effect=_route_urlopen,
    ) as opener:
        rows = pull_testimonies_archive(
            target, archive_url=_SINGLE_ARCHIVE_URL, delay_seconds=0.0
        )
    assert rows == 2
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert [r["date"] for r in payload] == ["2024-03-06", "2024-05-01"]
    assert opener.call_count == 3


def test_pull_is_idempotent_when_cache_has_rows(tmp_path: Path) -> None:
    target = tmp_path / "congressional_testimonies.json"
    target.write_text(
        json.dumps(
            [
                {
                    "date": "2024-03-06",
                    "text": "x",
                    "document_type": "congressional_testimony",
                }
            ]
        ),
        encoding="utf-8",
    )
    with patch("app.services.scraper_testimonies.urllib.request.urlopen") as opener:
        rows = pull_testimonies_archive(
            target, archive_url=_SINGLE_ARCHIVE_URL, delay_seconds=0.0
        )
    assert rows == 1
    opener.assert_not_called()


def test_pull_force_re_walks_over_existing_cache(tmp_path: Path) -> None:
    target = tmp_path / "congressional_testimonies.json"
    target.write_text(json.dumps([{"date": "stale", "text": "x"}]), encoding="utf-8")
    with patch(
        "app.services.scraper_testimonies.urllib.request.urlopen",
        side_effect=_route_urlopen,
    ):
        rows = pull_testimonies_archive(
            target,
            archive_url=_SINGLE_ARCHIVE_URL,
            force=True,
            delay_seconds=0.0,
        )
    assert rows == 2


def test_pull_continues_when_one_page_404s(tmp_path: Path) -> None:
    target = tmp_path / "congressional_testimonies.json"

    def _route_with_one_404(url, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        url_str = url if isinstance(url, str) else url.full_url
        if "powell20240306" in url_str:
            raise urllib.error.HTTPError(url_str, 404, "Not Found", None, None)  # type: ignore[arg-type]
        return _route_urlopen(url, *args, **kwargs)

    with patch(
        "app.services.scraper_testimonies.urllib.request.urlopen",
        side_effect=_route_with_one_404,
    ):
        with pytest.warns(UserWarning, match="Testimony fetch failed"):
            rows = pull_testimonies_archive(
                target, archive_url=_SINGLE_ARCHIVE_URL, delay_seconds=0.0
            )
    assert rows == 1  # only the May testimony survived


def test_pull_raises_when_every_page_fails(tmp_path: Path) -> None:
    target = tmp_path / "congressional_testimonies.json"

    def _route_all_fail(url, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        url_str = url if isinstance(url, str) else url.full_url
        if url_str == _SINGLE_ARCHIVE_URL:
            return _FakeResponse(_LISTING_HTML.encode("utf-8"))
        raise urllib.error.HTTPError(url_str, 500, "x", None, None)  # type: ignore[arg-type]

    with patch(
        "app.services.scraper_testimonies.urllib.request.urlopen",
        side_effect=_route_all_fail,
    ):
        with pytest.warns(UserWarning):
            with pytest.raises(RuntimeError, match="zero rows"):
                pull_testimonies_archive(
                    target, archive_url=_SINGLE_ARCHIVE_URL, delay_seconds=0.0
                )
    assert not target.exists()
    assert not target.with_suffix(target.suffix + ".tmp").exists()


def test_pull_warns_and_continues_when_one_year_listing_404s(tmp_path: Path) -> None:
    target = tmp_path / "congressional_testimonies.json"
    good_listing = "https://www.federalreserve.gov/newsevents/testimony/2024-testimony.htm"
    bad_listing = "https://www.federalreserve.gov/newsevents/testimony/2099-testimony.htm"

    def _route_with_bad_year(url, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        url_str = url if isinstance(url, str) else url.full_url
        if url_str == bad_listing:
            raise urllib.error.HTTPError(url_str, 404, "Not Found", None, None)  # type: ignore[arg-type]
        if url_str == good_listing:
            return _FakeResponse(_LISTING_HTML.encode("utf-8"))
        if "powell20240306" in url_str:
            return _FakeResponse(
                _fake_page("Chair Powell", "March 2024").encode("utf-8")
            )
        if "waller20240501" in url_str:
            return _FakeResponse(
                _fake_page("Governor Waller", "May 2024").encode("utf-8")
            )
        raise AssertionError(f"unexpected URL fetched: {url_str}")

    with patch(
        "app.services.scraper_testimonies.urllib.request.urlopen",
        side_effect=_route_with_bad_year,
    ):
        with pytest.warns(UserWarning, match="Testimony listing fetch failed"):
            rows = pull_testimonies_archive(
                target, years=[2024, 2099], delay_seconds=0.0
            )
    assert rows == 2
