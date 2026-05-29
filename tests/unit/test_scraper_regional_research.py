from __future__ import annotations

import json
import re
import urllib.error
from pathlib import Path
from unittest.mock import patch

import pytest

from app.services.scraper_regional_research import (
    ARCHIVE_LISTING_URL,
    RegionalResearchListingEntry,
    ParsedRegionalResearch,
    extract_lse_listing,
    parse_lse_post,
    pull_regional_research_archive,
    write_regional_research_json,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def test_extract_lse_listing_returns_entries_with_expected_url_pattern() -> None:
    html = (FIXTURES / "lse_listing.html").read_text(encoding="utf-8")
    entries = extract_lse_listing(html)

    assert len(entries) >= 3
    for entry in entries:
        assert isinstance(entry, RegionalResearchListingEntry)
        assert entry.url.startswith("https://libertystreeteconomics.newyorkfed.org/")
        assert entry.date  # ISO yyyy-mm-01 (date precision is month-level from URL)
        assert entry.title or True  # title may be empty; not strictly required


def test_extract_lse_listing_returns_empty_on_unrelated_html() -> None:
    assert extract_lse_listing("<html><body>nothing</body></html>") == []


def test_extract_lse_listing_deduplicates_repeated_urls() -> None:
    repeated = "https://libertystreeteconomics.newyorkfed.org/2024/03/sample-post/"
    html = f'<html><body><a href="{repeated}">A</a><a href="{repeated}">B</a></body></html>'
    entries = extract_lse_listing(html)
    assert len(entries) == 1


def test_parse_lse_post_extracts_title_and_substantive_text() -> None:
    html = (FIXTURES / "lse_post_sample.html").read_text(encoding="utf-8")
    listing_html = (FIXTURES / "lse_listing.html").read_text(encoding="utf-8")
    match = re.search(r'https://libertystreeteconomics\.newyorkfed\.org/[0-9]{4}/[0-9]{2}/[a-z0-9-]+/?', listing_html)
    assert match
    source_url = match.group(0)

    parsed = parse_lse_post(html, source_url=source_url)

    assert isinstance(parsed, ParsedRegionalResearch)
    assert parsed.date.startswith("20")
    assert parsed.url == source_url
    assert parsed.title  # non-empty
    # Liberty Street posts are typically a few thousand chars
    assert len(parsed.text) > 1000


def test_write_regional_research_json_emits_one_row_per_parsed(tmp_path: Path) -> None:
    parsed = [
        ParsedRegionalResearch(
            date="2024-03-01",
            title="A Liberty Street Post",
            text="Body of the post " * 80,
            url="https://libertystreeteconomics.newyorkfed.org/2024/03/sample-post/",
            source_bank="ny_fed",
        ),
        ParsedRegionalResearch(
            date="",
            title="Empty date",
            text="something",
            url="https://libertystreeteconomics.newyorkfed.org/2024/04/another/",
            source_bank="ny_fed",
        ),
    ]

    output = tmp_path / "regional_research.json"
    written = write_regional_research_json(parsed, output)

    assert written == 1  # empty-date row is skipped
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert len(payload) == 1
    assert payload[0]["document_type"] == "regional_research"
    assert payload[0]["source_bank"] == "ny_fed"


# ----- pull_regional_research_archive -----


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
<a href="https://libertystreeteconomics.newyorkfed.org/2024/03/post-one/">Post One</a>
<a href="https://libertystreeteconomics.newyorkfed.org/2024/04/post-two/">Post Two</a>
</body></html>
"""

_POST_HTML_TEMPLATE = """
<html><head><meta property="og:title" content="LSE Post {month}"></head>
<body><div class="ts-article-text">
<p>{filler}</p>
<p>{filler2}</p>
<p>{filler3}</p>
</div></body></html>
"""


def _fake_post(month: str) -> str:
    filler = "Substantive analytical paragraph from the Liberty Street post body " * 8
    filler2 = "The data show that bond market liquidity has thinned materially since the prior quarter " * 6
    filler3 = "Cross sectional analysis across maturities suggests a persistent term premium adjustment " * 6
    return _POST_HTML_TEMPLATE.format(
        month=month, filler=filler, filler2=filler2, filler3=filler3
    )


def _route_urlopen(url, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
    url_str = url if isinstance(url, str) else url.full_url
    if url_str == ARCHIVE_LISTING_URL:
        return _FakeResponse(_LISTING_HTML.encode("utf-8"))
    if "post-one" in url_str:
        return _FakeResponse(_fake_post("March 2024").encode("utf-8"))
    if "post-two" in url_str:
        return _FakeResponse(_fake_post("April 2024").encode("utf-8"))
    raise AssertionError(f"unexpected URL fetched: {url_str}")


def test_pull_walks_listing_and_writes_json(tmp_path: Path) -> None:
    target = tmp_path / "regional_research.json"
    with patch(
        "app.services.scraper_regional_research.urllib.request.urlopen",
        side_effect=_route_urlopen,
    ) as opener:
        rows = pull_regional_research_archive(target, delay_seconds=0.0)
    assert rows == 2
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert [r["date"] for r in payload] == ["2024-03-01", "2024-04-01"]
    assert opener.call_count == 3


def test_pull_is_idempotent_when_cache_has_rows(tmp_path: Path) -> None:
    target = tmp_path / "regional_research.json"
    target.write_text(
        json.dumps(
            [{"date": "2024-03-01", "text": "x", "document_type": "regional_research"}]
        ),
        encoding="utf-8",
    )
    with patch(
        "app.services.scraper_regional_research.urllib.request.urlopen"
    ) as opener:
        rows = pull_regional_research_archive(target, delay_seconds=0.0)
    assert rows == 1
    opener.assert_not_called()


def test_pull_force_re_walks_over_existing_cache(tmp_path: Path) -> None:
    target = tmp_path / "regional_research.json"
    target.write_text(json.dumps([{"date": "stale", "text": "x"}]), encoding="utf-8")
    with patch(
        "app.services.scraper_regional_research.urllib.request.urlopen",
        side_effect=_route_urlopen,
    ):
        rows = pull_regional_research_archive(target, force=True, delay_seconds=0.0)
    assert rows == 2


def test_pull_continues_when_one_page_404s(tmp_path: Path) -> None:
    target = tmp_path / "regional_research.json"

    def _route_with_one_404(url, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        url_str = url if isinstance(url, str) else url.full_url
        if "post-one" in url_str:
            raise urllib.error.HTTPError(url_str, 404, "Not Found", None, None)  # type: ignore[arg-type]
        return _route_urlopen(url, *args, **kwargs)

    with patch(
        "app.services.scraper_regional_research.urllib.request.urlopen",
        side_effect=_route_with_one_404,
    ):
        with pytest.warns(UserWarning, match="Regional research fetch failed"):
            rows = pull_regional_research_archive(target, delay_seconds=0.0)
    assert rows == 1  # only post-two survived


def test_pull_raises_on_listing_http_error(tmp_path: Path) -> None:
    target = tmp_path / "regional_research.json"
    err = urllib.error.HTTPError(ARCHIVE_LISTING_URL, 503, "boom", None, None)  # type: ignore[arg-type]
    with patch(
        "app.services.scraper_regional_research.urllib.request.urlopen",
        side_effect=err,
    ):
        with pytest.raises(RuntimeError, match="HTTP 503"):
            pull_regional_research_archive(target, delay_seconds=0.0)
    assert not target.exists()
    assert not target.with_suffix(target.suffix + ".tmp").exists()


def test_pull_limit_caps_walk(tmp_path: Path) -> None:
    target = tmp_path / "regional_research.json"
    with patch(
        "app.services.scraper_regional_research.urllib.request.urlopen",
        side_effect=_route_urlopen,
    ) as opener:
        rows = pull_regional_research_archive(target, limit=1, delay_seconds=0.0)
    assert rows == 1
    assert opener.call_count == 2
