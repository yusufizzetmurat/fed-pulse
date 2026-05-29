from __future__ import annotations

import json
import urllib.error
from pathlib import Path
from unittest.mock import patch

import pytest

from app.services.scraper_speeches import (
    ParsedSpeech,
    SpeechListingEntry,
    extract_speech_listing,
    pull_speeches_archive,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def test_extract_speech_listing_returns_entries_from_real_archive_page() -> None:
    html = (FIXTURES / "fed_speech_archive_2024.html").read_text(encoding="utf-8")

    entries = extract_speech_listing(html)

    # The archive lists at least 10 speeches in 2024.
    assert len(entries) >= 10
    for entry in entries:
        assert isinstance(entry, SpeechListingEntry)
        assert entry.url.startswith("https://www.federalreserve.gov/newsevents/speech/")
        assert entry.url.endswith(".htm")
        assert entry.date  # ISO yyyy-mm-dd
        assert entry.title  # non-empty


def test_extract_speech_listing_returns_empty_on_unrelated_html() -> None:
    assert extract_speech_listing("<html><body>nothing here</body></html>") == []


def test_extract_speech_listing_deduplicates_repeated_urls() -> None:
    """Same URL listed twice in the archive must collapse to one entry."""
    repeated_url = "/newsevents/speech/powell20240131a.htm"
    html = f"""
    <html><body>
      <a href="{repeated_url}">Speech on inflation</a>
      <a href="{repeated_url}">Speech on inflation (duplicate)</a>
    </body></html>
    """
    entries = extract_speech_listing(html)
    assert len(entries) == 1
    assert entries[0].url.endswith(repeated_url)


from app.services.scraper_speeches import ParsedSpeech, parse_speech_page


def test_parse_speech_page_extracts_speaker_date_and_body() -> None:
    html = (FIXTURES / "fed_speech_powell_2024_sample.html").read_text(encoding="utf-8")
    source_url = "https://www.federalreserve.gov/newsevents/speech/powell20241114a.htm"

    parsed = parse_speech_page(html, source_url=source_url)

    assert isinstance(parsed, ParsedSpeech)
    assert "Powell" in parsed.speaker
    assert parsed.date == "2024-11-14"  # date is derivable from the URL
    assert len(parsed.text) > 500
    assert parsed.title  # non-empty
    assert parsed.url == source_url


import json

from app.services.scraper_speeches import (
    is_chair_speech,
    write_chair_speeches_json,
)


@pytest.mark.parametrize(
    "speaker,expected",
    [
        ("Chair Powell", True),
        ("Chairman Bernanke", True),
        ("Chair Jerome H. Powell", True),
        ("Chair Yellen", True),
        ("Chairwoman Yellen", True),
        ("Vice Chair Brainard", False),
        ("Vice Chairman Clarida", False),
        ("Governor Waller", False),
        ("Governor Bowman", False),
        ("", False),
    ],
)
def test_is_chair_speech_classifies_speaker_correctly(speaker: str, expected: bool) -> None:
    assert is_chair_speech(speaker) == expected


def test_write_chair_speeches_json_emits_one_row_per_chair_speech(tmp_path: Path) -> None:
    parsed = [
        ParsedSpeech(
            date="2024-01-31",
            speaker="Chair Powell",
            title="Speech on inflation",
            text="Full body of the speech " * 30,
            url="https://www.federalreserve.gov/newsevents/speech/powell20240131a.htm",
        ),
        ParsedSpeech(
            date="2024-02-15",
            speaker="Governor Waller",
            title="Speech on the labor market",
            text="Full body " * 30,
            url="https://www.federalreserve.gov/newsevents/speech/waller20240215a.htm",
        ),
    ]

    output = tmp_path / "chair_speeches.json"
    written = write_chair_speeches_json(parsed, output)

    assert written == 1

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert isinstance(payload, list)
    assert len(payload) == 1
    row = payload[0]
    assert row["title"] == "Speech on inflation"
    assert row["date"] == "2024-01-31"
    assert row["text"].startswith("Full body of the speech")
    assert row["document_type"] == "chair_speech"
    assert row["url"].endswith("powell20240131a.htm")
    assert "scraped_at_utc" in row


def test_write_chair_speeches_json_skips_rows_with_empty_body(tmp_path: Path) -> None:
    parsed = [
        ParsedSpeech(
            date="2024-01-31",
            speaker="Chair Powell",
            title="Empty speech",
            text="",
            url="https://www.federalreserve.gov/newsevents/speech/powell20240131a.htm",
        )
    ]
    output = tmp_path / "chair_speeches.json"
    written = write_chair_speeches_json(parsed, output)
    assert written == 0
    assert output.read_text(encoding="utf-8") == "[]"


from app.services.scraper_speeches import is_governor_speech, write_governor_speeches_json


@pytest.mark.parametrize(
    "speaker,expected",
    [
        ("Governor Waller", True),
        ("Governor Bowman", True),
        ("Governor Lael Brainard", True),
        ("Governor Christopher J. Waller", True),
        ("Vice Chair Brainard", False),
        ("Vice Chairman Clarida", False),
        ("Chair Powell", False),
        ("Chairman Bernanke", False),
        ("Chair Yellen", False),
        ("", False),
        ("Some Random Speaker", False),
    ],
)
def test_is_governor_speech_classifies_speaker_correctly(speaker: str, expected: bool) -> None:
    assert is_governor_speech(speaker) == expected


def test_write_governor_speeches_json_emits_one_row_per_governor_speech(tmp_path: Path) -> None:
    parsed = [
        ParsedSpeech(
            date="2024-02-15",
            speaker="Governor Waller",
            title="Speech on the labor market",
            text="Full body of the speech " * 30,
            url="https://www.federalreserve.gov/newsevents/speech/waller20240215a.htm",
        ),
        ParsedSpeech(
            date="2024-01-31",
            speaker="Chair Powell",
            title="Speech on inflation",
            text="Full body " * 30,
            url="https://www.federalreserve.gov/newsevents/speech/powell20240131a.htm",
        ),
        ParsedSpeech(
            date="2024-03-10",
            speaker="Vice Chair Brainard",
            title="Speech on financial stability",
            text="Full body " * 30,
            url="https://www.federalreserve.gov/newsevents/speech/brainard20240310a.htm",
        ),
    ]

    output = tmp_path / "governor_speeches.json"
    written = write_governor_speeches_json(parsed, output)

    assert written == 1  # only Waller is a non-Chair governor

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert isinstance(payload, list)
    assert len(payload) == 1
    row = payload[0]
    assert row["title"] == "Speech on the labor market"
    assert row["date"] == "2024-02-15"
    assert row["document_type"] == "governor_speech"
    assert row["url"].endswith("waller20240215a.htm")


def test_write_governor_speeches_json_skips_rows_with_empty_body(tmp_path: Path) -> None:
    parsed = [
        ParsedSpeech(
            date="2024-02-15",
            speaker="Governor Waller",
            title="Empty",
            text="",
            url="https://www.federalreserve.gov/newsevents/speech/waller20240215a.htm",
        )
    ]
    output = tmp_path / "governor_speeches.json"
    written = write_governor_speeches_json(parsed, output)
    assert written == 0
    assert output.read_text(encoding="utf-8") == "[]"


# ----- pull_speeches_archive -----


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
<div class="row">
  <div class="speaker">Chair Powell</div>
  <p><a href="/newsevents/speech/powell20240306a.htm">A Chair Speech</a></p>
</div>
<div class="row">
  <div class="speaker">Governor Waller</div>
  <p><a href="/newsevents/speech/waller20240501a.htm">A Governor Speech</a></p>
</div>
</body></html>
"""

_PAGE_HTML_TEMPLATE = """
<html><head><meta property="og:title" content="Speech - {speaker} - {label}"></head>
<body><div id="article">
<p class="speaker">{speaker}</p>
<p>{filler}</p>
<p>{filler2}</p>
<p>{filler3}</p>
</div></body></html>
"""


def _fake_page(speaker: str, label: str) -> str:
    filler = "Substantive speech paragraph on monetary policy and the economy " * 8
    filler2 = "Labor market conditions remain resilient and consistent with the dual mandate " * 6
    filler3 = "Inflation has eased materially over the past year toward the two percent objective " * 6
    return _PAGE_HTML_TEMPLATE.format(
        speaker=speaker, label=label, filler=filler, filler2=filler2, filler3=filler3
    )


_SINGLE_ARCHIVE_URL = "https://www.federalreserve.gov/newsevents/speech/2024-speeches.htm"


def _route_urlopen(url, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
    url_str = url if isinstance(url, str) else url.full_url
    if url_str == _SINGLE_ARCHIVE_URL:
        return _FakeResponse(_LISTING_HTML.encode("utf-8"))
    if "powell20240306" in url_str:
        return _FakeResponse(_fake_page("Chair Powell", "March 2024").encode("utf-8"))
    if "waller20240501" in url_str:
        return _FakeResponse(_fake_page("Governor Waller", "May 2024").encode("utf-8"))
    raise AssertionError(f"unexpected URL fetched: {url_str}")


def test_pull_walks_listing_and_writes_both_outputs(tmp_path: Path) -> None:
    chair_target = tmp_path / "chair_speeches.json"
    gov_target = tmp_path / "governor_speeches.json"
    with patch(
        "app.services.scraper_speeches.urllib.request.urlopen",
        side_effect=_route_urlopen,
    ) as opener:
        rows = pull_speeches_archive(
            chair_target,
            gov_target,
            archive_url=_SINGLE_ARCHIVE_URL,
            delay_seconds=0.0,
        )
    assert rows == 2  # one chair row + one governor row
    chair_payload = json.loads(chair_target.read_text(encoding="utf-8"))
    gov_payload = json.loads(gov_target.read_text(encoding="utf-8"))
    assert len(chair_payload) == 1
    assert chair_payload[0]["document_type"] == "chair_speech"
    assert len(gov_payload) == 1
    assert gov_payload[0]["document_type"] == "governor_speech"
    # 1 listing fetch + 2 page fetches
    assert opener.call_count == 3


def test_pull_is_idempotent_when_both_caches_have_rows(tmp_path: Path) -> None:
    chair_target = tmp_path / "chair_speeches.json"
    gov_target = tmp_path / "governor_speeches.json"
    chair_target.write_text(
        json.dumps([{"date": "2024-03-06", "text": "x", "document_type": "chair_speech"}]),
        encoding="utf-8",
    )
    gov_target.write_text(
        json.dumps([{"date": "2024-05-01", "text": "x", "document_type": "governor_speech"}]),
        encoding="utf-8",
    )
    with patch("app.services.scraper_speeches.urllib.request.urlopen") as opener:
        rows = pull_speeches_archive(
            chair_target,
            gov_target,
            archive_url=_SINGLE_ARCHIVE_URL,
            delay_seconds=0.0,
        )
    assert rows == 2
    opener.assert_not_called()


def test_pull_force_re_walks_over_existing_caches(tmp_path: Path) -> None:
    chair_target = tmp_path / "chair_speeches.json"
    gov_target = tmp_path / "governor_speeches.json"
    chair_target.write_text(json.dumps([{"date": "stale", "text": "x"}]), encoding="utf-8")
    gov_target.write_text(json.dumps([{"date": "stale", "text": "x"}]), encoding="utf-8")
    with patch(
        "app.services.scraper_speeches.urllib.request.urlopen",
        side_effect=_route_urlopen,
    ):
        rows = pull_speeches_archive(
            chair_target,
            gov_target,
            archive_url=_SINGLE_ARCHIVE_URL,
            force=True,
            delay_seconds=0.0,
        )
    assert rows == 2


def test_pull_continues_when_one_page_404s(tmp_path: Path) -> None:
    chair_target = tmp_path / "chair_speeches.json"
    gov_target = tmp_path / "governor_speeches.json"

    def _route_with_one_404(url, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        url_str = url if isinstance(url, str) else url.full_url
        if "powell20240306" in url_str:
            raise urllib.error.HTTPError(url_str, 404, "Not Found", None, None)  # type: ignore[arg-type]
        return _route_urlopen(url, *args, **kwargs)

    with patch(
        "app.services.scraper_speeches.urllib.request.urlopen",
        side_effect=_route_with_one_404,
    ):
        with pytest.warns(UserWarning, match="Speech fetch failed"):
            rows = pull_speeches_archive(
                chair_target,
                gov_target,
                archive_url=_SINGLE_ARCHIVE_URL,
                delay_seconds=0.0,
            )
    assert rows == 1  # only the governor row survived


def test_pull_raises_on_listing_http_error(tmp_path: Path) -> None:
    chair_target = tmp_path / "chair_speeches.json"
    gov_target = tmp_path / "governor_speeches.json"

    def _route_listing_503(url, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        url_str = url if isinstance(url, str) else url.full_url
        raise urllib.error.HTTPError(url_str, 503, "boom", None, None)  # type: ignore[arg-type]

    with patch(
        "app.services.scraper_speeches.urllib.request.urlopen",
        side_effect=_route_listing_503,
    ):
        with pytest.warns(UserWarning, match="Speech listing fetch failed"):
            with pytest.raises(RuntimeError, match="zero rows"):
                pull_speeches_archive(
                    chair_target,
                    gov_target,
                    archive_url=_SINGLE_ARCHIVE_URL,
                    delay_seconds=0.0,
                )
    assert not chair_target.exists()
    assert not gov_target.exists()


def test_pull_limit_caps_walk(tmp_path: Path) -> None:
    chair_target = tmp_path / "chair_speeches.json"
    gov_target = tmp_path / "governor_speeches.json"
    with patch(
        "app.services.scraper_speeches.urllib.request.urlopen",
        side_effect=_route_urlopen,
    ) as opener:
        rows = pull_speeches_archive(
            chair_target,
            gov_target,
            archive_url=_SINGLE_ARCHIVE_URL,
            limit=1,
            delay_seconds=0.0,
        )
    # First entry from the fixture listing is the chair; governor file
    # is written but empty.
    assert rows == 1
    # 1 listing + 1 page = 2 fetches
    assert opener.call_count == 2
