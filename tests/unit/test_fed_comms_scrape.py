"""Pure-function tests for the Fed communication scraper."""

from __future__ import annotations
from app.data import fed_comms_scrape as s


def test_date_and_speaker_from_url() -> None:
    assert s._date_from_url("/newsevents/speech/waller20241202a.htm") == "2024-12-02"
    assert s._date_from_url("/newsevents/pressreleases/monetary20240131a.htm") == "2024-01-31"
    assert s._date_from_url("/no/date/here.htm") is None
    assert s._date_from_url("/x/monetary20241302a.htm") is None  # month 13 invalid
    assert s._speaker_from_url("/newsevents/speech/waller20241202a.htm") == "waller"
    assert s._speaker_from_url("/newsevents/pressreleases/monetary20240131a.htm") == "monetary"


def test_assign_timestamp_fixed_vs_dateonly() -> None:
    assert s._assign_timestamp("statement", "2024-01-31") == ("2024-01-31 14:00", True)
    assert s._assign_timestamp("press_conference", "2024-01-31") == ("2024-01-31 14:30", True)
    ts, known = s._assign_timestamp("speech", "2024-12-02")
    assert known is False and ts.startswith("2024-12-02")


def test_index_doc_links_buckets() -> None:
    html = """<a href="/newsevents/pressreleases/monetary20180131a.htm">s</a>
    <a href="/monetarypolicy/fomcminutes20180131.htm">m</a>
    <a href="/mediacenter/files/FOMCpresconf20180131.pdf">p</a>
    <a href="/newsevents/speech/powell20180601a.htm">sp</a>
    <a href="/newsevents/testimony/powell20180717a.htm">t</a>
    <a href="/about.htm">x</a>"""
    b = s._index_doc_links(html)
    assert b["statement"] == ["/newsevents/pressreleases/monetary20180131a.htm"]
    assert b["minutes"] == ["/monetarypolicy/fomcminutes20180131.htm"]
    assert b["press_conference"] == ["/mediacenter/files/FOMCpresconf20180131.pdf"]
    assert b["speech"] and b["testimony"]


def test_clean_html_text_extracts_paragraphs() -> None:
    html = (
        '<html><body><div id="article"><p>First paragraph here.</p>'
        "<script>junk()</script><p>Second paragraph here.</p></div></body></html>"
    )
    out = s._clean_html_text(html)
    assert "First paragraph here." in out and "Second paragraph here." in out
    assert "junk" not in out
