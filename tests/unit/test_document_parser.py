from __future__ import annotations

import pytest

from app.services.document_parser import normalise, parse_paste


def test_normalise_collapses_internal_whitespace_and_blank_lines():
    raw = "Recent  indicators   suggest\n\n\n  economic activity has continued to expand.\r\n"
    cleaned = normalise(raw)
    assert "  " not in cleaned
    assert "\n\n\n" not in cleaned
    assert cleaned.endswith("expand.")


def test_normalise_handles_empty_input():
    assert normalise("") == ""
    assert normalise("   \n  \r\n   ") == ""


def test_parse_paste_returns_normalised_text():
    result = parse_paste("hello   world")
    assert result.source_kind == "paste"
    assert result.text == "hello world"
    assert result.char_count == len("hello world")


def test_parse_paste_strips_surrounding_whitespace():
    result = parse_paste("\n\n  meeting notes  \n")
    assert result.text == "meeting notes"
    assert result.char_count == len("meeting notes")


def test_parse_url_extracts_visible_text_with_mocked_client(monkeypatch):
    pytest.importorskip("httpx")
    httpx = pytest.importorskip("httpx")
    pytest.importorskip("bs4")

    from app.services import document_parser

    sample_html = (
        "<html><head><script>track()</script><style>.x{}</style></head>"
        "<body><nav>menu</nav>"
        "<article><h1>Statement</h1><p>The Committee decided to maintain the target range.</p>"
        "<p>Recent indicators expanded.</p></article>"
        "<footer>noise</footer></body></html>"
    )

    class _FakeResponse:
        status_code = 200
        text = sample_html
        content = sample_html.encode("utf-8")
        headers = {"content-type": "text/html; charset=utf-8", "content-length": str(len(sample_html))}

        def raise_for_status(self):
            return None

    class _FakeClient:
        async def get(self, _url):
            return _FakeResponse()

        async def aclose(self):
            return None

    async def _run():
        return await document_parser.parse_url("https://example.com/statement", client=_FakeClient())

    import asyncio

    result = asyncio.run(_run())
    assert result.source_kind == "url"
    assert "Committee decided to maintain" in result.text
    assert "menu" not in result.text
    assert "noise" not in result.text
    assert result.source_metadata["url"] == "https://example.com/statement"


def test_parse_pdf_stream_extracts_text_from_a_real_pdf(tmp_path):
    pdfplumber = pytest.importorskip("pdfplumber", reason="pdfplumber not installed")
    reportlab = pytest.importorskip("reportlab.pdfgen.canvas", reason="reportlab needed to author a test PDF")

    from app.services.document_parser import parse_pdf_stream

    pdf_path = tmp_path / "sample.pdf"
    canvas = reportlab.Canvas(str(pdf_path))
    canvas.drawString(100, 750, "FOMC statement excerpt")
    canvas.drawString(100, 730, "Recent indicators continued to expand.")
    canvas.save()

    with pdf_path.open("rb") as stream:
        parsed = parse_pdf_stream(stream, filename="sample.pdf")
    assert "FOMC statement excerpt" in parsed.text
    assert parsed.source_kind == "pdf"
    assert parsed.source_metadata["filename"] == "sample.pdf"
