"""Scrape the full typed Federal Reserve communication corpus.

Statements, minutes, press-conference transcripts, speeches and testimony from
federalreserve.gov — each typed, dated, and (where reliably knowable) stamped
with an Eastern-time release time. This is the text modality for the gated
text↔market fusion model; the dense intraday realized-vol series is the market
modality and the forecast target.

Timestamp reality (drives downstream alignment granularity):
  - statement / minutes  → 14:00 ET (fixed release time)
  - press conference     → 14:30 ET (transcript of the post-meeting briefing)
  - speech / testimony   → DATE ONLY; the page carries no reliable intraday
                            time, so `time_known=False` and these are aligned
                            at daily granularity (embargo to the next forecast
                            origin), never forced into an intraday window.

Discovery parses the Fed's own index pages (FOMC historical-materials and
calendar pages, per-year speech/testimony lists) and follows the real hrefs,
so it is robust to the URL-scheme changes across eras.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import re
import time
from pathlib import Path
from typing import Any

from app.config import DATA_DIR

DEFAULT_CORPUS_PARQUET = DATA_DIR / "external" / "fed_comms" / "fed_communications.parquet"
_BASE = "https://www.federalreserve.gov"
_UA = "Mozilla/5.0 (academic research; FOMC text study)"
# fixed ET release times by type (None → time unknown, date-only alignment)
_RELEASE_ET = {"statement": "14:00", "minutes": "14:00", "press_conference": "14:30"}
_DATE_RE = re.compile(r"(\d{8})")


@dataclasses.dataclass(frozen=True)
class FedDoc:
    doc_type: str  # statement | minutes | press_conference | speech | testimony
    date: str  # YYYY-MM-DD (release/delivery date)
    timestamp_et: str  # ISO 'YYYY-MM-DD HH:MM' ET
    time_known: bool  # True for fixed-time FOMC texts, False for speeches/testimony
    speaker: str | None
    title: str
    url: str
    text: str


def _date_from_url(url: str) -> str | None:
    """Pull the YYYYMMDD embedded in a Fed document URL → ISO date."""

    m = _DATE_RE.search(url)
    if not m:
        return None
    s = m.group(1)
    try:
        return datetime.date(int(s[:4]), int(s[4:6]), int(s[6:8])).isoformat()
    except ValueError:
        return None


def _speaker_from_url(url: str) -> str | None:
    """Speech/testimony URLs are /…/<speaker><YYYYMMDD><a>.htm."""

    name = url.rstrip("/").split("/")[-1]
    m = re.match(r"([a-zA-Z]+)\d{8}", name)
    return m.group(1).lower() if m else None


def _assign_timestamp(doc_type: str, date_iso: str) -> tuple[str, bool]:
    """(timestamp_et, time_known) — fixed ET time for FOMC texts, else 09:00 placeholder."""

    fixed = _RELEASE_ET.get(doc_type)
    if fixed is not None:
        return f"{date_iso} {fixed}", True
    return f"{date_iso} 09:00", False  # date-only; placeholder, time_known=False


def _clean_html_text(html: str) -> str:
    """Extract the article body paragraphs from a Fed HTML page."""

    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    node = soup.find(id="article") or soup.find("div", class_=re.compile(r"col-")) or soup
    for tag in node.find_all(["script", "style", "nav", "header", "footer"]):
        tag.decompose()
    paras = [p.get_text(" ", strip=True) for p in node.find_all("p")]
    paras = [p for p in paras if len(p) > 1]
    return "\n\n".join(paras).strip()


def _clean_pdf_text(content: bytes) -> str:
    """Extract text from a press-conference transcript PDF."""

    import io

    import pdfplumber

    out: list[str] = []
    with pdfplumber.open(io.BytesIO(content)) as pdf:
        for page in pdf.pages:
            txt = page.extract_text() or ""
            if txt.strip():
                out.append(txt.strip())
    return "\n\n".join(out).strip()


def _index_doc_links(html: str) -> dict[str, list[str]]:
    """From an index page, bucket document hrefs by type."""

    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    hrefs = [str(a["href"]) for a in soup.find_all("a", href=True)]
    buckets: dict[str, list[str]] = {
        "statement": [],
        "minutes": [],
        "press_conference": [],
        "speech": [],
        "testimony": [],
    }
    for href in hrefs:
        low = href.lower()
        if re.search(r"/(?:pressreleases|press)/monetary/?\d{8}a?\.htm", low):
            buckets["statement"].append(href)
        elif re.search(r"fomcminutes\d{8}\.htm", low):
            buckets["minutes"].append(href)
        elif re.search(r"presconf\d{8}\.pdf", low):
            buckets["press_conference"].append(href)
        elif re.search(r"/speech/[a-z]+\d{8}", low):
            buckets["speech"].append(href)
        elif re.search(r"/testimony/[a-z]+\d{8}", low):
            buckets["testimony"].append(href)
    return {k: sorted(set(v)) for k, v in buckets.items()}


def _abs(url: str) -> str:
    return url if url.startswith("http") else f"{_BASE}{url}"


def _get(client: Any, url: str) -> Any:
    return client.get(_abs(url), timeout=30, follow_redirects=True)


def discover_urls(client: Any, *, start_year: int, end_year: int) -> dict[str, list[str]]:
    """Walk index pages → de-duplicated document URLs bucketed by type."""

    found: dict[str, set[str]] = {
        k: set() for k in ("statement", "minutes", "press_conference", "speech", "testimony")
    }
    index_pages: list[str] = ["/monetarypolicy/fomccalendars.htm"]
    for yr in range(start_year, end_year + 1):
        index_pages.append(f"/monetarypolicy/fomchistorical{yr}.htm")
        # speech/testimony index URLs switched from "{yr}<type>" to "{yr}-<type>s"
        # around 2011 — request both schemes, missing ones 404 and are skipped.
        index_pages.append(f"/newsevents/speech/{yr}-speeches.htm")
        index_pages.append(f"/newsevents/speech/{yr}speech.htm")
        index_pages.append(f"/newsevents/testimony/{yr}-testimony.htm")
        index_pages.append(f"/newsevents/testimony/{yr}testimony.htm")
    for page in index_pages:
        try:
            r = _get(client, page)
        except Exception:  # noqa: BLE001 — index page may not exist for a given year
            continue
        if r.status_code != 200:
            continue
        for doc_type, links in _index_doc_links(r.text).items():
            for href in links:
                d = _date_from_url(href)
                if d and start_year <= int(d[:4]) <= end_year:
                    found[doc_type].add(href)
    # Press-conference transcript PDFs are not reliably linked from the index
    # pages, but they occur on FOMC meeting dates and follow a fixed path.
    # Derive candidates from the discovered statement dates; the fetcher drops
    # the pre-2011 dates (no press conferences then) as 404s.
    for stmt in found["statement"]:
        d = _date_from_url(stmt)
        if d:
            found["press_conference"].add(f"/mediacenter/files/FOMCpresconf{d.replace('-', '')}.pdf")
    return {k: sorted(v) for k, v in found.items()}


def _fetch_doc(client: Any, doc_type: str, url: str) -> FedDoc | None:
    """Fetch + parse a single document; None on failure or empty extraction."""

    date_iso = _date_from_url(url)
    if not date_iso:
        return None
    try:
        r = _get(client, url)
        if r.status_code != 200:
            return None
        is_pdf = "pdf" in r.headers.get("content-type", "") or url.lower().endswith(".pdf")
        text = _clean_pdf_text(r.content) if is_pdf else _clean_html_text(r.text)
    except Exception:  # noqa: BLE001 — skip a single unreachable doc
        return None
    if len(text) < 200:  # drop stubs / failed extractions
        return None
    ts, known = _assign_timestamp(doc_type, date_iso)
    return FedDoc(
        doc_type=doc_type,
        date=date_iso,
        timestamp_et=ts,
        time_known=known,
        speaker=_speaker_from_url(url),
        title=text.split("\n", 1)[0][:200],
        url=_abs(url),
        text=text,
    )


def scrape_corpus(
    *,
    start_year: int = 2005,
    end_year: int | None = None,
    out_path: Path | str = DEFAULT_CORPUS_PARQUET,
    request_interval_seconds: float = 0.5,
    sleep_fn: Any = time.sleep,
    client: Any = None,
) -> Path:
    """Discover + fetch + parse the full typed corpus; persist one parquet."""

    import httpx
    import pandas as pd

    end_year = end_year or datetime.datetime.now(datetime.timezone.utc).year
    owns = client is None
    if owns:
        client = httpx.Client(headers={"User-Agent": _UA})
    try:
        catalog = discover_urls(client, start_year=start_year, end_year=end_year)
        rows: list[dict[str, Any]] = []
        n = 0
        for doc_type, urls in catalog.items():
            for url in urls:
                if n and request_interval_seconds > 0:
                    sleep_fn(float(request_interval_seconds))
                n += 1
                doc = _fetch_doc(client, doc_type, url)
                if doc is not None:
                    rows.append(dataclasses.asdict(doc))
        frame = pd.DataFrame(rows)
        if not frame.empty:
            frame = frame.drop_duplicates("url").sort_values("timestamp_et").reset_index(drop=True)
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(out_path, index=False)
        by_type = frame["doc_type"].value_counts().to_dict() if not frame.empty else {}
        print(f"[fed_comms_scrape] wrote {len(frame)} docs to {out_path}  by_type={by_type}")
        return out_path
    finally:
        if owns:
            client.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Scrape the typed Fed communication corpus.")
    parser.add_argument("--start-year", type=int, default=2005)
    parser.add_argument("--end-year", type=int, default=None)
    parser.add_argument("--out-path", type=Path, default=DEFAULT_CORPUS_PARQUET)
    parser.add_argument("--request-interval-seconds", type=float, default=0.5)
    args = parser.parse_args()
    scrape_corpus(
        start_year=args.start_year,
        end_year=args.end_year,
        out_path=args.out_path,
        request_interval_seconds=args.request_interval_seconds,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
