"""Probe candidate external corpora for the multi-axis label schema.

Writes docs/corpora-inventory.md listing what's actually fetchable, what is
gated behind a paper-replication download, and what 404s. Re-running probes
each source fresh.

The script does not download anything; it only checks accessibility and
collects metadata.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "docs" / "corpora-inventory.md"
HTTP_TIMEOUT = 15


@dataclass
class CorpusCandidate:
    name: str
    kind: str  # "hf_dataset", "paper_replication", "kaggle", "fed_data_release"
    url: str
    citation: str
    expected_axes: list[str]
    notes: str = ""
    obtainable: bool | None = None
    rows_estimate: int | None = None
    license: str = "unknown"
    last_modified: str = ""

    def to_row(self) -> str:
        status = "✓" if self.obtainable else ("✗" if self.obtainable is False else "?")
        rows = str(self.rows_estimate) if self.rows_estimate is not None else "—"
        axes = ", ".join(self.expected_axes) or "—"
        return f"| {status} | {self.name} | {self.kind} | {axes} | {rows} | {self.license} | [link]({self.url}) |"


HF_CANDIDATES: list[CorpusCandidate] = [
    CorpusCandidate(
        name="FOMC Communication (Trillion Dollar Words)",
        kind="hf_dataset",
        url="https://huggingface.co/datasets/gtfintechlab/fomc_communication",
        citation="Shah, Paturi, Chava (ACL 2023). Hand-labelled hawkish / dovish / neutral on FOMC statements, minutes, and press conferences.",
        expected_axes=["stance"],
        notes="Primary labelled source; already ingested under the TDW alias.",
    ),
    CorpusCandidate(
        name="Financial PhraseBank",
        kind="hf_dataset",
        url="https://huggingface.co/datasets/takala/financial_phrasebank",
        citation="Malo et al. (2014). 4,840 finance news sentences labelled positive / negative / neutral.",
        expected_axes=["stance"],
        notes="Not FOMC-specific; useful as a domain-adaptive pretraining auxiliary task on the FinBERT-FedAdjacent checkpoint.",
    ),
]

PAPER_REPLICATION_CANDIDATES: list[CorpusCandidate] = [
    CorpusCandidate(
        name="Gürkaynak–Sack–Swanson factor decomposition",
        kind="paper_replication",
        url="https://www.federalreserve.gov/econresdata/notes/feds-notes/2015/effects-fomc-text-on-market-expectations-20151113.html",
        citation="Gürkaynak, Sack, Swanson (IJCB 2005). 'Do Actions Speak Louder Than Words?' Target-rate vs forward-guidance shock loadings per FOMC date.",
        expected_axes=["factor"],
        notes="Replication data historically posted on Sack's NYU page and the Federal Reserve Board's research-data archive. Manual download.",
    ),
    CorpusCandidate(
        name="Aruoba–Drechsel narrative shocks",
        kind="paper_replication",
        url="https://www.aruoba.econ.umd.edu/research/",
        citation="Aruoba & Drechsel (NBER w29307). Narrative identification of monetary policy shocks from FOMC text.",
        expected_axes=["factor"],
        notes="Posted on Aruoba's UMD page. Per-meeting shock series, csv format last time it was published.",
    ),
    CorpusCandidate(
        name="Cieslak–Schrimpf monetary-vs-growth news",
        kind="paper_replication",
        url="https://sites.google.com/view/anna-cieslak/",
        citation="Cieslak & Schrimpf (J. Int. Econ. 2019). Decomposition of FOMC-day price moves into monetary news and growth news.",
        expected_axes=["factor", "topic"],
        notes="Per-event labels for the FOMC release window. Posted on Cieslak's Duke page.",
    ),
    CorpusCandidate(
        name="Hansen–McMahon topic shares",
        kind="paper_replication",
        url="https://stephenhansen.eu/research/",
        citation="Hansen & McMahon (J. Int. Econ. 2016). 'Shocking Language' — LDA topic shares over FOMC statements.",
        expected_axes=["topic"],
        notes="Per-meeting topic distributions. Hansen's Oxford / Imperial pages have hosted the replication dataset.",
    ),
    CorpusCandidate(
        name="Lucca–Trebbi communication index",
        kind="paper_replication",
        url="https://www.newyorkfed.org/research/staff_reports/sr357",
        citation="Lucca & Trebbi (NBER w15367 / NY Fed Staff Report 357). Continuous hawkish-dovish index built from Google-search proximity of FOMC text to anchor terms.",
        expected_axes=["stance"],
        notes="Continuous score (not categorical). NY Fed staff-report page links a data appendix.",
    ),
    CorpusCandidate(
        name="Shapiro–Wilson FOMC tone series",
        kind="fed_data_release",
        url="https://www.frbsf.org/economic-research/indicators-data/daily-news-sentiment-index/",
        citation="Shapiro & Wilson (San Francisco Fed). FOMC-day tone series built from a constrained text-sentiment dictionary.",
        expected_axes=["stance"],
        notes="SF Fed publishes the tone series openly. Verify which subseries covers FOMC text vs general news.",
    ),
    CorpusCandidate(
        name="Bauer–Bernanke–Milstein risk-appetite",
        kind="paper_replication",
        url="https://www.michaeldbauer.com/research.html",
        citation="Bauer, Bernanke & Milstein (NBER 2023). Risk-appetite channel of monetary policy; FOMC-day shock decomposition.",
        expected_axes=["factor"],
        notes="Posted on Bauer's UC Irvine page.",
    ),
]


def _http_head(url: str) -> tuple[int | None, str]:
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "fed-pulse-inventory/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT) as response:
            return int(response.status), response.headers.get("Last-Modified", "")
    except Exception as exc:  # pragma: no cover - network reachability is the test
        return None, str(exc)


def _hf_dataset_info(slug: str) -> dict[str, Any] | None:
    api = f"https://huggingface.co/api/datasets/{slug}"
    try:
        with urllib.request.urlopen(api, timeout=HTTP_TIMEOUT) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception:
        return None


def _hf_slug_from_url(url: str) -> str | None:
    parsed = urllib.parse.urlparse(url)
    if "huggingface.co" not in parsed.netloc:
        return None
    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) >= 3 and parts[0] == "datasets":
        return "/".join(parts[1:3])
    return None


def probe(candidates: list[CorpusCandidate]) -> list[CorpusCandidate]:
    for cand in candidates:
        if cand.kind == "hf_dataset":
            slug = _hf_slug_from_url(cand.url)
            if slug:
                info = _hf_dataset_info(slug)
                if info is not None:
                    cand.obtainable = True
                    cand.last_modified = str(info.get("lastModified") or "")
                    card = info.get("cardData") or {}
                    if isinstance(card, dict):
                        cand.license = str(card.get("license") or cand.license)
                    continue
        status, last_modified = _http_head(cand.url)
        cand.last_modified = last_modified or cand.last_modified
        cand.obtainable = bool(status and 200 <= status < 400)
    return candidates


def render_markdown(candidates: list[CorpusCandidate]) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines = [
        "# External corpora inventory",
        "",
        f"_Probed: {now}_",
        "",
        "Status legend: ✓ accessible · ✗ not accessible at this URL · ? not probed (manual fetch).",
        "",
        "| Status | Name | Kind | Axes covered | Rows (est.) | License | URL |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for cand in candidates:
        lines.append(cand.to_row())
    lines.append("")
    lines.append("## Per-source notes")
    lines.append("")
    for cand in candidates:
        lines.append(f"### {cand.name}")
        lines.append("")
        lines.append(f"- **Kind:** {cand.kind}")
        lines.append(f"- **URL:** {cand.url}")
        lines.append(f"- **Citation:** {cand.citation}")
        lines.append(f"- **Axes covered:** {', '.join(cand.expected_axes) or '—'}")
        if cand.notes:
            lines.append(f"- **Notes:** {cand.notes}")
        if cand.last_modified:
            lines.append(f"- **Last-modified:** {cand.last_modified}")
        lines.append("")
    return "\n".join(lines)


def write_inventory(out_path: Path, candidates: list[CorpusCandidate]) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_markdown(candidates) + "\n", encoding="utf-8")
    return out_path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe candidate external corpora and write docs/corpora-inventory.md.")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--skip-network", action="store_true",
                        help="Render the inventory without probing remote URLs (uses candidate metadata only).")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    candidates = HF_CANDIDATES + PAPER_REPLICATION_CANDIDATES
    if not args.skip_network:
        candidates = probe(candidates)
    write_inventory(Path(args.out), candidates)
    obtainable = sum(1 for c in candidates if c.obtainable)
    print(f"[inventory] wrote {args.out} ({obtainable}/{len(candidates)} accessible)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
