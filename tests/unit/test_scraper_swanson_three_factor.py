"""Tests for the Swanson 2021 three-factor adapter (#420)."""

from __future__ import annotations

import json
import urllib.error
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

pytest.importorskip("bs4")  # source registry init imports HTML scrapers
pytest.importorskip("openpyxl")  # xlsx engine

from app.data.sources import SOURCES  # noqa: E402
from app.data.sources.swanson_three_factor import (  # noqa: E402
    SWANSON_THREE_FACTOR_UPSTREAM_URL,
    SwansonThreeFactorScraper,
    _parse_swanson_row,
    pull_swanson_three_factor_xlsx,
)


def _write_fixture_xlsx(path: Path, n: int = 3) -> None:
    """Write a Swanson-shape xlsx fixture mirroring the upstream layout.

    The real ``pre-and-post-ZLB-factors-extended.xlsx`` carries a single
    title row at openpyxl row 1 ("Estimated Factors" / etc.), the
    column-header row at openpyxl row 2 ("Federal Funds Rate factor",
    "Forward Guidance factor", ...), and the data from row 3 onwards.
    ``parse_entry``'s ``pd.read_excel(header=1)`` skips the title row
    and uses the column-header row as the frame's columns; the previous
    pandas-only fixture inserted the column-header row at openpyxl row 1
    (the implicit ``to_excel`` header) which made ``header=1`` consume
    the title row instead, every factor field then arrived as ``None``.

    Write directly via openpyxl so the three rows land in the order the
    parser expects.
    """

    from openpyxl import Workbook

    dates = pd.date_range("2010-01-27", periods=n, freq="60D")
    wb = Workbook()
    ws = wb.active
    ws.title = "Data"
    # Row 1 — single label row mirroring the upstream xlsx's title cell.
    ws.append([None, None, "Estimated Factors", None, None, None])
    # Row 2 — column headers consumed by ``pd.read_excel(header=1)``.
    ws.append(
        [
            "Unnamed: 0",
            "Unnamed: 1",
            "Federal Funds Rate factor",
            "Forward Guidance factor",
            "LSAP factor",
            " – LSAP factor",
        ]
    )
    # Rows 3+ — data with the same dates + factor sequence the original
    # fixture emitted.
    for i in range(n):
        ws.append(
            [
                None,
                dates[i].to_pydatetime(),
                -0.10 + i * 0.05,
                0.20 + i * 0.01,
                -0.05 + i * 0.02,
                0.05 - i * 0.02,
            ]
        )
    wb.save(path)


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


def test_swanson_scraper_is_registered() -> None:
    assert "swanson_three_factor" in SOURCES
    scraper = SOURCES["swanson_three_factor"]
    assert isinstance(scraper, SwansonThreeFactorScraper)
    assert scraper.metadata.provenance.value == "peer_reviewed"
    assert "Swanson" in scraper.metadata.name


def test_parse_row_returns_registry_shape() -> None:
    row = {
        "meeting_date": pd.Timestamp("2014-09-17"),
        "target_factor": 0.123,
        "forward_guidance_factor": -0.456,
        "lsap_factor": 0.789,
    }
    parsed = _parse_swanson_row(row)
    assert parsed is not None
    assert parsed["source_record_id"] == "swanson_2014-09-17"
    assert parsed["event_date_hint"] == "2014-09-17"
    assert parsed["document_type"] == "statement"
    assert parsed["license_scope"] == "research_only"
    assert parsed["citation_ref"] == "swanson_2021_jme"
    extras = parsed["multi_axis_extras"]
    assert extras["swanson_target_factor"] == pytest.approx(0.123)
    assert extras["swanson_forward_guidance_factor"] == pytest.approx(-0.456)
    assert extras["swanson_lsap_factor"] == pytest.approx(0.789)


def test_parse_row_drops_row_with_no_factors() -> None:
    row = {
        "meeting_date": pd.Timestamp("2014-09-17"),
        "target_factor": None,
        "forward_guidance_factor": None,
        "lsap_factor": None,
    }
    assert _parse_swanson_row(row) is None


def test_parse_row_drops_row_with_missing_date() -> None:
    assert (
        _parse_swanson_row(
            {
                "meeting_date": None,
                "target_factor": 0.1,
                "forward_guidance_factor": 0.2,
                "lsap_factor": 0.3,
            }
        )
        is None
    )


def test_pull_downloads_and_writes_xlsx(tmp_path: Path) -> None:
    target = tmp_path / "external" / "swanson" / "release.xlsx"
    src = tmp_path / "src.xlsx"
    _write_fixture_xlsx(src, n=3)
    body = src.read_bytes()
    with patch(
        "app.data.sources.swanson_three_factor.urllib.request.urlopen",
        return_value=_FakeResponse(body),
    ) as opener:
        rows = pull_swanson_three_factor_xlsx(target)
    assert rows == 3
    assert target.exists()
    opener.assert_called_once()


def test_pull_is_idempotent_when_cache_has_rows(tmp_path: Path) -> None:
    target = tmp_path / "release.xlsx"
    _write_fixture_xlsx(target, n=2)
    with patch(
        "app.data.sources.swanson_three_factor.urllib.request.urlopen"
    ) as opener:
        rows = pull_swanson_three_factor_xlsx(target)
    assert rows == 2
    opener.assert_not_called()


def test_pull_force_re_downloads_over_existing_cache(tmp_path: Path) -> None:
    target = tmp_path / "release.xlsx"
    _write_fixture_xlsx(target, n=1)
    src = tmp_path / "src.xlsx"
    _write_fixture_xlsx(src, n=4)
    body = src.read_bytes()
    with patch(
        "app.data.sources.swanson_three_factor.urllib.request.urlopen",
        return_value=_FakeResponse(body),
    ):
        rows = pull_swanson_three_factor_xlsx(target, force=True)
    assert rows == 4


def test_pull_raises_on_http_error(tmp_path: Path) -> None:
    target = tmp_path / "release.xlsx"
    err = urllib.error.HTTPError(
        SWANSON_THREE_FACTOR_UPSTREAM_URL, 404, "Not Found", None, None  # type: ignore[arg-type]
    )
    with patch(
        "app.data.sources.swanson_three_factor.urllib.request.urlopen",
        side_effect=err,
    ):
        with pytest.raises(RuntimeError, match="HTTP 404"):
            pull_swanson_three_factor_xlsx(target)
    assert not target.exists()
    assert not target.with_suffix(target.suffix + ".tmp").exists()


def test_scraper_fetch_listing_returns_row_dicts(tmp_path: Path) -> None:
    src = tmp_path / "src.xlsx"
    _write_fixture_xlsx(src, n=3)
    listing = SwansonThreeFactorScraper().fetch_listing(str(src))
    assert len(listing) == 3
    assert "meeting_date" in listing[0]
    assert "target_factor" in listing[0]


def test_scraper_parse_entry_round_trips(tmp_path: Path) -> None:
    src = tmp_path / "src.xlsx"
    _write_fixture_xlsx(src, n=1)
    scraper = SwansonThreeFactorScraper()
    listing = scraper.fetch_listing(str(src))
    row = listing[0]
    # Coerce non-JSON-serialisable Timestamp to isoformat string
    row["meeting_date"] = (
        row["meeting_date"].strftime("%Y-%m-%d")  # type: ignore[union-attr]
        if hasattr(row["meeting_date"], "strftime")
        else row["meeting_date"]
    )
    parsed = scraper.parse_entry(
        json.dumps(row), source_url="https://swanson.test"
    )
    assert parsed is not None
    assert parsed["source_url"] == "https://swanson.test"
