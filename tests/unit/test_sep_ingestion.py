"""Unit tests for SEP projection-table parsing (no network)."""

from __future__ import annotations

import pandas as pd

from app.data.sep_ingestion import (
    _canon_variable,
    _norm_horizon,
    _range,
    _scalar,
    fetch_projection_page,
    parse_projection_tables,
)


class _FakeResponse:
    def __init__(self, status_code: int, text: str = "") -> None:
        self.status_code = status_code
        self.text = text
        self.encoding = ""

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise AssertionError(f"unexpected status {self.status_code}")


class _FakeSession:
    """Returns 404 for the first URL spelling, 200 for the second."""

    def __init__(self) -> None:
        self.urls: list[str] = []

    def get(self, url: str, **_: object) -> _FakeResponse:
        self.urls.append(url)
        if "fomcprojtable" in url:  # the "projtable" spelling
            return _FakeResponse(200, "<html>ok</html>")
        return _FakeResponse(404)


def test_fetch_falls_back_to_alternate_spelling() -> None:
    session = _FakeSession()
    html = fetch_projection_page("20220316", session=session)  # type: ignore[arg-type]
    assert html == "<html>ok</html>"
    # both spellings were tried, in order
    assert any("fomcprojtabl2" in u for u in session.urls)
    assert any("fomcprojtable2" in u for u in session.urls)


class _AllMissingSession:
    def get(self, url: str, **_: object) -> _FakeResponse:
        return _FakeResponse(404)


def test_fetch_returns_none_when_all_variants_404() -> None:
    assert fetch_projection_page("20200316", session=_AllMissingSession()) is None  # type: ignore[arg-type]


class _ServerErrorSession:
    def get(self, url: str, **_: object) -> _FakeResponse:
        return _FakeResponse(503)


def test_fetch_raises_on_persistent_server_error() -> None:
    # A 5xx on every variant must surface, not be mistaken for "no SEP".
    import pytest

    with pytest.raises(RuntimeError, match="503"):
        fetch_projection_page("20240320", session=_ServerErrorSession())  # type: ignore[arg-type]

# 2015+ format: combined Median / Central Tendency / Range, scalar medians.
_MODERN_HTML = """
<table>
  <thead>
    <tr><th>Variable</th><th>Median 2024</th><th>Median Longer run</th>
        <th>Central Tendency 2024</th><th>Range 2024</th></tr>
  </thead>
  <tbody>
    <tr><td>Change in real GDP</td><td>2.1</td><td>1.8</td>
        <td>2.0&#8211;2.4</td><td>1.3&#8211;2.7</td></tr>
    <tr><td>December projection</td><td>1.4</td><td>1.8</td>
        <td>1.2&#8211;1.7</td><td>0.8&#8211;2.0</td></tr>
    <tr><td>Unemployment rate</td><td>4.0</td><td>4.1</td>
        <td>3.9&#8211;4.1</td><td>3.8&#8211;4.5</td></tr>
    <tr><td>PCE inflation</td><td>2.4</td><td>2.0</td>
        <td>2.3&#8211;2.6</td><td>2.2&#8211;2.9</td></tr>
    <tr><td>Core PCE inflation</td><td>2.6</td><td></td>
        <td>2.5&#8211;2.6</td><td>2.4&#8211;3.0</td></tr>
    <tr><td>Federal funds rate</td><td>4.6</td><td>2.6</td>
        <td>4.4&#8211;4.9</td><td>3.9&#8211;5.4</td></tr>
  </tbody>
</table>
"""

# 2012-2014 format: Central tendency / Range only, "low to high" cells, no median.
_LEGACY_HTML = """
<table>
  <thead>
    <tr><th>Variable</th><th>Central tendency 2014</th>
        <th>Central tendency Longer run</th><th>Range 2014</th></tr>
  </thead>
  <tbody>
    <tr><td>Change in real GDP</td><td>2.8 to 3.0</td>
        <td>2.2 to 2.3</td><td>2.1 to 3.0</td></tr>
    <tr><td>December projection</td><td>2.8 to 3.2</td>
        <td>2.2 to 2.4</td><td>2.2 to 3.3</td></tr>
    <tr><td>Unemployment rate</td><td>6.1 to 6.3</td>
        <td>5.2 to 5.6</td><td>6.0 to 6.5</td></tr>
  </tbody>
</table>
"""


def test_canon_variable_maps_and_skips() -> None:
    assert _canon_variable("Change in real GDP") == "gdp"
    assert _canon_variable("Unemployment rate") == "unemployment"
    assert _canon_variable("PCE inflation") == "pce"
    # core must win over the bare 'pce' pattern
    assert _canon_variable("Core PCE inflation") == "core_pce"
    assert _canon_variable("Federal funds rate") == "ffr"
    # prior-projection rows and footnotes are dropped
    assert _canon_variable("December projection") is None
    assert _canon_variable("") is None


def test_norm_horizon() -> None:
    assert _norm_horizon("2024") == "2024"
    assert _norm_horizon("Longer run") == "LR"
    assert _norm_horizon("Variable") is None


def test_scalar_parsing() -> None:
    assert _scalar("2.1") == 2.1
    assert _scalar("-1.1") == -1.1
    assert _scalar("-") is None
    assert _scalar("") is None


def test_range_parsing_both_separators() -> None:
    assert _range("2.0–2.4") == (2.0, 2.4)  # en-dash
    assert _range("2.8 to 3.0") == (2.8, 3.0)
    assert _range("2.0 - 2.4") == (2.0, 2.4)  # space-flanked hyphen
    assert _range("-") == (None, None)
    # single value -> (v, v)
    assert _range("2.5") == (2.5, 2.5)


def test_range_preserves_negative_boundaries() -> None:
    # A leading minus is a unary minus, not a range separator.
    assert _range("-0.1 to 0.1") == (-0.1, 0.1)
    assert _range("-0.1–0.4") == (-0.1, 0.4)
    assert _scalar("-0.1") == -0.1


def test_parse_modern_table() -> None:
    out = parse_projection_tables(_MODERN_HTML, "2024-03-20")
    assert not out.empty
    assert set(out["variable"]) == {"gdp", "unemployment", "pce", "core_pce", "ffr"}
    # prior "December projection" row excluded -> no duplicate gdp/2024 beyond one
    gdp_2024 = out[(out["variable"] == "gdp") & (out["horizon"] == "2024")]
    assert len(gdp_2024) == 1
    row = gdp_2024.iloc[0]
    assert row["median"] == 2.1
    assert row["central_low"] == 2.0
    assert row["central_high"] == 2.4
    assert row["range_low"] == 1.3
    assert row["range_high"] == 2.7
    # longer-run horizon captured
    ffr_lr = out[(out["variable"] == "ffr") & (out["horizon"] == "LR")]
    assert ffr_lr.iloc[0]["median"] == 2.6


def test_parse_legacy_table_no_median() -> None:
    out = parse_projection_tables(_LEGACY_HTML, "2014-03-19")
    assert not out.empty
    gdp_2014 = out[(out["variable"] == "gdp") & (out["horizon"] == "2014")].iloc[0]
    # legacy era: no median column -> NaN, but central tendency present
    assert pd.isna(gdp_2014["median"])
    assert gdp_2014["central_low"] == 2.8
    assert gdp_2014["central_high"] == 3.0
    # prior-projection comparison row excluded
    assert len(out[(out["variable"] == "gdp") & (out["horizon"] == "2014")]) == 1


def test_meeting_date_propagated() -> None:
    out = parse_projection_tables(_MODERN_HTML, "2024-03-20")
    assert (out["meeting_date"] == "2024-03-20").all()
