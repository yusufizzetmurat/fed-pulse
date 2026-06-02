"""Tests for the Workspace-spine MP-surprise serving wrapper.

The serving wrapper only picks the latest ``event_date`` row from
``mp_surprises.parquet`` and translates it into a wire response — the
heavy lifting (rate-curve reconstruction, PCA, no-look-ahead) is
covered by :mod:`tests.unit.test_mp_surprise`. These tests verify the
serving-layer contract:

- positive level above the band -> ``hawkish`` direction
- negative level below the band -> ``dovish`` direction
- ``|level| <= NO_SURPRISE_BAND_BPS`` -> ``no_surprise``
- ``ff_target_prior`` (percent) is reported in bps
- missing parquet raises :class:`MpSurpriseUnavailable`
- empty parquet raises :class:`MpSurpriseUnavailable`
- the *latest* row is selected regardless of file ordering
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from app.services.mp_surprise_service import (
    NO_SURPRISE_BAND_BPS,
    MpSurpriseUnavailable,
    load_latest_mp_surprise,
)


def _write_parquet(
    path: Path,
    *,
    rows: list[dict[str, object]],
) -> Path:
    df = pd.DataFrame(rows)
    df.to_parquet(path, index=False)
    return path


def _row(
    event_date: str,
    mp_surprise_level: float | None,
    *,
    is_intermeeting: bool = False,
    ff_target_prior: float | None = 3.625,
) -> dict[str, object]:
    return {
        "event_date": event_date,
        "meeting_id": int(event_date.replace("-", "")),
        "ff_target_prior": ff_target_prior,
        "ff_target_after": 3.625,
        "mp_surprise_level": mp_surprise_level,
        "mp_surprise_path_factor": 0.0,
        "is_intermeeting": is_intermeeting,
        "methodology": "ois_proxy",
    }


def test_latest_row_with_positive_level_classifies_as_hawkish(tmp_path: Path) -> None:
    parquet = _write_parquet(
        tmp_path / "mp_surprises.parquet",
        rows=[
            _row("2026-01-28", -14.5),
            _row("2026-03-18", -11.5),
            _row("2026-04-29", 12.0, ff_target_prior=3.625),
        ],
    )

    out = load_latest_mp_surprise(parquet)

    assert out.event_date == "2026-04-29"
    assert out.mp_surprise_level_bps == pytest.approx(12.0)
    assert out.direction == "hawkish"
    assert out.magnitude_bps == pytest.approx(12.0)
    assert out.is_intermeeting is False
    # 3.625% target -> 362.5 bps prior.
    assert out.ff_target_prior_bps == pytest.approx(362.5)


def test_latest_row_with_negative_level_classifies_as_dovish(tmp_path: Path) -> None:
    parquet = _write_parquet(
        tmp_path / "mp_surprises.parquet",
        rows=[
            _row("2026-01-28", 4.0),
            _row("2026-04-29", -14.5),
        ],
    )

    out = load_latest_mp_surprise(parquet)

    assert out.event_date == "2026-04-29"
    assert out.direction == "dovish"
    assert out.magnitude_bps == pytest.approx(14.5)


@pytest.mark.parametrize("level", [0.0, 1.5, -2.5, 2.5, -1.0])
def test_inside_band_reports_no_surprise(tmp_path: Path, level: float) -> None:
    assert abs(level) <= NO_SURPRISE_BAND_BPS
    parquet = _write_parquet(
        tmp_path / "mp_surprises.parquet",
        rows=[_row("2026-04-29", level)],
    )

    out = load_latest_mp_surprise(parquet)

    assert out.direction == "no_surprise"
    assert out.magnitude_bps == pytest.approx(abs(level))


def test_intermeeting_flag_propagates(tmp_path: Path) -> None:
    parquet = _write_parquet(
        tmp_path / "mp_surprises.parquet",
        rows=[_row("2020-03-15", -65.0, is_intermeeting=True, ff_target_prior=1.125)],
    )

    out = load_latest_mp_surprise(parquet)

    assert out.is_intermeeting is True
    assert out.direction == "dovish"
    assert out.ff_target_prior_bps == pytest.approx(112.5)


def test_latest_row_selected_regardless_of_file_ordering(tmp_path: Path) -> None:
    """The wrapper must sort by event_date — file insertion order is not load-bearing."""

    parquet = _write_parquet(
        tmp_path / "mp_surprises.parquet",
        rows=[
            _row("2026-04-29", 8.0),
            _row("2024-01-31", 20.0),  # earliest, written second
            _row("2026-01-28", -4.0),
        ],
    )

    out = load_latest_mp_surprise(parquet)

    assert out.event_date == "2026-04-29"
    assert out.mp_surprise_level_bps == pytest.approx(8.0)


def test_null_prior_target_yields_none(tmp_path: Path) -> None:
    parquet = _write_parquet(
        tmp_path / "mp_surprises.parquet",
        rows=[_row("2026-04-29", -14.5, ff_target_prior=None)],
    )

    out = load_latest_mp_surprise(parquet)

    assert out.ff_target_prior_bps is None


def test_missing_parquet_raises_unavailable(tmp_path: Path) -> None:
    missing = tmp_path / "nope.parquet"

    with pytest.raises(MpSurpriseUnavailable):
        load_latest_mp_surprise(missing)


def test_empty_parquet_raises_unavailable(tmp_path: Path) -> None:
    parquet = tmp_path / "mp_surprises.parquet"
    pd.DataFrame(
        columns=[
            "event_date",
            "mp_surprise_level",
            "is_intermeeting",
            "ff_target_prior",
        ]
    ).to_parquet(parquet, index=False)

    with pytest.raises(MpSurpriseUnavailable):
        load_latest_mp_surprise(parquet)


def test_null_latest_level_raises_unavailable(tmp_path: Path) -> None:
    parquet = _write_parquet(
        tmp_path / "mp_surprises.parquet",
        rows=[_row("2026-04-29", None)],
    )

    with pytest.raises(MpSurpriseUnavailable):
        load_latest_mp_surprise(parquet)


def test_null_event_date_raises_unavailable(tmp_path: Path) -> None:
    """A NaN ``event_date`` on the latest row must surface as a structured
    unavailable instead of leaking ``'nan'`` onto the wire chip."""

    parquet = tmp_path / "mp_surprises.parquet"
    pd.DataFrame(
        [
            {
                "event_date": None,
                "meeting_id": 20260429,
                "ff_target_prior": 3.625,
                "ff_target_after": 3.625,
                "mp_surprise_level": 12.0,
                "mp_surprise_path_factor": 0.0,
                "is_intermeeting": False,
                "methodology": "ois_proxy",
            }
        ]
    ).to_parquet(parquet, index=False)

    with pytest.raises(MpSurpriseUnavailable, match="event_date"):
        load_latest_mp_surprise(parquet)


def test_empty_event_date_raises_unavailable(tmp_path: Path) -> None:
    """An empty-after-strip ``event_date`` must not silently coerce to a
    blank chip — the wrapper rejects the row up front."""

    parquet = tmp_path / "mp_surprises.parquet"
    pd.DataFrame(
        [
            {
                "event_date": "   ",
                "meeting_id": 20260429,
                "ff_target_prior": 3.625,
                "ff_target_after": 3.625,
                "mp_surprise_level": 12.0,
                "mp_surprise_path_factor": 0.0,
                "is_intermeeting": False,
                "methodology": "ois_proxy",
            }
        ]
    ).to_parquet(parquet, index=False)

    with pytest.raises(MpSurpriseUnavailable, match="event_date"):
        load_latest_mp_surprise(parquet)


def test_null_is_intermeeting_raises_unavailable(tmp_path: Path) -> None:
    """A NaN ``is_intermeeting`` on the latest row must surface as a
    structured unavailable instead of coercing ``float('nan')`` to ``True``
    via ``bool(...)`` and emitting a misleading flag on the chip."""

    parquet = tmp_path / "mp_surprises.parquet"
    pd.DataFrame(
        [
            {
                "event_date": "2026-04-29",
                "meeting_id": 20260429,
                "ff_target_prior": 3.625,
                "ff_target_after": 3.625,
                "mp_surprise_level": 12.0,
                "mp_surprise_path_factor": 0.0,
                "is_intermeeting": float("nan"),
                "methodology": "ois_proxy",
            }
        ]
    ).to_parquet(parquet, index=False)

    with pytest.raises(MpSurpriseUnavailable, match="is_intermeeting"):
        load_latest_mp_surprise(parquet)


def test_endpoint_returns_503_when_parquet_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """``GET /fomc/latest-mp-surprise`` degrades to a structured 503."""

    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    import app.main as main_mod
    from app.services import mp_surprise_service

    def _raise(*_a: object, **_kw: object) -> None:
        raise mp_surprise_service.MpSurpriseUnavailable("parquet missing in test")

    monkeypatch.setattr(mp_surprise_service, "load_latest_mp_surprise", _raise)

    client = TestClient(main_mod.app)
    response = client.get("/fomc/latest-mp-surprise")

    assert response.status_code == 503
    detail = response.json()["detail"]
    assert detail["error"] == "mp_surprise_unavailable"
    assert "parquet missing in test" in detail["message"]


def test_endpoint_returns_payload_when_parquet_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    import app.main as main_mod
    from app.schemas import MonetaryPolicySurpriseResponse
    from app.services import mp_surprise_service

    canned = MonetaryPolicySurpriseResponse(
        event_date="2026-04-29",
        mp_surprise_level_bps=12.0,
        direction="hawkish",
        magnitude_bps=12.0,
        is_intermeeting=False,
        ff_target_prior_bps=362.5,
    )
    monkeypatch.setattr(
        mp_surprise_service,
        "load_latest_mp_surprise",
        lambda *_a, **_kw: canned,
    )

    client = TestClient(main_mod.app)
    response = client.get("/fomc/latest-mp-surprise")

    assert response.status_code == 200
    body = response.json()
    assert body["event_date"] == "2026-04-29"
    assert body["direction"] == "hawkish"
    assert body["magnitude_bps"] == pytest.approx(12.0)
    assert body["ff_target_prior_bps"] == pytest.approx(362.5)
