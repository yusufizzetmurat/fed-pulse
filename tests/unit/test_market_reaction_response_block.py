"""Cover the /analyze ``rates_reaction`` block (#293).

The multi-target retrofit from #292 mounts up to three rates heads
(``2y`` / ``5y`` / ``terminal``) on top of the shared encoder, then
exposes the per-head cards on :class:`AnalyzeResponse.rates_reaction`
so the panel surface in #293 can read off the unified /analyze
payload without a second roundtrip. The field is populated via
``_safe_rates_reaction``, a defensive wrapper over
:func:`build_market_reaction_panel`.

These tests pin:

- the schema field exists, defaults to ``None``, and validates as a
  list of :class:`RatesReactionCard`;
- ``_safe_rates_reaction`` returns ``None`` when the panel builder
  degrades to a structured status payload (legacy single-head
  checkpoint) so a pre-#292 response shape is byte-identical;
- the helper returns an empty list when the rates heads are mounted
  but the per-event forward produced no rows (active, no read);
- the helper short-circuits to ``None`` on a builder exception so
  /analyze never 500s on a rates inference path crash;
- the derived dicts hydrate cleanly into the pydantic card.
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Schema surface


def test_analyze_response_carries_rates_reaction_field() -> None:
    """:class:`AnalyzeResponse` must declare the new ``rates_reaction`` field."""

    from app.schemas import AnalyzeResponse

    assert "rates_reaction" in AnalyzeResponse.model_fields
    field = AnalyzeResponse.model_fields["rates_reaction"]
    # Optional + defaults to None so a legacy single-head checkpoint
    # serialises to the pre-#292 response shape with the field absent
    # in non-default model_dump(exclude_none=True) callers.
    assert field.default is None


def test_rates_reaction_card_validates_point_only() -> None:
    """A point-only card (no conformal sidecar, no aux classifier) must validate."""

    from app.schemas import RatesReactionCard

    card = RatesReactionCard(head="2y", point_bps=4.5)
    assert card.head == "2y"
    assert card.point_bps == pytest.approx(4.5)
    assert card.lower_bps is None
    assert card.upper_bps is None
    assert card.coverage is None
    assert card.directional_bucket is None
    assert card.bucket_probabilities is None
    assert card.predicted_set is None


def test_rates_reaction_card_validates_full_interval() -> None:
    """A populated conformal interval + aux classifier round-trips through the schema."""

    from app.schemas import RatesReactionCard

    card = RatesReactionCard(
        head="terminal",
        point_bps=10.0,
        lower_bps=5.0,
        upper_bps=15.0,
        coverage=0.8,
        directional_bucket="tightening",
        bucket_probabilities={"easing": 0.05, "neutral": 0.10, "tightening": 0.85},
        predicted_set=["tightening"],
    )
    assert card.lower_bps == pytest.approx(5.0)
    assert card.upper_bps == pytest.approx(15.0)
    assert card.coverage == pytest.approx(0.8)
    assert card.directional_bucket == "tightening"
    assert card.predicted_set == ["tightening"]


def test_rates_reaction_card_field_type_is_list_of_cards() -> None:
    """The schema annotation must be ``list[RatesReactionCard] | None`` (#293 acceptance)."""

    from app.schemas import AnalyzeResponse, RatesReactionCard

    field = AnalyzeResponse.model_fields["rates_reaction"]
    annotation = field.annotation
    # Pydantic stores the resolved union; sniff the args for the
    # element type so the cardinality (list of cards, not a single
    # card and not a list of dicts) is locked.
    annotation_str = str(annotation)
    assert "RatesReactionCard" in annotation_str
    assert "list" in annotation_str.lower()
    # The card type itself must export the head + point_bps surface
    # downstream consumers rely on.
    assert "head" in RatesReactionCard.model_fields
    assert "point_bps" in RatesReactionCard.model_fields


# ---------------------------------------------------------------------------
# Helper short-circuit behaviour


def test_safe_rates_reaction_returns_none_on_status_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A structured ``status=...`` payload from the builder yields ``None``.

    Mirrors the legacy single-head checkpoint branch: the panel
    builder emits ``{"status": "not_classification_mode"}`` and the
    helper drops it so the response field is absent rather than
    carrying an unusable status dict.
    """

    import app.main as main_mod
    from app.services import forecaster as forecaster_service

    monkeypatch.setattr(
        forecaster_service,
        "build_market_reaction_panel",
        lambda _vectors: {"status": "not_classification_mode"},
    )
    out = main_mod._safe_rates_reaction([])
    assert out is None


def test_safe_rates_reaction_returns_none_on_non_dict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The builder returning ``None`` short-circuits to ``None``."""

    import app.main as main_mod
    from app.services import forecaster as forecaster_service

    monkeypatch.setattr(
        forecaster_service, "build_market_reaction_panel", lambda _vectors: None
    )
    assert main_mod._safe_rates_reaction([]) is None


def test_safe_rates_reaction_empty_list_when_heads_active_no_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Heads mounted but no rows emitted -> empty list (active, no read)."""

    import app.main as main_mod
    from app.services import forecaster as forecaster_service

    monkeypatch.setattr(
        forecaster_service,
        "build_market_reaction_panel",
        lambda _vectors: {"rates": [], "vol_regime": None},
    )
    out = main_mod._safe_rates_reaction([])
    assert out == []


def test_safe_rates_reaction_returns_rows_when_panel_populated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A populated panel surfaces the rates rows verbatim."""

    import app.main as main_mod
    from app.services import forecaster as forecaster_service

    payload = {
        "rates": [
            {
                "head": "2y",
                "point_bps": 4.5,
                "lower_bps": 1.0,
                "upper_bps": 8.0,
                "coverage": 0.8,
                "directional_bucket": "tightening",
                "bucket_probabilities": {
                    "easing": 0.1,
                    "neutral": 0.3,
                    "tightening": 0.6,
                },
                "predicted_set": ["tightening"],
            }
        ],
        "vol_regime": None,
    }
    monkeypatch.setattr(
        forecaster_service, "build_market_reaction_panel", lambda _vectors: payload
    )
    out = main_mod._safe_rates_reaction([])
    assert out is not None
    assert len(out) == 1
    assert out[0]["head"] == "2y"
    assert out[0]["point_bps"] == pytest.approx(4.5)


def test_safe_rates_reaction_swallows_builder_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A panel-builder crash degrades to ``None`` so /analyze stays up."""

    import app.main as main_mod
    from app.services import forecaster as forecaster_service

    def _raises(_vectors):
        raise RuntimeError("boom")

    monkeypatch.setattr(forecaster_service, "build_market_reaction_panel", _raises)
    assert main_mod._safe_rates_reaction([]) is None


def test_safe_rates_reaction_rows_hydrate_into_pydantic_card(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each row from the helper must hydrate cleanly into ``RatesReactionCard``."""

    import app.main as main_mod
    from app.schemas import RatesReactionCard
    from app.services import forecaster as forecaster_service

    payload = {
        "rates": [
            {
                "head": "5y",
                "point_bps": -2.0,
                "lower_bps": None,
                "upper_bps": None,
                "coverage": None,
                "directional_bucket": None,
                "bucket_probabilities": None,
                "predicted_set": None,
            }
        ],
        "vol_regime": None,
    }
    monkeypatch.setattr(
        forecaster_service, "build_market_reaction_panel", lambda _vectors: payload
    )
    out = main_mod._safe_rates_reaction([])
    assert out is not None
    hydrated = [RatesReactionCard(**row) for row in out]
    assert hydrated[0].head == "5y"
    assert hydrated[0].point_bps == pytest.approx(-2.0)
    assert hydrated[0].directional_bucket is None
