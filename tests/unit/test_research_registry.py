"""Unit tests for the quant-facing encoder bake-off registry (#299)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.services.research_artifacts import (
    REGISTRY_MANIFEST_PATH,
    load_research_registry,
)


def test_manifest_exists_and_is_valid_json() -> None:
    payload = json.loads(REGISTRY_MANIFEST_PATH.read_text(encoding="utf-8"))
    assert payload["baseline"]["dual_f1"] == pytest.approx(0.3773)
    assert payload["baseline"]["cls_f1"] == pytest.approx(0.3934)
    assert len(payload["rows"]) == 5
    aliases = {row["encoder_alias"] for row in payload["rows"]}
    assert aliases == {
        "bge_large_en_v15",
        "finbert_fed_adjacent",
        "nomic_embed_text_v15",
        "finbert_fed_adjacent_xbank",
        "voyage_finance_2",
    }


def test_dual_surface_default_filters_out_voyage() -> None:
    out = load_research_registry(surface="dual")
    assert out["available"] is True
    assert out["surface"] == "dual"
    aliases = [row["encoder_alias"] for row in out["rows"]]
    # voyage is the only dual-surface loser (-0.013).
    assert "voyage_finance_2" not in aliases
    assert "finbert_fed_adjacent_xbank" in aliases  # xbank is +0.010 on dual
    assert out["rejected_count"] == 1


def test_cls_surface_default_filters_out_voyage_and_xbank() -> None:
    out = load_research_registry(surface="cls")
    assert out["surface"] == "cls"
    aliases = [row["encoder_alias"] for row in out["rows"]]
    # Both voyage (-0.023) and xbank (-0.022) lose on cls.
    assert "voyage_finance_2" not in aliases
    assert "finbert_fed_adjacent_xbank" not in aliases
    assert "bge_large_en_v15" in aliases
    assert "finbert_fed_adjacent" in aliases
    assert "nomic_embed_text_v15" in aliases
    assert out["rejected_count"] == 2


def test_include_rejected_returns_full_table() -> None:
    out = load_research_registry(surface="dual", include_rejected=True)
    assert len(out["rows"]) == 5
    assert out["rejected_count"] == 0
    voyage = next(r for r in out["rows"] if r["encoder_alias"] == "voyage_finance_2")
    assert voyage["is_winner"] is False
    assert voyage["delta_dual"] < 0


def test_deltas_round_to_four_decimals_and_match_wiki() -> None:
    out = load_research_registry(surface="dual", include_rejected=True)
    by_alias = {row["encoder_alias"]: row for row in out["rows"]}
    # Wiki §6.41 ground truth deltas.
    assert by_alias["bge_large_en_v15"]["delta_dual"] == pytest.approx(0.0574, abs=1e-4)
    assert by_alias["finbert_fed_adjacent"]["delta_dual"] == pytest.approx(0.0428, abs=1e-4)
    assert by_alias["nomic_embed_text_v15"]["delta_dual"] == pytest.approx(0.0225, abs=1e-4)
    assert by_alias["voyage_finance_2"]["delta_dual"] == pytest.approx(-0.0132, abs=1e-4)


def test_unsupported_surface_raises() -> None:
    with pytest.raises(ValueError, match="dual|cls"):
        load_research_registry(surface="regression")


def test_manifest_carries_provenance() -> None:
    out = load_research_registry(surface="dual")
    assert out["training_package_id"] == "canonical"
    assert out["head"] == "dual_lstm"
    assert out["seeds"] == [11, 29, 47, 71, 97]
    assert "06_Deep_Learning_Roadmap" in out["source_wiki_section"]
