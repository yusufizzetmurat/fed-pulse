"""Unit tests for app.evaluation.transfer_matrix."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.evaluation import transfer_matrix as tm


def _write_registry(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


@pytest.fixture
def two_bank_package(tmp_path: Path) -> Path:
    package_dir = tmp_path / "processed" / "tp_v1"
    _write_registry(
        package_dir / "registry_normalized.jsonl",
        [
            {
                "record_id": "ecb_1",
                "text": "rate path firm",
                "event_date": "2024-02-01",
                "mapped_label": "hawkish",
                "source": "gtfintechlab_european_central_bank",
                "provenance": "peer_reviewed_cross_bank",
                "sample_weight": 0.0,
            },
            {
                "record_id": "ecb_2",
                "text": "ease ahead",
                "event_date": "2024-03-01",
                "mapped_label": "dovish",
                "source": "gtfintechlab_european_central_bank",
                "provenance": "peer_reviewed_cross_bank",
                "sample_weight": 0.0,
            },
            {
                "record_id": "boj_1",
                "text": "accommodative",
                "event_date": "2024-04-01",
                "mapped_label": "dovish",
                "source": "gtfintechlab_bank_of_japan",
                "provenance": "peer_reviewed_cross_bank",
                "sample_weight": 0.0,
            },
        ],
    )
    return package_dir


def test_parse_model_checkpoints_handles_alias_pairs() -> None:
    parsed = tm._parse_model_checkpoints("finbert=/tmp/a,bge_large=/tmp/b")
    assert parsed == {"finbert": "/tmp/a", "bge_large": "/tmp/b"}


def test_parse_model_checkpoints_rejects_missing_equals() -> None:
    with pytest.raises(ValueError, match="alias=path"):
        tm._parse_model_checkpoints("bad-entry")


def test_split_banks_resolves_short_aliases() -> None:
    assert tm._split_banks("ecb,boj") == [
        "gtfintechlab_european_central_bank",
        "gtfintechlab_bank_of_japan",
    ]


def test_split_banks_defaults_to_all_five() -> None:
    out = tm._split_banks("")
    assert len(out) == 5
    assert "gtfintechlab_reserve_bank_of_australia" in out


def test_build_matrix_with_oracle_predictor(two_bank_package: Path) -> None:
    def oracle(rows):
        labels = [r.label for r in rows]
        return list(labels), list(labels), [0.5] * len(labels)

    matrix = tm.build_matrix(
        package_dir=two_bank_package,
        model_checkpoints={"finbert": "/tmp/a"},
        banks=[
            "gtfintechlab_european_central_bank",
            "gtfintechlab_bank_of_japan",
        ],
        seeds=[11, 29],
        n_resamples=50,
        coverage=0.9,
        rng_seed=11,
        predict_fn=oracle,
    )
    assert matrix["models"] == ["finbert"]
    per_bank = matrix["by_model"]["finbert"]["per_bank"]
    assert {"gtfintechlab_european_central_bank", "gtfintechlab_bank_of_japan"} == set(per_bank.keys())
    macro = per_bank["gtfintechlab_european_central_bank"]["summary"]["macro_f1"]
    # Oracle prediction; macro-F1 averages over dovish/hawkish/neutral and the
    # ECB fixture has no neutral row so the ceiling for an oracle is 2/3.
    assert macro["point"] == pytest.approx(2 / 3)
    accuracy = per_bank["gtfintechlab_european_central_bank"]["summary"]["accuracy"]
    assert accuracy["point"] == pytest.approx(1.0)


def test_render_csv_emits_header_and_per_cell_row(two_bank_package: Path) -> None:
    def oracle(rows):
        labels = [r.label for r in rows]
        return list(labels), list(labels), [0.5] * len(labels)

    matrix = tm.build_matrix(
        package_dir=two_bank_package,
        model_checkpoints={"finbert": "/tmp/a"},
        banks=["gtfintechlab_european_central_bank"],
        seeds=[11],
        n_resamples=20,
        rng_seed=11,
        predict_fn=oracle,
    )
    csv_text = tm.render_csv(matrix)
    assert "model,bank,support,macro_f1_mean" in csv_text
    assert "finbert,gtfintechlab_european_central_bank" in csv_text


def test_render_markdown_includes_ci_format(two_bank_package: Path) -> None:
    def oracle(rows):
        labels = [r.label for r in rows]
        return list(labels), list(labels), [0.5] * len(labels)

    matrix = tm.build_matrix(
        package_dir=two_bank_package,
        model_checkpoints={"finbert": "/tmp/a"},
        banks=["gtfintechlab_european_central_bank"],
        seeds=[11],
        n_resamples=20,
        coverage=0.95,
        rng_seed=11,
        predict_fn=oracle,
    )
    md = tm.render_markdown(matrix)
    assert "95% CI" in md
    assert "`finbert`" in md
