"""Unit tests for app.evaluation.transfer_matrix."""

from __future__ import annotations

import json
import math
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


def test_parse_model_checkpoints_handles_single_alias_pairs() -> None:
    parsed = tm._parse_model_checkpoints("finbert=/tmp/a,bge_large=/tmp/b")
    assert parsed == {"finbert": ["/tmp/a"], "bge_large": ["/tmp/b"]}


def test_parse_model_checkpoints_accumulates_repeated_aliases() -> None:
    """Same alias appearing multiple times → list of checkpoints for that alias."""

    parsed = tm._parse_model_checkpoints(
        "finbert=/tmp/s11,finbert=/tmp/s29,finbert=/tmp/s47,bge_large=/tmp/s11"
    )
    assert parsed["finbert"] == ["/tmp/s11", "/tmp/s29", "/tmp/s47"]
    assert parsed["bge_large"] == ["/tmp/s11"]


def test_parse_model_checkpoints_rejects_missing_equals() -> None:
    with pytest.raises(ValueError, match="alias=path"):
        tm._parse_model_checkpoints("bad-entry")


def test_parse_model_checkpoints_rejects_empty_alias_or_path() -> None:
    with pytest.raises(ValueError, match="empty"):
        tm._parse_model_checkpoints("=path")


def test_split_banks_resolves_short_aliases() -> None:
    assert tm._split_banks("ecb,boj") == [
        "gtfintechlab_european_central_bank",
        "gtfintechlab_bank_of_japan",
    ]


def test_split_banks_defaults_to_all_five() -> None:
    out = tm._split_banks("")
    assert len(out) == 5
    assert "gtfintechlab_reserve_bank_of_australia" in out


def _oracle_predict(rows):
    labels = [r.label for r in rows]
    return list(labels), list(labels), [0.5] * len(labels)


def _constant_predict(value: str):
    def _impl(rows):
        return [r.label for r in rows], [value] * len(rows), [0.5] * len(rows)

    return _impl


def test_build_matrix_single_checkpoint_emits_point_estimate(two_bank_package: Path) -> None:
    matrix = tm.build_matrix(
        package_dir=two_bank_package,
        model_checkpoints={"finbert": ["/tmp/a"]},
        banks=["gtfintechlab_european_central_bank"],
        n_resamples=50,
        rng_seed=11,
        predict_fn=_oracle_predict,
    )
    macro = matrix["by_model"]["finbert"]["per_bank"][
        "gtfintechlab_european_central_bank"
    ]["summary"]["macro_f1"]
    assert macro["ci_kind"] == "point_estimate"
    assert macro["n_checkpoints"] == 1
    assert "lo" not in macro and "hi" not in macro
    # ECB has no neutral row so the oracle ceiling is 2/3.
    assert macro["point"] == pytest.approx(2 / 3)


def test_build_matrix_multi_checkpoint_emits_real_ci(two_bank_package: Path) -> None:
    """With 3 distinct predictors yielding genuinely different macro-F1s,
    the bootstrap CI must have non-zero width."""

    matrix = tm.build_matrix(
        package_dir=two_bank_package,
        # Three different "checkpoints" — the test uses predict_fn so paths
        # are nominal — wired through one predict_fn that varies by call.
        model_checkpoints={"finbert": ["/tmp/a", "/tmp/b", "/tmp/c"]},
        banks=["gtfintechlab_european_central_bank"],
        n_resamples=100,
        rng_seed=11,
        # Predictor that walks oracle → constant-neutral → constant-hawkish
        # by mutating an external counter so each "checkpoint" yields a
        # different macro-F1.
        predict_fn=_make_varying_predictor(),
    )
    macro = matrix["by_model"]["finbert"]["per_bank"][
        "gtfintechlab_european_central_bank"
    ]["summary"]["macro_f1"]
    assert macro["ci_kind"] == "block_bootstrap"
    assert macro["n_checkpoints"] == 3
    # CI has real width because the 3 predictors disagree.
    assert macro["hi"] > macro["lo"]


def _make_varying_predictor():
    calls = {"n": 0}

    def _impl(rows):
        idx = calls["n"]
        calls["n"] += 1
        if idx == 0:
            return [r.label for r in rows], [r.label for r in rows], [0.5] * len(rows)
        if idx == 1:
            return [r.label for r in rows], ["neutral"] * len(rows), [0.5] * len(rows)
        return [r.label for r in rows], ["hawkish"] * len(rows), [0.5] * len(rows)

    return _impl


def test_render_csv_emits_n_checkpoints_and_ci_kind_columns(two_bank_package: Path) -> None:
    matrix = tm.build_matrix(
        package_dir=two_bank_package,
        model_checkpoints={"finbert": ["/tmp/a"]},
        banks=["gtfintechlab_european_central_bank"],
        n_resamples=20,
        rng_seed=11,
        predict_fn=_oracle_predict,
    )
    csv_text = tm.render_csv(matrix)
    assert "n_checkpoints,ci_kind" in csv_text
    assert ",point_estimate," in csv_text


def test_render_markdown_marks_point_estimate_when_single_checkpoint(two_bank_package: Path) -> None:
    matrix = tm.build_matrix(
        package_dir=two_bank_package,
        model_checkpoints={"finbert": ["/tmp/a"]},
        banks=["gtfintechlab_european_central_bank"],
        n_resamples=20,
        coverage=0.95,
        rng_seed=11,
        predict_fn=_oracle_predict,
    )
    md = tm.render_markdown(matrix, coverage=0.95)
    assert "(point)" in md
    assert "95% CI when n≥2" in md


def test_render_markdown_emits_ci_when_multi_checkpoint(two_bank_package: Path) -> None:
    matrix = tm.build_matrix(
        package_dir=two_bank_package,
        model_checkpoints={"finbert": ["/tmp/a", "/tmp/b", "/tmp/c"]},
        banks=["gtfintechlab_european_central_bank"],
        n_resamples=100,
        rng_seed=11,
        predict_fn=_make_varying_predictor(),
    )
    md = tm.render_markdown(matrix)
    # Multi-checkpoint cell renders as "point [lo, hi]" — square-brackets present.
    assert " [" in md and "]" in md


def test_scrub_nan_converts_floats_to_none() -> None:
    payload = {
        "ok": 0.5,
        "nan_field": float("nan"),
        "inf_field": float("inf"),
        "list": [1.0, float("nan"), {"nested": float("-inf")}],
    }
    out = tm._scrub_nan(payload)
    assert out["ok"] == 0.5
    assert out["nan_field"] is None
    assert out["inf_field"] is None
    assert out["list"][1] is None
    assert out["list"][2]["nested"] is None
    # Round-trips through strict JSON.
    json.dumps(out, allow_nan=False)


def test_scrub_nan_preserves_normal_data() -> None:
    payload = {"a": 1, "b": [1, 2, 3], "c": {"d": "string"}}
    assert tm._scrub_nan(payload) == payload


def test_build_matrix_failures_get_captured_not_raised(tmp_path: Path) -> None:
    package_dir = tmp_path / "processed" / "tp_empty"
    _write_registry(package_dir / "registry_normalized.jsonl", [])

    def _exploding(rows):
        raise RuntimeError("boom")

    matrix = tm.build_matrix(
        package_dir=package_dir,
        model_checkpoints={"finbert": ["/tmp/a"]},
        banks=["gtfintechlab_european_central_bank"],
        n_resamples=20,
        rng_seed=11,
        predict_fn=_exploding,
    )
    failures = matrix["by_model"]["finbert"].get("failures") or []
    assert any("boom" in f.get("error", "") or "No labeled rows" in f.get("error", "") for f in failures)
