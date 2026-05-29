"""Unit tests for the family zero-out smoke (#505 A.1.b)."""

from __future__ import annotations

from unittest.mock import patch

from scripts.run_family_zeroout_smoke import (
    FAMILIES,
    _baseline_flags,
    run_smoke,
)


def test_baseline_flags_turn_every_family_on() -> None:
    flags = _baseline_flags()
    assert set(flags) == {f"use_{f}" for f in FAMILIES}
    assert all(flags.values())


def test_family_list_covers_every_loader_use_flag() -> None:
    """The loader's per-family ablation flags must each have a matching
    entry in FAMILIES so the smoke covers them. Lock the list here.
    """

    expected = {
        "credibility",
        "linguistic",
        "mp_surprise",
        "multi_axis",
        "llm_features",
        "retrieval_analogs",
        "regime_conditioning",
        "sep",
        "press_conf",
        "statement_delta",
        "vote_features",
    }
    assert set(FAMILIES) == expected


def test_run_smoke_flags_silent_zero_family() -> None:
    """When a family OFF returns the SAME F1 as the baseline, the smoke
    flags it. Two families return identical metrics under the mock;
    both end up on the silent-zero list.
    """

    baseline = {"regime_f1_macro": 0.40, "regime_accuracy": 0.50}

    def fake_run_arm(*, loader_flags, **kwargs):
        # First call (all True) = baseline.
        if all(loader_flags.values()):
            return baseline
        # Identify which family is off by finding the False flag.
        off = [k for k, v in loader_flags.items() if not v]
        family = off[0].replace("use_", "")
        # Two families return identical-to-baseline metrics (silent zero
        # bug); the rest move the F1 by a small amount.
        if family in {"credibility", "press_conf"}:
            return {"regime_f1_macro": 0.40, "regime_accuracy": 0.50}
        return {"regime_f1_macro": 0.40 - 0.01, "regime_accuracy": 0.49}

    with patch(
        "scripts.run_family_zeroout_smoke._run_one_arm",
        side_effect=fake_run_arm,
    ):
        report = run_smoke(
            training_package_id="tp_x",
            fold_id="wf_fold_3",
            seed=11,
            epochs=2,
            hidden_size=64,
        )

    assert report["baseline_metrics"] == baseline
    families = [arm["family"] for arm in report["per_family"]]
    assert set(families) == set(FAMILIES)
    flagged = set(report["flagged_silent_zero"])
    assert flagged == {"credibility", "press_conf"}


def test_run_smoke_returns_empty_flag_list_when_every_family_moves_metric() -> None:
    """Every family OFF moves the F1; flag list stays empty."""

    baseline = {"regime_f1_macro": 0.42, "regime_accuracy": 0.51}

    def fake_run_arm(*, loader_flags, **kwargs):
        if all(loader_flags.values()):
            return baseline
        return {"regime_f1_macro": 0.42 - 0.01, "regime_accuracy": 0.50}

    with patch(
        "scripts.run_family_zeroout_smoke._run_one_arm",
        side_effect=fake_run_arm,
    ):
        report = run_smoke(
            training_package_id="tp_y",
            fold_id="wf_fold_3",
            seed=11,
            epochs=2,
            hidden_size=64,
        )

    assert report["flagged_silent_zero"] == []
    for arm in report["per_family"]:
        assert arm["delta_f1"] != 0.0
