from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "run_regime_baseline_tiers.py"


@pytest.fixture(scope="module")
def harness_module():
    spec = importlib.util.spec_from_file_location(
        "fed_pulse_regime_baseline_tiers", _SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------


def test_parses_required_package_id(harness_module) -> None:
    args = harness_module._parse_args(
        ["--training-package-id", "pkg-abc"]
    )
    assert args.training_package_id == "pkg-abc"
    # Defaults: 5 official seeds, 4 walk-forward folds, 3-class regime.
    assert args.seeds == [11, 29, 47, 71, 97]
    assert args.folds == ["wf_fold_1", "wf_fold_2", "wf_fold_3", "wf_fold_4"]
    assert args.vol_regime_classes == 3
    assert args.tiers == (
        "tier1_market_only",
        "tier2_market_rich",
        "tier3_market_rich_nlp",
    )


def test_rejects_missing_package_id(harness_module) -> None:
    with pytest.raises(SystemExit):
        harness_module._parse_args([])


# ---------------------------------------------------------------------------
# Per-tier argument overlays
# ---------------------------------------------------------------------------


def test_tier1_disables_rich_and_text(harness_module, tmp_path) -> None:
    args = harness_module._parse_args(["--training-package-id", "pkg"])
    cmd = harness_module._tier_args(
        "tier1_market_only", args, tmp_path / "tier1.json"
    )
    assert "--no-rich-features" in cmd
    assert "--text-encoder" in cmd
    encoder_idx = cmd.index("--text-encoder")
    assert cmd[encoder_idx + 1] == "none"
    # Classification dispatch must be on for every tier.
    assert "--output-mode" in cmd
    assert cmd[cmd.index("--output-mode") + 1] == "classification"
    assert "--vol-regime-classes" in cmd
    assert cmd[cmd.index("--vol-regime-classes") + 1] == "3"


def test_tier2_enables_rich_but_no_text(harness_module, tmp_path) -> None:
    args = harness_module._parse_args(["--training-package-id", "pkg"])
    cmd = harness_module._tier_args(
        "tier2_market_rich", args, tmp_path / "tier2.json"
    )
    assert "--rich-features" in cmd
    assert "--no-rich-features" not in cmd
    assert cmd[cmd.index("--text-encoder") + 1] == "none"


def test_tier3_uses_configured_nlp_encoder(harness_module, tmp_path) -> None:
    args = harness_module._parse_args(
        [
            "--training-package-id",
            "pkg",
            "--nlp-text-encoder",
            "voyage_finance_2",
        ]
    )
    cmd = harness_module._tier_args(
        "tier3_market_rich_nlp", args, tmp_path / "tier3.json"
    )
    assert "--rich-features" in cmd
    assert cmd[cmd.index("--text-encoder") + 1] == "voyage_finance_2"
    # Adapter dims forwarded only on tier 3.
    assert "--text-adapter-dims" in cmd


def test_tier2_does_not_forward_text_adapter_dims(harness_module, tmp_path) -> None:
    args = harness_module._parse_args(["--training-package-id", "pkg"])
    cmd = harness_module._tier_args(
        "tier2_market_rich", args, tmp_path / "tier2.json"
    )
    assert "--text-adapter-dims" not in cmd


def test_unknown_tier_raises(harness_module, tmp_path) -> None:
    args = harness_module._parse_args(["--training-package-id", "pkg"])
    with pytest.raises(ValueError, match="Unknown tier"):
        harness_module._tier_args("tier_bogus", args, tmp_path / "x.json")


# ---------------------------------------------------------------------------
# Dry-run end-to-end
# ---------------------------------------------------------------------------


def test_dry_run_prints_commands_for_each_tier(harness_module, tmp_path, capsys) -> None:
    rc = harness_module.main(
        [
            "--training-package-id",
            "pkg-dry",
            "--report-root",
            str(tmp_path),
            "--dry-run",
        ]
    )
    assert rc == 0
    captured = capsys.readouterr().out
    assert "tier1_market_only" in captured
    assert "tier2_market_rich" in captured
    assert "tier3_market_rich_nlp" in captured
    # Each per-tier line records the cmd that would have run.
    assert captured.count("[regime_tiers] cmd:") == 3


def test_dry_run_can_restrict_tier_subset(harness_module, tmp_path, capsys) -> None:
    rc = harness_module.main(
        [
            "--training-package-id",
            "pkg-subset",
            "--report-root",
            str(tmp_path),
            "--tiers",
            "tier1_market_only",
            "tier3_market_rich_nlp",
            "--dry-run",
        ]
    )
    assert rc == 0
    captured = capsys.readouterr().out
    assert "tier1_market_only" in captured
    assert "tier3_market_rich_nlp" in captured
    assert "tier2_market_rich" not in captured
