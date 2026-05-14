from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("yaml")

from app.training.config_loader import AblationConfig, load_ablation_config


def test_loads_no_text_ablation():
    config = load_ablation_config(Path("configs/ablation_no_text.yaml"))
    assert isinstance(config, AblationConfig)
    assert config.name == "no_text"
    assert config.zero_text is True
    assert config.calendar_only is False
    assert config.feature_overrides == {"sentiment_score": 0.0}


def test_loads_calendar_only_ablation():
    config = load_ablation_config(Path("configs/ablation_calendar_only.yaml"))
    assert config.calendar_only is True
    assert config.feature_overrides["close_change_pct"] == 0.0


def test_rejects_unknown_keys(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: x\nmystery_key: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown keys"):
        load_ablation_config(bad)


def test_rejects_missing_name(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("text_channel: scalar\n", encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty 'name'"):
        load_ablation_config(bad)


def test_rejects_invalid_text_channel(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: x\ntext_channel: hybrid\n", encoding="utf-8")
    with pytest.raises(ValueError, match="text_channel must be"):
        load_ablation_config(bad)
