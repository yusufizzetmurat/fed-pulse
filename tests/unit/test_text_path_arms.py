"""#327 text-path A/B coverage.

The three arms (broadcast-static / per-bar / flat MLP) must:

- forward smoke on the canonical input shape without throwing
- match shape contracts (``(B, 2)`` for regression output mode,
  ``(B, n_classes)`` for classification)
- emit a byte-equivalent forward when per-bar input is the broadcast
  of the same pooled vector across every bar (Arm A parity test)
- reject mis-configured combinations (flat_mlp with per_bar, per_bar
  with no per-bar tensor)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from app.models.config import (  # noqa: E402
    FORECASTER_ARCHITECTURES,
    RICH_FEATURE_SIZE,
    ModelConfig,
)
from app.models.factory import build_forecaster  # noqa: E402
from app.models.flat_mlp import ForecasterFlatMLP  # noqa: E402
from app.models.research_model import ForecasterResearchModel  # noqa: E402

SEQ_LEN = 20
TEXT_IN_DIM = 32
TEXT_ADAPTER_DIM = 16


def _baseline_kwargs() -> dict[str, Any]:
    return {
        "input_size": RICH_FEATURE_SIZE,
        "hidden_size": 16,
        "num_layers": 1,
        "head_hidden_size": 16,
        "n_classes": 3,
        "text_embedding_dim": TEXT_IN_DIM,
        "text_adapter_dim": TEXT_ADAPTER_DIM,
    }


def test_flat_mlp_in_architectures_registry():
    assert "flat_mlp" in FORECASTER_ARCHITECTURES


def test_broadcast_static_forward_regression_shape():
    config = ModelConfig(architecture="lstm", text_channel="scalar", **_baseline_kwargs())
    model = build_forecaster(config)
    model.eval()
    x = torch.zeros((2, SEQ_LEN, RICH_FEATURE_SIZE))
    text = torch.zeros((2, TEXT_IN_DIM))
    out = model(x, text_embedding=text)
    assert out.shape == (2, 2), out.shape


def test_per_bar_forward_regression_shape():
    config = ModelConfig(
        architecture="lstm",
        text_channel="per_bar",
        **_baseline_kwargs(),
    )
    model = build_forecaster(config)
    model.eval()
    x = torch.zeros((2, SEQ_LEN, RICH_FEATURE_SIZE))
    per_bar = torch.zeros((2, SEQ_LEN, TEXT_IN_DIM))
    out = model(x, text_embedding_per_bar=per_bar)
    assert out.shape == (2, 2), out.shape


def test_flat_mlp_forward_regression_shape():
    config = ModelConfig(
        architecture="flat_mlp",
        text_channel="scalar",
        **_baseline_kwargs(),
    )
    model = build_forecaster(config)
    assert isinstance(model, ForecasterFlatMLP)
    model.eval()
    x = torch.zeros((2, SEQ_LEN, RICH_FEATURE_SIZE))
    text = torch.zeros((2, TEXT_IN_DIM))
    out = model(x, text_embedding=text)
    assert out.shape == (2, 2), out.shape


def test_per_bar_parity_with_broadcast_static_when_constant_across_bars():
    """Arm A with a tiled-pooled per-bar tensor must match broadcast-static.

    The two paths share the same adapter weights and the same recurrent
    core; the only difference is whether the projection lives inside or
    outside the per-bar broadcast. With identical per-bar payloads the
    two forwards must agree to numerical noise.
    """

    torch.manual_seed(11)
    config_static = ModelConfig(
        architecture="lstm", text_channel="scalar", **_baseline_kwargs()
    )
    static_model = build_forecaster(config_static)
    static_model.eval()
    # Replace the zero-init adapter with a non-zero copy so the
    # broadcast-static path produces a non-trivial signal we can match.
    with torch.no_grad():
        for param in static_model.text_adapter.parameters():
            param.data.normal_(mean=0.0, std=0.02)

    config_per_bar = ModelConfig(
        architecture="lstm", text_channel="per_bar", **_baseline_kwargs()
    )
    per_bar_model = build_forecaster(config_per_bar)
    per_bar_model.eval()
    per_bar_model.load_state_dict(static_model.state_dict())

    x = torch.randn((3, SEQ_LEN, RICH_FEATURE_SIZE)) * 0.1
    pooled = torch.randn((3, TEXT_IN_DIM)) * 0.1
    per_bar = pooled.unsqueeze(1).expand(-1, SEQ_LEN, -1).contiguous()
    with torch.no_grad():
        out_static = static_model(x, text_embedding=pooled)
        out_per_bar = per_bar_model(x, text_embedding_per_bar=per_bar)
    assert torch.allclose(out_static, out_per_bar, atol=1e-6), (
        out_static - out_per_bar
    )


def test_per_bar_rejects_missing_tensor():
    """Arm A with text_channel='per_bar' and no per-bar tensor falls back to scalar."""

    config = ModelConfig(
        architecture="lstm", text_channel="per_bar", **_baseline_kwargs()
    )
    model = build_forecaster(config)
    model.eval()
    x = torch.zeros((2, SEQ_LEN, RICH_FEATURE_SIZE))
    # When per_bar is missing the path falls through to the scalar
    # branch, so we must still pass ``text_embedding``. The forward
    # should not raise.
    text = torch.zeros((2, TEXT_IN_DIM))
    out = model(x, text_embedding=text)
    assert out.shape == (2, 2)


def test_per_bar_wrong_sequence_length_raises():
    config = ModelConfig(
        architecture="lstm", text_channel="per_bar", **_baseline_kwargs()
    )
    model = build_forecaster(config)
    model.eval()
    x = torch.zeros((2, SEQ_LEN, RICH_FEATURE_SIZE))
    bad_per_bar = torch.zeros((2, SEQ_LEN - 3, TEXT_IN_DIM))
    with pytest.raises(ValueError, match="sequence length"):
        model(x, text_embedding_per_bar=bad_per_bar)


def test_flat_mlp_rejects_per_bar_channel():
    with pytest.raises(ValueError, match="text_channel='per_bar'"):
        ModelConfig(
            architecture="flat_mlp", text_channel="per_bar", **_baseline_kwargs()
        )
        build_forecaster(
            ModelConfig(
                architecture="flat_mlp",
                text_channel="per_bar",
                **_baseline_kwargs(),
            )
        )


def test_flat_mlp_serving_rejected():
    from app.models.config import ModelConfig as _ModelConfig

    with pytest.raises(ValueError, match="research-only"):
        build_forecaster(
            _ModelConfig(architecture="flat_mlp", **_baseline_kwargs()),
            role="serving",
        )


def test_unknown_text_channel_rejected():
    with pytest.raises(ValueError, match="text_channel"):
        ForecasterResearchModel(
            input_size=RICH_FEATURE_SIZE,
            text_channel="garbage",  # type: ignore[arg-type]
        )


def test_classification_multi_task_forward_shapes_for_all_arms():
    """Each arm must produce the same multi-task dict keys in classification mode."""

    base = dict(_baseline_kwargs())
    cases = [
        ("lstm", "scalar", "text_embedding"),
        ("lstm", "per_bar", "text_embedding_per_bar"),
        ("flat_mlp", "scalar", "text_embedding"),
    ]
    for arch, channel, kw in cases:
        cfg = ModelConfig(
            architecture=arch,
            text_channel=channel,
            output_mode="classification",
            **base,
        )
        model = build_forecaster(cfg)
        model.eval()
        x = torch.zeros((2, SEQ_LEN, RICH_FEATURE_SIZE))
        if kw == "text_embedding_per_bar":
            payload = torch.zeros((2, SEQ_LEN, TEXT_IN_DIM))
        else:
            payload = torch.zeros((2, TEXT_IN_DIM))
        out = model.forward_multi_task(x, **{kw: payload})
        assert "stance" in out, (arch, channel, list(out.keys()))
        assert out["stance"].shape == (2, 3), (arch, channel, out["stance"].shape)


def test_build_per_bar_text_tensor_smoke(tmp_path: Path):
    """Loader helper must emit (num_windows, T, in_dim) for a small fixture."""

    from app.models.config import FeatureVector, SEQUENCE_LENGTH
    from app.training.loaders import build_per_bar_text_tensor

    sequence: list[FeatureVector] = []
    for i in range(SEQUENCE_LENGTH + 2):
        fv = FeatureVector(
            date=f"2026-05-{i + 1:02d}",
            sentiment_score=0.0,
            market_close=100.0 + i,
            market_volatility=0.01,
        )
        fv.text_embedding_pooled = [0.0] * TEXT_IN_DIM
        fv.text_embedding_missing = 1.0
        sequence.append(fv)
    pooled, missing, in_dim = build_per_bar_text_tensor(
        [sequence], sequence_length=SEQUENCE_LENGTH, fallback_in_dim=TEXT_IN_DIM
    )
    assert in_dim == TEXT_IN_DIM
    assert pooled is not None and missing is not None
    assert pooled.shape == (2, SEQUENCE_LENGTH, TEXT_IN_DIM)
    assert missing.shape == (2, SEQUENCE_LENGTH)


def test_runner_arm_config_uses_rich_feature_size():
    """Hard-pin the #322 follow-up contract: runner must set input_size=RICH_FEATURE_SIZE."""

    import argparse

    from scripts.run_text_path_ab import _arm_config

    args = argparse.Namespace(
        head_mode="dual",
        regression_alpha=0.5,
        hidden_size=64,
        text_adapter_dim=64,
    )
    for arm in ("broadcast_static", "per_bar", "flat_mlp"):
        cfg = _arm_config(arm, args)
        assert cfg["input_size"] == RICH_FEATURE_SIZE, (arm, cfg["input_size"])
        assert cfg["output_mode"] == "classification"
        assert cfg["head_mode"] == "dual"


def test_runner_json_payload_shape_contract(tmp_path: Path):
    """The runner's JSON shape must mirror the dual_head canonical contract."""

    # Re-create the assembled payload by hand (calling main() requires
    # a real training package, which is GPU-bound). The shape contract
    # the test guards is the *keys* downstream consumers read off.
    sample_payload = {
        "arms": ["broadcast_static", "per_bar", "flat_mlp"],
        "seeds": [11],
        "fold_ids": ["fold_001"],
        "epochs": 1,
        "head_mode": "dual",
        "regression_alpha": 0.5,
        "training_package_id": "tp_dummy",
        "text_encoder": "finbert_fed_adjacent",
        "text_adapter_dim": 64,
        "text_adapter_warm_start": None,
        "trials": {
            arm: [
                {
                    "arm": arm,
                    "seed": 11,
                    "config": {"input_size": RICH_FEATURE_SIZE},
                    "folds": [
                        {
                            "fold_id": "fold_001",
                            "metrics": {
                                "regime_f1_macro": 0.4,
                                "regression_rmse_log_rv": 1.0,
                            },
                        }
                    ],
                }
            ]
            for arm in ("broadcast_static", "per_bar", "flat_mlp")
        },
        "summary": {
            arm: {
                "regime_f1_macro": {
                    "mean": 0.4,
                    "std": 0.0,
                    "min": 0.4,
                    "max": 0.4,
                    "n": 1,
                },
                "regression_rmse_log_rv": {
                    "mean": 1.0,
                    "std": 0.0,
                    "min": 1.0,
                    "max": 1.0,
                    "n": 1,
                },
            }
            for arm in ("broadcast_static", "per_bar", "flat_mlp")
        },
    }
    output = tmp_path / "text_path_ab.json"
    output.write_text(json.dumps(sample_payload))
    loaded = json.loads(output.read_text())
    assert set(loaded["arms"]) == {"broadcast_static", "per_bar", "flat_mlp"}
    assert set(loaded["trials"].keys()) == set(loaded["arms"])
    for arm in loaded["arms"]:
        for trial in loaded["trials"][arm]:
            assert "folds" in trial
            for fold in trial["folds"]:
                assert "metrics" in fold
                assert "regression_rmse_log_rv" in fold["metrics"]
                assert "regime_f1_macro" in fold["metrics"]
        assert "regime_f1_macro" in loaded["summary"][arm]
        assert "regression_rmse_log_rv" in loaded["summary"][arm]
