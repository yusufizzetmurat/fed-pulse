"""Per-axis class weights + lambdas survive the checkpoint round-trip (#273).

When ``multi_task_loss=True``, the training-loop builds per-axis class
weights for stance / certainty / topic on the train slice and constructs
a ``MultiTaskLoss`` with those weights plus the four lambda coefficients
from the ModelConfig. The fitted weights + lambdas must land on the
serialised checkpoint payload so a resume reads back the same loss
config -- otherwise the post-resume gradient signal silently drifts.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

import torch

from app.models.config import FeatureVector, ModelConfig
from app.training.loop import train_model


def _dummy_vec(
    *,
    day: int,
    vol: float,
    stance: int,
    factor: float,
    certainty: int,
    topic: int,
) -> FeatureVector:
    return FeatureVector(
        date=_dt.date(2025, 1, 1) + _dt.timedelta(days=day - 1),
        sentiment_score=0.0,
        market_close=100.0,
        market_volatility=0.01,
        close_change_pct=0.0,
        volatility_change=0.0,
        elapsed_time=0.0,
        forward_realized_vol_10d=vol,
        target_stance_idx=stance,
        target_stance_present=True,
        target_factor=factor,
        target_factor_present=True,
        target_certainty_idx=certainty,
        target_certainty_present=True,
        target_topic_idx=topic,
        target_topic_present=True,
    )


def _make_groups(n: int = 40) -> list[list[FeatureVector]]:
    return [
        [
            _dummy_vec(
                day=i + 1,
                vol=0.01 + 0.001 * i,
                stance=i % 3,
                factor=((i % 5) - 2) / 5.0,
                certainty=i % 3,
                topic=i % 4,
            )
            for i in range(n)
        ]
    ]


def test_checkpoint_carries_per_axis_class_weights_and_lambdas(tmp_path: Path) -> None:
    """One-epoch smoke: --multi-task-loss=on, then load the .pt and
    assert the per-axis weights + lambdas round-trip into the payload.
    """

    config = ModelConfig(
        output_mode="classification",
        multi_task_loss=True,
        multi_task_lambda_stance=1.0,
        multi_task_lambda_factor=0.25,
        multi_task_lambda_certainty=0.35,
        multi_task_lambda_topic=0.40,
        n_classes=3,
    )
    groups = _make_groups()
    ckpt = tmp_path / "mt.pt"

    result = train_model(
        model_config=config,
        train_sequence_groups=groups,
        val_sequence_groups=groups,
        test_sequence_groups=groups,
        epochs=1,
        batch_size=8,
        seed=11,
        save_checkpoint=True,
        checkpoint_path=ckpt,
        use_compile=False,
        use_amp=False,
    )

    assert result.summary.epochs_completed == 1
    # The summary itself carries the per-axis weights so a sweep
    # aggregator can ingest them without re-loading the .pt file.
    mt = result.summary.multi_task_class_weights
    assert mt is not None
    assert "certainty" in mt and "topic" in mt and "lambdas" in mt
    assert isinstance(mt["certainty"], list) and len(mt["certainty"]) >= 2
    assert isinstance(mt["topic"], list) and len(mt["topic"]) >= 2
    lambdas = mt["lambdas"]
    assert lambdas["stance"] == 1.0
    assert lambdas["factor"] == 0.25
    assert lambdas["certainty"] == 0.35
    assert lambdas["topic"] == 0.40

    # The on-disk checkpoint must also carry the same payload so a
    # resume reads back the exact loss config the run trained under.
    assert ckpt.exists(), "checkpoint .pt not written"
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    ts = payload.get("training_summary")
    assert isinstance(ts, dict)
    persisted = ts.get("multi_task_class_weights")
    assert persisted is not None, (
        "multi_task_class_weights missing from serialised training_summary; "
        "the resume path would silently rebuild a uniform-weighted loss."
    )
    assert persisted["lambdas"] == lambdas
    assert persisted["certainty"] == mt["certainty"]
    assert persisted["topic"] == mt["topic"]


def test_coerce_payload_config_threads_multi_task_fields() -> None:
    """`_coerce_payload_config` rebuilds the multi-task knob + lambdas (#423).

    Pre-#423 the helper rebuilt a ``multi_task_loss=False`` config from a
    checkpoint dict regardless of what the run trained under, so
    ``eval_checkpoint_directional`` / ``calibrate_regime_classifier``
    silently saw a single-task config on every multi-task checkpoint.
    Mirror the #292 rates-fields landing and round-trip the four knobs
    through the coercion path.
    """

    from app.training.checkpoint import _coerce_payload_config

    config = ModelConfig(
        output_mode="classification",
        multi_task_loss=True,
        multi_task_lambda_stance=1.0,
        multi_task_lambda_factor=0.25,
        multi_task_lambda_certainty=0.35,
        multi_task_lambda_topic=0.40,
        n_classes=3,
    )

    rebuilt = _coerce_payload_config({"model_config": config.to_dict()})

    assert rebuilt.multi_task_loss is True
    assert rebuilt.multi_task_lambda_stance == 1.0
    assert rebuilt.multi_task_lambda_factor == 0.25
    assert rebuilt.multi_task_lambda_certainty == 0.35
    assert rebuilt.multi_task_lambda_topic == 0.40


def test_coerce_payload_config_defaults_when_multi_task_absent() -> None:
    """Pre-#273 checkpoints leave the keys absent; the rebuilt config
    must collapse to the single-task CE path so the legacy contract
    stays byte-identical.
    """

    from app.training.checkpoint import _coerce_payload_config

    rebuilt = _coerce_payload_config({"model_config": {"input_size": 6}})

    assert rebuilt.multi_task_loss is False
    assert rebuilt.multi_task_lambda_stance == 1.0
    assert rebuilt.multi_task_lambda_factor == 0.3
    assert rebuilt.multi_task_lambda_certainty == 0.3
    assert rebuilt.multi_task_lambda_topic == 0.3


def test_coerce_payload_config_threads_vol_target_mode() -> None:
    """`_coerce_payload_config` rebuilds the vol-target-mode (#435).

    A checkpoint trained under ``--vol-target-mode garch_residual`` must
    rebuild a config that names the residual column on the eval /
    calibration paths. Pre-#435 the helper silently dropped the key.
    """

    from app.training.checkpoint import _coerce_payload_config

    config = ModelConfig(
        output_mode="regression",
        vol_target_mode="garch_residual",
        n_classes=3,
    )

    rebuilt = _coerce_payload_config({"model_config": config.to_dict()})

    assert rebuilt.vol_target_mode == "garch_residual"


def test_coerce_payload_config_defaults_vol_target_mode_when_absent() -> None:
    """Pre-#435 checkpoints leave the key absent; the rebuilt config
    must collapse to ``raw`` so the legacy contract stays byte-identical.
    """

    from app.training.checkpoint import _coerce_payload_config

    rebuilt = _coerce_payload_config({"model_config": {"input_size": 6}})

    assert rebuilt.vol_target_mode == "raw"


def test_checkpoint_omits_multi_task_weights_when_flag_off(tmp_path: Path) -> None:
    """Default (multi_task_loss=False): no per-axis payload, no contract
    drift for every pre-#273 checkpoint.
    """

    config = ModelConfig(output_mode="classification", n_classes=3)
    groups = _make_groups()
    ckpt = tmp_path / "single.pt"

    result = train_model(
        model_config=config,
        train_sequence_groups=groups,
        val_sequence_groups=groups,
        test_sequence_groups=groups,
        epochs=1,
        batch_size=8,
        seed=11,
        save_checkpoint=True,
        checkpoint_path=ckpt,
        use_compile=False,
        use_amp=False,
    )

    assert result.summary.multi_task_class_weights is None
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    ts = payload.get("training_summary")
    assert isinstance(ts, dict)
    # Either the key is absent or it serialised as None -- both leave
    # the resume path on the legacy single-task CE branch.
    assert ts.get("multi_task_class_weights") is None
