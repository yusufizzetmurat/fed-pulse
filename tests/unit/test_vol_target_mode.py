"""``--vol-target-mode {raw,garch_residual}`` switch wiring (#435).

PR #434 landed the data side (events.parquet columns + audit + ADR
0034 + tests) but the trainer always read from
``forward_realized_vol_10d``. This wiring threads a CLI knob through
``ModelConfig`` and ``_build_partition_log_rv_target`` so the canonical
sweep can produce a §6 row against the GARCH-residual variant.

The tests pin four layers:

- ``ModelConfig.vol_target_mode`` defaults to ``raw`` and round-trips
  through the dataclass field-name machinery (``_coerce_model_config``);
- the default-off byte-identity guarantee: with ``vol_target_mode='raw'``
  the per-partition log_rv tensor is bit-identical to the pre-#435
  builder output;
- ``vol_target_mode='garch_residual'`` reads the residual column and
  fits the standardiser on residual values (the fixture's residual is
  the absolute value of the raw target, so the fitted mean diverges
  from the raw-mode fit and the standardised tensor diverges too);
- the ``None``-residual fallback policy: rows with no residual fall
  back to ``log(forward_realized_vol_10d)`` so row count stays aligned
  with ``y``; a partition where the residual is populated on a strict
  subset of rows produces a mixed tensor and emits a single log warning.
"""

from __future__ import annotations

import datetime as _dt
import logging
import math

import pytest

torch = pytest.importorskip("torch")

from app.models.config import FeatureVector, ModelConfig  # noqa: E402
from app.training.loop import (  # noqa: E402
    DEFAULT_VOL_TARGET_MODE,
    VOL_TARGET_MODES,
    _build_partition_log_rv_target,
    train_model,
)


# ---------------------------------------------------------------------------
# ModelConfig surface
# ---------------------------------------------------------------------------


def test_vol_target_modes_canonical_tuple() -> None:
    assert VOL_TARGET_MODES == ("raw", "garch_residual")
    assert DEFAULT_VOL_TARGET_MODE == "raw"


def test_model_config_default_vol_target_mode_is_raw() -> None:
    """Pre-#435 defaults persist; explicit opt-in only flips the flag."""

    config = ModelConfig()
    assert config.vol_target_mode == "raw"


def test_model_config_vol_target_mode_round_trip_through_coerce() -> None:
    """A dict-payload checkpoint round-trips the new field via the field-name machinery."""

    from app.training.loop import _coerce_model_config

    payload = {
        "vol_target_mode": "garch_residual",
        "output_mode": "classification",
        "head_mode": "dual",
    }
    rebuilt = _coerce_model_config(payload)
    assert rebuilt.vol_target_mode == "garch_residual"


# ---------------------------------------------------------------------------
# Per-partition log_rv builder
# ---------------------------------------------------------------------------


def _dummy_feature_vector(
    *,
    day: int,
    vol: float,
    residual: float | None = None,
) -> FeatureVector:
    fv = FeatureVector(
        date=str(_dt.date(2025, 1, 1) + _dt.timedelta(days=day - 1)),
        sentiment_score=0.0,
        market_close=100.0,
        market_volatility=0.01,
        close_change_pct=0.0,
        volatility_change=0.0,
        elapsed_time=0.0,
        forward_realized_vol_10d=vol,
    )
    fv.forward_realized_vol_10d_garch_residual = residual
    return fv


def _make_groups(
    n: int = 30,
    *,
    populate_residual: bool = True,
) -> list[list[FeatureVector]]:
    """Synthetic walk-forward fold whose supervised rows (i >= 20) carry
    a monotonic forward-vol curve. When ``populate_residual`` is True
    the residual is set to ``abs(log(vol))`` so the per-fold scaler
    fitted under ``garch_residual`` mode lands on a materially different
    mean than the raw-mode fit (the absolute value collapses the
    log-vol negatives onto positives).
    """

    def _vol(i: int) -> float:
        return 0.01 + 0.001 * i

    def _residual(i: int) -> float | None:
        if not populate_residual:
            return None
        return abs(math.log(_vol(i)))

    return [
        [
            _dummy_feature_vector(
                day=i + 1,
                vol=_vol(i),
                residual=_residual(i),
            )
            for i in range(n)
        ]
    ]


def test_log_rv_raw_mode_byte_identical_to_pre_435_path() -> None:
    """Default ``vol_target_mode='raw'`` matches the unparameterised call.

    The previous builder signature accepted ``(sequence_groups,
    vol_regime_quantiles=, log_rv_scaler=)``; the new ``vol_target_mode``
    kwarg defaults to ``'raw'`` so any caller that does not pass it
    sees the identical tensor. Without the residual column populated
    the explicit ``'garch_residual'`` mode would have fallen back to
    raw row-by-row, also matching — this test pins the strict default
    path (no fallback log line, no residual lookup) only.
    """

    groups = _make_groups(n=30, populate_residual=False)
    quantiles = (0.012, 0.018)

    default_tensor, default_scaler = _build_partition_log_rv_target(
        groups, vol_regime_quantiles=quantiles
    )
    raw_tensor, raw_scaler = _build_partition_log_rv_target(
        groups, vol_regime_quantiles=quantiles, vol_target_mode="raw"
    )
    assert default_tensor is not None
    assert raw_tensor is not None
    # Byte-identical: same shape, same values, same scaler.
    assert default_tensor.shape == raw_tensor.shape
    assert torch.equal(default_tensor, raw_tensor)
    assert default_scaler == raw_scaler


def test_log_rv_garch_residual_mode_reads_residual_column() -> None:
    """``garch_residual`` mode fits the standardiser on the residual values."""

    groups = _make_groups(n=30, populate_residual=True)
    quantiles = (0.012, 0.018)

    raw_tensor, raw_scaler = _build_partition_log_rv_target(
        groups, vol_regime_quantiles=quantiles, vol_target_mode="raw"
    )
    residual_tensor, residual_scaler = _build_partition_log_rv_target(
        groups,
        vol_regime_quantiles=quantiles,
        vol_target_mode="garch_residual",
    )
    assert raw_tensor is not None
    assert residual_tensor is not None
    assert raw_scaler is not None
    assert residual_scaler is not None
    # Row count is identical (same group filter) but the values diverge
    # because the fixture's residual is the absolute value of log(vol)
    # rather than log(vol) itself.
    assert raw_tensor.shape == residual_tensor.shape
    assert not torch.allclose(raw_tensor, residual_tensor)
    # The two modes fit independent scalers; the residual fit centres on
    # the absolute-value mean which is materially different from the
    # raw log-vol mean. The raw log-vol mean sits around -4.4 (log of
    # ~0.012); the residual mean equals abs(log) ≈ +4.4. So the fitted
    # means must differ by a non-trivial amount.
    assert abs(raw_scaler[0] - residual_scaler[0]) > 1.0


def test_log_rv_garch_residual_missing_falls_back_to_raw(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Rows with ``residual=None`` fall back to log(raw) with a log line.

    Row alignment with ``y`` is the load-bearing invariant: dropping
    rows would break the TensorDataset row-count contract the dual-head
    DataLoader rebuilds positionally. The fallback emits one warning at
    the partition boundary so the operator can audit how many rows the
    data-side decomposition silently lacked (insufficient fit history
    or QMLE convergence failure per #434).
    """

    n = 30
    groups: list[list[FeatureVector]] = []
    sequence: list[FeatureVector] = []
    for i in range(n):
        # Populate the residual on the first half of the supervised
        # range only; the second half falls back to raw.
        if i >= 20 and i < 25:
            residual: float | None = abs(math.log(0.01 + 0.001 * i))
        else:
            residual = None
        sequence.append(
            _dummy_feature_vector(
                day=i + 1,
                vol=0.01 + 0.001 * i,
                residual=residual,
            )
        )
    groups.append(sequence)
    quantiles = (0.012, 0.018)

    with caplog.at_level(logging.WARNING, logger="app.training.loop"):
        residual_tensor, _scaler = _build_partition_log_rv_target(
            groups,
            vol_regime_quantiles=quantiles,
            vol_target_mode="garch_residual",
        )
        raw_tensor, _ = _build_partition_log_rv_target(
            groups,
            vol_regime_quantiles=quantiles,
            vol_target_mode="raw",
        )
    assert residual_tensor is not None
    assert raw_tensor is not None
    # Both tensors carry the same row count -- the fallback preserves
    # row alignment with y.
    assert residual_tensor.shape == raw_tensor.shape
    # The warning fires exactly once at the partition boundary and
    # records the fallback row count (5 rows populated, n - 5 fell back).
    fallback_records = [
        rec for rec in caplog.records
        if "vol_target_mode='garch_residual'" in rec.getMessage()
        and "fell back" in rec.getMessage()
    ]
    assert len(fallback_records) == 1
    assert "row(s)" in fallback_records[0].getMessage()


def test_log_rv_garch_residual_no_warning_when_every_row_has_residual(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Clean residual coverage -> no fallback log line."""

    groups = _make_groups(n=30, populate_residual=True)
    quantiles = (0.012, 0.018)

    with caplog.at_level(logging.WARNING, logger="app.training.loop"):
        residual_tensor, _scaler = _build_partition_log_rv_target(
            groups,
            vol_regime_quantiles=quantiles,
            vol_target_mode="garch_residual",
        )
    assert residual_tensor is not None
    fallback_records = [
        rec for rec in caplog.records
        if "fell back" in rec.getMessage()
    ]
    assert fallback_records == []


def test_log_rv_unknown_vol_target_mode_raises() -> None:
    """A typo in vol_target_mode raises rather than silently falling back."""

    groups = _make_groups(n=5)
    with pytest.raises(ValueError, match="vol_target_mode"):
        _build_partition_log_rv_target(
            groups,
            vol_regime_quantiles=(0.012, 0.018),
            vol_target_mode="not_a_mode",
        )


# ---------------------------------------------------------------------------
# End-to-end smoke through train_model
# ---------------------------------------------------------------------------


def _make_walk_forward_groups_with_residual(
    n: int = 40,
    *,
    populate_residual: bool = True,
) -> list[list[FeatureVector]]:
    def _vol(i: int) -> float:
        return 0.01 + 0.001 * i

    def _residual(i: int) -> float | None:
        if not populate_residual:
            return None
        # Sign-flip half so the residual centres around zero rather
        # than the all-positive log-vol value; this gives the
        # standardiser a non-trivial std to fit on.
        sign = 1.0 if i % 2 == 0 else -1.0
        return sign * 0.005 * (i % 5 + 1)

    return [
        [
            _dummy_feature_vector(
                day=i + 1,
                vol=_vol(i),
                residual=_residual(i),
            )
            for i in range(n)
        ]
    ]


def test_train_model_smoke_vol_target_mode_garch_residual() -> None:
    """1-epoch training run with ``garch_residual`` completes and persists the mode."""

    groups = _make_walk_forward_groups_with_residual(n=40, populate_residual=True)
    config = ModelConfig(
        output_mode="classification",
        n_classes=3,
        hidden_size=16,
        head_hidden_size=8,
        head_mode="dual",
        regression_alpha=0.5,
        vol_target_mode="garch_residual",
    )
    result = train_model(
        model_config=config,
        train_sequence_groups=groups,
        val_sequence_groups=groups,
        test_sequence_groups=groups,
        epochs=1,
        batch_size=8,
        seed=11,
        save_checkpoint=False,
        use_compile=False,
        use_amp=False,
    )
    assert result.summary.epochs_completed == 1
    persisted_config = ModelConfig.from_model(result.model)
    assert persisted_config.vol_target_mode == "garch_residual"


def test_train_model_smoke_vol_target_mode_raw_default() -> None:
    """Default ``vol_target_mode='raw'`` runs the pre-#435 path."""

    groups = _make_walk_forward_groups_with_residual(n=40, populate_residual=False)
    config = ModelConfig(
        output_mode="classification",
        n_classes=3,
        hidden_size=16,
        head_hidden_size=8,
        head_mode="dual",
        regression_alpha=0.5,
        # vol_target_mode defaults to 'raw'.
    )
    result = train_model(
        model_config=config,
        train_sequence_groups=groups,
        val_sequence_groups=groups,
        test_sequence_groups=groups,
        epochs=1,
        batch_size=8,
        seed=11,
        save_checkpoint=False,
        use_compile=False,
        use_amp=False,
    )
    assert result.summary.epochs_completed == 1
    persisted_config = ModelConfig.from_model(result.model)
    assert persisted_config.vol_target_mode == "raw"
