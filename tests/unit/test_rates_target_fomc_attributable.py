"""FOMC-attributable rates target wiring (#305).

The new ``rates_target_mode='fomc_attributable'`` path projects the
observed 5d rates move onto the strict-prior policy-surprise direction
``sign(mp_surprise_level)`` and supervises the rates heads against the
projected scalar. These tests pin the surface at four layers:

- the projection helper (hand-computable fixture; pause-meeting edge
  case marks the target missing);
- the per-fold target builder (the new mode reads the
  ``_fomc_attributable`` FeatureVector field and standardises on the
  train slice only; val / test reuse the train scaler);
- ``ModelConfig.rates_target_mode`` round-trips through the dataclass
  field-name machinery in ``_coerce_model_config``;
- a 1-epoch smoke through ``train_model`` runs to completion with
  ``rates_target_mode='fomc_attributable'`` and persists the mode on
  the run summary's ``ModelConfig``.

See ADR 0027 for the projection definition.
"""

from __future__ import annotations

import datetime as _dt
import math

import pytest

torch = pytest.importorskip("torch")

from app.models.config import FeatureVector, ModelConfig  # noqa: E402
from app.training.loop import train_model  # noqa: E402
from app.training.rates_targets import (  # noqa: E402
    DEFAULT_RATES_TARGET_MODE,
    RATES_TARGET_MODES,
    SURPRISE_DIRECTION_EPSILON_BPS,
    _rates_field_for,
    build_partition_rates_targets,
    fomc_attributable_projection,
)


# ---------------------------------------------------------------------------
# Projection helper
# ---------------------------------------------------------------------------


def test_projection_hawkish_surprise_signs_positive() -> None:
    """Observed +10 bps move with +25 bps hawkish surprise -> +10 bps target.

    The 1-D projection keeps the magnitude and signs it relative to the
    surprise direction. A positive scalar means the observed move agreed
    with the surprise (hawkish surprise -> yields up was the agreement).
    """

    assert fomc_attributable_projection(10.0, 25.0) == pytest.approx(10.0)


def test_projection_dovish_surprise_signs_observation() -> None:
    """Observed +10 bps move with -25 bps dovish surprise -> -10 bps target.

    The model now predicts a negative attributable component, encoding
    "yields rose but the surprise was dovish -> the move opposed the
    surprise direction."
    """

    assert fomc_attributable_projection(10.0, -25.0) == pytest.approx(-10.0)


def test_projection_negative_observation_dovish_surprise() -> None:
    """Observed -8 bps move with -12 bps dovish surprise -> +8 bps target.

    Both factors flip sign; the projection magnitude survives unchanged
    and the sign agrees with the surprise direction (negative move,
    negative surprise -> positive attributable component).
    """

    assert fomc_attributable_projection(-8.0, -12.0) == pytest.approx(8.0)


def test_projection_pause_meeting_marks_missing() -> None:
    """No-change meeting (|surprise| < epsilon) returns None, not zero.

    Coercing the pause-meeting target to zero would inject a fake
    "no FOMC-attributable component" label across every pause and bias
    the regression toward the origin. The gate fires below the 1-bp
    epsilon.
    """

    assert SURPRISE_DIRECTION_EPSILON_BPS == 1.0
    # |surprise| = 0 -> direction ill-defined.
    assert fomc_attributable_projection(15.0, 0.0) is None
    # Below the epsilon -> still missing.
    assert fomc_attributable_projection(15.0, 0.5) is None
    assert fomc_attributable_projection(15.0, -0.5) is None
    # At the epsilon boundary the gate releases (>= 1.0 bp passes).
    assert fomc_attributable_projection(15.0, 1.0) == pytest.approx(15.0)


def test_projection_none_inputs_propagate_missing() -> None:
    """Either None input collapses to None so the loader masks the row."""

    assert fomc_attributable_projection(None, 25.0) is None
    assert fomc_attributable_projection(10.0, None) is None
    assert fomc_attributable_projection(None, None) is None


def test_projection_non_finite_inputs_collapse_to_missing() -> None:
    """NaN / inf surprises or moves emit None rather than propagating."""

    assert fomc_attributable_projection(float("nan"), 25.0) is None
    assert fomc_attributable_projection(10.0, float("nan")) is None
    assert fomc_attributable_projection(float("inf"), 25.0) is None
    assert fomc_attributable_projection(10.0, float("inf")) is None


def test_rates_field_for_dispatch_on_target_mode() -> None:
    """``_rates_field_for`` returns the right FeatureVector attribute name."""

    assert _rates_field_for("2y") == "target_yield_2y_change_5d"
    assert _rates_field_for("2y", target_mode="raw") == "target_yield_2y_change_5d"
    assert (
        _rates_field_for("2y", target_mode="fomc_attributable")
        == "target_yield_2y_change_5d_fomc_attributable"
    )
    assert (
        _rates_field_for("terminal", target_mode="fomc_attributable")
        == "target_terminal_rate_change_5d_fomc_attributable"
    )


def test_rates_target_modes_canonical_tuple() -> None:
    """The literal vocabulary is pinned at the module level."""

    assert RATES_TARGET_MODES == ("raw", "fomc_attributable")
    assert DEFAULT_RATES_TARGET_MODE == "raw"


def test_build_partition_rates_targets_rejects_unknown_target_mode() -> None:
    """A typo in target_mode raises rather than silently dropping the head."""

    with pytest.raises(ValueError, match="rates_target_mode"):
        build_partition_rates_targets(
            sequence_groups=[],
            head_names=("2y",),
            target_mode="kuttner",
        )


def test_rates_field_for_rejects_unknown_target_mode() -> None:
    with pytest.raises(ValueError, match="rates_target_mode"):
        _rates_field_for("2y", target_mode="kuttner")


# ---------------------------------------------------------------------------
# Per-fold target builder
# ---------------------------------------------------------------------------


def _dummy_feature_vector(
    *,
    day: int,
    vol: float,
    raw_2y: float = 0.0,
    raw_5y: float = 0.0,
    raw_terminal: float = 0.0,
    fomc_2y: float | None = None,
    fomc_5y: float | None = None,
    fomc_terminal: float | None = None,
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
    fv.target_yield_2y_change_5d = raw_2y
    fv.target_yield_5y_change_5d = raw_5y
    fv.target_terminal_rate_change_5d = raw_terminal
    fv.target_yield_2y_change_5d_fomc_attributable = fomc_2y
    fv.target_yield_5y_change_5d_fomc_attributable = fomc_5y
    fv.target_terminal_rate_change_5d_fomc_attributable = fomc_terminal
    return fv


def _make_walk_forward_groups(
    n: int = 30,
    *,
    populate_fomc: bool = True,
) -> list[list[FeatureVector]]:
    """Build a walk-forward group whose supervised rows (i >= 20) span
    both signs of the raw bps move, so the standardiser fit under
    ``raw`` and ``fomc_attributable`` modes lands on materially different
    means (abs() collapses the negatives onto positives).
    """

    def _raw_2y(i: int) -> float:
        # i in [20, 29] -> values [-9, -7, ..., 9]; mean 0, mean(|.|) = 5.
        return float(2 * (i - 24) - 1)

    def _raw_5y(i: int) -> float:
        return float(1.5 * (i - 24) - 2)

    def _raw_terminal(i: int) -> float:
        return float((i - 24) - 1)

    return [
        [
            _dummy_feature_vector(
                day=i + 1,
                vol=0.01 + 0.001 * i,
                raw_2y=_raw_2y(i),
                raw_5y=_raw_5y(i),
                raw_terminal=_raw_terminal(i),
                fomc_2y=abs(_raw_2y(i)) if populate_fomc else None,
                fomc_5y=abs(_raw_5y(i)) if populate_fomc else None,
                fomc_terminal=abs(_raw_terminal(i)) if populate_fomc else None,
            )
            for i in range(n)
        ]
    ]


def test_build_partition_rates_targets_fomc_attributable_reads_projected_field() -> None:
    """The new mode pulls supervised values off the ``_fomc_attributable`` field.

    Builds two side-by-side partitions from the same group fixture, one
    in raw mode and one in fomc-attributable mode. The fixture's
    ``_fomc_attributable`` values are the absolute value of the raw
    ones, so the per-fold scaler fitted under the new mode picks up a
    different mean than the raw-mode fit and the standardised tensor
    diverges.
    """

    groups = _make_walk_forward_groups(n=30, populate_fomc=True)
    raw_bps, _, _, _, raw_scalers, _ = build_partition_rates_targets(
        groups, head_names=("2y",), target_mode="raw"
    )
    fomc_bps, _, _, _, fomc_scalers, _ = build_partition_rates_targets(
        groups, head_names=("2y",), target_mode="fomc_attributable"
    )
    # Mean of the raw 2y series is ~9.0 (signed); mean of |raw| is ~10.5.
    # The two scalers must therefore differ.
    assert raw_scalers["2y"].mean != fomc_scalers["2y"].mean
    # The standardised tensors must agree on row count (the row filter
    # is the same -- forward_realized_vol_10d gate).
    assert raw_bps["2y"].shape == fomc_bps["2y"].shape


def test_build_partition_rates_targets_fomc_attributable_uses_train_scaler() -> None:
    """Val / test partitions reuse the train-fitted scaler under the new mode."""

    train_groups = _make_walk_forward_groups(n=30, populate_fomc=True)
    val_groups = _make_walk_forward_groups(n=20, populate_fomc=True)
    _, _, _, _, train_scalers, train_edges = build_partition_rates_targets(
        train_groups,
        head_names=("2y",),
        target_mode="fomc_attributable",
    )
    _, _, _, _, val_scalers, val_edges = build_partition_rates_targets(
        val_groups,
        head_names=("2y",),
        scalers=train_scalers,
        edges_by_head=train_edges,
        target_mode="fomc_attributable",
    )
    assert val_scalers["2y"] == train_scalers["2y"]
    assert val_edges["2y"].lower == train_edges["2y"].lower


def test_build_partition_rates_targets_fomc_attributable_masks_missing_rows() -> None:
    """Rows whose projected target is None are masked out, not coerced to zero.

    Builds a group whose first 5 emitted rows carry a valid projection
    and whose last 5 emitted rows have no surprise direction (None);
    the resulting bps_mask must be True only on the populated rows so
    the per-fold scaler ignores the missing entries.
    """

    groups = [
        [
            _dummy_feature_vector(
                day=i + 1,
                vol=0.01 + 0.001 * i,
                raw_2y=float(2 * i - 20),
                # Supervised rows are i in [20, 29]; populate the first
                # five (i in [20, 24]) and leave the rest as None.
                fomc_2y=float(abs(2 * i - 20)) if i < 25 else None,
            )
            for i in range(30)
        ]
    ]
    _, bps_mask, _, _, _, _ = build_partition_rates_targets(
        groups, head_names=("2y",), target_mode="fomc_attributable"
    )
    mask = bps_mask["2y"]
    populated = int(mask.sum().item())
    # Five rows populated, five masked False.
    assert populated == 5
    assert int(mask.numel()) == 10


# ---------------------------------------------------------------------------
# ModelConfig round-trip
# ---------------------------------------------------------------------------


def test_model_config_default_rates_target_mode_is_raw() -> None:
    """Pre-#305 defaults persist; explicit opt-in only flips the flag."""

    config = ModelConfig()
    assert config.rates_target_mode == "raw"


def test_model_config_rates_target_mode_round_trip_through_coerce() -> None:
    """A dict-payload checkpoint round-trips the new field via the field-name machinery."""

    from app.training.loop import _coerce_model_config

    payload = {
        "rates_heads": ["2y"],
        "rates_head_mode": "regression",
        "rates_alpha": 0.5,
        "rates_target_mode": "fomc_attributable",
        "output_mode": "classification",
        "head_mode": "classification",
    }
    rebuilt = _coerce_model_config(payload)
    assert rebuilt.rates_target_mode == "fomc_attributable"
    assert rebuilt.rates_heads == ("2y",)


# ---------------------------------------------------------------------------
# Smoke training run
# ---------------------------------------------------------------------------


def test_train_model_smoke_rates_target_mode_fomc_attributable() -> None:
    """1-epoch training run with the new mode completes and records the mode.

    The smoke uses an in-memory fixture (no parquet I/O) so it runs
    under a second on CPU. The `_fomc_attributable` fields are
    pre-populated on the FeatureVectors; the per-fold target builder
    must pick them up and the run must complete without crashing on
    the new code path.
    """

    groups = _make_walk_forward_groups(n=40, populate_fomc=True)
    config = ModelConfig(
        output_mode="classification",
        n_classes=3,
        hidden_size=16,
        head_hidden_size=8,
        rates_heads=("2y",),
        rates_head_mode="regression",
        rates_alpha=0.5,
        rates_target_mode="fomc_attributable",
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
    # The persisted ModelConfig must record the active target mode so
    # downstream consumers (the §16 comparison table, the checkpoint
    # auditor) can verify which target the head trained against.
    persisted_config = ModelConfig.from_model(result.model)
    assert persisted_config.rates_target_mode == "fomc_attributable"
    # Per-head metrics still emit -- the surface is unchanged, only the
    # supervised target shifted.
    metrics = result.summary.metrics
    assert metrics is not None
    assert metrics.rates_metrics is not None
    assert "2y" in metrics.rates_metrics
    payload = metrics.rates_metrics["2y"]
    # The mae_bps panel must have a finite point estimate; if every
    # row got masked the panel would be None and the per-head metric
    # block on the comparison sweep would be empty.
    assert payload["n_rows"] > 0
    mae = payload["mae_bps"]
    assert mae is not None
    assert math.isfinite(mae["point"])


def test_train_model_smoke_rates_target_mode_raw_default() -> None:
    """Default ``rates_target_mode='raw'`` runs the pre-#305 path.

    Verifies the default opt-out semantics: an unset / 'raw' flag
    reads the raw observed-move target, and the persisted config
    records ``raw`` so an auditor can tell raw runs from opt-in runs.
    """

    groups = _make_walk_forward_groups(n=40, populate_fomc=False)
    config = ModelConfig(
        output_mode="classification",
        n_classes=3,
        hidden_size=16,
        head_hidden_size=8,
        rates_heads=("2y",),
        rates_head_mode="regression",
        rates_alpha=0.5,
        # rates_target_mode defaults to 'raw'.
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
    assert persisted_config.rates_target_mode == "raw"
