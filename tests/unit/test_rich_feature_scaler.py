from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from app.models.config import (
    FEATURE_SIZE,
    RICH_EXTRA_FEATURE_SIZE,
    RICH_FEATURE_SIZE,
    RichFeatureScalerParams,
)
from app.training.loaders import (
    apply_rich_feature_scaler_tensor,
    fit_rich_feature_scaler_tensor,
)


# ---------------------------------------------------------------------------
# RichFeatureScalerParams dataclass
# ---------------------------------------------------------------------------


def _ones_params(**overrides):
    base = dict(
        medians=tuple(0.0 for _ in range(RICH_EXTRA_FEATURE_SIZE)),
        iqrs=tuple(1.0 for _ in range(RICH_EXTRA_FEATURE_SIZE)),
        epsilon=1e-6,
        fitted_at_utc="2026-05-17T00:00:00Z",
        n_train_observations=100,
    )
    base.update(overrides)
    return RichFeatureScalerParams(**base)


def test_dataclass_round_trips_through_dict() -> None:
    params = _ones_params(
        medians=tuple(float(i) for i in range(RICH_EXTRA_FEATURE_SIZE)),
        iqrs=tuple(0.5 + i for i in range(RICH_EXTRA_FEATURE_SIZE)),
    )
    restored = RichFeatureScalerParams.from_dict(params.to_dict())
    assert restored is not None
    assert restored.medians == params.medians
    assert restored.iqrs == params.iqrs
    assert restored.epsilon == params.epsilon
    assert restored.fitted_at_utc == params.fitted_at_utc
    assert restored.n_train_observations == params.n_train_observations


def test_dataclass_rejects_wrong_length_medians() -> None:
    with pytest.raises(ValueError, match="medians"):
        RichFeatureScalerParams(
            medians=(0.0, 0.0),  # too short
            iqrs=tuple(1.0 for _ in range(RICH_EXTRA_FEATURE_SIZE)),
        )


def test_dataclass_rejects_non_positive_iqr() -> None:
    bad_iqrs = list(1.0 for _ in range(RICH_EXTRA_FEATURE_SIZE))
    bad_iqrs[3] = 0.0
    with pytest.raises(ValueError, match=r"iqrs\[3\]"):
        RichFeatureScalerParams(
            medians=tuple(0.0 for _ in range(RICH_EXTRA_FEATURE_SIZE)),
            iqrs=tuple(bad_iqrs),
        )


@pytest.mark.parametrize("payload", [None, {}, [], "string", 12, {"medians": [0.0]}])
def test_from_dict_returns_none_on_malformed(payload) -> None:
    assert RichFeatureScalerParams.from_dict(payload) is None


# ---------------------------------------------------------------------------
# fit_rich_feature_scaler_tensor
# ---------------------------------------------------------------------------


def test_fit_returns_none_for_legacy_six_feature_tensor() -> None:
    x = torch.randn(8, 20, FEATURE_SIZE)
    assert fit_rich_feature_scaler_tensor(x) is None


def test_fit_returns_none_for_empty_tensor() -> None:
    x = torch.empty(0, 20, RICH_FEATURE_SIZE)
    assert fit_rich_feature_scaler_tensor(x) is None
    assert fit_rich_feature_scaler_tensor(None) is None  # type: ignore[arg-type]


def test_fit_recovers_known_median_and_iqr() -> None:
    """Construct a tensor where every rich column has known stats."""
    # 100 windows x 1 bar each → 100 observations per column
    n = 100
    market = torch.zeros(n, 1, FEATURE_SIZE)
    # Rich block: column 0 is values 0..99, column 1 is 100..199, etc.
    rich = torch.stack(
        [torch.arange(n).float() + col * n for col in range(RICH_EXTRA_FEATURE_SIZE)],
        dim=1,
    ).unsqueeze(1)  # shape (n, 1, RICH_EXTRA_FEATURE_SIZE)
    x = torch.cat([market, rich], dim=-1)

    params = fit_rich_feature_scaler_tensor(x)
    assert params is not None
    assert params.n_train_observations == n
    # For values 0..99, the median is 49.5 and IQR is (74.75 - 24.75) = 50.0
    # numpy uses linear interpolation by default; allow a tiny tolerance.
    for col in range(RICH_EXTRA_FEATURE_SIZE):
        expected_median = 49.5 + col * n
        expected_iqr = 49.5  # numpy linear-interp IQR for 0..99
        assert math.isclose(params.medians[col], expected_median, abs_tol=1e-6)
        assert math.isclose(params.iqrs[col], expected_iqr, abs_tol=1e-6)


def test_fit_floors_constant_column_iqr_to_one() -> None:
    """A column whose IQR is 0 must not produce divide-by-zero downstream."""
    n = 50
    x = torch.zeros(n, 20, RICH_FEATURE_SIZE)
    # Column 4 of the rich block has a real distribution; everything else
    # stays at the constant 0.0 so its IQR is 0 → must be floored to 1.0.
    x[..., FEATURE_SIZE + 4] = torch.arange(20).float().unsqueeze(0).expand(n, 20)
    params = fit_rich_feature_scaler_tensor(x)
    assert params is not None
    for col in range(RICH_EXTRA_FEATURE_SIZE):
        if col == 4:
            assert params.iqrs[col] > 1.0
        else:
            assert params.iqrs[col] == 1.0   # floored


def test_fit_records_n_train_observations_over_all_bars() -> None:
    """Stats are estimated over (n_windows * sequence_length) population."""
    n_windows = 7
    seq_len = 13
    x = torch.randn(n_windows, seq_len, RICH_FEATURE_SIZE)
    params = fit_rich_feature_scaler_tensor(x)
    assert params is not None
    assert params.n_train_observations == n_windows * seq_len


# ---------------------------------------------------------------------------
# apply_rich_feature_scaler_tensor
# ---------------------------------------------------------------------------


def test_apply_is_noop_when_scaler_is_none() -> None:
    x = torch.randn(4, 20, RICH_FEATURE_SIZE)
    out = apply_rich_feature_scaler_tensor(x, None)
    assert torch.equal(out, x)


def test_apply_is_noop_for_legacy_six_feature_tensor() -> None:
    """Even with a scaler, a 6-dim tensor must pass through unchanged."""
    params = _ones_params()
    x = torch.randn(4, 20, FEATURE_SIZE)
    out = apply_rich_feature_scaler_tensor(x, params)
    assert torch.equal(out, x)


def test_apply_market_block_passes_through() -> None:
    params = _ones_params(medians=tuple(10.0 for _ in range(RICH_EXTRA_FEATURE_SIZE)))
    x = torch.randn(5, 20, RICH_FEATURE_SIZE)
    out = apply_rich_feature_scaler_tensor(x, params)
    assert torch.equal(out[..., :FEATURE_SIZE], x[..., :FEATURE_SIZE])


def test_apply_rich_block_is_z_scored() -> None:
    """(x - median) / iqr applied column-wise to the rich block."""
    medians = tuple(float(i) for i in range(RICH_EXTRA_FEATURE_SIZE))
    iqrs = tuple(float(i + 1) for i in range(RICH_EXTRA_FEATURE_SIZE))
    params = _ones_params(medians=medians, iqrs=iqrs)
    x = torch.ones(2, 3, RICH_FEATURE_SIZE) * 5.0
    out = apply_rich_feature_scaler_tensor(x, params)
    for col in range(RICH_EXTRA_FEATURE_SIZE):
        expected = (5.0 - medians[col]) / iqrs[col]
        assert torch.allclose(out[..., FEATURE_SIZE + col], torch.tensor(expected))


def test_fit_then_apply_yields_unit_iqr_on_train_slice() -> None:
    """Round-trip invariant: applying the scaler to its own train slice
    should produce a slice whose IQR is ~1.0 column-wise."""
    n_windows = 80
    seq_len = 20
    # Use a fixed seed so the IQR estimate is stable.
    g = torch.Generator()
    g.manual_seed(11)
    x = torch.randn(n_windows, seq_len, RICH_FEATURE_SIZE, generator=g) * 5.0

    params = fit_rich_feature_scaler_tensor(x)
    assert params is not None
    out = apply_rich_feature_scaler_tensor(x, params)
    rich_block = out[..., FEATURE_SIZE:RICH_FEATURE_SIZE].reshape(
        -1, RICH_EXTRA_FEATURE_SIZE
    ).numpy()
    q1 = np.quantile(rich_block, 0.25, axis=0)
    q3 = np.quantile(rich_block, 0.75, axis=0)
    iqrs = q3 - q1
    assert np.allclose(iqrs, 1.0, atol=1e-5)


def test_apply_preserves_dtype_and_device() -> None:
    params = _ones_params()
    x = torch.randn(2, 3, RICH_FEATURE_SIZE, dtype=torch.float64)
    out = apply_rich_feature_scaler_tensor(x, params)
    assert out.dtype == torch.float64
    assert out.device == x.device
