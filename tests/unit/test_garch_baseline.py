"""Unit tests for the GARCH(1,1) reference baseline."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")
pytest.importorskip("arch")


# scripts/ lives outside the python package, so make sure the module is importable.
_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import garch_baseline  # noqa: E402


def test_bin_to_regime_respects_cutoffs() -> None:
    """Three-class quantile binning maps values strictly to (low, med, high)."""

    assert garch_baseline._bin_to_regime(0.001, q33=0.005, q67=0.010) == 0
    assert garch_baseline._bin_to_regime(0.005, q33=0.005, q67=0.010) == 0  # boundary -> low
    assert garch_baseline._bin_to_regime(0.007, q33=0.005, q67=0.010) == 1
    assert garch_baseline._bin_to_regime(0.010, q33=0.005, q67=0.010) == 1  # boundary -> med
    assert garch_baseline._bin_to_regime(0.020, q33=0.005, q67=0.010) == 2


def test_quantile_cutoffs_round_trip_uniform() -> None:
    """Quantile cutoffs on a uniform distribution land at the 33rd/67th
    percentiles within numerical tolerance."""

    values = np.linspace(0.0, 1.0, num=300)
    q33, q67 = garch_baseline._quantile_cutoffs(values)
    assert q33 == pytest.approx(0.333, abs=0.005)
    assert q67 == pytest.approx(0.667, abs=0.005)


def test_bootstrap_macro_f1_brackets_point_estimate() -> None:
    """Bootstrap CI must bracket the empirical macro-F1 on a balanced
    synthetic pool. With 600 rows and ~85% accuracy the CI is tight
    enough to assert containment without flakiness."""

    rng = np.random.default_rng(11)
    n = 600
    targets = [int(x) for x in rng.integers(low=0, high=3, size=n)]
    # 85% match the target, 15% are off by one class.
    preds: list[int] = []
    for t in targets:
        if rng.uniform() < 0.85:
            preds.append(int(t))
        else:
            preds.append(int((t + 1) % 3))
    ci = garch_baseline._bootstrap_macro_f1(
        preds, targets, block_size=20, n_resamples=500, seed=11
    )
    assert ci.lo <= ci.point <= ci.hi
    assert 0.7 <= ci.point <= 0.95


@pytest.fixture
def garch_synthetic_package(tmp_path: Path, monkeypatch) -> tuple[Path, Path]:
    """Materialise a tiny training package + SPX returns parquet so the
    GARCH baseline can run end-to-end without touching real artefacts."""

    package_id = "tp_unit_garch_baseline"
    package_dir = tmp_path / "processed" / package_id
    package_dir.mkdir(parents=True)

    # Synthetic AR(1)-GARCH-ish daily returns: 2000 bars starting 2010-01-04.
    # Mix a low-vol regime in the first half and a high-vol regime in the
    # second half so the quantile-binned regime label has signal for GARCH
    # to lock onto.
    rng = np.random.default_rng(11)
    n_days = 2000
    half = n_days // 2
    returns_low = rng.normal(loc=0.0, scale=0.005, size=half)
    returns_high = rng.normal(loc=0.0, scale=0.020, size=n_days - half)
    returns = np.concatenate([returns_low, returns_high])
    dates = pd.bdate_range("2010-01-04", periods=n_days).strftime("%Y-%m-%d").tolist()
    closes = (np.exp(np.cumsum(returns)) * 100.0)
    spx = pd.DataFrame({"date": dates, "close": closes})
    spx_path = tmp_path / "spx.parquet"
    spx.to_parquet(spx_path, index=False)

    # Construct events at every 30 trading days so a 10-day forward window
    # exists for each event without colliding into the next.
    event_indices = list(range(60, n_days - 30, 30))
    event_rows = []
    for idx in event_indices:
        forward_window = returns[idx + 1 : idx + 11]
        realized_vol = float(np.std(forward_window))
        event_rows.append(
            {
                "event_date": dates[idx],
                "text_hash": f"hash_{idx:04d}",
                "forward_realized_vol_10d": realized_vol,
            }
        )
    pd.DataFrame(event_rows).to_parquet(package_dir / "events.parquet", index=False)

    # Single fold so the fold loop is exercised once.
    train_cut = len(event_rows) * 3 // 5
    val_cut = train_cut + len(event_rows) // 5
    manifest = {
        "folds": [
            {
                "fold_id": "wf_fold_1",
                "train_start": event_rows[0]["event_date"],
                "train_end": event_rows[train_cut - 1]["event_date"],
                "val_start": event_rows[train_cut]["event_date"],
                "val_end": event_rows[val_cut - 1]["event_date"],
                "test_start": event_rows[val_cut]["event_date"],
                "test_end": event_rows[-1]["event_date"],
            }
        ]
    }
    (package_dir / "fold_manifest_expanding_walk_forward.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    output_root = tmp_path / "garch_baseline"
    monkeypatch.setattr(garch_baseline, "DATA_DIR", tmp_path)
    return package_dir, output_root, spx_path, package_id


def test_garch_baseline_end_to_end(garch_synthetic_package, capsys, monkeypatch) -> None:
    """End-to-end smoke: the baseline should fit, forecast, bin, and emit
    a pooled macro-F1 JSON with a bracketed bootstrap CI on a controlled
    AR(1)-GARCH-style fixture."""

    package_dir, output_root, spx_path, package_id = garch_synthetic_package

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "garch_baseline",
            "--training-package-id",
            package_id,
            "--spx-path",
            str(spx_path),
            "--output-root",
            str(output_root),
            "--min-event-date",
            "2010-01-01",
            "--n-resamples",
            "200",
        ],
    )
    rc = garch_baseline.main()
    assert rc == 0

    out_path = output_root / package_id / "garch_pooled_test_macro_f1.json"
    assert out_path.exists()
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["training_package_id"] == package_id
    assert payload["n_pooled"] > 0
    assert math.isfinite(payload["macro_f1"])
    assert payload["macro_f1_ci"]["lo"] <= payload["macro_f1_ci"]["point"]
    assert payload["macro_f1_ci"]["point"] <= payload["macro_f1_ci"]["hi"]
    assert len(payload["per_fold"]) == 1
