"""Sequence-length plumbing on ``load_walk_forward_split`` (#476).

Pins three contracts:

- the new ``sequence_length`` kwarg defaults to the module-level
  ``SEQUENCE_LENGTH`` so existing callers stay byte-identical;
- passing ``sequence_length=60`` against a TP whose ``prior_bars_json``
  carries 60 bars yields 60-bar prior windows on every supervised
  sequence;
- the same 60-bar TP run with the default 20 kwarg drops every row
  that lacks 60 bars only when the override raises the gate above the
  TP width — at the default 20-bar gate the loader still admits the
  rows (it only checks ``len(bars) >= sequence_length``).
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
from pathlib import Path

import pytest


pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")
pytest.importorskip("torch")

from app.models.config import SEQUENCE_LENGTH  # noqa: E402
from app.training import loaders  # noqa: E402


_TRAINING_PACKAGE_ID_60 = "tp_seq_len_60_v1"
_TRAINING_PACKAGE_ID_20 = "tp_seq_len_20_v1"


def _synth_prior_bars(*, event_date: _dt.date, base_close: float, n_bars: int) -> str:
    payload = []
    for offset in range(n_bars, 0, -1):
        bar_date = _dt.date.fromordinal(event_date.toordinal() - offset)
        payload.append(
            {
                "date": bar_date.isoformat(),
                "close": round(base_close + (n_bars - offset) * 1.5, 10),
                "volume": 1_000_000.0,
                "vol_5d": 0.012,
                "vol_20d": 0.018,
                "vol_60d": 0.022,
                "cum_return_20d": 0.0,
                "vix_close": 15.0,
                "dxy_close": 103.0,
                "tnx_close": 4.20,
                "gold_close": 2050.0,
                "vix3m_close": 17.0,
                "irx_close": 5.10,
                "vix_term_slope": 0.0,
                "yield_curve_slope_10y_3m": -0.90,
            }
        )
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


def _make_event_row(
    *,
    event_date: str,
    text: str,
    axis_stance: str | None,
    base_close: float,
    n_bars: int,
) -> dict:
    ed = _dt.date.fromisoformat(event_date)
    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return {
        "event_date": event_date,
        "event_kind": "statement",
        "document_id": text_hash[:16],
        "text_hash": text_hash,
        "source": "scraped_fed",
        "source_record_id": f"src:{text_hash[:8]}",
        "as_of_ts": f"{event_date}T19:00:00Z",
        "text": text,
        "token_count": len(text.split()),
        "axis_stance": axis_stance,
        "axis_time": None,
        "axis_certainty": None,
        "axis_factor": None,
        "axis_time_label": None,
        "axis_certain_label": None,
        "credibility_drift_score": 0.0,
        "credibility_realized_vs_stated_gap": 0.0,
        "credibility_market_implied_gap": 0.0,
        "credibility_months_since_reversal": 0,
        "prior_window_sha256": "0" * 64,
        "prior_bars_json": _synth_prior_bars(
            event_date=ed, base_close=base_close, n_bars=n_bars
        ),
        "asset_symbol": "^GSPC",
        "horizon": 1,
        "realized_return": 0.001,
        "abnormal_return": 0.001,
        "alpha": 0.0,
        "beta": 1.0,
        "direction_t1d": 1,
        "volatility_shift": 0.0,
        "concurrent_macro_release": False,
        "intra_meeting_stance_shift": 0.0,
        "intra_meeting_certainty_shift": 0.0,
        "intra_meeting_factor_shift": 0.0,
        "realized_date": (ed + _dt.timedelta(days=1)).isoformat(),
        "forward_realized_vol_10d": 0.015,
        "yield_2y_change_5d": 1.0,
        "yield_5y_change_5d": 0.5,
        "terminal_rate_change_5d": 0.0,
    }


def _materialise_package(
    *,
    root: Path,
    package_id: str,
    n_bars: int,
) -> Path:
    package_dir = root / "processed" / package_id
    package_dir.mkdir(parents=True)
    rows = [
        _make_event_row(
            event_date="2023-09-20",
            text=f"Event A ({n_bars} bars).",
            axis_stance="hawkish",
            base_close=4400.0,
            n_bars=n_bars,
        ),
        _make_event_row(
            event_date="2023-12-13",
            text=f"Event B ({n_bars} bars).",
            axis_stance="neutral",
            base_close=4500.0,
            n_bars=n_bars,
        ),
        _make_event_row(
            event_date="2024-03-20",
            text=f"Event C ({n_bars} bars).",
            axis_stance="neutral",
            base_close=4600.0,
            n_bars=n_bars,
        ),
    ]
    pd.DataFrame(rows).to_parquet(package_dir / "events.parquet", index=False)

    split_rows = [
        {"text_hash": row["text_hash"], "split_tag": "train" if i < 2 else "test"}
        for i, row in enumerate(rows)
    ]
    pd.DataFrame(split_rows).to_parquet(
        package_dir / "splits_train_val_test.parquet", index=False
    )
    return package_dir


@pytest.fixture
def package_60bar(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setattr(loaders, "DATA_DIR", tmp_path)
    return _materialise_package(
        root=tmp_path, package_id=_TRAINING_PACKAGE_ID_60, n_bars=60
    )


@pytest.fixture
def package_20bar(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setattr(loaders, "DATA_DIR", tmp_path)
    return _materialise_package(
        root=tmp_path, package_id=_TRAINING_PACKAGE_ID_20, n_bars=20
    )


def test_default_kwarg_matches_module_constant(package_20bar: Path) -> None:
    """``load_walk_forward_split(...)`` w/o ``sequence_length`` -> 20-bar windows.

    Default-path byte-identity: every supervised sequence carries
    ``SEQUENCE_LENGTH`` lookback bars + 1 appended target frame.
    """

    split = loaders.load_walk_forward_split(
        _TRAINING_PACKAGE_ID_20,
        rich_features=False,
        text_encoder=None,
    )
    assert split.train, "20-bar fixture must produce at least one train sequence"
    for partition in (split.train, split.val, split.test):
        for sequence in partition:
            assert len(sequence) == SEQUENCE_LENGTH + 1


def test_explicit_default_matches_implicit_default(package_20bar: Path) -> None:
    """Passing ``sequence_length=SEQUENCE_LENGTH`` matches the kwarg-elided call.

    Same fixture, same per-sequence lengths, same partition counts.
    """

    elided = loaders.load_walk_forward_split(
        _TRAINING_PACKAGE_ID_20,
        rich_features=False,
        text_encoder=None,
    )
    explicit = loaders.load_walk_forward_split(
        _TRAINING_PACKAGE_ID_20,
        rich_features=False,
        text_encoder=None,
        sequence_length=SEQUENCE_LENGTH,
    )
    assert len(elided.train) == len(explicit.train)
    assert len(elided.val) == len(explicit.val)
    assert len(elided.test) == len(explicit.test)
    for partition_a, partition_b in zip(
        (elided.train, elided.val, elided.test),
        (explicit.train, explicit.val, explicit.test),
    ):
        for seq_a, seq_b in zip(partition_a, partition_b):
            assert len(seq_a) == len(seq_b) == SEQUENCE_LENGTH + 1


def test_sequence_length_60_yields_60_bar_windows(package_60bar: Path) -> None:
    """``sequence_length=60`` against a 60-bar TP -> 60-bar lookback windows."""

    split = loaders.load_walk_forward_split(
        _TRAINING_PACKAGE_ID_60,
        rich_features=False,
        text_encoder=None,
        sequence_length=60,
    )
    assert split.train, "60-bar fixture must produce at least one train sequence"
    for partition in (split.train, split.val, split.test):
        for sequence in partition:
            # 60 prior bars + 1 appended event-day target frame.
            assert len(sequence) == 60 + 1


def test_sequence_length_60_against_20bar_tp_drops_every_event(
    package_20bar: Path,
) -> None:
    """``sequence_length=60`` against a 20-bar TP -> empty test (every row dropped).

    The gate ``len(bars) >= sequence_length`` short-circuits each event
    before the target frame appends. With no event surviving the
    walk-forward split the loader raises rather than silently training
    on no held-out rows.
    """

    with pytest.raises(ValueError, match="empty test partition"):
        loaders.load_walk_forward_split(
            _TRAINING_PACKAGE_ID_20,
            rich_features=False,
            text_encoder=None,
            sequence_length=60,
        )


def test_build_training_tensors_emits_60bar_timestep_dim(package_60bar: Path) -> None:
    """``_build_training_tensors(..., sequence_length=60)`` -> ``x.shape[1] == 60``.

    Loader-side plumbing is necessary but not sufficient: the tensor
    builder slices its own window off the sequence groups, so the
    timestep dim of the x tensor must also pick up the override.
    """

    import torch

    split = loaders.load_walk_forward_split(
        _TRAINING_PACKAGE_ID_60,
        rich_features=False,
        text_encoder=None,
        sequence_length=60,
    )
    x, y, _scale = loaders._build_training_tensors(
        split.train, sequence_length=60
    )
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    # Window count = sum(max(0, len(group) - sequence_length)) over groups;
    # the 60-bar fixture emits one (sequence_length + 1)-row group per
    # split partition so the per-group window count is exactly 1, and the
    # batch dim must be `len(split.train) * 1`. The explicit product
    # locks the multi-window contract in case a future fixture variant
    # widens the per-group row count.
    expected_windows = sum(
        max(0, len(group) - 60) for group in split.train
    )
    assert x.shape[0] == expected_windows
    assert x.shape[1] == 60
    assert x.shape[0] == y.shape[0]


def test_build_training_tensors_default_matches_module_constant(
    package_20bar: Path,
) -> None:
    """Default kwarg drops to ``SEQUENCE_LENGTH`` -> ``x.shape[1] == 20``."""

    import torch

    split = loaders.load_walk_forward_split(
        _TRAINING_PACKAGE_ID_20,
        rich_features=False,
        text_encoder=None,
    )
    x, _y, _scale = loaders._build_training_tensors(split.train)
    assert isinstance(x, torch.Tensor)
    assert x.shape[1] == SEQUENCE_LENGTH
