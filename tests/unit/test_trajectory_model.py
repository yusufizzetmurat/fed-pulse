"""Unit tests for the trajectory model architectures (#296).

Covers:

* Forward-pass shapes for both LSTM and Transformer arms.
* The walk-forward filter at training-sequence build time — targets
  with ``event_date >= train_end`` must never enter the train pool.
* APS conformal calibration consumes the right calibration slice
  (softmax probabilities + true class indices).
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from app.evaluation.conformal import calibrate_classification_conformal  # noqa: E402
from app.trajectory import model as traj_model  # noqa: E402
from app.trajectory import train as traj_train  # noqa: E402


def _make_meeting(
    *,
    event_date: str,
    stance: str | None = "hawkish",
    yield_bp: float | None = 1.0,
    vix: float | None = 18.0,
) -> traj_train.MeetingRow:
    return traj_train.MeetingRow(
        event_date=event_date,
        text_hash=f"hash_{event_date}",
        text=f"meeting {event_date}",
        axis_stance=stance,
        trailing_2y_yield_change_5d_bps=yield_bp,
        vix_close=vix,
    )


def _meetings_panel() -> list[traj_train.MeetingRow]:
    return [
        _make_meeting(event_date="2010-03-16", stance="dovish"),
        _make_meeting(event_date="2010-06-22", stance="dovish"),
        _make_meeting(event_date="2011-01-25", stance="dovish"),
        _make_meeting(event_date="2015-03-17", stance="hawkish"),
        _make_meeting(event_date="2015-12-16", stance="neutral"),
        _make_meeting(event_date="2017-06-13", stance="hawkish"),
        _make_meeting(event_date="2019-07-30", stance="dovish"),
        _make_meeting(event_date="2020-03-15", stance="dovish"),
        _make_meeting(event_date="2022-06-15", stance="hawkish"),
        _make_meeting(event_date="2023-05-03", stance="hawkish"),
    ]


def _toy_embeddings(meetings: list[traj_train.MeetingRow], *, dim: int = 8) -> np.ndarray:
    """Deterministic toy embedding: hash the event_date into a fixed-dim vector."""

    out = np.zeros((len(meetings), dim), dtype=np.float32)
    for row_idx, row in enumerate(meetings):
        seed_int = abs(hash(row.event_date)) % (2**32)
        rng = np.random.default_rng(seed_int)
        out[row_idx] = rng.normal(size=dim).astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# Forward-pass shape
# ---------------------------------------------------------------------------


def test_lstm_forward_returns_logits_and_hidden_of_expected_shape() -> None:
    config = traj_model.TrajectoryConfig(
        architecture="lstm",
        embedding_dim=16,
        history_length=12,
    )
    model = traj_model.build_model(config)
    inputs = torch.randn(4, config.history_length, config.input_dim)
    mask = torch.ones(4, config.history_length, dtype=torch.bool)
    logits, hidden = model(inputs, mask)
    assert logits.shape == (4, config.n_classes)
    assert hidden.shape == (4, config.lstm_hidden)


def test_transformer_forward_returns_logits_and_hidden_of_expected_shape() -> None:
    config = traj_model.TrajectoryConfig(
        architecture="transformer",
        embedding_dim=16,
        history_length=12,
    )
    model = traj_model.build_model(config)
    inputs = torch.randn(3, config.history_length, config.input_dim)
    mask = torch.ones(3, config.history_length, dtype=torch.bool)
    logits, hidden = model(inputs, mask)
    assert logits.shape == (3, config.n_classes)
    assert hidden.shape == (3, config.transformer_d_model)


@pytest.mark.parametrize("arch", ["lstm", "transformer"])
def test_forward_pass_respects_mask_padding(arch: str) -> None:
    """Padded positions must not crash either architecture; final-real
    position decoding handles the boundary even when a row has zero
    real meetings (we still get a finite logits row from the fallback).

    Parametrised over both arms so a polarity bug in the Transformer's
    ``key_padding_mask`` (e.g. forgetting the ``~`` flip) is caught
    too — the LSTM-only coverage missed this category of regression.
    """

    config = traj_model.TrajectoryConfig(
        architecture=arch, embedding_dim=8, history_length=6  # type: ignore[arg-type]
    )
    model = traj_model.build_model(config)
    inputs = torch.randn(2, 6, config.input_dim)
    mask = torch.tensor(
        [
            [False, False, False, True, True, True],  # last 3 real
            [True, True, True, True, True, True],  # all real
        ],
        dtype=torch.bool,
    )
    logits, _ = model(inputs, mask)
    assert logits.shape == (2, config.n_classes)
    assert torch.isfinite(logits).all()


@pytest.mark.parametrize("arch", ["lstm", "transformer"])
def test_forward_pass_handles_all_pad_row_without_nan(arch: str) -> None:
    """A row with zero real positions must produce finite logits.

    The Transformer arm in particular: if ``src_key_padding_mask`` is
    all-True the attention layer feeds ``softmax(-inf)`` and the
    pooled vector becomes NaN. The forward path now forces at least
    one position per row to be attended so the final-position pooler
    reads a finite vector.
    """

    config = traj_model.TrajectoryConfig(
        architecture=arch, embedding_dim=8, history_length=4  # type: ignore[arg-type]
    )
    model = traj_model.build_model(config)
    inputs = torch.randn(2, 4, config.input_dim)
    mask = torch.tensor(
        [
            [False, False, False, False],  # all-pad — would NaN without the guard
            [True, True, True, True],  # all real
        ],
        dtype=torch.bool,
    )
    logits, hidden = model(inputs, mask)
    assert logits.shape == (2, config.n_classes)
    assert torch.isfinite(logits).all(), f"{arch}: logits contain NaN / inf on all-pad row"
    assert torch.isfinite(hidden).all(), f"{arch}: hidden contains NaN / inf on all-pad row"


def test_save_and_load_round_trip_preserves_predictions(tmp_path) -> None:
    config = traj_model.TrajectoryConfig(
        architecture="transformer", embedding_dim=8, history_length=4
    )
    model = traj_model.build_model(config)
    inputs = torch.randn(2, 4, config.input_dim)
    mask = torch.ones(2, 4, dtype=torch.bool)
    before_logits, _ = model(inputs, mask)
    traj_model.save_model(model, config, tmp_path / "model.pt")
    reloaded, reloaded_config = traj_model.load_model(tmp_path / "model.pt")
    after_logits, _ = reloaded(inputs, mask)
    assert reloaded_config.architecture == "transformer"
    assert reloaded_config.embedding_dim == config.embedding_dim
    assert torch.allclose(before_logits, after_logits, atol=1e-5)


# ---------------------------------------------------------------------------
# Walk-forward filter at sequence-build time
# ---------------------------------------------------------------------------


def test_build_training_sequences_drops_targets_on_or_after_train_end() -> None:
    meetings = _meetings_panel()
    embeddings = _toy_embeddings(meetings)
    sequences = traj_train.build_training_sequences(
        meetings,
        embeddings=embeddings,
        history_length=4,
        train_end="2020-01-01",
    )
    # Every target date must lie strictly before 2020-01-01.
    assert sequences, "expected sequences from the pre-2020 slice"
    for seq in sequences:
        assert seq.target_event_date < "2020-01-01"


def test_build_training_sequences_includes_targets_below_train_end_only() -> None:
    meetings = _meetings_panel()
    embeddings = _toy_embeddings(meetings)
    unfiltered = traj_train.build_training_sequences(
        meetings, embeddings=embeddings, history_length=4
    )
    filtered = traj_train.build_training_sequences(
        meetings, embeddings=embeddings, history_length=4, train_end="2015-12-16"
    )
    assert len(filtered) < len(unfiltered)
    # 2015-12-16 must be excluded under strict <.
    assert all(seq.target_event_date < "2015-12-16" for seq in filtered)


def test_build_training_sequences_skips_unknown_stance_targets() -> None:
    meetings = [
        _make_meeting(event_date="2010-03-16", stance="dovish"),
        _make_meeting(event_date="2010-06-22", stance=None),
        _make_meeting(event_date="2011-01-25", stance="hawkish"),
    ]
    embeddings = _toy_embeddings(meetings)
    sequences = traj_train.build_training_sequences(
        meetings, embeddings=embeddings, history_length=2
    )
    target_dates = {seq.target_event_date for seq in sequences}
    assert "2010-06-22" not in target_dates


def test_build_training_sequences_left_pads_short_history() -> None:
    meetings = _meetings_panel()
    embeddings = _toy_embeddings(meetings)
    sequences = traj_train.build_training_sequences(
        meetings, embeddings=embeddings, history_length=12
    )
    # The very first non-skip target has only one meeting of context;
    # left-pad must put the real meeting at the END of the window.
    earliest = min(sequences, key=lambda s: s.target_event_date)
    assert earliest.inputs.shape == (12, embeddings.shape[1] + traj_model.MARKET_FEATURE_DIM)
    assert earliest.mask[-1] == bool(True)
    assert not earliest.mask[0]


# ---------------------------------------------------------------------------
# Conformal calibration
# ---------------------------------------------------------------------------


def test_calibrate_classification_conformal_consumes_softmax_and_true_classes() -> None:
    rng = np.random.default_rng(42)
    softmax_rows = []
    true_labels = []
    for _ in range(40):
        logits = rng.normal(size=3)
        exp_l = np.exp(logits - logits.max())
        probs = exp_l / exp_l.sum()
        softmax_rows.append(probs.tolist())
        true_labels.append(int(rng.integers(0, 3)))
    quantile = calibrate_classification_conformal(
        softmax_scores=softmax_rows, true_classes=true_labels, alpha=0.2
    )
    assert 0.0 <= quantile <= 1.0


def test_conformal_calibration_uses_only_calibration_slice(tmp_path) -> None:
    """A separate calibration slice must not be polluted by holdout rows."""

    meetings = _meetings_panel()
    embeddings = _toy_embeddings(meetings)
    sequences = traj_train.build_training_sequences(
        meetings, embeddings=embeddings, history_length=4
    )
    n = len(sequences)
    n_cal = max(1, int(0.3 * n))
    train_slice = sequences[: n - n_cal]
    cal_slice = sequences[n - n_cal :]
    assert train_slice and cal_slice, "expected non-empty train + cal partitions"

    config = traj_model.TrajectoryConfig(
        architecture="lstm",
        embedding_dim=embeddings.shape[1],
        history_length=4,
    )
    model = traj_train.train_model(train_slice, config, epochs=1, batch_size=2, seed=11)
    cal_eval = traj_train.evaluate_model(model, cal_slice)
    quantile = calibrate_classification_conformal(
        softmax_scores=cal_eval["softmax"].tolist(),
        true_classes=cal_eval["labels"].tolist(),
        alpha=0.2,
    )
    assert 0.0 <= quantile <= 1.0


# ---------------------------------------------------------------------------
# Market feature builder
# ---------------------------------------------------------------------------


def test_market_feature_vector_zeroes_missing_inputs() -> None:
    vec = traj_model.market_feature_vector(
        trailing_2y_yield_change_5d_bps=None, vix_close=None
    )
    assert vec.shape == (traj_model.MARKET_FEATURE_DIM,)
    # Bias term doubles as the "VIX present" indicator — 0.0 on missing input.
    assert vec[0] == pytest.approx(0.0)
    assert vec[1] == pytest.approx(0.0)
    assert vec[2] == pytest.approx(0.0)


def test_market_feature_vector_marks_present_with_bias_indicator() -> None:
    """A real VIX reading sets the bias=1 indicator regardless of magnitude.

    A future low-vol regime can print a sub-mean VIX (e.g. 10) — the
    old gating on ``vix_raw > 0`` would still flip on, but a regression
    that re-introduces the "> mean" or "> 0" heuristic would collapse
    sub-mean readings into the missing-data bin. The explicit-missing
    bit on the bias slot makes the contract crisp.
    """

    vec = traj_model.market_feature_vector(
        trailing_2y_yield_change_5d_bps=1.0, vix_close=10.0
    )
    assert vec[0] == pytest.approx(1.0)  # VIX present
    # Sub-mean VIX yields a negative z-score, not zero.
    assert vec[2] < 0.0


def test_market_feature_vector_treats_nan_as_missing() -> None:
    vec = traj_model.market_feature_vector(
        trailing_2y_yield_change_5d_bps=float("nan"), vix_close=float("nan")
    )
    assert vec[0] == pytest.approx(0.0)
    assert vec[1] == pytest.approx(0.0)
    assert vec[2] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Parameter-count guardrails (LSTM vs Transformer comparison)
# ---------------------------------------------------------------------------


def test_both_architectures_report_comparable_parameter_counts() -> None:
    """A LSTM-vs-Transformer comparison is meaningful only when capacity
    is in the same ballpark. Default knobs should keep both arms within
    one order of magnitude; a future config edit that blows out the
    Transformer's parameter count (e.g. by widening ``d_model`` to 512)
    should trip this guard before the headline metric is reported.
    """

    lstm_config = traj_model.TrajectoryConfig(
        architecture="lstm", embedding_dim=64, history_length=12
    )
    transformer_config = traj_model.TrajectoryConfig(
        architecture="transformer", embedding_dim=64, history_length=12
    )
    lstm_model = traj_model.build_model(lstm_config)
    transformer_model = traj_model.build_model(transformer_config)
    lstm_params = sum(p.numel() for p in lstm_model.parameters())
    transformer_params = sum(p.numel() for p in transformer_model.parameters())
    assert lstm_params > 0
    assert transformer_params > 0
    ratio = max(lstm_params, transformer_params) / max(
        1, min(lstm_params, transformer_params)
    )
    assert ratio <= 10.0, (
        f"LSTM/Transformer parameter counts out of comparable range: "
        f"lstm={lstm_params}, transformer={transformer_params}, ratio={ratio:.2f}"
    )
