from __future__ import annotations

import numpy as np
import pytest

from app.data.intraday_rv_arch import build_sequences

torch = pytest.importorskip("torch")


def test_window_ends_at_origin_and_is_leak_safe() -> None:
    rng = np.random.default_rng(0)
    n, d, seq_len = 40, 6, 22
    feat = rng.standard_normal((n, d))
    valid = np.ones(n, dtype=bool)
    out = build_sequences(feat, valid, seq_len=seq_len)
    for j, t in enumerate(out["origin"]):
        # window is exactly rows [t-L+1 .. t]: last row is t, first is t-L+1,
        # strictly rows <= t so no future row can leak in.
        assert out["seq"].shape[1] == seq_len
        np.testing.assert_array_equal(out["seq"][j, -1], feat[t])
        np.testing.assert_array_equal(out["seq"][j, 0], feat[t - seq_len + 1])
        np.testing.assert_array_equal(out["seq"][j], feat[t - seq_len + 1 : t + 1])


def test_drops_short_history_and_invalid_origins() -> None:
    rng = np.random.default_rng(1)
    n, seq_len = 40, 22
    feat = rng.standard_normal((n, 6))
    valid = np.ones(n, dtype=bool)
    valid[30] = False
    out = build_sequences(feat, valid, seq_len=seq_len)
    origins = set(out["origin"].tolist())
    assert min(origins) == seq_len - 1  # no origin with insufficient history
    assert 30 not in origins  # invalid origin excluded
    assert origins == {t for t in range(seq_len - 1, n) if t != 30}


def test_tcn_is_causal_future_cannot_change_past_output() -> None:
    """A change to a future input row must not move the TCN's last-step output.

    Build the model, push a window through, then perturb a row that is AFTER the
    output position and confirm the output is bit-identical. Since the TCN pools
    the LAST step, we instead verify the per-step output stream: perturbing input
    at step k leaves outputs at steps < k unchanged (strict causality).
    """

    from app.data.intraday_rv_arch import _build_tcn

    torch.manual_seed(0)
    d_in, length = 6, 22
    model = _build_tcn(d_in)
    model.eval()

    # tap the conv stack directly to get the full per-step output stream
    def conv_stream(x_seq: "torch.Tensor") -> "torch.Tensor":
        x = x_seq.transpose(1, 2)
        for blk in model.blocks:
            x = blk(x)
        return x  # (B, C, L)

    x = torch.randn(1, length, d_in)
    with torch.no_grad():
        out_a = conv_stream(x)
        x2 = x.clone()
        k = 15
        x2[0, k] += 5.0  # perturb input at step k
        out_b = conv_stream(x2)
    # outputs at steps < k must be unchanged; step k and later may move
    torch.testing.assert_close(out_a[:, :, :k], out_b[:, :, :k])
    assert not torch.allclose(out_a[:, :, k], out_b[:, :, k])
