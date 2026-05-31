from __future__ import annotations

import numpy as np

from app.data.fed_comms_seqlstm import build_sequences


def _toy(n: int = 30, d_market: int = 4, d_text: int = 5) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    return {
        "market_feat": rng.standard_normal((n, d_market)),
        "text_emb": rng.standard_normal((n, d_text)),
        "text_mask": (np.arange(n) % 3 == 0).astype(float),
        "valid": np.ones(n, dtype=bool),
    }


def test_window_ends_at_origin_and_is_leak_safe() -> None:
    d = _toy()
    seq_len = 22
    out = build_sequences(
        d["market_feat"], d["text_emb"], d["text_mask"], d["valid"], seq_len=seq_len
    )
    # every origin has enough history; the window's LAST row is exactly row t and
    # the FIRST is row t-L+1 — strictly rows <= t, never a future row.
    for j, t in enumerate(out["origin"]):
        assert out["seq"].shape[1] == seq_len
        np.testing.assert_array_equal(out["seq"][j, -1], d["market_feat"][t])
        np.testing.assert_array_equal(out["seq"][j, 0], d["market_feat"][t - seq_len + 1])
        np.testing.assert_array_equal(
            out["seq"][j], d["market_feat"][t - seq_len + 1 : t + 1]
        )


def test_text_is_origin_day_embedding() -> None:
    d = _toy()
    out = build_sequences(d["market_feat"], d["text_emb"], d["text_mask"], d["valid"])
    for j, t in enumerate(out["origin"]):
        np.testing.assert_array_equal(out["text_emb"][j], d["text_emb"][t])
        assert out["text_mask"][j] == d["text_mask"][t]


def test_drops_short_history_and_invalid_origins() -> None:
    d = _toy(n=30)
    seq_len = 22
    d["valid"][25] = False  # an otherwise-eligible origin marked invalid
    out = build_sequences(
        d["market_feat"], d["text_emb"], d["text_mask"], d["valid"], seq_len=seq_len
    )
    origins = set(out["origin"].tolist())
    # no origin with t < L-1 (insufficient history)
    assert all(t >= seq_len - 1 for t in origins)
    assert min(origins) == seq_len - 1
    # the invalid origin is excluded
    assert 25 not in origins
    # everything else in [L-1, n) that is valid is kept
    expected = {t for t in range(seq_len - 1, 30) if t != 25}
    assert origins == expected
