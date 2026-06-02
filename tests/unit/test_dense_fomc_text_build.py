"""build_fomc_embeddings must size embeddings to the actual encoder, not a
hardcoded 768. FOMC-RoBERTa is 1024-dim; the old ``np.zeros(768)`` empty-text
fallback would emit mixed-width rows (NaNs past column 768)."""

from __future__ import annotations

import types

import pandas as pd

import app.data.dense_fomc_text as dft
import app.services.text_encoder as te


def test_build_fomc_embeddings_dim_follows_encoder_not_hardcoded(tmp_path, monkeypatch):
    # Two statements; the second yields no encodable chunks (empty-text fallback).
    monkeypatch.setattr(
        dft,
        "statement_dates_and_text",
        lambda _p: {"2024-01-31": "a hawkish statement", "2024-03-20": "blank"},
    )

    def _fake_encode(text: str, classifier=None):
        if text == "blank":
            return []
        return [types.SimpleNamespace(embedding=[0.1] * 1024)]

    monkeypatch.setattr(te, "encode_chunks", _fake_encode)
    monkeypatch.setattr(te, "assert_primary_model_loaded", lambda: None)
    monkeypatch.setattr(
        te,
        "loaded_encoder_provenance",
        lambda: {"model_id": "gtfintechlab/FOMC-RoBERTa", "revision": "r", "hidden_size": 1024},
    )

    out = tmp_path / "emb.parquet"
    dft.build_fomc_embeddings("ignored.parquet", out, force=True)

    df = pd.read_parquet(out)
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    assert len(emb_cols) == 1024
    # The empty-text row must be zeros(1024) — no NaNs from a 768/1024 width clash.
    assert not df[emb_cols].isna().any().any()
