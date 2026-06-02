"""Encoder-provenance sidecar for embedding artifacts (MLC1 hardening).

Every built embedding artifact gets a ``<artifact>.encoder.json`` sidecar
recording which encoder produced it, so the provenance ambiguity that made the
May-31 ``fomc_embeddings.parquet`` unverifiable can never recur.
"""

from __future__ import annotations

from app.services.encoder_provenance import (
    read_encoder_sidecar,
    write_encoder_sidecar,
)


def test_write_then_read_roundtrip(tmp_path):
    art = tmp_path / "fomc_embeddings.parquet"
    art.write_bytes(b"fake-parquet")
    prov = {"model_id": "gtfintechlab/FOMC-RoBERTa", "revision": "abc123", "hidden_size": 768}

    sidecar = write_encoder_sidecar(art, prov, built_at="2026-06-02T00:00:00+00:00")

    assert sidecar.name == "fomc_embeddings.parquet.encoder.json"
    got = read_encoder_sidecar(art)
    assert got is not None
    assert got["model_id"] == "gtfintechlab/FOMC-RoBERTa"
    assert got["revision"] == "abc123"
    assert got["hidden_size"] == 768
    assert got["built_at"] == "2026-06-02T00:00:00+00:00"


def test_built_at_is_stamped_when_not_supplied(tmp_path):
    art = tmp_path / "emb.parquet"
    art.write_bytes(b"x")
    write_encoder_sidecar(art, {"model_id": "m", "revision": None, "hidden_size": 384})
    got = read_encoder_sidecar(art)
    assert got is not None
    assert isinstance(got["built_at"], str) and got["built_at"]


def test_read_returns_none_when_sidecar_absent(tmp_path):
    art = tmp_path / "missing.parquet"
    assert read_encoder_sidecar(art) is None
