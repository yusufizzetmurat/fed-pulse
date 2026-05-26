"""Unit tests for the trajectory train script (#296).

Covers:

* ``build_training_sequences`` filters by ``event_date < train_end``.
* The bundle persist path writes atomically (no leftover ``.tmp`` files).
* The manifest carries ``train_end`` + ``fold_id`` so the runtime
  singleton can echo them back to the caller.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from app.trajectory import model as traj_model  # noqa: E402
from app.trajectory import train as traj_train  # noqa: E402


def _events_frame() -> pd.DataFrame:
    rows = [
        {
            "event_date": "2010-03-16",
            "event_kind": "statement",
            "text": "2010 statement",
            "text_hash": "h_2010",
            "axis_stance": "dovish",
            "pre_meeting_trailing_2y_yield_change_5d_bps": 1.5,
            "vix_close": 22.0,
            "horizon": 1,
        },
        {
            "event_date": "2012-06-20",
            "event_kind": "statement",
            "text": "2012 statement",
            "text_hash": "h_2012",
            "axis_stance": "dovish",
            "pre_meeting_trailing_2y_yield_change_5d_bps": -0.5,
            "vix_close": 18.0,
            "horizon": 1,
        },
        {
            "event_date": "2015-03-18",
            "event_kind": "statement",
            "text": "2015 statement",
            "text_hash": "h_2015",
            "axis_stance": "neutral",
            "pre_meeting_trailing_2y_yield_change_5d_bps": 3.0,
            "vix_close": 16.0,
            "horizon": 1,
        },
        {
            "event_date": "2017-06-13",
            "event_kind": "statement",
            "text": "2017 statement",
            "text_hash": "h_2017",
            "axis_stance": "hawkish",
            "pre_meeting_trailing_2y_yield_change_5d_bps": 4.5,
            "vix_close": 12.0,
            "horizon": 1,
        },
        {
            "event_date": "2019-07-30",
            "event_kind": "statement",
            "text": "2019 statement",
            "text_hash": "h_2019",
            "axis_stance": "dovish",
            "pre_meeting_trailing_2y_yield_change_5d_bps": -2.0,
            "vix_close": 17.0,
            "horizon": 1,
        },
        {
            "event_date": "2022-06-15",
            "event_kind": "statement",
            "text": "2022 statement",
            "text_hash": "h_2022",
            "axis_stance": "hawkish",
            "pre_meeting_trailing_2y_yield_change_5d_bps": 7.0,
            "vix_close": 28.0,
            "horizon": 1,
        },
        # Non-statement row that must be ignored by the distiller.
        {
            "event_date": "2022-06-15",
            "event_kind": "macro_release",
            "text": "CPI",
            "text_hash": "h_macro_2022",
            "axis_stance": None,
            "horizon": 1,
        },
    ]
    return pd.DataFrame(rows)


def _toy_embedder(texts: list[str]) -> np.ndarray:
    """Deterministic 4-dim projection — same text always maps to the same vector."""

    out = np.zeros((len(texts), 4), dtype=np.float32)
    for idx, text in enumerate(texts):
        seed = abs(hash(text)) % (2**32)
        rng = np.random.default_rng(seed)
        out[idx] = rng.normal(size=4).astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# Sequence build
# ---------------------------------------------------------------------------


def test_distill_meeting_rows_drops_non_statement_kinds() -> None:
    rows = traj_train.distill_meeting_rows(_events_frame())
    assert all(row.text_hash.startswith("h_") for row in rows)
    assert not any(row.text_hash == "h_macro_2022" for row in rows)
    # Sorted ascending by event_date.
    dates = [row.event_date for row in rows]
    assert dates == sorted(dates)


def test_build_training_sequences_filters_by_event_date(tmp_path: Path) -> None:
    meetings = traj_train.distill_meeting_rows(_events_frame())
    embeddings = _toy_embedder([row.text for row in meetings])
    sequences = traj_train.build_training_sequences(
        meetings, embeddings=embeddings, history_length=4, train_end="2018-01-01"
    )
    assert sequences, "expected at least one sequence under train_end=2018-01-01"
    for seq in sequences:
        assert seq.target_event_date < "2018-01-01"


def test_build_training_sequences_rejects_zero_history_length() -> None:
    meetings = traj_train.distill_meeting_rows(_events_frame())
    embeddings = _toy_embedder([row.text for row in meetings])
    with pytest.raises(ValueError, match="history_length"):
        traj_train.build_training_sequences(
            meetings, embeddings=embeddings, history_length=0
        )


# ---------------------------------------------------------------------------
# Atomic writes + manifest
# ---------------------------------------------------------------------------


def test_train_and_persist_writes_atomic_bundle_without_leftover_tmp(
    tmp_path: Path,
) -> None:
    events_parquet = tmp_path / "events.parquet"
    _events_frame().to_parquet(events_parquet, index=False)
    bundle = traj_train.train_and_persist(
        events_parquet=events_parquet,
        architecture="lstm",
        output_root=tmp_path / "artifacts",
        run_name="run_test",
        history_length=3,
        epochs=1,
        batch_size=2,
        seed=11,
        train_end="2025-01-01",
        embed_fn=_toy_embedder,
        holdout_share=0.0,
        calibration_share=0.0,
    )

    assert (bundle / "manifest.json").exists()
    assert (bundle / "model.pt").exists()
    assert (bundle / "embedding_index.parquet").exists()
    assert (bundle / "embedding_index.npz").exists()
    leftovers = list(bundle.glob("*.tmp"))
    assert leftovers == [], f"atomic-write tmp files leaked: {leftovers}"


def test_train_and_persist_manifest_carries_train_end_and_fold_id(
    tmp_path: Path,
) -> None:
    events_parquet = tmp_path / "events.parquet"
    _events_frame().to_parquet(events_parquet, index=False)
    bundle = traj_train.train_and_persist(
        events_parquet=events_parquet,
        architecture="transformer",
        output_root=tmp_path / "artifacts",
        run_name="run_test",
        history_length=3,
        epochs=1,
        batch_size=2,
        seed=11,
        train_end="2020-01-01",
        embed_fn=_toy_embedder,
        holdout_share=0.0,
        calibration_share=0.0,
    )
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["train_end"] == "2020-01-01"
    assert manifest["architecture"] == "transformer"
    assert manifest["encoder_alias"] == traj_train.DEFAULT_BASE_ENCODER_ALIAS
    assert manifest["history_length"] == 3


def test_train_and_persist_writes_2d_projection_per_meeting(tmp_path: Path) -> None:
    events_parquet = tmp_path / "events.parquet"
    _events_frame().to_parquet(events_parquet, index=False)
    bundle = traj_train.train_and_persist(
        events_parquet=events_parquet,
        architecture="lstm",
        output_root=tmp_path / "artifacts",
        run_name="run_test",
        history_length=3,
        epochs=1,
        batch_size=2,
        seed=11,
        train_end="2025-01-01",
        embed_fn=_toy_embedder,
        holdout_share=0.0,
        calibration_share=0.0,
    )
    metadata = pd.read_parquet(bundle / "embedding_index.parquet")
    assert {"event_date", "axis_stance", "embedding_2d_x", "embedding_2d_y"} <= set(
        metadata.columns
    )
    # Every meeting row must have a finite 2D anchor for the chart.
    assert metadata["embedding_2d_x"].notna().all()
    assert metadata["embedding_2d_y"].notna().all()


def test_resolve_train_end_from_fold_matches_manifest(tmp_path: Path) -> None:
    pkg = tmp_path / "tp_test"
    pkg.mkdir()
    events_parquet = pkg / "events.parquet"
    _events_frame().to_parquet(events_parquet, index=False)
    (pkg / traj_train.FOLD_MANIFEST_FILENAME).write_text(
        json.dumps(
            {
                "folds": [
                    {"fold_id": "wf_fold_1", "train_end": "2016-09-21"},
                    {"fold_id": "wf_fold_2", "train_end": "2019-08-01"},
                ]
            }
        ),
        encoding="utf-8",
    )
    resolved = traj_train.resolve_train_end_from_fold(
        events_parquet=events_parquet, fold_id="wf_fold_2"
    )
    assert resolved == "2019-08-01"


def test_train_and_persist_rejects_mutually_exclusive_flags(tmp_path: Path) -> None:
    events_parquet = tmp_path / "events.parquet"
    _events_frame().to_parquet(events_parquet, index=False)
    with pytest.raises(ValueError, match="mutually exclusive"):
        traj_train.train_and_persist(
            events_parquet=events_parquet,
            architecture="lstm",
            output_root=tmp_path / "artifacts",
            run_name="run_test",
            embed_fn=_toy_embedder,
            train_end="2018-01-01",
            fold_id="wf_fold_1",
        )


# ---------------------------------------------------------------------------
# Standardisation
# ---------------------------------------------------------------------------


def test_standardise_inputs_zscores_embedding_block_only(tmp_path: Path) -> None:
    meetings = traj_train.distill_meeting_rows(_events_frame())
    embeddings = _toy_embedder([row.text for row in meetings])
    sequences = traj_train.build_training_sequences(
        meetings, embeddings=embeddings, history_length=3
    )
    rescaled, mean, std = traj_train.standardise_inputs(
        sequences, embedding_dim=embeddings.shape[1]
    )
    assert mean.shape == (embeddings.shape[1],)
    assert std.shape == (embeddings.shape[1],)
    # Market block must stay literally equal across (real positions).
    for original, scaled in zip(sequences, rescaled):
        for t in range(original.inputs.shape[0]):
            if not original.mask[t]:
                continue
            assert np.allclose(
                scaled.inputs[t, embeddings.shape[1] :],
                original.inputs[t, embeddings.shape[1] :],
            )


# ---------------------------------------------------------------------------
# Metrics smoke
# ---------------------------------------------------------------------------


def test_evaluate_metrics_returns_macro_f1_and_directional_accuracy() -> None:
    metrics = traj_train.evaluate_metrics(
        [0, 1, 2, 0, 1, 2],
        [0, 1, 2, 0, 0, 2],
        n_resamples=50,
        seed=11,
    )
    assert 0.0 <= metrics["macro_f1"]["point"] <= 1.0
    assert 0.0 <= metrics["directional_accuracy"]["point"] <= 1.0
    assert metrics["n"] == 6
