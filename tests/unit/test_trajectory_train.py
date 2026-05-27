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
from typing import Any

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
        enforce_param_cap=False,  # toy 4-dim embedder + 4x64 transformer exceeds the #332 cap; legit override for trainer wiring tests.
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


# ---------------------------------------------------------------------------
# Walk-forward carve + leakage tests
# ---------------------------------------------------------------------------


def test_train_and_persist_standardisation_fits_on_train_slice_only(
    tmp_path: Path,
) -> None:
    """The persisted ``feature_mean`` must match what the train slice
    alone produces — fitting on the full pool would leak holdout
    statistics into the inference path.
    """

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
        train_end="2018-01-01",  # pre-cutoff sequences only ↔ pre-2018 targets
        embed_fn=_toy_embedder,
        holdout_share=0.0,
        calibration_share=0.0,
    )
    persisted = np.load(bundle / "embedding_index.npz", allow_pickle=False)
    persisted_mean = persisted["feature_mean"]
    persisted_std = persisted["feature_std"]

    # Rebuild the expected statistics from scratch using ONLY pre-2018 targets.
    meetings = traj_train.distill_meeting_rows(_events_frame())
    embeddings = _toy_embedder([row.text for row in meetings])
    all_sequences = traj_train.build_training_sequences(
        meetings, embeddings=embeddings, history_length=3, train_end=None
    )
    pre = [seq for seq in all_sequences if seq.target_event_date < "2018-01-01"]
    expected_mean, expected_std = traj_train.fit_standardisation_stats(
        pre, embedding_dim=embeddings.shape[1]
    )
    assert np.allclose(persisted_mean, expected_mean, atol=1e-5)
    assert np.allclose(persisted_std, expected_std, atol=1e-5)


def test_train_and_persist_pca_axes_fit_on_train_slice_only(
    tmp_path: Path,
) -> None:
    """The persisted PCA axes must match a train-slice-only fit.

    A regression that re-fits ``project_2d`` on the full corpus would
    surface here as a mismatch between the persisted ``pca_mean`` /
    ``pca_components`` and the train-only reconstruction.
    """

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
        train_end="2018-01-01",
        embed_fn=_toy_embedder,
        holdout_share=0.0,
        calibration_share=0.0,
    )
    persisted = np.load(bundle / "embedding_index.npz", allow_pickle=False)
    assert "pca_mean" in persisted.files
    assert "pca_components" in persisted.files

    meetings = traj_train.distill_meeting_rows(_events_frame())
    embeddings = _toy_embedder([row.text for row in meetings])
    train_mask = np.array(
        [row.event_date < "2018-01-01" for row in meetings], dtype=bool
    )
    expected_mean, expected_components = traj_train.fit_pca_axes(embeddings[train_mask])
    assert np.allclose(persisted["pca_mean"], expected_mean, atol=1e-5)
    assert np.allclose(persisted["pca_components"], expected_components, atol=1e-5)


def test_train_and_persist_parquet_filters_to_train_end(tmp_path: Path) -> None:
    """The persisted ``embedding_index.parquet`` must drop post-train_end rows.

    The runtime singleton's history window walks this file; surfacing
    a meeting the model never trained on would be a walk-forward leak
    at inference time.
    """

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
        train_end="2018-01-01",
        embed_fn=_toy_embedder,
        holdout_share=0.0,
        calibration_share=0.0,
    )
    metadata = pd.read_parquet(bundle / "embedding_index.parquet")
    assert not metadata.empty
    assert (metadata["event_date"] < "2018-01-01").all(), (
        f"persisted parquet leaked post-train_end rows: "
        f"{metadata.loc[metadata['event_date'] >= '2018-01-01', 'event_date'].tolist()}"
    )


def test_train_and_persist_parquet_carries_market_columns(tmp_path: Path) -> None:
    """The persisted parquet must carry the market columns the runtime
    ``_market_for`` reads — otherwise inference reads None and the
    market feature vector collapses to a no-signal block (train-vs-
    inference distribution skew).
    """

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
    assert "pre_meeting_trailing_2y_yield_change_5d_bps" in metadata.columns
    assert "vix_close" in metadata.columns
    # Every row must carry finite market inputs (the fixture frame
    # provides them for every meeting).
    assert metadata["pre_meeting_trailing_2y_yield_change_5d_bps"].notna().all()
    assert metadata["vix_close"].notna().all()


def test_train_and_persist_manifest_includes_parameter_count(tmp_path: Path) -> None:
    """``manifest.json`` must report ``model_parameter_count`` so the
    LSTM-vs-Transformer comparison can be audited later.
    """

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
        train_end="2025-01-01",
        embed_fn=_toy_embedder,
        holdout_share=0.0,
        calibration_share=0.0,
        enforce_param_cap=False,  # toy 4-dim embedder + 4x64 transformer exceeds the #332 cap; legit override for trainer wiring tests.
    )
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert "model_parameter_count" in manifest
    assert manifest["model_parameter_count"] > 0


def test_train_and_persist_feeds_calibration_slice_to_conformal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Spy on ``calibrate_classification_conformal`` to confirm the
    trainer hands it the calibration partition's predictions — not
    the train or holdout slices.
    """

    captured: dict[str, Any] = {}

    def _spy(*, softmax_scores, true_classes, alpha):
        captured["softmax_rows"] = list(softmax_scores)
        captured["true_classes"] = list(true_classes)
        captured["alpha"] = alpha
        # Return a plausible quantile so the trainer persists conformal.json.
        return 0.5

    monkeypatch.setattr(
        "app.trajectory.train.calibrate_classification_conformal", _spy
    )

    events_parquet = tmp_path / "events.parquet"
    _events_frame().to_parquet(events_parquet, index=False)
    traj_train.train_and_persist(
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
        calibration_share=0.34,  # ensure non-empty cal partition
        conformal_alpha=0.2,
    )
    assert captured, "calibrate_classification_conformal was not called"

    # Reconstruct the calibration partition from the same upstream
    # builder and confirm sizes line up. The trainer carves cal as
    # the temporal tail of the train slice.
    from typing import Any as _A  # noqa: F401 — local import keeps the spy block self-contained.

    meetings = traj_train.distill_meeting_rows(_events_frame())
    embeddings = _toy_embedder([row.text for row in meetings])
    all_sequences = traj_train.build_training_sequences(
        meetings, embeddings=embeddings, history_length=3, train_end=None
    )
    pre = [seq for seq in all_sequences if seq.target_event_date < "2025-01-01"]
    n_pre = len(pre)
    n_cal = max(0, int(n_pre * 0.34))
    expected_cal = pre[max(1, n_pre - n_cal) :]
    assert len(captured["softmax_rows"]) == len(expected_cal), (
        f"spy saw {len(captured['softmax_rows'])} rows, expected {len(expected_cal)}"
    )
    assert len(captured["true_classes"]) == len(expected_cal)
    assert captured["alpha"] == 0.2


# ---------------------------------------------------------------------------
# Atomic-write crash safety + npz fallback
# ---------------------------------------------------------------------------


def test_train_and_persist_does_not_leave_partial_bundle_on_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If ``os.replace`` crashes part-way through the persist, no
    half-built bundle should survive — the manifest is the truth
    marker the runtime keys on, so any earlier file without a
    matching manifest is dead weight that the next training run
    must overwrite cleanly.
    """

    real_replace = traj_train.os.replace
    call_count = {"n": 0}

    def _crash_after_first(src, dst):  # type: ignore[no-untyped-def]
        call_count["n"] += 1
        if call_count["n"] >= 2:
            raise RuntimeError("simulated crash after first atomic-write")
        return real_replace(src, dst)

    monkeypatch.setattr(traj_train.os, "replace", _crash_after_first)

    events_parquet = tmp_path / "events.parquet"
    _events_frame().to_parquet(events_parquet, index=False)
    with pytest.raises(RuntimeError, match="simulated crash"):
        traj_train.train_and_persist(
            events_parquet=events_parquet,
            architecture="lstm",
            output_root=tmp_path / "artifacts",
            run_name="run_crash",
            history_length=3,
            epochs=1,
            batch_size=2,
            seed=11,
            train_end="2025-01-01",
            embed_fn=_toy_embedder,
            holdout_share=0.0,
            calibration_share=0.0,
        )
    bundle = tmp_path / "artifacts" / "run_crash"
    if bundle.exists():
        # If the crash landed after some writes the manifest must NOT
        # be present (it is the truth marker; without it the runtime
        # treats the bundle as missing).
        assert not (bundle / "manifest.json").exists(), (
            "crash left a manifest behind — runtime would treat the partial "
            "bundle as complete"
        )


# ---------------------------------------------------------------------------
# NpzFile fallback (finding 9)
# ---------------------------------------------------------------------------


def test_service_load_state_falls_back_when_npz_missing_feature_mean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A hand-built npz WITHOUT ``feature_mean`` must NOT crash the
    runtime loader — the previous ``npz.get`` call raised KeyError
    (NpzFile has no .get) and the wrapping ``except Exception``
    swallowed the bundle as "missing". The probe is now an explicit
    ``in npz.files`` check + a logged warning.
    """

    from app.services import trajectory as trajectory_service

    bundle = tmp_path / "trajectory_test"
    bundle.mkdir()
    # Minimal parquet + npz pair (deliberately omit feature_mean / feature_std).
    pd.DataFrame(
        [
            {
                "event_date": "2020-01-01",
                "text_hash": "h_2020",
                "axis_stance": "neutral",
                "embedding_2d_x": 0.0,
                "embedding_2d_y": 0.0,
                "pre_meeting_trailing_2y_yield_change_5d_bps": 0.0,
                "vix_close": 20.0,
            }
        ]
    ).to_parquet(bundle / "embedding_index.parquet", index=False)
    np.savez(
        bundle / "embedding_index.npz",
        embeddings=np.zeros((1, 4), dtype=np.float32),
    )
    # Minimal model + manifest so bundle_available returns True.
    config = traj_model.TrajectoryConfig(
        architecture="lstm", embedding_dim=4, history_length=2
    )
    model = traj_model.build_model(config)
    traj_model.save_model(model, config, bundle / "model.pt")
    (bundle / "manifest.json").write_text(
        json.dumps(
            {
                "architecture": "lstm",
                "encoder_alias": "test",
                "encoder_revision": "",
                "train_end": "2025-01-01",
                "history_length": 2,
                "embedding_dim": 4,
                "n_classes": 3,
                "row_count": 1,
                "stance_classes": ["hawkish", "dovish", "neutral"],
                "config": config.to_dict(),
                "built_at_utc": "2026-05-26T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("FED_PULSE_TRAJECTORY_DIR", str(bundle))
    trajectory_service.reset_state()
    try:
        state = trajectory_service.get_state()
        assert state is not None, "bundle without feature_mean should still load"
        # Defaults must be sane — zeros for mean, ones for std (post-floor).
        assert np.allclose(state.feature_mean, 0.0)
        assert np.allclose(state.feature_std, 1.0)
    finally:
        trajectory_service.reset_state()
