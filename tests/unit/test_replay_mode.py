"""Replay-mode (time-machine) coverage.

Covers three surfaces:

1. :func:`app.services.replay.resolve_fold_for_date` — picks the right
   walk-forward fold for a given as-of date, returns a structured
   ``FoldRef.unavailable(...)`` when the manifest / checkpoint is
   missing on disk.

2. The /analyze API path under ``as_of_date`` — emits the ``replay``
   block + ``realised_outcome`` reveal on a happy path, and surfaces a
   422 when the fold is unavailable.

3. :func:`app.services.forecaster.load_for_fold` — loads the per-fold
   checkpoint into an isolated model (does not touch the live serving
   singleton), caches repeated loads on the same path, and raises
   ``FileNotFoundError`` when the checkpoint file is absent.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")
pytest.importorskip("torch")
pytest.importorskip("transformers")

from fastapi.testclient import TestClient  # noqa: E402

import app.main as main_mod  # noqa: E402
from app.services import forecaster as forecaster_service  # noqa: E402
from app.services import replay as replay_service  # noqa: E402


# ---------------------------------------------------------------------------
# resolve_fold_for_date
# ---------------------------------------------------------------------------


def _write_manifest(
    tmp_path: Path, folds: list[dict], checkpoint_dir: Path | None = None
) -> Path:
    if checkpoint_dir is not None:
        for fold in folds:
            fold.setdefault("checkpoint_dir", str(checkpoint_dir))
    manifest = tmp_path / "fold_manifest.json"
    manifest.write_text(
        json.dumps({"training_package_id": "canonical", "folds": folds}),
        encoding="utf-8",
    )
    return manifest


def _seed_fold_checkpoint(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "forecaster_best.pt").write_bytes(b"stub")
    return root


def test_resolve_returns_unavailable_when_manifest_missing(tmp_path):
    ref = replay_service.resolve_fold_for_date(
        date(2024, 1, 5), manifest_path=tmp_path / "missing.json"
    )
    assert ref.available is False
    assert ref.reason == "fold_manifest_missing"


def test_resolve_returns_unavailable_when_no_fold_predates_as_of(tmp_path):
    manifest = _write_manifest(
        tmp_path,
        [
            {
                "fold_id": "wf_fold_1",
                "train_end": "2024-12-31",
                "test_start": "2025-01-02",
                "test_end": "2025-06-30",
            }
        ],
    )
    ref = replay_service.resolve_fold_for_date(
        date(2024, 6, 1), manifest_path=manifest
    )
    assert ref.available is False
    assert ref.reason == "no_fold_before_as_of"


def test_resolve_returns_unavailable_when_checkpoint_file_missing(tmp_path):
    manifest = _write_manifest(
        tmp_path,
        [
            {
                "fold_id": "wf_fold_1",
                "train_end": "2023-12-31",
                "test_start": "2024-01-02",
                "test_end": "2024-06-30",
                "checkpoint_dir": str(tmp_path / "nope"),
            }
        ],
    )
    ref = replay_service.resolve_fold_for_date(
        date(2024, 3, 1), manifest_path=manifest
    )
    assert ref.available is False
    assert ref.reason == "fold_checkpoint_missing"


def test_resolve_picks_latest_fold_whose_train_end_precedes_as_of(tmp_path):
    ckpt_a = _seed_fold_checkpoint(tmp_path / "fold_a")
    ckpt_b = _seed_fold_checkpoint(tmp_path / "fold_b")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "training_package_id": "canonical",
                "folds": [
                    {
                        "fold_id": "wf_fold_1",
                        "train_end": "2022-12-31",
                        "test_start": "2023-01-02",
                        "test_end": "2023-06-30",
                        "checkpoint_dir": str(ckpt_a),
                    },
                    {
                        "fold_id": "wf_fold_2",
                        "train_end": "2023-06-30",
                        "test_start": "2023-07-03",
                        "test_end": "2023-12-31",
                        "checkpoint_dir": str(ckpt_b),
                    },
                    {
                        "fold_id": "wf_fold_3",
                        "train_end": "2024-06-30",
                        "test_start": "2024-07-01",
                        "test_end": "2024-12-31",
                        "checkpoint_dir": str(tmp_path / "no_ckpt"),
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    # ``as_of`` = 2024-01-15 sits inside wf_fold_2's test window; the
    # right serving fold has train_end < 2024-01-15, which is wf_fold_2
    # itself (train_end 2023-06-30, strictly before 2024-01-15).
    ref = replay_service.resolve_fold_for_date(
        date(2024, 1, 15), manifest_path=manifest
    )
    assert ref.available is True
    assert ref.fold_id == "wf_fold_2"
    assert ref.train_end == date(2023, 6, 30)
    assert ref.forecaster_checkpoint == ckpt_b / "forecaster_best.pt"


# ---------------------------------------------------------------------------
# /analyze API path
# ---------------------------------------------------------------------------


def _stub_market_path(monkeypatch):
    monkeypatch.setattr(
        main_mod,
        "analyze_text",
        lambda _: {
            "label": "HAWKISH",
            "score": 0.62,
            "raw": [{"label": "HAWKISH", "score": 0.62}],
        },
    )
    monkeypatch.setattr(
        main_mod,
        "fetch_market_snapshot",
        lambda **_: {
            "symbol": "^GSPC",
            "requested_date": "2024-03-15",
            "date_used": "2024-03-15",
            "lookback_days": 5,
            "close": 5000.0,
            "volatility_5d": 0.01,
        },
    )
    monkeypatch.setattr(
        main_mod,
        "fetch_market_history",
        lambda **_: [
            {"date": "2024-03-12", "close": 4980.0, "volatility_5d": 0.011},
            {"date": "2024-03-13", "close": 5000.0, "volatility_5d": 0.010},
        ],
    )
    monkeypatch.setattr(main_mod, "parse_horizon_steps", lambda _: 3)
    monkeypatch.setattr(
        main_mod,
        "fetch_forward_trading_dates",
        lambda **_: ["2024-03-18", "2024-03-19", "2024-03-20"],
    )
    monkeypatch.setattr(
        main_mod,
        "forecast_quantitative_series",
        lambda **_: {
            "prediction": {"close": 5050.0, "volatility": 0.012, "horizon": "3d"},
            "model": {
                "checkpoint_path": "backend/models/forecaster_best.pt",
                "checkpoint_exists": True,
                "checkpoint_loaded": True,
                "runtime_mode": "fast",
                "hidden_size": 64,
                "num_layers": 2,
                "dropout": 0.15,
                "head_hidden_size": 32,
                "close_scale": 10000.0,
                "sequence_length": 5,
            },
            "series": {
                "timestamps": ["2024-03-12", "2024-03-13"],
                "history_close": [4980.0, 5000.0],
                "history_volatility": [0.011, 0.01],
                "forecast_timestamps": ["2024-03-18", "2024-03-19", "2024-03-20"],
                "forecast_close": [5020.0, 5040.0, 5050.0],
                "forecast_close_lower": [5000.0, 5015.0, 5020.0],
                "forecast_close_upper": [5040.0, 5060.0, 5080.0],
                "forecast_volatility": [0.011, 0.012, 0.012],
                "forecast_volatility_lower": [0.009, 0.010, 0.010],
                "forecast_volatility_upper": [0.013, 0.014, 0.015],
                "forecast_confidence_level": 0.8,
                "volatility_scale": {"suggested_ymin": 0.0, "suggested_ymax": 0.02},
            },
        },
    )
    monkeypatch.setattr(main_mod, "checkpoint_exists", lambda: True)


def test_replay_mode_returns_422_when_per_fold_checkpoints_missing(
    monkeypatch, tmp_path
):
    _stub_market_path(monkeypatch)
    # Point the manifest at a tmp file that resolves a fold whose
    # checkpoint_dir directory exists but carries no ``forecaster_best.pt``
    # -- mirrors the production state today where the manifest has been
    # extended with the new ``checkpoint_dir`` field but per-fold
    # training has not been run.
    empty_dir = tmp_path / "fold_dir"
    empty_dir.mkdir(parents=True)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "training_package_id": "canonical",
                "folds": [
                    {
                        "fold_id": "wf_fold_1",
                        "train_end": "2023-12-31",
                        "test_start": "2024-01-02",
                        "test_end": "2024-06-30",
                        "checkpoint_dir": str(empty_dir),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(replay_service, "_DEFAULT_MANIFEST_PATH", manifest)
    client = TestClient(main_mod.app)

    response = client.post(
        "/analyze",
        json={
            "text": "Recent indicators…",
            "date": "2024-03-15",
            "symbol": "^GSPC",
            "horizon": "3d",
            "include_realized": False,
            "as_of_date": "2024-03-15",
        },
    )
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert isinstance(detail, dict), detail
    assert detail["error"] == "replay_unavailable"
    assert detail["message"] == "fold_checkpoint_missing"


def test_replay_mode_emits_replay_and_realised_blocks_when_fold_resolves(
    monkeypatch, tmp_path
):
    _stub_market_path(monkeypatch)
    ckpt = tmp_path / "fold"
    ckpt.mkdir(parents=True)
    (ckpt / "forecaster_best.pt").write_bytes(b"stub")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "training_package_id": "canonical",
                "folds": [
                    {
                        "fold_id": "wf_fold_2",
                        "train_end": "2023-12-31",
                        "test_start": "2024-01-02",
                        "test_end": "2024-06-30",
                        "checkpoint_dir": str(ckpt),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(replay_service, "_DEFAULT_MANIFEST_PATH", manifest)
    # The stub ``forecaster_best.pt`` is unparseable as a torch payload;
    # patch the per-fold loader to a no-op so the wire under test (422 vs
    # 200, replay block populated, ``forecaster_checkpoint_rewound``
    # flipped) is exercised without standing up a real per-fold model.
    # ``load_for_fold`` now returns ``(model, metadata)`` directly so the
    # caller doesn't have to re-enter the cache.
    monkeypatch.setattr(
        forecaster_service, "load_for_fold", lambda path: (object(), None)
    )
    monkeypatch.setattr(
        forecaster_service, "get_fold_metadata", lambda path: None
    )
    monkeypatch.setattr(
        replay_service,
        "realised_outcome",
        lambda as_of, symbol="^GSPC": {
            "as_of_date": as_of.isoformat(),
            "symbol": symbol,
            "horizons": [
                {
                    "horizon": 1,
                    "log_return": 0.01,
                    "realised_volatility_5d_post_event": 0.005,
                    "close": 5050.0,
                    "date": "2024-03-18",
                },
                {
                    "horizon": 5,
                    "log_return": 0.02,
                    "realised_volatility_5d_post_event": 0.007,
                    "close": 5100.0,
                    "date": "2024-03-22",
                },
                {
                    "horizon": 10,
                    "log_return": None,
                    "realised_volatility_5d_post_event": None,
                    "close": None,
                    "date": None,
                },
            ],
        },
    )

    client = TestClient(main_mod.app)
    response = client.post(
        "/analyze",
        json={
            "text": "Recent indicators…",
            "date": "2024-03-15",
            "symbol": "^GSPC",
            "horizon": "3d",
            "include_realized": False,
            "as_of_date": "2024-03-15",
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["replay"] is not None
    assert body["replay"]["as_of_date"] == "2024-03-15"
    assert body["replay"]["fold_id"] == "wf_fold_2"
    assert body["replay"]["train_end"] == "2023-12-31"
    assert body["replay"]["classifier_rewind"] is False
    # The per-fold checkpoint resolved AND the load wire fired
    # successfully (load_for_fold returned a non-None model), so the
    # rewound flag must be True on this happy path.
    assert body["replay"]["forecaster_checkpoint_rewound"] is True
    assert any("classifier rewind" in note.lower() for note in body["replay"]["notes"])
    assert body["realised_outcome"] is not None
    horizons = {h["horizon"]: h for h in body["realised_outcome"]["horizons"]}
    assert horizons[1]["log_return"] == pytest.approx(0.01)
    assert horizons[10]["log_return"] is None


def test_live_mode_payload_is_unchanged_when_as_of_date_omitted(monkeypatch):
    _stub_market_path(monkeypatch)
    client = TestClient(main_mod.app)
    response = client.post(
        "/analyze",
        json={
            "text": "Recent indicators…",
            "date": "2026-03-15",
            "symbol": "^GSPC",
            "horizon": "3d",
            "include_realized": False,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body.get("replay") is None
    assert body.get("realised_outcome") is None


# ---------------------------------------------------------------------------
# Per-fold manifest + load_for_fold wire
# ---------------------------------------------------------------------------


def test_canonical_manifest_carries_checkpoint_dir_for_every_fold():
    """The shipped manifest under data/processed/canonical/ must point
    every fold at a ``checkpoint_dir`` so resolve_fold_for_date can
    surface the per-fold checkpoint path. Regression on the gap that
    /analyze replay 422'd before the manifest was extended."""

    repo_root = Path(__file__).resolve().parents[2]
    manifest_path = (
        repo_root
        / "data"
        / "processed"
        / "canonical"
        / "fold_manifest_expanding_walk_forward.json"
    )
    assert manifest_path.exists(), (
        f"canonical fold manifest missing at {manifest_path}; replay "
        "mode cannot resolve a per-fold checkpoint without it"
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    folds = payload.get("folds")
    assert isinstance(folds, list) and folds, "manifest carries no folds"
    for fold in folds:
        fold_id = fold.get("fold_id")
        ckpt_dir = fold.get("checkpoint_dir")
        assert isinstance(ckpt_dir, str) and ckpt_dir, (
            f"fold {fold_id!r} is missing the checkpoint_dir field; "
            "_resolve_path will return None and replay will 422"
        )
        # The relative path convention is anchored at the repo root.
        resolved = repo_root / ckpt_dir
        assert resolved.parent.exists(), (
            f"checkpoint_dir parent does not exist on disk: {resolved.parent}"
        )


def test_load_for_fold_raises_when_checkpoint_file_missing(tmp_path):
    """Regression on the pre-#655 422 surface: when the per-fold
    checkpoint file is absent, ``load_for_fold`` raises
    :class:`FileNotFoundError` so the route handler can convert it to a
    422 ``fold_checkpoint_missing`` -- not a generic 500."""

    forecaster_service.clear_fold_load_cache()
    missing = tmp_path / "wf_fold_99" / "forecaster_best.pt"
    with pytest.raises(FileNotFoundError):
        forecaster_service.load_for_fold(missing)


def test_load_for_fold_returns_isolated_model_distinct_from_live_singleton(
    tmp_path, monkeypatch
):
    """When a per-fold checkpoint exists on disk, ``load_for_fold``
    must:

    * return a ``ForecasterServingModel`` instance distinct from the
      module-level ``_model`` singleton (live mode stays isolated from
      per-request fold loads), AND
    * cache subsequent calls on the same path so repeated replay
      requests do not re-read the .pt file off disk.

    The loader is patched at the ``_read_checkpoint_payload`` +
    ``build_serving_forecaster`` boundary so the test runs without a
    real .pt artefact; the assertions cover the isolation contract,
    not the on-disk format."""

    forecaster_service.clear_fold_load_cache()

    # Sentinel "live" singleton -- distinguishable by ``identity``.
    live_singleton = object()
    monkeypatch.setattr(forecaster_service, "_model", live_singleton)

    # Stub a per-fold checkpoint on disk.
    ckpt_path = tmp_path / "wf_fold_3" / "forecaster_best.pt"
    ckpt_path.parent.mkdir(parents=True)
    ckpt_path.write_bytes(b"stub-bytes")

    # Synthetic payload returned by the patched checkpoint reader -- the
    # shape ``_get_model`` / ``load_for_fold`` expects (a dict carrying
    # ``model_state_dict`` + ``model_config``).
    fake_payload = {"model_state_dict": {}, "model_config": {}}
    monkeypatch.setattr(
        forecaster_service,
        "_read_checkpoint_payload",
        lambda path, device: fake_payload,
    )
    monkeypatch.setattr(
        forecaster_service,
        "_validate_serving_contract",
        lambda path, *, record_status=True: (True, "sidecar_absent"),
    )
    monkeypatch.setattr(
        forecaster_service,
        "_coerce_model_config",
        lambda raw: object(),
    )
    monkeypatch.setattr(
        forecaster_service,
        "_load_state_dict_loose",
        lambda model, state, path: None,
    )
    monkeypatch.setattr(
        forecaster_service,
        "_resolve_device",
        lambda: __import__("torch").device("cpu"),
    )

    # Per-fold model double: tracks ``eval()`` + ``.to(device)`` calls
    # so we can verify the loader walked them, and supplies enough
    # surface for the cache stash.
    class _Stub:
        def __init__(self):
            self.eval_called = False
            self.to_called_with = None

        def to(self, device):
            self.to_called_with = device
            return self

        def eval(self):
            self.eval_called = True
            return self

    stubs: list[_Stub] = []

    def _factory(resolved):
        s = _Stub()
        stubs.append(s)
        return s

    import app.models.factory as factory_mod

    monkeypatch.setattr(factory_mod, "build_serving_forecaster", _factory)
    monkeypatch.setattr(
        forecaster_service,
        "_checkpoint_metadata",
        lambda payload, ckpt, model: {"close_scale": 7777.0, "encoder_key": "fold_stub"},
    )

    first_model, first_meta = forecaster_service.load_for_fold(ckpt_path)
    assert first_model is not live_singleton, (
        "load_for_fold leaked the live singleton; replay mode must "
        "return an isolated per-fold instance"
    )
    assert forecaster_service._model is live_singleton, (
        "load_for_fold mutated the module-level singleton; live /analyze "
        "would now serve the per-fold weights"
    )
    assert first_model.eval_called and first_model.to_called_with is not None
    assert first_meta is not None and first_meta.get("close_scale") == 7777.0

    # ``get_fold_metadata`` is the standalone fallback for callers that
    # only need metadata; the tuple return covers the common combined
    # case but the standalone surface stays.
    standalone_meta = forecaster_service.get_fold_metadata(ckpt_path)
    assert standalone_meta is first_meta

    # Second call on the same path must return the cached instance, not
    # re-invoke the factory.
    second_model, _ = forecaster_service.load_for_fold(ckpt_path)
    assert second_model is first_model, "load_for_fold did not hit the LRU cache"
    assert len(stubs) == 1, "factory was invoked twice for the same path"
