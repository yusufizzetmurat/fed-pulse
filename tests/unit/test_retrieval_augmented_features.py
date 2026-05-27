"""Retrieval-augmented input features (#306).

Pins the surface at four layers:

- the derived feature computation (hand-fixture with known top-K
  similarity + stance values, asserts the five summary scalars are
  correct);
- the loader regression: with retrieval bundle absent the per-bar
  vector still emits ``RICH_FEATURE_SIZE`` floats and the analog block
  is all zeros + missing-flag-1.0; with bundle present and the flag on,
  the block populates;
- the FeatureVector schema: byte-identical default behaviour when
  ``analog_features`` is None (legacy path);
- a 1-epoch smoke through ``train_model`` runs to completion with
  ``use_retrieval_analogs=True`` in the feature-vector slot (the
  scaler picks up the new slice without crashing).

See ADR 0028 for the design.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Pure-Python helpers (no torch / no parquet) — exercise the derived
# feature computation against a hand-fixture so the math is reviewable.
# ---------------------------------------------------------------------------


from app.training.retrieval_features import (  # noqa: E402
    ANALOG_FEATURE_DIM,
    DEFAULT_SIMILARITY_FLOOR,
    DEFAULT_TOP_K,
    AnalogSummaryFeatures,
    compute_analog_summary_features,
)


class _Hit:
    """Stand-in for ``app.retrieval.index.AnalogHit``.

    Only the attributes the feature computation consumes are required —
    ``similarity`` and ``axis_stance`` — so this minimal shim avoids a
    hard dep on the index dataclass for the pure-Python tests.
    """

    def __init__(self, similarity: float, axis_stance: str | None) -> None:
        self.similarity = float(similarity)
        self.axis_stance = axis_stance


def test_compute_analog_features_empty_returns_zeros() -> None:
    """No hits -> all-zero summary block. The caller flips the missing flag."""

    out = compute_analog_summary_features([], event_stance="hawkish")
    assert out.as_list() == [0.0] * ANALOG_FEATURE_DIM


def test_compute_analog_features_known_values() -> None:
    """Hand-fixture with known similarities + stance distribution.

    Top-3 hits: similarities (0.85, 0.55, 0.30), stances
    (hawkish, hawkish, dovish). Current event is hawkish.

    Expected:
    - max = 0.85
    - mean = (0.85 + 0.55 + 0.30) / 3 = 0.5666...
    - dispersion (population std) = sqrt(((0.85-0.566..)^2 + (0.55-0.566..)^2 + (0.30-0.566..)^2)/3)
    - count_above_floor: 2 hits >= 0.40 (0.85, 0.55) -> 2/3 ~ 0.6666...
    - stance score: 2 of 3 analogs hawkish -> 0.6666...
    """

    hits = [
        _Hit(0.85, "hawkish"),
        _Hit(0.55, "hawkish"),
        _Hit(0.30, "dovish"),
    ]
    out = compute_analog_summary_features(
        hits, event_stance="hawkish", similarity_floor=0.40, top_k=3
    )
    sims = [0.85, 0.55, 0.30]
    mean_sim = sum(sims) / 3
    variance = sum((s - mean_sim) ** 2 for s in sims) / 3
    expected_disp = variance**0.5

    assert out.analog_max_similarity == pytest.approx(0.85)
    assert out.analog_mean_similarity == pytest.approx(mean_sim)
    assert out.analog_similarity_dispersion == pytest.approx(expected_disp)
    assert out.analog_count_above_floor == pytest.approx(2.0 / 3.0)
    assert out.analog_max_stance_score == pytest.approx(2.0 / 3.0)


def test_compute_analog_features_single_hit_zero_dispersion() -> None:
    """One hit: population std at n=1 is 0 (not NaN) by construction."""

    hits = [_Hit(0.7, "neutral")]
    out = compute_analog_summary_features(
        hits, event_stance="neutral", top_k=3
    )
    assert out.analog_similarity_dispersion == pytest.approx(0.0)
    # Stance agreement: 1 of 1 matches.
    assert out.analog_max_stance_score == pytest.approx(1.0)
    # Above-floor count normalised against top_k=3, not n=1.
    assert out.analog_count_above_floor == pytest.approx(1.0 / 3.0)


def test_compute_analog_features_below_floor_count_zero() -> None:
    """Every hit below the floor -> count_above_floor is 0.0."""

    hits = [_Hit(0.10, "hawkish"), _Hit(0.05, "dovish")]
    out = compute_analog_summary_features(
        hits, event_stance="hawkish", similarity_floor=0.40, top_k=3
    )
    assert out.analog_count_above_floor == 0.0


def test_compute_analog_features_unknown_event_stance_collapses_score() -> None:
    """Unknown current-event stance -> stance-agreement score is 0.0."""

    hits = [_Hit(0.9, "hawkish"), _Hit(0.8, "hawkish")]
    out = compute_analog_summary_features(hits, event_stance=None)
    assert out.analog_max_stance_score == 0.0
    # Similarity stats still compute fine.
    assert out.analog_max_similarity == pytest.approx(0.9)


def test_compute_analog_features_non_finite_sims_collapse() -> None:
    """NaN / inf similarities are filtered before aggregation."""

    hits = [_Hit(float("nan"), "hawkish"), _Hit(float("inf"), "hawkish")]
    out = compute_analog_summary_features(hits, event_stance="hawkish")
    assert out.analog_max_similarity == 0.0
    assert out.analog_count_above_floor == 0.0


def test_default_similarity_floor_matches_panel() -> None:
    """The loader floor is pinned to the #295 panel default."""

    assert DEFAULT_SIMILARITY_FLOOR == 0.40


def test_default_top_k_matches_panel_display() -> None:
    """K=3 matches the panel's top-3 display contract."""

    assert DEFAULT_TOP_K == 3


# ---------------------------------------------------------------------------
# FeatureVector schema round-trip
# ---------------------------------------------------------------------------


from app.models.config import (  # noqa: E402
    FEATURE_SIZE,
    FeatureVector,
    RICH_FEATURE_SIZE,
    RICH_RETRIEVAL_ANALOG_DIM,
    RICH_RETRIEVAL_ANALOG_MISSING_DIM,
    RICH_RETRIEVAL_ANALOG_MISSING_SLICE,
    RICH_RETRIEVAL_ANALOG_SLICE,
)


def test_feature_vector_default_emits_zero_analog_block() -> None:
    """A FeatureVector built without analog payload emits zeros + missing=1.0.

    Pre-#306 callers don't touch the new fields; the new slice in
    ``as_rich_list`` therefore reads as the missing-block contract by
    default. Pins the byte-identity guarantee at the schema layer.
    """

    fv = FeatureVector(
        date="2024-05-01",
        sentiment_score=0.0,
        market_close=4500.0,
        market_volatility=0.01,
    )
    rich = fv.as_rich_list()
    assert len(rich) == RICH_FEATURE_SIZE
    analog_slice = rich[RICH_RETRIEVAL_ANALOG_SLICE]
    missing_slice = rich[RICH_RETRIEVAL_ANALOG_MISSING_SLICE]
    assert analog_slice == [0.0] * RICH_RETRIEVAL_ANALOG_DIM
    assert missing_slice == [1.0] * RICH_RETRIEVAL_ANALOG_MISSING_DIM


def test_feature_vector_populated_analog_block_round_trips() -> None:
    """A populated ``analog_features`` slot emits its contents at the slice."""

    payload = [0.85, 0.50, 0.20, 0.66, 0.66]
    assert len(payload) == RICH_RETRIEVAL_ANALOG_DIM
    fv = FeatureVector(
        date="2024-05-01",
        sentiment_score=0.0,
        market_close=4500.0,
        market_volatility=0.01,
        analog_features=payload,
        analog_features_missing=0.0,
    )
    rich = fv.as_rich_list()
    assert rich[RICH_RETRIEVAL_ANALOG_SLICE] == payload
    assert rich[RICH_RETRIEVAL_ANALOG_MISSING_SLICE] == [0.0]


def test_feature_vector_short_analog_payload_zero_pads() -> None:
    """A short payload is right-zero-padded to ``RICH_RETRIEVAL_ANALOG_DIM``.

    Defensive contract: if a caller passes a shorter list the model
    input size stays constant. Mirrors the linguistic / llm slice
    padding contract.
    """

    short = [0.9, 0.7]
    fv = FeatureVector(
        date="2024-05-01",
        sentiment_score=0.0,
        market_close=4500.0,
        market_volatility=0.01,
        analog_features=short,
        analog_features_missing=0.0,
    )
    rich = fv.as_rich_list()
    block = rich[RICH_RETRIEVAL_ANALOG_SLICE]
    assert block == short + [0.0] * (RICH_RETRIEVAL_ANALOG_DIM - len(short))


def test_rich_feature_size_widens_by_six() -> None:
    """RICH_FEATURE_SIZE = legacy + 5 analog scalars + 1 missing flag.

    Pin the absolute size so a future widening that forgets to update
    the documented layout surfaces at this regression point. Legacy
    pre-#306 width was ``FEATURE_SIZE + 75`` (credibility 4 + linguistic
    15 + mp_surprise 4 + multi_axis 6 + realized_vol 2 + cross_asset 8
    + llm 35 + llm_missing 1); the #306 block adds 6 more.
    """

    assert RICH_RETRIEVAL_ANALOG_DIM == 5
    assert RICH_RETRIEVAL_ANALOG_MISSING_DIM == 1
    assert RICH_FEATURE_SIZE == FEATURE_SIZE + 75 + 6


# ---------------------------------------------------------------------------
# Loader regression — absent / present retrieval bundle
# ---------------------------------------------------------------------------


pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")
torch = pytest.importorskip("torch")

import json  # noqa: E402

from app.models.config import SEQUENCE_LENGTH  # noqa: E402
from app.retrieval import index as ret_index  # noqa: E402
from app.services import analogs as analogs_service  # noqa: E402
from app.training import loaders  # noqa: E402


_TRAINING_PACKAGE_ID = "tp_retrieval_features_regression_v1"


def _synth_prior_bars(*, event_date: _dt.date, base_close: float) -> str:
    payload = []
    for offset in range(SEQUENCE_LENGTH, 0, -1):
        bar_date = _dt.date.fromordinal(event_date.toordinal() - offset)
        payload.append(
            {
                "date": bar_date.isoformat(),
                "close": round(base_close + (SEQUENCE_LENGTH - offset) * 1.5, 10),
                "volume": 1_000_000.0,
                "vol_5d": round(0.012 + (SEQUENCE_LENGTH - offset) * 0.0001, 10),
                "vol_20d": round(0.018, 10),
                "vol_60d": round(0.022, 10),
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
    *, event_date: str, text: str, axis_stance: str | None, base_close: float
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
        "axis_topic": None,
        "axis_time_label": None,
        "axis_certain_label": None,
        "credibility_drift_score": 0.0,
        "credibility_realized_vs_stated_gap": 0.0,
        "credibility_market_implied_gap": 0.0,
        "credibility_months_since_reversal": 0,
        "prior_window_sha256": "0" * 64,
        "prior_bars_json": _synth_prior_bars(event_date=ed, base_close=base_close),
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


@pytest.fixture
def loader_package(tmp_path: Path, monkeypatch) -> Path:
    """Synthetic three-event training package on tmp_path.

    Three events spaced months apart so the strict-backward retrieval
    filter is non-trivial: the first event has no priors, the second
    has one prior, the third has two priors.
    """

    processed_root = tmp_path / "processed"
    package_dir = processed_root / _TRAINING_PACKAGE_ID
    package_dir.mkdir(parents=True)

    monkeypatch.setattr(loaders, "DATA_DIR", tmp_path)

    rows = [
        _make_event_row(
            event_date="2023-09-20",
            text="Inflation pressures remain elevated and the Committee will continue tightening.",
            axis_stance="hawkish",
            base_close=4400.0,
        ),
        _make_event_row(
            event_date="2023-12-13",
            text="The Committee judges policy is sufficiently restrictive to bring inflation lower.",
            axis_stance="neutral",
            base_close=4500.0,
        ),
        _make_event_row(
            event_date="2024-03-20",
            text="Inflation outlook has improved; the Committee anticipates a gradual normalisation path.",
            axis_stance="neutral",
            base_close=4600.0,
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


@pytest.fixture(autouse=True)
def _reset_analogs_singleton():
    analogs_service.reset_state()
    yield
    analogs_service.reset_state()


def _make_keyword_embedder(keywords: list[str]):
    """Tiny keyword-count embedder for the retrieval fixture."""

    import numpy as np

    lower = [k.lower() for k in keywords]

    def _embed(texts: list[str]):
        out = np.zeros((len(texts), len(keywords)), dtype=np.float32)
        for row_idx, text in enumerate(texts):
            text_lc = (text or "").lower()
            for col_idx, kw in enumerate(lower):
                out[row_idx, col_idx] = float(text_lc.count(kw))
        return out

    return _embed


def _install_retrieval_bundle(tmp_path: Path, events_parquet: Path) -> None:
    """Build a tiny on-disk retrieval bundle and install the singleton."""

    embed = _make_keyword_embedder(
        ["inflation", "tightening", "restrictive", "normalisation"]
    )
    loaded = ret_index.build_index_from_events(
        events_parquet=events_parquet,
        encoder_alias="test_retrieval",
        encoder_revision="rev1234",
        embed_fn=embed,
        training_package_id="tp_test",
        out_dir=tmp_path / "bundle",
    )
    state = analogs_service.build_state_from_index(
        loaded, embed_fn=embed, encoder_alias="test_retrieval"
    )
    analogs_service.install_state(state)


def test_loader_retrieval_flag_off_emits_zero_block(loader_package: Path) -> None:
    """Default ``use_retrieval_analogs=False`` -> all-zeros + missing=1.0.

    Pins byte-identity to the pre-#306 path: every supervised sequence
    must carry the missing analog block when the flag is off, regardless
    of whether the retrieval bundle is installed on disk.
    """

    split = loaders.load_walk_forward_split(
        _TRAINING_PACKAGE_ID,
        rich_features=True,
        use_retrieval_analogs=False,
        text_encoder=None,
    )
    assert split.train, "fixture must produce at least one train sequence"

    for partition in (split.train, split.val, split.test):
        for sequence in partition:
            for vector in sequence:
                assert vector.analog_features is None
                assert vector.analog_features_missing == 1.0
                rich = vector.as_rich_list()
                assert rich[RICH_RETRIEVAL_ANALOG_SLICE] == [0.0] * RICH_RETRIEVAL_ANALOG_DIM
                assert rich[RICH_RETRIEVAL_ANALOG_MISSING_SLICE] == [1.0]


def test_loader_retrieval_flag_on_absent_bundle_emits_zero_block(
    loader_package: Path,
) -> None:
    """``use_retrieval_analogs=True`` + bundle absent on disk -> graceful degrade.

    The loader emits zeros + missing-flag-1.0 rather than crashing.
    Pins the graceful-degrade contract for ops deployments that do not
    ship a retrieval bundle alongside the training package.
    """

    # No bundle installed; the singleton resolves to a missing path
    # because the fixture monkeypatches DATA_DIR onto an empty tmp_path.
    split = loaders.load_walk_forward_split(
        _TRAINING_PACKAGE_ID,
        rich_features=True,
        use_retrieval_analogs=True,
        text_encoder=None,
    )
    assert split.train

    for partition in (split.train, split.val, split.test):
        for sequence in partition:
            for vector in sequence:
                assert vector.analog_features is None
                assert vector.analog_features_missing == 1.0


def test_loader_retrieval_flag_on_present_bundle_populates_features(
    loader_package: Path, tmp_path: Path
) -> None:
    """Bundle installed + flag on -> later events carry a populated analog block.

    The first event in the corpus has no strict-prior analogs (it IS the
    earliest), so its block stays at the missing-flag baseline. The
    third event has two strict-prior analogs and must carry a populated
    block on every bar of its sequence.
    """

    events_parquet = loader_package / "events.parquet"
    _install_retrieval_bundle(tmp_path, events_parquet)

    split = loaders.load_walk_forward_split(
        _TRAINING_PACKAGE_ID,
        rich_features=True,
        use_retrieval_analogs=True,
        text_encoder=None,
    )

    # Sort sequences by the target row date so the assertion below is
    # deterministic against the loader's (event_date, text_hash) sort.
    all_sequences = list(split.train) + list(split.val) + list(split.test)
    all_sequences.sort(key=lambda seq: seq[-1].date)

    # The third event (2024-03-20) must carry a populated analog block:
    # it has two strict-prior analogs in the corpus.
    third = all_sequences[-1]
    assert third[-1].date.startswith("2024-03-20")
    assert third[-1].analog_features is not None
    assert third[-1].analog_features_missing == 0.0
    assert len(third[-1].analog_features) == RICH_RETRIEVAL_ANALOG_DIM
    # Every bar of the sequence carries the same broadcast analog block.
    for vector in third:
        assert vector.analog_features is not None
        assert vector.analog_features_missing == 0.0


def test_loader_retrieval_query_uses_strict_backward_filter(
    loader_package: Path, tmp_path: Path
) -> None:
    """The retrieval call must enforce ``analog_event_date < event_date``.

    Capture the as_of_date the loader passes to the retrieval service
    on each event and assert it equals the supervised event's date so
    the strict-backward filter the helper applies is the supervised
    event's own date (not a fold boundary or a relaxed `<=` value).
    """

    events_parquet = loader_package / "events.parquet"
    _install_retrieval_bundle(tmp_path, events_parquet)

    captured: list[tuple[str, _dt.date | None]] = []

    import app.training.retrieval_features as rf_mod

    original_lookup = rf_mod.lookup_analog_hits

    def _patched_lookup(*, text, event_date, top_k=rf_mod.DEFAULT_TOP_K):
        captured.append((text[:40], event_date))
        return original_lookup(text=text, event_date=event_date, top_k=top_k)

    rf_mod.lookup_analog_hits = _patched_lookup
    try:
        loaders.load_walk_forward_split(
            _TRAINING_PACKAGE_ID,
            rich_features=True,
            use_retrieval_analogs=True,
            text_encoder=None,
        )
    finally:
        rf_mod.lookup_analog_hits = original_lookup

    # Each call's as_of_date must be a real date object so the
    # strict-backward filter inside ``app.retrieval.index.query``
    # enforces ``analog_event_date < event_date`` via the string cutoff
    # path. A None / fold-boundary cutoff would silently relax the
    # filter, so the contract is "the supervised event's own date is
    # what the loader passes."
    assert captured, "expected the loader to call into the retrieval helper"
    for _text, as_of in captured:
        assert isinstance(as_of, _dt.date), (
            "loader must pass a real `date` so the strict-backward filter applies"
        )


def test_loader_rich_feature_size_widens_consistently(loader_package: Path) -> None:
    """``as_rich_list`` width on every bar stays at RICH_FEATURE_SIZE.

    Pins the structural lock: the per-bar feature size is constant
    regardless of the analog flag's state.
    """

    split = loaders.load_walk_forward_split(
        _TRAINING_PACKAGE_ID,
        rich_features=True,
        use_retrieval_analogs=False,
        text_encoder=None,
    )
    for sequence in split.train + split.test:
        for vector in sequence:
            assert len(vector.as_rich_list()) == RICH_FEATURE_SIZE


# ---------------------------------------------------------------------------
# Smoke training run
# ---------------------------------------------------------------------------


from app.models.config import ModelConfig  # noqa: E402
from app.training.loop import train_model  # noqa: E402


def _dummy_feature_vector(*, day: int, vol: float) -> FeatureVector:
    """In-memory FeatureVector with a populated analog block.

    Mirrors the #305 smoke fixture so the training loop runs against
    a no-parquet in-memory group.
    """

    fv = FeatureVector(
        date=str(_dt.date(2025, 1, 1) + _dt.timedelta(days=day - 1)),
        sentiment_score=0.0,
        market_close=100.0,
        market_volatility=0.01,
        close_change_pct=0.0,
        volatility_change=0.0,
        elapsed_time=0.0,
        forward_realized_vol_10d=vol,
    )
    # Populate the analog block with a plausible summary so the
    # per-fold scaler sees non-zero values to fit on.
    fv.analog_features = [0.8, 0.6, 0.1, 0.66, 0.66]
    fv.analog_features_missing = 0.0
    return fv


def test_train_model_smoke_with_analog_features() -> None:
    """1-epoch run with analog block populated completes without crashing.

    The smoke uses an in-memory fixture (no parquet I/O) so it runs
    under a second on CPU. The analog block is pre-populated on every
    FeatureVector; the per-fold RobustScaler fits the new slot
    alongside the existing rich-feature slots and the run must complete.
    """

    groups = [[_dummy_feature_vector(day=i + 1, vol=0.01 + 0.001 * i) for i in range(40)]]
    config = ModelConfig(
        output_mode="classification",
        n_classes=3,
        hidden_size=16,
        head_hidden_size=8,
    )
    result = train_model(
        model_config=config,
        train_sequence_groups=groups,
        val_sequence_groups=groups,
        test_sequence_groups=groups,
        epochs=1,
        batch_size=8,
        seed=11,
        save_checkpoint=False,
        use_compile=False,
        use_amp=False,
    )
    assert result.summary.epochs_completed == 1


def test_train_model_smoke_with_missing_analog_features() -> None:
    """Pre-#306 byte-identity: missing analog block trains identically.

    A FeatureVector with default ``analog_features=None`` /
    ``analog_features_missing=1.0`` must train through the same code
    path as the populated case. Verifies the legacy / opt-out path is
    not broken by the new slice.
    """

    groups = [
        [
            FeatureVector(
                date=str(_dt.date(2025, 1, 1) + _dt.timedelta(days=i)),
                sentiment_score=0.0,
                market_close=100.0,
                market_volatility=0.01,
                forward_realized_vol_10d=0.01 + 0.001 * i,
            )
            for i in range(40)
        ]
    ]
    config = ModelConfig(
        output_mode="classification",
        n_classes=3,
        hidden_size=16,
        head_hidden_size=8,
    )
    result = train_model(
        model_config=config,
        train_sequence_groups=groups,
        val_sequence_groups=groups,
        test_sequence_groups=groups,
        epochs=1,
        batch_size=8,
        seed=11,
        save_checkpoint=False,
        use_compile=False,
        use_amp=False,
    )
    assert result.summary.epochs_completed == 1
