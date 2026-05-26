"""Unit tests for ``app.retrieval.index`` (historical analog index, #294)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.retrieval import index as ret_index


def _fixture_statements() -> list[dict]:
    """A small canonical set of historical FOMC statements for the index.

    Three rows so the top-k path has both a winner and an ordering to
    assert. Each row carries the columns the production builder reads
    from ``events.parquet`` plus the supervised stance / realised vol
    columns the endpoint surfaces back to the caller.
    """

    rows = [
        {
            "event_date": "2008-12-16",
            "event_kind": "statement",
            "text": "The Committee will employ all available tools to promote economic recovery and price stability.",
            "axis_stance": "dovish",
            "forward_realized_vol_10d": 0.045,
            "horizon": 1,
        },
        {
            "event_date": "2022-09-21",
            "event_kind": "statement",
            "text": "Inflation remains elevated. The Committee anticipates ongoing increases in the target range.",
            "axis_stance": "hawkish",
            "forward_realized_vol_10d": 0.022,
            "horizon": 1,
        },
        {
            "event_date": "2015-12-16",
            "event_kind": "statement",
            "text": "The Committee judges that economic conditions warrant a gradual removal of policy accommodation.",
            "axis_stance": "neutral",
            "forward_realized_vol_10d": 0.011,
            "horizon": 1,
        },
    ]
    for row in rows:
        row["text_hash"] = hashlib.sha256(row["text"].encode("utf-8")).hexdigest()
    return rows


def _write_events_parquet(directory: Path, rows: list[dict]) -> Path:
    parquet = directory / "events.parquet"
    pd.DataFrame(rows).to_parquet(parquet, index=False)
    return parquet


def _make_keyword_embedder(keywords: list[str]):
    """Build a deterministic bag-of-keywords embedder for the tests.

    Each input string is projected onto a ``len(keywords)``-dim vector
    where entry ``i`` is the number of times ``keywords[i]`` appears
    (case-insensitive). This gives the tests a stable retrieval signal
    without standing up a real transformer.
    """

    lower = [k.lower() for k in keywords]

    def _embed(texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), len(keywords)), dtype=np.float32)
        for row_idx, text in enumerate(texts):
            text_lc = (text or "").lower()
            for col_idx, kw in enumerate(lower):
                out[row_idx, col_idx] = float(text_lc.count(kw))
        return out

    return _embed


def test_build_index_persists_parquet_embeddings_manifest(tmp_path: Path) -> None:
    rows = _fixture_statements()
    events_parquet = _write_events_parquet(tmp_path, rows)
    embed = _make_keyword_embedder(["inflation", "recovery", "accommodation"])

    loaded = ret_index.build_index_from_events(
        events_parquet=events_parquet,
        encoder_alias="test_retrieval",
        encoder_revision="rev1234",
        embed_fn=embed,
        training_package_id="tp_test",
        out_dir=tmp_path / "bundle",
    )

    assert loaded.size == 3
    assert loaded.embedding_dim == 3
    assert loaded.encoder_alias == "test_retrieval"

    bundle = tmp_path / "bundle"
    assert (bundle / "index.parquet").exists()
    assert (bundle / "embeddings.npy").exists()
    assert (bundle / "manifest.json").exists()

    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["encoder_alias"] == "test_retrieval"
    assert manifest["row_count"] == 3
    assert manifest["embedding_dim"] == 3
    assert manifest["training_package_id"] == "tp_test"

    persisted = pd.read_parquet(bundle / "index.parquet")
    assert set(persisted.columns) >= {
        "event_date",
        "text_hash",
        "axis_stance",
        "subsequent_vol_regime",
        "excerpt",
    }
    # Supervised target column must never land in the persisted bundle.
    assert "forward_realized_vol_10d" not in persisted.columns
    assert persisted["axis_stance"].tolist() == ["dovish", "hawkish", "neutral"]
    # Buckets pinned by VOL_REGIME_BUCKET_EDGES = (0.012, 0.020):
    # 0.045 -> high, 0.022 -> high, 0.011 -> calm.
    assert persisted["subsequent_vol_regime"].tolist() == ["high", "high", "calm"]


def test_query_returns_top1_matching_known_keyword(tmp_path: Path) -> None:
    """Embed a query that dominates on a single fixture row; top-1 must match."""

    rows = _fixture_statements()
    events_parquet = _write_events_parquet(tmp_path, rows)
    embed = _make_keyword_embedder(["inflation", "recovery", "accommodation"])
    loaded = ret_index.build_index_from_events(
        events_parquet=events_parquet,
        encoder_alias="test_retrieval",
        encoder_revision="rev1234",
        embed_fn=embed,
        training_package_id="tp_test",
        out_dir=tmp_path / "bundle",
    )

    # Query embedded with the same projection: dominated by the
    # "inflation" axis, so the 2022 hawkish statement must rank first.
    query_vec = embed(["Inflation outlook remains elevated"])[0]
    hits = ret_index.query(loaded, query_vec, k=3)

    assert len(hits) == 3
    assert hits[0].event_date == "2022-09-21"
    assert hits[0].axis_stance == "hawkish"
    # 0.022 > VOL_REGIME_BUCKET_EDGES[1] (0.020) -> "high".
    assert hits[0].subsequent_vol_regime == "high"
    assert hits[0].similarity == pytest.approx(1.0, abs=1e-5)
    # Strictly descending ordering by similarity.
    assert hits[0].similarity >= hits[1].similarity >= hits[2].similarity


def test_query_recovery_keyword_routes_to_2008_dovish_row(tmp_path: Path) -> None:
    rows = _fixture_statements()
    events_parquet = _write_events_parquet(tmp_path, rows)
    embed = _make_keyword_embedder(["inflation", "recovery", "accommodation"])
    loaded = ret_index.build_index_from_events(
        events_parquet=events_parquet,
        encoder_alias="test_retrieval",
        encoder_revision="rev1234",
        embed_fn=embed,
        training_package_id="tp_test",
        out_dir=tmp_path / "bundle",
    )

    query_vec = embed(["The Committee will promote economic recovery"])[0]
    hits = ret_index.query(loaded, query_vec, k=1)

    assert len(hits) == 1
    assert hits[0].event_date == "2008-12-16"
    assert hits[0].axis_stance == "dovish"


def test_load_index_round_trips_a_built_bundle(tmp_path: Path) -> None:
    rows = _fixture_statements()
    events_parquet = _write_events_parquet(tmp_path, rows)
    embed = _make_keyword_embedder(["inflation", "recovery", "accommodation"])
    bundle = tmp_path / "bundle"
    built = ret_index.build_index_from_events(
        events_parquet=events_parquet,
        encoder_alias="test_retrieval",
        encoder_revision="rev1234",
        embed_fn=embed,
        training_package_id="tp_test",
        out_dir=bundle,
    )

    reloaded = ret_index.load_index(bundle)
    assert reloaded.size == built.size
    assert reloaded.embedding_dim == built.embedding_dim
    assert reloaded.encoder_alias == "test_retrieval"
    assert reloaded.encoder_revision == "rev1234"
    assert reloaded.training_package_id == "tp_test"
    assert reloaded.metadata["event_date"].tolist() == [
        "2008-12-16",
        "2022-09-21",
        "2015-12-16",
    ]


def test_build_index_skips_non_statement_event_kinds(tmp_path: Path) -> None:
    rows = _fixture_statements()
    rows.append(
        {
            "event_date": "2024-03-20",
            "event_kind": "minutes",
            "text": "The Committee discussed inflation dynamics at length.",
            "axis_stance": "hawkish",
            "forward_realized_vol_10d": 0.018,
            "horizon": 1,
            "text_hash": hashlib.sha256(b"minutes-2024").hexdigest(),
        }
    )
    rows.append(
        {
            "event_date": "2024-03-21",
            "event_kind": "macro_release",
            "text": "CPI release.",
            "axis_stance": "neutral",
            "forward_realized_vol_10d": 0.005,
            "horizon": 1,
            "text_hash": hashlib.sha256(b"macro-2024").hexdigest(),
        }
    )
    events_parquet = _write_events_parquet(tmp_path, rows)
    embed = _make_keyword_embedder(["inflation", "recovery", "accommodation"])

    loaded = ret_index.build_index_from_events(
        events_parquet=events_parquet,
        encoder_alias="test_retrieval",
        encoder_revision="rev1234",
        embed_fn=embed,
        training_package_id=None,
        out_dir=tmp_path / "bundle",
    )

    assert loaded.size == 3
    assert "2024-03-20" not in loaded.metadata["event_date"].tolist()
    assert "2024-03-21" not in loaded.metadata["event_date"].tolist()


def test_query_clips_k_to_index_size(tmp_path: Path) -> None:
    rows = _fixture_statements()
    events_parquet = _write_events_parquet(tmp_path, rows)
    embed = _make_keyword_embedder(["inflation", "recovery", "accommodation"])
    loaded = ret_index.build_index_from_events(
        events_parquet=events_parquet,
        encoder_alias="test_retrieval",
        encoder_revision="rev1234",
        embed_fn=embed,
        training_package_id=None,
        out_dir=tmp_path / "bundle",
    )

    query_vec = embed(["inflation"])[0]
    hits = ret_index.query(loaded, query_vec, k=100)
    assert len(hits) == 3


def test_query_returns_empty_for_empty_index() -> None:
    empty = ret_index.LoadedIndex(
        embeddings=np.zeros((0, 4), dtype=np.float32),
        metadata=pd.DataFrame(columns=list(ret_index.METADATA_COLUMNS)),
        encoder_alias="test",
        encoder_revision="",
        training_package_id=None,
        built_at_utc="",
    )
    assert ret_index.query(empty, np.zeros(4, dtype=np.float32), k=5) == []


def test_query_returns_empty_when_query_vector_is_zero(tmp_path: Path) -> None:
    rows = _fixture_statements()
    events_parquet = _write_events_parquet(tmp_path, rows)
    embed = _make_keyword_embedder(["inflation", "recovery", "accommodation"])
    loaded = ret_index.build_index_from_events(
        events_parquet=events_parquet,
        encoder_alias="test_retrieval",
        encoder_revision="rev1234",
        embed_fn=embed,
        training_package_id=None,
        out_dir=tmp_path / "bundle",
    )

    zero_vec = np.zeros(loaded.embedding_dim, dtype=np.float32)
    assert ret_index.query(loaded, zero_vec, k=3) == []


def test_query_raises_on_dimension_mismatch(tmp_path: Path) -> None:
    rows = _fixture_statements()
    events_parquet = _write_events_parquet(tmp_path, rows)
    embed = _make_keyword_embedder(["inflation", "recovery", "accommodation"])
    loaded = ret_index.build_index_from_events(
        events_parquet=events_parquet,
        encoder_alias="test_retrieval",
        encoder_revision="rev1234",
        embed_fn=embed,
        training_package_id=None,
        out_dir=tmp_path / "bundle",
    )

    with pytest.raises(ValueError, match="dim"):
        ret_index.query(loaded, np.zeros(loaded.embedding_dim + 1, dtype=np.float32), k=3)


def test_load_index_raises_when_bundle_is_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="retrieval index incomplete"):
        ret_index.load_index(tmp_path / "does-not-exist")


def test_query_self_match_suppression(tmp_path: Path) -> None:
    """A query whose text_hash matches an indexed row drops the trivial hit.

    The runtime singleton always passes the cleaned query's sha256 as
    ``exclude_text_hash`` so submitting the literal text of an indexed
    statement never returns the similarity ≈ 1.0 self-row.
    """

    rows = [
        {
            "event_date": "2024-03-20",
            "event_kind": "statement",
            "text": "FOO",
            "axis_stance": "neutral",
            "forward_realized_vol_10d": 0.011,
            "horizon": 1,
            "text_hash": hashlib.sha256(b"FOO").hexdigest(),
        }
    ]
    events_parquet = _write_events_parquet(tmp_path, rows)
    embed = _make_keyword_embedder(["foo"])
    loaded = ret_index.build_index_from_events(
        events_parquet=events_parquet,
        encoder_alias="test_retrieval",
        encoder_revision="rev1234",
        embed_fn=embed,
        training_package_id=None,
        out_dir=tmp_path / "bundle",
    )

    query_vec = embed(["FOO"])[0]
    exclude = ret_index.text_hash_for_query("FOO")
    hits = ret_index.query(loaded, query_vec, k=5, exclude_text_hash=exclude)
    assert hits == []


def test_query_as_of_date_filters_strictly_backward(tmp_path: Path) -> None:
    """Only rows with event_date < as_of_date stay in the candidate pool."""

    rows = _fixture_statements()  # 2008-12-16, 2022-09-21, 2015-12-16
    events_parquet = _write_events_parquet(tmp_path, rows)
    embed = _make_keyword_embedder(["inflation", "recovery", "accommodation"])
    loaded = ret_index.build_index_from_events(
        events_parquet=events_parquet,
        encoder_alias="test_retrieval",
        encoder_revision="rev1234",
        embed_fn=embed,
        training_package_id=None,
        out_dir=tmp_path / "bundle",
    )

    query_vec = embed(["inflation"])[0]
    hits = ret_index.query(loaded, query_vec, k=5, as_of_date="2016-01-01")
    dates = {hit.event_date for hit in hits}
    assert dates <= {"2008-12-16", "2015-12-16"}
    assert "2022-09-21" not in dates

    # as_of_date earlier than every indexed row -> empty result (no
    # padding from future rows).
    early = ret_index.query(loaded, query_vec, k=5, as_of_date="2000-01-01")
    assert early == []


def test_build_index_persists_train_end_in_manifest(tmp_path: Path) -> None:
    rows = _fixture_statements()
    events_parquet = _write_events_parquet(tmp_path, rows)
    embed = _make_keyword_embedder(["inflation", "recovery", "accommodation"])
    ret_index.build_index_from_events(
        events_parquet=events_parquet,
        encoder_alias="test_retrieval",
        encoder_revision="rev1234",
        embed_fn=embed,
        training_package_id="tp_test",
        out_dir=tmp_path / "bundle",
        train_end="2018-01-01",
    )

    manifest = json.loads((tmp_path / "bundle" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["train_end"] == "2018-01-01"
    reloaded = ret_index.load_index(tmp_path / "bundle")
    assert reloaded.train_end == "2018-01-01"


def test_build_index_atomic_writes_leave_no_tmp_files(tmp_path: Path) -> None:
    rows = _fixture_statements()
    events_parquet = _write_events_parquet(tmp_path, rows)
    embed = _make_keyword_embedder(["inflation", "recovery", "accommodation"])
    ret_index.build_index_from_events(
        events_parquet=events_parquet,
        encoder_alias="test_retrieval",
        encoder_revision="rev1234",
        embed_fn=embed,
        training_package_id=None,
        out_dir=tmp_path / "bundle",
    )

    leftovers = list((tmp_path / "bundle").glob("*.tmp"))
    assert leftovers == [], f"atomic-write tmp files leaked: {leftovers}"


def test_build_index_excerpt_truncates_long_text(tmp_path: Path) -> None:
    long_text = "Inflation " * 200
    rows = [
        {
            "event_date": "2020-03-15",
            "event_kind": "statement",
            "text": long_text,
            "axis_stance": "dovish",
            "forward_realized_vol_10d": 0.07,
            "horizon": 1,
            "text_hash": hashlib.sha256(long_text.encode("utf-8")).hexdigest(),
        }
    ]
    events_parquet = _write_events_parquet(tmp_path, rows)
    embed = _make_keyword_embedder(["inflation"])

    loaded = ret_index.build_index_from_events(
        events_parquet=events_parquet,
        encoder_alias="test_retrieval",
        encoder_revision="rev1234",
        embed_fn=embed,
        training_package_id=None,
        out_dir=tmp_path / "bundle",
    )

    excerpt = loaded.metadata.iloc[0]["excerpt"]
    assert len(excerpt) <= ret_index.EXCERPT_CHARS
