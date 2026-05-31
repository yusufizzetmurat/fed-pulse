"""Phase 8 event-row dataset builder.

Produces one row per ``(FOMC event date x event_kind x forecast horizon)``
for downstream event-study forecasting. Each row carries the document text,
multi-axis labels (where available), a 4-vector credibility check, a 20
trading-day prior market window, and abnormal-return targets at horizons
``h in {1d, 5d, 10d, 30d}``.

Two outputs are emitted side-by-side in the same CLI run:

- ``data/processed/<pkg>/events.parquet``        -- collapsed view, one row
  per ``(event_date, event_kind, asset_symbol, horizon)``. Multi-source
  duplicates are pinned to one preferred source via ``_SOURCE_PREFERENCE``.
- ``data/processed/<pkg>/events_full.parquet``    -- full view, one row per
  ``(event_date, event_kind, source, source_record_id, asset_symbol, horizon)``.
  Keeps sentence-level data from every source for source-stratified and
  sentence-level analyses.

Both parquets share the same column schema; downstream consumers pick the
view that fits their question.

Schema (one row per event x kind x horizon x asset):

- ``event_date``                ISO date (string)
- ``event_kind``                One of ``{statement, minutes, speech,
                                 testimony, press_conference, macro_release}``
- ``document_id``               sha256(source|event_date|event_kind)[:16]
- ``text_hash``                 sha256 of the concatenated document text
- ``source``                    Registry source (preferred order documented
                                 in ``_SOURCE_PREFERENCE``)
- ``as_of_ts``                  ISO 8601 timestamp at which the event
                                 announcement becomes public knowledge.
                                 Placeholder convention:
                                 ``T19:00:00Z`` for FOMC statements /
                                 minutes / press conferences / testimonies
                                 (2pm ET = 19:00 UTC during US standard time
                                 -- we use the wall-clock 19:00Z throughout
                                 so the dataset is reproducible regardless
                                 of DST), ``T14:00:00Z`` for speeches
                                 (typical morning ET delivery), and
                                 ``T13:00:00Z`` for macro-release events
                                 (BLS CPI / NFP 8:30 ET slot, rounded to
                                 stay clear of DST hand-waving).
- ``text``                      Raw concatenated document text
- ``token_count``               Whitespace-token count of ``text`` (used as
                                 a coarse truncation budget upstream)
- ``axis_stance``               ``hawkish | dovish | neutral`` or None
- ``axis_time``, ``axis_certainty``, ``axis_factor``
                                Multi-axis labels (None when unavailable)
- ``credibility_*``             Four credibility-vector axes
- ``prior_window_sha256``       sha256 over the concatenated prior bars,
                                 written for reproducibility
- ``prior_bars_json``           JSON-encoded list of 20 prior bars
                                 (``close``, ``volume``, ``vol_5d``,
                                 ``cum_return_20d``) -- parquet does not
                                 store nested arrays cleanly across
                                 versions, so JSON keeps the bytes stable.
- ``asset_symbol``              Default ``^GSPC``; schema supports per-asset
                                 expansion in future sweeps without
                                 changing the contract
- ``horizon``                   Trading-day horizon (1, 5, 10, 30)
- ``realized_return``           Raw close-to-close return over horizon
- ``abnormal_return``           ``realized_return - (alpha + beta * SPX)``;
                                 when asset is ``^GSPC`` itself this is
                                 just the raw return (alpha=0, beta=1)
- ``alpha``, ``beta``           Market-model parameters fit on the
                                 trailing 252-day window ending strictly
                                 before ``as_of_ts``
- ``direction_t1d``             Sign of the t+1d realized return:
                                 ``-1``, ``0``, ``+1``
- ``volatility_shift``          10d post-event realized vol minus 10d
                                 pre-event realized vol
- ``concurrent_macro_release``  Boolean -- True when a major macro release
                                 (CPI, NFP, ISM) falls within +/-2 trading
                                 days. Heuristic dates only; we flag, never
                                 drop.
- ``intra_meeting_stance_shift``    Signed shift in ``axis_stance`` between
                                     the same-date press_conference and
                                     statement rows. Numeric encoding
                                     ``hawkish=+1, dovish=-1, neutral=0``
                                     applies to categorical stance values;
                                     ``shift = press - statement``. NaN
                                     when either kind is missing for the
                                     event_date, or when an axis value is
                                     unencodable.
- ``intra_meeting_certainty_shift`` Signed shift in ``axis_certainty``
                                     between same-date press_conference
                                     and statement. ``axis_certainty`` is
                                     a regression score in ``[0, 1]`` per
                                     ``data/schema/labels.yaml``; numeric
                                     values are subtracted directly. If a
                                     label-encoded variant is ever
                                     introduced upstream, the encoder
                                     applies ``uncertain=-1, neutral=0,
                                     certain=+1`` as a fallback.
- ``intra_meeting_factor_shift``    Signed shift in ``axis_factor``
                                     (Gürkaynak-Sack-Swanson forward-
                                     guidance loading, regression in
                                     ``[-1, 1]``) between the same-date
                                     press_conference and statement.
                                     Numeric subtraction.

The three ``intra_meeting_*_shift`` columns are a property of the FOMC
date, not of the source. On the collapsed view the value comes from the
preferred statement and the preferred press_conference. On the full view
the same value is replicated to every (source x asset x horizon) row
sharing the event_date, so cross-source analyses see a consistent
intra-meeting tone shift regardless of which row they read.

Methodological constraints, enforced by assertions in the builder:

1. No look-ahead. The last bar in the 20-day prior window has a date
   strictly less than ``as_of_ts.date()``. The market-model regression
   window ends strictly before that same date.
2. No survivorship filter. Every FOMC event with text + event_date + a
   usable prior window produces a row, regardless of whether the post-
   event move was large.
3. Deterministic: builds on the same inputs are byte-identical. Rows are
   sorted by ``(event_date, event_kind, asset_symbol, horizon)`` before
   writing, and parquet is written with snappy compression (which has no
   timestamp metadata) via pandas.
4. Idempotent: re-running with the same training package leaves the output
   parquet content unchanged.

CLI:
    python -m app.data.event_dataset_builder \
        --training-package-id <id> --asset ^GSPC --output events.parquet
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import logging
import math
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

from app.config import DATA_DIR as DEFAULT_DATA_DIR
from app.data.macro_releases import (
    DEFAULT_MACRO_RELEASES_CSV,
    MacroReleaseCalendar,
    build_heuristic_calendar,
    load_macro_release_calendar,
)
from app.data.rates_event_features import (
    FORWARD_TARGET_COLUMNS,
    PRE_MEETING_LEVEL_COLUMNS,
    PreMeetingFeatures,
    compute_pre_meeting_features,
    forward_yield_change_bps,
)
from app.data.rates_panel import RatesPanelLookup, load_rates_panel_lookup
from app.data.statement_delta import (
    STATEMENT_DELTA_EMBEDDING_DIM,
    compute_delta_for_event,
    select_prior_statement_text,
)
from app.data.vote_tally import VoteTally, parse_vote_tally
from app.features.credibility import CredibilityVector
from app.services.credibility_loader import load_credibility_for_run

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_ASSET = "^GSPC"
DEFAULT_BENCHMARK = "^GSPC"
DEFAULT_HORIZONS = (1, 5, 10, 30)
PRIOR_WINDOW_DAYS = 20
# A2 (#207) realised-vol horizons beyond the default 5-day window. The
# 20d horizon captures the canonical "one month" of trading days and
# the 60d horizon catches the slow-moving regime component (Engle &
# Rangel, 2008; Christoffersen, 2012). Both fit the historical
# vol-regime literature; together with the existing 5d window they
# give the model access to short / medium / long horizons.
EXTRA_VOL_WINDOWS: tuple[int, ...] = (20, 60)
# A3 (#208) cross-asset yfinance symbols joined onto every prior bar
# by date. Symbol → FeatureVector field-name mapping; consumed by the
# pipeline orchestrator below and the per-bar attachment helper.
CROSS_ASSET_SYMBOLS: tuple[tuple[str, str], ...] = (
    ("^VIX", "vix_close"),
    ("DX-Y.NYB", "dxy_close"),
    ("^TNX", "tnx_close"),
    ("GC=F", "gold_close"),
    # Chunk 1 of Path B (#239 / Round 1 follow-up): VIX term structure
    # + short-end yield. ``^VIX3M`` is the 3-month CBOE volatility
    # index; the ratio against the 30-day ^VIX is the textbook
    # vol-regime warning signal (backwardation = stressed). ``^IRX``
    # is the 13-week T-bill yield (×100); combined with ^TNX it pins
    # the 10Y–3M yield-curve slope, which the vol-forecasting
    # literature consistently uses as a regime predictor.
    ("^VIX3M", "vix3m_close"),
    ("^IRX", "irx_close"),
    # #478 VIX term-structure + VRP block. ^VIX1M is the 1-month CBOE
    # volatility index (rolled constant-maturity); ^VIX6M is the 6-month
    # series. Coverage: ^VIX from 1990, ^VIX3M from 2007-12, ^VIX1M and
    # ^VIX6M from 2008-01 (CBOE constant-maturity series start).
    # Pre-coverage events emit ``None`` on the per-event VIX-term
    # columns and the loader's missing-flag pattern handles them.
    ("^VIX1M", "vix1m_close"),
    ("^VIX6M", "vix6m_close"),
)
# Per-asset forward realised-vol targets (#481). One column per symbol
# carries the strict-forward 10-trading-day realised vol of log returns
# computed on that symbol's own close series. The canonical
# ``forward_realized_vol_10d`` column stays as the ^GSPC alias so the
# existing classifier contract is unchanged; per-asset training arrives
# downstream of the symbol-conditioned head work. Missing data (pre-
# listing, holiday, or the symbol cache failed to fetch) keeps the
# column ``None`` so the regime classifier learns to skip rather than
# treating an absent quote as a zero.
#
# Relationship to ``app.models.config.SUPPORTED_SYMBOLS`` (the
# 5-symbol id table the symbol-conditioned regime head's nn.Embedding
# is keyed off): SUPPORTED_SYMBOLS is a strict subset of this tuple.
# The two cannot be merged because SUPPORTED_SYMBOLS is byte-stable by
# contract -- the embedding ids are pinned forever so a checkpoint
# trained against id k=2 always sees the same symbol at id 2 on
# rehydrate (see the docstring on ``SUPPORTED_SYMBOLS``). This tuple
# is data-only -- per-asset target columns on events.parquet, no id
# stability requirement -- so it can grow independently as upstream
# symbol coverage expands. The subset invariant
# (``SUPPORTED_SYMBOLS <= PER_ASSET_TARGET_SYMBOLS``) is enforced by
# ``tests/unit/test_per_asset_vol_columns.py`` so a future symbol
# added to SUPPORTED_SYMBOLS without a matching per-asset target
# column fails at test time.
PER_ASSET_TARGET_SYMBOLS: tuple[str, ...] = (
    "^GSPC",
    "^NDX",
    "^DJI",
    "DX-Y.NYB",
    "^VIX",
    "EURUSD=X",
    "USDJPY=X",
    "GBPUSD=X",
)
MARKET_MODEL_WINDOW_DAYS = 252
VOL_WINDOW_DAYS = 10
ROLLING_VOL_DAYS = 5
CONCURRENT_MACRO_TRADING_DAY_RADIUS = 2

# Two announcement-time placeholders. Documented in module docstring; if a
# subsequent issue lands a real per-event timestamp source (#146 OIS surprise)
# replace these constants and re-build.
FOMC_AS_OF_TIME = "T19:00:00Z"
SPEECH_AS_OF_TIME = "T14:00:00Z"
# BLS macroeconomic releases (CPI, Employment Situation) drop at 8:30 ET
# (= 12:30 UTC under EST, 13:30 UTC under EDT). The placeholder rounds to
# 13:00 UTC to stay clear of DST hand-waving; intraday alignment will
# replace this if the announcement-window target ever lands.
MACRO_AS_OF_TIME = "T13:00:00Z"

def per_asset_target_slug(symbol: str) -> str:
    """Normalise a yfinance symbol to the suffix used for per-asset vol target columns.

    The rule: lowercase the symbol, then strip the yfinance prefix/suffix
    tokens ``^`` (index), ``=x`` (FX), ``.nyb`` (NY ICE futures venue), and
    finally drop any remaining ``-`` separators.

    Examples:

    - ``^GSPC`` -> ``gspc``
    - ``DX-Y.NYB`` -> ``dxy``
    - ``EURUSD=X`` -> ``eurusd``
    - ``^VIX`` -> ``vix``
    """

    slug = symbol.lower()
    for token in ("^", "=x", ".nyb"):
        slug = slug.replace(token, "")
    slug = slug.replace("-", "")
    return slug


def per_asset_target_column(symbol: str) -> str:
    """Return the events.parquet column name for ``symbol``'s 10d realised vol.

    See ``per_asset_target_slug`` for the normalisation rule.
    """

    return f"forward_realized_vol_10d_{per_asset_target_slug(symbol)}"


# Map registry document_type values (raw, mixed-case) onto the canonical
# event_kind taxonomy. Anything not listed is dropped silently and counted
# in the build summary.
_EVENT_KIND_MAP: dict[str, str] = {
    "statement": "statement",
    "Statement": "statement",
    "minutes": "minutes",
    "Minutes": "minutes",
    "meeting_transcript": "minutes",  # FOMC transcripts share the minutes window
    "press_conference": "press_conference",
    "congressional_testimony": "testimony",
    "chair_speech": "speech",
    "governor_speech": "speech",
    # Macro-release augmentation (variant A of the macro-event experiment).
    # The supervised target is forward-realised vol regime computed from
    # market data; the text channel emits the missing flag because the
    # placeholder text is not loaded into the embedding cache. The
    # canonical kind is ``macro_release`` so downstream consumers can
    # branch on it without parsing the source string.
    "macro_release_cpi": "macro_release",
    "macro_release_nfp": "macro_release",
}

# Speech-like kinds use the speech time placeholder; everything else uses
# the FOMC 2pm ET placeholder.
_SPEECH_KINDS = frozenset({"speech", "testimony"})

# Categorical encodings used by ``_axis_to_numeric`` when computing the
# ``intra_meeting_*_shift`` columns. ``axis_stance`` is the only axis
# that arrives as a categorical label in the bundled registry; the other
# two axes are regression-typed per ``data/schema/labels.yaml`` and pass
# their numeric values through unchanged. The ``certainty`` map is a
# best-effort fallback for any future upstream that emits label strings
# instead of a regression score.
_STANCE_ENCODING: dict[str, float] = {
    "hawkish": 1.0,
    "dovish": -1.0,
    "neutral": 0.0,
}
_CERTAINTY_ENCODING: dict[str, float] = {
    # ``axis_certainty`` is regression-typed in ``data/schema/labels.yaml``
    # (range [0.0, 1.0]), so there is no canonical categorical label
    # vocabulary to encode. The dict is kept for symmetry with
    # ``_STANCE_ENCODING`` and ``_FACTOR_ENCODING``; numeric passthrough
    # via ``_axis_to_numeric`` is the only supported path.
}
_FACTOR_ENCODING: dict[str, float] = {
    # ``axis_factor`` is regression-only in the schema, so the fallback
    # label map is intentionally empty; numeric passthrough is the only
    # supported path. The dict is kept for symmetry with stance/certainty.
}

# Map ``axis_*`` column name to its categorical encoding. Numeric inputs
# are accepted on every axis via the ``_axis_to_numeric`` fallback.
_INTRA_MEETING_AXIS_ENCODING: dict[str, dict[str, float]] = {
    "axis_stance": _STANCE_ENCODING,
    "axis_certainty": _CERTAINTY_ENCODING,
    "axis_factor": _FACTOR_ENCODING,
}

# Bounded numeric encodings for the gtfintechlab categorical axes when
# they bubble up through ``normalize_labels._Record.axes`` as strings.
# ``EventRowSchema`` requires ``axis_time`` ∈ [-10, 10] and
# ``axis_certainty`` ∈ [0, 1]; passing the raw strings through fails the
# schema gate (every TP build that includes gtfintechlab cross-bank
# rows). ``axis_*_label`` columns continue to carry the categorical
# representation for consumers that want it; the numeric encoding here
# only fills the bounded axis columns.
_AXIS_TIME_STR_TO_NUMERIC: dict[str, float] = {
    "forward looking": 1.0,
    "not forward looking": -1.0,
}
_AXIS_CERTAINTY_STR_TO_NUMERIC: dict[str, float] = {
    "certain": 1.0,
    "uncertain": 0.0,
}


def _encode_axis_time(value: Any) -> float | None:
    """Coerce ``axes.time`` to a schema-bounded numeric or ``None``.

    Accepts a numeric passthrough (already in [-10, 10] range), the
    two known gtfintechlab string categories, or returns ``None`` for
    anything else so the nullable ``axis_time`` column carries honest
    absence instead of a schema-fail value. ``bool`` is rejected
    explicitly — ``isinstance(True, int)`` is ``True`` in Python, so a
    boolean would silently coerce to 1.0 / 0.0 and pass the bounded
    range check, masking an upstream loader that mis-typed the field.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        s = value.strip().lower()
        return _AXIS_TIME_STR_TO_NUMERIC.get(s)
    return None


def _encode_axis_certainty(value: Any) -> float | None:
    """Coerce ``axes.certainty`` to a schema-bounded numeric or ``None``.

    Accepts a regression-typed numeric in [0, 1] (passthrough), the
    two known gtfintechlab string categories (``certain`` -> 1.0 /
    ``uncertain`` -> 0.0), or ``None``. Unknown shapes return ``None``.
    ``bool`` is rejected explicitly — see :func:`_encode_axis_time`.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        s = value.strip().lower()
        if s in _AXIS_CERTAINTY_STR_TO_NUMERIC:
            return _AXIS_CERTAINTY_STR_TO_NUMERIC[s]
        try:
            return float(s)
        except ValueError:
            return None
    return None


def _encode_axis_factor(value: Any) -> float | None:
    """Coerce ``axes.factor`` to a schema-bounded numeric or ``None``.

    ``EventRowSchema`` bounds ``axis_factor`` to [-1, 1]. Drop values
    that fall outside that range so the schema gate stays green; the
    bp-scale factor decompositions (GSS / Swanson) are preserved on
    ``multi_axis_extras`` and read by consumers that want them.
    Out-of-range / unparseable inputs land at ``None``. ``bool`` is
    rejected explicitly (subclasses ``int``) and negative-zero
    normalises to ``0.0`` so the parquet round-trip stays byte-identical.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(f):
        return None
    if -1.0 <= f <= 1.0:
        # Normalise -0.0 → 0.0; pyarrow rewrites -0.0 to 0.0 on the
        # parquet round-trip, so emitting either is unsafe for the
        # byte-determinism contract documented at the top of the module.
        return f if f != 0.0 else 0.0
    return None

# When multiple registry sources cover the same (event_date, event_kind),
# pick the row with the highest preference rank. This avoids near-duplicate
# event rows while keeping the choice deterministic. Higher-quality / more
# complete document sources are preferred over sentence-level shards.
_SOURCE_PREFERENCE: tuple[str, ...] = (
    "scraped_fed",
    "vtasca_fomc_archive",
    "op_fed",
    "gtfintechlab_federal_reserve_system",
    "hf_fomc_communication",
    "kaggle_fed_statements_minutes",
    "gss_factor",
    "fred_macro_releases",
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class _RegistryRow:
    source: str
    source_record_id: str
    document_type: str
    event_date: str
    text: str
    mapped_label: str | None
    multi_axis_extras: dict[str, Any]
    axes: dict[str, Any]


@dataclass
class _EventDoc:
    """One aggregated document for a single (source, event_date, event_kind)."""

    source: str
    event_date: str
    event_kind: str
    text: str
    record_ids: list[str]
    multi_axis: dict[str, str | None]
    # Populated from registry ``multi_axis_extras`` when available. Only the
    # gtfintechlab cross-bank corpora carry these labels today; every other
    # source leaves them ``None`` and downstream consumers encode them as
    # 0.0 in the rich-feature slot.
    time_label: str | None = None       # "forward looking" / "not forward looking"
    certain_label: str | None = None    # "certain" / "uncertain"
    # Merged ``multi_axis_extras`` payload from every record in the bucket.
    # Carries the GSS / Swanson bp-scale factor decompositions
    # (``gss_target_factor`` / ``gss_path_factor`` /
    # ``swanson_target_factor`` / ``swanson_forward_guidance_factor`` /
    # ``swanson_lsap_factor``) plus the gtfintechlab dataset-revision tags.
    # The bounded numeric ``axis_*`` columns can only hold values in the
    # schema range, so the bp-scale factors are stranded unless they ride
    # this dict onto events.parquet.
    multi_axis_extras: dict[str, Any] = field(default_factory=dict)

    @property
    def source_record_id(self) -> str:
        """Stable identifier for one (source, event_date, event_kind) shard.

        Uses the concatenation of all aggregated ``source_record_id`` values
        joined by ``|``. Sorted at aggregation time so the value is
        deterministic.
        """

        return "|".join(self.record_ids)


@dataclass
class _PriorBar:
    date: _dt.date
    close: float
    volume: float
    vol_5d: float
    cum_return_20d: float
    # A2 (#207): additional realised-vol horizons. Computed as the
    # sample standard deviation of log returns over the trailing N
    # trading days anchored at this bar (right-inclusive). Default
    # 0.0 keeps any legacy bar-builder path that doesn't compute
    # these working without crashing the schema.
    vol_20d: float = 0.0
    vol_60d: float = 0.0
    # A3 (#208): cross-asset close levels at this bar's date. Joined
    # in from independent yfinance caches at the top of the pipeline;
    # a series with no observation on the bar's date emits 0.0 rather
    # than blocking the whole bar (e.g. VIX pre-1990, gold front-month
    # contract roll days).
    vix_close: float = 0.0
    dxy_close: float = 0.0
    tnx_close: float = 0.0
    gold_close: float = 0.0
    # Path B Chunk 1: VIX term structure + short-end yield. Raw closes
    # for the new series; the per-bar emission below computes the two
    # derived slopes (``vix_term_slope``, ``yield_curve_slope_10y_3m``)
    # so the model sees both the levels and the stationary spreads.
    # Series with no observation default to 0.0 — same back-compat
    # contract as the original four cross-asset axes.
    vix3m_close: float = 0.0
    irx_close: float = 0.0
    vix_term_slope: float = 0.0
    yield_curve_slope_10y_3m: float = 0.0


@dataclass
class _BuildSummary:
    rows_written: int = 0
    events_emitted: int = 0
    dropped_no_text: int = 0
    dropped_unmapped_kind: int = 0
    dropped_no_prior_window: int = 0
    dropped_no_targets: int = 0
    concurrent_macro_release_rows: int = 0
    per_source_rows: dict[str, int] = field(default_factory=dict)
    per_kind_rows: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Registry loading + event aggregation
# ---------------------------------------------------------------------------


def _load_registry_rows(package_dir: Path) -> list[_RegistryRow]:
    path = package_dir / "registry_normalized.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing registry: {path}")
    rows: list[_RegistryRow] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        text = str(payload.get("text", "") or "").strip()
        event_date = str(payload.get("event_date", "") or "").strip()
        if not text or not event_date:
            continue
        mapped = payload.get("mapped_label")
        rows.append(
            _RegistryRow(
                source=str(payload.get("source", "") or ""),
                source_record_id=str(payload.get("source_record_id", "") or ""),
                document_type=str(payload.get("document_type", "") or ""),
                event_date=event_date,
                text=text,
                mapped_label=str(mapped).lower() if mapped else None,
                multi_axis_extras=dict(payload.get("multi_axis_extras") or {}),
                axes=dict(payload.get("axes") or {}),
            )
        )
    return rows


def _aggregate_events(rows: Iterable[_RegistryRow]) -> list[_EventDoc]:
    """Aggregate registry rows into one document per ``(source, date, kind)``.

    Sentence-level shards (TDW, gtfintechlab, GSS) get concatenated in
    ``source_record_id`` order; full-document sources stay as-is.
    """

    groups: dict[tuple[str, str, str], list[_RegistryRow]] = defaultdict(list)
    for row in rows:
        kind = _EVENT_KIND_MAP.get(row.document_type)
        if kind is None:
            continue
        groups[(row.source, row.event_date, kind)].append(row)

    docs: list[_EventDoc] = []
    for (source, event_date, kind), bucket in groups.items():
        bucket_sorted = sorted(bucket, key=lambda r: r.source_record_id)
        text = "\n".join(r.text for r in bucket_sorted if r.text)
        if not text.strip():
            continue
        multi_axis: dict[str, str | None] = {
            "stance": None,
            "time": None,
            "certainty": None,
            "factor": None,
        }
        # gtfintechlab cross-bank corpora ship two extra categorical labels
        # ("time_label" forward/not-forward, "certain_label" certain/uncertain)
        # on ``multi_axis_extras``. Lift them into the event doc so the
        # rich-feature loader can use them as Option-A indicators on the
        # multi-axis slot. Initialised per bucket iteration to prevent any
        # leakage between adjacent (source, date, kind) groups.
        time_label: str | None = None
        certain_label: str | None = None
        # Merged extras across the bucket. First-wins per key so the
        # earliest-sorted record's value sticks; tie-break is deterministic
        # via ``bucket_sorted`` (source_record_id ordering). Each ingester
        # parks per-row scalars on ``multi_axis_extras`` (gss_target_factor,
        # swanson_target_factor, etc.) and the bucket aggregator merges
        # them onto the event row so downstream trainers do not have to
        # rejoin to ``registry_normalized.parquet`` to read GSS / Swanson
        # decompositions or the gtfintechlab dataset-revision tags.
        merged_extras: dict[str, Any] = {}
        # Use the first row whose mapped_label or axes carry a value as the
        # document-level label. Deterministic because bucket is sorted.
        # ``val is not None`` (not truthy) preserves legitimate zeros for
        # the regression-typed axes (``certainty``, ``factor`` per
        # ``data/schema/labels.yaml``) -- otherwise a 0.0 factor would
        # silently be dropped to None and downstream consumers (the
        # ``intra_meeting_*_shift`` columns in particular) would lose
        # the signal.
        for r in bucket_sorted:
            if multi_axis["stance"] is None and r.mapped_label:
                multi_axis["stance"] = r.mapped_label
            for axis_name in ("time", "certainty", "factor"):
                if multi_axis[axis_name] is None:
                    val = r.axes.get(axis_name)
                    if val is not None:
                        multi_axis[axis_name] = str(val)
            if time_label is None:
                candidate = r.multi_axis_extras.get("gtfintechlab_time_label")
                if isinstance(candidate, str) and candidate.strip():
                    time_label = candidate.strip().lower()
            if certain_label is None:
                candidate = r.multi_axis_extras.get("gtfintechlab_certain_label")
                if isinstance(candidate, str) and candidate.strip():
                    certain_label = candidate.strip().lower()
            for k, v in (r.multi_axis_extras or {}).items():
                if v is None:
                    continue
                if k not in merged_extras:
                    merged_extras[k] = v
        docs.append(
            _EventDoc(
                source=source,
                event_date=event_date,
                event_kind=kind,
                text=text,
                record_ids=[r.source_record_id for r in bucket_sorted],
                multi_axis=multi_axis,
                multi_axis_extras=merged_extras,
                time_label=time_label,
                certain_label=certain_label,
            )
        )
    return docs


_STANCE_SCORE: dict[str, float] = {"hawkish": 1.0, "dovish": -1.0, "neutral": 0.0}


def _derive_stance_by_date(
    registry_rows: Sequence[_RegistryRow],
) -> tuple[tuple[str, float], ...]:
    """Aggregate per-date stance scores from registry ``mapped_label``.

    Encodes hawkish=+1, dovish=-1, neutral=0 and averages the score per
    ``event_date`` so the credibility loader has a stance history to
    correlate against the FRED-DFF realized path
    (``realized_vs_stated_gap``) and to count sign flips for the
    ``months_since_reversal`` axis. The auto-derivation is what previously
    forced both axes to 0.0 — the CLI never exposed ``--stance-by-date``,
    so every event_dataset_builder run before 2026-05-17 silently passed
    an empty tuple.

    Returns a chronologically-sorted tuple of (date_iso, mean_score)
    pairs. Empty for packages where no row carries a mapped stance label.
    """

    by_date: dict[str, list[float]] = defaultdict(list)
    for row in registry_rows:
        score = _STANCE_SCORE.get((row.mapped_label or "").lower())
        if score is None or not row.event_date:
            continue
        by_date[row.event_date].append(float(score))
    return tuple(
        (date, sum(scores) / len(scores))
        for date, scores in sorted(by_date.items())
    )


def _choose_preferred(docs: list[_EventDoc]) -> list[_EventDoc]:
    """Collapse multi-source duplicates to one chosen ``_EventDoc`` per event.

    For the same ``(event_date, event_kind)``, keep the source ranked highest
    in ``_SOURCE_PREFERENCE``; ties broken by source name (sorted).
    """

    rank = {src: i for i, src in enumerate(_SOURCE_PREFERENCE)}

    def _key(doc: _EventDoc) -> tuple[int, str]:
        return (rank.get(doc.source, len(_SOURCE_PREFERENCE)), doc.source)

    grouped: dict[tuple[str, str], list[_EventDoc]] = defaultdict(list)
    for doc in docs:
        grouped[(doc.event_date, doc.event_kind)].append(doc)

    chosen: list[_EventDoc] = []
    for bucket in grouped.values():
        bucket.sort(key=_key)
        chosen.append(bucket[0])
    chosen.sort(key=lambda d: (d.event_date, d.event_kind, d.source))
    return chosen


# ---------------------------------------------------------------------------
# Market series
# ---------------------------------------------------------------------------


def _date(value: str | _dt.date) -> _dt.date:
    if isinstance(value, _dt.date):
        return value
    return _dt.date.fromisoformat(str(value)[:10])


@dataclass
class _CloseSeries:
    """Lightweight close+volume series indexed by trading-day date."""

    dates: list[_dt.date]
    close: list[float]
    volume: list[float]

    def __len__(self) -> int:
        return len(self.dates)

    def index_strictly_before(self, target: _dt.date) -> int:
        """Largest index with ``dates[i] < target``; -1 if none."""
        lo, hi = 0, len(self.dates)
        while lo < hi:
            mid = (lo + hi) // 2
            if self.dates[mid] < target:
                lo = mid + 1
            else:
                hi = mid
        return lo - 1

    def index_on_or_after(self, target: _dt.date) -> int:
        """Smallest index with ``dates[i] >= target``; ``len`` if none."""
        lo, hi = 0, len(self.dates)
        while lo < hi:
            mid = (lo + hi) // 2
            if self.dates[mid] < target:
                lo = mid + 1
            else:
                hi = mid
        return lo


def _fetch_close_series(
    symbol: str,
    *,
    start: _dt.date,
    end: _dt.date,
    cache_dir: Path | None = None,
    force_refresh: bool = False,
) -> _CloseSeries:
    """Fetch (or load cached) daily close+volume for ``symbol``.

    The cache layout mirrors :mod:`app.services.fred_client`: one parquet per
    symbol in ``cache_dir`` with a ``SOURCES.lock`` recording the SHA. Tests
    write a parquet directly and never hit yfinance; the smoke run hits live
    yfinance once and reuses the cache forever after.
    """

    cache_path = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        safe = symbol.replace("^", "").replace("=", "_").replace("/", "_").replace(":", "_")
        cache_path = cache_dir / f"{safe}.parquet"
        if cache_path.exists() and not force_refresh:
            frame = pd.read_parquet(cache_path)
            return _frame_to_series(frame)

    import yfinance as yf

    ticker = yf.Ticker(symbol)
    frame = ticker.history(
        start=start.isoformat(),
        end=(end + _dt.timedelta(days=1)).isoformat(),
        auto_adjust=True,
    )
    if frame.empty:
        raise RuntimeError(f"yfinance returned no data for {symbol} in [{start}, {end}]")
    dates = [idx.date() for idx in frame.index]
    close = [float(v) for v in frame["Close"].to_numpy()]
    volume_series = frame["Volume"] if "Volume" in frame.columns else None
    if volume_series is not None:
        volume = [float(v) for v in volume_series.to_numpy()]
    else:
        volume = [0.0] * len(close)
    series = _CloseSeries(dates=dates, close=close, volume=volume)
    if cache_path is not None:
        out = pd.DataFrame(
            {
                "symbol": symbol,
                "date": [d.isoformat() for d in series.dates],
                "close": series.close,
                "volume": series.volume,
            }
        )
        out.to_parquet(cache_path, index=False)
        _update_sources_lock(cache_path, symbol)
    return series


def _frame_to_series(frame: pd.DataFrame) -> _CloseSeries:
    if "date" not in frame.columns or "close" not in frame.columns:
        raise RuntimeError(
            "Cached market parquet must contain 'date' and 'close' columns; "
            f"got {list(frame.columns)}"
        )
    parsed = sorted(
        zip(
            [_date(d) for d in frame["date"].tolist()],
            [float(c) for c in frame["close"].tolist()],
            [float(v) for v in (frame["volume"].tolist() if "volume" in frame.columns else [0.0] * len(frame))],
        ),
        key=lambda t: t[0],
    )
    return _CloseSeries(
        dates=[t[0] for t in parsed],
        close=[t[1] for t in parsed],
        volume=[t[2] for t in parsed],
    )


def _update_sources_lock(parquet_path: Path, symbol: str) -> None:
    lock_path = parquet_path.parent / "SOURCES.lock"
    digest = hashlib.sha256(parquet_path.read_bytes()).hexdigest()
    if lock_path.exists():
        try:
            payload = json.loads(lock_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
    else:
        payload = {}
    entries = payload.setdefault("entries", {})
    entries[symbol] = {
        "parquet_path": parquet_path.name,
        "sha256": digest,
    }
    payload["format_version"] = 1
    lock_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


# ---------------------------------------------------------------------------
# Market computations: prior bars, market-model, targets, vol shift
# ---------------------------------------------------------------------------


def _log_returns(closes: Sequence[float]) -> list[float]:
    import math

    out: list[float] = []
    for i in range(1, len(closes)):
        prev = closes[i - 1]
        cur = closes[i]
        if prev <= 0 or cur <= 0:
            out.append(0.0)
        else:
            out.append(math.log(cur / prev))
    return out


def _vix_term_structure_features(
    *,
    as_of: _dt.date,
    series_by_symbol: dict[str, _CloseSeries],
    asset_series: _CloseSeries,
) -> dict[str, float | None]:
    """Compose the #478 VIX term-structure + VRP block for one event.

    Reads the four constant-maturity VIX series (``^VIX`` 1990+,
    ``^VIX1M`` 2008+, ``^VIX3M`` 2007-12+, ``^VIX6M`` 2008+) and the
    asset close series strictly before ``as_of``. Emits six nullable
    scalars:

    - ``vix_t_minus_1`` — 30-day ATM implied vol at T-1.
    - ``vix1m_t_minus_1`` — 1-month constant-maturity implied vol at T-1.
    - ``vix3m_t_minus_1`` — 3-month constant-maturity implied vol at T-1.
    - ``vix6m_t_minus_1`` — 6-month constant-maturity implied vol at T-1.
    - ``vix_3m_over_1m_slope`` — ``vix3m / vix1m`` ratio at T-1 (>1.0
      contango, <1.0 backwardation; the textbook stressed-market signal).
    - ``vrp_t_minus_1`` — volatility risk premium at T-1: the implied
      30-day vol on a daily scale (``vix/100/sqrt(252)``) minus the
      trailing 30-trading-day realised vol of asset log returns. Both
      legs read strictly before ``as_of``.

    Any per-scalar input gap (pre-coverage event, holiday with no quote,
    asset history too short for the realised baseline) collapses that
    scalar to ``None``. Downstream the loader's missing-flag pattern
    fills zeros and flips the paired ``vix_features_missing`` flag.
    """

    out: dict[str, float | None] = {
        "vix_t_minus_1": None,
        "vix1m_t_minus_1": None,
        "vix3m_t_minus_1": None,
        "vix6m_t_minus_1": None,
        "vix_3m_over_1m_slope": None,
        "vrp_t_minus_1": None,
    }
    field_to_symbol = {field: sym for sym, field in CROSS_ASSET_SYMBOLS}
    column_for_field = {
        "vix_close": "vix_t_minus_1",
        "vix1m_close": "vix1m_t_minus_1",
        "vix3m_close": "vix3m_t_minus_1",
        "vix6m_close": "vix6m_t_minus_1",
    }
    for field_name, column in column_for_field.items():
        symbol = field_to_symbol.get(field_name)
        if symbol is None:
            continue
        series = series_by_symbol.get(symbol)
        if series is None or not series.dates:
            continue
        idx = series.index_strictly_before(as_of)
        if idx < 0:
            continue
        close = series.close[idx]
        if close > 0:
            out[column] = float(close)
    vix1m = out["vix1m_t_minus_1"]
    vix3m = out["vix3m_t_minus_1"]
    if vix1m is not None and vix3m is not None and vix1m > 0:
        out["vix_3m_over_1m_slope"] = float(vix3m) / float(vix1m)
    vix = out["vix_t_minus_1"]
    if vix is not None:
        implied_daily = float(vix) / 100.0 / (252.0 ** 0.5)
        realized = _rolling_realized_vol_t_minus_1(asset_series, as_of, window=30)
        if realized is not None:
            out["vrp_t_minus_1"] = implied_daily - realized
    return out


def _rolling_realized_vol_t_minus_1(
    series: _CloseSeries, as_of: _dt.date, *, window: int
) -> float | None:
    """Sample std of log returns over the trailing ``window`` bars at T-1.

    Strict-prior: reads ``series.close`` indices ``[base - window .. base]``
    where ``base = index_strictly_before(as_of)``; the resulting ``window``
    log returns are all dated strictly before ``as_of``.
    """

    base = series.index_strictly_before(as_of)
    if base < window:
        return None
    closes = series.close[base - window : base + 1]
    rets = _log_returns(closes)
    if len(rets) < 2:
        return None
    n = len(rets)
    mean = sum(rets) / n
    variance = sum((v - mean) ** 2 for v in rets) / (n - 1)
    return float(variance ** 0.5)


def _build_cross_asset_lookup(
    series_by_symbol: dict[str, _CloseSeries],
) -> dict[_dt.date, dict[str, float]]:
    """Build ``{date: {field_name: close}}`` for O(1) per-bar joins.

    Sparse: a date present in only one series's calendar still produces
    an entry, but only that series's field is populated. The
    ``_build_prior_window`` attachment fills missing fields with 0.0
    at read time.
    """

    out: dict[_dt.date, dict[str, float]] = {}
    field_by_symbol = dict(CROSS_ASSET_SYMBOLS)
    for symbol, series in series_by_symbol.items():
        field_name = field_by_symbol.get(symbol)
        if field_name is None:
            continue
        for d, c in zip(series.dates, series.close):
            row = out.setdefault(d, {})
            row[field_name] = float(c)
    return out


def _build_prior_window(
    series: _CloseSeries,
    as_of: _dt.date,
    *,
    window_days: int = PRIOR_WINDOW_DAYS,
    vol_window: int = ROLLING_VOL_DAYS,
    cross_asset_lookup: dict[_dt.date, dict[str, float]] | None = None,
) -> list[_PriorBar] | None:
    """Return ``window_days`` prior bars ending strictly before ``as_of``.

    Returns None when the series lacks enough history. Each bar carries
    rolling 5-day log-return std and cumulative simple return over the
    20-day window (anchored at the first bar in the window).
    """

    last_idx = series.index_strictly_before(as_of)
    if last_idx < 0:
        return None
    start_idx = last_idx - window_days + 1
    # Need vol_window+1 prior closes to compute rolling 5d std for the first
    # bar in the window, and window_days closes to compute cum_return.
    # cum_return baseline is the bar immediately preceding the window.
    if start_idx - 1 < 0:
        return None
    if start_idx - vol_window < 0:
        return None
    # A2 (#207) backstop: we also compute trailing 20d / 60d vol per
    # bar. Each requires N+1 prior closes from the bar itself; the
    # earliest bar (start_idx) must therefore have at least
    # ``max(EXTRA_VOL_WINDOWS)`` log returns behind it. When the
    # series is too short we emit 0.0 for the affected horizon rather
    # than dropping the event entirely.

    log_rets_all = _log_returns(series.close[: last_idx + 1])
    # log_rets_all[i] corresponds to close[i+1]. Rolling std at close-index j
    # uses log_rets_all[j - vol_window .. j-1].
    bars: list[_PriorBar] = []
    base_close = series.close[start_idx - 1]
    for offset in range(window_days):
        i = start_idx + offset
        # rolling 5d log-return std anchored on bar i
        if i - vol_window < 0:
            return None
        chunk = log_rets_all[i - vol_window : i]
        if len(chunk) < vol_window:
            return None
        mean = sum(chunk) / vol_window
        var = sum((x - mean) ** 2 for x in chunk) / (vol_window - 1)
        vol = var**0.5
        # A2 (#207) extra realised-vol horizons. We re-use the same
        # log-return slice machinery; insufficient history -> 0.0 for
        # that horizon only.
        extra_vols: dict[int, float] = {}
        for w in EXTRA_VOL_WINDOWS:
            if i - w < 0 or len(log_rets_all[i - w : i]) < w:
                extra_vols[w] = 0.0
                continue
            w_chunk = log_rets_all[i - w : i]
            w_mean = sum(w_chunk) / w
            w_var = sum((x - w_mean) ** 2 for x in w_chunk) / (w - 1)
            extra_vols[w] = w_var**0.5
        cum_return = (series.close[i] - base_close) / base_close if base_close > 0 else 0.0
        # A3 (#208) cross-asset join. Lookup is keyed on the bar's
        # date; absent fields default to 0.0 because non-trading days
        # for one asset (e.g. VIX before 1990, gold futures contract
        # roll gaps) should not block the bar.
        bar_date = series.dates[i]
        cross_asset_row = (
            cross_asset_lookup.get(bar_date, {})
            if cross_asset_lookup is not None
            else {}
        )
        vix_close_val = float(cross_asset_row.get("vix_close", 0.0))
        tnx_close_val = float(cross_asset_row.get("tnx_close", 0.0))
        vix3m_close_val = float(cross_asset_row.get("vix3m_close", 0.0))
        irx_close_val = float(cross_asset_row.get("irx_close", 0.0))
        # Derived slopes. Both go to 0.0 when either input is missing
        # (the join-day default) so the rich-feature block stays
        # numerically clean on pre-2002 bars where VIX3M does not exist
        # and the loader's per-fold scaler does not see NaN.
        #
        # Unit contract: ^TNX and ^IRX both come from Yahoo Finance and
        # both are quoted as percent yields on the same scale (e.g.
        # TNX ~= 4.20 means 4.20% on the 10Y, IRX ~= 5.10 means 5.10%
        # on the 13-week T-bill). The subtraction is therefore the
        # 10Y-3M slope in percentage points; an inversion shows up as
        # a negative number. VIX3M / VIX is a unitless ratio so the
        # log-slope is also unitless.
        vix_term_slope_val = (
            math.log(vix3m_close_val / vix_close_val)
            if vix3m_close_val > 0.0 and vix_close_val > 0.0
            else 0.0
        )
        yield_curve_slope_val = (
            tnx_close_val - irx_close_val
            if tnx_close_val > 0.0 and irx_close_val > 0.0
            else 0.0
        )
        bars.append(
            _PriorBar(
                date=bar_date,
                close=series.close[i],
                volume=series.volume[i],
                vol_5d=vol,
                cum_return_20d=cum_return,
                vol_20d=extra_vols.get(20, 0.0),
                vol_60d=extra_vols.get(60, 0.0),
                vix_close=vix_close_val,
                dxy_close=float(cross_asset_row.get("dxy_close", 0.0)),
                tnx_close=tnx_close_val,
                gold_close=float(cross_asset_row.get("gold_close", 0.0)),
                vix3m_close=vix3m_close_val,
                irx_close=irx_close_val,
                vix_term_slope=vix_term_slope_val,
                yield_curve_slope_10y_3m=yield_curve_slope_val,
            )
        )
    # Enforce no look-ahead: last prior bar must be < as_of
    assert bars[-1].date < as_of, (
        f"prior-window contract violated: last bar {bars[-1].date} not < as_of {as_of}"
    )
    return bars


def _fit_market_model(
    asset_returns: Sequence[float],
    bench_returns: Sequence[float],
) -> tuple[float, float]:
    """OLS regression ``asset = alpha + beta * bench``.

    Returns ``(alpha, beta)``. Falls back to ``(0.0, 1.0)`` when input is
    degenerate (empty, single point, or zero benchmark variance).
    """

    n = min(len(asset_returns), len(bench_returns))
    if n < 2:
        return (0.0, 1.0)
    a = list(asset_returns[-n:])
    b = list(bench_returns[-n:])
    mean_a = sum(a) / n
    mean_b = sum(b) / n
    cov = sum((ai - mean_a) * (bi - mean_b) for ai, bi in zip(a, b)) / n
    var_b = sum((bi - mean_b) ** 2 for bi in b) / n
    if var_b <= 1e-18:
        return (0.0, 1.0)
    beta = cov / var_b
    alpha = mean_a - beta * mean_b
    return (alpha, beta)


def _market_model_for_event(
    asset_series: _CloseSeries,
    bench_series: _CloseSeries,
    as_of: _dt.date,
    *,
    window_days: int = MARKET_MODEL_WINDOW_DAYS,
) -> tuple[float, float] | None:
    """Fit alpha, beta on the trailing ``window_days`` daily returns ending
    strictly before ``as_of``. Returns None when either series lacks
    enough history."""

    a_last = asset_series.index_strictly_before(as_of)
    b_last = bench_series.index_strictly_before(as_of)
    if a_last < window_days or b_last < window_days:
        return None

    # Daily returns aligned on date. Build a join on dates within the window.
    a_dates = asset_series.dates[a_last - window_days : a_last + 1]
    a_closes = asset_series.close[a_last - window_days : a_last + 1]
    b_date_to_close = dict(
        zip(
            bench_series.dates[: b_last + 1],
            bench_series.close[: b_last + 1],
        )
    )
    paired_a: list[float] = []
    paired_b: list[float] = []
    for i in range(1, len(a_dates)):
        d_prev, d_cur = a_dates[i - 1], a_dates[i]
        if d_prev not in b_date_to_close or d_cur not in b_date_to_close:
            continue
        a_ret = (a_closes[i] - a_closes[i - 1]) / a_closes[i - 1] if a_closes[i - 1] > 0 else 0.0
        bp, bc = b_date_to_close[d_prev], b_date_to_close[d_cur]
        b_ret = (bc - bp) / bp if bp > 0 else 0.0
        paired_a.append(a_ret)
        paired_b.append(b_ret)
    if len(paired_a) < window_days // 4:
        return None
    return _fit_market_model(paired_a, paired_b)


def _realized_returns(
    series: _CloseSeries,
    as_of: _dt.date,
    horizons: Sequence[int],
) -> dict[int, tuple[float, _dt.date | None]] | None:
    """Compute ``(close_{t-1} -> close_{t-1+h})`` returns for each horizon.

    The base is the last close strictly before ``as_of``. The target close
    is the ``h``-th trading day on-or-after ``as_of`` (i.e. event-day close
    counted as h=1 when event_date is itself a trading day; otherwise the
    next trading day).
    """

    base_idx = series.index_strictly_before(as_of)
    if base_idx < 0:
        return None
    base = series.close[base_idx]
    if base <= 0:
        return None
    on_or_after = series.index_on_or_after(as_of)
    if on_or_after >= len(series):
        return {}
    # Trading-day offset 1 == first trading day with date >= as_of.date()
    out: dict[int, tuple[float, _dt.date | None]] = {}
    for h in horizons:
        idx = on_or_after + (h - 1)
        if idx < 0 or idx >= len(series):
            continue
        close_h = series.close[idx]
        if close_h <= 0:
            continue
        out[h] = ((close_h - base) / base, series.dates[idx])
    return out


def _volatility_shift(
    series: _CloseSeries,
    as_of: _dt.date,
    *,
    window: int = VOL_WINDOW_DAYS,
) -> float | None:
    """Post-event vol minus pre-event vol around an as-of bar.

    Both windows are computed from log returns with sample std (ddof=1).
    The pre-window covers returns on days ``t-window .. t-1`` (strictly
    before the event) and the post-window covers returns on days
    ``t+1 .. t+window`` (strictly after the event), where ``t`` is the
    first bar on or after ``as_of``. ``close[t]`` participates only as
    the denominator of the post-window's first return, never as a
    numerator, so the announcement-day close-to-close move does not
    enter either window. Matches the strict-forward convention used by
    ``_forward_realized_vol``.
    """

    base_idx = series.index_strictly_before(as_of)
    if base_idx < window:
        return None
    on_or_after = series.index_on_or_after(as_of)
    if on_or_after + window >= len(series):
        return None
    pre_closes = series.close[base_idx - window : base_idx + 1]
    pre_rets = _log_returns(pre_closes)
    post_closes = series.close[on_or_after : on_or_after + window + 1]
    # post_closes spans (t, t+1, ..., t+window) -> window strict-forward
    # returns log(close[t+1]/close[t]) ... log(close[t+window]/close[t+window-1]).
    post_rets = _log_returns(post_closes)
    if len(pre_rets) < 2 or len(post_rets) < 2:
        return None

    def _std(values: list[float]) -> float:
        n = len(values)
        mean = sum(values) / n
        return (sum((v - mean) ** 2 for v in values) / (n - 1)) ** 0.5

    return _std(post_rets) - _std(pre_rets)


def _forward_realized_vol(
    series: _CloseSeries,
    as_of: _dt.date,
    *,
    window: int = 10,
) -> float | None:
    """Realised volatility of log returns over the *strictly* next ``window`` trading days.

    Target for the Phase 9 V2 vol-regime classifier (#195). 10 trading
    days is the default to keep the window clean of the FOMC blackout
    period (~10 trading days) and the next scheduled meeting. Returns
    ``None`` when the event sits within ``window`` days of the end of
    the price series (no forward window available).

    Computed as ``std(log_return(close[t+1 .. t+window]))`` with sample
    std (ddof=1), where ``t`` is the index of the event bar (first close
    on or after ``as_of``). The returns produced are
    ``log(close[t+1]/close[t])`` through
    ``log(close[t+window]/close[t+window-1])`` — strictly post-event.
    ``close[t]`` participates only as the denominator of the first
    return, never as a numerator, so the announcement-day close-to-close
    move (``log(close[t]/close[t-1])``) does not enter the target. This
    keeps the target disjoint from any feature derived from ``close[t]``
    (notably ``FeatureVector.market_close`` at the as-of bar) and the
    classification contract clean of look-ahead.

    A previous convention used the slice
    ``close[t-1 .. t+window-1]``, which folded
    ``log(close[t]/close[t-1])`` (the announcement-day reaction) into
    the target as the first return. That created a mechanical overlap
    with the day-``t`` close feature on the input side. The current
    slice ``close[t .. t+window]`` is the strict-forward replacement.
    """

    on_or_after = series.index_on_or_after(as_of)
    # Need ``window`` strictly-forward returns: indices
    # ``on_or_after, on_or_after+1, ..., on_or_after+window`` must
    # all be in range, so ``on_or_after + window`` must be a valid
    # index (i.e. ``< len(series)``).
    if on_or_after + window >= len(series):
        return None
    closes = series.close[on_or_after : on_or_after + window + 1]
    rets = _log_returns(closes)
    if len(rets) < 1:
        return None
    if len(rets) == 1:
        # window=1 (the auxiliary multi-horizon target). Sample std with
        # ddof=1 is undefined for a single observation; the standard
        # daily-RV approximation in this degenerate case is the absolute
        # log return. The 10d canonical path is unaffected (always 10
        # returns).
        return abs(rets[0])
    n = len(rets)
    mean = sum(rets) / n
    return (sum((v - mean) ** 2 for v in rets) / (n - 1)) ** 0.5


# ---------------------------------------------------------------------------
# Concurrent macro release calendar (real BLS / ISM / FRED-derived dates)
# ---------------------------------------------------------------------------


def _resolve_macro_calendar(
    csv_path: Path | str | None,
) -> MacroReleaseCalendar:
    """Pick the real-release calendar when available, fall back to heuristic.

    Resolution order:

    1. ``csv_path`` argument (parquet or CSV). Used by tests.
    2. ``data/external/macro_releases.csv`` -- a curated CSV shipped in the
       repo. Loaded once per build call.
    3. Pure-rule heuristic (legacy behaviour). Used when neither file is
       present, so the builder still runs on a fresh checkout without
       network calls.
    """

    if csv_path is not None:
        return load_macro_release_calendar(Path(csv_path))
    default_path = DEFAULT_MACRO_RELEASES_CSV
    if default_path.exists():
        return load_macro_release_calendar(default_path)
    return build_heuristic_calendar()


def _has_concurrent_macro_release(
    event_date: _dt.date,
    series: _CloseSeries,
    *,
    radius: int = CONCURRENT_MACRO_TRADING_DAY_RADIUS,
    calendar: MacroReleaseCalendar | None = None,
) -> bool:
    """Return True iff a major US macro release sits within ``radius``
    trading days of ``event_date``.

    The ``calendar`` argument carries the real release dates (BLS / ISM /
    FRED-derived). When omitted we fall back to the rule-based heuristic
    (legacy behaviour) so callers that never had a calendar continue to
    work.
    """

    if calendar is None:
        calendar = build_heuristic_calendar()
    macro_dates = calendar.dates_in_year(event_date.year)
    macro_dates |= calendar.dates_in_year(event_date.year - 1)
    macro_dates |= calendar.dates_in_year(event_date.year + 1)
    base_idx = series.index_on_or_after(event_date)
    if base_idx >= len(series):
        return False
    lo_idx = max(0, base_idx - radius)
    hi_idx = min(len(series) - 1, base_idx + radius)
    window = set(series.dates[lo_idx : hi_idx + 1])
    return any(d in macro_dates for d in window)


# ---------------------------------------------------------------------------
# Intra-meeting tone shift (statement vs same-day press conference)
# ---------------------------------------------------------------------------


def _axis_to_numeric(value: Any, encoding: dict[str, float]) -> float:
    """Best-effort encode an axis value to a numeric scalar.

    Returns ``math.nan`` when the value is missing or unencodable so the
    downstream shift propagates NaN honestly instead of coercing to 0.
    Order of preference:

    1. ``None`` / empty -> NaN.
    2. ``int`` / ``float`` (not NaN itself) -> passthrough.
    3. ``str`` looked up case-insensitively in ``encoding``.
    4. ``str`` parseable as ``float`` (e.g. "0.42") -> the float.
    5. Otherwise NaN.
    """

    if value is None:
        return math.nan
    if isinstance(value, bool):
        # bool is a subclass of int; treat as NaN to avoid surprising True->1.0
        return math.nan
    if isinstance(value, int | float):
        if isinstance(value, float) and math.isnan(value):
            return math.nan
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return math.nan
        encoded = encoding.get(stripped.lower())
        if encoded is not None:
            return float(encoded)
        try:
            return float(stripped)
        except ValueError:
            return math.nan
    return math.nan


def _compute_intra_meeting_shifts(
    docs: Sequence[_EventDoc],
) -> dict[str, dict[str, float]]:
    """Return ``{event_date: {axis_name: shift_value}}`` for every date.

    The shift is ``press_conference_axis - statement_axis`` after
    encoding each side through ``_axis_to_numeric``. When either side
    is missing for the date, all three shift values are NaN -- never
    coerced to zero. Multi-source duplicates are collapsed via
    ``_choose_preferred`` so the shift is computed on the *preferred*
    statement and the *preferred* press_conference per date.

    The result is keyed by ``event_date`` (string, ISO date) so callers
    can join the per-date shifts onto every row regardless of source,
    asset, or horizon -- the shift is a property of the date.
    """

    nan_payload = {
        "intra_meeting_stance_shift": math.nan,
        "intra_meeting_certainty_shift": math.nan,
        "intra_meeting_factor_shift": math.nan,
    }
    preferred = _choose_preferred(list(docs))
    by_date: dict[str, dict[str, _EventDoc]] = defaultdict(dict)
    for doc in preferred:
        by_date[doc.event_date][doc.event_kind] = doc

    shifts: dict[str, dict[str, float]] = {}
    for event_date, kinds in by_date.items():
        stmt = kinds.get("statement")
        pc = kinds.get("press_conference")
        if stmt is None or pc is None:
            shifts[event_date] = dict(nan_payload)
            continue
        payload: dict[str, float] = {}
        for axis_name, encoding in _INTRA_MEETING_AXIS_ENCODING.items():
            stmt_val = _axis_to_numeric(stmt.multi_axis.get(axis_name[5:]), encoding)
            pc_val = _axis_to_numeric(pc.multi_axis.get(axis_name[5:]), encoding)
            if math.isnan(stmt_val) or math.isnan(pc_val):
                payload[f"intra_meeting_{axis_name[5:]}_shift"] = math.nan
            else:
                payload[f"intra_meeting_{axis_name[5:]}_shift"] = pc_val - stmt_val
        shifts[event_date] = payload
    return shifts


# ---------------------------------------------------------------------------
# Per-event row construction
# ---------------------------------------------------------------------------


def _document_id(source: str, event_date: str, event_kind: str) -> str:
    h = hashlib.sha256(f"{source}|{event_date}|{event_kind}".encode("utf-8"))
    return h.hexdigest()[:16]


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _prior_window_sha(bars: Sequence[_PriorBar]) -> str:
    parts: list[str] = []
    for b in bars:
        parts.append(
            "|".join(
                [
                    b.date.isoformat(),
                    f"{b.close:.10f}",
                    f"{b.volume:.4f}",
                    f"{b.vol_5d:.10f}",
                    f"{b.vol_20d:.10f}",
                    f"{b.vol_60d:.10f}",
                    f"{b.cum_return_20d:.10f}",
                    f"{b.vix_close:.6f}",
                    f"{b.dxy_close:.6f}",
                    f"{b.tnx_close:.6f}",
                    f"{b.gold_close:.6f}",
                    f"{b.vix3m_close:.6f}",
                    f"{b.irx_close:.6f}",
                    f"{b.vix_term_slope:.6f}",
                    f"{b.yield_curve_slope_10y_3m:.6f}",
                ]
            )
        )
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def _bars_to_json(bars: Sequence[_PriorBar]) -> str:
    payload = [
        {
            "date": b.date.isoformat(),
            "close": round(b.close, 10),
            "volume": round(b.volume, 4),
            "vol_5d": round(b.vol_5d, 10),
            "vol_20d": round(b.vol_20d, 10),
            "vol_60d": round(b.vol_60d, 10),
            "cum_return_20d": round(b.cum_return_20d, 10),
            "vix_close": round(b.vix_close, 6),
            "dxy_close": round(b.dxy_close, 6),
            "tnx_close": round(b.tnx_close, 6),
            "gold_close": round(b.gold_close, 6),
            "vix3m_close": round(b.vix3m_close, 6),
            "irx_close": round(b.irx_close, 6),
            "vix_term_slope": round(b.vix_term_slope, 6),
            "yield_curve_slope_10y_3m": round(b.yield_curve_slope_10y_3m, 6),
        }
        for b in bars
    ]
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


def _as_of_for_event(event_date: str, event_kind: str) -> str:
    """Apply the placeholder announcement time.

    FOMC kinds (statement, minutes, press_conference, testimony) use the
    2pm ET / 19:00 UTC placeholder; speech kinds use 14:00 UTC; macro
    releases (CPI, NFP) use the 8:30 ET BLS slot rounded to 13:00 UTC.
    Documented in the module docstring.
    """

    if event_kind == "macro_release":
        return f"{event_date}{MACRO_AS_OF_TIME}"
    if event_kind in _SPEECH_KINDS:
        return f"{event_date}{SPEECH_AS_OF_TIME}"
    return f"{event_date}{FOMC_AS_OF_TIME}"


def _assert_no_lookahead(as_of: _dt.date, bars: Sequence[_PriorBar]) -> None:
    if not bars:
        return
    last = bars[-1].date
    if not (last < as_of):
        raise ValueError(
            f"prior-window contract violated: last bar {last} is not strictly "
            f"before as_of {as_of}"
        )


def _build_event_rows(
    doc: _EventDoc,
    *,
    asset: str,
    benchmark: str,
    asset_series: _CloseSeries,
    bench_series: _CloseSeries,
    horizons: Sequence[int],
    credibility_kwargs: dict[str, Any],
    macro_release_calendar: MacroReleaseCalendar,
    concurrent_macro_window_days: int = CONCURRENT_MACRO_TRADING_DAY_RADIUS,
    intra_meeting_shift: dict[str, float] | None = None,
    cross_asset_lookup: dict[_dt.date, dict[str, float]] | None = None,
    cross_asset_series: dict[str, _CloseSeries] | None = None,
    rates_lookup: RatesPanelLookup | None = None,
    rates_horizon: int = 5,
    prior_statement_text: str | None = None,
    vote_tally: VoteTally | None = None,
    delta_encoder: Any | None = None,
    per_asset_target_series: dict[str, _CloseSeries] | None = None,
    prior_window_days: int = PRIOR_WINDOW_DAYS,
) -> list[dict[str, Any]]:
    event_date = _date(doc.event_date)
    as_of_ts = _as_of_for_event(doc.event_date, doc.event_kind)
    as_of_date = event_date  # placeholder time is same-day; window cuts on date

    prior_bars = _build_prior_window(
        asset_series,
        as_of_date,
        cross_asset_lookup=cross_asset_lookup,
        window_days=prior_window_days,
    )
    if prior_bars is None:
        return []
    _assert_no_lookahead(as_of_date, prior_bars)

    targets = _realized_returns(asset_series, as_of_date, horizons)
    if not targets:
        return []

    # t+1d direction
    t1d = targets.get(1)
    if t1d is None:
        direction_t1d = 0
    else:
        ret, _ = t1d
        direction_t1d = (1 if ret > 0 else (-1 if ret < 0 else 0))

    if asset == benchmark:
        alpha, beta = 0.0, 1.0
    else:
        fit = _market_model_for_event(asset_series, bench_series, as_of_date)
        if fit is None:
            return []
        alpha, beta = fit

    # Benchmark realized returns at same horizons for abnormal-return calc
    if asset == benchmark:
        bench_targets: dict[int, tuple[float, _dt.date | None]] = targets
    else:
        bench_targets = _realized_returns(bench_series, as_of_date, horizons) or {}

    vol_shift = _volatility_shift(asset_series, as_of_date)
    # #478 VIX term-structure + VRP at T-1. Reads strictly-prior closes
    # from the ^VIX family cross-asset series plus the asset series for
    # the realised baseline. Pre-coverage events (notably ^VIX1M /
    # ^VIX3M / ^VIX6M before 2008) emit ``None`` on the affected
    # scalars; the loader's missing-flag pattern handles the per-event
    # widening.
    vix_term_block = _vix_term_structure_features(
        as_of=as_of_date,
        series_by_symbol=cross_asset_series or {},
        asset_series=asset_series,
    )
    # Phase 9 V2 (#195) target: realised volatility over the next 10
    # trading days. Class labels (calm / normal / high) are NOT
    # assigned here -- the per-fold quantile cutoffs are computed at
    # training time on the train slice to avoid look-ahead leakage.
    forward_vol_10d = _forward_realized_vol(asset_series, as_of_date, window=10)
    # Multi-horizon auxiliary targets (#480, foundation for the
    # multi-asset / multi-horizon roadmap). Same generator, parametrised
    # window. ``10d`` remains the canonical regime target; the auxiliary
    # horizons ride alongside on the parquet so downstream heads can be
    # mounted without a rebuild. Each horizon degrades to ``None``
    # independently when the post-event window runs off the end of the
    # asset's price series.
    forward_vol_multi_horizon: dict[int, float | None] = {
        h: _forward_realized_vol(asset_series, as_of_date, window=h)
        for h in (1, 3, 5, 20, 30)
    }
    # #236 GARCH(1,1)-residual decomposition of the same target. Fits
    # GARCH(1,1) on log returns dated strictly before ``as_of_date`` and
    # forecasts a 1-day-equivalent vol over the same 10-trading-day
    # forward window. The residual = raw - baseline isolates the
    # unanticipated component of realised vol given the standard
    # conditional-variance model. Strict-prior contract: the fit reads
    # only closes ``< as_of_date``; the forecast is conditional-on-fit
    # and reads no close at or after ``as_of_date``. Below
    # ``MIN_FIT_RETURNS`` (252 trading days) or on convergence failure
    # both columns degrade to None and the supervised row keeps the raw
    # target intact. See ADR 0034.
    from app.data.garch_residual import compute_for_event as _garch_compute_for_event

    garch_result = _garch_compute_for_event(
        dates=asset_series.dates,
        closes=asset_series.close,
        event_date=as_of_date,
        forward_realized_vol_10d=forward_vol_10d,
    )
    concurrent_macro = _has_concurrent_macro_release(
        event_date,
        asset_series,
        radius=concurrent_macro_window_days,
        calendar=macro_release_calendar,
    )

    # Credibility vector (degrades to zeros when inputs are absent)
    cred = _safe_credibility(as_of_ts, credibility_kwargs)

    prior_sha = _prior_window_sha(prior_bars)
    prior_json = _bars_to_json(prior_bars)
    text_hash = _text_hash(doc.text)
    document_id = _document_id(doc.source, doc.event_date, doc.event_kind)
    token_count = len(doc.text.split())

    if intra_meeting_shift is None:
        intra_meeting_shift = {
            "intra_meeting_stance_shift": math.nan,
            "intra_meeting_certainty_shift": math.nan,
            "intra_meeting_factor_shift": math.nan,
        }

    # Rates-complex forward targets (#291). Strict-forward bps changes
    # over the rates panel's t -> t+rates_horizon trading-day window.
    # Empty lookup -> every target is None and the column degrades
    # cleanly to a nullable float on the parquet write side.
    rates_forward_targets: dict[str, float | None] = {
        target_column: None for _, target_column in FORWARD_TARGET_COLUMNS
    }
    if rates_lookup is not None:
        for column, target_column in FORWARD_TARGET_COLUMNS:
            rates_forward_targets[target_column] = forward_yield_change_bps(
                rates_lookup,
                asset_series.dates,
                column=column,
                event_date=as_of_date,
                horizon=rates_horizon,
            )

    # Pre-meeting expectation features (#291). Strict-backward at t-1.
    pre_meeting: PreMeetingFeatures | None = None
    if rates_lookup is not None:
        pre_meeting = compute_pre_meeting_features(
            rates_lookup, asset_series.dates, event_date=as_of_date
        )

    # #443 statement-delta. Only fires when a strict-prior statement is
    # available (cold-start of the corpus, or non-statement event kinds
    # like minutes / press_conference, get a null triple + missing flag).
    # The encoder pass is opt-in via ``delta_encoder``; without one, the
    # text spans are stamped on the row and the embedding column carries
    # ``None``.
    statement_delta_inserted: str | None = None
    statement_delta_deleted: str | None = None
    statement_delta_substituted: str | None = None
    statement_delta_embedding: list[float] | None = None
    if doc.event_kind == "statement" and prior_statement_text:
        delta = compute_delta_for_event(
            current_text=doc.text,
            prior_text=prior_statement_text,
            encode_text=delta_encoder,
        )
        if delta is not None:
            statement_delta_inserted = delta.inserted_text
            statement_delta_deleted = delta.deleted_text
            # The substituted-pairs list is JSON-serialised on the column
            # so the parquet stays a flat string surface. A consumer
            # reading the column re-parses with ``json.loads`` to get back
            # the ``list[tuple[str, str]]`` structure.
            statement_delta_substituted = json.dumps(
                delta.substituted_pairs, separators=(",", ":"), sort_keys=False
            )
            statement_delta_embedding = (
                list(delta.embedding) if delta.embedding is not None else None
            )

    # #444 vote tally. The block fires only on statement-kind rows --
    # the vote is in the FOMC statement document, not in minutes /
    # speeches / press conferences. Non-statement rows carry None.
    if vote_tally is None and doc.event_kind == "statement":
        vote_tally = parse_vote_tally(doc.text)

    # #481 per-asset 10d forward realised-vol targets. One value per
    # symbol per event_date, computed on each symbol's own close series
    # under the strict-forward convention pinned by
    # ``_forward_realized_vol``. The canonical
    # ``forward_realized_vol_10d`` stays as the ^GSPC alias above.
    # Missing series (the symbol's cache fetch failed, or the event
    # sits before the symbol's listing) collapse to ``None`` so the
    # downstream classifier treats it as "skip" rather than zero.
    per_asset_target_values: dict[str, float | None] = {
        per_asset_target_column(sym): None for sym in PER_ASSET_TARGET_SYMBOLS
    }
    if per_asset_target_series:
        for sym in PER_ASSET_TARGET_SYMBOLS:
            series = per_asset_target_series.get(sym)
            if series is None or not series.dates:
                continue
            value = _forward_realized_vol(series, as_of_date, window=10)
            per_asset_target_values[per_asset_target_column(sym)] = (
                float(value) if value is not None else None
            )

    rows: list[dict[str, Any]] = []
    for h in horizons:
        tgt = targets.get(h)
        if tgt is None:
            continue
        realized_return, realized_date = tgt
        if asset == benchmark:
            # No market-model adjustment: the benchmark IS the asset.
            # By contract (alpha=0, beta=1) and abnormal = raw return.
            abnormal = realized_return
        else:
            entry = bench_targets.get(h)
            bench_ret = entry[0] if entry else 0.0
            abnormal = realized_return - (alpha + beta * bench_ret)
        rows.append(
            {
                "event_date": doc.event_date,
                "event_kind": doc.event_kind,
                "document_id": document_id,
                "text_hash": text_hash,
                "source": doc.source,
                "source_record_id": doc.source_record_id,
                "as_of_ts": as_of_ts,
                "text": doc.text,
                "token_count": token_count,
                "axis_stance": doc.multi_axis.get("stance"),
                "axis_time": _encode_axis_time(doc.multi_axis.get("time")),
                "axis_certainty": _encode_axis_certainty(
                    doc.multi_axis.get("certainty")
                ),
                "axis_factor": _encode_axis_factor(doc.multi_axis.get("factor")),
                "axis_time_label": doc.time_label,
                "axis_certain_label": doc.certain_label,
                "multi_axis_extras": doc.multi_axis_extras or None,
                "credibility_drift_score": float(cred.drift_score),
                "credibility_realized_vs_stated_gap": float(cred.realized_vs_stated_gap),
                "credibility_market_implied_gap": float(cred.market_implied_gap),
                "credibility_months_since_reversal": int(cred.months_since_reversal),
                "prior_window_sha256": prior_sha,
                "prior_bars_json": prior_json,
                "asset_symbol": asset,
                "horizon": int(h),
                "realized_return": float(realized_return),
                "abnormal_return": float(abnormal),
                "alpha": float(alpha),
                "beta": float(beta),
                "direction_t1d": int(direction_t1d),
                "volatility_shift": float(vol_shift) if vol_shift is not None else None,
                "forward_realized_vol_10d": (
                    float(forward_vol_10d) if forward_vol_10d is not None else None
                ),
                # Multi-horizon auxiliary targets (#480). Same generator
                # as the canonical 10d, parametrised window. Independent
                # null degradation per horizon when the post-event window
                # runs off the end of the asset price series.
                "forward_realized_vol_1d": (
                    float(forward_vol_multi_horizon[1])
                    if forward_vol_multi_horizon[1] is not None
                    else None
                ),
                "forward_realized_vol_3d": (
                    float(forward_vol_multi_horizon[3])
                    if forward_vol_multi_horizon[3] is not None
                    else None
                ),
                "forward_realized_vol_5d": (
                    float(forward_vol_multi_horizon[5])
                    if forward_vol_multi_horizon[5] is not None
                    else None
                ),
                "forward_realized_vol_20d": (
                    float(forward_vol_multi_horizon[20])
                    if forward_vol_multi_horizon[20] is not None
                    else None
                ),
                "forward_realized_vol_30d": (
                    float(forward_vol_multi_horizon[30])
                    if forward_vol_multi_horizon[30] is not None
                    else None
                ),
                # #236 GARCH(1,1)-residual decomposition. See
                # ``app.data.garch_residual`` + ADR 0034. Both columns
                # are nullable: ``baseline`` is None when GARCH did not
                # converge or the strict-prior window is short, and the
                # residual additionally requires a non-null raw target.
                "forward_realized_vol_10d_garch_baseline": (
                    float(garch_result.baseline)
                    if garch_result.baseline is not None
                    else None
                ),
                "forward_realized_vol_10d_garch_residual": (
                    float(garch_result.residual)
                    if garch_result.residual is not None
                    else None
                ),
                "concurrent_macro_release": bool(concurrent_macro),
                "intra_meeting_stance_shift": float(
                    intra_meeting_shift["intra_meeting_stance_shift"]
                ),
                "intra_meeting_certainty_shift": float(
                    intra_meeting_shift["intra_meeting_certainty_shift"]
                ),
                "intra_meeting_factor_shift": float(
                    intra_meeting_shift["intra_meeting_factor_shift"]
                ),
                "realized_date": realized_date.isoformat() if realized_date else None,
                # --- Rates-complex forward targets (#291), raw bps ---
                "yield_2y_change_5d": _maybe_float(
                    rates_forward_targets.get("yield_2y_change_5d")
                ),
                "yield_5y_change_5d": _maybe_float(
                    rates_forward_targets.get("yield_5y_change_5d")
                ),
                "terminal_rate_change_5d": _maybe_float(
                    rates_forward_targets.get("terminal_rate_change_5d")
                ),
                # --- Pre-meeting expectation features (#291) ---
                "pre_meeting_yield_1y": _pre_meeting_field(pre_meeting, "pre_meeting_yield_1y"),
                "pre_meeting_yield_2y": _pre_meeting_field(pre_meeting, "pre_meeting_yield_2y"),
                "pre_meeting_yield_5y": _pre_meeting_field(pre_meeting, "pre_meeting_yield_5y"),
                "pre_meeting_yield_10y": _pre_meeting_field(pre_meeting, "pre_meeting_yield_10y"),
                "pre_meeting_slope_10y_2y": _pre_meeting_field(
                    pre_meeting, "pre_meeting_slope_10y_2y"
                ),
                "pre_meeting_slope_10y_3m": _pre_meeting_field(
                    pre_meeting, "pre_meeting_slope_10y_3m"
                ),
                "pre_meeting_trailing_2y_yield_change_5d_bps": _pre_meeting_field(
                    pre_meeting, "pre_meeting_trailing_2y_yield_change_5d_bps"
                ),
                "pre_meeting_implied_next_move_bps": _pre_meeting_field(
                    pre_meeting, "pre_meeting_implied_next_move_bps"
                ),
                "pre_meeting_implied_hike_prob": _pre_meeting_field(
                    pre_meeting, "pre_meeting_implied_hike_prob"
                ),
                "pre_meeting_implied_cut_prob": _pre_meeting_field(
                    pre_meeting, "pre_meeting_implied_cut_prob"
                ),
                "pre_meeting_implied_pause_prob": _pre_meeting_field(
                    pre_meeting, "pre_meeting_implied_pause_prob"
                ),
                "pre_meeting_days_since_last_rate_change": _pre_meeting_field(
                    pre_meeting, "pre_meeting_days_since_last_rate_change"
                ),
                # --- #443 statement-delta (redline) text spans + embedding ---
                "statement_delta_inserted": statement_delta_inserted,
                "statement_delta_deleted": statement_delta_deleted,
                "statement_delta_substituted_pairs": statement_delta_substituted,
                "statement_delta_embedding": statement_delta_embedding,
                # --- #444 vote tally + dissent structural columns ---
                "votes_for": (
                    int(vote_tally.votes_for) if vote_tally is not None else None
                ),
                "votes_against": (
                    int(vote_tally.votes_against)
                    if vote_tally is not None
                    else None
                ),
                "dissent_count": (
                    int(vote_tally.dissent_count)
                    if vote_tally is not None
                    else None
                ),
                "is_unanimous": (
                    bool(vote_tally.is_unanimous)
                    if vote_tally is not None
                    else None
                ),
                "dissent_direction": (
                    vote_tally.dissent_direction
                    if vote_tally is not None
                    else None
                ),
                # --- #481 per-asset 10d forward realised-vol targets ---
                # One nullable float per workspace asset-picker symbol.
                # Same strict-forward 10-trading-day convention as the
                # canonical alias above; ``None`` when the symbol's
                # series is absent or the event sits within the forward
                # window of the series tail.
                **per_asset_target_values,
                # --- #478 VIX term-structure + VRP at T-1 ---
                # Strict-prior scalars read off the ^VIX family
                # cross-asset series plus the asset's own close history
                # for the realised baseline. Per-scalar ``None`` when
                # the underlying series is missing or pre-coverage.
                **{k: _maybe_float(v) for k, v in vix_term_block.items()},
            }
        )
    return rows


def _maybe_float(value: float | None) -> float | None:
    if value is None:
        return None
    return float(value)


def _pre_meeting_field(
    features: PreMeetingFeatures | None, field_name: str
) -> float | None:
    if features is None:
        return None
    value = getattr(features, field_name)
    if value is None:
        return None
    return float(value)




_CREDIBILITY_EMPTY_INPUTS_WARNED = False


def _safe_credibility(as_of_ts: str, kwargs: dict[str, Any]) -> CredibilityVector:
    """Wrap ``load_credibility_for_run`` so missing inputs degrade to zeros.

    The loader already handles "no embeddings / no FRED cache" gracefully,
    but we still catch ValueError so a malformed ``as_of_ts`` placeholder
    never aborts the whole build. Documented behaviour: zero vector means
    "credibility unknown", not "credibility zero".

    Logs a one-shot warning when ``embedding_path`` is None AND
    ``stance_by_date`` is empty — that combination guarantees a zero
    credibility vector and was the silent-failure mode in pre-2026-05-17
    events.parquet builds.
    """

    global _CREDIBILITY_EMPTY_INPUTS_WARNED
    if (
        not _CREDIBILITY_EMPTY_INPUTS_WARNED
        and kwargs.get("embedding_path") is None
        and not kwargs.get("stance_by_date")
    ):
        logging.getLogger(__name__).warning(
            "credibility inputs empty (no embedding_path, no stance_by_date); "
            "all four credibility axes will be 0.0 for every event. "
            "Pass --embedding-path data/raw/embeddings/<encoder>_<rev>.parquet "
            "to enable drift_score, and confirm the registry carries "
            "mapped_label values so the auto-derived stance series is non-empty."
        )
        _CREDIBILITY_EMPTY_INPUTS_WARNED = True

    try:
        return load_credibility_for_run(as_of_ts=as_of_ts, **kwargs)
    except (ValueError, FileNotFoundError):
        return CredibilityVector(0.0, 0.0, 0.0, 0)


# ---------------------------------------------------------------------------
# Top-level builder
# ---------------------------------------------------------------------------


COLUMN_ORDER = (
    "event_date",
    "event_kind",
    "document_id",
    "text_hash",
    "source",
    "source_record_id",
    "as_of_ts",
    "text",
    "token_count",
    "axis_stance",
    "axis_time",
    "axis_certainty",
    "axis_factor",
    "axis_time_label",
    "axis_certain_label",
    # Merged ``multi_axis_extras`` payload from the bucketed source rows.
    # Carries GSS / Swanson bp-scale factor decompositions
    # (``gss_target_factor`` / ``swanson_target_factor`` etc.) and the
    # gtfintechlab dataset-revision tags. Nullable; ``None`` when the
    # bucket has no extras to merge.
    "multi_axis_extras",
    "credibility_drift_score",
    "credibility_realized_vs_stated_gap",
    "credibility_market_implied_gap",
    "credibility_months_since_reversal",
    "prior_window_sha256",
    "prior_bars_json",
    "asset_symbol",
    "horizon",
    "realized_return",
    "abnormal_return",
    "alpha",
    "beta",
    "direction_t1d",
    "volatility_shift",
    "forward_realized_vol_10d",
    # #480 multi-horizon auxiliary targets. Same generator, parametrised
    # window. 10d remains canonical; these ride alongside.
    "forward_realized_vol_1d",
    "forward_realized_vol_3d",
    "forward_realized_vol_5d",
    "forward_realized_vol_20d",
    "forward_realized_vol_30d",
    # #236 GARCH(1,1)-residual variant of the forward-vol target.
    "forward_realized_vol_10d_garch_baseline",
    "forward_realized_vol_10d_garch_residual",
    # #481 per-asset forward realised-vol targets (10d). Eight columns,
    # one per workspace asset-picker symbol; ``_gspc`` is the canonical
    # alias of ``forward_realized_vol_10d`` and the remaining seven
    # cover the other indices, dollar index, VIX, and the three major
    # FX pairs. All nullable, required=False so older parquets validate.
    *(per_asset_target_column(sym) for sym in PER_ASSET_TARGET_SYMBOLS),
    # #478 VIX term-structure + VRP at T-1. Six nullable scalars read
    # strictly before event_date. ``None`` per scalar on pre-coverage
    # events (^VIX1M / ^VIX3M / ^VIX6M before 2008) or asset-history
    # gaps; downstream loader handles the per-event missing flag.
    "vix_t_minus_1",
    "vix1m_t_minus_1",
    "vix3m_t_minus_1",
    "vix6m_t_minus_1",
    "vix_3m_over_1m_slope",
    "vrp_t_minus_1",
    "concurrent_macro_release",
    "intra_meeting_stance_shift",
    "intra_meeting_certainty_shift",
    "intra_meeting_factor_shift",
    "realized_date",
    # Rates-complex forward targets (#291)
    "yield_2y_change_5d",
    "yield_5y_change_5d",
    "terminal_rate_change_5d",
    # Pre-meeting expectation features (#291)
    "pre_meeting_yield_1y",
    "pre_meeting_yield_2y",
    "pre_meeting_yield_5y",
    "pre_meeting_yield_10y",
    "pre_meeting_slope_10y_2y",
    "pre_meeting_slope_10y_3m",
    "pre_meeting_trailing_2y_yield_change_5d_bps",
    "pre_meeting_implied_next_move_bps",
    "pre_meeting_implied_hike_prob",
    "pre_meeting_implied_cut_prob",
    "pre_meeting_implied_pause_prob",
    "pre_meeting_days_since_last_rate_change",
    # #443 statement-delta (redline). Three text spans + a nullable
    # mean-pooled embedding vector. The embedding is populated only when
    # the build pass wires in a delta_encoder; otherwise the column
    # carries None and the loader collapses to the missing-1.0 slot.
    "statement_delta_inserted",
    "statement_delta_deleted",
    "statement_delta_substituted_pairs",
    "statement_delta_embedding",
    # #444 vote tally + dissent. Structural columns; the vote IS the
    # event so no leak surface. Five fields: counts (for/against),
    # alias dissent_count, is_unanimous boolean, and the inferred
    # dissent_direction ("hawkish_dissent" / "dovish_dissent" / None).
    "votes_for",
    "votes_against",
    "dissent_count",
    "is_unanimous",
    "dissent_direction",
)


def build_event_rows(
    *,
    package_dir: Path,
    asset: str = DEFAULT_ASSET,
    benchmark: str = DEFAULT_BENCHMARK,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    asset_series: _CloseSeries | None = None,
    bench_series: _CloseSeries | None = None,
    market_cache_dir: Path | None = None,
    embedding_path: Path | str | None = None,
    fred_cache_dir: Path | None = None,
    fred_series_id: str = "DFF",
    stance_by_date: Sequence[tuple[str, float]] = (),
    summary: _BuildSummary | None = None,
    keep_all_sources: bool = False,
    macro_release_calendar: MacroReleaseCalendar | None = None,
    macro_release_csv_path: Path | str | None = None,
    concurrent_macro_window_days: int = CONCURRENT_MACRO_TRADING_DAY_RADIUS,
    rates_panel_path: Path | str | None = None,
    rates_lookup: RatesPanelLookup | None = None,
    rates_horizon: int = 5,
    delta_encoder: Any | None = None,
    per_asset_target_series: dict[str, _CloseSeries] | None = None,
    per_asset_target_cache_dir: Path | None = None,
    per_asset_target_force_refresh: bool = False,
    prior_window_days: int = PRIOR_WINDOW_DAYS,
) -> pd.DataFrame:
    """Build the events DataFrame for one training package.

    Tests inject ``asset_series`` and ``bench_series`` directly. The smoke
    run leaves them ``None`` and we fetch (and cache) live yfinance bars.

    When ``keep_all_sources=False`` (default) the frame collapses to one
    row per ``(event_date, event_kind, asset_symbol, horizon)`` via the
    ``_SOURCE_PREFERENCE`` order. When ``keep_all_sources=True`` every
    source survives, producing one row per
    ``(event_date, event_kind, source, asset_symbol, horizon)``.

    The ``concurrent_macro_release`` flag consults ``macro_release_calendar``
    (a real BLS/ISM/FRED-derived calendar) when one is provided; otherwise it
    falls back to the deterministic heuristic. Pass ``macro_release_csv_path``
    to load a pre-built calendar parquet/CSV from disk.
    """

    if summary is None:
        summary = _BuildSummary()

    if macro_release_calendar is None:
        macro_release_calendar = _resolve_macro_calendar(macro_release_csv_path)

    registry_rows = _load_registry_rows(package_dir)

    # Auto-derive a stance series from the registry when the caller did not
    # supply one. The credibility loader needs this to compute
    # ``realized_vs_stated_gap`` (Pearson against the FRED-DFF window) and
    # ``months_since_reversal`` (sign flips); without it both axes degrade
    # silently to 0.0. Explicit ``stance_by_date`` callers (tests, smoke
    # runs) keep their override unchanged.
    if not stance_by_date:
        stance_by_date = _derive_stance_by_date(registry_rows)

    docs_all = _aggregate_events(registry_rows)
    docs = list(docs_all) if keep_all_sources else _choose_preferred(docs_all)
    if not docs:
        return _empty_frame()

    # Pre-compute intra_meeting_*_shift on the preferred-source pair per
    # event_date. The shift is a property of the date and is replicated
    # to every (source x asset x horizon) row sharing that date, so the
    # full and collapsed views report identical shifts even when the
    # full view holds multiple source-level rows per kind.
    intra_meeting_shifts = _compute_intra_meeting_shifts(docs_all)

    # Decide the window we need to fetch.
    event_dates = sorted({_date(d.event_date) for d in docs})
    earliest = event_dates[0] - _dt.timedelta(days=MARKET_MODEL_WINDOW_DAYS * 2 + 60)
    latest = event_dates[-1] + _dt.timedelta(days=max(horizons) * 2 + 60)

    # Capture caller-injection state up front so the #481 per-asset
    # target block below can opt out of outbound yfinance fetches when
    # the caller is in test mode (asset + benchmark series both
    # explicitly supplied).
    asset_series_was_injected = asset_series is not None
    bench_series_was_injected = bench_series is not None

    if asset_series is None:
        asset_series = _fetch_close_series(
            asset, start=earliest, end=latest, cache_dir=market_cache_dir
        )
    if bench_series is None:
        if benchmark == asset:
            bench_series = asset_series
        else:
            bench_series = _fetch_close_series(
                benchmark, start=earliest, end=latest, cache_dir=market_cache_dir
            )

    # A3 (#208) cross-asset cache pre-fetch + lookup index. Each symbol
    # fails gracefully -- a missing symbol's contribution to every bar
    # collapses to 0.0 rather than blocking the whole pipeline. This
    # matters because the historical coverage of VIX (1990+),
    # DX-Y.NYB, ^TNX, and GC=F is not uniform and we still want event
    # rows from earlier years to land with their existing features.
    cross_asset_series: dict[str, _CloseSeries] = {}
    for symbol, _field_name in CROSS_ASSET_SYMBOLS:
        try:
            cross_asset_series[symbol] = _fetch_close_series(
                symbol, start=earliest, end=latest, cache_dir=market_cache_dir
            )
        except RuntimeError as exc:
            print(
                f"[event_dataset_builder] cross-asset fetch failed for "
                f"{symbol}: {exc}; field will be 0.0 across all bars.",
                file=sys.stderr,
            )
    cross_asset_lookup = _build_cross_asset_lookup(cross_asset_series)

    # #481 per-asset 10d forward realised-vol targets. Same fetch-or-
    # cache contract as the cross-asset feed above: one parquet per
    # symbol under ``per_asset_target_cache_dir`` (defaults to
    # ``<DATA_DIR>/external/yfinance``), and a missing/failing symbol
    # collapses the column to ``None`` rather than blocking the build.
    # The ^GSPC / benchmark entries reuse the canonical series the
    # caller already loaded so the ``_gspc`` column and
    # ``forward_realized_vol_10d`` alias come from byte-identical inputs.
    if per_asset_target_series is None:
        per_asset_target_series = {}
        # Detect test mode: callers (unit tests) that supply both
        # ``asset_series`` and ``bench_series`` upfront expect no
        # outbound network. Reuse the injected series for ^GSPC /
        # benchmark and leave every other symbol absent (column ->
        # None). The smoke / CLI path leaves both None and fetches.
        injected_canonical_series = (
            asset_series_was_injected and bench_series_was_injected
        )
        target_cache_dir: Path | None = per_asset_target_cache_dir
        if target_cache_dir is None:
            target_cache_dir = DEFAULT_DATA_DIR / "external" / "yfinance"
        for sym in PER_ASSET_TARGET_SYMBOLS:
            if sym == asset and asset_series is not None:
                per_asset_target_series[sym] = asset_series
                continue
            if sym == benchmark and bench_series is not None:
                per_asset_target_series[sym] = bench_series
                continue
            if injected_canonical_series:
                # Test mode: skip the outbound fetch. The per-asset
                # column for ``sym`` lands as ``None`` on every row,
                # exercising the missing-data path explicitly.
                continue
            try:
                per_asset_target_series[sym] = _fetch_close_series(
                    sym,
                    start=earliest,
                    end=latest,
                    cache_dir=target_cache_dir,
                    force_refresh=per_asset_target_force_refresh,
                )
            except Exception as exc:
                # Broadened from RuntimeError to Exception (#481 review): yfinance
                # can surface HTTP 429 / network / parser errors that aren't our
                # synthetic RuntimeError. A single rate-limit on one symbol must
                # not crash the rebuild — the column lands as None for that symbol
                # and downstream events still build.
                print(
                    f"[event_dataset_builder] per-asset target fetch failed for "
                    f"{sym}: {type(exc).__name__}: {exc}; column "
                    f"{per_asset_target_column(sym)} will be None across all rows.",
                    file=sys.stderr,
                )

    credibility_kwargs = {
        "embedding_path": embedding_path,
        "stance_by_date": tuple(stance_by_date),
        "fred_cache_dir": fred_cache_dir,
        "fred_series_id": fred_series_id,
    }

    # Rates panel lookup (#291). Tests inject ``rates_lookup`` directly;
    # the pipeline path resolves the parquet at the supplied
    # ``rates_panel_path`` or under ``fred_cache_dir / 'rates_panel.parquet'``
    # by default, and degrades cleanly to an empty lookup when the file
    # is absent so a fresh checkout still produces an events.parquet.
    if rates_lookup is None:
        resolved_rates_path: Path | None = None
        if rates_panel_path is not None:
            resolved_rates_path = Path(rates_panel_path)
        elif fred_cache_dir is not None:
            resolved_rates_path = Path(fred_cache_dir) / "rates_panel.parquet"
        if resolved_rates_path is not None:
            rates_lookup = load_rates_panel_lookup(resolved_rates_path)

    # #443 strict-prior statement index. Built once over the
    # preferred-statement set so each event can read its own
    # immediately-prior statement text without a quadratic re-scan. The
    # selector asserts ``prior.event_date < this.event_date`` so a
    # same-date prior never enters the diff window.
    statement_text_index: list[tuple[str, str]] = sorted(
        (
            (str(d.event_date), d.text)
            for d in docs
            if d.event_kind == "statement" and d.text
        ),
        key=lambda pair: pair[0],
    )

    out_rows: list[dict[str, Any]] = []
    for doc in docs:
        prior_statement_text: str | None = None
        if doc.event_kind == "statement":
            prior_statement_text = select_prior_statement_text(
                event_date=str(doc.event_date),
                prior_index=statement_text_index,
            )
        rows = _build_event_rows(
            doc,
            asset=asset,
            benchmark=benchmark,
            asset_series=asset_series,
            bench_series=bench_series,
            horizons=horizons,
            credibility_kwargs=credibility_kwargs,
            macro_release_calendar=macro_release_calendar,
            concurrent_macro_window_days=concurrent_macro_window_days,
            intra_meeting_shift=intra_meeting_shifts.get(doc.event_date),
            cross_asset_lookup=cross_asset_lookup,
            cross_asset_series=cross_asset_series,
            rates_lookup=rates_lookup,
            rates_horizon=rates_horizon,
            prior_statement_text=prior_statement_text,
            delta_encoder=delta_encoder,
            per_asset_target_series=per_asset_target_series,
            prior_window_days=prior_window_days,
        )
        if not rows:
            summary.dropped_no_prior_window += 1
            continue
        summary.events_emitted += 1
        for r in rows:
            summary.per_source_rows[r["source"]] = summary.per_source_rows.get(r["source"], 0) + 1
            summary.per_kind_rows[r["event_kind"]] = summary.per_kind_rows.get(r["event_kind"], 0) + 1
            if r["concurrent_macro_release"]:
                summary.concurrent_macro_release_rows += 1
        out_rows.extend(rows)

    summary.rows_written = len(out_rows)
    if not out_rows:
        return _empty_frame()

    df = pd.DataFrame(out_rows)
    # Deterministic ordering. The full view sorts on source + source_record_id
    # so the parquet bytes don't shift when sources interleave.
    sort_cols = ["event_date", "event_kind", "source", "source_record_id", "asset_symbol", "horizon"]
    df = df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    df = df[list(COLUMN_ORDER)]
    return df


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in COLUMN_ORDER})


def write_events_parquet(df: pd.DataFrame, output_path: Path) -> None:
    """Write the events frame deterministically.

    Uses pandas + pyarrow with snappy compression (no creation-time
    metadata) and stable column order. Same input dataframe -> same bytes.
    Validates each row against ``EventRowSchema`` before writing so a
    contract violation raises at the write site rather than downstream.
    Set ``FED_PULSE_SKIP_SCHEMA_VALIDATION=1`` for diagnostic re-runs.
    """

    from app.data.schemas import EventRowSchema, validate_frame

    validate_frame(EventRowSchema, df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, engine="pyarrow", index=False, compression="snappy")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the event-row parquet for Phase 8 event-study forecasting."
    )
    parser.add_argument("--training-package-id", required=True)
    parser.add_argument("--asset", default=DEFAULT_ASSET)
    parser.add_argument(
        "--benchmark",
        default=DEFAULT_BENCHMARK,
        help="Market-model benchmark. Defaults to ^GSPC (forces beta=1, alpha=0 when asset==benchmark).",
    )
    parser.add_argument(
        "--horizons",
        default=",".join(str(h) for h in DEFAULT_HORIZONS),
        help="Comma-separated trading-day horizons (default: 1,5,10,30).",
    )
    parser.add_argument(
        "--output",
        default="events.parquet",
        help=(
            "Name of the collapsed parquet (one row per event_date x "
            "event_kind x asset_symbol x horizon). The full-source view is "
            "always written alongside as 'events_full.parquet'."
        ),
    )
    parser.add_argument(
        "--full-output",
        default="events_full.parquet",
        help=(
            "Name of the full parquet (keeps every source/source_record_id). "
            "Set to '' to skip the full view."
        ),
    )
    parser.add_argument(
        "--market-cache-dir",
        default=None,
        help="Where to cache the yfinance pull (default: <package_dir>/_market_cache).",
    )
    parser.add_argument(
        "--embedding-path",
        default=str(
            DEFAULT_DATA_DIR / "raw" / "embeddings" / "finbert_4556d1301521.parquet"
        ),
        help=(
            "Per-encoder embedding parquet for the credibility drift axis. "
            "Defaults to the cached FinBERT parquet so the builder is "
            "batteries-included; pass 'none' (or an empty string) to skip "
            "the drift computation. Non-'none' values must point at an "
            "existing file -- missing-file is a hard error."
        ),
    )
    parser.add_argument(
        "--fred-cache-dir",
        default=str(DEFAULT_DATA_DIR / "external" / "fred"),
        help="FRED cache directory for credibility realized-vs-stated gap.",
    )
    parser.add_argument(
        "--macro-release-csv",
        default=None,
        help=(
            "Override the macro-release calendar source (CSV or parquet). "
            "Defaults to data/external/macro_releases.csv when present, "
            "else the legacy rule-based heuristic."
        ),
    )
    parser.add_argument(
        "--concurrent-macro-window-days",
        type=int,
        default=CONCURRENT_MACRO_TRADING_DAY_RADIUS,
        help=(
            "Trading-day radius around each FOMC event to flag as "
            "concurrent_macro_release. Default 2 (matches the original heuristic "
            "and the real-calendar flag rate of ~49%). Set 1 for sharper "
            "attribution (~25%); set 0 to require exact same-day overlap."
        ),
    )
    parser.add_argument(
        "--rates-panel-path",
        default=None,
        help=(
            "Path to data/external/fred/rates_panel.parquet for the "
            "#291 rates-complex feature columns. Defaults to "
            "<fred-cache-dir>/rates_panel.parquet; a missing file "
            "degrades to null columns on every event row."
        ),
    )
    parser.add_argument(
        "--rates-horizon",
        type=int,
        default=5,
        help=(
            "Trading-day horizon for the strict-forward yield-change "
            "targets (default: 5)."
        ),
    )
    parser.add_argument(
        "--per-asset-target-cache-dir",
        default=None,
        help=(
            "Where to cache the per-asset forward-vol target price series "
            "(#481). Defaults to <DATA_DIR>/external/yfinance. One parquet "
            "per symbol; a missing/failing symbol collapses its column to "
            "None across all rows. The parquet filename uses yfinance's "
            "raw symbol (e.g. ``DX-Y.NYB.parquet``, ``EURUSD=X.parquet``), "
            "not the column slug on events.parquet (``dxy``, ``eurusd``) -- "
            "the slug normalisation lives downstream on the column write, "
            "not on the cache write, so the cache stays interoperable with "
            "any yfinance consumer that addresses by raw symbol."
        ),
    )
    parser.add_argument(
        "--per-asset-target-force-refresh",
        action="store_true",
        help=(
            "Bypass the cache for the per-asset forward-vol target "
            "series and force a fresh yfinance fetch on every symbol. "
            "Default off keeps the existing cache-once-pinned-forever "
            "behaviour. Use when the upstream series has been revised "
            "or when the cache parquet for one symbol is suspected "
            "corrupted (delete-the-parquet is the alternative)."
        ),
    )
    parser.add_argument(
        "--prior-window",
        type=int,
        default=PRIOR_WINDOW_DAYS,
        help=(
            "Number of trailing trading-day bars to attach as the "
            "prior_bars_json window (#209/#476). Defaults to 20 for "
            "back-compat with existing checkpoints; bump to 60 when "
            "rebuilding events.parquet for the extended-window sweep arm. "
            "Larger values shift walk-forward fold boundaries because the "
            "first usable event moves later in the calendar."
        ),
    )
    parser.add_argument(
        "--delta-encoder",
        default=None,
        help=(
            "Registry alias (or owner/repo HF id) of the encoder used to "
            "embed the #443 statement-delta spans (inserted / deleted / "
            "substituted) into the 768-d statement_delta_embedding column. "
            "Resolves through models/registry.yaml when an alias is given "
            "(picks up the pinned revision); a raw owner/repo id loads "
            "at HEAD. Without this flag the column stays None on every "
            "row and the loader collapses to the missing-1.0 slot."
        ),
    )
    return parser.parse_args(argv)


def _build_delta_encoder_callable(repo_or_alias: str):
    """Resolve ``repo_or_alias`` and return a ``Callable[[str], list[float]]``.

    Loads the encoder once on import, mean-pools its last-hidden-state
    over the non-pad tokens, and returns a 768-d list per call. CUDA
    used opportunistically. The pooled width is asserted against
    :data:`STATEMENT_DELTA_EMBEDDING_DIM` so a mismatched encoder fails
    at build time rather than silently writing odd-width rows.
    """

    import torch
    from transformers import AutoModel, AutoTokenizer

    from app.models.registry import encoder_ref

    ref = encoder_ref(repo_or_alias)
    repo = ref.repo if ref is not None else repo_or_alias
    revision = ref.revision if ref is not None else None

    tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision)
    model = AutoModel.from_pretrained(repo, revision=revision)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(
        f"[event-rows] delta encoder loaded: alias={repo_or_alias!r} "
        f"repo={repo!r} revision={revision!r} device={device}"
    )

    @torch.no_grad()
    def encode_text(text: str) -> list[float]:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=False,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        hidden = outputs.last_hidden_state  # (1, T, H)
        mask = inputs["attention_mask"].unsqueeze(-1).float()  # (1, T, 1)
        summed = (hidden * mask).sum(dim=1)  # (1, H)
        denom = mask.sum(dim=1).clamp(min=1.0)  # (1, 1)
        pooled = (summed / denom).squeeze(0)  # (H,)
        vec = pooled.detach().cpu().tolist()
        if len(vec) != STATEMENT_DELTA_EMBEDDING_DIM:
            raise SystemExit(
                f"--delta-encoder produced width {len(vec)} but the "
                f"statement_delta_embedding column expects "
                f"{STATEMENT_DELTA_EMBEDDING_DIM}. Pin a BERT-base "
                "encoder (e.g. finbert_fed_adjacent)."
            )
        return vec

    return encode_text


def _resolve_embedding_path(raw: str | None) -> Path | None:
    """Resolve the ``--embedding-path`` CLI value.

    Empty string or the literal ``"none"`` (case-insensitive) opts out of
    the drift axis. Any other value must exist on disk -- a typo that
    misses the cache parquet has to be a hard error so the
    silent-zero-credibility regression from pre-2026-05-17 cannot recur.
    """

    if raw is None:
        return None
    cleaned = str(raw).strip()
    if not cleaned or cleaned.lower() == "none":
        return None
    path = Path(cleaned)
    if not path.exists():
        raise SystemExit(
            f"--embedding-path points at a missing file: {path}. "
            "Pass 'none' to skip drift computation, or point at an existing "
            "data/raw/embeddings/<encoder>_<rev>.parquet."
        )
    return path


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    package_dir = DEFAULT_DATA_DIR / "processed" / args.training_package_id
    if not package_dir.exists():
        raise SystemExit(f"Training package not found: {package_dir}")

    horizons = tuple(int(h) for h in args.horizons.split(",") if h.strip())
    market_cache_dir = (
        Path(args.market_cache_dir)
        if args.market_cache_dir
        else (package_dir / "_market_cache")
    )

    macro_csv = Path(args.macro_release_csv) if args.macro_release_csv else None
    macro_calendar = _resolve_macro_calendar(macro_csv)

    per_asset_target_cache_dir = (
        Path(args.per_asset_target_cache_dir)
        if args.per_asset_target_cache_dir
        else None
    )

    delta_encoder = (
        _build_delta_encoder_callable(args.delta_encoder)
        if args.delta_encoder
        else None
    )

    # Collapsed view
    summary = _BuildSummary()
    df = build_event_rows(
        package_dir=package_dir,
        asset=args.asset,
        benchmark=args.benchmark,
        horizons=horizons,
        market_cache_dir=market_cache_dir,
        concurrent_macro_window_days=int(args.concurrent_macro_window_days),
        embedding_path=_resolve_embedding_path(args.embedding_path),
        fred_cache_dir=Path(args.fred_cache_dir) if args.fred_cache_dir else None,
        summary=summary,
        macro_release_calendar=macro_calendar,
        rates_panel_path=args.rates_panel_path,
        rates_horizon=int(args.rates_horizon),
        delta_encoder=delta_encoder,
        per_asset_target_cache_dir=per_asset_target_cache_dir,
        per_asset_target_force_refresh=bool(args.per_asset_target_force_refresh),
        prior_window_days=int(args.prior_window),
    )
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = package_dir / output_path
    write_events_parquet(df, output_path)

    print(f"[event-rows] collapsed view: {summary.rows_written} rows -> {output_path}")
    print(f"[event-rows] unique events: {summary.events_emitted}")
    print(f"[event-rows] dropped (no prior window or no targets): {summary.dropped_no_prior_window}")
    print(
        f"[event-rows] concurrent_macro_release rows: {summary.concurrent_macro_release_rows} "
        f"({_pct(summary.concurrent_macro_release_rows, summary.rows_written)})"
    )
    print(f"[event-rows] macro calendar source: {macro_calendar.source_label}")
    print("[event-rows] per-source breakdown (collapsed):")
    for src, count in sorted(summary.per_source_rows.items(), key=lambda x: -x[1]):
        print(f"  {src}: {count}")
    print("[event-rows] per-kind breakdown (collapsed):")
    for kind, count in sorted(summary.per_kind_rows.items(), key=lambda x: -x[1]):
        print(f"  {kind}: {count}")

    # Full view -- emit only when --full-output is set to a non-empty value.
    full_output_arg = (args.full_output or "").strip()
    if full_output_arg:
        full_summary = _BuildSummary()
        df_full = build_event_rows(
            package_dir=package_dir,
            asset=args.asset,
            benchmark=args.benchmark,
            horizons=horizons,
            market_cache_dir=market_cache_dir,
            concurrent_macro_window_days=int(args.concurrent_macro_window_days),
            embedding_path=_resolve_embedding_path(args.embedding_path),
            fred_cache_dir=Path(args.fred_cache_dir) if args.fred_cache_dir else None,
            summary=full_summary,
            macro_release_calendar=macro_calendar,
            keep_all_sources=True,
            rates_panel_path=args.rates_panel_path,
            rates_horizon=int(args.rates_horizon),
            delta_encoder=delta_encoder,
            per_asset_target_cache_dir=per_asset_target_cache_dir,
        per_asset_target_force_refresh=bool(args.per_asset_target_force_refresh),
            prior_window_days=int(args.prior_window),
        )
        full_output_path = Path(full_output_arg)
        if not full_output_path.is_absolute():
            full_output_path = package_dir / full_output_path
        write_events_parquet(df_full, full_output_path)
        print(
            f"[event-rows] full view: {full_summary.rows_written} rows -> {full_output_path}"
        )
        print(
            f"[event-rows] full unique (date x kind x source) docs: "
            f"{full_summary.events_emitted}"
        )
        print("[event-rows] per-source breakdown (full):")
        for src, count in sorted(full_summary.per_source_rows.items(), key=lambda x: -x[1]):
            print(f"  {src}: {count}")

    # Column / dtype summary so the smoke run is self-describing
    print("[event-rows] column dtypes (collapsed):")
    for col, dtype in df.dtypes.items():
        print(f"  {col}: {dtype}")
    return 0


def _pct(num: int, denom: int) -> str:
    if denom <= 0:
        return "0.00%"
    return f"{(num / denom) * 100:.2f}%"


if __name__ == "__main__":
    sys.exit(main())
