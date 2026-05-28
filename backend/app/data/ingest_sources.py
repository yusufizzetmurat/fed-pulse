from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from app.data.source_type import infer_source_type

from app.config import DATA_DIR as DEFAULT_DATA_DIR
DEFAULT_OUTPUT_DIR = DEFAULT_DATA_DIR / "raw" / "phase2"
DEFAULT_OUTPUT_FILE = "source_registry.jsonl"

HF_DATASET_ID = "gtfintechlab/fomc_communication"
GTFINTECHLAB_FED_DATASET_ID = "gtfintechlab/federal_reserve_system"
VTASCA_FOMC_ARCHIVE_DATASET_ID = "vtasca/fomc-statements-minutes"
KAGGLE_DATASET_ID = "drlexus/fed-statements-and-minutes"

# Pinned HF dataset revisions. record_id derives from
# sha256(source:source_record_id:event_date); upstream revision drift would
# otherwise rotate the entire hash chain. SHAs captured 2026-05-15.
_DATASET_REVISIONS: dict[str, str] = {
    "gtfintechlab/federal_reserve_system": "de0b1e8cb3a0fcfa601eec97d49d5c6f883804a1",
    "gtfintechlab/european_central_bank": "867cee85784ce569826e0104797b6e017205867b",
    "gtfintechlab/bank_of_japan": "1885e21cf1c33c4aea19a824ba40eac886c7a122",
    "gtfintechlab/bank_of_england": "de1123cf9d747dbb3e0c2224467f501692d5a310",
    "gtfintechlab/bank_of_canada": "ab15ea2271bfa3208874a5517afc439640fd9200",
    "gtfintechlab/reserve_bank_of_australia": "7a91206b56f2841b2586e409feade2518284894b",
    "vtasca/fomc-statements-minutes": "1d6c65eb96786ea921a29f4008c447f1cff5f7ff",
}


def _dataset_revision(dataset_id: str) -> str | None:
    """Return the pinned revision SHA for a dataset id, or None if unpinned."""
    return _DATASET_REVISIONS.get(dataset_id)
OP_FED_DEFAULT_RELATIVE = Path("external") / "op_fed" / "opfed_v1.csv"
GSS_FACTORS_DEFAULT_RELATIVE = Path("external") / "gss" / "gss_factors.csv"
GSS_SURPRISES_DEFAULT_RELATIVE = Path("external") / "gss" / "gss_surprises.csv"
SCRAPED_FILES = (
    "fomc_statements.json",
    "fomc_minutes.json",
    "chair_speeches.json",
    "governor_speeches.json",
    "congressional_testimonies.json",
    "press_conferences.json",
    "beige_book.json",
    "regional_research.json",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest approved Phase 2 text sources into a unified provenance registry."
    )
    parser.add_argument(
        "--data-dir",
        default=str(DEFAULT_DATA_DIR),
        help="Base data directory for scraped and exported artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where unified source registry artifacts are written.",
    )
    parser.add_argument(
        "--output-file",
        default=DEFAULT_OUTPUT_FILE,
        help="Output JSONL file name for unified source registry.",
    )
    parser.add_argument(
        "--include-hf",
        action="store_true",
        help=f"Ingest Hugging Face dataset: {HF_DATASET_ID}",
    )
    parser.add_argument(
        "--include-kaggle",
        action="store_true",
        help=f"Ingest Kaggle dataset: {KAGGLE_DATASET_ID}",
    )
    parser.add_argument(
        "--include-scraped",
        action="store_true",
        help="Ingest local scraped files (fomc_statements.json, fomc_minutes.json).",
    )
    parser.add_argument(
        "--include-op-fed",
        action="store_true",
        help="Ingest Op-Fed sentence-level stance + multi-axis annotations (Keith et al. 2025, MIT).",
    )
    parser.add_argument(
        "--include-gss-factors",
        action="store_true",
        help="Ingest GSS (Gürkaynak-Sack-Swanson 2005 IJCB) per-FOMC target/path factor decomposition and 30min/1hr/1day surprise windows.",
    )
    parser.add_argument(
        "--include-gtfintechlab-fed",
        action="store_true",
        help=f"Ingest {GTFINTECHLAB_FED_DATASET_ID}: 3,000 multi-axis FOMC sentence labels (stance + time + certainty).",
    )
    parser.add_argument(
        "--include-gtfintechlab-cross-bank",
        action="store_true",
        help=(
            "Ingest gtfintechlab cross-bank datasets (ECB / BoJ / BoE / BoC / RBA): "
            "~15,000 multi-axis sentences held out from FOMC training (sample_weight=0) "
            "for the cross-CB generalization study."
        ),
    )
    parser.add_argument(
        "--include-fomc-archive",
        action="store_true",
        help=f"Ingest {VTASCA_FOMC_ARCHIVE_DATASET_ID}: full FOMC statement + minutes archive (unlabelled, for credibility drift).",
    )
    parser.add_argument(
        "--all-sources",
        action="store_true",
        help="Ingest all configured sources.",
    )
    return parser.parse_args()


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def _text_hash(text: str) -> str:
    return hashlib.sha256(_normalize_text(text).encode("utf-8")).hexdigest()


def _coerce_str(record: dict[str, Any], keys: Iterable[str]) -> str:
    for key in keys:
        value = record.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _coerce_event_date(record: dict[str, Any], keys: Iterable[str]) -> str:
    raw = _coerce_str(record, keys)
    if not raw:
        year_value = _coerce_str(record, ("year",))
        if year_value.isdigit() and len(year_value) == 4:
            return f"{year_value}-01-01"
        return ""

    digits = re.sub(r"[^0-9]", "", raw)
    if len(digits) == 8:
        return f"{digits[0:4]}-{digits[4:6]}-{digits[6:8]}"
    if len(digits) == 4:
        return f"{digits}-01-01"
    return raw


def _map_kaggle_document_type(raw_type: str) -> str:
    cleaned = _normalize_text(raw_type).lower()
    if cleaned in {"0", "minutes"}:
        return "minutes"
    if cleaned in {"1", "statement"}:
        return "statement"
    return cleaned or "unknown"


def _coerce_label_origin(label: str) -> str:
    # Audit Tier 1.6: previous behaviour coerced empty labels to
    # ``"pseudo"`` -- the same value emitted for actual teacher-model
    # pseudo-labels. Downstream filters that select for ``human`` (or
    # exclude ``pseudo``) therefore silently mixed unlabeled rows with
    # genuine teacher predictions. Emit ``"unlabeled"`` so the three
    # cases stay distinguishable end-to-end. Whitespace-only labels
    # are treated as empty.
    if not label or not str(label).strip():
        return "unlabeled"
    return "human"


def _build_registry_record(
    *,
    source: str,
    source_record_id: str,
    event_date: str,
    document_type: str,
    title: str,
    text: str,
    label: str,
    license_scope: str,
    citation_ref: str,
    source_type: str | None = None,
) -> dict[str, Any] | None:
    cleaned_text = _normalize_text(text)
    if not cleaned_text or not event_date:
        return None
    record_id = hashlib.sha256(f"{source}:{source_record_id}:{event_date}".encode("utf-8")).hexdigest()[:16]
    resolved_source_type = source_type or infer_source_type(
        document_type=document_type, title=title
    )
    return {
        "record_id": record_id,
        "source": source,
        "source_record_id": source_record_id,
        "document_type": document_type or "unknown",
        "source_type": resolved_source_type,
        "event_date": event_date,
        "title": title,
        "text": cleaned_text,
        "label": label,
        "label_origin": _coerce_label_origin(label),
        "license_scope": license_scope,
        "citation_ref": citation_ref,
        "ingested_at_utc": datetime.now(timezone.utc).isoformat(),
        "text_hash": _text_hash(cleaned_text),
    }


def _iter_hf_records() -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "datasets package is required for --include-hf. Install dependencies first."
        ) from exc

    ds = load_dataset(HF_DATASET_ID)
    records: list[dict[str, Any]] = []

    for split_name, split in ds.items():
        for idx, row in enumerate(split):
            item = dict(row)
            event_date = _coerce_event_date(item, ("date", "event_date", "published_date", "timestamp", "year"))
            text = _coerce_str(item, ("text", "sentence", "content", "document", "statement"))
            label = _coerce_str(item, ("label_text", "label", "stance", "class"))
            title = _coerce_str(item, ("title", "headline"))
            document_type = _coerce_str(item, ("document_type", "type")) or "statement"
            source_record_id = _coerce_str(item, ("id", "uid", "record_id")) or f"{split_name}:{idx}"
            built = _build_registry_record(
                source="hf_fomc_communication",
                source_record_id=source_record_id,
                event_date=event_date,
                document_type=document_type,
                title=title,
                text=text,
                label=label,
                license_scope="research_only",
                citation_ref="shah_etal_2023_trillion_dollar_words",
            )
            if built:
                records.append(built)
    return records


_OP_FED_STANCE_MAP = {
    "entailment": "hawkish",
    "contradiction": "dovish",
    "neutral": "neutral",
}

_GTFINTECHLAB_STANCE_MAP = {
    "hawkish": "hawkish",
    "dovish": "dovish",
    "neutral": "neutral",
}


# Cross-bank gtfintechlab datasets — same 3,000-row {sentences, stance_label,
# time_label, certain_label, year} schema as federal_reserve_system. Held out
# from the FOMC headline training pool via provenance="peer_reviewed_cross_bank"
# (sample_weight 0.0) so they only contribute to the cross-bank generalization
# evaluation.
GTFINTECHLAB_CROSS_BANK_DATASETS: tuple[tuple[str, str, str], ...] = (
    # (bank_key, hf_dataset_id, document_type_hint)
    ("european_central_bank", "gtfintechlab/european_central_bank", "ecb_communication"),
    ("bank_of_japan", "gtfintechlab/bank_of_japan", "boj_communication"),
    ("bank_of_england", "gtfintechlab/bank_of_england", "boe_communication"),
    ("bank_of_canada", "gtfintechlab/bank_of_canada", "boc_communication"),
    ("reserve_bank_of_australia", "gtfintechlab/reserve_bank_of_australia", "rba_communication"),
)


def _iter_gtfintechlab_records(
    *,
    dataset_id: str,
    source_name: str,
    provenance: str,
    document_type: str,
    title_prefix: str,
    citation_ref: str = "shah_etal_2024_gtfintechlab_central_banks",
    license_scope: str = "research_only",
) -> list[dict[str, Any]]:
    """Generic loader for the gtfintechlab multi-axis schema.

    Every dataset under the gtfintechlab umbrella that hosts central-bank
    sentence annotations shares the row shape ``sentences, stance_label,
    time_label, certain_label, year``. This function walks every config /
    split combination, normalises stance, populates ``multi_axis_extras``
    with the time + certainty axes, and dedupes by ``text_hash``.

    Reproducibility: pins the dataset revision from ``_DATASET_REVISIONS`` so
    upstream HF pushes can't rotate row indices, and derives
    ``source_record_id`` from the text hash (not the iterator's positional
    index) so insertions/deletions in the dataset do not change ``record_id``.
    """
    try:
        from datasets import (  # type: ignore
            get_dataset_config_names,
            get_dataset_split_names,
            load_dataset,
        )
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "datasets package is required for the gtfintechlab loader. "
            "Install dependencies first."
        ) from exc

    revision = _dataset_revision(dataset_id)
    records: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()

    configs = list(get_dataset_config_names(dataset_id, revision=revision))
    for config in configs:
        splits = list(get_dataset_split_names(dataset_id, config, revision=revision))
        for split in splits:
            ds = load_dataset(dataset_id, config, split=split, revision=revision)
            for row in ds:
                item = dict(row)
                sentence = (item.get("sentences") or "").strip()
                if not sentence:
                    continue
                stance_raw = (item.get("stance_label") or "").strip().lower()
                label = _GTFINTECHLAB_STANCE_MAP.get(stance_raw, "")
                year_value = item.get("year")
                try:
                    event_date = (
                        f"{int(year_value):04d}-01-01" if year_value not in (None, "") else ""
                    )
                except (TypeError, ValueError):
                    event_date = ""
                if not event_date:
                    continue

                normalized_text = _normalize_text(sentence)
                content_hash = _text_hash(normalized_text)
                if content_hash in seen_hashes:
                    continue

                built = _build_registry_record(
                    source=source_name,
                    source_record_id=content_hash[:16],
                    event_date=event_date,
                    document_type=document_type,
                    title=f"{title_prefix} sentence {content_hash[:8]}",
                    text=sentence,
                    label=label,
                    license_scope=license_scope,
                    citation_ref=citation_ref,
                )
                if built is None:
                    continue
                seen_hashes.add(content_hash)
                built["provenance"] = provenance
                extras = {
                    "gtfintechlab_time_label": (item.get("time_label") or "").strip(),
                    "gtfintechlab_certain_label": (item.get("certain_label") or "").strip(),
                    "gtfintechlab_config": str(config),
                    "gtfintechlab_split": str(split),
                    "gtfintechlab_dataset_revision": revision or "",
                }
                built["multi_axis_extras"] = {k: v for k, v in extras.items() if v}
                records.append(built)
    return records


def _iter_gtfintechlab_federal_reserve_records() -> list[dict[str, Any]]:
    """Load gtfintechlab/federal_reserve_system into the FOMC training pool."""
    return _iter_gtfintechlab_records(
        dataset_id=GTFINTECHLAB_FED_DATASET_ID,
        source_name="gtfintechlab_federal_reserve_system",
        provenance="peer_reviewed",
        document_type="statement",
        title_prefix="Federal Reserve System",
    )


def _iter_gtfintechlab_cross_bank_records() -> list[dict[str, Any]]:
    """Load every cross-bank gtfintechlab dataset into the cross-bank generalization pool.

    Rows carry ``provenance="peer_reviewed_cross_bank"`` (weight 0.0) so they
    are visible in the source registry but excluded from the supervised
    training loss. The cross-bank evaluation harness opts them in explicitly.
    """
    records: list[dict[str, Any]] = []
    for bank_key, dataset_id, document_type in GTFINTECHLAB_CROSS_BANK_DATASETS:
        bank_records = _iter_gtfintechlab_records(
            dataset_id=dataset_id,
            source_name=f"gtfintechlab_{bank_key}",
            provenance="peer_reviewed_cross_bank",
            document_type=document_type,
            title_prefix=bank_key.replace("_", " ").title(),
        )
        records.extend(bank_records)
    return records


_FOMC_ARCHIVE_TYPE_MAP = {
    "statement": "statement",
    "minutes": "minutes",
    "minute": "minutes",
}


def _iter_fomc_archive_records() -> list[dict[str, Any]]:
    """Load vtasca/fomc-statements-minutes: 463 whole-document FOMC texts.

    Schema per row: ``Date, Release Date, Type, Text``. Rows are *unlabelled*
    — they feed the credibility module (drift of one statement vs the prior
    four meetings) and supplement the continued-pretraining substrate. Routed
    through ``provenance="scraped"`` so they receive ``sample_weight=0`` and
    do not enter the supervised training pool.

    Reproducibility: pins the dataset revision and discriminates the
    ``source_record_id`` with the text hash so corrected-release variants
    (same date + document_type but different text) do not collide.
    """
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "datasets package is required for --include-fomc-archive. Install dependencies first."
        ) from exc

    revision = _dataset_revision(VTASCA_FOMC_ARCHIVE_DATASET_ID)
    ds = load_dataset(VTASCA_FOMC_ARCHIVE_DATASET_ID, split="train", revision=revision)
    records: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()

    for row in ds:
        item = dict(row)
        raw_type = (item.get("Type") or "").strip().lower()
        document_type = _FOMC_ARCHIVE_TYPE_MAP.get(raw_type, "")
        if not document_type:
            continue
        event_date = _coerce_event_date(item, ("Date", "Release Date"))
        if not event_date:
            continue
        text = (item.get("Text") or "").strip()
        if not text:
            continue
        release_date = _coerce_event_date(item, ("Release Date",))

        normalized_text = _normalize_text(text)
        content_hash = _text_hash(normalized_text)
        if content_hash in seen_hashes:
            continue

        built = _build_registry_record(
            source="vtasca_fomc_archive",
            source_record_id=f"{event_date}:{document_type}:{content_hash[:8]}",
            event_date=event_date,
            document_type=document_type,
            title=f"FOMC {document_type} {event_date}",
            text=text,
            label="",
            license_scope="public_source_scrape_terms_required",
            citation_ref="vtasca_2024_fomc_statements_minutes",
        )
        if built is None:
            continue
        seen_hashes.add(content_hash)
        built["provenance"] = "scraped"
        extras: dict[str, str] = {}
        if release_date and release_date != event_date:
            extras["release_date"] = release_date
        if revision:
            extras["vtasca_dataset_revision"] = revision
        if extras:
            built["multi_axis_extras"] = extras
        records.append(built)
    return records


def _iter_op_fed_records(csv_path: Path) -> list[dict[str, Any]]:
    if not csv_path.exists():
        warnings.warn(f"Op-Fed CSV not found at {csv_path}; skipping.", stacklevel=2)
        return []

    records: list[dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            unique_id = (row.get("unique_id") or "").strip()
            if not unique_id:
                continue
            date_part = unique_id.split("_", 1)[0]
            event_date = _coerce_event_date({"date": date_part}, ("date",))
            if not event_date:
                continue

            sentence = (row.get("sentence") or "").strip().strip('"')
            if not sentence:
                continue

            stance_raw = (row.get("4_stance_nli") or "").strip().lower()
            label = _OP_FED_STANCE_MAP.get(stance_raw, "")

            title_speaker = (row.get("speaker") or "").strip()
            title = f"FOMC meeting transcript {date_part} — {title_speaker}".strip(" —")

            built = _build_registry_record(
                source="op_fed",
                source_record_id=unique_id,
                event_date=event_date,
                document_type="meeting_transcript",
                title=title,
                text=sentence,
                label=label,
                license_scope="mit",
                citation_ref="keith_etal_2025_op_fed",
                source_type="fomc_meeting_transcript",
            )
            if built is None:
                continue
            built["provenance"] = "peer_reviewed"
            extra = {
                "op_fed_opinion": (row.get("1_opinion") or "").strip(),
                "op_fed_mp": (row.get("2_mp") or "").strip(),
                "op_fed_mp_context": (row.get("3_mp_context") or "").strip(),
                "op_fed_stance_nli": stance_raw,
                "op_fed_stance_nli_context": (row.get("5_stance_nli_context") or "").strip(),
            }
            built["multi_axis_extras"] = {k: v for k, v in extra.items() if v}
            records.append(built)
    return records


def _iter_gss_factors_records(
    factors_csv: Path,
    surprises_csv: Path | None = None,
) -> list[dict[str, Any]]:
    """Load the GSS 2005 (Gürkaynak-Sack-Swanson) per-FOMC factor decomposition.

    Each FOMC meeting becomes one registry row with the target / path factors
    and (when ``surprises_csv`` is also present) the 30-min / 1-hour / 1-day
    monetary-policy-surprise windows on ``multi_axis_extras``. Rows are
    stance-unlabelled — the factor decomposition is continuous, not
    categorical — so they populate the factor axis of the multi-axis schema
    without polluting the hawkish/dovish/neutral training pool.
    """

    if not factors_csv.exists():
        warnings.warn(f"GSS factors CSV not found at {factors_csv}; skipping.", stacklevel=2)
        return []

    surprises_by_date: dict[str, dict[str, Any]] = {}
    if surprises_csv is not None and surprises_csv.exists():
        with surprises_csv.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                event_date = _coerce_event_date(row, ("meeting_date", "date", "event_date"))
                if not event_date:
                    continue
                extras: dict[str, Any] = {}
                for key in (
                    "surprise_30min_bp",
                    "surprise_1hour_bp",
                    "surprise_1day_bp",
                    "diff_wide_minus_tight",
                    "diff_daily_minus_tight",
                ):
                    raw = (row.get(key) or "").strip()
                    if not raw:
                        continue
                    try:
                        extras[key] = float(raw)
                    except ValueError:
                        continue
                if extras:
                    surprises_by_date[event_date] = extras

    records: list[dict[str, Any]] = []
    with factors_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            event_date = _coerce_event_date(row, ("meeting_date", "date", "event_date"))
            if not event_date:
                continue
            target_raw = (row.get("target_factor") or "").strip()
            path_raw = (row.get("path_factor") or "").strip()
            if not target_raw and not path_raw:
                continue
            extras: dict[str, Any] = {}
            try:
                extras["gss_target_factor"] = float(target_raw) if target_raw else None
            except ValueError:
                extras["gss_target_factor"] = None
            try:
                extras["gss_path_factor"] = float(path_raw) if path_raw else None
            except ValueError:
                extras["gss_path_factor"] = None
            statement_flag = (row.get("fomc_statement") or "").strip().upper() == "T"
            extras["gss_fomc_statement"] = statement_flag
            extras.update(surprises_by_date.get(event_date, {}))

            target_repr = (
                f"{extras['gss_target_factor']:+.2f}"
                if extras.get("gss_target_factor") is not None
                else "n/a"
            )
            path_repr = (
                f"{extras['gss_path_factor']:+.2f}"
                if extras.get("gss_path_factor") is not None
                else "n/a"
            )
            text = (
                f"GSS factor decomposition for {event_date}: "
                f"target={target_repr} bp, path={path_repr} bp"
            )

            built = _build_registry_record(
                source="gss_factor",
                source_record_id=f"gss_{event_date}",
                event_date=event_date,
                document_type="statement",
                title=f"GSS target/path factors {event_date}",
                text=text,
                label="",  # factor axis is continuous; no categorical stance label
                license_scope="research_only",
                citation_ref="gurkaynak_sack_swanson_2005_ijcb",
                source_type="gss_factor_decomposition",
            )
            if built is None:
                continue
            built["provenance"] = "peer_reviewed"
            built["multi_axis_extras"] = extras
            records.append(built)
    return records


def _read_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if path.suffix == ".jsonl":
        output: list[dict[str, Any]] = []
        for line in text.splitlines():
            payload = json.loads(line)
            if isinstance(payload, dict):
                output.append(payload)
        return output
    payload = json.loads(text)
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if isinstance(payload, dict):
        for key in ("records", "rows", "data", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                return [x for x in value if isinstance(x, dict)]
    return []


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _iter_candidate_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix in {".json", ".jsonl"}:
        return _read_json_or_jsonl(path)
    if path.suffix == ".csv":
        return _read_csv(path)
    return []


def _iter_kaggle_records() -> list[dict[str, Any]]:
    try:
        import kagglehub  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "kagglehub package is required for --include-kaggle. Install dependencies first."
        ) from exc

    dataset_path = Path(kagglehub.dataset_download(KAGGLE_DATASET_ID))
    records: list[dict[str, Any]] = []
    for file_path in sorted(dataset_path.rglob("*")):
        if not file_path.is_file():
            continue
        rows = _iter_candidate_records(file_path)
        for idx, row in enumerate(rows):
            event_date = _coerce_event_date(row, ("date", "event_date", "published_date", "timestamp", "Date"))
            text = _coerce_str(row, ("text", "Text", "content", "statement", "minutes"))
            label = _coerce_str(row, ("label_text", "label", "stance", "class"))
            title = _coerce_str(row, ("title", "headline"))
            document_type = _map_kaggle_document_type(_coerce_str(row, ("document_type", "type", "Type")))
            source_record_id = _coerce_str(row, ("id", "uid", "record_id")) or f"{file_path.name}:{idx}"
            built = _build_registry_record(
                source="kaggle_fed_statements_minutes",
                source_record_id=source_record_id,
                event_date=event_date,
                document_type=document_type,
                title=title,
                text=text,
                label=label,
                license_scope="source_terms_required",
                citation_ref="kaggle_drlexus_fed_statements_and_minutes",
            )
            if built:
                records.append(built)
    return records


def _iter_scraped_records(data_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for filename in SCRAPED_FILES:
        path = data_dir / filename
        if not path.exists():
            continue
        # Filename is the authoritative signal for legacy scraped files;
        # the row-level document_type is sometimes missing.
        if filename == "fomc_minutes.json":
            file_source_type = "fomc_minutes"
        elif filename == "fomc_statements.json":
            file_source_type = "fomc_statement"
        elif filename == "chair_speeches.json":
            file_source_type = "chair_speech"
        elif filename == "governor_speeches.json":
            file_source_type = "governor_speech"
        elif filename == "congressional_testimonies.json":
            file_source_type = "congressional_testimony"
        elif filename == "press_conferences.json":
            file_source_type = "press_conference"
        elif filename == "beige_book.json":
            file_source_type = "beige_book"
        elif filename == "regional_research.json":
            file_source_type = "regional_research"
        else:
            file_source_type = None
        payload = _read_json_or_jsonl(path)
        for idx, row in enumerate(payload):
            event_date = _coerce_str(row, ("date", "event_date", "published_date"))
            text = _coerce_str(row, ("text", "content"))
            label = _coerce_str(row, ("label_text", "label"))
            title = _coerce_str(row, ("title",))
            document_type = _coerce_str(row, ("document_type", "type")) or "unknown"
            source_record_id = _coerce_str(row, ("id", "uid", "record_id")) or f"{filename}:{idx}"
            built = _build_registry_record(
                source="scraped_fed",
                source_record_id=source_record_id,
                event_date=event_date,
                document_type=document_type,
                title=title,
                text=text,
                label=label,
                license_scope="public_source_scrape_terms_required",
                citation_ref="federalreserve_primary_source",
                source_type=file_source_type,
            )
            if built:
                records.append(built)
    return records


def _validate_ingested_rows(rows: list[dict[str, Any]]) -> None:
    """Run ``IngestedDocSchema`` on the unified registry frame.

    Pandera reports every column / row violation in a single
    ``SchemaErrors`` so the ingestion stage halts before any downstream
    consumer reads malformed JSONL. The ``FED_PULSE_SKIP_SCHEMA_VALIDATION``
    env var bypasses validation for diagnostic re-runs.
    """

    if not rows:
        return
    try:
        import pandas as pd  # type: ignore
    except Exception:
        return
    from app.data.schemas import IngestedDocSchema, validate_frame

    frame = pd.DataFrame(rows)
    validate_frame(IngestedDocSchema, frame)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    _validate_ingested_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    by_source: dict[str, int] = {}
    by_source_type: dict[str, int] = {}
    labeled = 0
    for row in rows:
        by_source[row["source"]] = by_source.get(row["source"], 0) + 1
        st = str(row.get("source_type", ""))
        if st:
            by_source_type[st] = by_source_type.get(st, 0) + 1
        if row.get("label"):
            labeled += 1
    payload = {
        "record_count": len(rows),
        "labeled_count": labeled,
        "source_counts": by_source,
        "source_type_counts": by_source_type,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    warnings.warn(
        "app.data.ingest_sources is deprecated. Use app.data.source_ingestion instead.",
        DeprecationWarning,
        stacklevel=1,
    )
    args = _parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_path = output_dir / args.output_file

    include_hf = args.all_sources or args.include_hf
    include_kaggle = args.all_sources or args.include_kaggle
    include_scraped = args.all_sources or args.include_scraped
    include_op_fed = args.all_sources or args.include_op_fed
    include_gss_factors = args.all_sources or args.include_gss_factors
    include_gtfintechlab_fed = args.all_sources or args.include_gtfintechlab_fed
    include_gtfintechlab_cross_bank = args.all_sources or args.include_gtfintechlab_cross_bank
    include_fomc_archive = args.all_sources or args.include_fomc_archive
    if not (
        include_hf
        or include_kaggle
        or include_scraped
        or include_op_fed
        or include_gss_factors
        or include_gtfintechlab_fed
        or include_gtfintechlab_cross_bank
        or include_fomc_archive
    ):
        print(
            "No source selected. Use --all-sources or one of "
            "--include-hf/--include-kaggle/--include-scraped/--include-op-fed/"
            "--include-gss-factors/--include-gtfintechlab-fed/"
            "--include-gtfintechlab-cross-bank/--include-fomc-archive."
        )
        return 1

    unified: list[dict[str, Any]] = []
    if include_hf:
        hf_records = _iter_hf_records()
        print(f"Ingested Hugging Face records: {len(hf_records)}")
        unified.extend(hf_records)
    if include_kaggle:
        kaggle_records = _iter_kaggle_records()
        print(f"Ingested Kaggle records: {len(kaggle_records)}")
        unified.extend(kaggle_records)
    if include_scraped:
        scraped_records = _iter_scraped_records(data_dir)
        print(f"Ingested scraped records: {len(scraped_records)}")
        unified.extend(scraped_records)
    if include_op_fed:
        op_fed_records = _iter_op_fed_records(data_dir / OP_FED_DEFAULT_RELATIVE)
        labelled = sum(1 for r in op_fed_records if r.get("label"))
        print(f"Ingested Op-Fed records: {len(op_fed_records)} (stance-labelled: {labelled})")
        unified.extend(op_fed_records)
    if include_gss_factors:
        gss_records = _iter_gss_factors_records(
            data_dir / GSS_FACTORS_DEFAULT_RELATIVE,
            data_dir / GSS_SURPRISES_DEFAULT_RELATIVE,
        )
        print(f"Ingested GSS factor records: {len(gss_records)} (per-FOMC target/path factors; factor axis only)")
        unified.extend(gss_records)
    if include_gtfintechlab_fed:
        gtfintechlab_records = _iter_gtfintechlab_federal_reserve_records()
        labelled = sum(1 for r in gtfintechlab_records if r.get("label"))
        print(
            f"Ingested gtfintechlab/federal_reserve_system records: {len(gtfintechlab_records)} "
            f"(stance-labelled: {labelled}; multi-axis time+certainty in multi_axis_extras)"
        )
        unified.extend(gtfintechlab_records)
    if include_gtfintechlab_cross_bank:
        cross_bank_records = _iter_gtfintechlab_cross_bank_records()
        labelled = sum(1 for r in cross_bank_records if r.get("label"))
        per_bank: dict[str, int] = {}
        for record in cross_bank_records:
            per_bank[record.get("source", "")] = per_bank.get(record.get("source", ""), 0) + 1
        print(
            f"Ingested gtfintechlab cross-bank records: {len(cross_bank_records)} "
            f"(stance-labelled: {labelled}; sample_weight=0; banks={per_bank})"
        )
        unified.extend(cross_bank_records)
    if include_fomc_archive:
        archive_records = _iter_fomc_archive_records()
        statement_count = sum(1 for r in archive_records if r.get("document_type") == "statement")
        minutes_count = sum(1 for r in archive_records if r.get("document_type") == "minutes")
        print(
            f"Ingested vtasca/fomc-statements-minutes records: {len(archive_records)} "
            f"(statements: {statement_count}, minutes: {minutes_count}; unlabelled, credibility-only)"
        )
        unified.extend(archive_records)

    unified.sort(key=lambda row: (row.get("event_date", ""), row.get("source", ""), row.get("source_record_id", "")))
    _write_jsonl(output_path, unified)
    _write_summary(output_dir / "ingestion_summary.json", unified)
    print(f"Unified source registry written to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

