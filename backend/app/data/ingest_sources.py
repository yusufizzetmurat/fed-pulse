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

BACKEND_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = Path("/data") if Path("/data").exists() else BACKEND_ROOT.parent / "data"
DEFAULT_OUTPUT_DIR = DEFAULT_DATA_DIR / "raw" / "phase2"
DEFAULT_OUTPUT_FILE = "source_registry.jsonl"

HF_DATASET_ID = "gtfintechlab/fomc_communication"
KAGGLE_DATASET_ID = "drlexus/fed-statements-and-minutes"
SCRAPED_FILES = ("fomc_statements.json", "fomc_minutes.json")


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
    return "human" if label else "pseudo"


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
) -> dict[str, Any] | None:
    cleaned_text = _normalize_text(text)
    if not cleaned_text or not event_date:
        return None
    record_id = hashlib.sha256(f"{source}:{source_record_id}:{event_date}".encode("utf-8")).hexdigest()[:16]
    return {
        "record_id": record_id,
        "source": source,
        "source_record_id": source_record_id,
        "document_type": document_type or "unknown",
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
        payload = _read_json_or_jsonl(path)
        for idx, row in enumerate(payload):
            event_date = _coerce_str(row, ("date", "event_date", "published_date"))
            text = _coerce_str(row, ("text", "content"))
            label = _coerce_str(row, ("label_text", "label"))  # likely empty for scraped
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
            )
            if built:
                records.append(built)
    return records


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    by_source: dict[str, int] = {}
    labeled = 0
    for row in rows:
        by_source[row["source"]] = by_source.get(row["source"], 0) + 1
        if row.get("label"):
            labeled += 1
    payload = {
        "record_count": len(rows),
        "labeled_count": labeled,
        "source_counts": by_source,
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
    if not (include_hf or include_kaggle or include_scraped):
        print("No source selected. Use --all-sources or one of --include-hf/--include-kaggle/--include-scraped.")
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

    unified.sort(key=lambda row: (row.get("event_date", ""), row.get("source", ""), row.get("source_record_id", "")))
    _write_jsonl(output_path, unified)
    _write_summary(output_dir / "ingestion_summary.json", unified)
    print(f"Unified source registry written to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

