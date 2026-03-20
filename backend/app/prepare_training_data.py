from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from app.services.market_data import fetch_market_snapshot
from app.services.sentiment import analyze_text

DEFAULT_DATA_DIR = Path("/data") if Path("/data").exists() else Path(__file__).resolve().parents[2] / "data"
RAW_DOCUMENT_FILES = ("fomc_statements.json", "fomc_minutes.json")
DEFAULT_SYMBOLS = ("^GSPC", "^VIX", "DX-Y.NYB", "^TNX", "BTC-USD")
DEFAULT_OUTPUT_FILE = "train_dataset.json"


@dataclass
class PreparedTrainingRecord:
    date: str
    sentiment_score: float
    close: float
    volatility_5d: float
    symbol: str
    document_type: str
    title: str
    source_file: str


@dataclass
class PreparationSummary:
    total_documents: int
    valid_documents: int
    prepared_records: int
    skipped_documents: int
    skipped_symbol_pairs: int


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert raw scraped FOMC documents into market-aligned training records."
    )
    parser.add_argument(
        "--data-dir",
        default=str(DEFAULT_DATA_DIR),
        help="Directory containing raw scraped documents and output dataset.",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=list(DEFAULT_SYMBOLS),
        help="Symbol list to enrich for each document.",
    )
    parser.add_argument(
        "--output-file",
        default=DEFAULT_OUTPUT_FILE,
        help="Output JSON filename relative to the data directory.",
    )
    return parser.parse_args()


def _load_json_list(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{path.name} must contain a top-level JSON list.")
    return [item for item in payload if isinstance(item, dict)]


def load_raw_documents(data_dir: str | Path) -> list[dict[str, Any]]:
    root = Path(data_dir)
    documents: list[dict[str, Any]] = []

    for filename in RAW_DOCUMENT_FILES:
        path = root / filename
        if not path.exists():
            continue
        for record in _load_json_list(path):
            documents.append({**record, "_source_file": filename})

    documents.sort(key=lambda item: str(item.get("date", "")))
    return documents


def _iter_valid_documents(documents: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    for record in documents:
        date_value = str(record.get("date", "")).strip()
        text_value = str(record.get("text", "")).strip()
        if not date_value or not text_value:
            continue
        yield record


def build_training_groups(
    documents: Iterable[dict[str, Any]],
    symbols: list[str],
) -> tuple[list[dict[str, Any]], PreparationSummary]:
    grouped_records: dict[str, list[PreparedTrainingRecord]] = {symbol: [] for symbol in symbols}
    total_documents = 0
    valid_documents = 0
    skipped_documents = 0
    skipped_symbol_pairs = 0

    for document in documents:
        total_documents += 1
        date_value = str(document.get("date", "")).strip()
        text_value = str(document.get("text", "")).strip()
        if not date_value or not text_value:
            skipped_documents += 1
            continue

        try:
            sentiment = analyze_text(text_value)
        except Exception:
            skipped_documents += 1
            continue

        valid_documents += 1
        sentiment_score = float(sentiment["score"])
        title = str(document.get("title", ""))
        document_type = str(document.get("document_type", "Unknown"))
        source_file = str(document.get("_source_file", "unknown"))

        for symbol in symbols:
            try:
                market = fetch_market_snapshot(target_date=date_value, symbol=symbol)
            except Exception:
                skipped_symbol_pairs += 1
                continue
            grouped_records[symbol].append(
                PreparedTrainingRecord(
                    date=str(market["date_used"]),
                    sentiment_score=sentiment_score,
                    close=float(market["close"]),
                    volatility_5d=float(market["volatility_5d"]),
                    symbol=symbol,
                    document_type=document_type,
                    title=title,
                    source_file=source_file,
                )
            )

    groups: list[dict[str, Any]] = []
    for symbol, records in grouped_records.items():
        deduped: dict[str, PreparedTrainingRecord] = {}
        for record in records:
            deduped[record.date] = record
        ordered = [asdict(deduped[key]) for key in sorted(deduped)]
        groups.append({"symbol": symbol, "records": ordered})
    summary = PreparationSummary(
        total_documents=total_documents,
        valid_documents=valid_documents,
        prepared_records=sum(len(group["records"]) for group in groups),
        skipped_documents=skipped_documents,
        skipped_symbol_pairs=skipped_symbol_pairs,
    )
    return groups, summary


def prepare_training_dataset(
    *,
    data_dir: str | Path,
    symbols: list[str],
    output_file: str = DEFAULT_OUTPUT_FILE,
) -> Path:
    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)

    documents = load_raw_documents(root)
    groups, summary = build_training_groups(documents, symbols)
    output_path = root / output_file
    payload = {
        "groups": groups,
        "metadata": {
            "document_count": summary.valid_documents,
            "total_documents": summary.total_documents,
            "prepared_records": summary.prepared_records,
            "skipped_documents": summary.skipped_documents,
            "skipped_symbol_pairs": summary.skipped_symbol_pairs,
            "symbols": symbols,
            "source_files": [name for name in RAW_DOCUMENT_FILES if (root / name).exists()],
        },
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def main() -> int:
    args = _parse_args()
    data_dir = Path(args.data_dir)
    symbols = [str(symbol).strip() for symbol in args.symbols if str(symbol).strip()]
    if not symbols:
        print("No symbols provided.")
        return 1

    documents = load_raw_documents(data_dir)
    valid_documents = list(_iter_valid_documents(documents))
    print(f"Data directory: {data_dir}")
    print(f"Raw documents loaded: {len(documents)}")
    print(f"Valid documents for enrichment: {len(valid_documents)}")
    print(f"Symbols: {', '.join(symbols)}")

    if not valid_documents:
        print("No valid raw documents found. Ensure scraper output exists and includes date/text fields.")
        return 1

    output_path = prepare_training_dataset(
        data_dir=data_dir,
        symbols=symbols,
        output_file=args.output_file,
    )
    print(f"Prepared training dataset written to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
