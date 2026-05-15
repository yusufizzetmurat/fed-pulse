// Compare-page CSV: two runs side by side, plus the computed deltas.
//
// Layout: header row [field, run_a, run_b, delta]. Delta is filled when the
// computeCompareDelta result has a non-null value for that axis (close,
// volatility, sentiment score, multi-axis fields); cells the helper does
// not compute leave delta blank so the schema stays consistent.

import { computeCompareDelta, computeMultiAxisDelta } from "@/lib/analyze/compare";
import type { AnalyzeResult, HistoryDetail } from "@/lib/analyze/types";
import { downloadCsvBlob, toCsv, type CsvCell, type CsvRow } from "./csv";

function _result(detail: HistoryDetail): AnalyzeResult {
  return (detail.payload || {}) as AnalyzeResult;
}

function _row(field: string, a: CsvCell, b: CsvCell, delta: CsvCell = ""): CsvRow {
  return [field, a, b, delta];
}

export interface CompareCsv {
  filename: string;
  csv: string;
}

export function buildCompareRows(a: HistoryDetail, b: HistoryDetail): CsvRow[] {
  const ra = _result(a);
  const rb = _result(b);
  const delta = computeCompareDelta(a, b);
  const maxis = computeMultiAxisDelta(a, b);

  const rows: CsvRow[] = [["field", "run_a", "run_b", "delta_a_minus_b"]];
  rows.push(_row("run_id", a.id, b.id));
  rows.push(_row("created_at", a.created_at, b.created_at));
  rows.push(_row("symbol", a.symbol, b.symbol));
  rows.push(_row("document_date", a.document_date, b.document_date));
  rows.push(_row("horizon", a.horizon, b.horizon));
  rows.push(_row("forecast_mode", a.forecast_mode, b.forecast_mode));
  rows.push(_row(
    "sentiment.label",
    ra.sentiment?.label ?? null,
    rb.sentiment?.label ?? null,
  ));
  rows.push(_row(
    "sentiment.score",
    ra.sentiment?.score ?? null,
    rb.sentiment?.score ?? null,
    delta.scoreDelta,
  ));
  rows.push(_row(
    "prediction.close",
    ra.prediction?.close ?? null,
    rb.prediction?.close ?? null,
    delta.closeAbsolute,
  ));
  rows.push(_row(
    "prediction.close_percent",
    null,
    null,
    delta.closePercent,
  ));
  rows.push(_row(
    "prediction.volatility",
    ra.prediction?.volatility ?? null,
    rb.prediction?.volatility ?? null,
    delta.volatilityAbsolute,
  ));
  rows.push(_row(
    "stance.shift",
    a.stance,
    b.stance,
    delta.stanceShift,
  ));
  rows.push(_row(
    "multi_axis.stance.label",
    ra.multi_axis?.stance?.label ?? null,
    rb.multi_axis?.stance?.label ?? null,
    maxis.stanceRankDelta,
  ));
  rows.push(_row(
    "multi_axis.stance.confidence",
    ra.multi_axis?.stance?.confidence ?? null,
    rb.multi_axis?.stance?.confidence ?? null,
    maxis.stanceConfidenceDelta,
  ));
  rows.push(_row(
    "multi_axis.factor.value",
    ra.multi_axis?.factor?.value ?? null,
    rb.multi_axis?.factor?.value ?? null,
    maxis.factorDelta,
  ));
  rows.push(_row(
    "multi_axis.factor.confidence",
    ra.multi_axis?.factor?.confidence ?? null,
    rb.multi_axis?.factor?.confidence ?? null,
    maxis.factorConfidenceDelta,
  ));
  rows.push(_row(
    "multi_axis.certainty.label",
    ra.multi_axis?.certainty?.label ?? null,
    rb.multi_axis?.certainty?.label ?? null,
    maxis.certaintyShift,
  ));
  rows.push(_row(
    "multi_axis.certainty.confidence",
    ra.multi_axis?.certainty?.confidence ?? null,
    rb.multi_axis?.certainty?.confidence ?? null,
    maxis.certaintyConfidenceDelta,
  ));
  rows.push(_row(
    "multi_axis.topic.primary",
    ra.multi_axis?.topic?.primary ?? null,
    rb.multi_axis?.topic?.primary ?? null,
    maxis.topicChanged == null ? "" : maxis.topicChanged ? "changed" : "unchanged",
  ));
  return rows;
}

export function buildCompareCsv(a: HistoryDetail, b: HistoryDetail): CompareCsv {
  return {
    filename: `fed-pulse-compare-${a.id.slice(0, 8)}-vs-${b.id.slice(0, 8)}.csv`,
    csv: toCsv(buildCompareRows(a, b)),
  };
}

export function downloadCompareCsv(a: HistoryDetail, b: HistoryDetail): void {
  const { filename, csv } = buildCompareCsv(a, b);
  downloadCsvBlob(csv, filename);
}
