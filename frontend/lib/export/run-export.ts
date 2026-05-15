// Per-run CSV export emitted from the history detail page.
//
// Schema is deliberately flat — one row per metric. The reader is expected
// to be Excel / Sheets / pandas; we do not emit nested objects (forecast
// series, attribution arrays). For the forecast series the caller can drop
// down to `buildForecastSeriesCsv` and emit it as a second download if a
// user asks for the chart data.

import type { AnalyzeResult, HistoryDetail } from "@/lib/analyze/types";
import { downloadCsvBlob, toCsv, type CsvRow } from "./csv";

function _result(detail: HistoryDetail): AnalyzeResult {
  return (detail.payload || {}) as AnalyzeResult;
}

export interface RunSummaryCsv {
  filename: string;
  csv: string;
}

// Header row first, then one (field, value) row per metric the dashboard
// surfaces. Fields are added even when the underlying value is null so the
// schema is stable across runs (you can diff two CSVs and trust the row
// order matches).
export function buildRunSummaryRows(detail: HistoryDetail): CsvRow[] {
  const result = _result(detail);
  const sentiment = result.sentiment || {};
  const prediction = result.prediction || {};
  const market = result.market || {};
  const model = result.model || {};
  const stance = result.multi_axis?.stance;
  const factor = result.multi_axis?.factor;
  const certainty = result.multi_axis?.certainty;
  const topic = result.multi_axis?.topic;
  const credibility = result.credibility;

  return [
    ["field", "value"],
    ["run_id", detail.id],
    ["created_at", detail.created_at],
    ["symbol", detail.symbol],
    ["document_date", detail.document_date],
    ["horizon", detail.horizon],
    ["forecast_mode", detail.forecast_mode],
    ["sentiment.label", sentiment.label ?? null],
    ["sentiment.score", sentiment.score ?? null],
    ["sentiment.ood_energy", sentiment.ood_energy ?? null],
    ["sentiment.is_in_distribution", sentiment.is_in_distribution ?? null],
    ["prediction.close", prediction.close ?? null],
    ["prediction.volatility", prediction.volatility ?? null],
    ["prediction.horizon", prediction.horizon ?? null],
    ["market.symbol", market.symbol ?? null],
    ["market.requested_date", market.requested_date ?? null],
    ["market.date_used", market.date_used ?? null],
    ["market.close", market.close ?? null],
    ["market.volatility_5d", market.volatility_5d ?? null],
    ["model.checkpoint_loaded", model.checkpoint_loaded ?? null],
    ["model.runtime_mode", model.runtime_mode ?? null],
    ["model.combined_rmse", model.combined_rmse ?? null],
    ["multi_axis.stance.label", stance?.label ?? null],
    ["multi_axis.stance.confidence", stance?.confidence ?? null],
    ["multi_axis.factor.value", factor?.value ?? null],
    ["multi_axis.factor.confidence", factor?.confidence ?? null],
    ["multi_axis.certainty.label", certainty?.label ?? null],
    ["multi_axis.certainty.confidence", certainty?.confidence ?? null],
    ["multi_axis.topic.primary", topic?.primary ?? null],
    ["multi_axis.topic.confidence", topic?.confidence ?? null],
    ["credibility.drift_score", credibility?.drift_score ?? null],
    ["credibility.realized_vs_stated_gap", credibility?.realized_vs_stated_gap ?? null],
    ["credibility.market_implied_gap", credibility?.market_implied_gap ?? null],
    ["credibility.months_since_reversal", credibility?.months_since_reversal ?? null],
    ["series.forecast_band_source", result.series?.forecast_band_source ?? null],
    ["series.forecast_confidence_level", result.series?.forecast_confidence_level ?? null],
    ["series.conformal_coverage", result.series?.conformal_coverage ?? null],
  ];
}

// One row per forecast timestep with the close/volatility forecast plus
// upper/lower band edges (if present). Empty when the run has no series
// payload (older history rows).
export function buildForecastSeriesRows(detail: HistoryDetail): CsvRow[] {
  const series = _result(detail).series;
  if (!series || !series.forecast_timestamps?.length) return [];
  const rows: CsvRow[] = [
    [
      "timestamp",
      "forecast_close",
      "forecast_close_lower",
      "forecast_close_upper",
      "forecast_volatility",
      "forecast_volatility_lower",
      "forecast_volatility_upper",
    ],
  ];
  series.forecast_timestamps.forEach((ts, i) => {
    rows.push([
      ts,
      series.forecast_close?.[i] ?? null,
      series.forecast_close_lower?.[i] ?? null,
      series.forecast_close_upper?.[i] ?? null,
      series.forecast_volatility?.[i] ?? null,
      series.forecast_volatility_lower?.[i] ?? null,
      series.forecast_volatility_upper?.[i] ?? null,
    ]);
  });
  return rows;
}

export function buildRunSummaryCsv(detail: HistoryDetail): RunSummaryCsv {
  const summary = buildRunSummaryRows(detail);
  const series = buildForecastSeriesRows(detail);
  // We emit a single CSV file with two sections separated by a blank row
  // so the user gets the whole run in one click. The two sections share
  // different schemas, but Excel / pandas both handle this with a single
  // blank-row sniff.
  const blocks: string[] = [toCsv(summary)];
  if (series.length > 0) {
    blocks.push("");
    blocks.push("forecast_series");
    blocks.push(toCsv(series));
  }
  return {
    filename: `fed-pulse-run-${detail.symbol.replace(/[^A-Za-z0-9_-]/g, "_")}-${detail.document_date}-${detail.id.slice(0, 8)}.csv`,
    csv: blocks.join("\r\n"),
  };
}

export function downloadRunCsv(detail: HistoryDetail): void {
  const { filename, csv } = buildRunSummaryCsv(detail);
  downloadCsvBlob(csv, filename);
}
