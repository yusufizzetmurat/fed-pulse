// Per-run CSV export — flat field/value rows for the regime workspace.
//
// One row per emitted metric so the file opens cleanly in Excel / pandas
// without nested structures. Regime + multi-axis + credibility + XAI
// summary rows replace the legacy close-price prediction surface.

import type { AnalyzeResult, HistoryDetail } from "@/lib/analyze/types";
import { downloadCsvBlob, toCsv, type CsvRow } from "./csv";

function _result(detail: HistoryDetail): AnalyzeResult {
  return (detail.payload || {}) as AnalyzeResult;
}

export interface RunSummaryCsv {
  filename: string;
  csv: string;
}

export function buildRunSummaryRows(detail: HistoryDetail): CsvRow[] {
  const result = _result(detail);
  const sentiment = result.sentiment || {};
  const model = result.model || {};
  const stance = result.multi_axis?.stance;
  const factor = result.multi_axis?.factor;
  const certainty = result.multi_axis?.certainty;
  const credibility = result.credibility;
  const regime = result.regime_classification;

  return [
    ["field", "value"],
    ["run_id", detail.id],
    ["created_at", detail.created_at],
    ["symbol", detail.symbol],
    ["document_date", detail.document_date],
    ["horizon", detail.horizon],
    ["model.encoder_key", (model as { encoder_key?: string | null }).encoder_key ?? null],
    ["model.checkpoint_loaded", model.checkpoint_loaded ?? null],
    ["model.runtime_mode", model.runtime_mode ?? null],
    ["sentiment.label", sentiment.label ?? null],
    ["sentiment.score", sentiment.score ?? null],
    ["sentiment.ood_energy", sentiment.ood_energy ?? null],
    ["sentiment.is_in_distribution", sentiment.is_in_distribution ?? null],
    ["regime.argmax", regime?.argmax_class ?? null],
    ["regime.set", regime?.predicted_set?.join("|") ?? null],
    ["regime.set_label", regime?.set_label ?? null],
    ["regime.set_size", regime?.set_size ?? null],
    ["regime.coverage", regime?.coverage ?? null],
    [
      "regime.argmax_probability",
      regime?.argmax_class && regime?.distribution
        ? regime.distribution[regime.argmax_class] ?? null
        : null,
    ],
    ["multi_axis.stance.label", stance?.label ?? null],
    ["multi_axis.stance.confidence", stance?.confidence ?? null],
    ["multi_axis.factor.value", factor?.value ?? null],
    ["multi_axis.factor.confidence", factor?.confidence ?? null],
    ["multi_axis.certainty.label", certainty?.label ?? null],
    ["multi_axis.certainty.confidence", certainty?.confidence ?? null],
    ["credibility.drift_score", credibility?.drift_score ?? null],
    ["credibility.realized_vs_stated_gap", credibility?.realized_vs_stated_gap ?? null],
    ["credibility.market_implied_gap", credibility?.market_implied_gap ?? null],
    ["credibility.months_since_reversal", credibility?.months_since_reversal ?? null],
  ];
}

export function buildXaiSentencesRows(detail: HistoryDetail): CsvRow[] {
  const xai = _result(detail).xai;
  if (!xai?.sentences?.length) return [];
  const rows: CsvRow[] = [["sentence_index", "score", "top_tokens", "text"]];
  xai.sentences.forEach((sentence, idx) => {
    const topTokens = (sentence.topTokens ?? [])
      .map((token) => `${token.token}:${token.weight.toFixed(3)}`)
      .join(" ");
    rows.push([idx, sentence.score, topTokens, sentence.text]);
  });
  return rows;
}

export function buildRunSummaryCsv(detail: HistoryDetail): RunSummaryCsv {
  const summary = buildRunSummaryRows(detail);
  const xai = buildXaiSentencesRows(detail);
  // Single CSV with two sections separated by a blank row so the reader
  // gets the whole run in one click. Excel and pandas both handle the
  // blank-row sniff cleanly.
  const blocks: string[] = [toCsv(summary)];
  if (xai.length > 0) {
    blocks.push("");
    blocks.push("xai_sentences");
    blocks.push(toCsv(xai));
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
