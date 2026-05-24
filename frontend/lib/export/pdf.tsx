// Per-run + compare PDF exports backed by @react-pdf/renderer.
//
// Layout is intentionally text-only: title bar (date, symbol, horizon,
// forecast mode), a multi-axis card grid (stance / factor / certainty /
// topic), a forecast metrics block, and a model + series metadata footer.
// The forecast chart embed lives in the deferred bucket — @react-pdf does
// not consume Recharts elements directly, and the cleanest path (snapshot
// the DOM SVG via XMLSerializer and feed it back through <Svg>) collides
// with react-pdf's restricted SVG dialect. Text rendering covers every
// numeric column already present in the per-run CSV, which is the most
// asked-for downstream artefact.
//
// Compare PDF mirrors the same shape: two columns side by side plus the
// per-axis Δ table the compare CSV produces.

import { Document, Page, StyleSheet, Text, View, pdf } from "@react-pdf/renderer";
import React from "react";

import { computeCompareDelta, computeMultiAxisDelta } from "@/lib/analyze/compare";
import type { AnalyzeResult, HistoryDetail } from "@/lib/analyze/types";

const styles = StyleSheet.create({
  page: {
    padding: 36,
    fontSize: 10,
    fontFamily: "Helvetica",
    color: "#111827",
  },
  titleRow: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    marginBottom: 12,
    paddingBottom: 8,
    borderBottomWidth: 1,
    borderBottomColor: "#d1d5db",
  },
  title: { fontSize: 16, fontWeight: 700 },
  subtitle: { fontSize: 10, color: "#4b5563" },
  sectionHeading: {
    fontSize: 11,
    fontWeight: 700,
    marginTop: 14,
    marginBottom: 6,
    textTransform: "uppercase",
    letterSpacing: 0.5,
    color: "#374151",
  },
  cardGrid: { flexDirection: "row", flexWrap: "wrap", gap: 8 },
  card: {
    width: "48%",
    padding: 8,
    borderWidth: 1,
    borderColor: "#e5e7eb",
    borderRadius: 4,
    marginBottom: 8,
  },
  cardLabel: { fontSize: 8, color: "#6b7280", textTransform: "uppercase", marginBottom: 2 },
  cardValue: { fontSize: 12, fontWeight: 700 },
  cardSubvalue: { fontSize: 9, color: "#4b5563", marginTop: 2 },
  table: { width: "100%", marginTop: 4 },
  tableRow: { flexDirection: "row", paddingVertical: 3 },
  tableRowAlt: { backgroundColor: "#f9fafb" },
  tableHeader: { fontWeight: 700, backgroundColor: "#e5e7eb" },
  tableCell: { flex: 1, paddingHorizontal: 4, fontSize: 9 },
  tableCellNarrow: { width: 80, paddingHorizontal: 4, fontSize: 9 },
  twoColumn: { flexDirection: "row", gap: 12 },
  column: { flex: 1 },
  footer: {
    position: "absolute",
    bottom: 18,
    left: 36,
    right: 36,
    fontSize: 8,
    color: "#6b7280",
    borderTopWidth: 1,
    borderTopColor: "#e5e7eb",
    paddingTop: 6,
  },
});

const PDF_MAGIC = "%PDF-";

function asResult(detail: HistoryDetail): AnalyzeResult {
  return (detail.payload || {}) as AnalyzeResult;
}

function fmtNumber(value: number | null | undefined, digits = 4): string {
  if (value == null || Number.isNaN(value)) return "—";
  return Number(value).toFixed(digits);
}

function fmtPercent(value: number | null | undefined, digits = 2): string {
  if (value == null || Number.isNaN(value)) return "—";
  return `${Number(value).toFixed(digits)}%`;
}

function fmtSigned(value: number | null | undefined, digits = 4): string {
  if (value == null || Number.isNaN(value)) return "—";
  const v = Number(value);
  const sign = v > 0 ? "+" : "";
  return `${sign}${v.toFixed(digits)}`;
}

function MultiAxisCardGrid({ detail }: { detail: HistoryDetail }) {
  const result = asResult(detail);
  const stance = result.multi_axis?.stance;
  const factor = result.multi_axis?.factor;
  const certainty = result.multi_axis?.certainty;
  const topic = result.multi_axis?.topic;

  return (
    <View style={styles.cardGrid}>
      <View style={styles.card}>
        <Text style={styles.cardLabel}>Stance</Text>
        <Text style={styles.cardValue}>{stance?.label ?? "—"}</Text>
        <Text style={styles.cardSubvalue}>
          confidence {fmtNumber(stance?.confidence, 3)}
        </Text>
      </View>
      <View style={styles.card}>
        <Text style={styles.cardLabel}>Hawkish/Dovish factor</Text>
        <Text style={styles.cardValue}>{fmtNumber(factor?.value, 3)}</Text>
        <Text style={styles.cardSubvalue}>
          confidence {fmtNumber(factor?.confidence, 3)}
        </Text>
      </View>
      <View style={styles.card}>
        <Text style={styles.cardLabel}>Certainty</Text>
        <Text style={styles.cardValue}>{certainty?.label ?? "—"}</Text>
        <Text style={styles.cardSubvalue}>
          confidence {fmtNumber(certainty?.confidence, 3)}
        </Text>
      </View>
      <View style={styles.card}>
        <Text style={styles.cardLabel}>Primary topic</Text>
        <Text style={styles.cardValue}>{topic?.primary ?? "—"}</Text>
        <Text style={styles.cardSubvalue}>
          confidence {fmtNumber(topic?.confidence, 3)}
        </Text>
      </View>
    </View>
  );
}

function ForecastTable({ detail }: { detail: HistoryDetail }) {
  const result = asResult(detail);
  const prediction = result.prediction;
  const market = result.market;
  const series = result.series;
  const conformal = series?.forecast_band_source === "conformal";

  const rows: Array<[string, string]> = [
    ["forecast.close", fmtNumber(prediction?.close, 4)],
    ["forecast.volatility", fmtNumber(prediction?.volatility, 6)],
    ["forecast.horizon", prediction?.horizon ?? detail.horizon ?? "—"],
    ["market.symbol", market?.symbol ?? "—"],
    ["market.date_used", market?.date_used ?? "—"],
    ["market.close", fmtNumber(market?.close, 4)],
    ["market.volatility_5d", fmtNumber(market?.volatility_5d, 6)],
    [
      "forecast.band_source",
      conformal ? `conformal (coverage ${fmtNumber(series?.conformal_coverage, 3)})` : (series?.forecast_band_source ?? "—"),
    ],
    [
      "forecast.confidence_level",
      fmtNumber(series?.forecast_confidence_level, 3),
    ],
  ];

  return (
    <View style={styles.table}>
      <View style={[styles.tableRow, styles.tableHeader]}>
        <Text style={styles.tableCell}>Field</Text>
        <Text style={styles.tableCell}>Value</Text>
      </View>
      {rows.map(([field, value], i) => (
        <View
          key={field}
          style={[styles.tableRow, i % 2 === 1 ? styles.tableRowAlt : {}]}
        >
          <Text style={styles.tableCell}>{field}</Text>
          <Text style={styles.tableCell}>{value}</Text>
        </View>
      ))}
    </View>
  );
}

function RunDocument({ detail }: { detail: HistoryDetail }) {
  const result = asResult(detail);
  const model = result.model;
  const sentiment = result.sentiment;

  return (
    <Document>
      <Page size="A4" style={styles.page}>
        <View style={styles.titleRow}>
          <View>
            <Text style={styles.title}>Fed Pulse run</Text>
            <Text style={styles.subtitle}>
              {detail.document_date} · {detail.symbol} · horizon {detail.horizon} · {detail.forecast_mode}
            </Text>
          </View>
          <View>
            <Text style={styles.subtitle}>run {detail.id.slice(0, 8)}</Text>
            <Text style={styles.subtitle}>created {detail.created_at}</Text>
          </View>
        </View>

        <Text style={styles.sectionHeading}>Multi-axis labels</Text>
        <MultiAxisCardGrid detail={detail} />

        <Text style={styles.sectionHeading}>Forecast</Text>
        <ForecastTable detail={detail} />

        <Text style={styles.sectionHeading}>Sentiment</Text>
        <View style={styles.table}>
          <View style={[styles.tableRow, styles.tableHeader]}>
            <Text style={styles.tableCell}>Field</Text>
            <Text style={styles.tableCell}>Value</Text>
          </View>
          <View style={styles.tableRow}>
            <Text style={styles.tableCell}>sentiment.label</Text>
            <Text style={styles.tableCell}>{sentiment?.label ?? "—"}</Text>
          </View>
          <View style={[styles.tableRow, styles.tableRowAlt]}>
            <Text style={styles.tableCell}>sentiment.score</Text>
            <Text style={styles.tableCell}>{fmtNumber(sentiment?.score, 4)}</Text>
          </View>
          <View style={styles.tableRow}>
            <Text style={styles.tableCell}>sentiment.ood_energy</Text>
            <Text style={styles.tableCell}>{fmtNumber(sentiment?.ood_energy, 4)}</Text>
          </View>
          <View style={[styles.tableRow, styles.tableRowAlt]}>
            <Text style={styles.tableCell}>sentiment.is_in_distribution</Text>
            <Text style={styles.tableCell}>
              {sentiment?.is_in_distribution == null ? "—" : String(sentiment.is_in_distribution)}
            </Text>
          </View>
        </View>

        <View fixed style={styles.footer}>
          <Text>
            model: checkpoint_loaded={String(model?.checkpoint_loaded ?? "—")} · runtime_mode={model?.runtime_mode ?? "—"} · combined_rmse={fmtNumber(model?.combined_rmse, 6)}
          </Text>
        </View>
      </Page>
    </Document>
  );
}

function CompareDocument({ a, b }: { a: HistoryDetail; b: HistoryDetail }) {
  const ra = asResult(a);
  const rb = asResult(b);
  const delta = computeCompareDelta(a, b);
  const maxis = computeMultiAxisDelta(a, b);

  const rows: Array<[string, string, string, string]> = [
    [
      "stance.label",
      ra.multi_axis?.stance?.label ?? ra.sentiment?.label ?? "—",
      rb.multi_axis?.stance?.label ?? rb.sentiment?.label ?? "—",
      maxis.stanceRankDelta == null ? "—" : fmtSigned(maxis.stanceRankDelta, 0),
    ],
    [
      "stance.confidence",
      fmtNumber(ra.multi_axis?.stance?.confidence, 3),
      fmtNumber(rb.multi_axis?.stance?.confidence, 3),
      fmtSigned(maxis.stanceConfidenceDelta, 3),
    ],
    [
      "factor.value",
      fmtNumber(ra.multi_axis?.factor?.value, 3),
      fmtNumber(rb.multi_axis?.factor?.value, 3),
      fmtSigned(maxis.factorDelta, 3),
    ],
    [
      "factor.confidence",
      fmtNumber(ra.multi_axis?.factor?.confidence, 3),
      fmtNumber(rb.multi_axis?.factor?.confidence, 3),
      fmtSigned(maxis.factorConfidenceDelta, 3),
    ],
    [
      "certainty.label",
      ra.multi_axis?.certainty?.label ?? "—",
      rb.multi_axis?.certainty?.label ?? "—",
      maxis.certaintyShift,
    ],
    [
      "certainty.confidence",
      fmtNumber(ra.multi_axis?.certainty?.confidence, 3),
      fmtNumber(rb.multi_axis?.certainty?.confidence, 3),
      fmtSigned(maxis.certaintyConfidenceDelta, 3),
    ],
    [
      "topic.primary",
      ra.multi_axis?.topic?.primary ?? "—",
      rb.multi_axis?.topic?.primary ?? "—",
      maxis.topicChanged == null ? "—" : maxis.topicChanged ? "changed" : "unchanged",
    ],
    [
      "regime.argmax",
      ra.regime_classification?.argmax_class ?? "—",
      rb.regime_classification?.argmax_class ?? "—",
      delta.regime.argmaxChanged == null
        ? "—"
        : delta.regime.argmaxChanged
        ? "changed"
        : "same",
    ],
    [
      "regime.set_size",
      ra.regime_classification?.set_size != null
        ? String(ra.regime_classification.set_size)
        : "—",
      rb.regime_classification?.set_size != null
        ? String(rb.regime_classification.set_size)
        : "—",
      delta.regime.setSizeA != null && delta.regime.setSizeB != null
        ? fmtSigned(delta.regime.setSizeA - delta.regime.setSizeB, 0)
        : "—",
    ],
    [
      "credibility.drift_score",
      fmtNumber(ra.credibility?.drift_score, 3),
      fmtNumber(rb.credibility?.drift_score, 3),
      fmtSigned(delta.driftDelta, 3),
    ],
    [
      "sentiment.score",
      fmtNumber(ra.sentiment?.score, 4),
      fmtNumber(rb.sentiment?.score, 4),
      fmtSigned(delta.scoreDelta, 4),
    ],
  ];

  return (
    <Document>
      <Page size="A4" style={styles.page}>
        <View style={styles.titleRow}>
          <View>
            <Text style={styles.title}>Fed Pulse compare</Text>
            <Text style={styles.subtitle}>
              {a.document_date} {a.symbol} vs {b.document_date} {b.symbol}
            </Text>
          </View>
          <View>
            <Text style={styles.subtitle}>
              {a.id.slice(0, 8)} vs {b.id.slice(0, 8)}
            </Text>
          </View>
        </View>

        <Text style={styles.sectionHeading}>Run A</Text>
        <View style={styles.twoColumn}>
          <View style={styles.column}>
            <MultiAxisCardGrid detail={a} />
          </View>
          <View style={styles.column}>
            <MultiAxisCardGrid detail={b} />
          </View>
        </View>

        <Text style={styles.sectionHeading}>Per-axis Δ</Text>
        <View style={styles.table}>
          <View style={[styles.tableRow, styles.tableHeader]}>
            <Text style={styles.tableCell}>Field</Text>
            <Text style={styles.tableCell}>Run A</Text>
            <Text style={styles.tableCell}>Run B</Text>
            <Text style={styles.tableCell}>Δ (A − B)</Text>
          </View>
          {rows.map((row, i) => (
            <View
              key={row[0]}
              style={[styles.tableRow, i % 2 === 1 ? styles.tableRowAlt : {}]}
            >
              <Text style={styles.tableCell}>{row[0]}</Text>
              <Text style={styles.tableCell}>{row[1]}</Text>
              <Text style={styles.tableCell}>{row[2]}</Text>
              <Text style={styles.tableCell}>{row[3]}</Text>
            </View>
          ))}
        </View>

        <View fixed style={styles.footer}>
          <Text>
            horizon A={a.horizon} B={b.horizon} · forecast_mode A={a.forecast_mode} B={b.forecast_mode}
          </Text>
        </View>
      </Page>
    </Document>
  );
}

// Convert a NodeJS-style ReadableStream into a Uint8Array. Tests reach this
// branch (jsdom + react-pdf falls back to the Node renderer); the browser
// path uses `pdf().toBlob()` directly.
async function streamToBuffer(stream: NodeJS.ReadableStream): Promise<Uint8Array> {
  return new Promise((resolve, reject) => {
    const chunks: Uint8Array[] = [];
    stream.on("data", (chunk: Uint8Array | Buffer) => {
      chunks.push(chunk instanceof Uint8Array ? chunk : new Uint8Array(chunk));
    });
    stream.on("end", () => {
      const total = chunks.reduce((n, c) => n + c.length, 0);
      const out = new Uint8Array(total);
      let offset = 0;
      for (const c of chunks) {
        out.set(c, offset);
        offset += c.length;
      }
      resolve(out);
    });
    stream.on("error", reject);
  });
}

export async function buildRunPdfBuffer(detail: HistoryDetail): Promise<Uint8Array> {
  const instance = pdf(<RunDocument detail={detail} />);
  const stream = await instance.toBuffer();
  return streamToBuffer(stream);
}

export async function buildComparePdfBuffer(
  a: HistoryDetail,
  b: HistoryDetail,
): Promise<Uint8Array> {
  const instance = pdf(<CompareDocument a={a} b={b} />);
  const stream = await instance.toBuffer();
  return streamToBuffer(stream);
}

function _filenameSafe(value: string): string {
  return value.replace(/[^A-Za-z0-9_-]/g, "_");
}

export function buildRunPdfFilename(detail: HistoryDetail): string {
  return `fed-pulse-run-${_filenameSafe(detail.symbol)}-${detail.document_date}-${detail.id.slice(0, 8)}.pdf`;
}

export function buildComparePdfFilename(a: HistoryDetail, b: HistoryDetail): string {
  return `fed-pulse-compare-${a.id.slice(0, 8)}-vs-${b.id.slice(0, 8)}.pdf`;
}

function _triggerDownload(blob: Blob, filename: string): void {
  if (typeof document === "undefined" || typeof URL === "undefined") return;
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

export async function downloadRunPdf(detail: HistoryDetail): Promise<void> {
  const instance = pdf(<RunDocument detail={detail} />);
  const blob = await instance.toBlob();
  _triggerDownload(blob, buildRunPdfFilename(detail));
}

export async function downloadComparePdf(a: HistoryDetail, b: HistoryDetail): Promise<void> {
  const instance = pdf(<CompareDocument a={a} b={b} />);
  const blob = await instance.toBlob();
  _triggerDownload(blob, buildComparePdfFilename(a, b));
}

// Exported only so tests can assert the magic bytes without importing
// from inside the function body.
export const _PDF_MAGIC = PDF_MAGIC;
