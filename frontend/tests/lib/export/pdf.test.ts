import { describe, expect, it } from "vitest";

import {
  _PDF_MAGIC,
  buildComparePdfBuffer,
  buildComparePdfFilename,
  buildRunPdfBuffer,
  buildRunPdfFilename,
} from "@/lib/export/pdf";
import type { HistoryDetail } from "@/lib/analyze/types";

function makeDetail(overrides: Partial<HistoryDetail> = {}): HistoryDetail {
  return {
    id: "abcdef1234567890",
    created_at: "2024-09-18T12:00:00Z",
    symbol: "^GSPC",
    document_date: "2024-09-18",
    horizon: "3d",
    forecast_mode: "fast",
    stance: "hawkish",
    sentiment_score: 0.82,
    predicted_close: 5500,
    current_close: 5400,
    predicted_volatility: 0.012,
    payload: {
      sentiment: {
        label: "hawkish",
        score: 0.82,
        is_in_distribution: true,
        ood_energy: -1.2,
      },
      prediction: { close: 5500, volatility: 0.012, horizon: "3d" },
      market: {
        symbol: "^GSPC",
        date_used: "2024-09-17",
        close: 5400,
        volatility_5d: 0.011,
      },
      model: {
        checkpoint_loaded: true,
        runtime_mode: "fast",
        combined_rmse: 0.0021,
      },
      multi_axis: {
        stance: { label: "hawkish", confidence: 0.91 },
        factor: { value: 0.42, confidence: 0.18 },
        certainty: { label: "decisive", confidence: 0.74 },
        topic: { primary: "inflation", confidence: 0.66 },
      },
      series: {
        forecast_band_source: "conformal",
        forecast_confidence_level: 0.8,
        conformal_coverage: 0.82,
      },
    } as HistoryDetail["payload"],
    ...overrides,
  };
}

function hasPdfMagic(buf: Uint8Array): boolean {
  const head = String.fromCharCode(...buf.slice(0, _PDF_MAGIC.length));
  return head === _PDF_MAGIC;
}

describe("per-run PDF export", () => {
  it("emits a non-empty buffer that starts with the PDF magic", async () => {
    const detail = makeDetail();
    const buf = await buildRunPdfBuffer(detail);
    expect(buf.length).toBeGreaterThan(0);
    expect(hasPdfMagic(buf)).toBe(true);
  });

  it("filenames include the symbol, document date, and run-id prefix", () => {
    const detail = makeDetail();
    expect(buildRunPdfFilename(detail)).toBe(
      "fed-pulse-run-_GSPC-2024-09-18-abcdef12.pdf",
    );
  });
});

describe("compare PDF export", () => {
  it("emits a non-empty buffer that starts with the PDF magic", async () => {
    const a = makeDetail({ id: "aaaaaaaa11111111", document_date: "2024-09-18" });
    const b = makeDetail({
      id: "bbbbbbbb22222222",
      document_date: "2024-07-31",
      stance: "dovish",
      sentiment_score: 0.41,
      payload: {
        sentiment: { label: "dovish", score: 0.41, is_in_distribution: true },
        prediction: { close: 5300, volatility: 0.015, horizon: "3d" },
        market: { symbol: "^GSPC", date_used: "2024-07-30", close: 5320 },
        model: { checkpoint_loaded: true, runtime_mode: "fast" },
        multi_axis: {
          stance: { label: "dovish", confidence: 0.83 },
          factor: { value: -0.31, confidence: 0.22 },
          certainty: { label: "measured", confidence: 0.61 },
          topic: { primary: "employment", confidence: 0.55 },
        },
      } as HistoryDetail["payload"],
    });

    const buf = await buildComparePdfBuffer(a, b);
    expect(buf.length).toBeGreaterThan(0);
    expect(hasPdfMagic(buf)).toBe(true);
  });

  it("compare filename pairs both run id prefixes", () => {
    const a = makeDetail({ id: "aaaaaaaa11111111" });
    const b = makeDetail({ id: "bbbbbbbb22222222" });
    expect(buildComparePdfFilename(a, b)).toBe("fed-pulse-compare-aaaaaaaa-vs-bbbbbbbb.pdf");
  });
});
