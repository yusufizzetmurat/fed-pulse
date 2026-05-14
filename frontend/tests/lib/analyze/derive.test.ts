import { describe, expect, it } from "vitest";

import {
  buildAttention,
  buildCloseSeries,
  buildVolatilitySeries,
  computeCurrentSpotBandCheck,
  computeErrorMetrics,
  computeRealizedBandCheck,
} from "@/lib/analyze/derive";
import type { AnalyzeResult } from "@/lib/analyze/types";

const baseResult: AnalyzeResult = {
  market: { close: 5050, volatility_5d: 0.01 },
  prediction: { close: 5050, volatility: 0.012, horizon: "3d" },
  sentiment: { label: "HAWKISH", score: 0.81 },
  series: {
    timestamps: ["2024-09-15", "2024-09-16", "2024-09-17"],
    history_close: [4900, 4950, 5000],
    history_volatility: [0.011, 0.012, 0.01],
    forecast_timestamps: ["2024-09-18", "2024-09-19", "2024-09-20"],
    forecast_close: [5025, 5040, 5050],
    forecast_close_lower: [5000, 5015, 5020],
    forecast_close_upper: [5060, 5070, 5080],
    forecast_volatility: [0.011, 0.012, 0.012],
    forecast_volatility_lower: [0.009, 0.01, 0.01],
    forecast_volatility_upper: [0.014, 0.015, 0.016],
    realized_timestamps: ["2024-09-18", "2024-09-19"],
    realized_close: [5030, 5045],
    realized_volatility: [0.012, 0.013],
    forecast_confidence_level: 0.8,
    volatility_scale: { suggested_ymin: 0.0, suggested_ymax: 0.05 },
  },
};

describe("buildCloseSeries", () => {
  it("merges history, forecast and realized by timestamp", () => {
    const rows = buildCloseSeries(baseResult);
    expect(rows.map((r) => r.timestamp)).toEqual([
      "2024-09-15",
      "2024-09-16",
      "2024-09-17",
      "2024-09-18",
      "2024-09-19",
      "2024-09-20",
    ]);
    const sept18 = rows.find((r) => r.timestamp === "2024-09-18");
    expect(sept18?.forecast).toBe(5025);
    expect(sept18?.forecastBand).toBe(60);
    expect(sept18?.realized).toBe(5030);
    const sept17 = rows.find((r) => r.timestamp === "2024-09-17");
    expect(sept17?.history).toBe(5000);
    expect(sept17?.forecast).toBeNull();
  });

  it("returns empty array when there is no series", () => {
    expect(buildCloseSeries(null)).toEqual([]);
    expect(buildCloseSeries({})).toEqual([]);
  });
});

describe("buildVolatilitySeries", () => {
  it("uses the volatility fields", () => {
    const rows = buildVolatilitySeries(baseResult);
    const sept18 = rows.find((r) => r.timestamp === "2024-09-18");
    expect(sept18?.forecast).toBeCloseTo(0.011);
    expect(sept18?.realized).toBeCloseTo(0.012);
  });
});

describe("computeErrorMetrics", () => {
  it("computes MAPE and RMSE on overlapping timestamps", () => {
    const bundle = computeErrorMetrics(baseResult);
    expect(bundle.hasRealized).toBe(true);
    expect(bundle.close.mape).not.toBeNull();
    expect(bundle.close.rmse).not.toBeNull();
    expect(bundle.vol.mape).not.toBeNull();
  });

  it("returns empty bundle when no realized data overlaps", () => {
    const bundle = computeErrorMetrics({
      series: {
        forecast_timestamps: ["2024-09-18"],
        forecast_close: [5000],
        realized_timestamps: [],
        realized_close: [],
      },
    });
    expect(bundle.hasRealized).toBe(false);
    expect(bundle.close.mape).toBeNull();
  });
});

describe("band checks", () => {
  it("computeCurrentSpotBandCheck identifies in-band spot", () => {
    const check = computeCurrentSpotBandCheck(baseResult);
    expect(check).not.toBeNull();
    expect(check?.withinBand).toBe(true);
    expect(check?.tone).toBe("good");
  });

  it("computeRealizedBandCheck identifies out-of-band realized", () => {
    const result: AnalyzeResult = {
      ...baseResult,
      series: {
        ...baseResult.series,
        realized_timestamps: ["2024-09-19"],
        realized_close: [5400],
      },
    };
    const check = computeRealizedBandCheck(result);
    expect(check).not.toBeNull();
    expect(check?.withinBand).toBe(false);
    expect(["caution", "danger"]).toContain(check?.tone);
  });
});

describe("buildAttention", () => {
  it("returns null when chunk attention is missing", () => {
    expect(buildAttention(baseResult)).toBeNull();
  });

  it("normalizes weights and decay coefficients", () => {
    const bundle = buildAttention({
      model: {
        chunk_attention: {
          weights: [0.4, 0.6],
          decay_coeffs: [0.9, 0.5],
          chunk_previews: ["alpha statement", "beta passage"],
          lambda_value: 0.05,
          chunk_count: 2,
        },
      },
    });
    expect(bundle).not.toBeNull();
    expect(bundle?.rows).toHaveLength(2);
    expect(bundle?.rows[0].weightPct).toBe("40.0");
    expect(bundle?.lambdaValue).toBeCloseTo(0.05);
  });
});
