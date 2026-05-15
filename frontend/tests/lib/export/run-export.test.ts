import { describe, expect, it } from "vitest";

import {
  buildForecastSeriesRows,
  buildRunSummaryCsv,
  buildRunSummaryRows,
} from "@/lib/export/run-export";
import type { HistoryDetail } from "@/lib/analyze/types";

function makeDetail(): HistoryDetail {
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
      sentiment: { label: "hawkish", score: 0.82, is_in_distribution: true },
      prediction: { close: 5500, volatility: 0.012, horizon: "3d" },
      market: { symbol: "^GSPC", date_used: "2024-09-17", close: 5400, volatility_5d: 0.011 },
      model: { checkpoint_loaded: true, runtime_mode: "fast", combined_rmse: 0.0021 },
      multi_axis: {
        stance: { label: "hawkish", confidence: 0.91 },
        factor: { value: 0.42, confidence: 0.18 },
        certainty: { label: "decisive", confidence: 0.74 },
        topic: { primary: "inflation", confidence: 0.66 },
      },
      credibility: {
        drift_score: 0.21,
        realized_vs_stated_gap: -0.03,
        market_implied_gap: 0.07,
        months_since_reversal: 12,
      },
      series: {
        forecast_band_source: "conformal",
        forecast_confidence_level: 0.8,
        conformal_coverage: 0.81,
        forecast_timestamps: ["2024-09-18", "2024-09-19", "2024-09-20"],
        forecast_close: [5500, 5510, 5520],
        forecast_close_lower: [5450, 5455, 5460],
        forecast_close_upper: [5550, 5565, 5580],
        forecast_volatility: [0.012, 0.013, 0.014],
        forecast_volatility_lower: [0.011, 0.012, 0.013],
        forecast_volatility_upper: [0.013, 0.014, 0.015],
      },
    },
  };
}

describe("buildRunSummaryRows", () => {
  it("starts with the schema header row", () => {
    const rows = buildRunSummaryRows(makeDetail());
    expect(rows[0]).toEqual(["field", "value"]);
  });

  it("emits the run-id, sentiment, prediction, and multi-axis fields", () => {
    const rows = buildRunSummaryRows(makeDetail());
    const asObject: Record<string, unknown> = {};
    for (const row of rows.slice(1)) {
      asObject[String(row[0])] = row[1];
    }
    expect(asObject.run_id).toBe("abcdef1234567890");
    expect(asObject["sentiment.label"]).toBe("hawkish");
    expect(asObject["sentiment.score"]).toBe(0.82);
    expect(asObject["prediction.close"]).toBe(5500);
    expect(asObject["multi_axis.stance.label"]).toBe("hawkish");
    expect(asObject["multi_axis.factor.value"]).toBe(0.42);
    expect(asObject["credibility.drift_score"]).toBe(0.21);
    expect(asObject["series.forecast_band_source"]).toBe("conformal");
  });

  it("keeps the schema stable when the payload is empty", () => {
    const empty: HistoryDetail = { ...makeDetail(), payload: {} };
    const rows = buildRunSummaryRows(empty);
    // Same number of rows + same field order even though the values are null.
    expect(rows.length).toBe(buildRunSummaryRows(makeDetail()).length);
    expect(rows[1][0]).toBe("run_id");
  });
});

describe("buildForecastSeriesRows", () => {
  it("emits one row per forecast timestamp with bands", () => {
    const rows = buildForecastSeriesRows(makeDetail());
    expect(rows[0]).toEqual([
      "timestamp",
      "forecast_close",
      "forecast_close_lower",
      "forecast_close_upper",
      "forecast_volatility",
      "forecast_volatility_lower",
      "forecast_volatility_upper",
    ]);
    expect(rows.length).toBe(4); // header + 3 timesteps
    expect(rows[1]).toEqual(["2024-09-18", 5500, 5450, 5550, 0.012, 0.011, 0.013]);
  });

  it("returns empty when the run has no series payload", () => {
    const empty: HistoryDetail = { ...makeDetail(), payload: {} };
    expect(buildForecastSeriesRows(empty)).toEqual([]);
  });
});

describe("buildRunSummaryCsv", () => {
  it("produces a filename anchored to symbol + date + run-id prefix", () => {
    const { filename } = buildRunSummaryCsv(makeDetail());
    expect(filename).toContain("GSPC");
    expect(filename).toContain("2024-09-18");
    expect(filename.endsWith(".csv")).toBe(true);
  });

  it("includes both summary and forecast_series sections", () => {
    const { csv } = buildRunSummaryCsv(makeDetail());
    expect(csv).toContain("field,value");
    expect(csv).toContain("forecast_series");
    expect(csv).toContain("forecast_close_lower");
  });
});
