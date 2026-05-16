import { describe, expect, it } from "vitest";

import { buildCompareCsv, buildCompareRows } from "@/lib/export/compare-export";
import type { HistoryDetail } from "@/lib/analyze/types";

function makeDetail(
  overrides: Partial<HistoryDetail> & { payload?: Record<string, unknown> },
): HistoryDetail {
  return {
    id: "fixture-id-aaa",
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
    payload: {},
    ...overrides,
  };
}

describe("buildCompareRows", () => {
  it("emits header + per-field rows with delta column", () => {
    const a = makeDetail({
      id: "fixture-id-aaa",
      payload: {
        prediction: { close: 5500, volatility: 0.012 },
        sentiment: { label: "hawkish", score: 0.82 },
        multi_axis: {
          stance: { label: "hawkish", confidence: 0.9 },
          factor: { value: 0.4, confidence: 0.2 },
          certainty: { label: "decisive", confidence: 0.75 },
          topic: { primary: "inflation", confidence: 0.6 },
        },
      },
    });
    const b = makeDetail({
      id: "fixture-id-bbb",
      stance: "dovish",
      payload: {
        prediction: { close: 5400, volatility: 0.018 },
        sentiment: { label: "dovish", score: 0.71 },
        multi_axis: {
          stance: { label: "dovish", confidence: 0.8 },
          factor: { value: -0.1, confidence: 0.25 },
          certainty: { label: "tentative", confidence: 0.55 },
          topic: { primary: "growth", confidence: 0.62 },
        },
      },
    });
    const rows = buildCompareRows(a, b);
    expect(rows[0]).toEqual(["field", "run_a", "run_b", "delta_a_minus_b"]);

    const asObject: Record<string, [unknown, unknown, unknown]> = {};
    for (const row of rows.slice(1)) {
      asObject[String(row[0])] = [row[1], row[2], row[3]];
    }
    expect(asObject["prediction.close"]?.[2]).toBeCloseTo(100, 5);
    expect(asObject["prediction.volatility"]?.[2]).toBeCloseTo(-0.006, 5);
    expect(asObject["multi_axis.stance.label"]).toEqual(["hawkish", "dovish", 2]);
    expect(asObject["multi_axis.topic.primary"]?.[2]).toBe("changed");
    expect(asObject["stance.shift"]?.[2]).toBe("more_hawkish");
  });

  it("falls back to empty delta when one side lacks the axis", () => {
    const a = makeDetail({
      payload: { multi_axis: { factor: { value: 0.4, confidence: 0.2 } } },
    });
    const b = makeDetail({ payload: {} });
    const rows = buildCompareRows(a, b);
    const factorRow = rows.find((row) => row[0] === "multi_axis.factor.value");
    expect(factorRow?.[3]).toBe(null);
  });
});

describe("buildCompareCsv", () => {
  it("uses the run-id prefixes in the filename", () => {
    const a = makeDetail({ id: "aaaaaaaaaaaa" });
    const b = makeDetail({ id: "bbbbbbbbbbbb" });
    const { filename } = buildCompareCsv(a, b);
    expect(filename).toContain("aaaaaaaa");
    expect(filename).toContain("bbbbbbbb");
    expect(filename.endsWith(".csv")).toBe(true);
  });
});
