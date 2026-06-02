import { describe, expect, it } from "vitest";

import { buildCompareCsv, buildCompareRows } from "@/lib/export/compare-export";
import type { HistoryDetail } from "@/lib/analyze/types";

function makeDetail(
  overrides: Partial<HistoryDetail> & { payload?: Record<string, unknown> },
): HistoryDetail {
  return {
    id: "fixture-id-aaa",
    created_at: "2026-05-24T12:00:00Z",
    symbol: "^GSPC",
    document_date: "2026-05-01",
    horizon: "10d",
    forecast_mode: "fast",
    stance: "hawkish",
    sentiment_score: 0.82,
    predicted_close: null,
    current_close: null,
    predicted_volatility: null,
    payload: {},
    ...overrides,
  };
}

describe("buildCompareRows", () => {
  it("emits header + per-field rows with delta column", () => {
    const a = makeDetail({
      id: "fixture-id-aaa",
      payload: {
        sentiment: { label: "hawkish", score: 0.82 },
        multi_axis: {
          stance: { label: "hawkish", confidence: 0.9 },
          factor: { value: 0.4, confidence: 0.2 },
          certainty: { label: "decisive", confidence: 0.75 },
        },
        regime_classification: {
          argmax_class: "high",
          predicted_set: ["normal", "high"],
          set_size: 2,
          coverage: 0.8,
          distribution: { calm: 0.1, normal: 0.3, high: 0.6 },
          set_label: "{normal, high}",
        },
        credibility: {
          drift_score: 0.42,
          realized_vs_stated_gap: 0.12,
          market_implied_gap: 0,
          months_since_reversal: 9,
        },
      },
    });
    const b = makeDetail({
      id: "fixture-id-bbb",
      stance: "dovish",
      payload: {
        sentiment: { label: "dovish", score: 0.71 },
        multi_axis: {
          stance: { label: "dovish", confidence: 0.8 },
          factor: { value: -0.1, confidence: 0.25 },
          certainty: { label: "tentative", confidence: 0.55 },
        },
        regime_classification: {
          argmax_class: "normal",
          predicted_set: ["calm", "normal"],
          set_size: 2,
          coverage: 0.8,
          distribution: { calm: 0.45, normal: 0.5, high: 0.05 },
          set_label: "{calm, normal}",
        },
        credibility: {
          drift_score: 0.28,
          realized_vs_stated_gap: -0.04,
          market_implied_gap: 0,
          months_since_reversal: 4,
        },
      },
    });
    const rows = buildCompareRows(a, b);
    expect(rows[0]).toEqual(["field", "run_a", "run_b", "delta_a_minus_b"]);

    const asObject: Record<string, [unknown, unknown, unknown]> = {};
    for (const row of rows.slice(1)) {
      asObject[String(row[0])] = [row[1], row[2], row[3]];
    }
    expect(asObject["regime.argmax"]).toEqual(["high", "normal", "changed"]);
    expect(asObject["regime.set"]?.[0]).toBe("normal|high");
    expect(asObject["regime.set"]?.[1]).toBe("calm|normal");
    expect(asObject["credibility.drift_score"]?.[2] as number).toBeCloseTo(0.14, 5);
    expect(asObject["multi_axis.stance.label"]).toEqual(["hawkish", "dovish", 2]);
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
