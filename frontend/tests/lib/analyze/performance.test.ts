import { describe, expect, it } from "vitest";

import {
  aggregatePerformance,
  computeRunPerformance,
} from "@/lib/analyze/performance";
import type { HistoryEntry } from "@/lib/analyze/types";

function makeEntry(overrides: Partial<HistoryEntry> = {}): HistoryEntry {
  return {
    id: "run-1",
    created_at: "2024-09-18T12:00:00Z",
    symbol: "^GSPC",
    document_date: "2024-09-18",
    horizon: "3d",
    forecast_mode: "fast",
    stance: "hawkish",
    sentiment_score: 0.7,
    predicted_close: 5500,
    current_close: 5400,
    predicted_volatility: 0.012,
    text_excerpt: null,
    ...overrides,
  };
}

describe("computeRunPerformance", () => {
  it("marks direction correct when predicted and realized move the same way from spot", () => {
    const result = computeRunPerformance(makeEntry(), 5520);
    expect(result.direction_correct).toBe(true);
    expect(result.absolute_error).toBeCloseTo(20, 5);
    expect(result.percent_error).toBeCloseTo(20 / 5520, 5);
  });

  it("marks direction incorrect when realized moves opposite to predicted", () => {
    const result = computeRunPerformance(makeEntry(), 5380);
    expect(result.direction_correct).toBe(false);
  });

  it("returns null fields when realized close is missing", () => {
    const result = computeRunPerformance(makeEntry(), null);
    expect(result.direction_correct).toBeNull();
    expect(result.absolute_error).toBeNull();
    expect(result.percent_error).toBeNull();
  });

  it("guards percent_error against a zero realized close", () => {
    const result = computeRunPerformance(makeEntry(), 0);
    expect(result.absolute_error).toBeCloseTo(5500, 5);
    expect(result.percent_error).toBeNull();
  });
});

describe("aggregatePerformance", () => {
  it("computes hit rate, MAPE, and per-symbol breakdown from resolved rows", () => {
    const rows = [
      computeRunPerformance(makeEntry({ id: "a", symbol: "^GSPC", predicted_close: 5500, current_close: 5400 }), 5520),
      computeRunPerformance(makeEntry({ id: "b", symbol: "^GSPC", predicted_close: 5500, current_close: 5400 }), 5380),
      computeRunPerformance(makeEntry({ id: "c", symbol: "^NDX", predicted_close: 18000, current_close: 17900 }), 18100),
      computeRunPerformance(makeEntry({ id: "d", symbol: "^NDX", predicted_close: 18000, current_close: 17900 }), null),
    ];
    const agg = aggregatePerformance(rows);
    expect(agg.total).toBe(4);
    expect(agg.resolved).toBe(3);
    expect(agg.hitRate).toBeCloseTo(2 / 3, 5);
    expect(agg.mape).not.toBeNull();
    const gspc = agg.bySymbol.find((entry) => entry.symbol === "^GSPC");
    const ndx = agg.bySymbol.find((entry) => entry.symbol === "^NDX");
    expect(gspc?.resolved).toBe(2);
    expect(gspc?.hitRate).toBeCloseTo(0.5, 5);
    expect(ndx?.resolved).toBe(1);
    expect(ndx?.hitRate).toBeCloseTo(1, 5);
  });

  it("returns null hit rate and MAPE when no rows resolve", () => {
    const rows = [computeRunPerformance(makeEntry({ id: "a" }), null)];
    const agg = aggregatePerformance(rows);
    expect(agg.resolved).toBe(0);
    expect(agg.hitRate).toBeNull();
    expect(agg.mape).toBeNull();
  });
});
