import { describe, expect, it } from "vitest";

import { computeCompareDelta, describeStanceShift } from "@/lib/analyze/compare";
import type { HistoryDetail } from "@/lib/analyze/types";

function makeDetail(overrides: Partial<HistoryDetail> & { payload?: Record<string, unknown> }): HistoryDetail {
  return {
    id: "fixture",
    created_at: "2024-09-18T12:00:00Z",
    symbol: "^GSPC",
    document_date: "2024-09-18",
    horizon: "3d",
    forecast_mode: "fast",
    stance: "hawkish",
    sentiment_score: 0.8,
    predicted_close: 5000,
    current_close: 4900,
    predicted_volatility: 0.012,
    payload: {},
    ...overrides,
  };
}

describe("computeCompareDelta", () => {
  it("computes close, vol, and stance deltas from payload predictions", () => {
    const a = makeDetail({
      payload: {
        prediction: { close: 5500, volatility: 0.012 },
        sentiment: { label: "hawkish", score: 0.82 },
      },
    });
    const b = makeDetail({
      payload: {
        prediction: { close: 5400, volatility: 0.018 },
        sentiment: { label: "dovish", score: 0.71 },
      },
      stance: "dovish",
    });
    const delta = computeCompareDelta(a, b);
    expect(delta.closeAbsolute).toBeCloseTo(100, 5);
    expect(delta.closePercent).toBeCloseTo((100 / 5400) * 100, 5);
    expect(delta.volatilityAbsolute).toBeCloseTo(-0.006, 5);
    expect(delta.stanceShift).toBe("more_hawkish");
    expect(delta.scoreDelta).toBeCloseTo(0.11, 5);
  });

  it("falls back to history-row stance when payload sentiment is absent", () => {
    const a = makeDetail({ stance: "dovish", payload: {} });
    const b = makeDetail({ stance: "hawkish", payload: {} });
    const delta = computeCompareDelta(a, b);
    expect(delta.stanceShift).toBe("more_dovish");
  });

  it("emits null deltas when prediction fields are missing", () => {
    const empty = makeDetail({ payload: {} });
    const delta = computeCompareDelta(empty, empty);
    expect(delta.closeAbsolute).toBe(null);
    expect(delta.closePercent).toBe(null);
    expect(delta.volatilityAbsolute).toBe(null);
  });

  it("describes stance shifts in human-readable form", () => {
    expect(describeStanceShift("more_hawkish")).toMatch(/hawkish/i);
    expect(describeStanceShift("more_dovish")).toMatch(/dovish/i);
    expect(describeStanceShift("unchanged")).toMatch(/unchanged/i);
    expect(describeStanceShift("unknown")).toMatch(/unknown/i);
  });
});
