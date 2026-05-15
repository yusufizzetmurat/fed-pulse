import { describe, expect, it } from "vitest";

import {
  computeCompareDelta,
  computeMultiAxisDelta,
  describeStanceShift,
} from "@/lib/analyze/compare";
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

describe("computeMultiAxisDelta", () => {
  it("emits per-axis deltas and shifts", () => {
    const a = makeDetail({
      payload: {
        multi_axis: {
          stance: { label: "hawkish", confidence: 0.9 },
          factor: { value: 0.4, confidence: 0.2 },
          certainty: { label: "decisive", confidence: 0.7 },
          topic: { primary: "inflation", confidence: 0.6 },
        },
      },
    });
    const b = makeDetail({
      payload: {
        multi_axis: {
          stance: { label: "dovish", confidence: 0.8 },
          factor: { value: -0.1, confidence: 0.25 },
          certainty: { label: "tentative", confidence: 0.55 },
          topic: { primary: "growth", confidence: 0.5 },
        },
      },
    });
    const d = computeMultiAxisDelta(a, b);
    expect(d.stanceRankDelta).toBe(2);
    expect(d.stanceConfidenceDelta).toBeCloseTo(0.1, 5);
    expect(d.factorDelta).toBeCloseTo(0.5, 5);
    expect(d.factorConfidenceDelta).toBeCloseTo(-0.05, 5);
    expect(d.certaintyShift).toBe("more_decisive");
    expect(d.certaintyConfidenceDelta).toBeCloseTo(0.15, 5);
    expect(d.topicChanged).toBe(true);
  });

  it("returns nulls when the axis is missing on either side", () => {
    const a = makeDetail({ payload: {} });
    const b = makeDetail({
      payload: {
        multi_axis: { stance: { label: "neutral", confidence: 0.5 } },
      },
    });
    const d = computeMultiAxisDelta(a, b);
    expect(d.stanceRankDelta).toBe(null);
    expect(d.factorDelta).toBe(null);
    expect(d.certaintyShift).toBe("unknown");
    expect(d.topicChanged).toBe(null);
  });
});
