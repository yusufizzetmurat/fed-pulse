import { describe, expect, it } from "vitest";

import {
  buildNarrativeSummary,
  computeCompareDelta,
  computeMultiAxisDelta,
  computeRegimeDelta,
  describeStanceShift,
  type MultiAxisDelta,
} from "@/lib/analyze/compare";
import type { HistoryDetail } from "@/lib/analyze/types";

function makeMultiAxisDelta(overrides: Partial<MultiAxisDelta> = {}): MultiAxisDelta {
  return {
    stanceRankDelta: null,
    stanceConfidenceDelta: null,
    factorDelta: null,
    factorConfidenceDelta: null,
    certaintyConfidenceDelta: null,
    certaintyShift: "unknown",
    ...overrides,
  };
}

function makeDetail(overrides: Partial<HistoryDetail> & { payload?: Record<string, unknown> }): HistoryDetail {
  return {
    id: "fixture",
    created_at: "2024-09-18T12:00:00Z",
    symbol: "^GSPC",
    document_date: "2024-09-18",
    horizon: "10d",
    forecast_mode: "fast",
    stance: "hawkish",
    sentiment_score: 0.8,
    predicted_close: null,
    current_close: null,
    predicted_volatility: null,
    payload: {},
    ...overrides,
  };
}

describe("computeCompareDelta", () => {
  it("computes regime, stance, and credibility deltas", () => {
    const a = makeDetail({
      payload: {
        sentiment: { label: "hawkish", score: 0.82 },
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
      payload: {
        sentiment: { label: "dovish", score: 0.71 },
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
      stance: "dovish",
    });
    const delta = computeCompareDelta(a, b);
    expect(delta.regime.argmaxA).toBe("high");
    expect(delta.regime.argmaxB).toBe("normal");
    expect(delta.regime.argmaxChanged).toBe(true);
    expect(delta.regime.setAddedToA).toContain("high");
    expect(delta.regime.setDroppedFromA).toContain("calm");
    expect(delta.stanceShift).toBe("more_hawkish");
    expect(delta.scoreDelta).toBeCloseTo(0.11, 5);
    expect(delta.driftDelta).toBeCloseTo(0.14, 5);
    expect(delta.realizedGapDelta).toBeCloseTo(0.16, 5);
  });

  it("falls back to history-row stance when payload sentiment is absent", () => {
    const a = makeDetail({ stance: "dovish", payload: {} });
    const b = makeDetail({ stance: "hawkish", payload: {} });
    const delta = computeCompareDelta(a, b);
    expect(delta.stanceShift).toBe("more_dovish");
    expect(delta.regime.argmaxA).toBe(null);
    expect(delta.regime.argmaxChanged).toBe(null);
  });

  it("describes stance shifts in human-readable form", () => {
    expect(describeStanceShift("more_hawkish")).toMatch(/hawkish/i);
    expect(describeStanceShift("more_dovish")).toMatch(/dovish/i);
    expect(describeStanceShift("unchanged")).toMatch(/unchanged/i);
    expect(describeStanceShift("unknown")).toMatch(/unknown/i);
  });
});

describe("computeRegimeDelta", () => {
  it("reports argmax probability delta when both sides share the argmax", () => {
    const a = makeDetail({
      payload: {
        regime_classification: {
          argmax_class: "high",
          predicted_set: ["high"],
          set_size: 1,
          coverage: 0.8,
          distribution: { calm: 0.1, normal: 0.2, high: 0.7 },
          set_label: "{high}",
        },
      },
    });
    const b = makeDetail({
      payload: {
        regime_classification: {
          argmax_class: "high",
          predicted_set: ["normal", "high"],
          set_size: 2,
          coverage: 0.8,
          distribution: { calm: 0.05, normal: 0.4, high: 0.55 },
          set_label: "{normal, high}",
        },
      },
    });
    const delta = computeRegimeDelta(a, b);
    expect(delta.argmaxChanged).toBe(false);
    expect(delta.argmaxProbDelta).toBeCloseTo(0.15, 5);
    expect(delta.setSizeA).toBe(1);
    expect(delta.setSizeB).toBe(2);
    expect(delta.setDroppedFromA).toContain("normal");
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
        },
      },
    });
    const b = makeDetail({
      payload: {
        multi_axis: {
          stance: { label: "dovish", confidence: 0.8 },
          factor: { value: -0.1, confidence: 0.25 },
          certainty: { label: "tentative", confidence: 0.55 },
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
  });
});

describe("buildNarrativeSummary", () => {
  it("leads with the largest absolute stance delta when stance dominates", () => {
    const sentence = buildNarrativeSummary(
      makeMultiAxisDelta({ stanceRankDelta: 1.5, factorDelta: 0.2 }),
    );
    expect(sentence).not.toBeNull();
    // Stance wins the primary clause.
    expect(sentence!.startsWith("Run A is more hawkish on rate guidance")).toBe(true);
    // Factor is demoted to the secondary clause.
    expect(sentence).toContain("but more hawkish on inflation tone");
  });

  it("leads with the largest absolute factor delta when factor dominates", () => {
    const sentence = buildNarrativeSummary(
      makeMultiAxisDelta({ stanceRankDelta: 0.1, factorDelta: -0.9 }),
    );
    expect(sentence).not.toBeNull();
    // Factor leads with a dovish framing.
    expect(sentence!.startsWith("more dovish on inflation tone")).toBe(true);
  });

  it("returns null when both axes carry a zero delta", () => {
    const sentence = buildNarrativeSummary(
      makeMultiAxisDelta({ stanceRankDelta: 0, factorDelta: 0 }),
    );
    expect(sentence).toBeNull();
  });

  it("returns null when neither axis is present", () => {
    expect(buildNarrativeSummary(makeMultiAxisDelta())).toBeNull();
  });

  it("handles mixed signs across the two axes", () => {
    const sentence = buildNarrativeSummary(
      makeMultiAxisDelta({ stanceRankDelta: 0.8, factorDelta: -0.4 }),
    );
    expect(sentence).not.toBeNull();
    expect(sentence).toContain("more hawkish on rate guidance");
    expect(sentence).toContain("more dovish on inflation tone");
  });

  it("returns null when both inputs are NaN", () => {
    const sentence = buildNarrativeSummary(
      makeMultiAxisDelta({ stanceRankDelta: Number.NaN, factorDelta: Number.NaN }),
    );
    expect(sentence).toBeNull();
  });
});
