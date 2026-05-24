import { describe, expect, it } from "vitest";

import {
  REGIME_CLASSES,
  aggregateRegimePerformance,
  buildRunRegimeRecord,
} from "@/lib/analyze/performance";
import type { HistoryEntry, HistoryRealizedResponse } from "@/lib/analyze/types";

function makeEntry(overrides: Partial<HistoryEntry> = {}): HistoryEntry {
  return {
    id: overrides.id ?? "run-1",
    created_at: "2026-05-24T12:00:00Z",
    symbol: overrides.symbol ?? "^GSPC",
    document_date: overrides.document_date ?? "2026-05-01",
    horizon: overrides.horizon ?? "10d",
    forecast_mode: "fast",
    stance: "neutral",
    sentiment_score: 0,
    predicted_close: null,
    current_close: null,
    predicted_volatility: null,
    text_excerpt: null,
    argmax_regime: overrides.argmax_regime ?? "normal",
    argmax_probability: overrides.argmax_probability ?? 0.5,
    regime_set_size: overrides.regime_set_size ?? 2,
  };
}

function makeRealized(overrides: Partial<HistoryRealizedResponse> = {}): HistoryRealizedResponse {
  return {
    run_id: overrides.run_id ?? "run-1",
    symbol: overrides.symbol ?? "^GSPC",
    document_date: overrides.document_date ?? "2026-05-01",
    horizon: overrides.horizon ?? "10d",
    timestamps: [],
    close: [],
    volatility: [],
    realized_regime: overrides.realized_regime ?? "normal",
  };
}

describe("buildRunRegimeRecord", () => {
  it("flags set-hit when the realised regime is inside the predicted set", () => {
    const record = buildRunRegimeRecord({
      entry: makeEntry({ argmax_regime: "normal" }),
      realized: makeRealized({ realized_regime: "high" }),
      payload: {
        regime_classification: { predicted_set: ["normal", "high"] },
      },
    });
    expect(record.setHit).toBe(true);
    expect(record.argmax).toBe("normal");
    expect(record.realized).toBe("high");
  });

  it("flags set-miss when the realised regime is outside the predicted set", () => {
    const record = buildRunRegimeRecord({
      entry: makeEntry({ argmax_regime: "normal" }),
      realized: makeRealized({ realized_regime: "calm" }),
      payload: {
        regime_classification: { predicted_set: ["normal", "high"] },
      },
    });
    expect(record.setHit).toBe(false);
  });

  it("leaves setHit null when either side is missing", () => {
    const noRealized = buildRunRegimeRecord({
      entry: makeEntry({ argmax_regime: "normal" }),
      realized: null,
      payload: { regime_classification: { predicted_set: ["normal"] } },
    });
    expect(noRealized.setHit).toBe(null);

    const noPayload = buildRunRegimeRecord({
      entry: makeEntry({ argmax_regime: "normal" }),
      realized: makeRealized({ realized_regime: "high" }),
      payload: null,
    });
    expect(noPayload.setHit).toBe(null);
  });
});

describe("aggregateRegimePerformance", () => {
  it("computes argmax accuracy and empirical coverage from resolved runs", () => {
    // Three predictions, all argmax=normal. Realised: normal, normal, high.
    // Predicted sets: {calm,normal}, {normal,high}, {normal} respectively.
    const rows = [
      buildRunRegimeRecord({
        entry: makeEntry({ id: "a", argmax_regime: "normal" }),
        realized: makeRealized({ run_id: "a", realized_regime: "normal" }),
        payload: { regime_classification: { predicted_set: ["calm", "normal"] } },
      }),
      buildRunRegimeRecord({
        entry: makeEntry({ id: "b", argmax_regime: "normal" }),
        realized: makeRealized({ run_id: "b", realized_regime: "normal" }),
        payload: { regime_classification: { predicted_set: ["normal", "high"] } },
      }),
      buildRunRegimeRecord({
        entry: makeEntry({ id: "c", argmax_regime: "normal" }),
        realized: makeRealized({ run_id: "c", realized_regime: "high" }),
        payload: { regime_classification: { predicted_set: ["normal"] } },
      }),
    ];
    const agg = aggregateRegimePerformance(rows);
    expect(agg.resolved).toBe(3);
    expect(agg.argmaxAccuracy).toBeCloseTo(2 / 3, 5);
    expect(agg.empiricalCoverage).toBeCloseTo(2 / 3, 5);
    // Only the "normal" class has full P/R signal here; calm/high have
    // zero TP and zero FP so their F1 is null and the macro fallback
    // returns null rather than averaging over an incomplete set.
    expect(agg.macroF1).toBe(null);
  });

  it("falls back to empty perClass entries when no runs resolve", () => {
    const agg = aggregateRegimePerformance([]);
    expect(agg.resolved).toBe(0);
    expect(agg.argmaxAccuracy).toBe(null);
    expect(agg.empiricalCoverage).toBe(null);
    expect(agg.macroF1).toBe(null);
    expect(agg.perClass.map((entry) => entry.klass)).toEqual([...REGIME_CLASSES]);
  });
});
