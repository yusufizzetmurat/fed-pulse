import { describe, expect, it } from "vitest";

import {
  buildRunSummaryCsv,
  buildRunSummaryRows,
  buildXaiSentencesRows,
} from "@/lib/export/run-export";
import type { HistoryDetail } from "@/lib/analyze/types";

function makeDetail(): HistoryDetail {
  return {
    id: "abcdef1234567890",
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
    argmax_regime: "high",
    argmax_probability: 0.62,
    regime_set_size: 2,
    payload: {
      sentiment: { label: "hawkish", score: 0.82, is_in_distribution: true },
      model: {
        checkpoint_loaded: true,
        runtime_mode: "fast",
        encoder_key: "finbert_fed_adjacent",
      },
      multi_axis: {
        stance: { label: "hawkish", confidence: 0.91 },
        factor: { value: 0.42, confidence: 0.18 },
        certainty: { label: "decisive", confidence: 0.74 },
        topic: { primary: "inflation", confidence: 0.66 },
      },
      credibility: {
        drift_score: 0.21,
        realized_vs_stated_gap: -0.03,
        market_implied_gap: 0,
        months_since_reversal: 12,
      },
      regime_classification: {
        argmax_class: "high",
        predicted_set: ["normal", "high"],
        set_size: 2,
        set_label: "{normal, high}",
        coverage: 0.8,
        distribution: { calm: 0.1, normal: 0.28, high: 0.62 },
      },
      xai: {
        method: "keyword_salience_v1",
        sentences: [
          {
            text: "Inflation remains elevated.",
            score: 0.6,
            topTokens: [{ token: "inflation", weight: 0.5 }],
          },
        ],
      },
    },
  };
}

describe("buildRunSummaryRows", () => {
  it("starts with the schema header row", () => {
    const rows = buildRunSummaryRows(makeDetail());
    expect(rows[0]).toEqual(["field", "value"]);
  });

  it("emits regime, multi-axis, and credibility fields", () => {
    const rows = buildRunSummaryRows(makeDetail());
    const asObject: Record<string, unknown> = {};
    for (const row of rows.slice(1)) {
      asObject[String(row[0])] = row[1];
    }
    expect(asObject.run_id).toBe("abcdef1234567890");
    expect(asObject["sentiment.label"]).toBe("hawkish");
    expect(asObject["sentiment.score"]).toBe(0.82);
    expect(asObject["regime.argmax"]).toBe("high");
    expect(asObject["regime.set"]).toBe("normal|high");
    expect(asObject["regime.argmax_probability"]).toBe(0.62);
    expect(asObject["multi_axis.stance.label"]).toBe("hawkish");
    expect(asObject["multi_axis.factor.value"]).toBe(0.42);
    expect(asObject["credibility.drift_score"]).toBe(0.21);
    expect(asObject["model.encoder_key"]).toBe("finbert_fed_adjacent");
  });

  it("keeps the schema stable when the payload is empty", () => {
    const empty: HistoryDetail = { ...makeDetail(), payload: {} };
    const rows = buildRunSummaryRows(empty);
    expect(rows.length).toBe(buildRunSummaryRows(makeDetail()).length);
    expect(rows[1][0]).toBe("run_id");
  });
});

describe("buildXaiSentencesRows", () => {
  it("emits one row per sentence with top-token weights", () => {
    const rows = buildXaiSentencesRows(makeDetail());
    expect(rows[0]).toEqual(["sentence_index", "score", "top_tokens", "text"]);
    expect(rows.length).toBe(2);
    const [, score, topTokens, text] = rows[1];
    expect(score).toBe(0.6);
    expect(topTokens).toContain("inflation");
    expect(text).toBe("Inflation remains elevated.");
  });

  it("returns empty when the run has no xai payload", () => {
    const empty: HistoryDetail = { ...makeDetail(), payload: {} };
    expect(buildXaiSentencesRows(empty)).toEqual([]);
  });
});

describe("buildRunSummaryCsv", () => {
  it("produces a filename anchored to symbol + date + run-id prefix", () => {
    const { filename } = buildRunSummaryCsv(makeDetail());
    expect(filename).toContain("GSPC");
    expect(filename).toContain("2026-05-01");
    expect(filename.endsWith(".csv")).toBe(true);
  });

  it("includes the summary block and xai_sentences section", () => {
    const { csv } = buildRunSummaryCsv(makeDetail());
    expect(csv).toContain("field,value");
    expect(csv).toContain("regime.argmax");
    expect(csv).toContain("xai_sentences");
    expect(csv).toContain("inflation");
  });
});
