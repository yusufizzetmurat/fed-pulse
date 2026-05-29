import { describe, expect, it } from "vitest";

import {
  bandLabel,
  errorToneLabel,
  formatPercentDelta,
  formatPrice,
  formatPriceDelta,
  formatVol,
  getErrorTone,
  normalizeTimestamp,
  stanceLabel,
  toNumericOrNull,
  toStance,
} from "@/lib/analyze/format";

describe("format helpers", () => {
  it("toNumericOrNull returns null for non-numeric", () => {
    expect(toNumericOrNull("abc")).toBeNull();
    expect(toNumericOrNull(null)).toBeNull();
    expect(toNumericOrNull(undefined)).toBeNull();
  });

  it("toNumericOrNull parses numeric strings", () => {
    expect(toNumericOrNull("4.5")).toBe(4.5);
  });

  it("normalizeTimestamp strips trailing timezone offset", () => {
    expect(normalizeTimestamp("2024-09-18T00:00:00+00:00")).toBe("2024-09-18T00:00:00");
    expect(normalizeTimestamp(null)).toBe("");
  });

  it("toStance maps known stance labels", () => {
    expect(toStance("HAWKISH")).toBe("hawkish");
    expect(toStance("hawkish")).toBe("hawkish");
    expect(toStance("Dovish")).toBe("dovish");
    expect(toStance("Neutral")).toBe("neutral");
    expect(toStance("LABEL_0")).toBe("dovish");
    expect(toStance("LABEL_1")).toBe("neutral");
    expect(toStance("LABEL_2")).toBe("hawkish");
    expect(toStance("???")).toBe("unknown");
  });

  it("toStance does NOT silently relabel sst-2 news-sentiment labels as monetary stance", () => {
    // Regression guard for the 'banana banana banana' bug. When the FOMC
    // sentiment model fails to load and the backend falls back to
    // distilbert-sst-2, the labels are POSITIVE / NEGATIVE — news-sentiment,
    // not monetary-policy stance. The UI must surface this as "unknown" so
    // the dashboard renders "Sentiment unavailable" instead of pretending
    // POSITIVE means hawkish.
    expect(toStance("POSITIVE")).toBe("unknown");
    expect(toStance("NEGATIVE")).toBe("unknown");
    expect(toStance("positive")).toBe("unknown");
    expect(toStance("negative")).toBe("unknown");
  });

  it("stanceLabel renders human-readable form", () => {
    expect(stanceLabel("hawkish")).toBe("Hawkish");
    expect(stanceLabel("unknown")).toBe("Unknown");
  });

  it("formatters guard against non-numeric input", () => {
    expect(formatPrice(null)).toBe("N/A");
    expect(formatPriceDelta(undefined)).toBe("N/A");
    expect(formatPercentDelta(NaN)).toBe("N/A");
    expect(formatVol(null)).toBe("N/A");
  });

  it("formatPriceDelta keeps sign", () => {
    expect(formatPriceDelta(5)).toBe("+$5");
    expect(formatPriceDelta(-2.5)).toBe("-$2.5");
  });

  it("getErrorTone bins by magnitude", () => {
    expect(getErrorTone("mape", 1.5)).toBe("low");
    expect(getErrorTone("mape", 3)).toBe("medium");
    expect(getErrorTone("mape", 8)).toBe("high");
    expect(getErrorTone("rmse", null)).toBe("neutral");
    expect(getErrorTone("rmse", 30, 5000)).toBe("low");
    expect(errorToneLabel("medium")).toBe("Medium error");
  });

  it("bandLabel marks conformal vs Gaussian-z vs unknown", () => {
    expect(bandLabel(80, "conformal")).toBe("80% calibrated range");
    expect(bandLabel(80, "gaussian_z")).toBe("80% Gaussian range");
    // Null / missing source falls back to the legacy generic label so older
    // history rows (saved before the band-source field existed) still render.
    expect(bandLabel(80, null)).toBe("80% confidence range");
    expect(bandLabel(80, undefined)).toBe("80% confidence range");
    expect(bandLabel(95, "conformal")).toBe("95% calibrated range");
  });
});
