import { describe, expect, it } from "vitest";

import {
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

  it("toStance maps known labels", () => {
    expect(toStance("HAWKISH")).toBe("hawkish");
    expect(toStance("LABEL_0")).toBe("dovish");
    expect(toStance("Neutral")).toBe("neutral");
    expect(toStance("???")).toBe("unknown");
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
});
