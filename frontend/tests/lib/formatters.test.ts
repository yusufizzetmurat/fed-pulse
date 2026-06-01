import { describe, expect, it } from "vitest";

import {
  formatBps,
  formatLogResidual,
  formatPctVsBaseline,
  formatProbabilityPct,
} from "@/lib/analyze/formatters";

describe("formatBps", () => {
  it("renders a positive value with an explicit + sign by default", () => {
    expect(formatBps(12)).toBe("+12 bps");
  });

  it("renders a negative value with a minus sign", () => {
    expect(formatBps(-7)).toBe("-7 bps");
  });

  it("renders zero without a sign even when signed=true", () => {
    expect(formatBps(0)).toBe("0 bps");
  });

  it("drops the leading + when signed is false", () => {
    expect(formatBps(12, { signed: false })).toBe("12 bps");
    expect(formatBps(-12, { signed: false })).toBe("-12 bps");
  });

  it("honors fractional digits", () => {
    expect(formatBps(12.345, { fractionDigits: 1 })).toBe("+12.3 bps");
  });

  it("returns N/A for null/undefined/NaN", () => {
    expect(formatBps(null)).toBe("N/A");
    expect(formatBps(undefined)).toBe("N/A");
    expect(formatBps(Number.NaN)).toBe("N/A");
  });
});

describe("formatPctVsBaseline", () => {
  it("renders a positive percent with a + sign and the baseline tag", () => {
    expect(formatPctVsBaseline(12.3)).toBe("+12.3% vs baseline");
  });

  it("renders a negative percent with a minus sign", () => {
    expect(formatPctVsBaseline(-4.5)).toBe("-4.5% vs baseline");
  });

  it("returns N/A for null", () => {
    expect(formatPctVsBaseline(null)).toBe("N/A");
  });
});

describe("formatLogResidual", () => {
  it("renders a positive value with a + sign and three decimals", () => {
    expect(formatLogResidual(0.118)).toBe("+0.118");
  });

  it("renders a negative value with a minus sign", () => {
    expect(formatLogResidual(-0.05)).toBe("-0.050");
  });

  it("returns N/A for null", () => {
    expect(formatLogResidual(null)).toBe("N/A");
  });
});

describe("formatProbabilityPct", () => {
  it("renders a probability in [0,1] as an integer percent", () => {
    expect(formatProbabilityPct(0.62)).toBe("62%");
  });

  it("rounds to the nearest integer percent", () => {
    expect(formatProbabilityPct(0.6249)).toBe("62%");
    expect(formatProbabilityPct(0.625)).toBe("63%");
  });

  it("clamps values above 1.0 to 100%", () => {
    expect(formatProbabilityPct(1.0000001)).toBe("100%");
  });

  it("clamps negative values to 0%", () => {
    expect(formatProbabilityPct(-0.01)).toBe("0%");
  });

  it("returns N/A for null/undefined", () => {
    expect(formatProbabilityPct(null)).toBe("N/A");
    expect(formatProbabilityPct(undefined)).toBe("N/A");
  });
});
