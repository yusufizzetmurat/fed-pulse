import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";

import { VolatilityOutlookCard } from "@/components/analyze/VolatilityOutlookCard";
import type { RealizedVolForecastResponse } from "@/lib/analyze/types";

function fixture(): RealizedVolForecastResponse {
  return {
    symbol: "^GSPC",
    horizons: [
      {
        h: 1,
        point: 1e-4,
        band_lo_80: 5e-5,
        band_hi_80: 2e-4,
        band_lo_90: 4e-5,
        band_hi_90: 3e-4,
        qlike_model: 0.197,
        qlike_har: 0.223,
        coverage_empirical_90: 0.855,
      },
      {
        h: 5,
        point: 1.2e-4,
        band_lo_80: 6e-5,
        band_hi_80: 2.2e-4,
        band_lo_90: 5e-5,
        band_hi_90: 3.2e-4,
        qlike_model: 0.197,
        qlike_har: 0.219,
        coverage_empirical_90: 0.876,
      },
      {
        h: 22,
        point: 1.5e-4,
        band_lo_80: 8e-5,
        band_hi_80: 2.5e-4,
        band_lo_90: 7e-5,
        band_hi_90: 3.5e-4,
        qlike_model: 0.327,
        qlike_har: 0.360,
        coverage_empirical_90: 0.919,
      },
    ],
    history: Array.from({ length: 30 }, (_, i) => 1e-4 + i * 1e-6),
    history_dates: Array.from(
      { length: 30 },
      (_, i) => `2026-04-${String(i + 1).padStart(2, "0")}`,
    ),
    model_revision: "intraday_rv_production@2026-05-29",
  };
}

describe("VolatilityOutlookCard", () => {
  it("renders one column per forecast horizon", () => {
    render(<VolatilityOutlookCard forecast={fixture()} />);
    expect(screen.getByText(/1 day/i)).toBeInTheDocument();
    expect(screen.getByText(/1 week/i)).toBeInTheDocument();
    expect(screen.getByText(/1 month/i)).toBeInTheDocument();
  });

  it("shows the beats-HAR badge when qlike_model < qlike_har", () => {
    render(<VolatilityOutlookCard forecast={fixture()} />);
    const beats = screen.getAllByText(/beats HAR by/i);
    // Three horizons, three badges.
    expect(beats.length).toBe(3);
  });

  it("hides the beats-HAR badge when the model loses to HAR", () => {
    const data = fixture();
    data.horizons.forEach((h) => {
      h.qlike_model = 0.4;
      h.qlike_har = 0.3;
    });
    render(<VolatilityOutlookCard forecast={data} />);
    expect(screen.queryByText(/beats HAR by/i)).toBeNull();
  });

  it("surfaces an inline message when the model artifact is unavailable", () => {
    render(<VolatilityOutlookCard forecast={null} error="503: model_unavailable" />);
    expect(screen.getByText(/Forecast unavailable/i)).toBeInTheDocument();
    expect(screen.getByText(/503/i)).toBeInTheDocument();
  });

  it("shows a loading placeholder while the request is in flight", () => {
    render(<VolatilityOutlookCard forecast={null} loading />);
    expect(screen.getByText(/Loading forecast/i)).toBeInTheDocument();
  });

  it("renders the historical 80% bands overlay when historical_bands is present", () => {
    const data = fixture();
    data.historical_bands = data.history_dates.slice(8).map((date, i) => ({
      date,
      band_lo_80: 7e-5 + i * 1e-6,
      band_hi_80: 2.5e-4 + i * 1e-6,
      realized_rv: 1e-4 + i * 1e-6,
    }));
    render(<VolatilityOutlookCard forecast={data} />);
    // Legend chip surfaces only when bands are present.
    expect(screen.getByText(/Past 80% bands/i)).toBeInTheDocument();
    expect(screen.getByTestId("rv-bands-legend")).toBeInTheDocument();
  });

  it("falls back to the bare sparkline when historical_bands is absent", () => {
    const data = fixture();
    data.historical_bands = null;
    render(<VolatilityOutlookCard forecast={data} />);
    expect(screen.queryByText(/Past 80% bands/i)).toBeNull();
    expect(screen.queryByTestId("rv-bands-legend")).toBeNull();
  });

  it("falls back cleanly when historical_bands is an empty list", () => {
    const data = fixture();
    data.historical_bands = [];
    render(<VolatilityOutlookCard forecast={data} />);
    expect(screen.queryByText(/Past 80% bands/i)).toBeNull();
    expect(screen.queryByTestId("rv-bands-legend")).toBeNull();
  });
});
