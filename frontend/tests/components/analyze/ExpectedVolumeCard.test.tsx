import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";

import { ExpectedVolumeCard } from "@/components/analyze/ExpectedVolumeCard";
import type { ExpectedVolumeForecastResponse } from "@/lib/analyze/types";

function fixture(): ExpectedVolumeForecastResponse {
  return {
    symbol: "^GSPC",
    horizons: [
      {
        h: 1,
        point_log_residual: 0.118,
        point_pct_vs_baseline: 12.5,
        band_lo_80: 2.1,
        band_hi_80: 23.4,
        band_lo_90: -1.4,
        band_hi_90: 28.2,
        r2_har: 0.82,
        calendar_adjusted: true,
      },
      {
        h: 5,
        point_log_residual: 0.052,
        point_pct_vs_baseline: 5.3,
        band_lo_80: -2.2,
        band_hi_80: 13.1,
        band_lo_90: -5.0,
        band_hi_90: 16.4,
        r2_har: 0.74,
        calendar_adjusted: true,
      },
      {
        h: 22,
        point_log_residual: -0.034,
        point_pct_vs_baseline: -3.3,
        band_lo_80: -10.1,
        band_hi_80: 4.2,
        band_lo_90: -13.5,
        band_hi_90: 8.0,
        r2_har: 0.61,
        calendar_adjusted: false,
      },
    ],
    model_revision: "volume_har@2026-05-30",
    generated_at: "2026-05-31T08:00:00+00:00",
  };
}

describe("ExpectedVolumeCard", () => {
  it("renders one column per forecast horizon", () => {
    render(<ExpectedVolumeCard forecast={fixture()} />);
    expect(screen.getByText(/1 day/i)).toBeInTheDocument();
    expect(screen.getByText(/1 week/i)).toBeInTheDocument();
    expect(screen.getByText(/1 month/i)).toBeInTheDocument();
  });

  it("formats the headline as percent vs baseline", () => {
    render(<ExpectedVolumeCard forecast={fixture()} />);
    expect(screen.getByTestId("expected-volume-headline-1")).toHaveTextContent(
      /\+12\.5% vs baseline/i,
    );
    expect(screen.getByTestId("expected-volume-headline-22")).toHaveTextContent(
      /-3\.3% vs baseline/i,
    );
  });

  it("shows the log-residual subscript on every column", () => {
    render(<ExpectedVolumeCard forecast={fixture()} />);
    expect(screen.getByTestId("expected-volume-subscript-1")).toHaveTextContent(
      /\+0\.118 log-residual/i,
    );
    expect(screen.getByTestId("expected-volume-subscript-22")).toHaveTextContent(
      /-0\.034 log-residual/i,
    );
  });

  it("renders the 80% and 90% band ranges", () => {
    render(<ExpectedVolumeCard forecast={fixture()} />);
    expect(screen.getByTestId("expected-volume-band80-1")).toHaveTextContent(
      /\+2\.1% – \+23\.4%/i,
    );
    expect(screen.getByTestId("expected-volume-band90-1")).toHaveTextContent(
      /-1\.4% – \+28\.2%/i,
    );
  });

  it("surfaces the R² chip per horizon", () => {
    render(<ExpectedVolumeCard forecast={fixture()} />);
    expect(screen.getByTestId("expected-volume-r2-1")).toHaveTextContent(
      /R² 0\.82/i,
    );
    expect(screen.getByTestId("expected-volume-r2-22")).toHaveTextContent(
      /R² 0\.61/i,
    );
  });

  it("shows the calendar-adjusted chip only when the artifact applied it", () => {
    render(<ExpectedVolumeCard forecast={fixture()} />);
    expect(screen.getByTestId("expected-volume-cal-1")).toBeInTheDocument();
    expect(
      screen.queryByTestId("expected-volume-cal-22"),
    ).not.toBeInTheDocument();
  });

  it("renders the workspace forecast badge", () => {
    render(<ExpectedVolumeCard forecast={fixture()} />);
    const badge = screen.getByTestId("workspace-section-badge");
    expect(badge).toHaveTextContent(/forecast/i);
  });

  it("shows a loading state while the request is in flight", () => {
    render(<ExpectedVolumeCard forecast={null} loading />);
    expect(screen.getByTestId("expected-volume-loading")).toBeInTheDocument();
  });

  it("shows an error state when the request fails", () => {
    render(
      <ExpectedVolumeCard
        forecast={null}
        error="503: model_unavailable"
      />,
    );
    expect(
      screen.getByTestId("expected-volume-unavailable"),
    ).toHaveTextContent(/503: model_unavailable/i);
  });

  it("shows the empty state when the response carries no horizons", () => {
    render(
      <ExpectedVolumeCard
        forecast={{
          symbol: "^GSPC",
          horizons: [],
          model_revision: "volume_har@2026-05-30",
          generated_at: "2026-05-31T08:00:00+00:00",
        }}
      />,
    );
    expect(
      screen.getByTestId("expected-volume-unavailable"),
    ).toBeInTheDocument();
  });
});
