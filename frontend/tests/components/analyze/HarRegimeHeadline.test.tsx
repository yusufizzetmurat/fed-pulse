import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";

import { HarRegimeHeadline } from "@/components/analyze/HarRegimeHeadline";
import type { HarTercileBaselineResponse } from "@/lib/analyze/types";

function fixture(): HarTercileBaselineResponse {
  return {
    symbol: "^GSPC",
    horizons: [
      {
        h: 1,
        top_pick: "medium",
        probabilities: { low: 0.18, medium: 0.55, high: 0.27 },
        predicted_rv: 1e-4,
        macro_f1: 0.687,
        n: 412,
      },
      {
        h: 5,
        top_pick: "medium",
        probabilities: { low: 0.2, medium: 0.52, high: 0.28 },
        predicted_rv: 1.2e-4,
        macro_f1: 0.685,
        n: 408,
      },
      {
        h: 22,
        top_pick: "high",
        probabilities: { low: 0.15, medium: 0.3, high: 0.55 },
        predicted_rv: 1.5e-4,
        macro_f1: 0.654,
        n: 391,
      },
    ],
    source_wiki_section: "20_Gated_Fusion_InfoNCE_Comprehensive_Null",
  };
}

describe("HarRegimeHeadline", () => {
  it("renders one column per horizon", () => {
    render(<HarRegimeHeadline baselines={fixture()} symbol="^GSPC" />);
    expect(screen.getByText(/1 day/i)).toBeInTheDocument();
    expect(screen.getByText(/1 week/i)).toBeInTheDocument();
    expect(screen.getByText(/1 month/i)).toBeInTheDocument();
  });

  it("shows the macro-F1 chip with horizon-specific values", () => {
    render(<HarRegimeHeadline baselines={fixture()} symbol="^GSPC" />);
    expect(screen.getByText(/macro-F1 0\.687 \(wiki §20, n=412\)/i)).toBeInTheDocument();
    expect(screen.getByText(/macro-F1 0\.685 \(wiki §20, n=408\)/i)).toBeInTheDocument();
    expect(screen.getByText(/macro-F1 0\.654 \(wiki §20, n=391\)/i)).toBeInTheDocument();
  });

  it("renders the HEADLINE primacy badge", () => {
    render(<HarRegimeHeadline baselines={fixture()} symbol="^GSPC" />);
    expect(screen.getByText(/headline/i)).toBeInTheDocument();
  });

  it("renders the disclosure caption referencing the wiki source", () => {
    render(<HarRegimeHeadline baselines={fixture()} symbol="^GSPC" />);
    expect(
      screen.getByText(/Beats both market-only and fused text\+market models/i),
    ).toBeInTheDocument();
  });

  it("renders annualized predicted RV beside the tercile chip", () => {
    render(<HarRegimeHeadline baselines={fixture()} symbol="^GSPC" />);
    // sqrt(1e-4 * 252) * 100 ≈ 15.9% — fixture h=1 column
    expect(screen.getByText(/predicted RV 15\.9%/i)).toBeInTheDocument();
  });

  it("shows a skeleton loading state", () => {
    render(<HarRegimeHeadline baselines={null} loading symbol="^GSPC" />);
    expect(screen.getByText(/Loading HAR baseline/i)).toBeInTheDocument();
  });

  it("shows an error state when the request fails", () => {
    render(
      <HarRegimeHeadline
        baselines={null}
        error="503: baseline_unavailable"
        symbol="^GSPC"
      />,
    );
    expect(screen.getByText(/HAR baseline unavailable/i)).toBeInTheDocument();
    expect(screen.getByText(/503: baseline_unavailable/i)).toBeInTheDocument();
  });

  it("shows the empty state when the response carries no horizons", () => {
    render(
      <HarRegimeHeadline
        baselines={{
          symbol: "^GSPC",
          horizons: [],
          source_wiki_section: "20_Gated_Fusion_InfoNCE_Comprehensive_Null",
        }}
        symbol="^GSPC"
      />,
    );
    expect(screen.getByText(/HAR baseline unavailable/i)).toBeInTheDocument();
  });
});
