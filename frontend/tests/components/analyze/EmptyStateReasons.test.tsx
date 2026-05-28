import { describe, expect, it } from "vitest";
import { render as rtlRender, screen } from "@testing-library/react";

import { CredibilityKpis } from "@/components/analyze/CredibilityKpis";
import { MultiAxisInterpretation } from "@/components/analyze/MultiAxisInterpretation";
import { TooltipProvider } from "@/components/ui/tooltip";

function render(ui: React.ReactElement) {
  return rtlRender(<TooltipProvider>{ui}</TooltipProvider>);
}

describe("Inline 'awaiting checkpoint' reasons", () => {
  it("MultiAxisInterpretation explains why no axes are present when all four are null", () => {
    render(
      <MultiAxisInterpretation
        multiAxis={{ stance: null, factor: null, certainty: null, topic: null }}
      />,
    );
    expect(
      screen.getByText(/sentiment breakdown returned no labels/i),
    ).toBeInTheDocument();
    expect(screen.getByText(/active model file is missing/i)).toBeInTheDocument();
  });

  it("MultiAxisInterpretation renders the tile grid when at least one axis is present", () => {
    render(
      <MultiAxisInterpretation
        multiAxis={{
          stance: { label: "hawkish", confidence: 0.7 },
          factor: null,
          certainty: null,
          topic: null,
        }}
      />,
    );
    expect(
      screen.queryByText(/sentiment breakdown returned no labels/i),
    ).not.toBeInTheDocument();
    expect(screen.getByText(/stance/i)).toBeInTheDocument();
  });

  it("CredibilityKpis explains why every value is at its placeholder", () => {
    render(
      <CredibilityKpis
        credibility={{
          drift_score: 0,
          drift_trend: [],
          realized_vs_stated_gap: null,
          market_implied_gap: 0,
          months_since_reversal: null,
        }}
      />,
    );
    expect(screen.getByText(/credibility signals not yet available/i)).toBeInTheDocument();
    expect(screen.getByText(/federal funds rate/i)).toBeInTheDocument();
  });

  it("CredibilityKpis renders the KPI tiles when drift trend has real history", () => {
    render(
      <CredibilityKpis
        credibility={{
          drift_score: 0.42,
          drift_trend: [0.3, 0.36, 0.42],
          realized_vs_stated_gap: 0.12,
          market_implied_gap: 0,
          months_since_reversal: 9,
        }}
      />,
    );
    expect(screen.queryByText(/credibility signals not yet available/i)).not.toBeInTheDocument();
    expect(screen.getByText(/shift score/i)).toBeInTheDocument();
  });
});
