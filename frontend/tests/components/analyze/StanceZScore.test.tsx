import { describe, expect, it } from "vitest";
import { render as rtlRender, screen } from "@testing-library/react";

import { MultiAxisInterpretation } from "@/components/analyze/MultiAxisInterpretation";
import { TooltipProvider } from "@/components/ui/tooltip";

function render(ui: React.ReactElement) {
  return rtlRender(<TooltipProvider>{ui}</TooltipProvider>);
}

const HAWKISH_STANCE = {
  label: "hawkish" as const,
  confidence: 0.95,
  // s = P(hawk) - P(dove) = 0.95 - 0.04 = 0.91
  distribution: { hawkish: 0.95, dovish: 0.04, neutral: 0.01 },
};

describe("StanceTile rolling z-score", () => {
  it("renders the z-score badge when a usable trailing context is provided", () => {
    // baseline mean 0.5, std 0.2 → current s=0.91 → z = +2.05σ
    render(
      <MultiAxisInterpretation
        multiAxis={{ stance: HAWKISH_STANCE, factor: null, certainty: null }}
        stanceContext={{
          n: 10,
          mean: 0.5,
          std: 0.2,
          history: [],
        }}
      />,
    );
    expect(screen.getByTestId("stance-zscore")).toBeInTheDocument();
    expect(screen.getByTestId("stance-zscore").textContent).toMatch(/\+2\.05σ/);
    // Caption flips to the rolling-z explainer when z is shown.
    expect(
      screen.getByText(/Rolling z-score vs recent meetings/i),
    ).toBeInTheDocument();
  });

  it("falls back to the raw confidence badge when context is null", () => {
    render(
      <MultiAxisInterpretation
        multiAxis={{ stance: HAWKISH_STANCE, factor: null, certainty: null }}
        stanceContext={null}
      />,
    );
    expect(screen.queryByTestId("stance-zscore")).not.toBeInTheDocument();
    expect(
      screen.getByText(/Hawkish \(\+\) favours tighter policy/i),
    ).toBeInTheDocument();
  });

  it("falls back when fewer than two history rows are usable", () => {
    render(
      <MultiAxisInterpretation
        multiAxis={{ stance: HAWKISH_STANCE, factor: null, certainty: null }}
        stanceContext={{ n: 1, mean: 0.6, std: null, history: [] }}
      />,
    );
    expect(screen.queryByTestId("stance-zscore")).not.toBeInTheDocument();
  });

  it("falls back when the trailing series is degenerate (std=0)", () => {
    // A constant series has no meaningful spread — the z-score would be
    // undefined, so the tile must NOT render one.
    render(
      <MultiAxisInterpretation
        multiAxis={{ stance: HAWKISH_STANCE, factor: null, certainty: null }}
        stanceContext={{ n: 5, mean: 0.5, std: 0, history: [] }}
      />,
    );
    expect(screen.queryByTestId("stance-zscore")).not.toBeInTheDocument();
  });

  it("falls back when the current stance has no distribution to score", () => {
    render(
      <MultiAxisInterpretation
        multiAxis={{
          stance: { label: "hawkish", confidence: 0.8 }, // no distribution
          factor: null,
          certainty: null,
        }}
        stanceContext={{ n: 10, mean: 0.5, std: 0.2, history: [] }}
      />,
    );
    expect(screen.queryByTestId("stance-zscore")).not.toBeInTheDocument();
  });
});
