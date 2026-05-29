import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";

import { PanelAttributionRow, XaiPanel } from "@/components/analyze/XaiPanel";
import { TooltipProvider } from "@/components/ui/tooltip";
import { SAMPLE_XAI } from "@/lib/analyze/fixtures";
import type { XaiPanelAttribution } from "@/lib/analyze/types";

function renderWithTooltip(node: React.ReactNode) {
  return render(<TooltipProvider>{node}</TooltipProvider>);
}

describe("XaiPanel", () => {
  it("renders the title and one chip per sentence", () => {
    renderWithTooltip(<XaiPanel xai={SAMPLE_XAI} />);
    expect(screen.getByText(/per-sentence explanation/i)).toBeInTheDocument();
    for (const sentence of SAMPLE_XAI.sentences) {
      expect(screen.getByText(sentence.text)).toBeInTheDocument();
    }
  });

  it("renders the empty-state placeholder when there are no sentences", () => {
    renderWithTooltip(<XaiPanel xai={{ sentences: [] }} />);
    expect(screen.getByText(/per-sentence explanation/i)).toBeInTheDocument();
    expect(screen.getByText(/no high-impact sentences found/i)).toBeInTheDocument();
  });

  it("shows the integrated-gradients method label", () => {
    renderWithTooltip(<XaiPanel xai={SAMPLE_XAI} />);
    expect(screen.getByText(/integrated_gradients/i)).toBeInTheDocument();
  });

  it("renders feature-family bars for each panel attribution", () => {
    const panels: XaiPanelAttribution[] = [
      {
        panel: "regime",
        target: "argmax_logit[0]",
        families: [
          { family: "linguistic", magnitude: 0.42, signed: 0.31 },
          { family: "credibility", magnitude: 0.18, signed: -0.14 },
        ],
        n_steps: 20,
        unavailable: false,
        reason: null,
      },
    ];
    renderWithTooltip(
      <XaiPanel xai={{ ...SAMPLE_XAI, panels }} />,
    );
    expect(screen.getByTestId("panel-attributions")).toBeInTheDocument();
    expect(screen.getByTestId("panel-attribution-regime")).toBeInTheDocument();
    expect(screen.getByText("Linguistic")).toBeInTheDocument();
    expect(screen.getByText("Credibility")).toBeInTheDocument();
    expect(screen.getByText(/20 attribution steps/)).toBeInTheDocument();
  });

  it("renders the explanation-unavailable badge when a panel is unavailable", () => {
    const panel: XaiPanelAttribution = {
      panel: "rates_2y",
      target: "rates_2y_bps",
      families: [],
      n_steps: 0,
      unavailable: true,
      reason: "head_not_mounted",
    };
    renderWithTooltip(<PanelAttributionRow panel={panel} />);
    expect(screen.getByTestId("panel-attribution-rates_2y")).toBeInTheDocument();
    expect(screen.getByText(/explanation not available/i)).toBeInTheDocument();
    expect(
      screen.getByText(/This prediction is not enabled on the active model\./i),
    ).toBeInTheDocument();
  });

  it("does not render the panel section when no panels are supplied", () => {
    renderWithTooltip(<XaiPanel xai={SAMPLE_XAI} />);
    expect(screen.queryByTestId("panel-attributions")).not.toBeInTheDocument();
  });
});
