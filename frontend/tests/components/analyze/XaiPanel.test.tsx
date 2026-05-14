import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";

import { XaiPanel } from "@/components/analyze/XaiPanel";
import { TooltipProvider } from "@/components/ui/tooltip";
import { SAMPLE_XAI } from "@/lib/analyze/fixtures";

function renderWithTooltip(node: React.ReactNode) {
  return render(<TooltipProvider>{node}</TooltipProvider>);
}

describe("XaiPanel", () => {
  it("renders the title and one chip per sentence", () => {
    renderWithTooltip(<XaiPanel xai={SAMPLE_XAI} />);
    expect(screen.getByText(/sentence attribution/i)).toBeInTheDocument();
    for (const sentence of SAMPLE_XAI.sentences) {
      expect(screen.getByText(sentence.text)).toBeInTheDocument();
    }
  });

  it("renders the empty-state placeholder when there are no sentences", () => {
    renderWithTooltip(<XaiPanel xai={{ sentences: [] }} />);
    expect(screen.getByText(/sentence attribution/i)).toBeInTheDocument();
    expect(screen.getByText(/no salient sentences detected/i)).toBeInTheDocument();
  });

  it("shows the integrated-gradients method label", () => {
    renderWithTooltip(<XaiPanel xai={SAMPLE_XAI} />);
    expect(screen.getByText(/integrated_gradients/i)).toBeInTheDocument();
  });
});
