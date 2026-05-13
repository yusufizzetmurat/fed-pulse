import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";

import { XaiPanel } from "@/components/analyze/XaiPanel";
import { SAMPLE_XAI } from "@/lib/analyze/fixtures";

describe("XaiPanel", () => {
  it("renders the title and one chip per sentence", () => {
    render(<XaiPanel xai={SAMPLE_XAI} />);
    expect(screen.getByText(/sentence attribution/i)).toBeInTheDocument();
    for (const sentence of SAMPLE_XAI.sentences) {
      expect(screen.getByText(sentence.text)).toBeInTheDocument();
    }
  });

  it("returns null when there are no sentences", () => {
    const { container } = render(<XaiPanel xai={{ sentences: [] }} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("shows the integrated-gradients method label", () => {
    render(<XaiPanel xai={SAMPLE_XAI} />);
    expect(screen.getByText(/integrated_gradients/i)).toBeInTheDocument();
  });
});
