import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";

import { CredibilityPanel } from "@/components/analyze/CredibilityPanel";
import { SAMPLE_CREDIBILITY } from "@/lib/analyze/fixtures";

describe("CredibilityPanel", () => {
  it("renders the four credibility surface items from the fixture", () => {
    render(<CredibilityPanel credibility={SAMPLE_CREDIBILITY} />);
    expect(screen.getByText(/shift vs\. last 4 statements/i)).toBeInTheDocument();
    expect(screen.getByText(/what was done vs\. said/i)).toBeInTheDocument();
    expect(screen.getByText(/gap to market expectations/i)).toBeInTheDocument();
    expect(screen.getByText(/time since last reversal/i)).toBeInTheDocument();
    expect(screen.getByText(/14 mo/)).toBeInTheDocument();
  });

  it("handles a sparse credibility payload without crashing", () => {
    render(<CredibilityPanel credibility={{ drift_score: 0.5 }} />);
    expect(screen.getAllByText(/—/).length).toBeGreaterThanOrEqual(3);
  });
});
