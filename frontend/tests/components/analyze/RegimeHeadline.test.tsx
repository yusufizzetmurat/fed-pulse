import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";

import { RegimeHeadline } from "@/components/analyze/RegimeHeadline";
import type { RegimeClassificationResponse } from "@/lib/analyze/types";

const REGIME: RegimeClassificationResponse = {
  predicted_set: ["calm", "normal"],
  set_label: "calm|normal",
  set_size: 2,
  coverage: 0.8,
  distribution: { calm: 0.35, normal: 0.45, high: 0.2 },
  argmax_class: "normal",
};

describe("RegimeHeadline coverage chip", () => {
  it("shows the run-level coverage badge when no empirical figure is provided", () => {
    render(<RegimeHeadline regime={REGIME} symbol="^GSPC" documentDate="2024-09-18" />);
    expect(screen.getByText(/80% coverage · set size 2/i)).toBeInTheDocument();
    expect(screen.queryByText(/empirical/i)).not.toBeInTheDocument();
  });

  it("merges nominal + empirical into one chip when empirical coverage is available", () => {
    render(
      <RegimeHeadline
        regime={REGIME}
        symbol="^GSPC"
        documentDate="2024-09-18"
        empiricalCoverage={0.76}
        empiricalCoverageSampleSize={42}
      />,
    );
    expect(screen.getByText(/Nominal 80% · Empirical 76%/i)).toBeInTheDocument();
  });

  it("renders the drift badge when empirical drops more than 10pp under nominal", () => {
    render(
      <RegimeHeadline
        regime={REGIME}
        symbol="^GSPC"
        documentDate="2024-09-18"
        empiricalCoverage={0.55}
        empiricalCoverageSampleSize={25}
      />,
    );
    expect(screen.getByText(/-25pp drift/i)).toBeInTheDocument();
  });

  it("hides empirical when sample size is zero (coverage not yet meaningful)", () => {
    render(
      <RegimeHeadline
        regime={REGIME}
        symbol="^GSPC"
        documentDate="2024-09-18"
        empiricalCoverage={0.5}
        empiricalCoverageSampleSize={0}
      />,
    );
    expect(screen.queryByText(/empirical/i)).not.toBeInTheDocument();
    expect(screen.getByText(/80% coverage/i)).toBeInTheDocument();
  });
});
