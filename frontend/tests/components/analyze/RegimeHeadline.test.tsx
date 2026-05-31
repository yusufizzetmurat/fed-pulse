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

const REGIME_WITH_BAND: RegimeClassificationResponse = {
  ...REGIME,
  log_rv_point: -0.25,
  log_rv_lower: -0.85,
  log_rv_upper: 0.35,
  bucket_source: "regression",
};

describe("RegimeHeadline coverage chip", () => {
  it("shows the run-level coverage badge when no empirical figure is provided", () => {
    render(<RegimeHeadline regime={REGIME} symbol="^GSPC" documentDate="2024-09-18" />);
    expect(screen.getByText(/80% confidence level · 2 labels in set/i)).toBeInTheDocument();
    expect(screen.queryByText(/Actual/i)).not.toBeInTheDocument();
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
    expect(screen.getByText(/Target 80% · Actual 76%/i)).toBeInTheDocument();
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
    expect(screen.queryByText(/Actual/i)).not.toBeInTheDocument();
    expect(screen.getByText(/80% confidence level · 2 labels in set/i)).toBeInTheDocument();
  });
});

describe("RegimeHeadline regression-canonical surface (#338)", () => {
  it("leads with the numeric volatility forecast when the dual-head fields are populated", () => {
    render(<RegimeHeadline regime={REGIME_WITH_BAND} symbol="^GSPC" documentDate="2024-09-18" />);
    expect(screen.getByText(/Volatility forecast · 10 days ahead/i)).toBeInTheDocument();
    expect(screen.getByText(/-0\.250/)).toBeInTheDocument();
    expect(screen.getByText(/confidence range \[-0\.850, 0\.350\]/)).toBeInTheDocument();
    expect(screen.getByText(/normal regime/i)).toBeInTheDocument();
    expect(screen.getByText(/regime source · regression/i)).toBeInTheDocument();
  });

  it("falls back to the classifier-led surface when no log_rv_point is present", () => {
    render(<RegimeHeadline regime={REGIME} symbol="^GSPC" documentDate="2024-09-18" />);
    expect(screen.queryByText(/Volatility forecast · 10 days ahead/i)).not.toBeInTheDocument();
    expect(screen.getByText(/Volatility Regime prediction · 10 days ahead/i)).toBeInTheDocument();
    expect(screen.getByText(/top pick · 45\.0%/i)).toBeInTheDocument();
  });


  it("demotes per-class probabilities + predicted set to a foldable section", () => {
    render(<RegimeHeadline regime={REGIME_WITH_BAND} symbol="^GSPC" documentDate="2024-09-18" />);
    const summary = screen.getByText(/Per-class probabilities and prediction set/i);
    expect(summary.tagName.toLowerCase()).toBe("summary");
  });

});
