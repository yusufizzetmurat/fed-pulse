import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";

import { SecondOpinionRegime } from "@/components/analyze/SecondOpinionRegime";
import type {
  HarTercileBaselineResponse,
  RegimeClassificationResponse,
} from "@/lib/analyze/types";

function regimeFixture(
  overrides: Partial<RegimeClassificationResponse> = {},
): RegimeClassificationResponse {
  return {
    predicted_set: ["calm", "normal"],
    set_label: "calm|normal",
    set_size: 2,
    coverage: 0.8,
    distribution: { calm: 0.45, normal: 0.4, high: 0.15 },
    argmax_class: "calm",
    ...overrides,
  };
}

function harFixture(
  topPick: "low" | "medium" | "high",
): HarTercileBaselineResponse {
  return {
    symbol: "^GSPC",
    horizons: [
      {
        h: 1,
        tercile: topPick,
        tercile_probs: { low: 0.2, medium: 0.4, high: 0.4 },
        predicted_rv: 1e-4,
        macro_f1: 0.687,
        macro_f1_source: "wiki §20",
      },
    ],
    source_wiki_section: "20_Gated_Fusion_InfoNCE_Comprehensive_Null",
  };
}

describe("SecondOpinionRegime", () => {
  it("renders the second-opinion strip and macro-F1 chip", () => {
    render(<SecondOpinionRegime regime={regimeFixture()} symbol="^GSPC" />);
    expect(screen.getAllByText(/second opinion/i).length).toBeGreaterThan(0);
    expect(screen.getByText(/Late-fusion text\+market classifier/i)).toBeInTheDocument();
    expect(screen.getByText(/macro-F1 0\.629 \(1-day\)/i)).toBeInTheDocument();
  });

  it("renders the disclosure paragraph and 95% CI", () => {
    render(<SecondOpinionRegime regime={regimeFixture()} symbol="^GSPC" />);
    expect(
      screen.getByText(/Still below the HAR baseline above/i),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/95% block CI includes 0/i),
    ).toBeInTheDocument();
  });

  it("fires the disagrees-with-HAR chip when top picks differ", () => {
    // Late-fusion argmax is "calm"; HAR maps "high" -> "high" so the
    // picks disagree.
    render(
      <SecondOpinionRegime
        regime={regimeFixture({ argmax_class: "calm" })}
        symbol="^GSPC"
        harBaselines={harFixture("high")}
      />,
    );
    expect(screen.getByText(/Disagrees with HAR/i)).toBeInTheDocument();
  });

  it("does not fire the disagrees chip when top picks agree", () => {
    // HAR "low" maps to "calm"; late-fusion argmax is also "calm".
    render(
      <SecondOpinionRegime
        regime={regimeFixture({ argmax_class: "calm" })}
        symbol="^GSPC"
        harBaselines={harFixture("low")}
      />,
    );
    expect(screen.queryByText(/Disagrees with HAR/i)).not.toBeInTheDocument();
  });

  it("fires the low-confidence collapse chip when argmax is calm and prob < 0.65", () => {
    render(
      <SecondOpinionRegime
        regime={regimeFixture({
          argmax_class: "calm",
          distribution: { calm: 0.45, normal: 0.4, high: 0.15 },
        })}
        symbol="^GSPC"
      />,
    );
    expect(screen.getByText(/Low-confidence collapse/i)).toBeInTheDocument();
  });

  it("does not fire the collapse chip when calm has high confidence", () => {
    render(
      <SecondOpinionRegime
        regime={regimeFixture({
          argmax_class: "calm",
          distribution: { calm: 0.78, normal: 0.15, high: 0.07 },
        })}
        symbol="^GSPC"
      />,
    );
    expect(screen.queryByText(/Low-confidence collapse/i)).not.toBeInTheDocument();
  });

  it("does not fire the collapse chip when argmax is not the calm majority class", () => {
    render(
      <SecondOpinionRegime
        regime={regimeFixture({
          argmax_class: "normal",
          distribution: { calm: 0.2, normal: 0.45, high: 0.35 },
        })}
        symbol="^GSPC"
      />,
    );
    expect(screen.queryByText(/Low-confidence collapse/i)).not.toBeInTheDocument();
  });

  it("still renders the late-fusion RegimeHeadline card body", () => {
    render(<SecondOpinionRegime regime={regimeFixture()} symbol="^GSPC" />);
    // Distribution % shows up inside the foldable details, and the
    // outer surface always renders the coverage chip.
    expect(
      screen.getByText(/80% confidence level · 2 labels in set/i),
    ).toBeInTheDocument();
  });
});
