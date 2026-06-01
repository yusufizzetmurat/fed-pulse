import { describe, expect, it } from "vitest";
import { render, screen, within } from "@testing-library/react";

import { SemanticDiffPanel } from "@/components/analyze/SemanticDiffPanel";
import type {
  SemanticDiffResponse,
  SemanticDiffSpan,
  SemanticDiffTopic,
} from "@/lib/analyze/types";

function span(
  overrides: Partial<SemanticDiffSpan> & { kind: SemanticDiffSpan["kind"] },
): SemanticDiffSpan {
  return {
    text: "placeholder",
    paired_text: null,
    ...overrides,
  };
}

function topic(
  overrides: Partial<SemanticDiffTopic> & { topic: string },
): SemanticDiffTopic {
  return {
    prior_emphasis: 0.2,
    current_emphasis: 0.3,
    delta: 0.1,
    sample_phrases: [],
    ...overrides,
  };
}

function fixture(
  overrides: Partial<SemanticDiffResponse> = {},
): SemanticDiffResponse {
  return {
    current_date: "2026-05-01",
    prior_date: "2026-03-18",
    token_spans: [
      span({ kind: "unchanged", text: "inflation remains" }),
      span({ kind: "added", text: "has eased" }),
      span({ kind: "removed", text: "elevated" }),
      span({
        kind: "substituted",
        text: "moderate",
        paired_text: "solid",
      }),
    ],
    topic_deltas: [
      topic({
        topic: "Inflation",
        delta: 0.15,
        sample_phrases: ["inflation", "2 percent"],
      }),
      topic({
        topic: "Labor",
        delta: -0.05,
        sample_phrases: ["labor market"],
      }),
    ],
    summary: "Largest emphasis shift: more weight on inflation.",
    ...overrides,
  };
}

describe("SemanticDiffPanel", () => {
  it("renders the unavailable placeholder when data is null", () => {
    render(<SemanticDiffPanel data={null} />);
    expect(screen.getByTestId("semantic-diff-unavailable")).toBeInTheDocument();
  });

  it("renders the loading placeholder when loading is true", () => {
    render(<SemanticDiffPanel data={null} loading />);
    expect(screen.getByTestId("semantic-diff-loading")).toBeInTheDocument();
  });

  it("renders the descriptive workspace variant (never a forecast card)", () => {
    render(<SemanticDiffPanel data={fixture()} />);
    const section = screen.getByLabelText("Semantic diff vs prior statement");
    expect(section).toHaveAttribute("data-variant", "descriptive");
    expect(section.className).toMatch(/border-dashed/);
  });

  it("renders the cold-start banner when prior_date is empty", () => {
    render(
      <SemanticDiffPanel
        data={fixture({
          prior_date: "",
          token_spans: [],
          topic_deltas: [],
          summary: "Earliest statement in dataset; no prior to compare.",
        })}
      />,
    );
    expect(screen.getByTestId("semantic-diff-cold-start")).toBeInTheDocument();
    expect(
      screen.queryByTestId("semantic-diff-wording-section"),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByTestId("semantic-diff-emphasis-section"),
    ).not.toBeInTheDocument();
  });

  it("renders the redline spans with kind-specific styling", () => {
    render(<SemanticDiffPanel data={fixture()} />);
    expect(screen.getByTestId("semantic-diff-span-unchanged")).toHaveTextContent(
      /inflation remains/i,
    );
    expect(screen.getByTestId("semantic-diff-span-added")).toHaveTextContent(
      /has eased/i,
    );
    expect(screen.getByTestId("semantic-diff-span-removed")).toHaveTextContent(
      /elevated/i,
    );
    const substituted = screen.getByTestId("semantic-diff-span-substituted");
    expect(substituted).toHaveTextContent(/solid/i);
    expect(substituted).toHaveTextContent(/moderate/i);
  });

  it("truncates unchanged runs longer than 25 words with an ellipsis", () => {
    const longText = Array.from({ length: 40 }, (_, i) => `word${i}`).join(" ");
    render(
      <SemanticDiffPanel
        data={fixture({
          token_spans: [span({ kind: "unchanged", text: longText })],
          topic_deltas: [],
        })}
      />,
    );
    const unchanged = screen.getByTestId("semantic-diff-span-unchanged");
    expect(unchanged.textContent).toMatch(/…/);
  });

  it("renders one row per emphasis topic with signed delta", () => {
    render(<SemanticDiffPanel data={fixture()} />);
    const rows = screen.getAllByTestId("semantic-diff-topic-row");
    expect(rows).toHaveLength(2);
    expect(rows[0]).toHaveTextContent(/Inflation/);
    expect(rows[0]).toHaveTextContent(/\+15\.0 pp/);
    expect(rows[1]).toHaveTextContent(/Labor/);
    expect(rows[1]).toHaveTextContent(/-5\.0 pp/);
  });

  it("renders the per-topic sample phrases as chips", () => {
    render(<SemanticDiffPanel data={fixture()} />);
    const rows = screen.getAllByTestId("semantic-diff-topic-row");
    const inflation = within(rows[0]);
    expect(inflation.getByText("inflation")).toBeInTheDocument();
    expect(inflation.getByText("2 percent")).toBeInTheDocument();
  });

  it("renders the summary line under the emphasis section", () => {
    render(<SemanticDiffPanel data={fixture()} />);
    expect(screen.getByTestId("semantic-diff-summary")).toHaveTextContent(
      /Largest emphasis shift/i,
    );
  });

  it("formats the prior date in the section description", () => {
    render(<SemanticDiffPanel data={fixture()} />);
    expect(screen.getByText(/Mar 18, 2026/)).toBeInTheDocument();
  });

  it("renders the no-input banner when status is no_input", () => {
    render(
      <SemanticDiffPanel
        data={fixture({
          prior_date: "",
          token_spans: [],
          topic_deltas: [],
          summary: "Input too short to diff (n=2 tokens).",
          status: "no_input",
        })}
      />,
    );
    const banner = screen.getByTestId("semantic-diff-no-input");
    expect(banner).toBeInTheDocument();
    expect(banner).toHaveTextContent(/Input too short to diff/i);
    expect(banner).toHaveTextContent(/n=2 tokens/);
    expect(
      screen.queryByTestId("semantic-diff-wording-section"),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByTestId("semantic-diff-cold-start"),
    ).not.toBeInTheDocument();
  });

  it("renders the non-english banner when status is non_english", () => {
    render(
      <SemanticDiffPanel
        data={fixture({
          prior_date: "",
          token_spans: [],
          topic_deltas: [],
          summary: "Non-Latin text — diff not run.",
          status: "non_english",
        })}
      />,
    );
    const banner = screen.getByTestId("semantic-diff-non-english");
    expect(banner).toBeInTheDocument();
    expect(banner).toHaveTextContent(/Non-Latin text/i);
    expect(
      screen.queryByTestId("semantic-diff-wording-section"),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByTestId("semantic-diff-cold-start"),
    ).not.toBeInTheDocument();
  });

  it("renders the cold-start banner when status is no_prior", () => {
    render(
      <SemanticDiffPanel
        data={fixture({
          prior_date: "",
          token_spans: [],
          topic_deltas: [],
          summary: "Earliest statement in dataset; no prior to compare.",
          status: "no_prior",
        })}
      />,
    );
    expect(screen.getByTestId("semantic-diff-cold-start")).toBeInTheDocument();
    expect(
      screen.queryByTestId("semantic-diff-no-input"),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByTestId("semantic-diff-non-english"),
    ).not.toBeInTheDocument();
  });

  it("renders the populated diff when status is ok", () => {
    render(<SemanticDiffPanel data={fixture({ status: "ok" })} />);
    expect(
      screen.getByTestId("semantic-diff-wording-section"),
    ).toBeInTheDocument();
    expect(
      screen.getByTestId("semantic-diff-emphasis-section"),
    ).toBeInTheDocument();
    expect(
      screen.queryByTestId("semantic-diff-no-input"),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByTestId("semantic-diff-non-english"),
    ).not.toBeInTheDocument();
  });
});
