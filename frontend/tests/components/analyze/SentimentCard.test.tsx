import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";

import { SentimentCard } from "@/components/analyze/SentimentCard";

describe("SentimentCard", () => {
  it("renders a hawkish stance with score", () => {
    render(<SentimentCard sentiment={{ label: "HAWKISH", score: 0.81 }} />);
    expect(screen.getAllByText(/hawkish/i).length).toBeGreaterThan(0);
    expect(screen.getByText(/0\.810/)).toBeInTheDocument();
  });

  it("falls back to 'Sentiment unavailable' when label is missing", () => {
    render(<SentimentCard sentiment={undefined} />);
    expect(screen.getByText(/sentiment unavailable/i)).toBeInTheDocument();
  });

  it("renders 'Sentiment unavailable' with the raw label when backend returns POSITIVE", () => {
    // Regression guard for the live bug: when the FOMC sentiment model
    // fails to load and distilbert-sst-2 takes over, the backend returns
    // POSITIVE / NEGATIVE. The dashboard must NOT silently re-render
    // these as hawkish/dovish — it must surface the failure.
    render(<SentimentCard sentiment={{ label: "POSITIVE", score: 0.99 }} />);
    expect(screen.getByText(/sentiment unavailable/i)).toBeInTheDocument();
    expect(screen.getByText(/raw: POSITIVE/)).toBeInTheDocument();
  });
});
