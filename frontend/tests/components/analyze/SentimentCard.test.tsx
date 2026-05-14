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

  it("renders OOD warning with energy and threshold when is_in_distribution is false", () => {
    render(
      <SentimentCard
        sentiment={{
          label: "hawkish",
          score: 0.94,
          ood_energy: 12.34,
          ood_threshold: 8.0,
          is_in_distribution: false,
        }}
      />,
    );
    expect(screen.getByText(/out of distribution/i)).toBeInTheDocument();
    expect(screen.getByText(/energy 12\.34/)).toBeInTheDocument();
    expect(screen.getByText(/threshold 8\.00/)).toBeInTheDocument();
  });

  it("shows in-distribution energy footer when is_in_distribution is true", () => {
    render(
      <SentimentCard
        sentiment={{
          label: "hawkish",
          score: 0.81,
          ood_energy: 5.2,
          ood_threshold: 8.0,
          is_in_distribution: true,
        }}
      />,
    );
    expect(screen.queryByText(/out of distribution/i)).not.toBeInTheDocument();
    expect(screen.getByText(/in-distribution · energy 5\.20/)).toBeInTheDocument();
  });

  it("omits the OOD row when the response carries no manifest fields", () => {
    render(<SentimentCard sentiment={{ label: "hawkish", score: 0.7 }} />);
    expect(screen.queryByText(/out of distribution/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/in-distribution/i)).not.toBeInTheDocument();
  });
});
