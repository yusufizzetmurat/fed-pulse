import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";

import { SentimentCard } from "@/components/analyze/SentimentCard";

describe("SentimentCard", () => {
  it("renders a hawkish stance with score", () => {
    render(<SentimentCard sentiment={{ label: "HAWKISH", score: 0.81 }} />);
    expect(screen.getAllByText(/hawkish/i).length).toBeGreaterThan(0);
    expect(screen.getByText(/0\.810/)).toBeInTheDocument();
  });

  it("falls back to Unknown when label is missing", () => {
    render(<SentimentCard sentiment={undefined} />);
    expect(screen.getAllByText(/unknown/i).length).toBeGreaterThan(0);
  });
});
