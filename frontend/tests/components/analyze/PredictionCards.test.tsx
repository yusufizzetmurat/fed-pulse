import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";

import { PredictionCards } from "@/components/analyze/PredictionCards";
import type { AnalyzeResult } from "@/lib/analyze/types";

describe("PredictionCards", () => {
  it("renders predicted close with positive delta vs spot", () => {
    const result: AnalyzeResult = {
      prediction: { close: 5050, volatility: 0.012, horizon: "3d" },
      market: { close: 5000 },
      series: {
        history_close: [4900, 5000],
        forecast_close: [5050],
      },
    };
    const { container } = render(<PredictionCards result={result} />);
    expect(screen.getByText(/\$5,050/)).toBeInTheDocument();
    expect(screen.getByText(/\+\$50/)).toBeInTheDocument();
    expect(container.textContent).toMatch(/horizon 3d/i);
  });

  it("renders volatility as percent", () => {
    const result: AnalyzeResult = {
      prediction: { close: 5000, volatility: 0.02, horizon: "5d" },
      market: { close: 5000 },
    };
    render(<PredictionCards result={result} />);
    expect(screen.getByText(/2\.00%/)).toBeInTheDocument();
  });
});
