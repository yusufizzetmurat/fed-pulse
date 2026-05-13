import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";

import { ErrorBadges } from "@/components/analyze/ErrorBadges";
import type { AnalyzeResult } from "@/lib/analyze/types";

const result: AnalyzeResult = {
  prediction: { close: 5050, volatility: 0.012 },
  market: { close: 5000, volatility_5d: 0.01 },
};

describe("ErrorBadges", () => {
  it("shows enable-overlay hint when there is no realized data", () => {
    render(
      <ErrorBadges
        result={result}
        metrics={{ close: { mape: null, rmse: null }, vol: { mape: null, rmse: null }, hasRealized: false }}
      />
    );
    expect(screen.getByText(/realized overlay/i)).toBeInTheDocument();
  });

  it("renders MAPE and RMSE values when realized data exists", () => {
    render(
      <ErrorBadges
        result={result}
        metrics={{
          close: { mape: 1.4, rmse: 12 },
          vol: { mape: 6.2, rmse: 0.0012 },
          hasRealized: true,
        }}
      />
    );
    expect(screen.getByText("1.40%")).toBeInTheDocument();
    expect(screen.getByText("12.0000")).toBeInTheDocument();
    expect(screen.getByText("6.20%")).toBeInTheDocument();
  });
});
