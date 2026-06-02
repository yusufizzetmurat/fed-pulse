import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";

import { HarAccuracyPanel } from "@/components/analyze/HarAccuracyPanel";
import type {
  HarTercileBacktestResponse,
  HarTercileBacktestRow,
} from "@/lib/analyze/types";

function makeRow(
  overrides: Partial<HarTercileBacktestRow> = {},
): HarTercileBacktestRow {
  return {
    event_date: "2024-05-01",
    predicted_tercile: "high",
    predicted_prob: 0.62,
    realized_tercile: "high",
    realized_rv: 0.018,
    correct: true,
    ...overrides,
  };
}

function fixture(
  overrides: Partial<HarTercileBacktestResponse> = {},
): HarTercileBacktestResponse {
  return {
    symbol: "^GSPC",
    horizon: 10,
    rows: [
      makeRow({
        event_date: "2024-05-01",
        predicted_tercile: "high",
        realized_tercile: "high",
        correct: true,
        realized_rv: 0.018,
      }),
      makeRow({
        event_date: "2024-03-20",
        predicted_tercile: "medium",
        realized_tercile: "low",
        correct: false,
        realized_rv: 0.006,
        predicted_prob: 0.48,
      }),
      makeRow({
        event_date: "2024-01-31",
        predicted_tercile: "low",
        realized_tercile: null,
        correct: null,
        realized_rv: null,
        predicted_prob: 0.55,
      }),
    ],
    metrics: {
      total_runs: 3,
      resolved_runs: 2,
      accuracy_overall: 0.5,
      per_tercile_hit_rate: { high: 1, medium: 0 },
    },
    generated_at: "2026-06-01T00:00:00+00:00",
    ...overrides,
  };
}

describe("HarAccuracyPanel", () => {
  it("renders the aggregate accuracy KPI and resolved counter", () => {
    render(<HarAccuracyPanel data={fixture()} symbol="^GSPC" />);
    expect(screen.getByTestId("har-accuracy-overall")).toHaveTextContent(
      "50.0%",
    );
    expect(screen.getByTestId("har-accuracy-counter")).toHaveTextContent(
      "2 / 3",
    );
  });

  it("renders per-tercile hit-rate chips for all three buckets", () => {
    render(<HarAccuracyPanel data={fixture()} symbol="^GSPC" />);
    expect(screen.getByTestId("har-accuracy-tercile-low")).toHaveTextContent(
      /low:\s*—/i,
    );
    expect(screen.getByTestId("har-accuracy-tercile-medium")).toHaveTextContent(
      /medium:\s*0\.0%/i,
    );
    expect(screen.getByTestId("har-accuracy-tercile-high")).toHaveTextContent(
      /high:\s*100\.0%/i,
    );
  });

  it("renders hit / miss / pending marks per row", () => {
    render(<HarAccuracyPanel data={fixture()} symbol="^GSPC" />);
    expect(screen.getByTestId("har-accuracy-row-hit-2024-05-01")).toHaveTextContent(
      "✓",
    );
    expect(screen.getByTestId("har-accuracy-row-hit-2024-03-20")).toHaveTextContent(
      "✗",
    );
    expect(screen.getByTestId("har-accuracy-row-hit-2024-01-31")).toHaveTextContent(
      "—",
    );
  });

  it("renders predicted + realized tercile chips per resolved row", () => {
    render(<HarAccuracyPanel data={fixture()} symbol="^GSPC" />);
    expect(
      screen.getByTestId("har-accuracy-row-pred-2024-05-01"),
    ).toHaveTextContent(/high/i);
    expect(
      screen.getByTestId("har-accuracy-row-real-2024-05-01"),
    ).toHaveTextContent(/high/i);
    // Pending row shows the "pending" placeholder instead of a realized chip.
    expect(
      screen.getByTestId("har-accuracy-row-real-2024-01-31-pending"),
    ).toHaveTextContent(/pending/i);
  });

  it("formats realized RV as annualized vol percent", () => {
    render(<HarAccuracyPanel data={fixture()} symbol="^GSPC" />);
    // sqrt(0.018 * 252) * 100 ≈ 213.0% per the formatter's annualisation.
    // We assert the formatted % shows up in the table row.
    const row = screen.getByTestId("har-accuracy-row-2024-05-01");
    expect(row.textContent).toMatch(/%/);
  });

  it("renders an empty state when rows are empty", () => {
    const empty = fixture({
      rows: [],
      metrics: {
        total_runs: 0,
        resolved_runs: 0,
        accuracy_overall: null,
        per_tercile_hit_rate: {},
      },
    });
    render(<HarAccuracyPanel data={empty} symbol="^GSPC" />);
    expect(screen.getByTestId("har-accuracy-empty")).toHaveTextContent(
      /No resolved FOMC runs for \^GSPC yet/i,
    );
  });

  it("renders a loading state when loading", () => {
    render(<HarAccuracyPanel data={null} loading symbol="^GSPC" />);
    expect(screen.getByTestId("har-accuracy-loading")).toBeInTheDocument();
  });

  it("renders an unavailable state on error or missing data", () => {
    render(
      <HarAccuracyPanel
        data={null}
        error="backtest service down"
        symbol="^GSPC"
      />,
    );
    expect(screen.getByTestId("har-accuracy-unavailable")).toHaveTextContent(
      /backtest service down/i,
    );
  });

  it("shows the headline accuracy as em-dash when nothing resolved", () => {
    const data = fixture({
      rows: [
        makeRow({
          event_date: "2024-04-04",
          predicted_tercile: "high",
          realized_tercile: null,
          correct: null,
          realized_rv: null,
        }),
      ],
      metrics: {
        total_runs: 1,
        resolved_runs: 0,
        accuracy_overall: null,
        per_tercile_hit_rate: {},
      },
    });
    render(<HarAccuracyPanel data={data} symbol="^GSPC" />);
    expect(screen.getByTestId("har-accuracy-overall")).toHaveTextContent("—");
    expect(screen.getByTestId("har-accuracy-counter")).toHaveTextContent(
      "0 / 1",
    );
  });
});
