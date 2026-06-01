import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";

vi.mock("recharts", () => {
  return {
    CartesianGrid: () => <div data-testid="rc-grid" />,
    ComposedChart: ({
      children,
      data,
    }: {
      children?: React.ReactNode;
      data?: unknown[];
    }) => (
      <div data-testid="rc-composed-chart" data-row-count={(data ?? []).length}>
        {children}
      </div>
    ),
    Line: ({ dataKey }: { dataKey?: string }) => (
      <div data-testid={`rc-line-${String(dataKey)}`} />
    ),
    ReferenceArea: ({
      fill,
      x1,
      x2,
    }: {
      fill?: string;
      x1?: number;
      x2?: number;
    }) => (
      <div
        data-testid="rc-reference-area"
        data-fill={fill}
        data-x1={String(x1)}
        data-x2={String(x2)}
      />
    ),
    ResponsiveContainer: ({ children }: { children?: React.ReactNode }) => (
      <div data-testid="rc-responsive-container">{children}</div>
    ),
    Tooltip: () => <div data-testid="rc-tooltip" />,
    XAxis: () => <div data-testid="rc-x-axis" />,
    YAxis: () => <div data-testid="rc-y-axis" />,
  };
});

import {
  EventStudyChart,
  buildEventStudyHeadline,
} from "@/components/analyze/EventStudyChart";
import type { HistoryEventStudyResponse } from "@/lib/analyze/types";

function makePayload(
  overrides: Partial<HistoryEventStudyResponse> = {},
): HistoryEventStudyResponse {
  return {
    event_date: "2024-09-18",
    symbol: "^GSPC",
    forward_dates: [
      "2024-09-19",
      "2024-09-20",
      "2024-09-23",
      "2024-09-24",
      "2024-09-25",
      "2024-09-26",
      "2024-09-27",
      "2024-09-30",
      "2024-10-01",
      "2024-10-02",
    ],
    forward_close: [5650, 5660, 5640, 5670, 5680, 5690, 5700, 5710, 5705, 5720],
    forward_log_returns: [0.001, 0.002, -0.003, 0.005, 0.002, 0.002, 0.002, 0.002, -0.001, 0.003],
    realized_vol_10d: 0.0123,
    predicted_regime: "calm",
    realized_regime: "normal",
    ...overrides,
  };
}

describe("EventStudyChart", () => {
  it("renders the loading skeleton when loading is true", () => {
    render(<EventStudyChart data={null} loading />);
    expect(screen.getByText(/Loading forward 10-day price path/i)).toBeInTheDocument();
    expect(screen.queryByTestId("rc-composed-chart")).toBeNull();
  });

  it("renders the error fallback when errorMessage is set", () => {
    render(<EventStudyChart data={null} errorMessage="yfinance failed" />);
    expect(screen.getByText(/Could not load market path/i)).toBeInTheDocument();
    expect(screen.getByText(/yfinance failed/i)).toBeInTheDocument();
    expect(screen.queryByTestId("rc-composed-chart")).toBeNull();
  });

  it("renders the empty fallback when payload has no forward bars", () => {
    render(
      <EventStudyChart
        data={makePayload({ forward_dates: [], forward_close: [], forward_log_returns: [] })}
      />,
    );
    expect(screen.getByText(/No forward bars yet/i)).toBeInTheDocument();
    expect(screen.queryByTestId("rc-composed-chart")).toBeNull();
  });

  it("renders the close line, regime band and headline on a populated payload", () => {
    render(<EventStudyChart data={makePayload()} />);
    const chart = screen.getByTestId("rc-composed-chart");
    expect(chart).toHaveAttribute("data-row-count", "10");
    expect(screen.getByTestId("rc-line-close")).toBeInTheDocument();
    const band = screen.getByTestId("rc-reference-area");
    expect(band.getAttribute("data-x1")).toBe("1");
    expect(band.getAttribute("data-x2")).toBe("10");
    expect(screen.getByText(/predicted calm, realized normal/i)).toBeInTheDocument();
    expect(screen.getByText(/Realised 10-day vol: 0\.0123/i)).toBeInTheDocument();
  });

  it("omits the regime band when no predicted regime is available", () => {
    render(<EventStudyChart data={makePayload({ predicted_regime: null })} />);
    expect(screen.queryByTestId("rc-reference-area")).toBeNull();
    expect(screen.getByText(/predicted n\/a, realized normal/i)).toBeInTheDocument();
  });

  it("buildEventStudyHeadline formats predicted + realized into the headline text", () => {
    expect(buildEventStudyHeadline("high", "calm")).toBe("predicted high, realized calm");
    expect(buildEventStudyHeadline(null, "normal")).toBe("predicted n/a, realized normal");
    expect(buildEventStudyHeadline("calm", null)).toBe("predicted calm, realized n/a");
  });
});
