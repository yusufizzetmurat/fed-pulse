import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";

vi.mock("recharts", () => {
  return {
    CartesianGrid: () => <div data-testid="rc-grid" />,
    ResponsiveContainer: ({ children }: { children?: React.ReactNode }) => (
      <div data-testid="rc-responsive-container">{children}</div>
    ),
    Scatter: () => <div data-testid="rc-scatter" />,
    ScatterChart: ({ children }: { children?: React.ReactNode }) => (
      <div data-testid="rc-scatter-chart">{children}</div>
    ),
    Tooltip: () => <div data-testid="rc-tooltip" />,
    XAxis: () => <div data-testid="rc-x-axis" />,
    YAxis: () => <div data-testid="rc-y-axis" />,
    ZAxis: () => <div data-testid="rc-z-axis" />,
  };
});

import {
  TrajectoryPanel,
  type TrajectoryResponse,
} from "@/components/analyze/TrajectoryPanel";

function basePayload(
  overrides: Partial<TrajectoryResponse> = {},
): TrajectoryResponse {
  return {
    available: true,
    history: [
      {
        event_date: "2026-01-29",
        axis_stance: "hawkish",
        embedding_2d: [0.1, 0.2],
      },
      {
        event_date: "2026-03-19",
        axis_stance: "neutral",
        embedding_2d: [0.15, 0.18],
      },
    ],
    projected_next: null,
    architecture: "transformer",
    encoder_alias: "bge-small-en-v1.5",
    history_length: 2,
    train_end: "2025-12-31",
    as_of_date: "2026-06-01",
    ...overrides,
  };
}

function mockFetchOnce(payload: TrajectoryResponse) {
  const fetchMock = vi.fn().mockResolvedValue({
    ok: true,
    status: 200,
    json: async () => payload,
  } as unknown as Response);
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

describe("TrajectoryPanel", () => {
  beforeEach(() => {
    vi.useRealTimers();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it("renders the backend warning banner when warning is present", async () => {
    mockFetchOnce(
      basePayload({
        warning:
          "as_of_date 2026-06-01 sits beyond the bundle train_end 2025-12-31; projection extrapolates beyond the fold.",
      }),
    );

    render(
      <TrajectoryPanel
        apiBaseUrl="http://api.test"
        asOfDate="2026-06-01"
        historyLength={2}
      />,
    );

    const banner = await waitFor(() =>
      screen.getByTestId("trajectory-warning"),
    );
    expect(banner).toHaveAttribute("role", "alert");
    expect(banner).toHaveTextContent(/extrapolates beyond the fold/i);
  });

  it("omits the warning banner when warning is null", async () => {
    mockFetchOnce(basePayload({ warning: null }));

    render(
      <TrajectoryPanel
        apiBaseUrl="http://api.test"
        asOfDate="2026-06-01"
        historyLength={2}
      />,
    );

    // Wait for the chart to render so we know the panel finished loading.
    await waitFor(() =>
      expect(screen.getByTestId("rc-scatter-chart")).toBeInTheDocument(),
    );
    expect(screen.queryByTestId("trajectory-warning")).toBeNull();
  });
});
