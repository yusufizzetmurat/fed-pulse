import { describe, expect, it, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";

vi.mock("sonner", () => ({
  toast: { success: vi.fn(), error: vi.fn(), info: vi.fn() },
  Toaster: () => null,
}));

let mockQuery: Record<string, string> = {};

vi.mock("next/router", () => ({
  useRouter: () => ({
    isReady: true,
    query: mockQuery,
    replace: vi.fn(),
  }),
}));

vi.mock("next/head", () => ({
  default: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}));

vi.mock("next/dynamic", () => ({
  default: () => () => null,
}));

const fetchHistoryRunMock = vi.fn();

vi.mock("@/lib/analyze/api", () => ({
  resolveApiBaseUrl: () => "http://localhost:8000",
  fetchHistoryRun: (...args: unknown[]) => fetchHistoryRunMock(...args),
}));

const DETAIL = {
  id: "run-9",
  created_at: "2024-09-18T12:00:00Z",
  symbol: "^GSPC",
  document_date: "2024-09-18",
  horizon: "3d",
  forecast_mode: "fast",
  stance: "hawkish",
  predicted_close: 5500,
  current_close: 5400,
  sentiment_score: 0.82,
  predicted_volatility: 0.012,
  text_excerpt: "September 18 statement excerpt …",
  payload: {
    sentiment: { label: "hawkish", score: 0.82 },
    prediction: { close: 5500, volatility: 0.012, horizon: "3d" },
    market: { symbol: "^GSPC", requested_date: "2024-09-18", date_used: "2024-09-18", close: 5400, volatility_5d: 0.01 },
    model: { runtime_mode: "fast", checkpoint_loaded: true },
    series: {
      timestamps: ["2024-09-17T00:00:00Z"],
      history_close: [5380],
      history_volatility: [0.01],
      forecast_timestamps: ["2024-09-18T00:00:00Z", "2024-09-19T00:00:00Z", "2024-09-20T00:00:00Z"],
      forecast_close: [5450, 5475, 5500],
      forecast_close_lower: [5400, 5410, 5430],
      forecast_close_upper: [5500, 5540, 5570],
      forecast_volatility: [0.011, 0.012, 0.013],
      forecast_volatility_lower: [0.009, 0.010, 0.011],
      forecast_volatility_upper: [0.013, 0.014, 0.015],
      forecast_confidence_level: 0.8,
      volatility_scale: { suggested_ymin: 0, suggested_ymax: 0.02 },
    },
  },
};

describe("HistoryDetailPage", () => {
  beforeEach(() => {
    mockQuery = { id: "run-9" };
    fetchHistoryRunMock.mockReset();
  });

  it("renders the run header and submitted-text card on success", async () => {
    fetchHistoryRunMock.mockResolvedValue(DETAIL);
    const { default: HistoryDetailPage } = await import("@/pages/history/[id]");
    render(<HistoryDetailPage />);
    expect(await screen.findByText(/September 18 statement excerpt/)).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: /2024-09-18/ })).toBeInTheDocument();
  });

  it("renders the not-found fallback when the backend rejects", async () => {
    fetchHistoryRunMock.mockRejectedValue(new Error("Run does not exist"));
    const { default: HistoryDetailPage } = await import("@/pages/history/[id]");
    render(<HistoryDetailPage />);
    // The categorical error voice surfaces a generic Model unavailable
    // string rather than the backend's raw message — the helper folds
    // both 404 and unknown errors into the same bucket.
    await waitFor(() => expect(screen.getByText(/Model unavailable/i)).toBeInTheDocument());
  });

  it("links the compare-with button to /compare?a=<id>", async () => {
    fetchHistoryRunMock.mockResolvedValue(DETAIL);
    const { default: HistoryDetailPage } = await import("@/pages/history/[id]");
    render(<HistoryDetailPage />);
    await waitFor(() => expect(screen.getByText(/Compare with/i)).toBeInTheDocument());
    const link = screen.getByText(/Compare with/i).closest("a");
    expect(link?.getAttribute("href")).toBe("/compare?a=run-9");
  });
});
