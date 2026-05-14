import { describe, expect, it, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";

vi.mock("sonner", () => ({
  toast: { success: vi.fn(), error: vi.fn(), info: vi.fn() },
  Toaster: () => null,
}));

vi.mock("next/router", () => ({
  useRouter: () => ({
    isReady: true,
    query: {},
    push: vi.fn(),
  }),
}));

vi.mock("next/head", () => ({
  default: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}));

const fetchHistoryMock = vi.fn();
const fetchHistoryRealizedMock = vi.fn();

vi.mock("@/lib/analyze/api", () => ({
  resolveApiBaseUrl: () => "http://localhost:8000",
  fetchHistory: (...args: unknown[]) => fetchHistoryMock(...args),
  fetchHistoryRealized: (...args: unknown[]) => fetchHistoryRealizedMock(...args),
}));

const SAMPLE_ROWS = [
  {
    id: "run-1",
    created_at: "2024-09-18T12:00:00Z",
    symbol: "^GSPC",
    document_date: "2024-09-18",
    horizon: "3d",
    forecast_mode: "fast",
    stance: "hawkish",
    sentiment_score: 0.7,
    predicted_close: 5500,
    current_close: 5400,
    predicted_volatility: 0.012,
    text_excerpt: null,
  },
  {
    id: "run-2",
    created_at: "2024-11-06T12:00:00Z",
    symbol: "^GSPC",
    document_date: "2024-11-06",
    horizon: "3d",
    forecast_mode: "fast",
    stance: "dovish",
    sentiment_score: 0.32,
    predicted_close: 5800,
    current_close: 5850,
    predicted_volatility: 0.015,
    text_excerpt: null,
  },
];

describe("PerformancePage", () => {
  beforeEach(() => {
    fetchHistoryMock.mockReset();
    fetchHistoryRealizedMock.mockReset();
  });

  it("renders empty-state when history is empty", async () => {
    fetchHistoryMock.mockResolvedValue({ total: 0, limit: 50, offset: 0, items: [] });
    const { default: PerformancePage } = await import("@/pages/performance");
    render(<PerformancePage />);
    await waitFor(() =>
      expect(screen.getByText(/submit analyses on the Analyze page/i)).toBeInTheDocument()
    );
  });

  it("renders aggregate hit-rate and run-level rows when realized data resolves", async () => {
    fetchHistoryMock.mockResolvedValue({ total: 2, limit: 50, offset: 0, items: SAMPLE_ROWS });
    fetchHistoryRealizedMock
      .mockResolvedValueOnce({
        run_id: "run-1",
        symbol: "^GSPC",
        document_date: "2024-09-18",
        horizon: "3d",
        timestamps: ["2024-09-19", "2024-09-20", "2024-09-23"],
        close: [5520, 5540, 5560],
        volatility: [0.011, 0.012, 0.013],
      })
      .mockResolvedValueOnce({
        run_id: "run-2",
        symbol: "^GSPC",
        document_date: "2024-11-06",
        horizon: "3d",
        timestamps: ["2024-11-07", "2024-11-08", "2024-11-11"],
        close: [5780, 5760, 5790],
        volatility: [0.014, 0.015, 0.014],
      });

    const { default: PerformancePage } = await import("@/pages/performance");
    render(<PerformancePage />);

    await waitFor(() => expect(fetchHistoryRealizedMock).toHaveBeenCalledTimes(2));
    expect(await screen.findByText(/Directional hit rate/i)).toBeInTheDocument();
    expect(screen.getByText(/Per-asset breakdown/i)).toBeInTheDocument();
    expect(screen.getByText(/Run-level detail/i)).toBeInTheDocument();
    expect(screen.getAllByText(/^\^GSPC$/).length).toBeGreaterThanOrEqual(2);
  });

  it("still renders the table when one realized fetch fails", async () => {
    fetchHistoryMock.mockResolvedValue({ total: 1, limit: 50, offset: 0, items: [SAMPLE_ROWS[0]] });
    fetchHistoryRealizedMock.mockRejectedValue(new Error("Market lookup failed"));

    const { default: PerformancePage } = await import("@/pages/performance");
    render(<PerformancePage />);

    await waitFor(() => expect(fetchHistoryRealizedMock).toHaveBeenCalledTimes(1));
    expect(await screen.findByText(/Run-level detail/i)).toBeInTheDocument();
  });
});
