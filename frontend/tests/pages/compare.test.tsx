import { describe, expect, it, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";

vi.mock("sonner", () => ({
  toast: { success: vi.fn(), error: vi.fn(), info: vi.fn() },
  Toaster: () => null,
}));

const replaceMock = vi.fn();
let mockQuery: Record<string, string> = {};

vi.mock("next/router", () => ({
  useRouter: () => ({
    isReady: true,
    query: mockQuery,
    replace: replaceMock,
  }),
}));

vi.mock("next/head", () => ({
  default: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}));

const fetchHistoryMock = vi.fn();
const fetchHistoryRunMock = vi.fn();

vi.mock("@/lib/analyze/api", () => ({
  resolveApiBaseUrl: () => "http://localhost:8000",
  fetchHistory: (...args: unknown[]) => fetchHistoryMock(...args),
  fetchHistoryRun: (...args: unknown[]) => fetchHistoryRunMock(...args),
  // `compare()` defers to fetchHistoryRun under the hood; the page calls
  // compare(baseUrl, id, id) for single-slot loads and compare(a, b) for
  // pairs. Mirror that fan-out here so the test asserts both slot fetches.
  compare: async (_base: string, idA: string, idB: string) => {
    const [a, b] = await Promise.all([
      fetchHistoryRunMock(_base, idA),
      fetchHistoryRunMock(_base, idB),
    ]);
    return { a, b };
  },
}));

const ENTRIES = [
  {
    id: "run-a",
    created_at: "2024-09-18T12:00:00Z",
    symbol: "^GSPC",
    document_date: "2024-09-18",
    horizon: "3d",
    forecast_mode: "fast",
    stance: "hawkish",
    predicted_close: 5500,
    current_close: 5400,
    sentiment_score: 0.82,
  },
  {
    id: "run-b",
    created_at: "2024-11-07T12:00:00Z",
    symbol: "^GSPC",
    document_date: "2024-11-07",
    horizon: "3d",
    forecast_mode: "fast",
    stance: "dovish",
    predicted_close: 5400,
    current_close: 5450,
    sentiment_score: 0.71,
  },
];

describe("ComparePage", () => {
  beforeEach(() => {
    mockQuery = {};
    fetchHistoryMock.mockReset();
    fetchHistoryRunMock.mockReset();
    replaceMock.mockReset();
  });

  it("renders the empty-state when there are no runs", async () => {
    fetchHistoryMock.mockResolvedValue({ total: 0, limit: 50, offset: 0, items: [] });
    const { default: ComparePage } = await import("@/pages/compare");
    render(<ComparePage />);
    await waitFor(() =>
      expect(screen.getByText(/no runs yet/i)).toBeInTheDocument(),
    );
  });

  it("renders both slot cards once history loads", async () => {
    fetchHistoryMock.mockResolvedValue({ total: 2, limit: 50, offset: 0, items: ENTRIES });
    const { default: ComparePage } = await import("@/pages/compare");
    render(<ComparePage />);
    await waitFor(() => expect(screen.getByText(/Run A/)).toBeInTheDocument());
    expect(screen.getByText(/Run B/)).toBeInTheDocument();
    expect(screen.getAllByText(/Pick a run/).length).toBeGreaterThan(0);
  });

  it("fetches and renders both runs when ?a and ?b query params are set", async () => {
    mockQuery = { a: "run-a", b: "run-b" };
    fetchHistoryMock.mockResolvedValue({ total: 2, limit: 50, offset: 0, items: ENTRIES });
    fetchHistoryRunMock.mockImplementation(async (_base: string, id: string) => ({
      ...ENTRIES.find((entry) => entry.id === id)!,
      payload: {
        prediction: { close: id === "run-a" ? 5500 : 5400, volatility: 0.012 },
        sentiment: { label: id === "run-a" ? "hawkish" : "dovish", score: id === "run-a" ? 0.82 : 0.71 },
      },
    }));
    const { default: ComparePage } = await import("@/pages/compare");
    render(<ComparePage />);
    await waitFor(() => {
      expect(fetchHistoryRunMock).toHaveBeenCalledWith("http://localhost:8000", "run-a");
      expect(fetchHistoryRunMock).toHaveBeenCalledWith("http://localhost:8000", "run-b");
    });
    await waitFor(() => expect(screen.getByText(/Δ A − B/)).toBeInTheDocument());
    expect(screen.getByText(/A shifts hawkish vs\. B/i)).toBeInTheDocument();
  });
});
