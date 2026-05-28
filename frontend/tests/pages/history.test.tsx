import { describe, expect, it, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";

vi.mock("sonner", () => ({
  toast: { success: vi.fn(), error: vi.fn(), info: vi.fn() },
  Toaster: () => null,
}));

const replaceMock = vi.fn();
vi.mock("next/router", () => ({
  useRouter: () => ({
    isReady: true,
    query: {},
    push: vi.fn(),
    replace: replaceMock,
  }),
}));

vi.mock("next/head", () => ({
  default: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}));

const fetchHistoryMock = vi.fn();
const fetchHistoryRealizedBatchMock = vi.fn();
const deleteHistoryRunMock = vi.fn();

const fetchSymbolsMock = vi.fn();

vi.mock("@/lib/analyze/api", () => ({
  resolveApiBaseUrl: () => "http://localhost:8000",
  fetchHistory: (...args: unknown[]) => fetchHistoryMock(...args),
  fetchHistoryRealizedBatch: (...args: unknown[]) => fetchHistoryRealizedBatchMock(...args),
  deleteHistoryRun: (...args: unknown[]) => deleteHistoryRunMock(...args),
  fetchSymbols: (...args: unknown[]) => fetchSymbolsMock(...args),
}));

describe("HistoryPage", () => {
  beforeEach(() => {
    fetchHistoryMock.mockReset();
    fetchHistoryRealizedBatchMock.mockReset();
    fetchHistoryRealizedBatchMock.mockResolvedValue({ items: {}, missing: [] });
    deleteHistoryRunMock.mockReset();
    fetchSymbolsMock.mockReset();
    fetchSymbolsMock.mockResolvedValue({ symbols: [] });
    replaceMock.mockReset();
  });

  it("renders the rows returned by the backend", async () => {
    fetchHistoryMock.mockResolvedValue({
      total: 2,
      limit: 20,
      offset: 0,
      items: [
        {
          id: "abc",
          created_at: "2024-09-18T12:00:00Z",
          symbol: "^GSPC",
          document_date: "2024-09-18",
          horizon: "3d",
          forecast_mode: "fast",
          stance: "hawkish",
          predicted_close: 5050.5,
          current_close: 5000.1,
        },
        {
          id: "def",
          created_at: "2024-11-06T12:00:00Z",
          symbol: "^NDX",
          document_date: "2024-11-06",
          horizon: "5d",
          forecast_mode: "real_train",
          stance: "dovish",
        },
      ],
    });
    const { default: HistoryPage } = await import("@/pages/history");
    render(<HistoryPage />);
    await waitFor(() => expect(screen.getAllByText("^GSPC").length).toBeGreaterThan(0));
    expect(screen.getAllByText("^NDX").length).toBeGreaterThan(0);
    expect(screen.getByText("2024-09-18")).toBeInTheDocument();
    expect(screen.getByText("2024-11-06")).toBeInTheDocument();
    expect(screen.getAllByText(/hawkish/i).length).toBeGreaterThan(0);
  });

  it("renders an empty-state when the backend returns no rows", async () => {
    fetchHistoryMock.mockResolvedValue({ total: 0, limit: 20, offset: 0, items: [] });
    const { default: HistoryPage } = await import("@/pages/history");
    render(<HistoryPage />);
    await waitFor(() =>
      expect(screen.getByText(/no history yet/i)).toBeInTheDocument(),
    );
  });

  it("fires exactly one batched realized request after loading the row list", async () => {
    fetchHistoryMock.mockResolvedValue({
      total: 2,
      limit: 20,
      offset: 0,
      items: [
        {
          id: "abc",
          created_at: "2024-09-18T12:00:00Z",
          symbol: "^GSPC",
          document_date: "2024-09-18",
          horizon: "10d",
          forecast_mode: "fast",
          stance: "hawkish",
        },
        {
          id: "def",
          created_at: "2024-11-06T12:00:00Z",
          symbol: "^NDX",
          document_date: "2024-11-06",
          horizon: "10d",
          forecast_mode: "fast",
          stance: "dovish",
        },
      ],
    });
    fetchHistoryRealizedBatchMock.mockResolvedValue({
      items: {
        abc: {
          run_id: "abc",
          symbol: "^GSPC",
          document_date: "2024-09-18",
          horizon: "10d",
          timestamps: ["2024-09-20"],
          close: [5050.0],
          volatility: [0.011],
          realized_regime: "normal",
        },
      },
      missing: ["def"],
    });

    const { default: HistoryPage } = await import("@/pages/history");
    render(<HistoryPage />);
    await waitFor(() =>
      expect(fetchHistoryRealizedBatchMock).toHaveBeenCalledTimes(1),
    );
    const [, ids] = fetchHistoryRealizedBatchMock.mock.calls[0];
    expect(ids).toEqual(["abc", "def"]);
  });
});
