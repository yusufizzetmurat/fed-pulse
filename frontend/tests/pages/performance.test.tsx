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
const fetchHistoryRunMock = vi.fn();

vi.mock("@/lib/analyze/api", () => ({
  resolveApiBaseUrl: () => "http://localhost:8000",
  fetchHistory: (...args: unknown[]) => fetchHistoryMock(...args),
  fetchHistoryRealized: (...args: unknown[]) => fetchHistoryRealizedMock(...args),
  fetchHistoryRun: (...args: unknown[]) => fetchHistoryRunMock(...args),
}));

const SAMPLE_ROWS = [
  {
    id: "run-1",
    created_at: "2026-01-27T20:00:00Z",
    symbol: "^GSPC",
    document_date: "2026-01-27",
    horizon: "10d",
    forecast_mode: "fast",
    stance: "hawkish",
    sentiment_score: 0.7,
    predicted_close: null,
    current_close: null,
    predicted_volatility: null,
    text_excerpt: null,
    argmax_regime: "high",
    argmax_probability: 0.62,
    regime_set_size: 2,
  },
  {
    id: "run-2",
    created_at: "2026-03-17T20:00:00Z",
    symbol: "^GSPC",
    document_date: "2026-03-17",
    horizon: "10d",
    forecast_mode: "fast",
    stance: "dovish",
    sentiment_score: 0.32,
    predicted_close: null,
    current_close: null,
    predicted_volatility: null,
    text_excerpt: null,
    argmax_regime: "calm",
    argmax_probability: 0.55,
    regime_set_size: 1,
  },
];

function realizedPayload(runId: string, regime: string | null) {
  return {
    run_id: runId,
    symbol: "^GSPC",
    document_date: "2026-01-27",
    horizon: "10d",
    timestamps: [],
    close: [],
    volatility: [],
    realized_regime: regime,
  };
}

function detailPayload(runId: string, set: string[]) {
  return {
    id: runId,
    created_at: "2026-01-27T20:00:00Z",
    symbol: "^GSPC",
    document_date: "2026-01-27",
    horizon: "10d",
    forecast_mode: "fast",
    stance: "hawkish",
    payload: {
      regime_classification: { predicted_set: set },
    },
  };
}

describe("PerformancePage", () => {
  beforeEach(() => {
    fetchHistoryMock.mockReset();
    fetchHistoryRealizedMock.mockReset();
    fetchHistoryRunMock.mockReset();
  });

  it("renders empty-state when history is empty", async () => {
    fetchHistoryMock.mockResolvedValue({ total: 0, limit: 100, offset: 0, items: [] });
    const { default: PerformancePage } = await import("@/pages/performance");
    render(<PerformancePage />);
    await waitFor(() =>
      expect(screen.getByText(/no runs in history/i)).toBeInTheDocument(),
    );
  });

  it("renders the regime KPI tiles and per-class table when realized data resolves", async () => {
    fetchHistoryMock.mockResolvedValue({
      total: 2,
      limit: 100,
      offset: 0,
      items: SAMPLE_ROWS,
    });
    fetchHistoryRealizedMock
      .mockResolvedValueOnce(realizedPayload("run-1", "high"))
      .mockResolvedValueOnce(realizedPayload("run-2", "normal"));
    fetchHistoryRunMock
      .mockResolvedValueOnce(detailPayload("run-1", ["high", "normal"]))
      .mockResolvedValueOnce(detailPayload("run-2", ["calm"]));

    const { default: PerformancePage } = await import("@/pages/performance");
    render(<PerformancePage />);

    await waitFor(() =>
      expect(fetchHistoryRealizedMock).toHaveBeenCalledTimes(2),
    );

    // Several of these labels appear in more than one place (the page
    // intro paragraph mentions them and the KPI tiles + card titles
    // repeat them). getAllByText with a count guard avoids the
    // ambiguous-match error while still asserting presence.
    await waitFor(() =>
      expect(screen.getAllByText(/Argmax accuracy/i).length).toBeGreaterThan(0),
    );
    expect(screen.getAllByText(/Empirical coverage/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/Confusion matrix/i).length).toBeGreaterThan(0);
    expect(screen.getByText(/Per-class metrics/i)).toBeInTheDocument();
    expect(screen.getByText(/Per-asset breakdown/i)).toBeInTheDocument();
    expect(screen.getByText(/Run-level detail/i)).toBeInTheDocument();
  });

  it("still renders the table when one realized fetch fails", async () => {
    fetchHistoryMock.mockResolvedValue({
      total: 1,
      limit: 100,
      offset: 0,
      items: [SAMPLE_ROWS[0]],
    });
    fetchHistoryRealizedMock.mockRejectedValue(new Error("Market lookup failed"));
    fetchHistoryRunMock.mockResolvedValueOnce(detailPayload("run-1", ["high"]));

    const { default: PerformancePage } = await import("@/pages/performance");
    render(<PerformancePage />);

    await waitFor(() =>
      expect(fetchHistoryRealizedMock).toHaveBeenCalledTimes(1),
    );
    expect(await screen.findByText(/Run-level detail/i)).toBeInTheDocument();
  });
});
