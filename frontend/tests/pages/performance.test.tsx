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
const fetchHistoryRealizedBatchMock = vi.fn();
const fetchHistoryRunMock = vi.fn();
const fetchClassificationBreakdownMock = vi.fn();

vi.mock("@/lib/analyze/api", () => ({
  resolveApiBaseUrl: () => "http://localhost:8000",
  fetchHistory: (...args: unknown[]) => fetchHistoryMock(...args),
  fetchHistoryRealizedBatch: (...args: unknown[]) => fetchHistoryRealizedBatchMock(...args),
  fetchHistoryRun: (...args: unknown[]) => fetchHistoryRunMock(...args),
  fetchClassificationBreakdown: (...args: unknown[]) =>
    fetchClassificationBreakdownMock(...args),
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
    fetchHistoryRealizedBatchMock.mockReset();
    fetchHistoryRunMock.mockReset();
    fetchClassificationBreakdownMock.mockReset();
    fetchClassificationBreakdownMock.mockResolvedValue({ available: false });
    fetchHistoryRealizedBatchMock.mockResolvedValue({ items: {}, missing: [] });
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
    fetchHistoryRealizedBatchMock.mockResolvedValue({
      items: {
        "run-1": realizedPayload("run-1", "high"),
        "run-2": realizedPayload("run-2", "normal"),
      },
      missing: [],
    });
    fetchHistoryRunMock
      .mockResolvedValueOnce(detailPayload("run-1", ["high", "normal"]))
      .mockResolvedValueOnce(detailPayload("run-2", ["calm"]));

    const { default: PerformancePage } = await import("@/pages/performance");
    render(<PerformancePage />);

    await waitFor(() =>
      expect(fetchHistoryRealizedBatchMock).toHaveBeenCalledTimes(1),
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

  it("still renders the table when the realized batch fetch fails", async () => {
    fetchHistoryMock.mockResolvedValue({
      total: 1,
      limit: 100,
      offset: 0,
      items: [SAMPLE_ROWS[0]],
    });
    fetchHistoryRealizedBatchMock.mockRejectedValue(new Error("Market lookup failed"));
    fetchHistoryRunMock.mockResolvedValueOnce(detailPayload("run-1", ["high"]));

    const { default: PerformancePage } = await import("@/pages/performance");
    render(<PerformancePage />);

    await waitFor(() =>
      expect(fetchHistoryRealizedBatchMock).toHaveBeenCalledTimes(1),
    );
    expect(await screen.findByText(/Run-level detail/i)).toBeInTheDocument();
  });

  it("renders ROC-AUC + provenance when the breakdown artifact is available", async () => {
    fetchHistoryMock.mockResolvedValue({
      total: 1,
      limit: 100,
      offset: 0,
      items: [SAMPLE_ROWS[0]],
    });
    fetchClassificationBreakdownMock.mockResolvedValue({
      available: true,
      confusion_matrix: [[5, 1, 0], [0, 4, 2], [1, 0, 3]],
      per_class: [
        { class_id: 0, precision: 0.83, recall: 0.83, f1: 0.83, support: 6, roc_auc: 0.91, pr_auc: 0.88 },
        { class_id: 1, precision: 0.80, recall: 0.67, f1: 0.73, support: 6, roc_auc: 0.85, pr_auc: 0.79 },
        { class_id: 2, precision: 0.60, recall: 0.75, f1: 0.67, support: 4, roc_auc: 0.78, pr_auc: 0.72 },
      ],
      macro_f1: 0.74,
      macro_roc_auc: 0.85,
      n_classes: 3,
      source: {
        relative_path: "regime_baseline_tiers/tp_fixture/forecaster_sweep_results.json",
        training_package_id: "tp_fixture",
        modified_at: "2026-05-25T00:00:00Z",
      },
    });
    fetchHistoryRunMock.mockResolvedValueOnce(detailPayload("run-1", ["high"]));

    const { default: PerformancePage } = await import("@/pages/performance");
    render(<PerformancePage />);

    await waitFor(() => expect(screen.getByText(/macro roc-auc/i)).toBeInTheDocument());
    expect(screen.getByText(/from training eval artifact/i)).toBeInTheDocument();
    // The "training-time classification breakdown" phrase appears in
    // both the per-class table and the confusion matrix descriptions
    // when the artifact is loaded — assert presence, not uniqueness.
    expect(screen.getAllByText(/training-time classification breakdown/i).length).toBeGreaterThan(0);
    expect(screen.getByText(/tp_fixture/)).toBeInTheDocument();
  });
});
