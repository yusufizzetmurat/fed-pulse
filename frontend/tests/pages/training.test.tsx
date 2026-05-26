import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";

vi.mock("sonner", () => ({
  toast: { success: vi.fn(), error: vi.fn(), info: vi.fn() },
  Toaster: () => null,
}));

const routerStore = { isReady: true, query: {} as Record<string, string>, push: vi.fn() };
vi.mock("next/router", () => ({
  useRouter: () => routerStore,
}));

vi.mock("next/head", () => ({
  default: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}));

const fetchTrainJobsMock = vi.fn();
const fetchTrainJobMock = vi.fn();
vi.mock("@/lib/analyze/api", () => ({
  resolveApiBaseUrl: () => "http://localhost:8000",
  fetchTrainJobs: (...args: unknown[]) => fetchTrainJobsMock(...args),
  fetchTrainJob: (...args: unknown[]) => fetchTrainJobMock(...args),
}));

const SAMPLE_JOBS = [
  {
    job_id: "11111111-1111-1111-1111-111111111111",
    status: "running",
    symbol: "^GSPC",
    date: "2026-05-15",
    created_at: "2026-05-15T10:00:00Z",
    started_at: "2026-05-15T10:00:05Z",
    finished_at: null,
    history_length: 252,
    error: null,
  },
  {
    job_id: "22222222-2222-2222-2222-222222222222",
    status: "queued",
    symbol: "QQQ",
    date: "2026-05-15",
    created_at: "2026-05-15T10:01:00Z",
    started_at: null,
    finished_at: null,
    history_length: 252,
    error: null,
  },
  {
    job_id: "33333333-3333-3333-3333-333333333333",
    status: "succeeded",
    symbol: "^GSPC",
    date: "2026-04-29",
    created_at: "2026-04-29T13:00:00Z",
    started_at: "2026-04-29T13:00:05Z",
    finished_at: "2026-04-29T13:01:11Z",
    history_length: 252,
    error: null,
  },
  {
    job_id: "44444444-4444-4444-4444-444444444444",
    status: "failed",
    symbol: "^GSPC",
    date: "2026-04-15",
    created_at: "2026-04-15T08:00:00Z",
    started_at: "2026-04-15T08:00:05Z",
    finished_at: "2026-04-15T08:00:30Z",
    history_length: 252,
    error: "fetch_market_history: HTTP 503",
  },
];

describe("TrainingPage (list)", () => {
  beforeEach(() => {
    fetchTrainJobsMock.mockReset();
    fetchTrainJobMock.mockReset();
  });
  afterEach(() => {
    // Drop any pending polling intervals — cleanup() runs from setup.ts
    // but the setInterval inside useEffect already cleared via the
    // return cleanup. Nothing extra to do here.
  });

  it("renders the empty state when no jobs are queued", async () => {
    fetchTrainJobsMock.mockResolvedValue({ items: [], total: 0, limit: 50, offset: 0 });
    const { default: TrainingPage } = await import("@/pages/training");
    render(<TrainingPage />);
    await waitFor(() =>
      expect(screen.getByText(/No training jobs in this backend instance/i)).toBeInTheDocument()
    );
  });

  it("renders status counts and job rows for queued / running / succeeded / failed", async () => {
    fetchTrainJobsMock.mockResolvedValue({
      items: SAMPLE_JOBS,
      total: SAMPLE_JOBS.length,
      limit: 50,
      offset: 0,
    });
    const { default: TrainingPage } = await import("@/pages/training");
    render(<TrainingPage />);
    await waitFor(() =>
      expect(screen.getByText(/Job queue/i)).toBeInTheDocument()
    );
    expect(screen.getAllByText(/running/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/queued/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/succeeded/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/failed/i).length).toBeGreaterThan(0);
    expect(screen.getByText(/^QQQ$/)).toBeInTheDocument();
  });
});

describe("TrainingDetailPage", () => {
  beforeEach(() => {
    fetchTrainJobsMock.mockReset();
    fetchTrainJobMock.mockReset();
    routerStore.query = { id: "abc-123" };
  });
  afterEach(() => {
    routerStore.query = {};
  });

  it("renders the running state details", async () => {
    fetchTrainJobMock.mockResolvedValue({
      job_id: "abc-123",
      status: "running",
      message: "Real Train queued.",
      error: null,
      result: null,
    });
    const { default: TrainingDetailPage } = await import("@/pages/training/[id]");
    render(<TrainingDetailPage />);
    await waitFor(() => expect(screen.getByText(/Real Train queued/i)).toBeInTheDocument());
    // Heading and id surface.
    expect(screen.getByText(/abc-123/)).toBeInTheDocument();
    expect(screen.getAllByText(/running/i).length).toBeGreaterThan(0);
  });

  it("renders predicted close and sentiment label when result is present", async () => {
    fetchTrainJobMock.mockResolvedValue({
      job_id: "abc-123",
      status: "succeeded",
      message: "ok",
      error: null,
      result: {
        sentiment: { label: "HAWKISH", score: 0.62 },
        prediction: { close: 5612.5, volatility: 0.014, horizon: "3d" },
        model: { runtime_mode: "real_train" },
      },
    });
    const { default: TrainingDetailPage } = await import("@/pages/training/[id]");
    render(<TrainingDetailPage />);
    await waitFor(() => expect(screen.getByText(/HAWKISH/i)).toBeInTheDocument());
    expect(screen.getByText("5612.50")).toBeInTheDocument();
    expect(screen.getByText("0.0140")).toBeInTheDocument();
  });
});
