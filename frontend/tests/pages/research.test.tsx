import { describe, expect, it, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";

vi.mock("sonner", () => ({
  toast: { success: vi.fn(), error: vi.fn(), info: vi.fn() },
  Toaster: () => null,
}));

vi.mock("next/router", () => ({
  useRouter: () => ({ isReady: true, query: {}, push: vi.fn() }),
}));

vi.mock("next/head", () => ({
  default: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}));

const fetchResearchArtifactsMock = vi.fn();
vi.mock("@/lib/analyze/api", () => ({
  resolveApiBaseUrl: () => "http://localhost:8000",
  fetchResearchArtifacts: (...args: unknown[]) => fetchResearchArtifactsMock(...args),
}));

const EMPTY_RESPONSE = {
  artifacts_root: "/data/artifacts",
  sections: {
    phase3: [],
    cross_bank: [],
    cross_asset: [],
    next_fomc: [],
  },
  encoder_bakeoff: {
    available: false,
    coverage: null,
    rows: [],
    source_files: [],
  },
  cross_bank_transfer: {
    available: false,
    metric_name: "macro_f1",
    sources: [],
    targets: [],
    cells: [],
    source_files: [],
  },
};

const POPULATED_RESPONSE = {
  artifacts_root: "/data/artifacts",
  sections: {
    phase3: [
      {
        relative_path: "phase3/run-a/aggregate.json",
        size_bytes: 4096,
        modified_at: "2026-05-01T12:00:00+00:00",
        suffix: ".json",
      },
    ],
    cross_bank: [],
    cross_asset: [],
    next_fomc: [],
  },
  encoder_bakeoff: {
    available: true,
    coverage: 0.95,
    rows: [
      {
        encoder_key: "bert-base-uncased",
        checkpoint: "bert-base-uncased",
        seeds: [11, 29, 47, 71, 97],
        macro_f1_values: [0.51, 0.52, 0.53, 0.55, 0.54],
        macro_f1_mean: 0.53,
        macro_f1_ci_low: 0.5,
        macro_f1_ci_high: 0.56,
        weighted_f1_mean: 0.57,
        accuracy_mean: 0.6,
        cohen_kappa: 0.31,
      },
      {
        encoder_key: "fomc-roberta",
        checkpoint: "yiyanghkust/finbert-tone",
        seeds: [11, 29, 47, 71, 97],
        macro_f1_values: [0.59, 0.61, 0.6, 0.62, 0.6],
        macro_f1_mean: 0.604,
        macro_f1_ci_low: 0.58,
        macro_f1_ci_high: 0.63,
        weighted_f1_mean: 0.62,
        accuracy_mean: 0.66,
        cohen_kappa: 0.4,
      },
    ],
    source_files: ["phase3/run-a/aggregate.json"],
  },
  cross_bank_transfer: {
    available: false,
    metric_name: "macro_f1",
    sources: [],
    targets: [],
    cells: [],
    source_files: [],
  },
};

describe("ResearchPage", () => {
  beforeEach(() => {
    fetchResearchArtifactsMock.mockReset();
  });

  it("renders the empty-state when no artefacts are present", async () => {
    fetchResearchArtifactsMock.mockResolvedValue(EMPTY_RESPONSE);
    const { default: ResearchPage } = await import("@/pages/research");
    render(<ResearchPage />);
    await waitFor(() =>
      expect(screen.getByText(/No bake-off artefacts/i)).toBeInTheDocument()
    );
  });

  it("renders the encoder bake-off table when artefacts are present", async () => {
    fetchResearchArtifactsMock.mockResolvedValue(POPULATED_RESPONSE);
    const { default: ResearchPage } = await import("@/pages/research");
    render(<ResearchPage />);
    // Two cells per encoder (encoder column + checkpoint column).
    await waitFor(() =>
      expect(screen.getAllByText(/bert-base-uncased/i).length).toBeGreaterThanOrEqual(2)
    );
    expect(screen.getAllByText(/fomc-roberta/i).length).toBeGreaterThan(0);
    // Macro-F1 mean appears in the row body.
    expect(screen.getByText("0.530")).toBeInTheDocument();
    expect(screen.getByText("0.604")).toBeInTheDocument();
  });
});
