import { describe, expect, it, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

async function openBakeoffTab() {
  // The default tab is "What it predicts"; the bake-off table renders
  // under the "Bake-off" tab via radix Tabs (only the active tab's
  // content is mounted), so click in before asserting on bake-off DOM.
  // userEvent dispatches pointer events the way radix-ui expects.
  const user = userEvent.setup();
  const trigger = await screen.findByRole("tab", { name: /bake-off/i });
  await user.click(trigger);
}

async function openTransferTab() {
  const user = userEvent.setup();
  const trigger = await screen.findByRole("tab", { name: /transfer/i });
  await user.click(trigger);
}

async function openFilesTab() {
  const user = userEvent.setup();
  const trigger = await screen.findByRole("tab", { name: /^files$/i });
  await user.click(trigger);
}

// Recharts renders its <Bar> children into SVG inside a sized
// ResponsiveContainer; in jsdom the container collapses to 0×0 and the
// nested Cell elements never reach the DOM. Mock the chart primitives so
// the rendered Cell surfaces its `fill` prop as a data attribute we can
// assert on.
vi.mock("recharts", () => {
  const passthrough = ({ children }: { children?: React.ReactNode }) => (
    <div>{children}</div>
  );
  return {
    Bar: passthrough,
    BarChart: passthrough,
    CartesianGrid: () => <div data-testid="rc-grid" />,
    Cell: ({ fill }: { fill?: string }) => (
      <div data-testid="rc-bar-cell" data-fill={fill ?? ""} />
    ),
    ErrorBar: () => <div data-testid="rc-error-bar" />,
    ResponsiveContainer: passthrough,
    Tooltip: () => <div data-testid="rc-tooltip" />,
    XAxis: () => <div data-testid="rc-x-axis" />,
    YAxis: () => <div data-testid="rc-y-axis" />,
  };
});

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
    await openBakeoffTab();
    await waitFor(() =>
      expect(screen.getByText(/No bake-off artefacts/i)).toBeInTheDocument()
    );
  });

  it("renders the encoder bake-off table when artefacts are present", async () => {
    fetchResearchArtifactsMock.mockResolvedValue(POPULATED_RESPONSE);
    const { default: ResearchPage } = await import("@/pages/research");
    render(<ResearchPage />);
    await openBakeoffTab();
    // Two cells per encoder (encoder column + checkpoint column).
    await waitFor(() =>
      expect(screen.getAllByText(/bert-base-uncased/i).length).toBeGreaterThanOrEqual(2)
    );
    expect(screen.getAllByText(/fomc-roberta/i).length).toBeGreaterThan(0);
    // Macro-F1 mean appears in the row body.
    expect(screen.getByText("0.530")).toBeInTheDocument();
    expect(screen.getByText("0.604")).toBeInTheDocument();
  });

  it("renders heatmap cells with the numeric metric and bank labels", async () => {
    const response = {
      ...POPULATED_RESPONSE,
      cross_bank_transfer: {
        available: true,
        metric_name: "macro_f1",
        sources: ["FED", "ECB"],
        targets: ["FED", "ECB"],
        cells: [
          { source: "FED", target: "FED", metric: 0.612 },
          { source: "FED", target: "ECB", metric: 0.481 },
          { source: "ECB", target: "FED", metric: 0.395 },
          { source: "ECB", target: "ECB", metric: 0.557 },
        ],
        source_files: ["cross_bank/transfer.json"],
      },
    };
    fetchResearchArtifactsMock.mockResolvedValue(response);
    const { default: ResearchPage } = await import("@/pages/research");
    render(<ResearchPage />);
    await openTransferTab();

    // Numeric metric renders inside each populated cell (3-decimal format).
    // Both heatmap + values tables carry the same tooltip title; the heatmap
    // cell is the one keyed `heat-…` and tinted via an inline background.
    const titledCells = await screen.findAllByTitle(/Trained on FED, evaluated on ECB/);
    const heatmapCell = titledCells.find(
      (el) => el.getAttribute("style")?.includes("background-color") ?? false,
    );
    expect(heatmapCell).toBeDefined();
    expect(heatmapCell!.textContent).toContain("0.481");

    // Source + target bank labels are present as row + column headers.
    expect(screen.getAllByText("FED").length).toBeGreaterThan(0);
    expect(screen.getAllByText("ECB").length).toBeGreaterThan(0);
  });

  it("renders the Files tab with at least one artefact row under the new label", async () => {
    fetchResearchArtifactsMock.mockResolvedValue(POPULATED_RESPONSE);
    const { default: ResearchPage } = await import("@/pages/research");
    render(<ResearchPage />);
    await openFilesTab();

    // Tab + card title both read "Files" (the rename) rather than the old
    // "Downloads" label that had no matching download endpoint.
    expect(await screen.findByRole("tab", { name: /^files$/i })).toBeInTheDocument();
    expect(screen.queryByText(/^Downloads$/)).toBeNull();

    // The populated phase3 artefact row surfaces its relative path.
    expect(
      screen.getByText("phase3/run-a/aggregate.json"),
    ).toBeInTheDocument();
  });

  it("paints bake-off bars with a non-default colour ramp", async () => {
    fetchResearchArtifactsMock.mockResolvedValue(POPULATED_RESPONSE);
    const { default: ResearchPage } = await import("@/pages/research");
    render(<ResearchPage />);
    await openBakeoffTab();

    const cells = await screen.findAllByTestId("rc-bar-cell");
    expect(cells.length).toBeGreaterThan(0);
    // The new ramp emits literal hsl() so Recharts can write it directly
    // onto the SVG fill attribute; the old `hsla(var(--primary) / α)` form
    // never resolved inside SVG and rendered as black.
    for (const cell of cells) {
      const fill = cell.getAttribute("data-fill") ?? "";
      expect(fill).toMatch(/^hsl\(/);
      expect(fill).not.toContain("var(");
    }
  });

  it("renders source-file badges with a friendly encoder label, not the raw path", async () => {
    const rawPath =
      "data/artifacts/continued_pretraining/finbert_fed_adjacent_20260515T104824Z_s11/checkpoint";
    const response = {
      ...POPULATED_RESPONSE,
      encoder_bakeoff: {
        ...POPULATED_RESPONSE.encoder_bakeoff,
        source_files: [rawPath],
      },
    };
    fetchResearchArtifactsMock.mockResolvedValue(response);
    const { default: ResearchPage } = await import("@/pages/research");
    render(<ResearchPage />);
    await openBakeoffTab();
    const badge = await screen.findByTitle(rawPath);
    expect(badge.textContent).toBe("FinBERT (Fed-adjacent)");
    expect(badge.textContent).not.toContain("/");
    expect(badge.textContent).not.toContain("checkpoint");
    // The raw path must not surface anywhere in the rendered DOM text.
    expect(screen.queryByText(rawPath)).toBeNull();
  });
});
