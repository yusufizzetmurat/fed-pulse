import { describe, expect, it, vi, beforeEach } from "vitest";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

const postAnalyzeMock = vi.fn();
const postAnalyzeAnalogsMock = vi.fn();
const postResearchBacktestMock = vi.fn();
const fetchResearchRegistryMock = vi.fn();

vi.mock("@/lib/analyze/api", () => ({
  resolveApiBaseUrl: () => "http://localhost:8000",
  postAnalyze: (...args: unknown[]) => postAnalyzeMock(...args),
  postAnalyzeAnalogs: (...args: unknown[]) => postAnalyzeAnalogsMock(...args),
  postResearchBacktest: (...args: unknown[]) => postResearchBacktestMock(...args),
  fetchResearchRegistry: (...args: unknown[]) => fetchResearchRegistryMock(...args),
}));

const REGISTRY_RESPONSE = {
  available: true,
  surface: "dual" as const,
  baseline: null,
  rows: [
    {
      encoder_alias: "bge-large",
      encoder_display: "BGE-large",
      dual_f1: 0.4321,
      cls_f1: 0.4011,
      regression_f1: null,
      delta_dual: 0.0231,
      delta_cls: 0.0102,
      is_winner: true,
      checkpoint_relpath: "ckpts/bge.pt",
      cache_uri: null,
      notes: "winner on dual",
    },
    {
      encoder_alias: "finbert",
      encoder_display: "FinBERT",
      dual_f1: 0.4000,
      cls_f1: 0.3801,
      regression_f1: null,
      delta_dual: -0.0090,
      delta_cls: -0.0108,
      is_winner: false,
      checkpoint_relpath: null,
      cache_uri: null,
      notes: "loser",
    },
  ],
  rejected_count: 3,
  training_package_id: "tp-2026-05",
  head: "dual",
  seeds: [11, 29, 47],
  source_wiki_section: "16",
};

const ANALYZE_RESULT = {
  sentiment: {
    label: "hawkish",
    score: 0.812,
    is_in_distribution: true,
    ood_energy: -3.21,
  },
  prediction: { close: 5500, volatility: 0.0123, horizon: "10d" },
  market: {
    symbol: "^GSPC",
    requested_date: "2024-09-18",
    date_used: "2024-09-18",
    close: 5400,
    volatility_5d: 0.011,
  },
  regime_classification: {
    argmax_class: "calm",
    coverage: 0.62,
  },
  multi_axis: {
    stance: { label: "hawkish", confidence: 0.74 },
    factor: null,
    certainty: null,
  },
};

const ANALOGS_RESULT = {
  analogs: [],
  index_size: 12,
  encoder_alias: "bge-large",
};

const BACKTEST_RESULT = {
  trades: [],
  n_trades: 8,
  sharpe: 1.21,
  hit_rate: 0.625,
  max_dd_pct: -3.4,
  cum_return_pct: 12.1,
  benchmark_cum_pct: 4.2,
  alpha_cum_pct: 7.9,
  horizon_days: 5,
  symbol: "^GSPC",
};

async function importComponent() {
  const mod = await import("@/components/research/TerminalTab");
  return mod.TerminalTab;
}

describe("TerminalTab", () => {
  beforeEach(() => {
    postAnalyzeMock.mockReset();
    postAnalyzeAnalogsMock.mockReset();
    postResearchBacktestMock.mockReset();
    fetchResearchRegistryMock.mockReset();
    fetchResearchRegistryMock.mockResolvedValue(REGISTRY_RESPONSE);
    postAnalyzeMock.mockResolvedValue(ANALYZE_RESULT);
    postAnalyzeAnalogsMock.mockResolvedValue(ANALOGS_RESULT);
    postResearchBacktestMock.mockResolvedValue(BACKTEST_RESULT);
  });

  it("renders the active checkpoint strip after fetching the registry", async () => {
    const TerminalTab = await importComponent();
    render(<TerminalTab />);
    expect(screen.getAllByText(/Active checkpoint/i).length).toBeGreaterThan(0);
    await waitFor(() => {
      expect(fetchResearchRegistryMock).toHaveBeenCalledWith(
        "http://localhost:8000",
        expect.objectContaining({ surface: "dual", includeRejected: false }),
      );
    });
    expect(
      await screen.findByText(/2 shown · 3 rejected · TP=tp-2026-05/i),
    ).toBeInTheDocument();
    expect(screen.getByText(/head=dual · seeds=11,29,47/i)).toBeInTheDocument();
  });

  it("refetches the registry when the surface picker flips to cls", async () => {
    const TerminalTab = await importComponent();
    render(<TerminalTab />);
    await waitFor(() => expect(fetchResearchRegistryMock).toHaveBeenCalledTimes(1));
    const user = userEvent.setup();
    await user.selectOptions(screen.getByLabelText(/surface/i), "cls");
    await waitFor(() =>
      expect(fetchResearchRegistryMock).toHaveBeenLastCalledWith(
        "http://localhost:8000",
        expect.objectContaining({ surface: "cls" }),
      ),
    );
  });

  it("populates Stance + Vol regime panels after Run analysis fires postAnalyze", async () => {
    const TerminalTab = await importComponent();
    render(<TerminalTab />);
    await waitFor(() => expect(fetchResearchRegistryMock).toHaveBeenCalled());
    const user = userEvent.setup();
    await user.click(screen.getByRole("button", { name: /run analysis/i }));
    await waitFor(() => expect(postAnalyzeMock).toHaveBeenCalledTimes(1));
    expect(postAnalyzeMock).toHaveBeenCalledWith(
      "http://localhost:8000",
      expect.objectContaining({ symbol: "^GSPC", horizon: "10d" }),
    );
    expect(postAnalyzeAnalogsMock).toHaveBeenCalledTimes(1);
    expect(await screen.findByText("HAWKISH")).toBeInTheDocument();
    expect(screen.getByText("0.812")).toBeInTheDocument();
    expect(screen.getByText("calm")).toBeInTheDocument();
    expect(screen.getByText("0.0123")).toBeInTheDocument();
    expect(screen.getByText("5500.00")).toBeInTheDocument();
  });

  it("auto-runs the backtest once analyze resolves and surfaces Sharpe / HitRate / MaxDD", async () => {
    const TerminalTab = await importComponent();
    render(<TerminalTab />);
    await waitFor(() => expect(fetchResearchRegistryMock).toHaveBeenCalled());
    const user = userEvent.setup();
    await user.click(screen.getByRole("button", { name: /run analysis/i }));
    await waitFor(() => expect(postResearchBacktestMock).toHaveBeenCalledTimes(1));
    expect(postResearchBacktestMock).toHaveBeenCalledWith(
      "http://localhost:8000",
      expect.objectContaining({ symbol: "^GSPC", horizon_days: 5 }),
    );
    expect(await screen.findByText("1.21")).toBeInTheDocument();
    expect(screen.getByText("62.5%")).toBeInTheDocument();
    expect(screen.getByText("-3.40%")).toBeInTheDocument();
  });

  it("loads a sample statement when the picker fires and updates the analysis date", async () => {
    const TerminalTab = await importComponent();
    render(<TerminalTab />);
    await waitFor(() => expect(fetchResearchRegistryMock).toHaveBeenCalled());
    const user = userEvent.setup();
    const picker = screen.getByLabelText(/load sample/i) as HTMLSelectElement;
    const options = within(picker).getAllByRole("option") as HTMLOptionElement[];
    const sample = options.find((opt) => opt.value !== "");
    expect(sample).toBeTruthy();
    await user.selectOptions(picker, sample!.value);
    const dateInput = screen.getByLabelText(/analysis date/i) as HTMLInputElement;
    expect(dateInput.value).toMatch(/^\d{4}-\d{2}-\d{2}$/);
  });

  it("renders the registry table rows in response order with winner / loser styling", async () => {
    const TerminalTab = await importComponent();
    render(<TerminalTab />);
    await waitFor(() => expect(fetchResearchRegistryMock).toHaveBeenCalled());
    const table = await screen.findByRole("table");
    const rows = within(table).getAllByRole("row");
    // Header + 2 data rows.
    expect(rows).toHaveLength(3);
    expect(within(rows[1]).getByText("BGE-large")).toBeInTheDocument();
    expect(within(rows[1]).getByText("0.4321")).toBeInTheDocument();
    expect(within(rows[1]).getByText("+0.0231")).toBeInTheDocument();
    expect(within(rows[2]).getByText("FinBERT")).toBeInTheDocument();
    expect(within(rows[2]).getByText("-0.0090")).toBeInTheDocument();
    // Loser row carries the dimmed class.
    expect(rows[2].className).toMatch(/opacity-50/);
    expect(rows[1].className).not.toMatch(/opacity-50/);
  });

  it("surfaces an error when postAnalyze rejects and clears the running state", async () => {
    postAnalyzeMock.mockRejectedValueOnce(new Error("backend down"));
    const TerminalTab = await importComponent();
    render(<TerminalTab />);
    await waitFor(() => expect(fetchResearchRegistryMock).toHaveBeenCalled());
    const user = userEvent.setup();
    await user.click(screen.getByRole("button", { name: /run analysis/i }));
    // The component pipes raw errors through errorMessage(), which
    // collapses unknown failures to a categorical "Model unavailable"
    // string. Asserting on that surface keeps the test pinned to the
    // visible contract rather than the inner exception text.
    expect(await screen.findByRole("alert")).toHaveTextContent(/Model unavailable/i);
    expect(screen.getByRole("button", { name: /run analysis/i })).toBeEnabled();
  });
});
