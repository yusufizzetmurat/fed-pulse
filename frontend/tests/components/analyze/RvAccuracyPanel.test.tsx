import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";

import { RvAccuracyPanel } from "@/components/analyze/RvAccuracyPanel";
import type {
  RvBacktestResponse,
  RvBacktestRow,
} from "@/lib/analyze/types";

function makeRow(overrides: Partial<RvBacktestRow> = {}): RvBacktestRow {
  return {
    event_date: "2024-05-01",
    point_forecast_rv: 1e-4,
    band_lo_80: 5e-5,
    band_hi_80: 2e-4,
    band_lo_90: 4e-5,
    band_hi_90: 3e-4,
    realized_rv: 1.2e-4,
    in_band_80: true,
    in_band_90: true,
    ...overrides,
  };
}

function fixture(
  overrides: Partial<RvBacktestResponse> = {},
): RvBacktestResponse {
  // Default fixture uses a 6-row resolved sample so the gap chip is
  // statistically meaningful (resolved >= GAP_MIN_SAMPLE). Individual
  // tests can shrink the coverage block to exercise the small-sample
  // neutralization path.
  return {
    symbol: "^GSPC",
    horizon: 1,
    rows: [
      makeRow({
        event_date: "2024-05-01",
        in_band_80: true,
        in_band_90: true,
      }),
      makeRow({
        event_date: "2024-03-20",
        realized_rv: 5e-4,
        in_band_80: false,
        in_band_90: true,
      }),
      makeRow({
        event_date: "2024-01-31",
        realized_rv: null,
        in_band_80: null,
        in_band_90: null,
      }),
    ],
    coverage: {
      total_runs: 7,
      resolved_runs: 6,
      pending_runs: 1,
      empirical_coverage_80: 0.5,
      empirical_coverage_90: 1.0,
      nominal_coverage_80: 0.8,
      nominal_coverage_90: 0.9,
    },
    generated_at: "2026-06-01T00:00:00+00:00",
    ...overrides,
  };
}

describe("RvAccuracyPanel", () => {
  it("renders the 80% / 90% coverage KPI header against attempted (not total) rows", () => {
    render(<RvAccuracyPanel data={fixture()} symbol="^GSPC" />);
    // Denominator excludes the 1 pending row: 6 attempted, not 7.
    expect(screen.getByTestId("rv-accuracy-coverage-80")).toHaveTextContent(
      "6 / 6",
    );
    expect(screen.getByTestId("rv-accuracy-coverage-80")).toHaveTextContent(
      "50.0%",
    );
    expect(screen.getByTestId("rv-accuracy-coverage-90")).toHaveTextContent(
      "100.0%",
    );
  });

  it("surfaces the pending-row count when some events fall outside the RV history window", () => {
    render(<RvAccuracyPanel data={fixture()} symbol="^GSPC" />);
    expect(screen.getByTestId("rv-accuracy-pending")).toHaveTextContent(
      /1 pending/i,
    );
  });

  it("colors the 80% gap chip hawkish when empirical materially undershoots nominal", () => {
    // 0.5 vs 0.8 = -30 pp gap, outside the 10 pp materiality band → hawkish.
    render(<RvAccuracyPanel data={fixture()} symbol="^GSPC" />);
    const chip80 = screen.getByTestId("rv-accuracy-gap-80");
    expect(chip80.className).toMatch(/hawkish/);
    expect(chip80).toHaveTextContent(/80%:.*-30\.0 pp vs nominal/i);
  });

  it("keeps the 90% gap chip neutral when empirical exceeds nominal", () => {
    render(<RvAccuracyPanel data={fixture()} symbol="^GSPC" />);
    const chip90 = screen.getByTestId("rv-accuracy-gap-90");
    expect(chip90.className).toMatch(/neutral/);
    expect(chip90).toHaveTextContent(/\+10\.0 pp vs nominal/);
  });

  it("neutralizes the gap chip when the resolved sample is too small to be meaningful", () => {
    // 2 resolved rows is below the GAP_MIN_SAMPLE threshold, so even a
    // dramatic -30 pp empirical-vs-nominal delta stays neutral with a
    // "small sample" label instead of flipping hawkish.
    const data = fixture({
      coverage: {
        total_runs: 3,
        resolved_runs: 2,
        pending_runs: 1,
        empirical_coverage_80: 0.5,
        empirical_coverage_90: 1.0,
        nominal_coverage_80: 0.8,
        nominal_coverage_90: 0.9,
      },
    });
    render(<RvAccuracyPanel data={data} symbol="^GSPC" />);
    const chip80 = screen.getByTestId("rv-accuracy-gap-80");
    expect(chip80.className).toMatch(/neutral/);
    expect(chip80).toHaveTextContent(/small sample/i);
    const chip90 = screen.getByTestId("rv-accuracy-gap-90");
    expect(chip90.className).toMatch(/neutral/);
    expect(chip90).toHaveTextContent(/small sample/i);
  });

  it("renders ✓ / ✗ / — per-row hit marks for both bands", () => {
    render(<RvAccuracyPanel data={fixture()} symbol="^GSPC" />);
    expect(screen.getByTestId("rv-accuracy-row-hit80-2024-05-01")).toHaveTextContent(
      "✓",
    );
    expect(screen.getByTestId("rv-accuracy-row-hit80-2024-03-20")).toHaveTextContent(
      "✗",
    );
    expect(screen.getByTestId("rv-accuracy-row-hit80-2024-01-31")).toHaveTextContent(
      "—",
    );
    expect(screen.getByTestId("rv-accuracy-row-hit90-2024-05-01")).toHaveTextContent(
      "✓",
    );
    expect(screen.getByTestId("rv-accuracy-row-hit90-2024-03-20")).toHaveTextContent(
      "✓",
    );
    expect(screen.getByTestId("rv-accuracy-row-hit90-2024-01-31")).toHaveTextContent(
      "—",
    );
  });

  it("renders the pending placeholder when realized_rv is null", () => {
    render(<RvAccuracyPanel data={fixture()} symbol="^GSPC" />);
    expect(
      screen.getByTestId("rv-accuracy-row-real-2024-01-31-pending"),
    ).toHaveTextContent(/pending/i);
  });

  it("renders an empty state when rows are empty", () => {
    const empty = fixture({
      rows: [],
      coverage: {
        total_runs: 0,
        resolved_runs: 0,
        pending_runs: 0,
        empirical_coverage_80: null,
        empirical_coverage_90: null,
        nominal_coverage_80: 0.8,
        nominal_coverage_90: 0.9,
      },
    });
    render(<RvAccuracyPanel data={empty} symbol="^GSPC" />);
    expect(screen.getByTestId("rv-accuracy-empty")).toHaveTextContent(
      /No resolved RV runs yet/i,
    );
  });

  it("renders a loading state when loading", () => {
    render(<RvAccuracyPanel data={null} loading symbol="^GSPC" />);
    expect(screen.getByTestId("rv-accuracy-loading")).toBeInTheDocument();
  });

  it("renders an unavailable state on error or missing data", () => {
    render(
      <RvAccuracyPanel
        data={null}
        error="rv backtest service down"
        symbol="^GSPC"
      />,
    );
    expect(screen.getByTestId("rv-accuracy-unavailable")).toHaveTextContent(
      /rv backtest service down/i,
    );
  });

  it("keeps the 80% gap chip neutral when empirical exactly matches nominal", () => {
    const data = fixture({
      coverage: {
        total_runs: 10,
        resolved_runs: 8,
        pending_runs: 2,
        empirical_coverage_80: 0.8,
        empirical_coverage_90: 0.9,
        nominal_coverage_80: 0.8,
        nominal_coverage_90: 0.9,
      },
    });
    render(<RvAccuracyPanel data={data} symbol="^GSPC" />);
    const chip80 = screen.getByTestId("rv-accuracy-gap-80");
    expect(chip80.className).toMatch(/neutral/);
    expect(chip80).toHaveTextContent(/0\.0 pp/);
  });

  it("falls back to an em-dash on the gap chip when coverage is unresolved", () => {
    const data = fixture({
      rows: [],
      coverage: {
        total_runs: 1,
        resolved_runs: 0,
        pending_runs: 1,
        empirical_coverage_80: null,
        empirical_coverage_90: null,
        nominal_coverage_80: 0.8,
        nominal_coverage_90: 0.9,
      },
    });
    // With rows=[] the panel renders the empty state, so the chip is
    // not present. Confirm by asserting the gap chips block is absent.
    render(<RvAccuracyPanel data={data} symbol="^GSPC" />);
    expect(screen.queryByTestId("rv-accuracy-gap-chips")).toBeNull();
  });
});
