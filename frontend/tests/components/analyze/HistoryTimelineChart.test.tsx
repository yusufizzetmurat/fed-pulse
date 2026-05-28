import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";

// Recharts renders inside a <ResponsiveContainer> sized via the parent's
// box, which jsdom collapses to 0×0 — the SVG never paints. The mock
// below substitutes each primitive with a deterministic <div> that
// preserves children and the props we want to assert on, so the smoke
// test can observe presence/absence without driving the real chart.
vi.mock("recharts", () => {
  const passthrough =
    (name: string) =>
    ({ children, ...rest }: { children?: React.ReactNode } & Record<string, unknown>) => (
      <div data-testid={`rc-${name}`} data-props={JSON.stringify(rest)}>
        {children}
      </div>
    );
  return {
    CartesianGrid: passthrough("cartesian-grid"),
    ComposedChart: ({ children, data }: { children?: React.ReactNode; data?: unknown[] }) => (
      <div data-testid="rc-composed-chart" data-row-count={(data ?? []).length}>
        {children}
        {/* Expose the per-row scatter dots as discrete nodes so the
            smoke test can count them and reason about the right-axis
            vol line covering only the rows that carry the metric. */}
        {(data ?? []).map((row, idx) => {
          const r = row as { vol?: number | null; date?: string };
          return (
            <div
              key={idx}
              data-testid="rc-row"
              data-has-vol={r.vol != null ? "true" : "false"}
              data-date={r.date}
            />
          );
        })}
      </div>
    ),
    Line: ({ dataKey, yAxisId }: { dataKey?: string; yAxisId?: string }) => (
      <div data-testid={`rc-line-${String(dataKey)}`} data-y-axis={yAxisId} />
    ),
    ResponsiveContainer: ({ children }: { children?: React.ReactNode }) => (
      <div data-testid="rc-responsive-container">{children}</div>
    ),
    Scatter: ({ dataKey, yAxisId }: { dataKey?: string; yAxisId?: string }) => (
      <div data-testid={`rc-scatter-${String(dataKey)}`} data-y-axis={yAxisId} />
    ),
    Tooltip: () => <div data-testid="rc-tooltip" />,
    XAxis: () => <div data-testid="rc-x-axis" />,
    YAxis: ({ yAxisId, orientation }: { yAxisId?: string; orientation?: string }) => (
      <div
        data-testid={`rc-y-axis-${String(yAxisId ?? "default")}`}
        data-orientation={orientation ?? "left"}
      />
    ),
  };
});

import { HistoryTimelineChart, type HistoryTimelineRow } from "@/components/analyze/HistoryTimelineChart";

function makeRow(overrides: Partial<HistoryTimelineRow> = {}): HistoryTimelineRow {
  return {
    id: overrides.id ?? "run-1",
    created_at: overrides.created_at ?? "2026-04-01T12:00:00Z",
    symbol: overrides.symbol ?? "^GSPC",
    document_date: overrides.document_date ?? "2026-04-01",
    horizon: overrides.horizon ?? "10d",
    forecast_mode: "fast",
    stance: overrides.stance ?? "neutral",
    sentiment_score: overrides.sentiment_score ?? 0,
    predicted_close: null,
    current_close: null,
    predicted_volatility: null,
    text_excerpt: null,
    argmax_regime: overrides.argmax_regime ?? "normal",
    argmax_probability: 0.5,
    regime_set_size: 2,
    forward_realized_vol_10d: overrides.forward_realized_vol_10d ?? null,
  };
}

describe("HistoryTimelineChart", () => {
  it("renders the empty-state card when rows is empty", () => {
    render(<HistoryTimelineChart rows={[]} />);
    expect(screen.getByText(/No history yet/i)).toBeInTheDocument();
    // No chart canvas should mount in the empty branch.
    expect(screen.queryByTestId("rc-composed-chart")).toBeNull();
  });

  it("renders one dot per row with mixed regimes and no collisions", () => {
    const rows = [
      makeRow({ id: "a", document_date: "2026-03-01", argmax_regime: "calm", sentiment_score: -0.7 }),
      makeRow({ id: "b", document_date: "2026-03-08", argmax_regime: "normal", sentiment_score: 0.1 }),
      makeRow({ id: "c", document_date: "2026-03-15", argmax_regime: "high", sentiment_score: 0.8 }),
    ];
    render(<HistoryTimelineChart rows={rows} />);
    const chart = screen.getByTestId("rc-composed-chart");
    expect(chart).toHaveAttribute("data-row-count", "3");
    // Scatter dots are emitted as one node per row; uniqueness on the
    // x-axis (document_date) means there are no overlapping label
    // collisions for this fixture.
    const dots = screen.getAllByTestId("rc-row");
    expect(dots).toHaveLength(3);
    const dates = dots.map((node) => node.getAttribute("data-date"));
    expect(new Set(dates).size).toBe(3);
    // The stance scatter primitive is mounted exactly once.
    expect(screen.getByTestId("rc-scatter-stance")).toBeInTheDocument();
  });

  it("hides the right-hand vol axis when no row carries forward_realized_vol_10d", () => {
    const rows = [
      makeRow({ id: "a", document_date: "2026-03-01" }),
      makeRow({ id: "b", document_date: "2026-03-08" }),
    ];
    render(<HistoryTimelineChart rows={rows} />);
    expect(screen.queryByTestId("rc-y-axis-vol")).toBeNull();
    expect(screen.queryByTestId("rc-line-vol")).toBeNull();
    // Card copy mirrors the axis state so the user sees a matching note.
    expect(screen.getByText(/No forward vol data on these rows\./i)).toBeInTheDocument();
  });

  it("shows the right axis and the vol line, with vol present only on the rows that carry it", () => {
    const rows = [
      makeRow({ id: "a", document_date: "2026-03-01", forward_realized_vol_10d: 0.12 }),
      makeRow({ id: "b", document_date: "2026-03-08", forward_realized_vol_10d: null }),
      makeRow({ id: "c", document_date: "2026-03-15", forward_realized_vol_10d: 0.18 }),
    ];
    render(<HistoryTimelineChart rows={rows} />);
    // Right axis primitive is mounted and orientation is right.
    const volAxis = screen.getByTestId("rc-y-axis-vol");
    expect(volAxis).toHaveAttribute("data-orientation", "right");
    // The vol line is bound to the vol axis.
    const volLine = screen.getByTestId("rc-line-vol");
    expect(volLine).toHaveAttribute("data-y-axis", "vol");
    // Only the two rows carrying a numeric vol contribute to the line;
    // the middle row is rendered but its vol slot stays null so the
    // primitive's connectNulls bridge is the only thing joining the
    // two real points.
    const dots = screen.getAllByTestId("rc-row");
    const volBearing = dots.filter((node) => node.getAttribute("data-has-vol") === "true");
    expect(volBearing).toHaveLength(2);
    expect(volBearing.map((d) => d.getAttribute("data-date"))).toEqual([
      "2026-03-01",
      "2026-03-15",
    ]);
  });
});
