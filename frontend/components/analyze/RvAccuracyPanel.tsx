import * as React from "react";

import { Badge } from "@/components/ui/badge";
import { WorkspaceSection } from "@/components/analyze/WorkspaceSection";
import { cn } from "@/lib/utils";
import type {
  RvBacktestCoverage,
  RvBacktestResponse,
  RvBacktestRow,
} from "@/lib/analyze/types";

// RvAccuracyPanel — workspace-spine forecast variant. Renders empirical
// band coverage for the QLIKE-RV ensemble: how often the published 80%
// / 90% conformal bands actually contained the realized RV on recent
// FOMC dates. Sits immediately below the HarAccuracyPanel card on
// /analyze and shares its layout idiom (KPI header + compact table).
//
// Calibration framing: nominal vs empirical. The 80% band is calibrated
// so 80% of out-of-sample residuals should land inside; the chip flags
// when the recent FOMC window materially undershoots that target.

// Materiality threshold for the calibration gap chip. Anything within
// 10 percentage points of the nominal target is "neutral" — the
// hit-rate on a small FOMC window is noisy, so a small gap should not
// shout "miscalibrated". Outside that band the chip turns hawkish.
const GAP_MATERIAL_THRESHOLD = 0.10;

// Minimum resolved-row count before the gap chip is allowed to switch
// off neutral. Below this, a single in-band hit or miss can flip the
// empirical coverage by 30+ percentage points, which would surface as a
// false "over-confident bands" signal. The chip stays neutral until we
// have enough events to make the hit rate statistically meaningful.
const GAP_MIN_SAMPLE = 5;

function formatPct(value: number | null | undefined, digits: number = 1): string {
  if (value == null || !Number.isFinite(value)) return "—";
  return `${(value * 100).toFixed(digits)}%`;
}

function formatRv(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value) || value <= 0) return "—";
  // Daily realized VARIANCE → annualized vol % via sqrt(var * 252).
  const ann = Math.sqrt(value * 252) * 100;
  return `${ann.toFixed(1)}%`;
}

function formatBand(
  lo: number | null | undefined,
  hi: number | null | undefined,
): string {
  if (
    lo == null ||
    hi == null ||
    !Number.isFinite(lo) ||
    !Number.isFinite(hi)
  ) {
    return "—";
  }
  return `${formatRv(lo)} – ${formatRv(hi)}`;
}

interface KpiHeaderProps {
  coverage: RvBacktestCoverage;
}

function KpiHeader({ coverage }: KpiHeaderProps) {
  const cov80 = formatPct(coverage.empirical_coverage_80);
  const cov90 = formatPct(coverage.empirical_coverage_90);
  // Denominator is the count of rows we actually attempted to score —
  // exclude pending rows (event date inside HAR warmup or outside the
  // available RV history). Falls back to total_runs when an older
  // backend response omits ``pending_runs`` so the panel keeps
  // rendering during a rolling deploy.
  const pending = coverage.pending_runs ?? 0;
  const attempted = Math.max(coverage.total_runs - pending, coverage.resolved_runs);
  return (
    <div
      className="flex flex-wrap items-baseline justify-between gap-3 rounded-md border border-border bg-card/50 p-3"
      data-testid="rv-accuracy-kpi"
    >
      <div className="space-y-1">
        <p className="text-[11px] uppercase tracking-wide text-muted-foreground">
          80% band coverage
        </p>
        <p
          className="numeric text-3xl font-semibold tabular-nums"
          data-testid="rv-accuracy-coverage-80"
        >
          {coverage.resolved_runs} / {attempted}
          <span className="ml-2 text-base font-medium text-muted-foreground">
            ({cov80})
          </span>
        </p>
        {pending > 0 ? (
          <p
            className="text-[11px] text-muted-foreground"
            data-testid="rv-accuracy-pending"
          >
            {pending} pending (outside available RV history)
          </p>
        ) : null}
      </div>
      <div className="space-y-1 text-right">
        <p className="text-[11px] uppercase tracking-wide text-muted-foreground">
          90% band coverage
        </p>
        <p
          className="numeric text-sm font-medium tabular-nums"
          data-testid="rv-accuracy-coverage-90"
        >
          {coverage.resolved_runs} / {attempted} ({cov90})
        </p>
      </div>
    </div>
  );
}

interface GapChipsProps {
  coverage: RvBacktestCoverage;
}

// Renders the nominal-vs-empirical gap as a chip pair. Hawkish (red)
// when the empirical hit-rate is materially BELOW the nominal target —
// the bands are under-covering, the model is over-confident. Neutral
// otherwise (including over-coverage, which is conservative but does
// not break the calibration story). When the resolved sample is too
// small (< GAP_MIN_SAMPLE) both chips stay neutral with a "small sample"
// label so a 1-of-2-miss event does not surface as miscalibration.
function GapChips({ coverage }: GapChipsProps) {
  const cov80 = coverage.empirical_coverage_80;
  const cov90 = coverage.empirical_coverage_90;
  const gap80 = cov80 == null ? null : cov80 - coverage.nominal_coverage_80;
  const gap90 = cov90 == null ? null : cov90 - coverage.nominal_coverage_90;
  const smallSample = coverage.resolved_runs < GAP_MIN_SAMPLE;
  const variantFor = (gap: number | null): "neutral" | "hawkish" => {
    if (gap == null || smallSample) return "neutral";
    return gap < -GAP_MATERIAL_THRESHOLD ? "hawkish" : "neutral";
  };
  const labelFor = (gap: number | null): string => {
    if (gap == null) return "—";
    if (smallSample) return "small sample";
    const pct = (gap * 100).toFixed(1);
    if (gap > 0) return `+${pct} pp vs nominal`;
    if (gap < 0) return `${pct} pp vs nominal`;
    return `± 0.0 pp vs nominal`;
  };
  return (
    <div
      className="flex flex-wrap items-center gap-2"
      data-testid="rv-accuracy-gap-chips"
    >
      <p className="mr-1 text-[11px] uppercase tracking-wide text-muted-foreground">
        Calibration gap
      </p>
      <Badge
        variant={variantFor(gap80)}
        data-testid="rv-accuracy-gap-80"
      >
        80%: {labelFor(gap80)}
      </Badge>
      <Badge
        variant={variantFor(gap90)}
        data-testid="rv-accuracy-gap-90"
      >
        90%: {labelFor(gap90)}
      </Badge>
    </div>
  );
}

interface BacktestTableProps {
  rows: RvBacktestRow[];
}

function BacktestTable({ rows }: BacktestTableProps) {
  return (
    <div
      className="overflow-x-auto rounded-md border border-border"
      data-testid="rv-accuracy-table"
    >
      <table className="w-full text-sm">
        <thead className="bg-muted/40">
          <tr className="text-left text-[11px] uppercase tracking-wide text-muted-foreground">
            <th className="px-3 py-2 font-medium">Date</th>
            <th className="px-3 py-2 font-medium">Point</th>
            <th className="px-3 py-2 font-medium">80% band</th>
            <th className="px-3 py-2 font-medium">Realized</th>
            <th className="px-3 py-2 font-medium">In 80%</th>
            <th className="px-3 py-2 font-medium">In 90%</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => {
            const renderHit = (value: boolean | null): string => {
              if (value === true) return "✓";
              if (value === false) return "✗";
              return "—";
            };
            const hitClass = (value: boolean | null): string =>
              cn(
                "numeric text-sm font-semibold",
                value === true && "text-dovish",
                value === false && "text-hawkish",
                value == null && "text-muted-foreground",
              );
            return (
              <tr
                key={row.event_date}
                className="border-t border-border/60"
                data-testid={`rv-accuracy-row-${row.event_date}`}
              >
                <td className="numeric whitespace-nowrap px-3 py-2 tabular-nums">
                  {row.event_date}
                </td>
                <td className="numeric px-3 py-2 tabular-nums">
                  {formatRv(row.point_forecast_rv)}
                </td>
                <td className="numeric px-3 py-2 tabular-nums text-muted-foreground">
                  {formatBand(row.band_lo_80, row.band_hi_80)}
                </td>
                <td className="numeric px-3 py-2 tabular-nums">
                  {row.realized_rv == null ? (
                    <span
                      className="text-[11px] text-muted-foreground"
                      data-testid={`rv-accuracy-row-real-${row.event_date}-pending`}
                    >
                      pending
                    </span>
                  ) : (
                    formatRv(row.realized_rv)
                  )}
                </td>
                <td className="px-3 py-2">
                  <span
                    className={hitClass(row.in_band_80)}
                    data-testid={`rv-accuracy-row-hit80-${row.event_date}`}
                    aria-label={
                      row.in_band_80 === true
                        ? "Inside 80% band"
                        : row.in_band_80 === false
                          ? "Outside 80% band"
                          : "Pending"
                    }
                  >
                    {renderHit(row.in_band_80)}
                  </span>
                </td>
                <td className="px-3 py-2">
                  <span
                    className={hitClass(row.in_band_90)}
                    data-testid={`rv-accuracy-row-hit90-${row.event_date}`}
                    aria-label={
                      row.in_band_90 === true
                        ? "Inside 90% band"
                        : row.in_band_90 === false
                          ? "Outside 90% band"
                          : "Pending"
                    }
                  >
                    {renderHit(row.in_band_90)}
                  </span>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export interface RvAccuracyPanelProps {
  data: RvBacktestResponse | null;
  loading?: boolean;
  error?: string | null;
  symbol?: string;
}

export function RvAccuracyPanel({
  data,
  loading = false,
  error = null,
  symbol,
}: RvAccuracyPanelProps) {
  if (loading) {
    return (
      <WorkspaceSection
        title="QLIKE-RV band coverage"
        description="Backtest of the published 80% / 90% bands against realized RV"
        variant="forecast"
      >
        <p
          className="text-xs text-muted-foreground"
          data-testid="rv-accuracy-loading"
        >
          Loading RV backtest…
        </p>
      </WorkspaceSection>
    );
  }

  if (error || !data) {
    return (
      <WorkspaceSection
        title="QLIKE-RV band coverage"
        description="Backtest of the published 80% / 90% bands against realized RV"
        variant="forecast"
      >
        <p
          className="text-xs text-muted-foreground"
          data-testid="rv-accuracy-unavailable"
        >
          {error ?? "Backtest unavailable. Retry shortly."}
        </p>
      </WorkspaceSection>
    );
  }

  const symbolBadge = symbol ?? data.symbol;

  if (data.rows.length === 0) {
    return (
      <WorkspaceSection
        title="QLIKE-RV band coverage"
        description="Backtest of the published 80% / 90% bands against realized RV"
        variant="forecast"
      >
        <div className="space-y-2">
          <div className="flex items-center justify-end">
            {symbolBadge ? (
              <Badge
                variant="outline"
                className="numeric text-[10px] tabular-nums"
                data-testid="rv-accuracy-symbol"
              >
                {symbolBadge}
              </Badge>
            ) : null}
          </div>
          <p
            className="text-xs text-muted-foreground"
            data-testid="rv-accuracy-empty"
          >
            No resolved RV runs yet
          </p>
        </div>
      </WorkspaceSection>
    );
  }

  return (
    <WorkspaceSection
      title="QLIKE-RV band coverage"
      description="Backtest of the published 80% / 90% bands against realized RV"
      variant="forecast"
    >
      <div className="space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <p className="text-[11px] text-muted-foreground">
            Empirical hit rate of the published conformal bands across
            the last persisted FOMC events. Resolved rows ride the h=1
            forecast on the leading RV prefix.
          </p>
          {symbolBadge ? (
            <Badge
              variant="outline"
              className="numeric text-[10px] tabular-nums"
              data-testid="rv-accuracy-symbol"
            >
              {symbolBadge}
            </Badge>
          ) : null}
        </div>
        <KpiHeader coverage={data.coverage} />
        <GapChips coverage={data.coverage} />
        <BacktestTable rows={data.rows} />
      </div>
    </WorkspaceSection>
  );
}

export default RvAccuracyPanel;
