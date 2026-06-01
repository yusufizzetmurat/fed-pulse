import * as React from "react";

import { Badge } from "@/components/ui/badge";
import { WorkspaceSection } from "@/components/analyze/WorkspaceSection";
import { cn } from "@/lib/utils";
import type {
  HarAccuracyMetrics,
  HarTercileBacktestResponse,
  HarTercileBacktestRow,
  HarTercileLabel,
} from "@/lib/analyze/types";

// HarAccuracyPanel — workspace-spine forecast variant. Surfaces the
// last 10 resolved FOMC runs for ^GSPC with predicted vs realized
// tercile + an aggregate accuracy KPI. Sits immediately below the
// HarRegimeHeadline card on /analyze and shares its tercile palette
// (low → dovish, medium → neutral, high → hawkish).

const TERCILE_ORDER: readonly HarTercileLabel[] = ["low", "medium", "high"];

function tercileBadgeVariant(
  label: HarTercileLabel,
): "hawkish" | "dovish" | "neutral" {
  if (label === "low") return "dovish";
  if (label === "high") return "hawkish";
  return "neutral";
}

function annualizedVolPct(rv: number | null | undefined): string {
  // ``realized_rv`` is daily realized VARIANCE — same convention the
  // HarRegimeHeadline + VolatilityOutlookCard cards use upstream, so
  // annualisation is sqrt(variance * 252) * 100. Mirrors
  // HarRegimeHeadline.annualizedVolPct deliberately so the two
  // forecast-spine cards report comparable numbers.
  if (rv == null || !Number.isFinite(rv) || rv <= 0) return "—";
  const ann = Math.sqrt(rv * 252) * 100;
  return `${ann.toFixed(1)}%`;
}

function formatProbPct(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return "—";
  return `${(value * 100).toFixed(0)}%`;
}

function formatAccuracyPct(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return "—";
  return `${(value * 100).toFixed(1)}%`;
}

interface KpiHeaderProps {
  metrics: HarAccuracyMetrics;
}

function KpiHeader({ metrics }: KpiHeaderProps) {
  const accuracy = formatAccuracyPct(metrics.accuracy_overall);
  return (
    <div
      className="flex flex-wrap items-baseline justify-between gap-3 rounded-md border border-border bg-card/50 p-3"
      data-testid="har-accuracy-kpi"
    >
      <div className="space-y-1">
        <p className="text-[11px] uppercase tracking-wide text-muted-foreground">
          Aggregate accuracy
        </p>
        <p
          className="numeric text-3xl font-semibold tabular-nums"
          data-testid="har-accuracy-overall"
        >
          {accuracy}
        </p>
      </div>
      <div className="space-y-1 text-right">
        <p className="text-[11px] uppercase tracking-wide text-muted-foreground">
          Resolved runs
        </p>
        <p
          className="numeric text-sm font-medium tabular-nums"
          data-testid="har-accuracy-counter"
        >
          {metrics.resolved_runs} / {metrics.total_runs}
        </p>
      </div>
    </div>
  );
}

interface PerTercileChipsProps {
  metrics: HarAccuracyMetrics;
}

function PerTercileChips({ metrics }: PerTercileChipsProps) {
  return (
    <div
      className="flex flex-wrap items-center gap-2"
      data-testid="har-accuracy-per-tercile"
    >
      <p className="mr-1 text-[11px] uppercase tracking-wide text-muted-foreground">
        Hit rate by predicted tercile
      </p>
      {TERCILE_ORDER.map((label) => {
        const value = metrics.per_tercile_hit_rate?.[label];
        const empty = value == null || !Number.isFinite(value);
        return (
          <Badge
            key={label}
            variant={empty ? "outline" : tercileBadgeVariant(label)}
            className="capitalize"
            data-testid={`har-accuracy-tercile-${label}`}
          >
            {label}: {empty ? "—" : formatAccuracyPct(value)}
          </Badge>
        );
      })}
    </div>
  );
}

interface BacktestTableProps {
  rows: HarTercileBacktestRow[];
}

function BacktestTable({ rows }: BacktestTableProps) {
  return (
    <div
      className="overflow-x-auto rounded-md border border-border"
      data-testid="har-accuracy-table"
    >
      <table className="w-full text-sm">
        <thead className="bg-muted/40">
          <tr className="text-left text-[11px] uppercase tracking-wide text-muted-foreground">
            <th className="px-3 py-2 font-medium">Date</th>
            <th className="px-3 py-2 font-medium">Predicted</th>
            <th className="px-3 py-2 font-medium">Realized</th>
            <th className="px-3 py-2 font-medium">Hit</th>
            <th className="px-3 py-2 text-right font-medium">Realized vol</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => {
            const hit =
              row.correct === true
                ? "✓"
                : row.correct === false
                  ? "✗"
                  : "—";
            const hitClass = cn(
              "numeric text-sm font-semibold",
              row.correct === true && "text-dovish",
              row.correct === false && "text-hawkish",
              row.correct == null && "text-muted-foreground",
            );
            return (
              <tr
                key={`${row.event_date}-${row.predicted_tercile}`}
                className="border-t border-border/60"
                data-testid={`har-accuracy-row-${row.event_date}`}
              >
                <td className="numeric whitespace-nowrap px-3 py-2 tabular-nums">
                  {row.event_date}
                </td>
                <td className="px-3 py-2">
                  <Badge
                    variant={tercileBadgeVariant(row.predicted_tercile)}
                    className="capitalize"
                    data-testid={`har-accuracy-row-pred-${row.event_date}`}
                  >
                    {row.predicted_tercile}
                  </Badge>
                  <span className="ml-1.5 numeric text-[11px] text-muted-foreground tabular-nums">
                    {formatProbPct(row.predicted_prob)}
                  </span>
                </td>
                <td className="px-3 py-2">
                  {row.realized_tercile ? (
                    <Badge
                      variant={tercileBadgeVariant(row.realized_tercile)}
                      className="capitalize"
                      data-testid={`har-accuracy-row-real-${row.event_date}`}
                    >
                      {row.realized_tercile}
                    </Badge>
                  ) : (
                    <span
                      className="text-[11px] text-muted-foreground"
                      data-testid={`har-accuracy-row-real-${row.event_date}-pending`}
                    >
                      pending
                    </span>
                  )}
                </td>
                <td className="px-3 py-2">
                  <span
                    className={hitClass}
                    data-testid={`har-accuracy-row-hit-${row.event_date}`}
                    aria-label={
                      row.correct === true
                        ? "Hit"
                        : row.correct === false
                          ? "Miss"
                          : "Pending"
                    }
                  >
                    {hit}
                  </span>
                </td>
                <td className="numeric px-3 py-2 text-right tabular-nums text-muted-foreground">
                  {annualizedVolPct(row.realized_rv)}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export interface HarAccuracyPanelProps {
  data: HarTercileBacktestResponse | null;
  loading?: boolean;
  error?: string | null;
  symbol?: string;
}

export function HarAccuracyPanel({
  data,
  loading = false,
  error = null,
  symbol,
}: HarAccuracyPanelProps) {
  if (loading) {
    return (
      <WorkspaceSection
        title="HAR-tercile accuracy"
        description="Backtest of the last persisted predictions vs realized forward vol"
        variant="forecast"
      >
        <p
          className="text-xs text-muted-foreground"
          data-testid="har-accuracy-loading"
        >
          Loading HAR-tercile backtest…
        </p>
      </WorkspaceSection>
    );
  }

  if (error || !data) {
    return (
      <WorkspaceSection
        title="HAR-tercile accuracy"
        description="Backtest of the last persisted predictions vs realized forward vol"
        variant="forecast"
      >
        <p
          className="text-xs text-muted-foreground"
          data-testid="har-accuracy-unavailable"
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
        title="HAR-tercile accuracy"
        description="Backtest of the last persisted predictions vs realized forward vol"
        variant="forecast"
      >
        <div className="space-y-2">
          <div className="flex items-center justify-end">
            {symbolBadge ? (
              <Badge
                variant="outline"
                className="numeric text-[10px] tabular-nums"
                data-testid="har-accuracy-symbol"
              >
                {symbolBadge}
              </Badge>
            ) : null}
          </div>
          <p
            className="text-xs text-muted-foreground"
            data-testid="har-accuracy-empty"
          >
            No resolved FOMC runs for ^GSPC yet
          </p>
        </div>
      </WorkspaceSection>
    );
  }

  return (
    <WorkspaceSection
      title="HAR-tercile accuracy"
      description="Backtest of the last persisted predictions vs realized forward vol"
      variant="forecast"
    >
      <div className="space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <p className="text-[11px] text-muted-foreground">
            Resolved tercile is bucketed off the forward 10 trading days
            of realized vol against the same cutoffs the prediction used.
          </p>
          {symbolBadge ? (
            <Badge
              variant="outline"
              className="numeric text-[10px] tabular-nums"
              data-testid="har-accuracy-symbol"
            >
              {symbolBadge}
            </Badge>
          ) : null}
        </div>
        <KpiHeader metrics={data.metrics} />
        <PerTercileChips metrics={data.metrics} />
        <BacktestTable rows={data.rows} />
      </div>
    </WorkspaceSection>
  );
}

export default HarAccuracyPanel;
