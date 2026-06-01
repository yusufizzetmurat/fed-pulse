import * as React from "react";

import { Badge } from "@/components/ui/badge";
import { WorkspaceSection } from "@/components/analyze/WorkspaceSection";
import {
  formatLogResidual,
  formatPctVsBaseline,
} from "@/lib/analyze/formatters";
import { cn } from "@/lib/utils";
import type {
  ExpectedVolumeForecastResponse,
  ExpectedVolumeHorizonForecast,
} from "@/lib/analyze/types";

// Expected Volume forecast card — workspace-spine forecast variant.
// HAR Corsi head over the last ~120 daily volumes plus an optional
// calendar-seasonality block (Mon..Thu / month-end / quarter-end).
// Market-data only; text features never feed this surface.

const HORIZON_LABELS: Record<number, string> = {
  1: "1 day",
  5: "1 week",
  22: "1 month",
};

function formatBandPct(value: number): string {
  if (!Number.isFinite(value)) return "N/A";
  const sign = value >= 0 ? "+" : "";
  return `${sign}${value.toFixed(1)}%`;
}

function pctToneClass(pct: number | null | undefined): string {
  if (pct == null || !Number.isFinite(pct)) return "text-muted-foreground";
  if (pct > 0.5) return "text-hawkish";
  if (pct < -0.5) return "text-dovish";
  return "text-foreground";
}

interface HorizonColumnProps {
  horizon: ExpectedVolumeHorizonForecast;
}

function HorizonColumn({ horizon }: HorizonColumnProps) {
  const label = HORIZON_LABELS[horizon.h] ?? `${horizon.h}d`;
  const headlineTone = pctToneClass(horizon.point_pct_vs_baseline);
  const r2 =
    horizon.r2_har !== null && Number.isFinite(horizon.r2_har)
      ? `R² ${horizon.r2_har.toFixed(2)}`
      : "R² N/A";
  return (
    <div
      data-testid={`expected-volume-horizon-${horizon.h}`}
      className="space-y-2 rounded-lg border border-border bg-card/50 p-3"
    >
      <div className="flex items-center justify-between gap-2">
        <p className="text-xs uppercase tracking-wide text-muted-foreground">
          {label}
        </p>
        <div className="flex items-center gap-1.5">
          <Badge
            variant="outline"
            className="numeric text-[10px] tabular-nums"
            data-testid={`expected-volume-r2-${horizon.h}`}
          >
            {r2}
          </Badge>
          {horizon.calendar_adjusted ? (
            <Badge
              variant="outline"
              className="text-[10px]"
              data-testid={`expected-volume-cal-${horizon.h}`}
            >
              calendar-adjusted
            </Badge>
          ) : null}
        </div>
      </div>
      <p
        className={cn(
          "numeric text-2xl font-semibold tabular-nums",
          headlineTone,
        )}
        data-testid={`expected-volume-headline-${horizon.h}`}
      >
        {formatPctVsBaseline(horizon.point_pct_vs_baseline)}
      </p>
      <p
        className="numeric text-[11px] tabular-nums text-muted-foreground"
        data-testid={`expected-volume-subscript-${horizon.h}`}
      >
        {formatLogResidual(horizon.point_log_residual)} log-residual
      </p>
      <div className="space-y-1 pt-1 text-[11px] text-muted-foreground">
        <p>
          <span className="font-mono">80%</span>{" "}
          <span
            className="numeric tabular-nums"
            data-testid={`expected-volume-band80-${horizon.h}`}
          >
            {formatBandPct(horizon.band_lo_80)} – {formatBandPct(horizon.band_hi_80)}
          </span>
        </p>
        <p>
          <span className="font-mono">90%</span>{" "}
          <span
            className="numeric tabular-nums"
            data-testid={`expected-volume-band90-${horizon.h}`}
          >
            {formatBandPct(horizon.band_lo_90)} – {formatBandPct(horizon.band_hi_90)}
          </span>
        </p>
      </div>
    </div>
  );
}

export interface ExpectedVolumeCardProps {
  forecast: ExpectedVolumeForecastResponse | null;
  loading?: boolean;
  error?: string | null;
  symbol?: string;
  collapsible?: boolean;
  storageKey?: string;
}

export function ExpectedVolumeCard({
  forecast,
  loading = false,
  error = null,
  symbol,
  collapsible = false,
  storageKey,
}: ExpectedVolumeCardProps) {
  if (loading) {
    return (
      <WorkspaceSection
        title="Expected Volume"
        description="HAR-volume forecast over market history"
        variant="forecast"
        collapsible={collapsible}
        storageKey={storageKey}
      >
        <p
          className="text-xs text-muted-foreground"
          data-testid="expected-volume-loading"
        >
          Loading HAR-volume forecast…
        </p>
      </WorkspaceSection>
    );
  }

  if (error || !forecast || forecast.horizons.length === 0) {
    return (
      <WorkspaceSection
        title="Expected Volume"
        description="HAR-volume forecast over market history"
        variant="forecast"
        collapsible={collapsible}
        storageKey={storageKey}
      >
        <p
          className="text-xs text-muted-foreground"
          data-testid="expected-volume-unavailable"
        >
          {error ?? "HAR-volume artifact has not loaded yet. Retry shortly."}
        </p>
      </WorkspaceSection>
    );
  }

  // Render horizons in a fixed 1d / 5d / 22d order regardless of how
  // the backend listed them; missing horizons just drop out.
  const horizonsByH = new Map<number, ExpectedVolumeHorizonForecast>();
  for (const h of forecast.horizons) horizonsByH.set(h.h, h);
  const ordered = [1, 5, 22]
    .map((h) => horizonsByH.get(h))
    .filter((h): h is ExpectedVolumeHorizonForecast => h !== undefined);

  const symbolBadge = symbol ?? forecast.symbol;

  return (
    <WorkspaceSection
      title="Expected Volume"
      description="HAR-volume forecast over market history"
      variant="forecast"
      collapsible={collapsible}
      storageKey={storageKey}
    >
      <div className="space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <p className="text-[11px] text-muted-foreground">
            HAR Corsi head on daily log-volume; bands are conformal at 80% and
            90% nominal coverage. Market-data only.
          </p>
          {symbolBadge ? (
            <Badge
              variant="outline"
              className="numeric text-[10px] tabular-nums"
              data-testid="expected-volume-symbol"
            >
              {symbolBadge}
            </Badge>
          ) : null}
        </div>
        <div
          className="grid gap-3 md:grid-cols-3"
          data-testid="expected-volume-grid"
        >
          {ordered.map((horizon) => (
            <HorizonColumn key={horizon.h} horizon={horizon} />
          ))}
        </div>
      </div>
    </WorkspaceSection>
  );
}

export default ExpectedVolumeCard;
