import * as React from "react";
import { Info } from "lucide-react";

import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { WorkspaceSection } from "@/components/analyze/WorkspaceSection";
import { formatBps, formatProbabilityPct } from "@/lib/analyze/formatters";
import type { FuturesConsensusResponse } from "@/lib/analyze/types";

// Descriptive workspace panel — fed-funds path consensus via the
// short-end DGS Treasury proxy. The panel renders three horizon
// columns (1m / 3m / 6m) with the implied rate, the change vs the
// current target band midpoint, and a stacked hike / cut / pause
// probability bar. Methodology footnote sits in fine print at the
// bottom so the panel never implies an OIS-clean expectation.

function formatMeetingDate(iso: string): string {
  const parsed = new Date(`${iso}T00:00:00Z`);
  if (Number.isNaN(parsed.getTime())) return iso;
  return parsed.toLocaleDateString("en-US", {
    year: "numeric",
    month: "short",
    day: "2-digit",
    timeZone: "UTC",
  });
}

const PROXY_TOOLTIP = (
  <div className="space-y-2 text-[11px] leading-snug">
    <p>
      Treasury proxy: the short-end constant-maturity series
      (DGS1MO / DGS3MO / DGS6MO) stand in for the fed-funds futures
      curve when CME settlement prices are not in scope. The level
      embeds a small term premium that is typically a few basis points
      at these tenors.
    </p>
    <p>
      Treat the panel as a direction indicator and a level proxy.
      Hike / cut / pause probabilities are derived from a normal CDF
      with a 25 bps threshold and a 12.5 bps sigma anchored at the
      current target band midpoint.
    </p>
    <p className="text-muted-foreground">
      Descriptive only — never feeds the forecast cards.
    </p>
  </div>
);

interface StackedProbabilityBarProps {
  hike: number;
  pause: number;
  cut: number;
}

function StackedProbabilityBar({
  hike,
  pause,
  cut,
}: StackedProbabilityBarProps) {
  // Clamp + renormalize defensively so a small backend rounding
  // miss does not push the segments past 100% width.
  const total = Math.max(hike + pause + cut, 1e-9);
  const widths = {
    cut: (cut / total) * 100,
    pause: (pause / total) * 100,
    hike: (hike / total) * 100,
  };
  return (
    <div
      role="img"
      aria-label={`Cut ${Math.round(widths.cut)}%, pause ${Math.round(
        widths.pause,
      )}%, hike ${Math.round(widths.hike)}%`}
      data-testid="futures-consensus-prob-bar"
      className="flex h-2 w-full overflow-hidden rounded-full border border-border/40 bg-muted"
    >
      <span
        data-testid="futures-consensus-prob-cut"
        className="block h-full bg-blue-500"
        style={{ width: `${widths.cut}%` }}
      />
      <span
        data-testid="futures-consensus-prob-pause"
        className="block h-full bg-muted-foreground/40"
        style={{ width: `${widths.pause}%` }}
      />
      <span
        data-testid="futures-consensus-prob-hike"
        className="block h-full bg-amber-500"
        style={{ width: `${widths.hike}%` }}
      />
    </div>
  );
}

export interface FuturesConsensusPanelProps {
  data: FuturesConsensusResponse | null;
  loading?: boolean;
  collapsible?: boolean;
  storageKey?: string;
}

export function FuturesConsensusPanel({
  data,
  loading = false,
  collapsible = false,
  storageKey,
}: FuturesConsensusPanelProps) {
  if (loading) {
    return (
      <WorkspaceSection
        title="FRED futures consensus"
        description="Fed-funds path via Treasury proxy (descriptive)"
        variant="descriptive"
        collapsible={collapsible}
        storageKey={storageKey}
      >
        <p
          className="text-xs text-muted-foreground"
          data-testid="futures-consensus-loading"
        >
          Loading short-end DGS proxy from FRED…
        </p>
      </WorkspaceSection>
    );
  }

  if (!data) {
    return (
      <WorkspaceSection
        title="FRED futures consensus"
        description="Fed-funds path via Treasury proxy (descriptive)"
        variant="descriptive"
        collapsible={collapsible}
        storageKey={storageKey}
      >
        <p
          className="text-xs text-muted-foreground"
          data-testid="futures-consensus-unavailable"
        >
          Futures consensus feed unavailable. The panel will appear once the
          FRED short-end series come back online.
        </p>
      </WorkspaceSection>
    );
  }

  const meetingLabel = formatMeetingDate(data.meeting_date);
  const targetMidBps =
    0.5 * (data.current_target_lo_bps + data.current_target_hi_bps);

  return (
    <WorkspaceSection
      title="FRED futures consensus"
      description="Fed-funds path via Treasury proxy (descriptive)"
      variant="descriptive"
      collapsible={collapsible}
      storageKey={storageKey}
    >
      <div className="space-y-3">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div className="space-y-1">
            <p className="text-xs uppercase tracking-wide text-muted-foreground">
              Next meeting
            </p>
            <p
              className="text-sm font-medium"
              data-testid="futures-consensus-meeting-date"
            >
              {meetingLabel}
            </p>
            <p className="text-[11px] text-muted-foreground">
              Current target{" "}
              <span className="numeric tabular-nums">
                {formatBps(data.current_target_lo_bps, { signed: false })}
              </span>
              {" – "}
              <span className="numeric tabular-nums">
                {formatBps(data.current_target_hi_bps, { signed: false })}
              </span>
              {" · midpoint "}
              <span
                className="numeric tabular-nums"
                data-testid="futures-consensus-midpoint"
              >
                {formatBps(targetMidBps, { signed: false })}
              </span>
            </p>
          </div>
          <TooltipProvider delayDuration={150}>
            <Tooltip>
              <TooltipTrigger asChild>
                <button
                  type="button"
                  aria-label="Treasury proxy methodology"
                  className="inline-flex h-7 items-center gap-1 rounded-full border border-dashed border-border px-2 text-[11px] font-medium text-muted-foreground transition hover:bg-muted/40 hover:text-foreground"
                  data-testid="futures-consensus-proxy-trigger"
                >
                  <Info className="h-3.5 w-3.5" aria-hidden="true" />
                  Treasury proxy
                </button>
              </TooltipTrigger>
              <TooltipContent side="left" className="max-w-xs">
                {PROXY_TOOLTIP}
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </div>
        <div
          className="grid grid-cols-1 gap-3 sm:grid-cols-3"
          data-testid="futures-consensus-grid"
        >
          {data.horizons.map((horizon) => (
            <div
              key={horizon.horizon_label}
              data-testid={`futures-consensus-horizon-${horizon.horizon_label}`}
              className="rounded-lg border border-dashed border-border/60 bg-background/40 p-3"
            >
              <p className="text-[11px] uppercase tracking-wide text-muted-foreground">
                {horizon.horizon_label}
              </p>
              <p
                className="numeric mt-1 text-base font-semibold tabular-nums"
                data-testid={`futures-consensus-implied-${horizon.horizon_label}`}
              >
                {formatBps(horizon.implied_rate_bps, { signed: false })}
              </p>
              <p
                className="numeric text-[11px] tabular-nums text-muted-foreground"
                data-testid={`futures-consensus-change-${horizon.horizon_label}`}
              >
                {formatBps(horizon.change_vs_current_bps)} vs current
              </p>
              <div className="mt-2 space-y-1">
                <StackedProbabilityBar
                  hike={horizon.probability_hike}
                  pause={horizon.probability_pause}
                  cut={horizon.probability_cut}
                />
                <div className="flex justify-between text-[10px] uppercase tracking-wide text-muted-foreground">
                  <span data-testid={`futures-consensus-pcut-${horizon.horizon_label}`}>
                    Cut {formatProbabilityPct(horizon.probability_cut)}
                  </span>
                  <span data-testid={`futures-consensus-ppause-${horizon.horizon_label}`}>
                    Pause {formatProbabilityPct(horizon.probability_pause)}
                  </span>
                  <span data-testid={`futures-consensus-phike-${horizon.horizon_label}`}>
                    Hike {formatProbabilityPct(horizon.probability_hike)}
                  </span>
                </div>
              </div>
            </div>
          ))}
        </div>
        <p
          className="text-[10px] leading-snug text-muted-foreground"
          data-testid="futures-consensus-methodology"
        >
          {data.methodology} Source: {data.data_source}.
        </p>
      </div>
    </WorkspaceSection>
  );
}

export default FuturesConsensusPanel;
