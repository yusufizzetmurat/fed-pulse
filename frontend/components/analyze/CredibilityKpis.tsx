import * as React from "react";
import { Activity, GitBranch, Scale, Timer } from "lucide-react";

import { KpiTile } from "@/components/ui/kpi-tile";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import type { CredibilityResponse } from "@/lib/analyze/types";

interface CredibilityKpisProps {
  credibility: CredibilityResponse;
}

function isMarketGapAvailable(value: number | undefined | null): boolean {
  // The backend currently returns 0.0 as a placeholder until the SEP /
  // Eurodollar scraper lands. Treat exact zero as "not yet available"
  // rather than as a real reading.
  if (value == null) return false;
  return Math.abs(value) > 1e-6;
}

export function CredibilityKpis({ credibility }: CredibilityKpisProps) {
  const drift = credibility.drift_score;
  const realizedGap = credibility.realized_vs_stated_gap;
  const marketGap = credibility.market_implied_gap;
  const monthsSince = credibility.months_since_reversal;
  const marketGapReady = isMarketGapAvailable(marketGap);
  // Backend now reports ``null`` (not zero) when the drift score is
  // unavailable, so the magnitude check is no longer the right gate.
  // Treat any non-null reading as a real value and let the trend list
  // (more than one point) keep gating the chart panel.
  const driftReady = drift != null && (credibility.drift_trend?.length ?? 0) > 1;
  const allFlat =
    !driftReady &&
    realizedGap == null &&
    !marketGapReady &&
    monthsSince == null;

  if (allFlat) {
    return (
      <div className="rounded-md border border-dashed border-border bg-muted/20 p-4 text-xs text-muted-foreground">
        <p className="mb-1 text-[10px] uppercase tracking-wide text-foreground">
          Credibility signals not yet available
        </p>
        <p>
          The credibility module ran but every signal is at its default value. Computing
          the shift score, the gap between what was done vs. said, and time since the last
          reversal requires the previous four FOMC statements plus federal funds rate
          history. The gap to market expectations stays unavailable until the SEP and OIS
          data feeds are connected.
        </p>
      </div>
    );
  }

  return (
    <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
      <KpiTile
        label="Shift score"
        icon={<GitBranch className="h-3.5 w-3.5" />}
        value={
          drift == null ? (
            <span className="text-xs text-muted-foreground">not available</span>
          ) : (
            <span className="numeric">{drift.toFixed(2)}</span>
          )
        }
        sparkline={credibility.drift_trend}
        tone={drift != null && drift > 0.6 ? "warn" : "neutral"}
        caption={
          drift == null
            ? "Drift signal unavailable for this statement"
            : drift > 0.6
            ? "Diverging from the last 4 statements"
            : drift < 0.3
            ? "Stable vs the last 4 statements"
            : "Mild shift vs the last 4"
        }
      />
      <KpiTile
        label="What was done vs. said"
        icon={<Scale className="h-3.5 w-3.5" />}
        value={
          realizedGap == null ? (
            <span className="text-muted-foreground">N/A</span>
          ) : (
            <span className="numeric">
              {realizedGap >= 0 ? "+" : ""}
              {realizedGap.toFixed(2)}
            </span>
          )
        }
        tone={
          realizedGap == null
            ? "neutral"
            : realizedGap > 0.05
            ? "up"
            : realizedGap < -0.05
            ? "down"
            : "neutral"
        }
        caption="90-day correlation between stance and actual rate moves"
      />
      <Tooltip>
        <TooltipTrigger asChild>
          <div>
            <KpiTile
              label="Gap to market expectations"
              icon={<Activity className="h-3.5 w-3.5" />}
              value={
                marketGapReady ? (
                  <span className="numeric">
                    {(marketGap as number) >= 0 ? "+" : ""}
                    {(marketGap as number).toFixed(2)}
                  </span>
                ) : (
                  <span className="text-muted-foreground">N/A</span>
                )
              }
              caption={
                marketGapReady
                  ? "Stance vs the rate path priced into OIS swaps"
                  : "Waiting on SEP and OIS data feeds"
              }
            />
          </div>
        </TooltipTrigger>
        <TooltipContent side="top" className="max-w-xs text-[11px]">
          Fills in once the SEP dot-plot and OIS curve data feeds are connected. For now the
          backend returns a placeholder value, so the UI shows N/A rather than implying a reading.
        </TooltipContent>
      </Tooltip>
      <KpiTile
        label="Time since last reversal"
        icon={<Timer className="h-3.5 w-3.5" />}
        value={
          monthsSince == null ? (
            <span className="text-muted-foreground">—</span>
          ) : (
            <span className="numeric">{monthsSince} mo</span>
          )
        }
        caption="Months since the stance last flipped direction"
      />
    </div>
  );
}
