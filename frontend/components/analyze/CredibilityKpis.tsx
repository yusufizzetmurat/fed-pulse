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
  const driftReady = Math.abs(drift) > 1e-6 || (credibility.drift_trend?.length ?? 0) > 1;
  const allFlat =
    !driftReady &&
    realizedGap == null &&
    !marketGapReady &&
    monthsSince == null;

  if (allFlat) {
    return (
      <div className="rounded-md border border-dashed border-border bg-muted/20 p-4 text-xs text-muted-foreground">
        <p className="mb-1 text-[10px] uppercase tracking-wide text-foreground">
          Credibility features unpopulated
        </p>
        <p>
          The credibility module ran but every signal is at its placeholder value. Needs the
          prior four FOMC statements + the DFF history under <code className="rounded bg-muted px-1">data/external/fred/</code>
          {" "}to compute drift, realized-vs-stated gap, and months-since-reversal. The
          market-implied gap stays N/A until the SEP / OIS curve scrapers ship.
        </p>
      </div>
    );
  }

  return (
    <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
      <KpiTile
        label="Drift score"
        icon={<GitBranch className="h-3.5 w-3.5" />}
        value={<span className="numeric">{drift.toFixed(2)}</span>}
        sparkline={credibility.drift_trend}
        tone={drift > 0.6 ? "warn" : drift < 0.3 ? "neutral" : "neutral"}
        caption={
          drift > 0.6
            ? "Diverging from prior 4 statements"
            : drift < 0.3
            ? "Stable vs prior 4 statements"
            : "Mild drift vs prior 4"
        }
      />
      <KpiTile
        label="Realized vs stated"
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
        caption="90-day Pearson · stance vs DFF moves"
      />
      <Tooltip>
        <TooltipTrigger asChild>
          <div>
            <KpiTile
              label="Market-implied gap"
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
                  ? "stance vs OIS-implied path"
                  : "Pending SEP / OIS curve ingest"
              }
            />
          </div>
        </TooltipTrigger>
        <TooltipContent side="top" className="max-w-xs text-[11px]">
          Populated once the SEP dot-plot and Eurodollar / OIS curve scrapers ship. Today the backend
          returns 0.0 as a placeholder; the UI shows N/A rather than implying confidence.
        </TooltipContent>
      </Tooltip>
      <KpiTile
        label="Since reversal"
        icon={<Timer className="h-3.5 w-3.5" />}
        value={
          monthsSince == null ? (
            <span className="text-muted-foreground">—</span>
          ) : (
            <span className="numeric">{monthsSince} mo</span>
          )
        }
        caption="Months since last stance sign flip"
      />
    </div>
  );
}
