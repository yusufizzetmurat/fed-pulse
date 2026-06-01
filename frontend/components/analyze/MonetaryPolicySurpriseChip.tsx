import * as React from "react";
import { ArrowDownRight, ArrowUpRight, Info, Minus } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { WorkspaceSection } from "@/components/analyze/WorkspaceSection";
import { formatBps } from "@/lib/analyze/formatters";
import type {
  MonetaryPolicySurpriseDirection,
  MonetaryPolicySurpriseResponse,
} from "@/lib/analyze/types";

const DIRECTION_BADGE: Record<
  MonetaryPolicySurpriseDirection,
  { label: string; variant: "hawkish" | "dovish" | "neutral" }
> = {
  hawkish: { label: "Hawkish", variant: "hawkish" },
  dovish: { label: "Dovish", variant: "dovish" },
  no_surprise: { label: "No surprise", variant: "neutral" },
};

function DirectionIcon({ direction }: { direction: MonetaryPolicySurpriseDirection }) {
  if (direction === "hawkish") {
    return <ArrowUpRight className="h-3.5 w-3.5" aria-hidden="true" />;
  }
  if (direction === "dovish") {
    return <ArrowDownRight className="h-3.5 w-3.5" aria-hidden="true" />;
  }
  return <Minus className="h-3.5 w-3.5" aria-hidden="true" />;
}

function formatEventDate(iso: string): string {
  // The wire shape is a zero-padded ISO date string. Render in a more
  // human-readable form ("Apr 29, 2026") without pulling in a locale
  // library — the dashboard is en-US only.
  const parsed = new Date(`${iso}T00:00:00Z`);
  if (Number.isNaN(parsed.getTime())) return iso;
  return parsed.toLocaleDateString("en-US", {
    year: "numeric",
    month: "short",
    day: "2-digit",
    timeZone: "UTC",
  });
}

export interface MonetaryPolicySurpriseChipProps {
  data: MonetaryPolicySurpriseResponse | null;
  loading?: boolean;
}

// Methodology summary for the chip tooltip. Keep this self-contained —
// no wiki section citations leak into the frontend; the prose covers
// the OIS-proxy caveat and the basis-point sign convention so a reader
// hovering the icon gets the same context the backend module docs
// carry.
const METHODOLOGY_TOOLTIP = (
  <div className="space-y-2 text-[11px] leading-snug">
    <p>
      Strict-prior monetary-policy surprise: realized rate change minus the
      pre-event 1-month-ahead implied policy path. Treasury yields (DGS1MO /
      DGS3MO / DGS6MO / DGS1 / DGS2) proxy the fed-funds futures curve when
      CME settlement prices are unavailable.
    </p>
    <p>
      Sign convention: positive bps = hawkish surprise (policy tighter than
      priced); negative = dovish. Inside the +/-2.5 bps band the panel
      reports &quot;no surprise&quot; rather than amplifying noise from the
      daily-window proxy.
    </p>
    <p className="text-muted-foreground">
      Descriptive only — never feeds the forecast cards.
    </p>
  </div>
);

export function MonetaryPolicySurpriseChip({
  data,
  loading = false,
}: MonetaryPolicySurpriseChipProps) {
  if (loading) {
    return (
      <WorkspaceSection
        title="Monetary policy surprise"
        description="Latest FOMC rate-path surprise (descriptive)"
        variant="descriptive"
      >
        <p className="text-xs text-muted-foreground" data-testid="mp-surprise-loading">
          Loading latest FOMC surprise…
        </p>
      </WorkspaceSection>
    );
  }

  if (!data) {
    return (
      <WorkspaceSection
        title="Monetary policy surprise"
        description="Latest FOMC rate-path surprise (descriptive)"
        variant="descriptive"
      >
        <p
          className="text-xs text-muted-foreground"
          data-testid="mp-surprise-unavailable"
        >
          MP-surprise feed unavailable. The latest FOMC surprise will appear
          here once the FRED rate panel is rebuilt.
        </p>
      </WorkspaceSection>
    );
  }

  const directionMeta = DIRECTION_BADGE[data.direction];
  const meetingLabel = formatEventDate(data.event_date);
  // Render the magnitude without the explicit "+" — the directional
  // badge already carries the sign, and the magnitude is an absolute
  // number by construction.
  const magnitudeLabel = formatBps(data.magnitude_bps, { signed: false });

  return (
    <WorkspaceSection
      title="Monetary policy surprise"
      description="Latest FOMC rate-path surprise (descriptive)"
      variant="descriptive"
    >
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="space-y-2">
          <div className="flex flex-wrap items-center gap-2">
            <Badge
              variant={directionMeta.variant}
              className="inline-flex items-center gap-1 capitalize"
              data-testid="mp-surprise-direction"
            >
              <DirectionIcon direction={data.direction} />
              {directionMeta.label}
            </Badge>
            <span
              className="numeric text-lg font-semibold tabular-nums"
              data-testid="mp-surprise-magnitude"
            >
              {magnitudeLabel}
            </span>
            {data.is_intermeeting ? (
              <Badge variant="outline" className="text-[10px] uppercase tracking-wide">
                Intermeeting
              </Badge>
            ) : null}
          </div>
          <p className="text-xs text-muted-foreground">
            Meeting <span data-testid="mp-surprise-event-date">{meetingLabel}</span>
            {data.ff_target_prior_bps != null ? (
              <>
                {" · prior target "}
                <span className="numeric tabular-nums">
                  {formatBps(data.ff_target_prior_bps, { signed: false })}
                </span>
              </>
            ) : null}
          </p>
        </div>
        <TooltipProvider delayDuration={150}>
          <Tooltip>
            <TooltipTrigger asChild>
              <button
                type="button"
                aria-label="Methodology for the monetary-policy surprise"
                className="inline-flex h-7 w-7 items-center justify-center rounded-full border border-dashed border-border text-muted-foreground transition hover:bg-muted/40 hover:text-foreground"
                data-testid="mp-surprise-methodology-trigger"
              >
                <Info className="h-3.5 w-3.5" aria-hidden="true" />
              </button>
            </TooltipTrigger>
            <TooltipContent side="left" className="max-w-xs">
              {METHODOLOGY_TOOLTIP}
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
      </div>
    </WorkspaceSection>
  );
}

export default MonetaryPolicySurpriseChip;
