import * as React from "react";
import { Calendar, CircleDot, Cpu, GitBranch } from "lucide-react";

import { useSharedCalendar } from "@/lib/analyze/shared-context";
import { friendlyEncoderName } from "@/lib/analyze/encoders";
import { cn } from "@/lib/utils";
import type { AnalyzeResult } from "@/lib/analyze/types";

interface StatusBarProps {
  result?: AnalyzeResult | null;
  loading?: boolean;
  symbol?: string;
  documentDate?: string;
  className?: string;
}

interface UpcomingMeeting {
  date: string;
  daysUntil: number;
}

function computeDaysUntil(dateIso: string): number {
  const target = new Date(`${dateIso}T00:00:00Z`).getTime();
  if (Number.isNaN(target)) return Number.NaN;
  const ms = target - Date.now();
  return Math.ceil(ms / 86_400_000);
}

export function StatusBar({
  result,
  loading,
  symbol,
  documentDate,
  className,
}: StatusBarProps) {
  const calendar = useSharedCalendar();
  const upcoming = React.useMemo<UpcomingMeeting | null>(() => {
    const next = calendar.data?.upcoming?.[0];
    if (!next) return null;
    const days = computeDaysUntil(next.meeting_date);
    if (Number.isNaN(days)) return null;
    return { date: next.meeting_date, daysUntil: days };
  }, [calendar.data]);

  const encoderKey = result?.model?.encoder_key ?? null;
  const checkpointLoaded = result?.model?.checkpoint_loaded;
  const runtimeMode = result?.model?.runtime_mode;

  // Tri-state: loading (running) > result present (checkpoint loaded /
  // no checkpoint based on diagnostics) > no result yet (awaiting).
  // Before this branch the third case rendered as "no checkpoint",
  // which read like an error on initial page load.
  const hasResult = Boolean(result);
  let stateLabel: string;
  let stateTone: string;
  if (loading) {
    stateLabel = "running";
    stateTone = "animate-pulse text-hawkish";
  } else if (!hasResult) {
    stateLabel = "awaiting analysis";
    stateTone = "text-muted-foreground";
  } else if (checkpointLoaded) {
    stateLabel = "checkpoint loaded";
    stateTone = "text-up";
  } else {
    stateLabel = "no checkpoint";
    stateTone = "text-down";
  }

  return (
    <div
      role="status"
      aria-live="polite"
      className={cn(
        "statusbar-surface flex flex-wrap items-center gap-x-4 gap-y-1 border-b border-border px-4 py-2 text-xs text-muted-foreground",
        className,
      )}
    >
      <div className="flex items-center gap-1.5">
        <CircleDot
          className={cn("h-2 w-2 fill-current", stateTone)}
          aria-hidden="true"
        />
        <span>{stateLabel}</span>
      </div>
      {symbol ? (
        <div className="flex items-center gap-1.5">
          <span>symbol</span>
          <span className="numeric text-foreground">{symbol}</span>
        </div>
      ) : null}
      {documentDate ? (
        <div className="flex items-center gap-1.5">
          <span>as-of</span>
          <span className="numeric text-foreground">{documentDate}</span>
        </div>
      ) : null}
      {encoderKey ? (
        <div className="flex items-center gap-1.5" title={encoderKey}>
          <Cpu className="h-3 w-3" aria-hidden="true" />
          <span className="numeric text-foreground">{friendlyEncoderName(encoderKey)}</span>
        </div>
      ) : null}
      {runtimeMode ? (
        <div className="flex items-center gap-1.5">
          <GitBranch className="h-3 w-3" aria-hidden="true" />
          <span className="numeric text-foreground">{runtimeMode}</span>
        </div>
      ) : null}
      {upcoming ? (
        <div className="ml-auto flex items-center gap-1.5">
          <Calendar className="h-3 w-3" aria-hidden="true" />
          <span>next FOMC</span>
          <span className="numeric text-foreground">{upcoming.date}</span>
          <span>
            ·{" "}
            {upcoming.daysUntil > 0
              ? `in ${upcoming.daysUntil}d`
              : upcoming.daysUntil === 0
              ? "today"
              : `${Math.abs(upcoming.daysUntil)}d ago`}
          </span>
        </div>
      ) : null}
    </div>
  );
}
