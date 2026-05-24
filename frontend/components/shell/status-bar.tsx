import * as React from "react";
import { Calendar, CircleDot, Cpu, GitBranch } from "lucide-react";

import { fetchFomcCalendar, resolveApiBaseUrl } from "@/lib/analyze/api";
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
  const apiBaseUrl = React.useMemo(() => resolveApiBaseUrl(), []);
  const [upcoming, setUpcoming] = React.useState<UpcomingMeeting | null>(null);

  React.useEffect(() => {
    const controller = new AbortController();
    fetchFomcCalendar(apiBaseUrl, { upcoming_limit: 1, past_limit: 0 }, controller.signal)
      .then((response) => {
        if (controller.signal.aborted) return;
        const next = response.upcoming?.[0];
        if (!next) return;
        const days = computeDaysUntil(next.meeting_date);
        if (Number.isNaN(days)) return;
        setUpcoming({ date: next.meeting_date, daysUntil: days });
      })
      .catch(() => {
        // Calendar is best-effort; the next-FOMC field just stays hidden.
      });
    return () => {
      controller.abort();
    };
  }, [apiBaseUrl]);

  const encoderKey = result?.model?.encoder_key ?? null;
  const checkpointLoaded = result?.model?.checkpoint_loaded;
  const runtimeMode = result?.model?.runtime_mode;

  return (
    <div
      role="status"
      aria-live="polite"
      className={cn(
        "statusbar-surface flex flex-wrap items-center gap-x-4 gap-y-1 border-b border-border px-4 py-1.5 text-[11px] text-muted-foreground",
        className,
      )}
    >
      <div className="flex items-center gap-1.5">
        <CircleDot
          className={cn(
            "h-2 w-2 fill-current",
            loading
              ? "animate-pulse text-hawkish"
              : checkpointLoaded
              ? "text-up"
              : "text-muted-foreground",
          )}
          aria-hidden="true"
        />
        <span>
          {loading
            ? "running"
            : checkpointLoaded
            ? "checkpoint loaded"
            : "no checkpoint"}
        </span>
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
        <div className="flex items-center gap-1.5">
          <Cpu className="h-3 w-3" aria-hidden="true" />
          <span className="numeric text-foreground">{encoderKey}</span>
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
