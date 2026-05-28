import * as React from "react";
import Head from "next/head";
import { useRouter } from "next/router";
import { Calendar as CalendarIcon, Clock } from "lucide-react";
import { toast } from "sonner";

import { Header } from "@/components/shell/header";
import { StatusBar } from "@/components/shell/status-bar";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import {
  fetchFomcCalendar,
  fetchNextFomcForecast,
  resolveApiBaseUrl,
} from "@/lib/analyze/api";
import type {
  FomcCalendarResponse,
  FomcMeeting,
  NextFomcForecastResponse,
} from "@/lib/analyze/types";

const ORDINAL_LABELS: Record<string, string> = {
  cut_50: "Cut 50",
  cut_25: "Cut 25",
  hold: "Hold",
  hike_25: "Hike 25",
  hike_50: "Hike 50",
  hike_75: "Hike 75",
};

function ordinalSuffix(n: number): string {
  const v = n % 100;
  if (v >= 11 && v <= 13) return `${n}th`;
  const last = n % 10;
  if (last === 1) return `${n}st`;
  if (last === 2) return `${n}nd`;
  if (last === 3) return `${n}rd`;
  return `${n}th`;
}

function formatCountdown(targetIso: string, nowMs: number): string {
  // Parse the target as UTC midnight so viewers in every timezone see the
  // same countdown to the same meeting. The displayed value is still a
  // relative duration ("in 3d 4h"), which is timezone-agnostic.
  const target = new Date(`${targetIso}T00:00:00Z`).getTime();
  if (!Number.isFinite(target)) return "—";
  const diffMs = target - nowMs;
  if (diffMs <= 0) return "today";
  const totalMinutes = Math.floor(diffMs / 60000);
  const days = Math.floor(totalMinutes / (60 * 24));
  const hours = Math.floor((totalMinutes % (60 * 24)) / 60);
  if (days >= 1) {
    return `${days} day${days === 1 ? "" : "s"}, ${hours} hour${hours === 1 ? "" : "s"}`;
  }
  if (hours >= 1) {
    return `${hours} hour${hours === 1 ? "" : "s"}`;
  }
  return "less than 1 hour";
}

function CountdownCard({ targetDate }: { targetDate: string }) {
  const [now, setNow] = React.useState(() => Date.now());
  React.useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), 60_000);
    return () => clearInterval(id);
  }, []);
  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center gap-2 text-base">
          <Clock className="h-4 w-4 text-primary" />
          FOMC meeting in {formatCountdown(targetDate, now)}
        </CardTitle>
        <CardDescription>
          Next scheduled meeting: <span className="font-mono">{targetDate}</span>
        </CardDescription>
      </CardHeader>
    </Card>
  );
}

interface MeetingRowProps {
  meeting: FomcMeeting;
  onAnalyze: (date: string) => void;
  predictedAction?: string | null;
  contextLabel?: string | null;
}

function MeetingRow({ meeting, onAnalyze, predictedAction, contextLabel }: MeetingRowProps) {
  const targetDate = meeting.statement_release_date ?? meeting.meeting_date;
  return (
    <li className="border-b border-border last:border-0">
      <button
        type="button"
        onClick={() => onAnalyze(targetDate)}
        className="flex w-full flex-wrap items-center justify-between gap-3 py-3 text-left hover:bg-accent/40 focus:outline-none focus-visible:ring-2 focus-visible:ring-ring rounded-sm px-2"
      >
        <div className="space-y-0.5">
          <p className="font-mono text-sm">{meeting.meeting_date}</p>
          <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
            {meeting.statement_release_date ? (
              <span>Statement {meeting.statement_release_date}</span>
            ) : null}
            {meeting.minutes_release_date ? (
              <span>Minutes {meeting.minutes_release_date}</span>
            ) : null}
            {contextLabel ? <span>· {contextLabel}</span> : null}
          </div>
        </div>
        <div className="flex items-center gap-2">
          {predictedAction ? (
            <Badge variant="outline" className="text-[10px]">
              forecast · {predictedAction}
            </Badge>
          ) : null}
          <Badge variant="outline" className="capitalize">
            {meeting.meeting_type}
          </Badge>
          <span className="text-xs text-muted-foreground">Analyze →</span>
        </div>
      </button>
    </li>
  );
}

// Compute historical context labels for the past meetings list (newest-first).
// Iterates oldest → newest accumulating streaks, then maps back so each
// row gets the label that describes "as of that meeting".
interface CalendarHistoricalContext {
  labels: Record<string, string>;
}

function computeHistoricalContext(past: FomcMeeting[], forecastHistory: NextFomcForecastResponse["history"]): CalendarHistoricalContext {
  const labels: Record<string, string> = {};
  if (!forecastHistory || forecastHistory.length === 0) return { labels };
  // Build date → realised action lookup from forecast history.
  const byDate = new Map<string, string | null>();
  for (const entry of forecastHistory) {
    if (entry.target_event_date) {
      byDate.set(entry.target_event_date, entry.target_class ?? null);
    }
  }
  // Walk past meetings oldest-first.
  const ordered = [...past].sort((a, b) =>
    a.meeting_date.localeCompare(b.meeting_date),
  );
  let prevAction: string | null = null;
  let streak = 0;
  let sinceLastCut = 0;
  for (const meeting of ordered) {
    const action = byDate.get(meeting.meeting_date) ?? null;
    if (!action) {
      streak = 0;
      sinceLastCut += 1;
      continue;
    }
    if (action === prevAction) {
      streak += 1;
    } else {
      streak = 1;
    }
    if (action.startsWith("cut")) {
      sinceLastCut = 0;
    } else {
      sinceLastCut += 1;
    }
    if (streak >= 2) {
      const word = action.startsWith("hike")
        ? "hike"
        : action.startsWith("cut")
        ? "cut"
        : "hold";
      labels[meeting.meeting_date] = `${ordinalSuffix(streak)} consecutive ${word}`;
    } else if (action.startsWith("hike") && sinceLastCut > 0) {
      labels[meeting.meeting_date] = `${sinceLastCut} meeting${sinceLastCut === 1 ? "" : "s"} since last cut`;
    }
    prevAction = action;
  }
  return { labels };
}

export default function CalendarPage() {
  const router = useRouter();
  const apiBaseUrl = React.useMemo(() => resolveApiBaseUrl(), []);
  const [data, setData] = React.useState<FomcCalendarResponse | null>(null);
  const [decisions, setDecisions] = React.useState<NextFomcForecastResponse | null>(null);
  const [loading, setLoading] = React.useState(true);

  React.useEffect(() => {
    let cancelled = false;
    fetchFomcCalendar(apiBaseUrl)
      .then((result) => {
        if (!cancelled) setData(result);
      })
      .catch((err) => {
        if (!cancelled) toast.error((err as Error).message || "Calendar fetch failed");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    // Best-effort decisions read — surfaces predicted action badges on
    // upcoming meetings. Silent failure when the artifact is absent.
    fetchNextFomcForecast(apiBaseUrl)
      .then((result) => {
        if (!cancelled) setDecisions(result);
      })
      .catch(() => {
        // Silent.
      });
    return () => {
      cancelled = true;
    };
  }, [apiBaseUrl]);

  const goAnalyze = (meetingDate: string) => {
    router.push({
      pathname: "/",
      query: { date: meetingDate, kind: "statement" },
    });
  };

  const nextMeeting = data?.upcoming?.[0] ?? null;
  const nextDate = nextMeeting?.statement_release_date ?? nextMeeting?.meeting_date ?? null;

  // Build a forecast lookup for upcoming meetings. The decisions endpoint
  // exposes one "headline" prediction for the next meeting; older
  // history entries cover past meetings.
  const upcomingForecastByDate = React.useMemo(() => {
    const map: Record<string, string> = {};
    if (!decisions?.headline) return map;
    const primaryModel = decisions.model_names?.[0];
    const predicted = primaryModel
      ? decisions.headline.predicted_class[primaryModel] ?? null
      : null;
    if (predicted && decisions.headline.target_event_date) {
      map[decisions.headline.target_event_date] = ORDINAL_LABELS[predicted] ?? predicted;
    }
    return map;
  }, [decisions]);

  const historicalContext = React.useMemo(
    () => computeHistoricalContext(data?.past ?? [], decisions?.history ?? []),
    [data, decisions],
  );

  return (
    <>
      <Head>
        <title>FOMC calendar — Fed Pulse</title>
      </Head>
      <div className="min-h-screen bg-background text-foreground">
        <Header />
        <StatusBar />
        <main id="main-content" tabIndex={-1} className="container space-y-6 py-8 focus:outline-none">
          <div className="space-y-2">
            <h1 className="flex items-center gap-2 text-2xl font-semibold tracking-tight sm:text-3xl">
              <CalendarIcon className="h-6 w-6 text-primary sm:h-7 sm:w-7" />
              FOMC calendar
            </h1>
            <p className="max-w-2xl text-muted-foreground">
              Scheduled FOMC meetings sourced from the Federal Reserve's published calendar.
              Click a past meeting to load its statement on the Workspace.
            </p>
          </div>

          {nextDate ? <CountdownCard targetDate={nextDate} /> : null}

          {loading ? (
            <div className="space-y-2">
              <Skeleton className="h-16 w-full" />
              <Skeleton className="h-16 w-full" />
              <Skeleton className="h-16 w-full" />
            </div>
          ) : data ? (
            <div className="grid gap-6 lg:grid-cols-2">
              <Card>
                <CardHeader>
                  <CardTitle>Upcoming</CardTitle>
                  <CardDescription>Next {data.upcoming.length} meetings</CardDescription>
                </CardHeader>
                <CardContent>
                  {data.upcoming.length === 0 ? (
                    <p className="text-muted-foreground">No upcoming meetings scheduled.</p>
                  ) : (
                    <ul>
                      {data.upcoming.map((meeting) => (
                        <MeetingRow
                          key={`up-${meeting.meeting_date}`}
                          meeting={meeting}
                          onAnalyze={goAnalyze}
                          predictedAction={upcomingForecastByDate[meeting.meeting_date] ?? null}
                        />
                      ))}
                    </ul>
                  )}
                </CardContent>
              </Card>
              <Card>
                <CardHeader>
                  <CardTitle>Past</CardTitle>
                  <CardDescription>Last {data.past.length} meetings</CardDescription>
                </CardHeader>
                <CardContent>
                  {data.past.length === 0 ? (
                    <p className="text-muted-foreground">No past meetings in window.</p>
                  ) : (
                    <ul>
                      {data.past.map((meeting) => (
                        <MeetingRow
                          key={`past-${meeting.meeting_date}`}
                          meeting={meeting}
                          onAnalyze={goAnalyze}
                          contextLabel={historicalContext.labels[meeting.meeting_date] ?? null}
                        />
                      ))}
                    </ul>
                  )}
                </CardContent>
              </Card>
            </div>
          ) : null}
        </main>
      </div>
    </>
  );
}
