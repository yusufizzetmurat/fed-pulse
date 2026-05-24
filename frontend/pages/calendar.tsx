import * as React from "react";
import Head from "next/head";
import { useRouter } from "next/router";
import { Calendar as CalendarIcon } from "lucide-react";
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
import { fetchFomcCalendar, resolveApiBaseUrl } from "@/lib/analyze/api";
import type { FomcCalendarResponse, FomcMeeting } from "@/lib/analyze/types";

function MeetingRow({ meeting, onAnalyze }: { meeting: FomcMeeting; onAnalyze: (date: string) => void }) {
  return (
    <li className="flex flex-wrap items-center justify-between gap-3 border-b border-border py-3 last:border-0">
      <div className="space-y-0.5">
        <p className="font-mono text-sm">{meeting.meeting_date}</p>
        <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
          {meeting.statement_release_date ? (
            <span>Statement {meeting.statement_release_date}</span>
          ) : null}
          {meeting.minutes_release_date ? (
            <span>Minutes {meeting.minutes_release_date}</span>
          ) : null}
        </div>
      </div>
      <div className="flex items-center gap-2">
        <Badge variant="outline" className="capitalize">{meeting.meeting_type}</Badge>
        <Button size="sm" variant="outline" onClick={() => onAnalyze(meeting.statement_release_date ?? meeting.meeting_date)}>
          Analyze
        </Button>
      </div>
    </li>
  );
}

export default function CalendarPage() {
  const router = useRouter();
  const apiBaseUrl = React.useMemo(() => resolveApiBaseUrl(), []);
  const [data, setData] = React.useState<FomcCalendarResponse | null>(null);
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
    return () => {
      cancelled = true;
    };
  }, [apiBaseUrl]);

  const goAnalyze = (meetingDate: string) => {
    // ?kind=statement asks the analyze page to fetch the matching FOMC
    // statement text from /documents/by-date and prefill the textarea.
    // Without it, only the date prefills and the user has to paste the
    // text themselves.
    router.push({
      pathname: "/analyze",
      query: { date: meetingDate, kind: "statement" },
    });
  };

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
            <h1 className="text-3xl font-semibold tracking-tight flex items-center gap-2">
              <CalendarIcon className="h-7 w-7 text-primary" />
              FOMC calendar
            </h1>
            <p className="max-w-2xl text-muted-foreground">
              Scheduled FOMC meetings sourced from the Federal Reserve's published calendar.
            </p>
          </div>

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
