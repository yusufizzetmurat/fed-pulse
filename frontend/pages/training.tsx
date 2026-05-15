import * as React from "react";
import Head from "next/head";
import Link from "next/link";
import { Cpu } from "lucide-react";
import { toast } from "sonner";

import { Header } from "@/components/shell/header";
import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { fetchTrainJobs, resolveApiBaseUrl } from "@/lib/analyze/api";
import type { TrainJobStatus, TrainJobSummary } from "@/lib/analyze/types";

const STATUS_VARIANT: Record<string, "hawkish" | "dovish" | "outline" | "secondary"> = {
  running: "hawkish",
  queued: "secondary",
  failed: "dovish",
  succeeded: "outline",
};

function StatusBadge({ status }: { status: string }) {
  const key = status.toLowerCase();
  const variant = STATUS_VARIANT[key] ?? "outline";
  return (
    <Badge variant={variant} className="capitalize">
      {status}
    </Badge>
  );
}

function StatusCount({
  label,
  status,
  jobs,
}: {
  label: string;
  status: TrainJobStatus;
  jobs: TrainJobSummary[];
}) {
  const count = jobs.filter((j) => j.status === status).length;
  return (
    <Card>
      <CardHeader className="space-y-0 pb-2">
        <CardTitle className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
          {label}
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="font-mono text-2xl font-semibold">{count}</div>
      </CardContent>
    </Card>
  );
}

function durationSeconds(started: string | null, finished: string | null): string {
  if (!started) return "—";
  const start = Date.parse(started);
  const end = finished ? Date.parse(finished) : Date.now();
  if (Number.isNaN(start) || Number.isNaN(end)) return "—";
  const seconds = Math.max(0, Math.round((end - start) / 1000));
  if (seconds < 60) return `${seconds}s`;
  const minutes = Math.floor(seconds / 60);
  const rest = seconds % 60;
  return `${minutes}m ${rest}s`;
}

export default function TrainingPage() {
  const apiBaseUrl = React.useMemo(() => resolveApiBaseUrl(), []);
  const [jobs, setJobs] = React.useState<TrainJobSummary[]>([]);
  const [loading, setLoading] = React.useState(true);

  React.useEffect(() => {
    let cancelled = false;
    const load = () => {
      fetchTrainJobs(apiBaseUrl)
        .then((result) => {
          if (!cancelled) setJobs(result.items);
        })
        .catch((err) => {
          if (!cancelled) toast.error((err as Error).message || "Failed to load training jobs.");
        })
        .finally(() => {
          if (!cancelled) setLoading(false);
        });
    };
    load();
    const id = setInterval(load, 5000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [apiBaseUrl]);

  return (
    <>
      <Head>
        <title>Training — Fed Pulse</title>
      </Head>
      <div className="min-h-screen bg-background text-foreground">
        <Header />
        <main className="container space-y-6 py-8">
          <div className="space-y-2">
            <h1 className="flex items-center gap-2 text-3xl font-semibold tracking-tight">
              <Cpu className="h-7 w-7 text-primary" />
              Training
            </h1>
            <p className="max-w-2xl text-muted-foreground">
              In-process Real Train jobs. Observational view — durable training runs flow through
              <code className="ml-1 rounded bg-muted px-1 py-0.5 font-mono text-xs">make train-batch</code> /
              <code className="ml-1 rounded bg-muted px-1 py-0.5 font-mono text-xs">make next-fomc</code>.
            </p>
          </div>

          <div className="grid gap-3 md:grid-cols-4">
            <StatusCount label="Running" status="running" jobs={jobs} />
            <StatusCount label="Queued" status="queued" jobs={jobs} />
            <StatusCount label="Succeeded" status="succeeded" jobs={jobs} />
            <StatusCount label="Failed" status="failed" jobs={jobs} />
          </div>

          {loading ? (
            <div className="space-y-2">
              <Skeleton className="h-12 w-full" />
              <Skeleton className="h-12 w-full" />
              <Skeleton className="h-12 w-full" />
            </div>
          ) : jobs.length === 0 ? (
            <Card>
              <CardContent className="py-10 text-center text-muted-foreground">
                No training jobs in this backend instance. Submit a Real Train forecast from the
                <Link href="/analyze" className="ml-1 underline">
                  Analyze page
                </Link>
                {" "}to enqueue one.
              </CardContent>
            </Card>
          ) : (
            <Card>
              <CardHeader>
                <CardTitle>Job queue</CardTitle>
                <CardDescription>Ordered by status (running first), then newest.</CardDescription>
              </CardHeader>
              <CardContent className="p-0">
                <table className="w-full text-sm">
                  <thead className="border-b border-border bg-muted/30 text-xs uppercase tracking-wide text-muted-foreground">
                    <tr>
                      <th className="px-4 py-2 text-left">Job</th>
                      <th className="px-4 py-2 text-left">Status</th>
                      <th className="px-4 py-2 text-left">Symbol</th>
                      <th className="px-4 py-2 text-left">Document date</th>
                      <th className="px-4 py-2 text-right">History len</th>
                      <th className="px-4 py-2 text-right">Duration</th>
                      <th className="px-4 py-2 text-left">Created</th>
                    </tr>
                  </thead>
                  <tbody>
                    {jobs.map((job) => (
                      <tr key={job.job_id} className="border-b border-border last:border-0 hover:bg-muted/40">
                        <td className="px-4 py-2 font-mono text-xs">
                          <Link href={`/training/${job.job_id}`} className="hover:underline">
                            {job.job_id.slice(0, 12)}…
                          </Link>
                        </td>
                        <td className="px-4 py-2">
                          <StatusBadge status={job.status} />
                        </td>
                        <td className="px-4 py-2 font-medium">{job.symbol ?? "—"}</td>
                        <td className="px-4 py-2 font-mono text-xs text-muted-foreground">
                          {job.date ?? "—"}
                        </td>
                        <td className="px-4 py-2 text-right font-mono text-muted-foreground">
                          {job.history_length ?? "—"}
                        </td>
                        <td className="px-4 py-2 text-right font-mono">
                          {durationSeconds(job.started_at, job.finished_at)}
                        </td>
                        <td className="px-4 py-2 font-mono text-xs text-muted-foreground">
                          {job.created_at ? job.created_at.slice(0, 19) : "—"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </CardContent>
            </Card>
          )}
        </main>
      </div>
    </>
  );
}
