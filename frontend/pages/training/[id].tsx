import * as React from "react";
import Head from "next/head";
import Link from "next/link";
import { useRouter } from "next/router";
import { ArrowLeft } from "lucide-react";
import { toast } from "sonner";

import { Header } from "@/components/shell/header";
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
import { fetchTrainJob, resolveApiBaseUrl } from "@/lib/analyze/api";
import type { TrainJobState } from "@/lib/analyze/types";

const STATUS_VARIANT: Record<string, "hawkish" | "dovish" | "outline" | "secondary"> = {
  running: "hawkish",
  queued: "secondary",
  failed: "dovish",
  succeeded: "outline",
};

export default function TrainingDetailPage() {
  const router = useRouter();
  const apiBaseUrl = React.useMemo(() => resolveApiBaseUrl(), []);
  const [job, setJob] = React.useState<TrainJobState | null>(null);
  const [loading, setLoading] = React.useState(true);
  const [errorMessage, setErrorMessage] = React.useState<string | null>(null);

  const jobId = React.useMemo(() => {
    const value = router.query.id;
    return typeof value === "string" ? value : null;
  }, [router.query.id]);

  React.useEffect(() => {
    if (!router.isReady) return;
    if (!jobId) return;
    let cancelled = false;
    const load = () => {
      fetchTrainJob(apiBaseUrl, jobId)
        .then((result) => {
          if (!cancelled) {
            setJob(result);
            setErrorMessage(null);
          }
        })
        .catch((err) => {
          if (!cancelled) {
            const message = (err as Error).message || "Failed to load job state.";
            setErrorMessage(message);
            toast.error(message);
          }
        })
        .finally(() => {
          if (!cancelled) setLoading(false);
        });
    };
    load();
    const id = setInterval(load, 4000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [apiBaseUrl, jobId, router.isReady]);

  const status = job?.status ?? "unknown";
  const variant = STATUS_VARIANT[status.toLowerCase()] ?? "outline";

  return (
    <>
      <Head>
        <title>Training job — Fed Pulse</title>
      </Head>
      <div className="min-h-screen bg-background text-foreground">
        <Header />
        <main className="container space-y-6 py-8">
          <div className="flex items-center justify-between gap-4">
            <div className="space-y-1">
              <h1 className="text-2xl font-semibold tracking-tight">Training job</h1>
              <p className="font-mono text-xs text-muted-foreground">{jobId ?? "—"}</p>
            </div>
            <Button asChild variant="outline" size="sm">
              <Link href="/training">
                <ArrowLeft className="h-4 w-4" />
                Back to queue
              </Link>
            </Button>
          </div>

          {loading && !job ? (
            <Skeleton className="h-48 w-full" />
          ) : errorMessage && !job ? (
            <Card>
              <CardContent className="py-10 text-center text-muted-foreground">{errorMessage}</CardContent>
            </Card>
          ) : job ? (
            <div className="grid gap-4 lg:grid-cols-2">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-3">
                    <Badge variant={variant} className="capitalize">
                      {status}
                    </Badge>
                    State
                  </CardTitle>
                  <CardDescription>Auto-refreshes every 4 seconds.</CardDescription>
                </CardHeader>
                <CardContent>
                  <dl className="grid grid-cols-2 gap-y-2 text-sm">
                    <dt className="text-muted-foreground">Status</dt>
                    <dd className="font-mono">{status}</dd>
                    {job.message ? (
                      <>
                        <dt className="text-muted-foreground">Message</dt>
                        <dd className="font-mono text-xs">{job.message}</dd>
                      </>
                    ) : null}
                    {job.error ? (
                      <>
                        <dt className="text-muted-foreground">Error</dt>
                        <dd className="font-mono text-xs text-rose-500">{job.error}</dd>
                      </>
                    ) : null}
                  </dl>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Result preview</CardTitle>
                  <CardDescription>
                    Populated on success. The full payload appears on the Analyze page once polling resolves.
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-2 text-sm">
                  {job.result ? (
                    <dl className="grid grid-cols-2 gap-y-2">
                      <dt className="text-muted-foreground">Sentiment</dt>
                      <dd className="font-mono">{job.result.sentiment?.label ?? "—"}</dd>
                      <dt className="text-muted-foreground">Predicted close</dt>
                      <dd className="font-mono">
                        {job.result.prediction?.close != null
                          ? job.result.prediction.close.toFixed(2)
                          : "—"}
                      </dd>
                      <dt className="text-muted-foreground">Predicted volatility</dt>
                      <dd className="font-mono">
                        {job.result.prediction?.volatility != null
                          ? job.result.prediction.volatility.toFixed(4)
                          : "—"}
                      </dd>
                      <dt className="text-muted-foreground">Runtime mode</dt>
                      <dd className="font-mono">{job.result.model?.runtime_mode ?? "—"}</dd>
                    </dl>
                  ) : (
                    <p className="text-muted-foreground">No result yet.</p>
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
