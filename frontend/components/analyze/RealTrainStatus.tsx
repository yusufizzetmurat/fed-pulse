import { Loader2 } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import type { TrainJobState } from "@/lib/analyze/types";

interface RealTrainStatusProps {
  job: TrainJobState | null;
}

const STATUS_VARIANT: Record<TrainJobState["status"], "outline" | "hawkish" | "dovish" | "neutral"> = {
  queued: "outline",
  running: "neutral",
  succeeded: "hawkish",
  failed: "dovish",
};

export function RealTrainStatus({ job }: RealTrainStatusProps) {
  if (!job) return null;
  const variant = STATUS_VARIANT[job.status] ?? "outline";
  const isActive = job.status === "queued" || job.status === "running";
  return (
    <Card>
      <CardHeader className="flex-row items-start justify-between space-y-0">
        <div>
          <CardTitle>Real train job</CardTitle>
          <CardDescription className="font-mono text-xs">{job.job_id}</CardDescription>
        </div>
        <Badge variant={variant} className="uppercase tracking-wide">
          {job.status}
        </Badge>
      </CardHeader>
      <CardContent>
        <div className="flex items-center gap-2 text-sm">
          {isActive ? <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" /> : null}
          <p>{job.message || "Waiting for job updates…"}</p>
        </div>
        {job.error ? (
          <p className="mt-2 text-sm text-destructive">Error: {job.error}</p>
        ) : null}
      </CardContent>
    </Card>
  );
}
