import * as React from "react";
import { Gauge } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/utils";
import type {
  HarTercileBaselineResponse,
  HarTercileHorizon,
  HarTercileLabel,
} from "@/lib/analyze/types";

const HORIZON_LABELS: Record<number, string> = {
  1: "1 day",
  5: "1 week",
  22: "1 month",
};

const TERCILE_ORDER: readonly HarTercileLabel[] = ["low", "medium", "high"];

function tercileBarClass(label: HarTercileLabel): string {
  if (label === "low") return "bg-dovish";
  if (label === "high") return "bg-hawkish";
  return "bg-neutral";
}

function tercileBadgeVariant(
  label: HarTercileLabel,
): "hawkish" | "dovish" | "neutral" {
  if (label === "low") return "dovish";
  if (label === "high") return "hawkish";
  return "neutral";
}

function annualizedVolPct(rv: number): string {
  const ann = Math.sqrt(Math.max(rv, 0) * 252) * 100;
  return `${ann.toFixed(1)}%`;
}

interface HorizonColumnProps {
  horizon: HarTercileHorizon;
}

function HorizonColumn({ horizon }: HorizonColumnProps) {
  const label = HORIZON_LABELS[horizon.h] ?? `${horizon.h}d`;
  const top = horizon.top_pick;
  return (
    <div className="space-y-3 rounded-md border border-border bg-card/50 p-3">
      <div className="flex items-center justify-between gap-2">
        <p className="text-xs uppercase tracking-wide text-muted-foreground">{label}</p>
        <Badge
          variant="outline"
          className="numeric text-[10px]"
          title={`Walk-forward macro-F1 from wiki §20 (n=${horizon.n}).`}
        >
          macro-F1 {horizon.macro_f1.toFixed(3)} (wiki §20, n={horizon.n})
        </Badge>
      </div>
      <div className="flex flex-wrap items-baseline gap-2">
        <Badge variant={tercileBadgeVariant(top)} className="capitalize">
          {top}
        </Badge>
        <span className="numeric text-sm text-muted-foreground">
          predicted RV {annualizedVolPct(horizon.predicted_rv)}
        </span>
      </div>
      <div className="space-y-1.5">
        {TERCILE_ORDER.map((key) => {
          const value = horizon.probabilities[key] ?? 0;
          const isTop = key === top;
          return (
            <div key={key} className="space-y-1">
              <div className="flex items-center justify-between text-[11px]">
                <span
                  className={cn(
                    "flex items-center gap-1.5 capitalize",
                    isTop ? "text-foreground" : "text-muted-foreground",
                  )}
                >
                  <span
                    className={cn(
                      "inline-block h-1.5 w-1.5 rounded-full",
                      tercileBarClass(key),
                    )}
                    aria-hidden="true"
                  />
                  {key}
                </span>
                <span
                  className={cn(
                    "numeric",
                    isTop ? "font-medium text-foreground" : "text-muted-foreground",
                  )}
                >
                  {(value * 100).toFixed(1)}%
                </span>
              </div>
              <Progress value={value} indicatorClassName={tercileBarClass(key)} />
            </div>
          );
        })}
      </div>
    </div>
  );
}

interface HarRegimeHeadlineProps {
  baselines: HarTercileBaselineResponse | null;
  loading?: boolean;
  error?: string | null;
  symbol?: string;
}

export function HarRegimeHeadline({
  baselines,
  loading = false,
  error = null,
  symbol,
}: HarRegimeHeadlineProps) {
  const headlineBadge = (
    <Badge variant="default" className="text-[10px] uppercase tracking-wide">
      Headline
    </Badge>
  );

  if (loading) {
    return (
      <Card className="overflow-hidden border-primary/40 bg-primary/5">
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between gap-3">
            <CardDescription className="flex items-center gap-1.5">
              <Gauge className="h-3.5 w-3.5" /> Volatility regime · HAR-tercile baseline
            </CardDescription>
            {headlineBadge}
          </div>
          <CardTitle className="text-base">Loading HAR baseline…</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid gap-2 md:grid-cols-3">
            <Skeleton className="h-40 w-full" />
            <Skeleton className="h-40 w-full" />
            <Skeleton className="h-40 w-full" />
          </div>
        </CardContent>
      </Card>
    );
  }

  if (error || !baselines || baselines.horizons.length === 0) {
    return (
      <Card className="overflow-hidden border-primary/40 bg-primary/5">
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between gap-3">
            <CardDescription className="flex items-center gap-1.5">
              <Gauge className="h-3.5 w-3.5" /> Volatility regime · HAR-tercile baseline
            </CardDescription>
            {headlineBadge}
          </div>
          <CardTitle className="text-base text-muted-foreground">
            HAR baseline unavailable
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-xs text-muted-foreground">
            {error ?? "Baseline artifact has not loaded yet. Retry shortly."}
          </p>
        </CardContent>
      </Card>
    );
  }

  // Render horizons in a fixed 1d / 5d / 22d order regardless of how
  // the backend listed them; missing horizons just drop out.
  const horizonsByH = new Map<number, HarTercileHorizon>();
  for (const h of baselines.horizons) horizonsByH.set(h.h, h);
  const ordered = [1, 5, 22]
    .map((h) => horizonsByH.get(h))
    .filter((h): h is HarTercileHorizon => h !== undefined);

  const symbolBadge = symbol ?? baselines.symbol;

  return (
    <Card className="overflow-hidden border-primary/40 bg-primary/5">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between gap-3">
          <CardDescription className="flex items-center gap-1.5">
            <Gauge className="h-3.5 w-3.5" /> Volatility regime · HAR-tercile baseline
          </CardDescription>
          <div className="flex flex-wrap items-center gap-2">
            {symbolBadge ? (
              <Badge variant="outline" className="numeric text-[10px]">
                {symbolBadge}
              </Badge>
            ) : null}
            {headlineBadge}
          </div>
        </div>
        <CardTitle className="text-base">3-class forward-vol classifier · Low / Medium / High</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="grid gap-3 md:grid-cols-3">
          {ordered.map((horizon) => (
            <HorizonColumn key={horizon.h} horizon={horizon} />
          ))}
        </div>
        <p className="text-[11px] text-muted-foreground">
          HAR-tercile baseline. 3-class forward-vol classifier (Low / Med / High).
          Macro-F1 from the walk-forward eval reported in wiki §20. Beats both market-only
          and fused text+market models — see Second opinion below.
        </p>
      </CardContent>
    </Card>
  );
}
