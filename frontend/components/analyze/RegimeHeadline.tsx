import * as React from "react";
import { AlertTriangle, ShieldCheck } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Sparkline } from "@/components/ui/sparkline";
import { cn } from "@/lib/utils";
import type { RegimeClassificationResponse, SentimentResponse } from "@/lib/analyze/types";

const REGIME_ORDER = ["calm", "normal", "high"] as const;
type Regime = (typeof REGIME_ORDER)[number] | string;

interface RegimeHeadlineProps {
  regime: RegimeClassificationResponse;
  sentiment?: SentimentResponse;
  history?: Array<{ documentDate: string; argmax: string | null; realized?: string | null }>;
  symbol?: string;
  documentDate?: string;
  // Empirical conformal coverage across recent history (fraction in
  // [0,1]). When provided alongside the run-level nominal coverage,
  // the headline renders a "Nominal X% · Empirical Y%" chip so the
  // calibration claim on the spine card is audited rather than
  // asserted. Null/undefined hides the chip.
  empiricalCoverage?: number | null;
  empiricalCoverageSampleSize?: number | null;
}

function regimeBarClass(label: Regime): string {
  if (label === "calm") return "bg-dovish";
  if (label === "high") return "bg-hawkish";
  return "bg-neutral";
}

function regimeChipVariant(
  label: Regime,
): "hawkish" | "dovish" | "neutral" | "outline" {
  if (label === "calm") return "dovish";
  if (label === "high") return "hawkish";
  if (label === "normal") return "neutral";
  return "outline";
}

function regimeFromIndex(values: Array<{ argmax: string | null }>): SparklineValue[] {
  return values.map((v) => {
    if (v.argmax === "calm") return -1;
    if (v.argmax === "high") return 1;
    if (v.argmax === "normal") return 0;
    return null;
  });
}

type SparklineValue = number | null;

export function RegimeHeadline({
  regime,
  sentiment,
  history,
  symbol,
  documentDate,
  empiricalCoverage,
  empiricalCoverageSampleSize,
}: RegimeHeadlineProps) {
  const distribution = regime.distribution ?? {};
  const coveragePct = Math.round(regime.coverage * 100);
  const hasEmpirical =
    typeof empiricalCoverage === "number" &&
    !Number.isNaN(empiricalCoverage) &&
    (empiricalCoverageSampleSize ?? 0) > 0;
  const empiricalPct = hasEmpirical
    ? Math.round((empiricalCoverage as number) * 100)
    : null;
  const coverageDeltaPct = hasEmpirical && empiricalPct !== null ? empiricalPct - coveragePct : 0;
  const coverageDriftLarge = Math.abs(coverageDeltaPct) >= 10;
  const knownOrder = new Set<string>(REGIME_ORDER);
  const extraLabels = Object.keys(distribution).filter((k) => !knownOrder.has(k));
  const renderOrder = [...REGIME_ORDER, ...extraLabels];
  const argmaxProb = distribution[regime.argmax_class] ?? 0;
  const oodFlag = sentiment?.is_in_distribution === false;
  const trendValues = history ? regimeFromIndex(history) : [];

  return (
    <Card className="overflow-hidden">
      <CardHeader className="space-y-2 pb-3">
        <div className="flex items-center justify-between gap-3">
          <CardDescription className="flex items-center gap-1.5">
            <ShieldCheck className="h-3.5 w-3.5" />
            Vol-regime prediction set · 10d forward
          </CardDescription>
          <div className="flex flex-wrap items-center gap-2">
            {symbol ? (
              <Badge variant="outline" className="numeric text-[10px]">
                {symbol}
              </Badge>
            ) : null}
            {documentDate ? (
              <Badge variant="outline" className="numeric text-[10px]">
                {documentDate}
              </Badge>
            ) : null}
            <Badge variant="outline" className="numeric text-[10px]">
              {hasEmpirical
                ? `Nominal ${coveragePct}% · Empirical ${empiricalPct}%`
                : `${coveragePct}% coverage`}
              {` · set size ${regime.set_size}`}
            </Badge>
            {hasEmpirical && coverageDriftLarge ? (
              <Badge
                variant={coverageDeltaPct < 0 ? "hawkish" : "neutral"}
                className="text-[10px]"
                title={`Empirical coverage drifted ${
                  coverageDeltaPct < 0 ? "below" : "above"
                } nominal across ${empiricalCoverageSampleSize ?? 0} runs.`}
              >
                {coverageDeltaPct > 0 ? "+" : ""}
                {coverageDeltaPct}pp drift
              </Badge>
            ) : null}
            {oodFlag ? (
              <Badge variant="hawkish" className="text-[10px]">
                <AlertTriangle className="h-3 w-3" /> OOD
              </Badge>
            ) : null}
          </div>
        </div>
        <CardTitle className="flex flex-wrap items-end gap-3 sm:gap-4">
          <span className="numeric text-4xl font-semibold capitalize tracking-tight sm:text-5xl">
            {regime.argmax_class}
          </span>
          <span className="numeric text-sm text-muted-foreground sm:text-base">
            argmax · {(argmaxProb * 100).toFixed(1)}%
          </span>
          <div className="flex flex-wrap items-center gap-1.5">
            {regime.predicted_set.map((label) => (
              <Badge
                key={label}
                variant={regimeChipVariant(label)}
                className="capitalize"
              >
                {label}
              </Badge>
            ))}
          </div>
        </CardTitle>
      </CardHeader>
      <CardContent className="grid gap-6 md:grid-cols-2">
        <div className="space-y-2.5">
          {renderOrder.map((key) => {
            const value = distribution[key];
            if (value === undefined) return null;
            const inSet = regime.predicted_set.includes(key);
            return (
              <div key={key} className="space-y-1">
                <div className="flex items-center justify-between text-xs">
                  <span
                    className={cn(
                      "flex items-center gap-1.5 capitalize",
                      inSet ? "text-foreground" : "text-muted-foreground",
                    )}
                  >
                    <span
                      className={cn(
                        "inline-block h-1.5 w-1.5 rounded-full",
                        regimeBarClass(key),
                      )}
                      aria-hidden="true"
                    />
                    {key}
                    {inSet ? <span className="text-[10px] text-muted-foreground">in set</span> : null}
                  </span>
                  <span
                    className={cn(
                      "numeric",
                      inSet ? "font-medium text-foreground" : "text-muted-foreground",
                    )}
                  >
                    {(value * 100).toFixed(1)}%
                  </span>
                </div>
                <Progress value={value} indicatorClassName={regimeBarClass(key)} />
              </div>
            );
          })}
        </div>
        <div className="space-y-3 rounded-md border border-border bg-muted/20 p-3">
          <p className="text-[10px] uppercase tracking-wide text-muted-foreground">
            Past {trendValues.length || 0} runs · argmax regime
          </p>
          {trendValues.length ? (
            <Sparkline
              values={trendValues}
              tone="neutral"
              height={56}
              formatTooltip={(value, label) => {
                const regimeLabel = value > 0 ? "high" : value < 0 ? "calm" : "normal";
                return `${label ?? ""} → ${regimeLabel}`;
              }}
              labels={history?.map((h) => h.documentDate)}
            />
          ) : (
            <p className="text-xs text-muted-foreground">
              No prior runs for this symbol yet — run history will appear here.
            </p>
          )}
          <div className="grid grid-cols-3 gap-1 text-center text-[10px] text-muted-foreground">
            <span className="numeric">calm</span>
            <span className="numeric">normal</span>
            <span className="numeric">high</span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
