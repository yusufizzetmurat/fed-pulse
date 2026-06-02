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
  // Optional text-channel attribution. ``marketOnlyArgmaxProb`` is the
  // top-pick probability under a market-only baseline (no text input or
  // text masked out); when provided the headline renders a "text
  // contribution ±X.Xpp" chip. When the live argmax probability sits
  // below uniform + 2pp the surface instead shows "text channel: weak"
  // to telegraph that the text barely moved the prediction.
  marketOnlyArgmaxProb?: number | null;
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
  marketOnlyArgmaxProb,
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

  // Text-channel contribution badge. Prefer the explicit delta vs a
  // market-only baseline when the caller threads one in; otherwise fall
  // back to a "weak" tag whenever argmax sits within 2pp of uniform.
  const classCount = Math.max(1, Object.keys(distribution).length);
  const uniformProb = 1 / classCount;
  let textContribLabel: string | null = null;
  let textContribTitle: string | null = null;
  if (typeof marketOnlyArgmaxProb === "number" && Number.isFinite(marketOnlyArgmaxProb)) {
    const deltaPp = (argmaxProb - marketOnlyArgmaxProb) * 100;
    const rounded = Math.round(deltaPp * 10) / 10;
    const sign = rounded > 0 ? "+" : rounded < 0 ? "−" : "±";
    textContribLabel = `text contribution ${sign}${Math.abs(rounded).toFixed(1)}pp`;
    textContribTitle = `Top-pick probability under text vs market-only: ${(
      argmaxProb * 100
    ).toFixed(1)}% vs ${(marketOnlyArgmaxProb * 100).toFixed(1)}%.`;
  } else if (argmaxProb - uniformProb < 0.02) {
    textContribLabel = "text channel: weak";
    textContribTitle = "Top-pick probability is within 2pp of uniform. The text barely moved the prediction.";
  }

  // #338 reframe: when the dual-head regression branch is mounted on
  // the active checkpoint we lead with the log(RV) band; per-class
  // softmax + conformal set become foldable detail. Older
  // classification-only checkpoints fall back to the previous
  // softmax-led surface.
  const hasRegressionBand =
    regime.log_rv_point != null
    && regime.log_rv_lower != null
    && regime.log_rv_upper != null;
  const bandWidth =
    hasRegressionBand
      ? Math.abs((regime.log_rv_upper as number) - (regime.log_rv_lower as number))
      : null;

  return (
    <Card className="overflow-hidden">
      <CardHeader className="space-y-2 pb-3">
        <div className="flex items-center justify-between gap-3">
          <CardDescription className="flex items-center gap-1.5">
            <ShieldCheck className="h-3.5 w-3.5" />
            {hasRegressionBand
              ? "Volatility forecast · 10 days ahead (log scale)"
              : "Volatility Regime prediction · 10 days ahead"}
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
                ? `Target ${coveragePct}% · Actual ${empiricalPct}%`
                : `${coveragePct}% confidence level`}
              {` · ${regime.set_size} label${regime.set_size === 1 ? "" : "s"} in set`}
            </Badge>
            {hasEmpirical && coverageDriftLarge ? (
              <Badge
                variant={coverageDeltaPct < 0 ? "hawkish" : "neutral"}
                className="text-[10px]"
                title={`Actual coverage drifted ${
                  coverageDeltaPct < 0 ? "below" : "above"
                } target across ${empiricalCoverageSampleSize ?? 0} runs.`}
              >
                {coverageDeltaPct > 0 ? "+" : ""}
                {coverageDeltaPct}pp drift
              </Badge>
            ) : null}
            {oodFlag ? (
              <Badge variant="hawkish" className="text-[10px]" title="Text looks unlike anything the model was trained on">
                <AlertTriangle className="h-3 w-3" /> Unfamiliar text
              </Badge>
            ) : null}
          </div>
        </div>
        {hasRegressionBand ? (
          <CardTitle className="flex flex-wrap items-end gap-3 sm:gap-4">
            <span className="numeric text-4xl font-semibold tracking-tight sm:text-5xl">
              {(regime.log_rv_point as number).toFixed(3)}
            </span>
            <span className="numeric text-sm text-muted-foreground sm:text-base">
              Point estimate · confidence range [{(regime.log_rv_lower as number).toFixed(3)},{" "}
              {(regime.log_rv_upper as number).toFixed(3)}]
              {bandWidth != null ? (
                <span className="ml-1 text-muted-foreground/80">
                  · width {bandWidth.toFixed(3)}
                </span>
              ) : null}
            </span>
            <Badge
              variant={regimeChipVariant(regime.argmax_class)}
              className="capitalize"
              title="Regime bucket derived from the numeric forecast using the per-fold cutoffs."
            >
              {regime.argmax_class} regime
            </Badge>
            {textContribLabel ? (
              <Badge
                variant="outline"
                className="text-[10px] uppercase tracking-wide"
                title={textContribTitle ?? undefined}
              >
                {textContribLabel}
              </Badge>
            ) : null}
            {regime.bucket_source ? (
              <Badge variant="outline" className="text-[10px] uppercase tracking-wide">
                regime source · {regime.bucket_source}
              </Badge>
            ) : null}
          </CardTitle>
        ) : (
          <CardTitle className="flex flex-wrap items-end gap-3 sm:gap-4">
            <span className="numeric text-4xl font-semibold capitalize tracking-tight sm:text-5xl">
              {regime.argmax_class}
            </span>
            <span className="numeric text-sm text-muted-foreground sm:text-base">
              top pick · {(argmaxProb * 100).toFixed(1)}%
            </span>
            {textContribLabel ? (
              <Badge
                variant="outline"
                className="text-[10px] uppercase tracking-wide"
                title={textContribTitle ?? undefined}
              >
                {textContribLabel}
              </Badge>
            ) : null}
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
        )}
      </CardHeader>
      <CardContent className="grid gap-6 md:grid-cols-2">
        <div className="space-y-3 rounded-md border border-border bg-muted/20 p-3">
          <p className="text-[10px] uppercase tracking-wide text-muted-foreground">
            Past {trendValues.length || 0} runs · top-pick regime
          </p>
          {trendValues.length >= 2 ? (
            <Sparkline
              values={trendValues}
              tone="neutral"
              height={56}
              yDomain={[-1.5, 1.5]}
              formatTooltip={(value, label) => {
                const regimeLabel = value > 0 ? "high" : value < 0 ? "calm" : "normal";
                return `${label ?? ""} → ${regimeLabel}`;
              }}
              labels={history?.map((h) => h.documentDate)}
            />
          ) : trendValues.length === 1 ? (
            <p className="text-xs text-muted-foreground">
              Insufficient history (1 run). Sparkline appears with two or more runs.
            </p>
          ) : (
            <p className="text-xs text-muted-foreground">
              No prior runs for this symbol yet. Run history will appear here.
            </p>
          )}
          <div className="grid grid-cols-3 gap-1 text-center text-[10px] text-muted-foreground">
            <span className="numeric">calm</span>
            <span className="numeric">normal</span>
            <span className="numeric">high</span>
          </div>
        </div>
        <details className="md:col-span-2 group rounded-md border border-border bg-muted/10 p-3 text-xs">
          <summary className="cursor-pointer select-none text-[11px] uppercase tracking-wide text-muted-foreground group-open:text-foreground">
            Per-class probabilities and prediction set (details)
          </summary>
          <div className="mt-3 grid gap-3 md:grid-cols-2">
            <div className="space-y-2">
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
            <div className="space-y-2 text-muted-foreground">
              <p>
                Calibrated prediction set at {coveragePct}% confidence level:{" "}
                <span className="text-foreground numeric">
                  {`{${regime.predicted_set.join(", ")}}`}
                </span>{" "}
                · {regime.set_size} label{regime.set_size === 1 ? "" : "s"} in set.
              </p>
              <p>
                Top pick: <span className="text-foreground capitalize">{regime.argmax_class}</span>{" "}
                ({(argmaxProb * 100).toFixed(1)}%). Numeric forecast above is the headline,
                with the classifier shown as supporting detail.
              </p>
            </div>
          </div>
        </details>
      </CardContent>
    </Card>
  );
}
