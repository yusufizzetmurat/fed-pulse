import * as React from "react";
import { AlertTriangle, ShieldCheck } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Sparkline } from "@/components/ui/sparkline";
import { cn } from "@/lib/utils";
import type { RegimeClassificationResponse, SentimentResponse } from "@/lib/analyze/types";
import { EvidenceLink } from "@/components/analyze/EvidenceLink";

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

// Fold-4 with/without numbers, computed from
// backend/artifacts/experiments/dual_head_comparison_canonical.json
// (dual head, 5 seeds × 5 folds, alpha=0.5). Hard-coded here because
// the canonical artefact is not exposed on /analyze and a dedicated
// endpoint just for one footnote is not worth its keep — the
// evidence chip points the reader to §6.7 / §6.15 / row 10b for the
// load-bearing context.
const FOLD4_DUAL_F1_WITH = 0.419;
const FOLD4_DUAL_F1_WITHOUT = 0.414;
const FOLD4_DUAL_F1_STD_WITH = 0.070;
const FOLD4_DUAL_F1_STD_WITHOUT = 0.079;
const FOLD4_DUAL_RMSE_WITH = 1.004;
const FOLD4_DUAL_RMSE_WITHOUT = 1.043;

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
              ? "log(RV) regression band · 10d forward"
              : "Vol-regime prediction set · 10d forward"}
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
            <EvidenceLink section="6.15" label="Three-way comparison · row 10b" />
          </div>
        </div>
        {hasRegressionBand ? (
          <CardTitle className="flex flex-wrap items-end gap-3 sm:gap-4">
            <span className="numeric text-4xl font-semibold tracking-tight sm:text-5xl">
              {(regime.log_rv_point as number).toFixed(3)}
            </span>
            <span className="numeric text-sm text-muted-foreground sm:text-base">
              log(RV) point · band [{(regime.log_rv_lower as number).toFixed(3)},{" "}
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
              title="UI-side bucket derived from the regression point against the per-fold tertile cutoffs."
            >
              {regime.argmax_class} bucket
            </Badge>
            {regime.bucket_source ? (
              <Badge variant="outline" className="text-[10px] uppercase tracking-wide">
                bucket source · {regime.bucket_source}
              </Badge>
            ) : null}
          </CardTitle>
        ) : (
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
        )}
      </CardHeader>
      <CardContent className="grid gap-6 md:grid-cols-2">
        <div className="space-y-3 rounded-md border border-border bg-muted/10 p-3">
          <p className="text-[10px] uppercase tracking-wide text-muted-foreground">
            Fold-4 with / without · dual head, 5 seeds × 5 folds
          </p>
          <div className="grid grid-cols-2 gap-3 text-xs">
            <div className="space-y-1">
              <p className="text-muted-foreground">With fold-4 (canonical)</p>
              <p className="numeric font-medium text-foreground">
                F1 {FOLD4_DUAL_F1_WITH.toFixed(3)} ± {FOLD4_DUAL_F1_STD_WITH.toFixed(3)}
              </p>
              <p className="numeric text-muted-foreground">
                RMSE log(RV) {FOLD4_DUAL_RMSE_WITH.toFixed(3)}
              </p>
            </div>
            <div className="space-y-1">
              <p className="text-muted-foreground">Without fold-4</p>
              <p className="numeric font-medium text-foreground">
                F1 {FOLD4_DUAL_F1_WITHOUT.toFixed(3)} ± {FOLD4_DUAL_F1_STD_WITHOUT.toFixed(3)}
              </p>
              <p className="numeric text-muted-foreground">
                RMSE log(RV) {FOLD4_DUAL_RMSE_WITHOUT.toFixed(3)}
              </p>
            </div>
          </div>
          <p className="text-[11px] leading-relaxed text-muted-foreground">
            Fold-4 sits at the R-17 zero-<span className="numeric">calm</span> slice. The delta is small (F1
            +0.005, RMSE −0.039) which means the canonical headline is not load-bearing on the
            degenerate fold. Cells from <code>dual_head_comparison_canonical.json</code>; full
            four-variant context in §6.7.
          </p>
          <EvidenceLink section="6.7" label="Honest headline reporting · four-variant table" />
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
        <details className="md:col-span-2 group rounded-md border border-border bg-muted/10 p-3 text-xs">
          <summary className="cursor-pointer select-none text-[11px] uppercase tracking-wide text-muted-foreground group-open:text-foreground">
            Per-class softmax + calibrated set (secondary detail)
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
                Calibrated APS prediction set at nominal {coveragePct}% coverage:{" "}
                <span className="text-foreground numeric">
                  {`{${regime.predicted_set.join(", ")}}`}
                </span>{" "}
                · size {regime.set_size}.
              </p>
              <p>
                Argmax: <span className="text-foreground capitalize">{regime.argmax_class}</span>{" "}
                ({(argmaxProb * 100).toFixed(1)}%). Per-class precision / recall / F1 publish on
                the §6.10 baselines table; the classifier-branch macro-F1 on this checkpoint
                lands at 0.418 ± 0.052 (row 10b sibling). Demoted from the headline because
                §6.15 / row 10b makes the regression band the canonical surface and the
                classification head sits inside its CI.
              </p>
              <EvidenceLink section="6.10" label="Baselines table · per-class detail" />
            </div>
          </div>
        </details>
      </CardContent>
    </Card>
  );
}
