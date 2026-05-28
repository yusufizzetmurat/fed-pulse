import { useState } from "react";
import { ChevronDown, ChevronUp, ShieldCheck } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import type {
  RegimeClassificationResponse,
  RegimeRegressionResponse,
} from "@/lib/analyze/types";

interface RegimeClassificationCardProps {
  regime: RegimeClassificationResponse;
  // #304 dual-head retrofit: optional sibling block carrying the
  // regression head's point estimate + conformal interval. When
  // provided, a "show details" toggle reveals it underneath the
  // classification surface; absence keeps the pre-#304 visual
  // language identical.
  regression?: RegimeRegressionResponse | null;
}

const REGIME_ORDER = ["calm", "normal", "high"];

function regimeIndicatorClass(label: string): string {
  switch (label) {
    case "calm":
      return "bg-dovish";
    case "high":
      return "bg-hawkish";
    case "normal":
    default:
      return "bg-neutral";
  }
}

function formatLogRv(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return "—";
  }
  return value.toFixed(3);
}

export function RegimeClassificationCard({ regime, regression }: RegimeClassificationCardProps) {
  const [showRegression, setShowRegression] = useState(false);
  const distribution = regime.distribution ?? {};
  const coveragePct = Math.round(regime.coverage * 100);
  const known = new Set(REGIME_ORDER);
  const extraLabels = Object.keys(distribution).filter((k) => !known.has(k));
  const renderOrder = [...REGIME_ORDER, ...extraLabels];
  const hasRegression = regression != null && Number.isFinite(regression.log_rv_point);
  const regressionCoveragePct =
    regression?.coverage != null && Number.isFinite(regression.coverage)
      ? Math.round(regression.coverage * 100)
      : coveragePct;
  return (
    <Card>
      <CardHeader className="pb-2">
        <CardDescription className="flex items-center gap-1.5">
          <ShieldCheck className="h-3.5 w-3.5" />
          Regime prediction set
        </CardDescription>
        <CardTitle className="flex items-center justify-between text-2xl">
          <span className="font-mono">{regime.set_label}</span>
          <Badge variant="outline">{coveragePct}% coverage</Badge>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="flex flex-wrap gap-1.5">
          {regime.predicted_set.map((label) => (
            <Badge key={label} variant="default" className="capitalize">
              {label}
            </Badge>
          ))}
        </div>
        <div className="space-y-2">
          {renderOrder.map((key) => {
            const value = distribution[key];
            if (value === undefined) return null;
            const inSet = regime.predicted_set.includes(key);
            return (
              <div key={key} className="space-y-1">
                <div className="flex items-center justify-between text-xs text-muted-foreground">
                  <span className="capitalize">
                    {key} {inSet ? "✓" : ""}
                  </span>
                  <span className={inSet ? "font-medium text-foreground" : "text-muted-foreground"}>
                    {value.toFixed(2)}
                  </span>
                </div>
                <Progress value={value} indicatorClassName={regimeIndicatorClass(key)} />
              </div>
            );
          })}
        </div>
        <p className="text-xs text-muted-foreground">
          Split-conformal prediction set at nominal {coveragePct}% coverage.
          Argmax: <span className="font-medium text-foreground capitalize">{regime.argmax_class}</span> · set size {regime.set_size}.
        </p>
        {hasRegression && (
          <div className="border-t pt-2">
            <Button
              type="button"
              variant="ghost"
              size="sm"
              className="h-7 px-2 text-xs text-muted-foreground hover:text-foreground"
              onClick={() => setShowRegression((open) => !open)}
              aria-expanded={showRegression}
              aria-controls="regime-regression-details"
            >
              {showRegression ? (
                <ChevronUp className="mr-1 h-3 w-3" />
              ) : (
                <ChevronDown className="mr-1 h-3 w-3" />
              )}
              {showRegression ? "Hide regression details" : "Show regression details"}
            </Button>
            {showRegression && regression && (
              <div
                id="regime-regression-details"
                className="mt-2 space-y-1 rounded-md border bg-muted/40 px-3 py-2 text-xs text-muted-foreground"
              >
                <div className="flex items-center justify-between">
                  <span>log(RV) point</span>
                  <span className="font-mono text-foreground">{formatLogRv(regression.log_rv_point)}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span>{regressionCoveragePct}% interval</span>
                  <span className="font-mono text-foreground">
                    [{formatLogRv(regression.log_rv_lower)}, {formatLogRv(regression.log_rv_upper)}]
                  </span>
                </div>
                <p className="pt-1 text-[10px] leading-snug">
                  Dual-head regression on standardised log(forward realized vol);
                  conformal interval at nominal {regressionCoveragePct}% coverage.
                  Classification surface above stays the headline.
                </p>
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
