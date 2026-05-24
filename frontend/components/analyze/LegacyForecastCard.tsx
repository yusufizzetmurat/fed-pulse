import * as React from "react";
import { ArrowUpRight, Info } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { KpiTile } from "@/components/ui/kpi-tile";
import type { AnalyzeResult, MarketResponse, PredictionResponse } from "@/lib/analyze/types";

interface LegacyForecastCardProps {
  prediction: PredictionResponse;
  market?: MarketResponse;
  documentDate?: string;
}

function formatPrice(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return "—";
  return `$${Number(value).toLocaleString("en-US", { maximumFractionDigits: 2 })}`;
}

function formatVol(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return "—";
  return `${(Number(value) * 100).toFixed(2)}%`;
}

function formatPercentDelta(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return "—";
  const v = Number(value);
  return `${v >= 0 ? "+" : ""}${v.toFixed(2)}%`;
}

/**
 * Fallback view when the active forecaster is regression-mode and so
 * does not emit a calibrated regime card. Shows the two scalar
 * outputs the regression head actually produces — predicted close and
 * predicted volatility — so the workspace still surfaces *something*
 * model-driven. Clearly framed as a legacy view so reviewers do not
 * confuse a point forecast with the calibrated regime headline.
 */
export function LegacyForecastCard({
  prediction,
  market,
  documentDate,
}: LegacyForecastCardProps) {
  const close = prediction.close;
  const vol = prediction.volatility;
  const spot = market?.close ?? null;
  const closeDeltaPct =
    typeof close === "number" && typeof spot === "number" && spot !== 0
      ? ((close - spot) / spot) * 100
      : null;
  const horizonLabel = prediction.horizon ?? "—";
  // spot is the snapshot close on the request date, not today. Surface
  // the as-of date in the caption so a user running a historical
  // analysis doesn't read the delta against today's live price.
  const spotAsOf = market?.date_used ?? documentDate ?? null;
  const spotCaption =
    spot != null
      ? spotAsOf
        ? `Spot ${formatPrice(spot)} · as-of ${spotAsOf}`
        : `Spot ${formatPrice(spot)}`
      : "no spot reference";

  return (
    <Card>
      <CardHeader className="space-y-2 pb-3">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <CardTitle className="flex items-center gap-2 text-base">
            <ArrowUpRight className="h-4 w-4 text-primary" />
            Legacy point forecast
          </CardTitle>
          <Badge variant="outline" className="text-[10px] uppercase tracking-wide">
            regression head
          </Badge>
        </div>
        <CardDescription className="flex items-start gap-1.5 text-xs">
          <Info className="mt-0.5 h-3.5 w-3.5 flex-shrink-0 text-muted-foreground" />
          <span>
            The deployed forecaster is in regression mode — it emits scalar close + volatility
            predictions instead of the calibrated <span className="numeric">calm / normal / high</span>{" "}
            regime set. This card surfaces those numbers as a fallback while the classification head
            ships; the demo headline target is still the regime card above.
          </span>
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="grid gap-3 sm:grid-cols-2">
          <KpiTile
            label={`Predicted close · ${horizonLabel}`}
            value={<span className="numeric">{formatPrice(close)}</span>}
            delta={closeDeltaPct}
            deltaFormatter={(v) => formatPercentDelta(v)}
            tone={closeDeltaPct == null ? "neutral" : closeDeltaPct > 0 ? "up" : "down"}
            caption={spotCaption}
          />
          <KpiTile
            label={`Predicted volatility · ${horizonLabel}`}
            value={<span className="numeric">{formatVol(vol)}</span>}
            caption={
              market?.volatility_5d != null
                ? `5d realised ${formatVol(market.volatility_5d)}`
                : "annualised stdev of log returns"
            }
          />
        </div>
      </CardContent>
    </Card>
  );
}
