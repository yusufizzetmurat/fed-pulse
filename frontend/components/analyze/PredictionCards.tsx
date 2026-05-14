import { ArrowDownRight, ArrowUpRight, Minus } from "lucide-react";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { formatPercentDelta, formatPrice, formatPriceDelta, formatVol } from "@/lib/analyze/format";
import type { AnalyzeResult } from "@/lib/analyze/types";

interface PredictionCardsProps {
  result: AnalyzeResult;
}

export function PredictionCards({ result }: PredictionCardsProps) {
  const predicted = Number(result.prediction?.close ?? NaN);
  const current = Number(result.market?.close ?? NaN);
  const delta = Number.isFinite(predicted) && Number.isFinite(current) ? predicted - current : null;
  const pct = delta != null && current ? (delta / current) * 100 : null;
  const tone: "up" | "down" | "flat" =
    delta == null ? "flat" : delta > 0 ? "up" : delta < 0 ? "down" : "flat";
  const Arrow = tone === "up" ? ArrowUpRight : tone === "down" ? ArrowDownRight : Minus;
  const toneClass =
    tone === "up" ? "text-hawkish" : tone === "down" ? "text-dovish" : "text-muted-foreground";

  const historyClose = result.series?.history_close;
  const forecastClose = result.series?.forecast_close;
  let closeChangePct = 0;
  if (historyClose?.length && forecastClose?.length) {
    const lastHist = Number(historyClose[historyClose.length - 1]);
    const lastFc = Number(forecastClose[forecastClose.length - 1]);
    if (lastHist) closeChangePct = ((lastFc - lastHist) / lastHist) * 100;
  }

  const horizon = result.prediction?.horizon || "3d";
  const volatility = Number(result.prediction?.volatility ?? 0);

  return (
    <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
      <Card>
        <CardHeader className="pb-2">
          <CardDescription>Predicted close</CardDescription>
          <CardTitle className="text-2xl">{formatPrice(predicted)}</CardTitle>
        </CardHeader>
        <CardContent>
          <div className={`flex items-center gap-2 text-sm font-medium ${toneClass}`}>
            <Arrow className="h-4 w-4" />
            <span>{delta == null ? "N/A" : formatPriceDelta(delta)}</span>
            <span>·</span>
            <span>{pct == null ? "—" : formatPercentDelta(pct)}</span>
          </div>
          <p className="mt-2 text-xs text-muted-foreground">
            Current spot {formatPrice(Number.isFinite(current) ? current : null)} · horizon {horizon}
          </p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="pb-2">
          <CardDescription>Forecast change</CardDescription>
          <CardTitle className="text-2xl">{formatPercentDelta(closeChangePct)}</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-xs text-muted-foreground">
            Last forecast close vs last history close.
          </p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="pb-2">
          <CardDescription>Predicted volatility</CardDescription>
          <CardTitle className="text-2xl">{formatVol(volatility)}</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-xs text-muted-foreground">
            5d realized-vol proxy, horizon {horizon}.
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
