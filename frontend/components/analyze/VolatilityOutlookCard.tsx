import * as React from "react";
import { Activity } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Sparkline } from "@/components/ui/sparkline";
import type {
  RealizedVolForecastResponse,
  RealizedVolHorizonForecast,
} from "@/lib/analyze/types";

const HORIZON_LABELS: Record<number, string> = {
  1: "1 day",
  5: "1 week",
  22: "1 month",
};

function formatVol(rv: number): string {
  // Display annualized volatility (%): sqrt(rv * 252) * 100
  const ann = Math.sqrt(Math.max(rv, 0) * 252) * 100;
  return `${ann.toFixed(1)}%`;
}

function formatGain(qlikeModel: number | null, qlikeHar: number | null): string | null {
  if (qlikeModel == null || qlikeHar == null) return null;
  if (!Number.isFinite(qlikeModel) || !Number.isFinite(qlikeHar) || qlikeHar <= 0) {
    return null;
  }
  const gain = (qlikeHar - qlikeModel) / qlikeHar;
  if (gain <= 0) return null;
  return `beats HAR by ${(gain * 100).toFixed(1)}%`;
}

interface HorizonColumnProps {
  horizon: RealizedVolHorizonForecast;
}

function HorizonColumn({ horizon }: HorizonColumnProps) {
  const label = HORIZON_LABELS[horizon.h] ?? `${horizon.h}d`;
  const beats = formatGain(horizon.qlike_model, horizon.qlike_har);
  const coverage = horizon.coverage_empirical_90;
  return (
    <div className="space-y-2 rounded-md border border-border bg-card/50 p-3">
      <div className="flex items-center justify-between">
        <p className="text-xs uppercase tracking-wide text-muted-foreground">{label}</p>
        {beats ? (
          <Badge variant="dovish" className="text-[10px]">
            {beats}
          </Badge>
        ) : null}
      </div>
      <p className="numeric text-2xl font-semibold">{formatVol(horizon.point)}</p>
      <div className="space-y-1 text-[11px] text-muted-foreground">
        <p>
          <span className="font-mono">80%</span>{" "}
          <span className="numeric">
            {formatVol(horizon.band_lo_80)}–{formatVol(horizon.band_hi_80)}
          </span>
        </p>
        <p>
          <span className="font-mono">90%</span>{" "}
          <span className="numeric">
            {formatVol(horizon.band_lo_90)}–{formatVol(horizon.band_hi_90)}
          </span>
        </p>
      </div>
      {coverage != null && Number.isFinite(coverage) ? (
        <Badge variant="outline" className="text-[10px]">
          90% band: {(coverage * 100).toFixed(0)}% covered
        </Badge>
      ) : null}
    </div>
  );
}

interface VolatilityOutlookCardProps {
  forecast: RealizedVolForecastResponse | null;
  loading?: boolean;
  error?: string | null;
}

export function VolatilityOutlookCard({
  forecast,
  loading = false,
  error = null,
}: VolatilityOutlookCardProps) {
  if (loading) {
    return (
      <Card>
        <CardHeader className="pb-2">
          <CardDescription className="flex items-center gap-1.5">
            <Activity className="h-3.5 w-3.5" /> Volatility Outlook
          </CardDescription>
          <CardTitle>Loading forecast…</CardTitle>
        </CardHeader>
      </Card>
    );
  }

  if (error || !forecast || forecast.horizons.length === 0) {
    return (
      <Card>
        <CardHeader className="pb-2">
          <CardDescription className="flex items-center gap-1.5">
            <Activity className="h-3.5 w-3.5" /> Volatility Outlook
          </CardDescription>
          <CardTitle className="text-base text-muted-foreground">
            Forecast unavailable
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-xs text-muted-foreground">
            {error ?? "Model artifact has not loaded yet. Retry shortly."}
          </p>
        </CardContent>
      </Card>
    );
  }

  // Annualize the realized history sparkline for visual parity with the cards.
  const sparkValues = (forecast.history || []).map((rv) =>
    Number.isFinite(rv) && rv > 0 ? Math.sqrt(rv * 252) * 100 : null,
  );

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardDescription className="flex items-center gap-1.5">
          <Activity className="h-3.5 w-3.5" /> Volatility Outlook
        </CardDescription>
        <CardTitle className="flex items-center justify-between text-base">
          <span>QLIKE-DLq ensemble · {forecast.symbol}</span>
          <Badge variant="outline" className="text-[10px] uppercase tracking-wide">
            Market readout
          </Badge>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <p className="text-[11px] text-muted-foreground">
          Annualized realized volatility, ensemble mean with conformal bands.
        </p>
        <div className="grid gap-2 md:grid-cols-3">
          {forecast.horizons.map((h) => (
            <HorizonColumn key={h.h} horizon={h} />
          ))}
        </div>
        <div className="space-y-1">
          <p className="text-[10px] uppercase tracking-wide text-muted-foreground">
            Last {sparkValues.length}d realized (annualized %)
          </p>
          <Sparkline
            values={sparkValues}
            tone="primary"
            height={40}
            labels={forecast.history_dates}
            formatTooltip={(value, label) =>
              label ? `${label}: ${value.toFixed(1)}%` : `${value.toFixed(1)}%`
            }
          />
        </div>
      </CardContent>
    </Card>
  );
}
