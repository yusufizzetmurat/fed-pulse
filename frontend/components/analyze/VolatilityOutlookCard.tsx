import * as React from "react";
import { Activity } from "lucide-react";
import {
  Area,
  ComposedChart,
  Line,
  ResponsiveContainer,
  Tooltip,
  YAxis,
} from "recharts";

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
  RealizedVolHistoricalBand,
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

function RealizedFeatureSourceBadge({
  source,
  date,
}: {
  source: RealizedVolForecastResponse["realized_features_source"] | undefined;
  date: RealizedVolForecastResponse["realized_features_date"] | undefined;
}) {
  if (source === "live") {
    const tooltip = date
      ? `Live intraday realized measures (rs_pos/rs_neg/bv/rq/rskew/rkurt/parkinson/log_rvol) from ${date} feed the QLIKE-DLq head this request. Full edge served.`
      : "Live intraday realized measures feed the QLIKE-DLq head this request. Full edge served.";
    return (
      <Badge
        variant="outline"
        className="text-[10px] uppercase tracking-wide text-emerald-700 border-emerald-700/40"
        title={tooltip}
        data-testid="rv-source-live"
      >
        QLIKE-full{date ? ` · ${date}` : ""}
      </Badge>
    );
  }
  return (
    <Badge
      variant="outline"
      className="text-[10px] uppercase tracking-wide text-amber-700 border-amber-700/40"
      title="Intraday 5m bars unavailable for this symbol. The QLIKE-DLq head falls back to training-set means; the forecast collapses to HAR-grade. The ~10% QLIKE-over-HAR edge does not apply here."
      data-testid="rv-source-fallback"
    >
      HAR-fallback
    </Badge>
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

  const bandRows = forecast.historical_bands ?? [];
  const hasBands = bandRows.length > 0;

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardDescription className="flex items-center gap-1.5">
          <Activity className="h-3.5 w-3.5" /> Volatility Outlook
        </CardDescription>
        <CardTitle className="flex items-center justify-between text-base">
          <span>QLIKE-DLq ensemble · {forecast.symbol}</span>
          <div className="flex items-center gap-1.5">
            <RealizedFeatureSourceBadge
              source={forecast.realized_features_source}
              date={forecast.realized_features_date}
            />
            <Badge variant="outline" className="text-[10px] uppercase tracking-wide">
              Market readout
            </Badge>
          </div>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <p className="text-[11px] text-muted-foreground">
          {forecast.realized_features_source === "live"
            ? "Annualized realized volatility, ensemble mean with conformal bands. Live intraday realized measures feed the QLIKE head."
            : "Annualized realized volatility, ensemble mean with conformal bands. Intraday measures unavailable; the head falls back to HAR-grade."}
        </p>
        <div className="grid gap-2 md:grid-cols-3">
          {forecast.horizons.map((h) => (
            <HorizonColumn key={h.h} horizon={h} />
          ))}
        </div>
        <div className="space-y-1">
          <div className="flex items-center justify-between gap-2">
            <p className="text-[10px] uppercase tracking-wide text-muted-foreground">
              Last {sparkValues.length}d realized (annualized %)
            </p>
            {hasBands ? (
              <span
                className="flex items-center gap-1 text-[10px] text-muted-foreground"
                data-testid="rv-bands-legend"
              >
                <span
                  aria-hidden="true"
                  className="inline-block h-2 w-3 rounded-sm"
                  style={{ background: "hsl(var(--primary) / 0.18)" }}
                />
                Past 80% bands
              </span>
            ) : null}
          </div>
          {hasBands ? (
            <RealizedHistoryChart
              values={sparkValues}
              labels={forecast.history_dates}
              bands={bandRows}
            />
          ) : (
            <Sparkline
              values={sparkValues}
              tone="primary"
              height={40}
              labels={forecast.history_dates}
              formatTooltip={(value, label) =>
                label ? `${label}: ${value.toFixed(1)}%` : `${value.toFixed(1)}%`
              }
            />
          )}
        </div>
      </CardContent>
    </Card>
  );
}

function annualizeRvPct(rv: number | null | undefined): number | null {
  if (rv == null || !Number.isFinite(rv) || rv <= 0) return null;
  return Math.sqrt(rv * 252) * 100;
}

interface RealizedHistoryChartProps {
  values: Array<number | null>;
  labels: string[];
  bands: RealizedVolHistoricalBand[];
}

function RealizedHistoryChart({
  values,
  labels,
  bands,
}: RealizedHistoryChartProps) {
  // Index bands by date so we can left-join onto the sparkline series
  // and render the bands at exactly the dates the prediction covered.
  const bandByDate = React.useMemo(() => {
    const map = new Map<string, RealizedVolHistoricalBand>();
    for (const row of bands) map.set(row.date, row);
    return map;
  }, [bands]);

  const data = React.useMemo(() => {
    return labels.map((label, index) => {
      const band = bandByDate.get(label);
      const lo = band ? annualizeRvPct(band.band_lo_80) : null;
      const hi = band ? annualizeRvPct(band.band_hi_80) : null;
      return {
        x: index,
        label,
        value: values[index] ?? null,
        bandLo: lo,
        // The Area renders bandLo + bandRange stacked, so the visible
        // band spans [bandLo, bandLo + bandRange] = [bandLo, bandHi].
        bandRange: lo != null && hi != null ? hi - lo : null,
        bandHi: hi,
      };
    });
  }, [labels, values, bandByDate]);

  // Recompute the chart's vertical extent so band_lo_80 cannot get
  // clipped at the bottom. Recharts otherwise auto-scales to the
  // dominant series (the realized line) and crops the muted band tails.
  const domain = React.useMemo<[number, number]>(() => {
    const numeric: number[] = [];
    for (const row of data) {
      if (row.value != null && Number.isFinite(row.value)) numeric.push(row.value);
      if (row.bandLo != null && Number.isFinite(row.bandLo)) numeric.push(row.bandLo);
      if (row.bandHi != null && Number.isFinite(row.bandHi)) numeric.push(row.bandHi);
    }
    if (numeric.length === 0) return [0, 1];
    const min = Math.min(...numeric);
    const max = Math.max(...numeric);
    const pad = Math.max(0.5, (max - min) * 0.08);
    return [Math.max(0, min - pad), max + pad];
  }, [data]);

  const stroke = "hsl(var(--primary))";
  const bandFill = "hsl(var(--primary) / 0.18)";
  const transparent = "rgba(0,0,0,0)";

  return (
    <div className="w-full" style={{ height: 56, minHeight: 56 }}>
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart
          data={data}
          margin={{ top: 4, right: 2, bottom: 2, left: 2 }}
        >
          <YAxis hide domain={domain} />
          <Tooltip
            cursor={{ stroke, strokeWidth: 0.75, strokeDasharray: "2 2" }}
            content={({ active, payload }) => {
              if (!active || !payload?.length) return null;
              const point = payload[0].payload as (typeof data)[number];
              const lines: string[] = [];
              if (point.value != null) lines.push(`Realized: ${point.value.toFixed(1)}%`);
              if (point.bandLo != null && point.bandHi != null) {
                lines.push(
                  `80% band: ${point.bandLo.toFixed(1)}–${point.bandHi.toFixed(1)}%`,
                );
              }
              if (lines.length === 0) return null;
              return (
                <div className="rounded-md border border-border bg-popover px-2 py-1 text-[11px] shadow-md">
                  {point.label ? (
                    <p className="text-[10px] text-muted-foreground">{point.label}</p>
                  ) : null}
                  {lines.map((line) => (
                    <p key={line} className="numeric font-medium">
                      {line}
                    </p>
                  ))}
                </div>
              );
            }}
          />
          <Area
            type="monotone"
            dataKey="bandLo"
            stackId="band"
            stroke={transparent}
            fill={transparent}
            isAnimationActive={false}
            dot={false}
            activeDot={false}
            connectNulls={false}
            // Invisible base of the stack so band_lo lifts the band area to
            // its bottom edge without painting a line.
            data-testid="rv-band-base"
          />
          <Area
            type="monotone"
            dataKey="bandRange"
            stackId="band"
            stroke={transparent}
            fill={bandFill}
            isAnimationActive={false}
            dot={false}
            activeDot={false}
            connectNulls={false}
            data-testid="rv-band-range"
          />
          <Line
            type="monotone"
            dataKey="value"
            stroke={stroke}
            strokeWidth={1.5}
            isAnimationActive={false}
            dot={false}
            activeDot={{ r: 2, stroke, fill: stroke }}
            connectNulls
            data-testid="rv-realized-line"
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
