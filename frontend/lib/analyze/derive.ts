import { normalizeTimestamp, toNumericOrNull } from "./format";
import type { AnalyzeResult, ChunkAttentionResponse } from "./types";

export interface ChartRow {
  timestamp: string;
  history: number | null;
  forecast: number | null;
  forecastLower: number | null;
  forecastUpper: number | null;
  forecastBand: number | null;
  realized: number | null;
}

export interface ErrorMetrics {
  mape: number | null;
  rmse: number | null;
}

export interface ErrorBundle {
  close: ErrorMetrics;
  vol: ErrorMetrics;
  hasRealized: boolean;
}

export interface BandCheck {
  withinBand: boolean;
  lower: number;
  upper: number;
  current?: number;
  realizedValue?: number;
  distance: number;
  distancePct: number;
  tone: "good" | "caution" | "danger";
}

type SeriesKind = "close" | "volatility";

function buildSeries(result: AnalyzeResult | null, kind: SeriesKind): ChartRow[] {
  const series = result?.series;
  if (!series) return [];

  const histKey = kind === "close" ? "history_close" : "history_volatility";
  const fcKey = kind === "close" ? "forecast_close" : "forecast_volatility";
  const fcLowerKey =
    kind === "close" ? "forecast_close_lower" : "forecast_volatility_lower";
  const fcUpperKey =
    kind === "close" ? "forecast_close_upper" : "forecast_volatility_upper";
  const realizedKey = kind === "close" ? "realized_close" : "realized_volatility";

  const byTs = new Map<string, ChartRow>();
  const ensureRow = (raw: unknown): ChartRow | null => {
    const ts = normalizeTimestamp(raw);
    if (!ts) return null;
    let row = byTs.get(ts);
    if (!row) {
      row = {
        timestamp: ts,
        history: null,
        forecast: null,
        forecastLower: null,
        forecastUpper: null,
        forecastBand: null,
        realized: null,
      };
      byTs.set(ts, row);
    }
    return row;
  };

  (series.timestamps || []).forEach((ts, idx) => {
    const row = ensureRow(ts);
    if (row) row.history = toNumericOrNull((series as Record<string, number[] | undefined>)[histKey]?.[idx]);
  });
  (series.forecast_timestamps || []).forEach((ts, idx) => {
    const row = ensureRow(ts);
    if (!row) return;
    const fc = toNumericOrNull((series as Record<string, number[] | undefined>)[fcKey]?.[idx]);
    const lo = toNumericOrNull((series as Record<string, number[] | undefined>)[fcLowerKey]?.[idx]);
    const hi = toNumericOrNull((series as Record<string, number[] | undefined>)[fcUpperKey]?.[idx]);
    row.forecast = fc;
    row.forecastLower = lo;
    row.forecastUpper = hi;
    row.forecastBand = lo != null && hi != null ? Math.max(hi - lo, 0) : null;
  });
  (series.realized_timestamps || []).forEach((ts, idx) => {
    const row = ensureRow(ts);
    if (row) row.realized = toNumericOrNull((series as Record<string, number[] | undefined>)[realizedKey]?.[idx]);
  });

  return Array.from(byTs.values()).sort((a, b) => a.timestamp.localeCompare(b.timestamp));
}

export function buildCloseSeries(result: AnalyzeResult | null): ChartRow[] {
  return buildSeries(result, "close");
}

export function buildVolatilitySeries(result: AnalyzeResult | null): ChartRow[] {
  return buildSeries(result, "volatility");
}

export function computeErrorMetrics(result: AnalyzeResult | null): ErrorBundle {
  const series = result?.series;
  const pair = (
    fcTs?: string[],
    fcVals?: number[],
    rzTs?: string[],
    rzVals?: number[]
  ): Array<[number, number]> => {
    const map = new Map<string, number>();
    (fcTs || []).forEach((ts, idx) => {
      const key = normalizeTimestamp(ts);
      const value = toNumericOrNull(fcVals?.[idx]);
      if (key && value != null) map.set(key, value);
    });
    const pairs: Array<[number, number]> = [];
    (rzTs || []).forEach((ts, idx) => {
      const key = normalizeTimestamp(ts);
      const realized = toNumericOrNull(rzVals?.[idx]);
      const forecast = key ? map.get(key) : null;
      if (forecast != null && realized != null) pairs.push([forecast, realized]);
    });
    return pairs;
  };

  const closePairs = pair(
    series?.forecast_timestamps,
    series?.forecast_close,
    series?.realized_timestamps,
    series?.realized_close
  );
  const volPairs = pair(
    series?.forecast_timestamps,
    series?.forecast_volatility,
    series?.realized_timestamps,
    series?.realized_volatility
  );

  const calc = (pairs: Array<[number, number]>): ErrorMetrics => {
    if (!pairs.length) return { mape: null, rmse: null };
    const mapeVals = pairs
      .filter(([, actual]) => Math.abs(actual) > 1e-12)
      .map(([pred, actual]) => Math.abs((actual - pred) / actual));
    const mse = pairs.reduce((acc, [pred, actual]) => {
      const err = actual - pred;
      return acc + err * err;
    }, 0) / pairs.length;
    const rmse = Math.sqrt(mse) || 0;
    const mape = mapeVals.length
      ? (mapeVals.reduce((acc, v) => acc + v, 0) / mapeVals.length) * 100
      : null;
    return { mape, rmse };
  };

  return {
    close: calc(closePairs),
    vol: calc(volPairs),
    hasRealized: Boolean(closePairs.length || volPairs.length),
  };
}

export function computeCurrentSpotBandCheck(result: AnalyzeResult | null): BandCheck | null {
  const series = result?.series;
  const lower = toNumericOrNull(series?.forecast_close_lower?.[series.forecast_close_lower.length - 1]);
  const upper = toNumericOrNull(series?.forecast_close_upper?.[series.forecast_close_upper.length - 1]);
  const current = toNumericOrNull(result?.market?.close);
  if (lower == null || upper == null || current == null) return null;
  const withinBand = current >= lower && current <= upper;
  const distance = withinBand ? 0 : current < lower ? lower - current : current - upper;
  const reference = Math.max(Math.abs(current), 1e-6);
  const distancePct = (distance / reference) * 100;
  return {
    withinBand,
    lower,
    upper,
    current,
    distance,
    distancePct,
    tone: withinBand ? "good" : distance / reference <= 0.015 ? "caution" : "danger",
  };
}

export function computeRealizedBandCheck(result: AnalyzeResult | null): BandCheck | null {
  const series = result?.series;
  const forecastTs = series?.forecast_timestamps || [];
  const lowerBand = series?.forecast_close_lower || [];
  const upperBand = series?.forecast_close_upper || [];
  const realizedTs = series?.realized_timestamps || [];
  const realizedClose = series?.realized_close || [];

  const byTs = new Map<string, { lower: number; upper: number }>();
  forecastTs.forEach((ts, idx) => {
    const key = normalizeTimestamp(ts);
    const lo = toNumericOrNull(lowerBand[idx]);
    const hi = toNumericOrNull(upperBand[idx]);
    if (key && lo != null && hi != null) byTs.set(key, { lower: lo, upper: hi });
  });

  const overlaps: Array<{ ts: string; realizedValue: number; lower: number; upper: number }> = [];
  realizedTs.forEach((ts, idx) => {
    const key = normalizeTimestamp(ts);
    const realizedValue = toNumericOrNull(realizedClose[idx]);
    const band = key ? byTs.get(key) : null;
    if (key && band && realizedValue != null) {
      overlaps.push({ ts: key, realizedValue, lower: band.lower, upper: band.upper });
    }
  });
  if (!overlaps.length) return null;
  overlaps.sort((a, b) => a.ts.localeCompare(b.ts));
  const latest = overlaps[overlaps.length - 1];
  const { realizedValue, lower, upper } = latest;
  const withinBand = realizedValue >= lower && realizedValue <= upper;
  const distance = withinBand ? 0 : realizedValue < lower ? lower - realizedValue : realizedValue - upper;
  const reference = Math.max(Math.abs(realizedValue), 1e-6);
  const distancePct = (distance / reference) * 100;
  return {
    withinBand,
    lower,
    upper,
    realizedValue,
    distance,
    distancePct,
    tone: withinBand ? "good" : distance / reference <= 0.02 ? "caution" : "danger",
  };
}

export interface AttentionRow {
  index: number;
  label: string;
  preview: string;
  weight: number;
  decay: number;
  weightPct: string;
}

export interface AttentionBundle {
  rows: AttentionRow[];
  lambdaValue: number;
  chunkCount: number;
}

export function buildAttention(result: AnalyzeResult | null): AttentionBundle | null {
  const attention: ChunkAttentionResponse | null | undefined = result?.model?.chunk_attention;
  if (!attention) return null;
  const weights = Array.isArray(attention.weights) ? attention.weights : [];
  if (!weights.length) return null;
  const decay = Array.isArray(attention.decay_coeffs) ? attention.decay_coeffs : [];
  const previews = Array.isArray(attention.chunk_previews) ? attention.chunk_previews : [];
  const rows: AttentionRow[] = weights.map((weight, idx) => {
    const w = Number(weight) || 0;
    return {
      index: idx,
      label: previews[idx] ? previews[idx].slice(0, 80) : `chunk ${idx}`,
      preview: previews[idx] || "",
      weight: w,
      decay: Number(decay[idx]) || 0,
      weightPct: (w * 100).toFixed(1),
    };
  });
  return {
    rows,
    lambdaValue: Number(attention.lambda_value) || 0,
    chunkCount: attention.chunk_count ?? rows.length,
  };
}
