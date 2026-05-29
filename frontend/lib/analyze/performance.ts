import type { HistoryEntry, HistoryRealizedResponse } from "./types";

export const REGIME_CLASSES = ["calm", "normal", "high"] as const;
export type RegimeClass = (typeof REGIME_CLASSES)[number];

export interface RunRegimeRecord {
  id: string;
  symbol: string;
  document_date: string;
  horizon: string;
  argmax: string | null;
  argmaxProbability: number | null;
  setSize: number | null;
  realized: string | null;
  // Realised regime was inside the calibrated prediction set; null when
  // either the predicted set or the realised regime is missing.
  setHit: boolean | null;
}

export interface ClassMetrics {
  klass: string;
  support: number;
  precision: number | null;
  recall: number | null;
  f1: number | null;
}

export interface ConfusionRow {
  truth: string;
  counts: Record<string, number>;
  total: number;
}

export interface PerformanceAggregate {
  total: number;
  resolved: number;
  argmaxAccuracy: number | null;
  macroF1: number | null;
  empiricalCoverage: number | null;
  perClass: ClassMetrics[];
  confusion: ConfusionRow[];
  bySymbol: Array<{
    symbol: string;
    resolved: number;
    argmaxAccuracy: number | null;
    empiricalCoverage: number | null;
  }>;
}

function safeDivide(num: number, den: number): number | null {
  return den > 0 ? num / den : null;
}

function readPredictedSet(payload: Record<string, unknown> | undefined): string[] | null {
  const regime =
    payload && typeof payload === "object"
      ? (payload as { regime_classification?: { predicted_set?: unknown } }).regime_classification
      : null;
  if (!regime || typeof regime !== "object") return null;
  const set = (regime as { predicted_set?: unknown }).predicted_set;
  if (!Array.isArray(set)) return null;
  return set.filter((label): label is string => typeof label === "string");
}

export interface RegimeInputs {
  entry: HistoryEntry;
  realized: HistoryRealizedResponse | null;
  // Optional payload from /history/{id}; lets the set-hit math compute
  // off the persisted predicted_set when present.
  payload?: Record<string, unknown> | null;
}

export function buildRunRegimeRecord({ entry, realized, payload }: RegimeInputs): RunRegimeRecord {
  const argmax = entry.argmax_regime ?? null;
  const realizedLabel = realized?.realized_regime ?? null;
  const predictedSet = readPredictedSet(payload ?? undefined);
  let setHit: boolean | null = null;
  if (realizedLabel && predictedSet) {
    setHit = predictedSet.includes(realizedLabel);
  }
  return {
    id: entry.id,
    symbol: entry.symbol,
    document_date: entry.document_date,
    horizon: entry.horizon,
    argmax,
    argmaxProbability: entry.argmax_probability ?? null,
    setSize: entry.regime_set_size ?? null,
    realized: realizedLabel,
    setHit,
  };
}

function emptyConfusion(): ConfusionRow[] {
  return REGIME_CLASSES.map((truth) => ({
    truth,
    counts: Object.fromEntries(REGIME_CLASSES.map((klass) => [klass, 0])) as Record<string, number>,
    total: 0,
  }));
}

export function aggregateRegimePerformance(rows: RunRegimeRecord[]): PerformanceAggregate {
  const resolved = rows.filter((row) => row.argmax != null && row.realized != null);

  // Confusion + per-class counts. We restrict to the canonical
  // REGIME_CLASSES; an unexpected label (legacy run, hand-stitched
  // checkpoint) lands in the totals but not the per-class breakdown.
  const confusion = emptyConfusion();
  const tp: Record<string, number> = { calm: 0, normal: 0, high: 0 };
  const fp: Record<string, number> = { calm: 0, normal: 0, high: 0 };
  const fn: Record<string, number> = { calm: 0, normal: 0, high: 0 };
  const support: Record<string, number> = { calm: 0, normal: 0, high: 0 };

  for (const row of resolved) {
    const truth = row.realized as RegimeClass;
    const pred = row.argmax as RegimeClass;
    const knownTruth = (REGIME_CLASSES as readonly string[]).includes(truth);
    const knownPred = (REGIME_CLASSES as readonly string[]).includes(pred);
    if (knownTruth) {
      support[truth] += 1;
    }
    // Only count rows where both axes carry canonical labels in the
    // confusion matrix. Off-axis legacy predictions ('unknown', LABEL_2)
    // are excluded entirely so the support number and the row-sum match
    // — readers can then trust the diagonal share as a real accuracy.
    if (knownTruth && knownPred) {
      const r = confusion.find((c) => c.truth === truth)!;
      r.counts[pred] = (r.counts[pred] ?? 0) + 1;
      r.total += 1;
      if (truth === pred) {
        tp[truth] += 1;
      } else {
        fn[truth] += 1;
        fp[pred] += 1;
      }
    }
  }

  const perClass: ClassMetrics[] = REGIME_CLASSES.map((klass) => {
    const precision = safeDivide(tp[klass], tp[klass] + fp[klass]);
    const recall = safeDivide(tp[klass], tp[klass] + fn[klass]);
    const f1 =
      precision != null && recall != null && precision + recall > 0
        ? (2 * precision * recall) / (precision + recall)
        : null;
    return { klass, support: support[klass], precision, recall, f1 };
  });

  const f1Values = perClass
    .map((entry) => entry.f1)
    .filter((value): value is number => value != null);
  const macroF1 = f1Values.length === REGIME_CLASSES.length ? mean(f1Values) : null;

  const argmaxCorrect = resolved.filter((row) => row.argmax === row.realized).length;
  const argmaxAccuracy = resolved.length > 0 ? argmaxCorrect / resolved.length : null;

  const setEligible = rows.filter((row) => row.setHit != null);
  const setHits = setEligible.filter((row) => row.setHit === true).length;
  const empiricalCoverage = setEligible.length > 0 ? setHits / setEligible.length : null;

  // Per-symbol breakdown — same accuracy / coverage math limited to
  // the rows that carry the symbol so an analyst can tell which assets
  // are pulling the headline up or down.
  const bySymbolMap = new Map<string, RunRegimeRecord[]>();
  for (const row of rows) {
    const list = bySymbolMap.get(row.symbol) ?? [];
    list.push(row);
    bySymbolMap.set(row.symbol, list);
  }
  const bySymbol = [...bySymbolMap.entries()].map(([symbol, group]) => {
    const groupResolved = group.filter((row) => row.argmax != null && row.realized != null);
    const groupCorrect = groupResolved.filter((row) => row.argmax === row.realized).length;
    const groupEligible = group.filter((row) => row.setHit != null);
    const groupHits = groupEligible.filter((row) => row.setHit === true).length;
    return {
      symbol,
      resolved: groupResolved.length,
      argmaxAccuracy: groupResolved.length > 0 ? groupCorrect / groupResolved.length : null,
      empiricalCoverage: groupEligible.length > 0 ? groupHits / groupEligible.length : null,
    };
  });
  bySymbol.sort((a, b) => b.resolved - a.resolved);

  return {
    total: rows.length,
    resolved: resolved.length,
    argmaxAccuracy,
    macroF1,
    empiricalCoverage,
    perClass,
    confusion,
    bySymbol,
  };
}

function mean(values: number[]): number {
  if (values.length === 0) return 0;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

// Wald-style 95% half-width for a proportion p over a support of n.
// Returns null when n is too small for the normal approximation to be
// meaningful or when p is degenerate (variance ≤ 0). Non-finite inputs
// also return null so the caller's "—" branch fires.
export function proportionHalfWidth(p: number | null, n: number): number | null {
  if (p == null || !Number.isFinite(p) || !Number.isFinite(n) || n < 5) return null;
  const variance = p * (1 - p);
  if (variance <= 0) return null;
  return 1.96 * Math.sqrt(variance / n);
}
