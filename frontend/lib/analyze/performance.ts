import type { HistoryEntry } from "./types";

export interface RunPerformance {
  id: string;
  symbol: string;
  document_date: string;
  horizon: string;
  predicted_close: number | null;
  spot_close: number | null;
  realized_close: number | null;
  direction_correct: boolean | null;
  absolute_error: number | null;
  percent_error: number | null;
}

export interface PerformanceAggregate {
  total: number;
  resolved: number;
  hitRate: number | null;
  mape: number | null;
  mae: number | null;
  bySymbol: Array<{
    symbol: string;
    resolved: number;
    hitRate: number | null;
    mape: number | null;
  }>;
}

export function computeRunPerformance(
  row: HistoryEntry,
  realizedClose: number | null
): RunPerformance {
  const predicted = row.predicted_close ?? null;
  const spot = row.current_close ?? null;

  let directionCorrect: boolean | null = null;
  if (predicted != null && spot != null && realizedClose != null) {
    const predictedDir = Math.sign(predicted - spot);
    const realizedDir = Math.sign(realizedClose - spot);
    directionCorrect = predictedDir !== 0 && predictedDir === realizedDir;
  }

  let absoluteError: number | null = null;
  let percentError: number | null = null;
  if (predicted != null && realizedClose != null) {
    absoluteError = Math.abs(predicted - realizedClose);
    if (Math.abs(realizedClose) > 1e-9) {
      percentError = absoluteError / Math.abs(realizedClose);
    }
  }

  return {
    id: row.id,
    symbol: row.symbol,
    document_date: row.document_date,
    horizon: row.horizon,
    predicted_close: predicted,
    spot_close: spot,
    realized_close: realizedClose,
    direction_correct: directionCorrect,
    absolute_error: absoluteError,
    percent_error: percentError,
  };
}

export function aggregatePerformance(rows: RunPerformance[]): PerformanceAggregate {
  const resolved = rows.filter((row) => row.realized_close != null);
  const directional = resolved.filter((row) => row.direction_correct != null);
  const hits = directional.filter((row) => row.direction_correct === true).length;

  const mapeValues = resolved
    .map((row) => row.percent_error)
    .filter((value): value is number => value != null);
  const maeValues = resolved
    .map((row) => row.absolute_error)
    .filter((value): value is number => value != null);

  const bySymbolMap = new Map<string, RunPerformance[]>();
  for (const row of rows) {
    const list = bySymbolMap.get(row.symbol) ?? [];
    list.push(row);
    bySymbolMap.set(row.symbol, list);
  }
  const bySymbol = [...bySymbolMap.entries()].map(([symbol, group]) => {
    const groupResolved = group.filter((row) => row.realized_close != null);
    const groupDirectional = groupResolved.filter((row) => row.direction_correct != null);
    const groupHits = groupDirectional.filter((row) => row.direction_correct === true).length;
    const groupMape = groupResolved
      .map((row) => row.percent_error)
      .filter((value): value is number => value != null);
    return {
      symbol,
      resolved: groupResolved.length,
      hitRate: groupDirectional.length > 0 ? groupHits / groupDirectional.length : null,
      mape: groupMape.length > 0 ? mean(groupMape) : null,
    };
  });

  return {
    total: rows.length,
    resolved: resolved.length,
    hitRate: directional.length > 0 ? hits / directional.length : null,
    mape: mapeValues.length > 0 ? mean(mapeValues) : null,
    mae: maeValues.length > 0 ? mean(maeValues) : null,
    bySymbol: bySymbol.sort((a, b) => b.resolved - a.resolved),
  };
}

function mean(values: number[]): number {
  if (values.length === 0) return 0;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}
