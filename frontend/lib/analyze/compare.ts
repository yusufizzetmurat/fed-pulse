import type { AnalyzeResult, HistoryDetail, Stance } from "./types";

export interface CompareDelta {
  closeAbsolute: number | null;
  closePercent: number | null;
  volatilityAbsolute: number | null;
  stanceShift: "more_hawkish" | "more_dovish" | "unchanged" | "unknown";
  scoreDelta: number | null;
}

function asResult(detail: HistoryDetail): AnalyzeResult {
  return (detail.payload || {}) as AnalyzeResult;
}

function stanceRank(value: Stance | string | undefined): number | null {
  switch ((value ?? "").toLowerCase()) {
    case "hawkish":
      return 1;
    case "neutral":
      return 0;
    case "dovish":
      return -1;
    default:
      return null;
  }
}

export function computeCompareDelta(a: HistoryDetail, b: HistoryDetail): CompareDelta {
  const ra = asResult(a);
  const rb = asResult(b);

  const closeA = ra.prediction?.close ?? null;
  const closeB = rb.prediction?.close ?? null;
  const volA = ra.prediction?.volatility ?? null;
  const volB = rb.prediction?.volatility ?? null;
  const stanceA = stanceRank((ra.sentiment?.label ?? a.stance) as string | undefined);
  const stanceB = stanceRank((rb.sentiment?.label ?? b.stance) as string | undefined);
  const scoreA = ra.sentiment?.score ?? a.sentiment_score ?? null;
  const scoreB = rb.sentiment?.score ?? b.sentiment_score ?? null;

  let stanceShift: CompareDelta["stanceShift"] = "unknown";
  if (stanceA != null && stanceB != null) {
    if (stanceA > stanceB) stanceShift = "more_hawkish";
    else if (stanceA < stanceB) stanceShift = "more_dovish";
    else stanceShift = "unchanged";
  }

  const closeAbsolute = closeA != null && closeB != null ? closeA - closeB : null;
  const closePercent =
    closeAbsolute != null && closeB != null && closeB !== 0
      ? (closeAbsolute / closeB) * 100
      : null;
  const volatilityAbsolute = volA != null && volB != null ? volA - volB : null;
  const scoreDelta = scoreA != null && scoreB != null ? scoreA - scoreB : null;

  return {
    closeAbsolute,
    closePercent,
    volatilityAbsolute,
    stanceShift,
    scoreDelta,
  };
}

export function describeStanceShift(shift: CompareDelta["stanceShift"]): string {
  switch (shift) {
    case "more_hawkish":
      return "A shifts hawkish vs. B";
    case "more_dovish":
      return "A shifts dovish vs. B";
    case "unchanged":
      return "Stance unchanged";
    default:
      return "Stance unknown";
  }
}
