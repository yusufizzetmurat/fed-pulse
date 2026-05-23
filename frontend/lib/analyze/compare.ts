import type {
  AnalyzeResult,
  HistoryDetail,
  MultiAxisResponse,
  Stance,
  StanceAxis,
} from "./types";

export interface CompareDelta {
  closeAbsolute: number | null;
  closePercent: number | null;
  volatilityAbsolute: number | null;
  stanceShift: "more_hawkish" | "more_dovish" | "unchanged" | "unknown";
  scoreDelta: number | null;
}

// Per-axis deltas for the multi-axis schema (stance / factor / certainty /
// topic). Each delta is A − B; null means at least one side is missing the
// axis. Stance delta is the same hawkish=+1 / neutral=0 / dovish=−1 ranking
// the headline `stanceShift` uses, but kept as a numeric value so the UI can
// render the magnitude alongside the direction.
export interface MultiAxisDelta {
  stanceRankDelta: number | null;
  stanceConfidenceDelta: number | null;
  factorDelta: number | null;
  factorConfidenceDelta: number | null;
  certaintyConfidenceDelta: number | null;
  certaintyShift:
    | "more_decisive"
    | "more_tentative"
    | "unchanged"
    | "unknown";
  topicChanged: boolean | null;
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

// Certainty axis ranks "certain" highest and "uncertain" lowest so the
// shift label tracks confidence movement, not lexical similarity. The
// legacy fixture labels ("decisive" / "measured" / "tentative") are
// kept here for back-compat with persisted history entries written
// before the canonical relabel landed.
const _CERTAINTY_RANK: Record<string, number> = {
  certain: 2,
  neutral: 1,
  uncertain: 0,
  decisive: 2,
  measured: 1,
  tentative: 0,
};

function _certaintyRank(label: string | undefined | null): number | null {
  if (!label) return null;
  const key = String(label).toLowerCase();
  return key in _CERTAINTY_RANK ? _CERTAINTY_RANK[key] : null;
}

function _multiAxis(detail: HistoryDetail): MultiAxisResponse | undefined {
  return ((detail.payload || {}) as AnalyzeResult).multi_axis;
}

export function computeMultiAxisDelta(
  a: HistoryDetail,
  b: HistoryDetail,
): MultiAxisDelta {
  const ma = _multiAxis(a);
  const mb = _multiAxis(b);

  const stanceA = ma?.stance ? stanceRank(ma.stance.label as StanceAxis) : null;
  const stanceB = mb?.stance ? stanceRank(mb.stance.label as StanceAxis) : null;
  const stanceRankDelta = stanceA != null && stanceB != null ? stanceA - stanceB : null;
  const stanceConfA = ma?.stance?.confidence ?? null;
  const stanceConfB = mb?.stance?.confidence ?? null;
  const stanceConfidenceDelta =
    stanceConfA != null && stanceConfB != null ? stanceConfA - stanceConfB : null;

  const factorA = ma?.factor?.value ?? null;
  const factorB = mb?.factor?.value ?? null;
  const factorDelta = factorA != null && factorB != null ? factorA - factorB : null;
  const factorConfA = ma?.factor?.confidence ?? null;
  const factorConfB = mb?.factor?.confidence ?? null;
  const factorConfidenceDelta =
    factorConfA != null && factorConfB != null ? factorConfA - factorConfB : null;

  const certA = _certaintyRank(ma?.certainty?.label);
  const certB = _certaintyRank(mb?.certainty?.label);
  let certaintyShift: MultiAxisDelta["certaintyShift"] = "unknown";
  if (certA != null && certB != null) {
    if (certA > certB) certaintyShift = "more_decisive";
    else if (certA < certB) certaintyShift = "more_tentative";
    else certaintyShift = "unchanged";
  }
  const certaintyConfidenceDelta =
    ma?.certainty?.confidence != null && mb?.certainty?.confidence != null
      ? ma.certainty.confidence - mb.certainty.confidence
      : null;

  const topicA = ma?.topic?.label ?? ma?.topic?.primary ?? null;
  const topicB = mb?.topic?.label ?? mb?.topic?.primary ?? null;
  const topicChanged =
    topicA != null && topicB != null ? topicA !== topicB : null;

  return {
    stanceRankDelta,
    stanceConfidenceDelta,
    factorDelta,
    factorConfidenceDelta,
    certaintyConfidenceDelta,
    certaintyShift,
    topicChanged,
  };
}
