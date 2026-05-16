import type { Stance } from "./types";

export function toNumericOrNull(value: unknown): number | null {
  if (value == null || value === "") return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

export function normalizeTimestamp(value: unknown): string {
  if (!value) return "";
  return String(value).split("+")[0];
}

// Strict mapping. The dashboard refuses to silently relabel a non-stance
// label (e.g. POSITIVE / NEGATIVE from distilbert-sst-2, which is what
// loads when the FOMC sentiment model fails) into a monetary-policy
// stance.
//
// Earlier versions mapped POSITIVE -> hawkish / NEGATIVE -> dovish /
// LABEL_1 -> hawkish blindly, which caused "interest rates are bad
// inflation is good" to display as "Hawkish 0.99" — the sst-2 fallback
// was reading "good" as positive news sentiment and the UI laundered
// that into a monetary-policy stance.
//
// LABEL_0 / LABEL_1 / LABEL_2 are still mapped because a checkpoint
// without an id2label config legitimately exposes those generic strings;
// the order matches our taxonomy (0=dovish, 1=neutral, 2=hawkish).
export function toStance(label: unknown): Stance {
  const value = String(label || "").trim();
  const lower = value.toLowerCase();
  if (lower === "hawkish") return "hawkish";
  if (lower === "dovish") return "dovish";
  if (lower === "neutral") return "neutral";
  if (lower.startsWith("hawk")) return "hawkish";
  if (lower.startsWith("dove")) return "dovish";
  if (value === "LABEL_0") return "dovish";
  if (value === "LABEL_1") return "neutral";
  if (value === "LABEL_2") return "hawkish";
  return "unknown";
}

export function stanceLabel(stance: Stance): string {
  if (stance === "hawkish") return "Hawkish";
  if (stance === "dovish") return "Dovish";
  if (stance === "neutral") return "Neutral";
  return "Unknown";
}

export function formatPrice(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return "N/A";
  return `$${Number(value).toLocaleString("en-US", { maximumFractionDigits: 2 })}`;
}

export function formatPriceDelta(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return "N/A";
  const v = Number(value);
  const sign = v >= 0 ? "+" : "-";
  return `${sign}$${Math.abs(v).toLocaleString("en-US", { maximumFractionDigits: 2 })}`;
}

export function formatPercentDelta(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return "N/A";
  const v = Number(value);
  return `${v >= 0 ? "+" : ""}${v.toFixed(2)}%`;
}

export function formatVol(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return "N/A";
  return `${(Number(value) * 100).toFixed(2)}%`;
}

export function formatDateTick(value: unknown): string {
  if (!value) return "";
  const clean = String(value).split("+")[0];
  const dateValue = new Date(clean);
  if (Number.isNaN(dateValue.getTime())) return String(value);
  return dateValue.toLocaleDateString("en-US", { month: "short", day: "numeric" });
}

export type ErrorTone = "low" | "medium" | "high" | "neutral";

export function getErrorTone(
  kind: "mape" | "rmse",
  value: number | null | undefined,
  baseline = 0
): ErrorTone {
  if (value == null || !Number.isFinite(value)) return "neutral";
  if (kind === "mape") {
    if (value <= 2) return "low";
    if (value <= 5) return "medium";
    return "high";
  }
  const normalized = baseline > 0 ? (value / baseline) * 100 : value;
  if (normalized <= 1) return "low";
  if (normalized <= 2.5) return "medium";
  return "high";
}

export function errorToneLabel(tone: ErrorTone): string {
  if (tone === "low") return "Low error";
  if (tone === "medium") return "Medium error";
  if (tone === "high") return "High error";
  return "Awaiting data";
}

// Label that goes next to the forecast band so the reader can tell which
// methodology produced it. The backend marks the band source in
// `series.forecast_band_source`:
//   - "conformal"   the band came from the conformal-prediction manifest
//                   (empirical coverage, calibrated against holdout residuals)
//   - "gaussian_z"  the band is the Gaussian-z fallback (volatility times the
//                   80th-percentile z-score, applied symmetrically)
//   - null/missing  the response did not carry a source — usually an older
//                   history payload from before the band-source field
//                   existed. We fall back to "confidence band" rather than
//                   guessing.
//
// Confidence level is the integer percent (e.g. 80) the chart already
// shows; this helper just decorates it with the methodology.
export function bandLabel(
  confidenceLevel: number,
  source: "conformal" | "gaussian_z" | null | undefined,
): string {
  if (source === "conformal") return `${confidenceLevel}% conformal band`;
  if (source === "gaussian_z") return `${confidenceLevel}% Gaussian-z band`;
  return `${confidenceLevel}% confidence band`;
}
