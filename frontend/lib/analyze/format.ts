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

export function toStance(label: unknown): Stance {
  const value = String(label || "").toUpperCase();
  if (value.includes("HAWK") || value.includes("POSITIVE") || value.includes("LABEL_1")) {
    return "hawkish";
  }
  if (value.includes("DOVE") || value.includes("NEGATIVE") || value.includes("LABEL_0")) {
    return "dovish";
  }
  if (value.includes("NEUTRAL") || value.includes("LABEL_2")) {
    return "neutral";
  }
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
