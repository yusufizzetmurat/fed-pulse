// Shared number formatters for the workspace-spine bundle.
//
// The descriptive panels (MP-surprise, FRED consensus) speak in basis
// points; the forecast cards speak in log-residual / percent-vs-baseline
// units; both surfaces need consistent sign handling so a reader
// scanning the page sees "+12 bps" and "+12.3% vs baseline" in the same
// visual register. These helpers are intentionally pure and don't pull
// in any locale state — the dashboard is en-US only.

export interface FormatBpsOptions {
  /** When true (default) the formatter prepends "+" for non-negative
   *  values. Set false to drop the explicit positive sign while still
   *  honoring the conventional minus sign for negatives. */
  signed?: boolean;
  /** Number of fractional digits to render. Defaults to 0 — bps are
   *  whole numbers on the canonical surfaces. */
  fractionDigits?: number;
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

export function formatBps(
  bps: number | null | undefined,
  opts: FormatBpsOptions = {},
): string {
  if (!isFiniteNumber(bps)) return "N/A";
  const { signed = true, fractionDigits = 0 } = opts;
  const rounded = Number(bps.toFixed(fractionDigits));
  const magnitude = Math.abs(rounded).toFixed(fractionDigits);
  if (rounded === 0) {
    // Render an explicit "0 bps" without a sign regardless of `signed`.
    return `0 bps`;
  }
  let prefix = "";
  if (rounded < 0) {
    prefix = "-";
  } else if (signed) {
    prefix = "+";
  }
  return `${prefix}${magnitude} bps`;
}

export function formatPctVsBaseline(
  pct: number | null | undefined,
): string {
  if (!isFiniteNumber(pct)) return "N/A";
  const sign = pct >= 0 ? "+" : "";
  return `${sign}${pct.toFixed(1)}% vs baseline`;
}

export function formatLogResidual(value: number | null | undefined): string {
  if (!isFiniteNumber(value)) return "N/A";
  const sign = value >= 0 ? "+" : "";
  return `${sign}${value.toFixed(3)}`;
}

export function formatProbabilityPct(
  p: number | null | undefined,
): string {
  if (!isFiniteNumber(p)) return "N/A";
  // The probability inputs are in [0, 1]. Clamp before rendering so a
  // rounding overflow on the backend (e.g. 1.0000001) still reads as
  // 100% rather than 100.0001%.
  const clamped = Math.min(Math.max(p, 0), 1);
  return `${Math.round(clamped * 100)}%`;
}
