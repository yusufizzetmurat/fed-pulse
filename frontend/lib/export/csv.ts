// Minimal RFC-4180 CSV helpers shared by the per-run and compare exports.
//
// The dashboard does not pull in `papaparse` or `csv-stringify`; the rows
// it emits are small (≤ a few hundred), values are scalars, and the only
// quoting rule that actually matters is RFC-4180: wrap a field in double
// quotes when it contains a comma, a double quote, or a newline; double up
// any inner double quotes. This file is the entire contract.

export type CsvCell = string | number | boolean | null | undefined;
export type CsvRow = CsvCell[];

function _escapeCell(value: CsvCell): string {
  if (value == null) return "";
  const text = String(value);
  if (text === "") return "";
  const needsQuoting = /[",\r\n]/.test(text);
  const escaped = text.replace(/"/g, '""');
  return needsQuoting ? `"${escaped}"` : escaped;
}

// Serialise a 2D table to a CSV string. Trailing CRLF is omitted so the
// caller can append further rows by string concatenation if needed.
export function toCsv(rows: CsvRow[]): string {
  return rows.map((row) => row.map(_escapeCell).join(",")).join("\r\n");
}

// Trigger a browser download for `csv` under `filename`. SSR-safe: returns
// silently when `document` is undefined (e.g. unit tests that import the
// caller but never click an export button).
export function downloadCsvBlob(csv: string, filename: string): void {
  if (typeof document === "undefined" || typeof URL === "undefined") return;
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}
