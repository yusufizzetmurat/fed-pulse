import type { Horizon, SymbolValue } from "./types";

export const DEFAULT_TEXT =
  "Recent indicators suggest economic activity has continued to expand at a solid pace.";

export interface SymbolOption {
  value: SymbolValue;
  label: string;
}

export const SYMBOL_OPTIONS: SymbolOption[] = [
  { value: "^GSPC", label: "S&P 500 (^GSPC)" },
  { value: "DX-Y.NYB", label: "Dollar Index (DX-Y.NYB)" },
  { value: "^NDX", label: "NASDAQ 100 (^NDX)" },
  { value: "^DJI", label: "Dow Jones (^DJI)" },
  { value: "^VIX", label: "CBOE Volatility Index (^VIX)" },
  { value: "GC=F", label: "Gold Futures (GC=F)" },
  { value: "CL=F", label: "WTI Crude Oil (CL=F)" },
  { value: "EURUSD=X", label: "EUR/USD (EURUSD=X)" },
  { value: "BTC-USD", label: "Bitcoin (BTC-USD)" },
  { value: "^TNX", label: "US 10Y Yield (^TNX)" },
  // Cross-bank flagship indices. Available so the cross-bank page's
  // "Open in Workspace" deep-link lands on a known symbol; the HAR-
  // tercile + RV backtest panels gracefully degrade for tickers not
  // in HAR_TERCILE_SUPPORTED_SYMBOLS below.
  { value: "^STOXX50E", label: "Euro Stoxx 50 (^STOXX50E)" },
  { value: "^FTSE", label: "FTSE 100 (^FTSE)" },
  { value: "^GSPTSE", label: "S&P/TSX Composite (^GSPTSE)" },
  { value: "^N225", label: "Nikkei 225 (^N225)" },
  { value: "^AXJO", label: "ASX 200 (^AXJO)" },
];

export const HORIZON_OPTIONS: Horizon[] = ["1d", "3d", "5d", "10d"];

// Symbols the HAR-tercile baseline + backtest endpoints support. ^GSPC
// serves from the pinned QLIKE-DLq artifact; ^NDX and ^DJI use a
// per-call OLS HAR fit when their per-asset artifact is not on disk
// yet. FX / commodity tickers stay out of scope — HAR-tercile is fit
// on equity-index RV.
export const HAR_TERCILE_SUPPORTED_SYMBOLS: readonly string[] = [
  "^GSPC",
  "^NDX",
  "^DJI",
];

// Symbols that have a per-asset QLIKE-DLq production artifact on disk
// (or are reachable via HF Hub). The workspace's QLIKE-RV band-coverage
// panel renders only for these tickers; everything else falls through
// to the "asset not supported" stub. Mirrors
// ``backend/app/services/rv_forecaster.py:SYMBOL_ARTIFACTS`` -- keep
// the two lists in sync.
export const QLIKE_RV_SUPPORTED_SYMBOLS: readonly string[] = [
  "^GSPC",
  "^NDX",
  "^DJI",
];

export const REAL_TRAIN_POLL_INTERVAL_MS = 2000;
export const REAL_TRAIN_POLL_MAX = 180;
