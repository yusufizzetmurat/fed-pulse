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
];

export const HORIZON_OPTIONS: Horizon[] = ["1d", "3d", "5d", "10d"];

export const REAL_TRAIN_POLL_INTERVAL_MS = 2000;
export const REAL_TRAIN_POLL_MAX = 180;
