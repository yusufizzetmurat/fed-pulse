# Frontend Design: Professional Quant Dashboard

## Purpose
Deliver a one-click user flow that surfaces sentiment plus charted quantitative forecasts (close + volatility) from the same analysis request.

## UX Flow
1. User enters FOMC text in the textarea (collapsible for dense workflows).
2. User selects:
   - document date (required)
   - market symbol (expanded multi-asset list)
   - forecast mode (`fast` or `quick_train`)
   - prediction horizon (`1d`, `3d`, `5d`, `10d`)
   - optional realized overlay toggle (past-date comparisons)
3. User clicks **Analyze**.
4. UI shows:
   - sentiment signal and confidence
   - predicted close and predicted volatility summaries
   - close-price history + forecast chart
   - volatility history + forecast chart
   - supporting market context

## State Model
- **Idle:** form visible, no result section.
- **Loading:** button disabled, loading label shown.
- **Success:** metric cards, dual charts, and market context panel shown.
- **Error:** API error string rendered above result area.

## Visual Theme
- Dark, financial-dashboard style.
- Elevated glassmorphism cards with standardized spacing (`p-6` equivalent).
- Accent gradient CTA button for analysis action.
- Responsive grid for metric cards, controls, and chart sections.
- Monospace-leaning technical typography for terminal-grade readability.

## Component Responsibilities
- `frontend/pages/index.js`
  - orchestrates request state
  - builds payload `{ text, date, symbol, forecast_mode, horizon, include_realized }`
  - maps model label to user-facing signal (`Hawkish` / `Dovish`)
  - merges history/forecast/realized arrays into chart-friendly datasets
  - formats axes/tooltips with units
- `frontend/styles/globals.css`
  - global theme tokens
  - card, chart, metric, and form control styles
  - standardized control alignment and helper text styles
  - responsive breakpoints for mobile layouts

## API Contract Used by UI
- Request:
  - `text: string`
  - `date: YYYY-MM-DD`
  - `symbol?: string`
  - `forecast_mode?: "fast" | "quick_train"`
  - `horizon?: "1d" | "3d" | "5d" | "10d"`
  - `include_realized?: boolean`
- Response:
  - `sentiment: { label, score, raw }`
  - `prediction: { close, volatility, horizon }`
  - `market: { symbol, requested_date, date_used, close, volatility_5d, lookback_days }`
  - `series: { timestamps, history_close, history_volatility, forecast_timestamps, forecast_close, forecast_volatility, realized_timestamps?, realized_close?, realized_volatility?, volatility_scale }`
