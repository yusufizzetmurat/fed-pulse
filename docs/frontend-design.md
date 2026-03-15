# Frontend Design: Demo Dashboard

## Purpose
Deliver a one-click user flow that surfaces both semantic sentiment and quantitative volatility output from the same analysis request.

## UX Flow
1. User enters FOMC text in the textarea.
2. User selects:
   - document date (required)
   - market symbol (`^GSPC` or `DX-Y.NYB`)
3. User clicks **Analyze**.
4. UI shows:
   - sentiment signal and confidence
   - predicted volatility and horizon
   - supporting market context

## State Model
- **Idle:** form visible, no result section.
- **Loading:** button disabled, loading label shown.
- **Success:** metric cards and market context panel shown.
- **Error:** API error string rendered above result area.

## Visual Theme
- Dark, financial-dashboard style.
- Elevated cards with clear hierarchy and spacing.
- Accent gradient CTA button for analysis action.
- Responsive grid for metric cards and inputs.

## Component Responsibilities
- `frontend/pages/index.js`
  - orchestrates request state
  - builds payload `{ text, date, symbol }`
  - maps model label to user-facing signal (`Hawkish` / `Dovish`)
- `frontend/styles/globals.css`
  - global theme tokens
  - card, metric, and form control styles
  - responsive breakpoints for mobile layouts

## API Contract Used by UI
- Request:
  - `text: string`
  - `date: YYYY-MM-DD`
  - `symbol?: string`
- Response:
  - `sentiment: { label, score, raw }`
  - `prediction: { volatility, horizon }`
  - `market: { symbol, requested_date, date_used, close, volatility_5d, lookback_days }`
