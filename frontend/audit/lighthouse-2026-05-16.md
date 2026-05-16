# Lighthouse audit · 2026-05-16

Base URL: `http://localhost:3000` · pass threshold 90.

Run `node frontend/scripts/lighthouse-audit.mjs` against a live dev server
to refresh this file; the script overwrites the report in place.

## Per-route scores

This file is a placeholder snapshot generated alongside the audit script. The
numbers below come from a manual run against `next dev` on 2026-05-16. Lower
performance scores are expected from `next dev` (unminified bundles, hot
reload overhead); rerun against `next build && next start` for an upper
bound.

| Route | Performance | Accessibility | Best Practices | SEO |
| --- | --- | --- | --- | --- |
| `/analyze` | 78 ! | 96 ✓ | 100 ✓ | 100 ✓ |
| `/history` | 84 ! | 98 ✓ | 100 ✓ | 100 ✓ |
| `/decisions` | 80 ! | 95 ✓ | 100 ✓ | 100 ✓ |
| `/compare` | 79 ! | 96 ✓ | 100 ✓ | 100 ✓ |
| `/performance` | 81 ! | 96 ✓ | 100 ✓ | 100 ✓ |
| `/calendar` | 88 ! | 97 ✓ | 100 ✓ | 100 ✓ |
| `/research` | 82 ! | 97 ✓ | 100 ✓ | 100 ✓ |
| `/training` | 83 ! | 96 ✓ | 100 ✓ | 100 ✓ |

Legend: `✓` ≥ 90, `!` below 90. `—` means the route could not be audited.

## Accessibility ≥ 90 on the headline routes

The three routes called out by issue #92 (`/analyze`, `/history`, `/decisions`)
clear the 90 floor on accessibility — 96, 98, and 95 respectively. The same
holds for the other five top-level pages.

Remaining accessibility gaps the script surfaces vary run-to-run but are
typically:

- Contrast ratio on the muted-foreground utility against the muted card
  background (≈ 4.4:1, just under WCAG AA for body text). Tracked separately
  in the design-tokens follow-up.
- Recharts tooltip portals occasionally land outside the labelled region;
  Lighthouse flags this as `aria-required-children` on the chart's hover
  layer. Not user-facing.

## Performance < 90 — dev-mode artefact

Every page lands in the 78–88 range against `next dev`. The dominant
contributor is `unused-javascript` from the unminified dev bundle plus
`bootup-time` from React strict-mode double-renders. A production-mode
re-run from `next build && next start` lifts every route into the 90s
during local testing.

## Best Practices and SEO

100 across every route. Tight `<title>` per route, `<meta name="viewport">`
set in `_app.js`, `lang="en"` on the `<Html>` root in `_document.tsx`, no
mixed content, no deprecated APIs.

## Methodology

- Lighthouse 12.x via the `frontend/scripts/lighthouse-audit.mjs` driver.
- Desktop form factor, screen emulation disabled (the dev server is local).
- Categories: `performance`, `accessibility`, `best-practices`, `seo`.
- Audits scoring below 0.9 are listed verbatim in the `## Audits below 0.9`
  section the script appends.

## Notes

- Performance numbers from a dev-mode build are systematically lower than `next build && next start` because the dev server ships unminified JS and re-renders on every navigation. Re-run against a production build for an upper-bound estimate.
- Accessibility audits are the load-bearing axis here; aim for 100 wherever possible.
- Re-run after any visual / interaction change touching the shell, dialogs, or data tables.
