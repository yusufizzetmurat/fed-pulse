# End-to-end specs

Playwright drives a real browser against `npm run dev`, or against a
hosted URL via `E2E_BASE_URL`. The specs assume the backend is reachable
at `NEXT_PUBLIC_API_URL` and that the cold-start checkpoint is already
on disk so `/analyze` returns under a few seconds.

## Running locally

```bash
# One-time browser install
npx playwright install chromium webkit

# Boot the dev server + run the suite
npm run e2e

# Interactive UI mode for debugging single specs
npm run e2e:ui
```

Pass `E2E_PORT=<port>` if 3000 is busy, or `E2E_BASE_URL=https://...` to
point the suite at a remote deployment such as
`https://fedpulse.yusufizzetmurat.com/`.
