#!/usr/bin/env node
// Lighthouse audit driver for the Next.js dev server.
//
// Runs Lighthouse against each route in ROUTES, captures Performance,
// Accessibility, Best Practices, and SEO scores, and writes a Markdown
// summary to frontend/audit/lighthouse-<YYYY-MM-DD>.md.
//
// The script does NOT fail the build when scores drop below 90 — it is a
// one-shot audit, not a CI gate. Failing audits are written verbatim into
// the report so they can be triaged.
//
// Usage:
//   1. Start the Next.js dev server: `make dev-cpu` (frontend at :3000)
//   2. Run: `node frontend/scripts/lighthouse-audit.mjs`
//
// Override the target host with LIGHTHOUSE_BASE_URL (default
// http://localhost:3000).
import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const FRONTEND_ROOT = path.resolve(__dirname, "..");
const AUDIT_DIR = path.join(FRONTEND_ROOT, "audit");
const BASE_URL = (process.env.LIGHTHOUSE_BASE_URL || "http://localhost:3000").replace(/\/$/, "");

const ROUTES = [
  { path: "/analyze", label: "Analyze" },
  { path: "/history", label: "History" },
  { path: "/decisions", label: "Decisions" },
  { path: "/compare", label: "Compare" },
  { path: "/performance", label: "Performance" },
  { path: "/calendar", label: "FOMC calendar" },
  { path: "/research", label: "Research" },
  { path: "/training", label: "Training" },
];

const CATEGORIES = ["performance", "accessibility", "best-practices", "seo"];
const PASS_THRESHOLD = 90;

function todayIso() {
  const d = new Date();
  const yyyy = d.getUTCFullYear();
  const mm = String(d.getUTCMonth() + 1).padStart(2, "0");
  const dd = String(d.getUTCDate()).padStart(2, "0");
  return `${yyyy}-${mm}-${dd}`;
}

function scoreCell(score) {
  if (score == null) return "—";
  const pct = Math.round(score * 100);
  const flag = pct >= PASS_THRESHOLD ? "✓" : "!";
  return `${pct} ${flag}`;
}

function audits(lhr) {
  const failing = [];
  for (const audit of Object.values(lhr.audits ?? {})) {
    if (audit.score != null && audit.score < 0.9 && audit.scoreDisplayMode !== "informative") {
      failing.push({ id: audit.id, title: audit.title, score: audit.score });
    }
  }
  return failing;
}

async function probe(url, attempts = 20, delayMs = 1500) {
  for (let i = 0; i < attempts; i += 1) {
    try {
      const response = await fetch(url, { redirect: "manual" });
      if (response.status < 500) return true;
    } catch {
      // ignore — dev server may not be up yet
    }
    await new Promise((r) => setTimeout(r, delayMs));
  }
  return false;
}

async function runLighthouse(targetUrl, { lighthouse, chrome }) {
  const flags = {
    port: chrome.port,
    output: "json",
    logLevel: "error",
    onlyCategories: CATEGORIES,
  };
  const config = {
    extends: "lighthouse:default",
    settings: { formFactor: "desktop", screenEmulation: { disabled: true } },
  };
  const result = await lighthouse(targetUrl, flags, config);
  return result?.lhr;
}

async function main() {
  await mkdir(AUDIT_DIR, { recursive: true });

  console.log(`Lighthouse audit · base ${BASE_URL}`);
  const reachable = await probe(`${BASE_URL}/analyze`);
  if (!reachable) {
    const message = `Dev server is not reachable at ${BASE_URL}. Start the frontend (e.g. \`make dev-cpu\`) and retry.`;
    const stamp = todayIso();
    const fallback = path.join(AUDIT_DIR, `lighthouse-${stamp}.md`);
    await writeFile(
      fallback,
      `# Lighthouse audit · ${stamp}\n\n> Skipped — ${message}\n`,
      "utf8"
    );
    console.error(message);
    process.exitCode = 0;
    return;
  }

  let lighthouse;
  let chromeLauncher;
  try {
    ({ default: lighthouse } = await import("lighthouse"));
    chromeLauncher = await import("chrome-launcher");
  } catch (err) {
    const message =
      "lighthouse / chrome-launcher are not installed. Run `npm install --save-dev lighthouse chrome-launcher` from the frontend/ directory.";
    console.error(message);
    console.error(err);
    process.exitCode = 1;
    return;
  }

  const chrome = await chromeLauncher.launch({
    chromeFlags: [
      "--headless=new",
      "--disable-gpu",
      "--no-sandbox",
      "--disable-dev-shm-usage",
    ],
  });

  const rows = [];
  const failingByRoute = [];
  try {
    for (const route of ROUTES) {
      const target = `${BASE_URL}${route.path}`;
      console.log(`  · ${route.path}`);
      try {
        const lhr = await runLighthouse(target, { lighthouse, chrome });
        const scores = Object.fromEntries(
          CATEGORIES.map((id) => [id, lhr?.categories?.[id]?.score ?? null])
        );
        rows.push({ route, scores });
        const failing = audits(lhr);
        if (failing.length) failingByRoute.push({ route, failing });
      } catch (err) {
        rows.push({ route, scores: null, error: (err && err.message) || String(err) });
      }
    }
  } finally {
    await chrome.kill();
  }

  const stamp = todayIso();
  const reportPath = path.join(AUDIT_DIR, `lighthouse-${stamp}.md`);
  const lines = [];
  lines.push(`# Lighthouse audit · ${stamp}`);
  lines.push("");
  lines.push(`Base URL: \`${BASE_URL}\` · pass threshold ${PASS_THRESHOLD}.`);
  lines.push("");
  lines.push("## Per-route scores");
  lines.push("");
  lines.push("| Route | Performance | Accessibility | Best Practices | SEO |");
  lines.push("| --- | --- | --- | --- | --- |");
  for (const row of rows) {
    if (!row.scores) {
      lines.push(`| \`${row.route.path}\` | — | — | — | — |`);
      continue;
    }
    lines.push(
      `| \`${row.route.path}\` | ${scoreCell(row.scores.performance)} | ${scoreCell(
        row.scores.accessibility
      )} | ${scoreCell(row.scores["best-practices"])} | ${scoreCell(row.scores.seo)} |`
    );
  }
  lines.push("");
  lines.push("Legend: `✓` ≥ 90, `!` below 90. `—` means the route could not be audited.");
  lines.push("");

  if (failingByRoute.length) {
    lines.push("## Audits below 0.9");
    lines.push("");
    for (const entry of failingByRoute) {
      lines.push(`### \`${entry.route.path}\` (${entry.route.label})`);
      lines.push("");
      for (const a of entry.failing) {
        const pct = a.score == null ? "—" : Math.round(a.score * 100);
        lines.push(`- \`${a.id}\` · ${a.title} · ${pct}`);
      }
      lines.push("");
    }
  } else {
    lines.push("## Audits below 0.9");
    lines.push("");
    lines.push("None — every audit on every route cleared 0.9.");
    lines.push("");
  }

  lines.push("## Notes");
  lines.push("");
  lines.push("- Performance numbers from a dev-mode build are systematically lower than `next build && next start` because the dev server ships unminified JS and re-renders on every navigation. Re-run against a production build for an upper-bound estimate.");
  lines.push("- Accessibility audits are the load-bearing axis here; aim for 100 wherever possible.");
  lines.push("- Re-run after any visual / interaction change touching the shell, dialogs, or data tables.");
  lines.push("");

  await writeFile(reportPath, lines.join("\n"), "utf8");
  console.log(`Report written: ${reportPath}`);
}

main().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});
