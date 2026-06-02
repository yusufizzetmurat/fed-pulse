"""Scripted end-to-end demo + screenshot capture (#298).

Drives the live front-end through every panel the 2026-06-05 report
cites and writes a stable set of PNG screenshots to the wiki repo
under ``fed-pulse.wiki/assets/demo/``. The script assumes a backend
listening on ``http://localhost:8000`` and a frontend on
``http://localhost:3000`` — both default ``make dev`` ports.

The capture is deterministic: a 1440×900 viewport, fixed FOMC statement,
fixed symbol + horizon, and explicit waits per panel so screenshots are
diffable run-to-run.

Panels captured
---------------

1. ``statement_decomposition_<ts>.png`` — paste-and-analyse form,
   regime headline, multi-axis interpretation, factor card (with the
   #328 null-result framing).
2. ``market_reaction_<ts>.png`` — rates cards (#292/#293) + vol-regime
   card (#322/#326).
3. ``historical_analogs_<ts>.png`` — top-3 hits (#294/#295).
4. ``trajectory_<ts>.png`` — projected next stance with the
   #332/#338 lift-vs-baseline badge.
5. ``settings_checkpoints_<ts>.png`` — /settings page surface with
   the #341/#342 sidecar contract + supplied_at_inference badges.

CLI
---

::

    # Backend + frontend up via `make dev`, then:
    python scripts/demo_end_to_end.py

    # Override the bases (e.g. against the deployed droplet):
    python scripts/demo_end_to_end.py \\
        --backend-url https://fedpulse.yusufizzetmurat.com/api \\
        --frontend-url https://fedpulse.yusufizzetmurat.com

The script writes to ``../fed-pulse.wiki/assets/demo/`` by default; pass
``--output-dir`` to redirect or ``--in-repo`` to write under
``docs/demo-screenshots/`` instead.

Dependencies
------------

The Playwright Python package + a Chromium download. On a fresh
checkout::

    pip install playwright
    playwright install chromium

The script imports ``playwright.sync_api`` lazily so the module can
still be imported (for ``--help`` etc.) without the dep installed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FOMC_TEXT = (
    "Recent indicators suggest economic activity has continued to expand at a "
    "solid pace. Job gains have moderated since earlier in the year but remain "
    "solid, and the unemployment rate has moved up slightly though it stays low. "
    "Inflation has eased over the past year but remains somewhat elevated. The "
    "Committee judges that the risks to achieving its employment and inflation "
    "goals continue to move into better balance."
)
DEFAULT_BACKEND = "http://localhost:8000"
DEFAULT_FRONTEND = "http://localhost:3000"
VIEWPORT = {"width": 1440, "height": 900}


@dataclass
class DemoArgs:
    backend_url: str
    frontend_url: str
    output_dir: Path
    fomc_text: str
    headless: bool
    wait_timeout_ms: int
    keep_open: bool
    strict: bool = False


def _parse_args(argv: Sequence[str] | None = None) -> DemoArgs:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend-url", default=DEFAULT_BACKEND)
    parser.add_argument("--frontend-url", default=DEFAULT_FRONTEND)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override the screenshot output directory. Defaults to ../fed-pulse.wiki/assets/demo/.",
    )
    parser.add_argument(
        "--in-repo",
        action="store_true",
        help="Write screenshots under docs/demo-screenshots/ inside the main repo instead.",
    )
    parser.add_argument(
        "--fomc-text",
        default=DEFAULT_FOMC_TEXT,
        help="Statement text pasted into the analyze form. Defaults to a sanitised FOMC excerpt.",
    )
    parser.add_argument(
        "--headed",
        action="store_true",
        help="Run with the browser visible. Default is headless.",
    )
    parser.add_argument(
        "--keep-open",
        action="store_true",
        help="Pause after capture so the operator can inspect the live UI before tear-down.",
    )
    parser.add_argument(
        "--wait-timeout-ms",
        type=int,
        default=30_000,
        help="Per-step wait timeout in milliseconds.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when any panel is skipped instead of treating skips as soft.",
    )
    args = parser.parse_args(argv)

    if args.output_dir is not None:
        out = Path(args.output_dir).resolve()
    elif args.in_repo:
        out = REPO_ROOT / "docs" / "demo-screenshots"
    else:
        # Walk back from REPO_ROOT looking for a sibling `fed-pulse.wiki` dir
        # so the demo works both from a top-level checkout and from a worktree.
        candidates = [
            REPO_ROOT.parent / "fed-pulse.wiki",
            REPO_ROOT.parent.parent / "fed-pulse.wiki",
            REPO_ROOT.parent.parent.parent / "fed-pulse.wiki",
            REPO_ROOT.parent.parent.parent.parent / "fed-pulse.wiki",
        ]
        out = next(
            (wiki / "assets" / "demo" for wiki in candidates if wiki.exists()),
            REPO_ROOT.parent / "fed-pulse.wiki" / "assets" / "demo",
        )

    return DemoArgs(
        backend_url=args.backend_url.rstrip("/"),
        frontend_url=args.frontend_url.rstrip("/"),
        output_dir=out,
        fomc_text=args.fomc_text,
        headless=not args.headed,
        wait_timeout_ms=args.wait_timeout_ms,
        keep_open=args.keep_open,
        strict=args.strict,
    )


def _import_playwright():
    """Lazy-import so ``--help`` works without the dep installed."""

    try:
        from playwright.sync_api import sync_playwright  # noqa: WPS433 (intentional lazy import)
    except ImportError as exc:
        raise SystemExit(
            "playwright is not installed. Run `pip install playwright && playwright install chromium`."
        ) from exc
    return sync_playwright


def _timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _wait_panel(page, locator_text: str, timeout_ms: int) -> None:
    """Scroll the panel with ``locator_text`` into view and let it settle."""

    locator = page.get_by_text(locator_text, exact=False).first
    locator.wait_for(timeout=timeout_ms)
    locator.scroll_into_view_if_needed()
    page.wait_for_timeout(400)  # give the chart libs a beat to paint


def _capture(page, output_dir: Path, name: str, timestamp: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}_{timestamp}.png"
    page.screenshot(path=str(path), full_page=True)
    print(f"[demo]   wrote {path}")
    return path


def run_demo(args: DemoArgs) -> int:
    """Drive the demo end-to-end and capture the panel screenshots.

    Returns 0 on success, 1 on any panel failure.
    """

    sync_playwright = _import_playwright()
    timestamp = _timestamp()
    output_dir = args.output_dir
    print(f"[demo] frontend={args.frontend_url} backend={args.backend_url}")
    print(f"[demo] output_dir={output_dir} timestamp={timestamp}")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=args.headless)
        context = browser.new_context(viewport=VIEWPORT)
        # The frontend resolves its API base URL at build time from
        # NEXT_PUBLIC_API_URL (see frontend/lib/analyze/api.ts). The
        # --backend-url flag is therefore a documentation knob; capture
        # against the deployed bundle by pointing --frontend-url at the
        # public origin (the prod build already wires `/api` correctly).
        page = context.new_page()
        page.set_default_timeout(args.wait_timeout_ms)

        # ---- Step 1: paste statement + analyse. -----------------------
        page.goto(args.frontend_url, wait_until="domcontentloaded")
        # The text area is rendered by AnalyzeForm with a placeholder; pasting
        # via fill keeps the diff small and avoids per-character debounce.
        textarea = page.locator("textarea").first
        textarea.wait_for(timeout=args.wait_timeout_ms)
        textarea.fill(args.fomc_text)

        # The submit button is the only primary CTA in the form. Match by
        # role rather than CSS so a className change does not break the demo.
        analyse_button = page.get_by_role("button", name="Analyze")
        if analyse_button.count() == 0:
            # Older copy: "Run analysis" / "Analyse".
            analyse_button = page.get_by_role("button").filter(
                has_text="naly"
            ).first
        analyse_button.click()

        # Wait for the regime headline to render (the first panel to settle).
        _wait_panel(page, "Regime", args.wait_timeout_ms)

        # ---- Step 2: statement decomposition screenshot. --------------
        # The form + regime headline + multi-axis cards sit at the top of /
        # so the full-page screenshot captures all three.
        _wait_panel(page, "Multi-axis interpretation", args.wait_timeout_ms)
        page.evaluate("window.scrollTo(0, 0)")
        page.wait_for_timeout(300)
        _capture(page, output_dir, "statement_decomposition", timestamp)

        skip_count = 0

        # ---- Step 3: market reaction. ---------------------------------
        # Cold-start dev backends produce a regression-only checkpoint;
        # the market panel then collapses to an "evidence unavailable"
        # state and the heading text never paints. Skip + log rather
        # than hard-fail so the rest of the capture still runs.
        try:
            _wait_panel(page, "Market reaction", args.wait_timeout_ms)
            _capture(page, output_dir, "market_reaction", timestamp)
        except Exception as exc:  # noqa: BLE001
            print(f"[demo]   market_reaction panel skipped: {exc}")
            skip_count += 1

        # ---- Step 4: historical analogs. ------------------------------
        try:
            _wait_panel(page, "Historical analog", args.wait_timeout_ms)
            _capture(page, output_dir, "historical_analogs", timestamp)
        except Exception as exc:  # noqa: BLE001 - analog index may be absent on a fresh deploy
            print(f"[demo]   historical_analogs panel skipped: {exc}")
            skip_count += 1

        # ---- Step 5: trajectory. --------------------------------------
        try:
            _wait_panel(page, "Trajectory", args.wait_timeout_ms)
            _capture(page, output_dir, "trajectory", timestamp)
        except Exception as exc:  # noqa: BLE001
            print(f"[demo]   trajectory panel skipped: {exc}")
            skip_count += 1

        # ---- Step 6: settings / checkpoint surface. -------------------
        page.goto(f"{args.frontend_url}/settings", wait_until="domcontentloaded")
        try:
            _wait_panel(page, "Checkpoint", args.wait_timeout_ms)
        except Exception:
            # Older settings copy: "Settings" or "Inference contract".
            page.wait_for_load_state("networkidle", timeout=args.wait_timeout_ms)
        page.wait_for_timeout(300)
        _capture(page, output_dir, "settings_checkpoints", timestamp)

        if args.keep_open:
            print("[demo] --keep-open set; press Enter to tear down...")
            try:
                input()
            except EOFError:
                pass

        context.close()
        browser.close()

    if args.strict and skip_count > 0:
        print(f"[demo] {skip_count} panels skipped in strict mode")
        return 1
    print("[demo] OK")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    start = time.time()
    rc = run_demo(args)
    elapsed = time.time() - start
    print(f"[demo] elapsed={elapsed:.1f}s exit={rc}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
