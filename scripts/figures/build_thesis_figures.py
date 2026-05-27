"""Thesis figure-generation entry point (#298).

Renders the figures the 2026-06-05 report cites into a single directory
under the wiki repo (``fed-pulse.wiki/assets/figures/``). Every figure
carries a reproducibility caption with the commit SHA at generation
time, the canonical training-package id, and the source artefact path
on disk.

The script is pure-PIL on purpose. The backend image already ships
Pillow as a transitive dep (see ``backend/app/evaluation/confusion_matrix_render.py``
for the prior precedent); adding matplotlib would inflate the wheel
weight by ~30 MB for a handful of tables. The per-cell rendering is
the same convention the confusion-matrix module already uses.

Figures produced
----------------

1. ``architecture.png`` — backend / frontend / external-data flow
   diagram derived from the §3 system-architecture wiki C4 sketch.
2. ``dual_head_comparison.png`` — three-way head-mode table from
   ``backend/artifacts/experiments/dual_head_comparison_canonical.json``.
3. ``text_path_ab.png`` — text-channel A/B table from
   ``backend/artifacts/experiments/text_path_ab_canonical.json``.
4. ``cross_bank_ladder.png`` — cross-bank transfer ladder transcribed
   from wiki §6.14 (no JSON artefact on disk; the table is the
   canonical source).

Each PNG ships a paired ``.caption.txt`` with the reproducibility
header so the report can drop the caption verbatim.

CLI
---

::

    python -m scripts.figures.build_thesis_figures \\
        --output-dir ../fed-pulse.wiki/assets/figures

Default output dir resolves to the wiki repo next to the main repo
checkout. The script also supports ``--in-repo`` which writes the
figures under ``docs/figures/`` instead — handy for previewing
without touching the wiki repo.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit(
        "Pillow is required to generate thesis figures. "
        "Install via `pip install pillow` or run inside the backend container."
    ) from exc


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTEFACTS_DIR = REPO_ROOT / "backend" / "artifacts" / "experiments"
CANONICAL_TP_ID = (
    "tp_v3_macro_aug_2026_05_25_fwd_strict_sentiment_market_core_v1.1_epv1_v1.0"
)

# Tuned for legibility against the report's two-column layout. Cells stay
# wide enough that the four-decimal metric strings fit without wrapping.
_CELL_PAD_X = 18
_CELL_PAD_Y = 12
_ROW_HEIGHT = 36
_HEADER_HEIGHT = 44
_TITLE_HEIGHT = 56
_CAPTION_HEIGHT = 72
_BG = (255, 255, 255)
_BORDER = (200, 206, 214)
_HEADER_BG = (240, 244, 250)
_TITLE_FG = (20, 30, 50)
_BODY_FG = (40, 46, 58)
_MUTED_FG = (96, 104, 118)
_ACCENT = (32, 96, 196)


def _load_font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    """Resolve a usable TTF; fall back to PIL's bitmap default."""

    candidates: list[str] = []
    if bold:
        candidates.extend(
            [
                "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
            ]
        )
    candidates.extend(
        [
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        ]
    )
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


def _git_sha() -> str:
    """Return the short commit SHA at script run time."""

    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "uncommitted"


def _measure(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    """Width / height of ``text`` rendered with ``font``."""

    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


@dataclass
class ReproducibilityHeader:
    """Caption header carried by every figure."""

    commit_sha: str
    training_package_id: str
    source_artefact: str

    def render(self) -> str:
        return (
            f"Commit {self.commit_sha} · "
            f"Training package: {self.training_package_id} · "
            f"Source: {self.source_artefact}"
        )


def _wrap_caption(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
) -> list[str]:
    """Greedy word-wrap so the caption fits the figure width."""

    words = text.split(" ")
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        trial = (" ".join(current + [word])).strip()
        width, _ = _measure(draw, trial, font)
        if width > max_width and current:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    return lines


def _render_table(
    *,
    title: str,
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    caption: str,
    output_path: Path,
    align: Sequence[str] | None = None,
) -> Path:
    """Render a single titled table with caption to ``output_path``."""

    if align is None:
        align = ["left"] + ["right"] * (len(headers) - 1)
    if len(align) != len(headers):
        raise ValueError("align length must match headers")

    title_font = _load_font(24, bold=True)
    header_font = _load_font(15, bold=True)
    body_font = _load_font(15)
    caption_font = _load_font(13)

    # First pass: measure widest cell per column.
    probe = Image.new("RGB", (10, 10), _BG)
    pdraw = ImageDraw.Draw(probe)
    col_widths: list[int] = []
    for c, header in enumerate(headers):
        w, _ = _measure(pdraw, header, header_font)
        col_widths.append(w)
    for row in rows:
        for c, cell in enumerate(row):
            w, _ = _measure(pdraw, str(cell), body_font)
            col_widths[c] = max(col_widths[c], w)
    col_widths = [w + 2 * _CELL_PAD_X for w in col_widths]

    table_width = sum(col_widths)
    title_w, _ = _measure(pdraw, title, title_font)
    width = max(table_width, title_w + 2 * _CELL_PAD_X, 720)

    caption_lines = _wrap_caption(pdraw, caption, caption_font, width - 2 * _CELL_PAD_X)
    caption_box_h = max(_CAPTION_HEIGHT, len(caption_lines) * 20 + 16)

    table_height = _HEADER_HEIGHT + len(rows) * _ROW_HEIGHT
    height = _TITLE_HEIGHT + table_height + caption_box_h + 16

    img = Image.new("RGB", (width, height), _BG)
    draw = ImageDraw.Draw(img)

    # Title.
    draw.text((_CELL_PAD_X, 16), title, fill=_TITLE_FG, font=title_font)

    # Table origin centred horizontally.
    table_x = (width - table_width) // 2
    table_y = _TITLE_HEIGHT

    # Header band.
    draw.rectangle(
        [table_x, table_y, table_x + table_width, table_y + _HEADER_HEIGHT],
        fill=_HEADER_BG,
        outline=_BORDER,
    )
    cx = table_x
    for c, header in enumerate(headers):
        text_w, text_h = _measure(draw, header, header_font)
        if align[c] == "right":
            tx = cx + col_widths[c] - _CELL_PAD_X - text_w
        elif align[c] == "center":
            tx = cx + (col_widths[c] - text_w) // 2
        else:
            tx = cx + _CELL_PAD_X
        ty = table_y + (_HEADER_HEIGHT - text_h) // 2
        draw.text((tx, ty), header, fill=_TITLE_FG, font=header_font)
        cx += col_widths[c]

    # Rows.
    for r, row in enumerate(rows):
        ry = table_y + _HEADER_HEIGHT + r * _ROW_HEIGHT
        draw.rectangle(
            [table_x, ry, table_x + table_width, ry + _ROW_HEIGHT],
            fill=_BG,
            outline=_BORDER,
        )
        cx = table_x
        for c, cell in enumerate(row):
            txt = str(cell)
            text_w, text_h = _measure(draw, txt, body_font)
            if align[c] == "right":
                tx = cx + col_widths[c] - _CELL_PAD_X - text_w
            elif align[c] == "center":
                tx = cx + (col_widths[c] - text_w) // 2
            else:
                tx = cx + _CELL_PAD_X
            ty = ry + (_ROW_HEIGHT - text_h) // 2
            draw.text((tx, ty), txt, fill=_BODY_FG, font=body_font)
            cx += col_widths[c]

    # Caption.
    cy = table_y + table_height + 12
    for line in caption_lines:
        draw.text((_CELL_PAD_X, cy), line, fill=_MUTED_FG, font=caption_font)
        cy += 20

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, format="PNG", optimize=True)

    caption_path = output_path.with_suffix(".caption.txt")
    caption_path.write_text(caption + "\n", encoding="utf-8")
    return output_path


def _format_stat(stat: dict | None) -> str:
    """Render ``{mean, std, n}`` as ``0.4190 ± 0.0697`` (or ``—`` when null)."""

    if stat is None:
        return "—"
    mean = stat.get("mean")
    std = stat.get("std")
    if mean is None:
        return "—"
    if std is None:
        return f"{mean:.4f}"
    return f"{mean:.4f} ± {std:.4f}"


def _read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"artefact missing on disk: {path} — run the corresponding sweep first"
        )
    with path.open() as fh:
        return json.load(fh)


def build_dual_head_table(header: ReproducibilityHeader, output_dir: Path) -> Path:
    src = ARTEFACTS_DIR / "dual_head_comparison_canonical.json"
    data = _read_json(src)
    summary = data["summary"]

    rows = []
    for mode in data["head_modes"]:
        block = summary[mode]
        rows.append(
            [
                mode,
                _format_stat(block.get("regime_f1_macro")),
                _format_stat(block.get("regression_rmse_log_rv")),
                str(block.get("regime_f1_macro", {}).get("n", "—") if block.get("regime_f1_macro") else "—"),
            ]
        )

    caption = (
        f"Three-way head-mode comparison on the canonical training package. "
        f"Pooled across {len(data['seeds'])} seeds × {len(data['fold_ids'])} walk-forward folds "
        f"({data.get('epochs', '?')} epochs, regression_alpha={data.get('regression_alpha', '?')}). "
        f"{header.render()}"
    )

    return _render_table(
        title="Dual-head methodology — three-way comparison",
        headers=["Head mode", "regime macro-F1", "RMSE log(RV)", "n (seed × fold)"],
        rows=rows,
        caption=caption,
        output_path=output_dir / "dual_head_comparison.png",
        align=["left", "right", "right", "right"],
    )


def build_text_path_ab_table(header: ReproducibilityHeader, output_dir: Path) -> Path:
    src = ARTEFACTS_DIR / "text_path_ab_canonical.json"
    data = _read_json(src)
    summary = data["summary"]

    rows = []
    for arm in data["arms"]:
        block = summary[arm]
        rows.append(
            [
                arm,
                _format_stat(block.get("regime_f1_macro")),
                _format_stat(block.get("regression_rmse_log_rv")),
            ]
        )

    caption = (
        f"Text-channel A/B sweep on the canonical training package. "
        f"Pooled across {len(data['seeds'])} seeds × {len(data['fold_ids'])} walk-forward folds "
        f"({data.get('epochs', '?')} epochs, head_mode={data.get('head_mode', '?')}, "
        f"regression_alpha={data.get('regression_alpha', '?')}, "
        f"encoder={data.get('text_encoder', '?')}). "
        f"{header.render()}"
    )

    return _render_table(
        title="Text-channel A/B — broadcast vs per-bar vs flat-MLP",
        headers=["Arm", "regime macro-F1", "RMSE log(RV)"],
        rows=rows,
        caption=caption,
        output_path=output_dir / "text_path_ab.png",
        align=["left", "right", "right"],
    )


# Cross-bank ladder. No JSON artefact on disk — the canonical source is
# wiki §6.14 ("Cross-bank ladder — final comparison"). Values transcribed
# from that table verbatim. If §6.14 changes, update this block and
# regenerate the figure.
_CROSS_BANK_ROWS: list[list[str]] = [
    ["FOMC-only (baseline)", "finbert_fed_adjacent", "0.4538", "[0.434, 0.469]", "5 × 4", "—"],
    ["Bundle A.2 arm B (weighted aux)", "finbert_fed_adjacent_xbank_aux_weighted", "0.4297", "[0.414, 0.447]", "5 × 4", "−0.024"],
    ["Bundle A.4 DAPT", "finbert_fed_adjacent_xbank_dapt", "0.4183", "σ 0.053", "1 × 4", "−0.036"],
    ["Bundle A.2 arm A (stance-masked aux)", "finbert_fed_adjacent_xbank_aux_stance_masked", "0.4066", "[0.391, 0.424]", "5 × 4", "−0.047"],
]


def build_cross_bank_table(header: ReproducibilityHeader, output_dir: Path) -> Path:
    caption = (
        "Cross-bank transfer ladder. Same Transformer cell (h=128, layers=2, "
        "dropout=0.2, lr=1e-3, weight-decay=1e-3, text-adapter-dim=128) on "
        f"{CANONICAL_TP_ID}; encoder swapped "
        "row-by-row. Every cross-bank-touched variant lands below the FOMC-only "
        "baseline CI, replicating the substitute-not-complement prior. "
        f"{header.render()}"
    )
    # Override the source artefact for this figure since it's transcribed
    # from wiki §6.14 rather than from a JSON file.
    caption = caption.replace(
        f"Source: {header.source_artefact}",
        "Source: fed-pulse.wiki/06_Deep_Learning_Roadmap.md §6.14",
    )

    return _render_table(
        title="Cross-bank transfer ladder",
        headers=["Variant", "Encoder alias", "macro-F1", "95% CI / Std", "Sample", "Δ vs baseline"],
        rows=_CROSS_BANK_ROWS,
        caption=caption,
        output_path=output_dir / "cross_bank_ladder.png",
        align=["left", "left", "right", "right", "right", "right"],
    )


def build_architecture_diagram(header: ReproducibilityHeader, output_dir: Path) -> Path:
    """Render the system-architecture diagram as a box-and-arrow PNG.

    Mirrors wiki §3 container diagram (Next.js dashboard / FastAPI
    backend / external data sources) plus the §3.3 component layout
    (forecaster + multi-axis classifier + analog retriever + trajectory
    heads off the shared text encoder).
    """

    title_font = _load_font(24, bold=True)
    box_font = _load_font(14, bold=True)
    sub_font = _load_font(12)
    caption_font = _load_font(13)

    width, height = 1280, 760
    img = Image.new("RGB", (width, height), _BG)
    draw = ImageDraw.Draw(img)

    draw.text(
        (_CELL_PAD_X, 16),
        "Fed Pulse — runtime architecture",
        fill=_TITLE_FG,
        font=title_font,
    )

    # Box layout. (x, y, w, h, title, subtitle, fill, outline)
    boxes: list[tuple[int, int, int, int, str, str, tuple[int, int, int], tuple[int, int, int]]] = [
        # Frontend column.
        (40, 96, 240, 88, "Next.js dashboard", "/, /settings, /history", (235, 244, 255), _ACCENT),
        # Backend column - API surface.
        (340, 96, 240, 88, "FastAPI /analyze", "+ /analyze/market /analogs /trajectory", (240, 244, 250), _BORDER),
        (340, 200, 240, 64, "/settings/checkpoints", "inference_contract sidecar", (240, 244, 250), _BORDER),
        # Backend column - heads.
        (640, 96, 280, 64, "Forecaster (dual head)", "regression(log RV) + 3-class CE", (250, 246, 240), (196, 128, 32)),
        (640, 176, 280, 64, "Multi-axis classifier", "stance / time / certainty", (250, 246, 240), (196, 128, 32)),
        (640, 256, 280, 64, "Analog retriever", "cross-bank DAPT encoder", (250, 246, 240), (196, 128, 32)),
        (640, 336, 280, 64, "Trajectory head", "LSTM / Transformer + baselines", (250, 246, 240), (196, 128, 32)),
        # Shared substrate.
        (640, 432, 280, 72, "Shared text encoder", "FinBERT-FedAdjacent (classifier role)\n+ xbank-DAPT (retrieval role)", (248, 240, 250), (128, 64, 196)),
        # External sources.
        (980, 96, 240, 64, "Hugging Face Hub", "training packages + checkpoints", (244, 244, 244), _MUTED_FG),
        (980, 176, 240, 64, "FRED + Yahoo", "macro-state + market series", (244, 244, 244), _MUTED_FG),
        (980, 256, 240, 64, "FOMC archives", "statements + minutes", (244, 244, 244), _MUTED_FG),
        # Storage.
        (340, 296, 240, 64, "Run history (SQLite)", "+ FOMC calendar", (244, 244, 244), _MUTED_FG),
    ]

    def _box(x: int, y: int, w: int, h: int, title: str, subtitle: str, fill: tuple, outline: tuple) -> None:
        draw.rectangle([x, y, x + w, y + h], fill=fill, outline=outline, width=2)
        tx, ty = _measure(draw, title, box_font)
        draw.text((x + (w - tx) // 2, y + 10), title, fill=_TITLE_FG, font=box_font)
        # Sub line(s) — split on newline so we can keep two-line subtitles tight.
        sub_y = y + 10 + ty + 6
        for line in subtitle.split("\n"):
            sx, _ = _measure(draw, line, sub_font)
            draw.text((x + (w - sx) // 2, sub_y), line, fill=_BODY_FG, font=sub_font)
            sub_y += 16

    for box in boxes:
        _box(*box)

    # Edges. (x1, y1, x2, y2)
    edges: list[tuple[int, int, int, int]] = [
        # Frontend → API.
        (280, 140, 340, 140),
        # Frontend → settings.
        (280, 160, 340, 220),
        # API → each head.
        (580, 130, 640, 128),
        (580, 140, 640, 208),
        (580, 150, 640, 288),
        (580, 160, 640, 368),
        # Heads → shared encoder.
        (780, 160, 780, 432),
        # API → run history.
        (460, 184, 460, 296),
        # API → HF Hub + FRED + FOMC.
        (920, 128, 980, 128),
        (920, 208, 980, 208),
        (920, 288, 980, 288),
    ]
    for x1, y1, x2, y2 in edges:
        draw.line([(x1, y1), (x2, y2)], fill=_MUTED_FG, width=2)
        # tiny arrowhead at (x2,y2).
        draw.polygon(
            [(x2, y2), (x2 - 8, y2 - 4), (x2 - 8, y2 + 4)],
            fill=_MUTED_FG,
        )

    # Legend.
    legend_y = 560
    draw.text((_CELL_PAD_X, legend_y), "Layers:", fill=_TITLE_FG, font=box_font)
    legend_items = [
        ("Frontend", (235, 244, 255), _ACCENT),
        ("API + storage", (240, 244, 250), _BORDER),
        ("Model heads", (250, 246, 240), (196, 128, 32)),
        ("Shared encoder", (248, 240, 250), (128, 64, 196)),
        ("External", (244, 244, 244), _MUTED_FG),
    ]
    lx = 110
    for label, fill, outline in legend_items:
        draw.rectangle([lx, legend_y - 4, lx + 18, legend_y + 14], fill=fill, outline=outline, width=2)
        draw.text((lx + 26, legend_y - 2), label, fill=_BODY_FG, font=sub_font)
        lw, _ = _measure(draw, label, sub_font)
        lx += 26 + lw + 24

    # Caption.
    caption = (
        "Runtime architecture (C4 container + component level). The Next.js "
        "dashboard calls four /analyze* endpoints in parallel; each backend "
        "head consumes the same shared text-encoder substrate (FinBERT-FedAdjacent "
        "as the classifier role; cross-bank DAPT as the retrieval role per ADR 0019). "
        f"{header.render()}"
    )
    caption = caption.replace(
        f"Source: {header.source_artefact}",
        "Source: fed-pulse.wiki/03_System_Architecture.md §1–§3",
    )
    caption_lines = _wrap_caption(draw, caption, caption_font, width - 2 * _CELL_PAD_X)
    cy = 620
    for line in caption_lines:
        draw.text((_CELL_PAD_X, cy), line, fill=_MUTED_FG, font=caption_font)
        cy += 20

    output_path = output_dir / "architecture.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, format="PNG", optimize=True)
    output_path.with_suffix(".caption.txt").write_text(caption + "\n", encoding="utf-8")
    return output_path


def _resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return Path(args.output_dir).resolve()
    if args.in_repo:
        return REPO_ROOT / "docs" / "figures"
    # Default: wiki repo next to the main checkout. Try a few plausible
    # parents — REPO_ROOT may resolve inside a worktree (`.claude/worktrees/<id>`)
    # in which case the sibling lookup needs to walk back to the original
    # checkout's parent before the wiki appears.
    candidates = [
        REPO_ROOT.parent / "fed-pulse.wiki",
        REPO_ROOT.parent.parent / "fed-pulse.wiki",
        REPO_ROOT.parent.parent.parent / "fed-pulse.wiki",
        REPO_ROOT.parent.parent.parent.parent / "fed-pulse.wiki",
    ]
    for wiki in candidates:
        if wiki.exists():
            return wiki / "assets" / "figures"
    # Fall back to the literal sibling path — the script will create it.
    return REPO_ROOT.parent / "fed-pulse.wiki" / "assets" / "figures"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override the output directory (defaults to ../fed-pulse.wiki/assets/figures).",
    )
    parser.add_argument(
        "--in-repo",
        action="store_true",
        help="Write figures to docs/figures/ inside the main repo instead of the wiki.",
    )
    parser.add_argument(
        "--only",
        choices=("architecture", "dual-head", "text-path-ab", "cross-bank"),
        default=None,
        help="Render only one figure (default: render all).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    output_dir = _resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    sha = _git_sha()
    print(f"[figures] commit={sha} output_dir={output_dir}")

    builders: list[tuple[str, str, callable]] = [
        (
            "architecture",
            "fed-pulse.wiki/03_System_Architecture.md",
            build_architecture_diagram,
        ),
        (
            "dual-head",
            "backend/artifacts/experiments/dual_head_comparison_canonical.json",
            build_dual_head_table,
        ),
        (
            "text-path-ab",
            "backend/artifacts/experiments/text_path_ab_canonical.json",
            build_text_path_ab_table,
        ),
        (
            "cross-bank",
            "fed-pulse.wiki/06_Deep_Learning_Roadmap.md §6.14",
            build_cross_bank_table,
        ),
    ]

    failures: list[tuple[str, BaseException]] = []
    for name, source, builder in builders:
        if args.only and args.only != name:
            continue
        header = ReproducibilityHeader(
            commit_sha=sha,
            training_package_id=CANONICAL_TP_ID,
            source_artefact=source,
        )
        try:
            path = builder(header, output_dir)
        except Exception as exc:  # noqa: BLE001 - figure builders surface their own context
            failures.append((name, exc))
            print(f"[figures] {name}: FAILED — {exc}", file=sys.stderr)
            continue
        print(f"[figures] {name}: {path.relative_to(output_dir.parent) if path.is_relative_to(output_dir.parent) else path}")

    if failures:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
