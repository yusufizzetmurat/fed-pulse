"""Pure-PIL confusion-matrix heatmap renderer.

The classification breakdown evaluator (#199) emits a confusion matrix
as a list-of-lists per trial; this module renders it to a PNG so the
appendix visuals + the UI cards can read the file directly. Matplotlib
is intentionally avoided -- the backend image already ships Pillow as
a transitive dependency and adding matplotlib would inflate the wheel
weight by ~30 MB for one heatmap.

Color ramp: a single-colour linear ramp from white (zero) to a saturated
primary at the matrix max. Per-cell text labels render in inverted
luminance so digits stay readable on both ends of the ramp.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw, ImageFont


def _ramp_color(value: float, vmax: int, base: tuple[int, int, int]) -> tuple[int, int, int]:
    """Linear-interpolate ``base`` color from white to base at vmax."""

    if vmax <= 0:
        return (255, 255, 255)
    t = min(1.0, float(value) / float(vmax))
    return (
        int(round(255 + t * (base[0] - 255))),
        int(round(255 + t * (base[1] - 255))),
        int(round(255 + t * (base[2] - 255))),
    )


def _text_color_for(bg: tuple[int, int, int]) -> tuple[int, int, int]:
    """Pick white or black text so the digits stay legible on the cell."""

    luma = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]
    return (255, 255, 255) if luma < 140 else (20, 20, 20)


def _load_font(size: int) -> ImageFont.ImageFont:
    """Try a small list of bundled DejaVu fallbacks, then PIL default."""

    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


_DEFAULT_BASE_COLOR: tuple[int, int, int] = (32, 96, 196)
_CELL_PX = 96
_MARGIN_PX = 84
_TITLE_PX = 36


def render_confusion_matrix_png(
    confusion_matrix: Sequence[Sequence[int]],
    output_path: Path | str,
    *,
    class_labels: Sequence[str] | None = None,
    title: str | None = None,
    base_color: tuple[int, int, int] = _DEFAULT_BASE_COLOR,
) -> Path:
    """Render ``confusion_matrix`` to a PNG heatmap.

    ``class_labels`` defaults to ``("0", "1", ...)`` when ``None``.
    ``title`` renders above the grid; ``None`` skips the title row.
    ``base_color`` is the cell ramp's saturated endpoint at the matrix
    max; cells linear-interpolate from white at zero to ``base_color``
    at vmax.

    Returns the path the PNG was written to.
    """

    if not confusion_matrix:
        raise ValueError("confusion_matrix must not be empty")
    n = len(confusion_matrix)
    if any(len(row) != n for row in confusion_matrix):
        raise ValueError(
            "confusion_matrix must be square; got rows of varying length"
        )

    labels = (
        list(class_labels)
        if class_labels is not None and len(class_labels) == n
        else [str(c) for c in range(n)]
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cell_px = _CELL_PX
    margin_px = _MARGIN_PX
    title_height = _TITLE_PX if title else 0
    img_width = margin_px + cell_px * n + margin_px // 2
    img_height = title_height + margin_px + cell_px * n + margin_px // 2

    img = Image.new("RGB", (img_width, img_height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = _load_font(14)
    label_font = _load_font(13)
    title_font = _load_font(16)

    if title:
        draw.text(
            (margin_px, 8),
            title,
            fill=(20, 20, 20),
            font=title_font,
        )

    grid_top = title_height + margin_px // 2 + 24
    grid_left = margin_px

    # Axis labels
    draw.text(
        (grid_left, grid_top - 20),
        "predicted →",
        fill=(80, 80, 80),
        font=label_font,
    )
    draw.text(
        (8, grid_top + (cell_px * n) // 2 - 8),
        "true",
        fill=(80, 80, 80),
        font=label_font,
    )

    vmax = max(max(row) for row in confusion_matrix)

    for r in range(n):
        for c in range(n):
            value = int(confusion_matrix[r][c])
            cell_x = grid_left + c * cell_px
            cell_y = grid_top + r * cell_px
            bg = _ramp_color(value, vmax, base_color)
            draw.rectangle(
                [cell_x, cell_y, cell_x + cell_px, cell_y + cell_px],
                fill=bg,
                outline=(220, 220, 220),
                width=1,
            )
            text_color = _text_color_for(bg)
            text = str(value)
            tw = draw.textlength(text, font=font)
            draw.text(
                (cell_x + (cell_px - tw) / 2, cell_y + cell_px / 2 - 8),
                text,
                fill=text_color,
                font=font,
            )

    # Column headers (predicted labels)
    for c in range(n):
        text = labels[c]
        tw = draw.textlength(text, font=label_font)
        draw.text(
            (grid_left + c * cell_px + (cell_px - tw) / 2, grid_top - 4),
            text,
            fill=(40, 40, 40),
            font=label_font,
        )
    # Row headers (true labels)
    for r in range(n):
        text = labels[r]
        tw = draw.textlength(text, font=label_font)
        draw.text(
            (grid_left - tw - 8, grid_top + r * cell_px + cell_px / 2 - 8),
            text,
            fill=(40, 40, 40),
            font=label_font,
        )

    img.save(output_path, format="PNG", optimize=True)
    return output_path


def aggregate_confusion_matrices(
    matrices: Sequence[Sequence[Sequence[int]]],
) -> list[list[int]]:
    """Element-wise sum a list of square confusion matrices.

    The mean per-tier "headline" confusion matrix the appendix renders
    is the elementwise sum across every (seed, fold) trial -- support is
    additive across trials, so the visual reads as a population-level
    count rather than a per-trial average. Callers that want a mean
    visualisation can divide by ``len(matrices)`` themselves.
    """

    if not matrices:
        raise ValueError("matrices must not be empty")
    n = len(matrices[0])
    for m in matrices:
        if len(m) != n or any(len(row) != n for row in m):
            raise ValueError("all confusion matrices must share the same shape")
    out = [[0 for _ in range(n)] for _ in range(n)]
    for m in matrices:
        for r in range(n):
            for c in range(n):
                out[r][c] += int(m[r][c])
    return out


__all__ = [
    "render_confusion_matrix_png",
    "aggregate_confusion_matrices",
]
