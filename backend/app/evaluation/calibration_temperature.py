"""Post-hoc temperature scaling for the vol-regime classifier.

Neural-network classifiers are typically overconfident — when the model
reports a softmax probability of 0.8 it is often only right 6-7 times
out of 10 (Guo et al. 2017, "On Calibration of Modern Neural Networks").
Temperature scaling fits a single scalar ``T > 0`` against held-out
validation logits, then applies ``softmax(logits / T)`` at inference.
The scalar is invariant under argmax (so accuracy and macro-F1 are
unchanged) but reshapes the probability mass so reported confidences
match empirical frequencies.

The module exposes:

- ``fit_temperature(val_logits, val_targets)`` -- gradient-descent
  optimum on cross-entropy over the calibration partition.
- ``apply_temperature(logits, T)`` -- the inference-time transform.
- ``reliability_curve(probs, targets, n_bins=10)`` -- per-bin
  empirical accuracy versus reported confidence, the input to the
  reliability-diagram plot.
- ``expected_calibration_error(probs, targets, n_bins=10)`` -- the
  standard ECE scalar.
- ``render_reliability_diagram_png(curve, output_path)`` -- pure-PIL
  reliability plot for the appendix; matches the
  ``confusion_matrix_render`` aesthetic.

No new third-party dependencies: temperature fit uses pure PyTorch
already in the stack; the renderer uses Pillow which is already a
transitive dep.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont


# ---------------------------------------------------------------------------
# Temperature fitting
# ---------------------------------------------------------------------------


def fit_temperature(
    val_logits: torch.Tensor,
    val_targets: torch.Tensor,
    *,
    max_iter: int = 200,
    lr: float = 0.01,
    initial_T: float = 1.0,
    eps: float = 1e-4,
) -> float:
    """Optimise the scalar temperature against the validation cross-entropy.

    ``val_logits`` shape: ``(N, n_classes)``; raw logits, NOT softmaxed.
    ``val_targets`` shape: ``(N,)`` of class indices.

    Uses LBFGS the way the Guo et al. reference implementation does:
    a single scalar parameter, log-parameterised so ``T`` stays
    strictly positive. Returns the fitted ``T`` as a Python float.
    """

    if val_logits.shape[0] != val_targets.shape[0]:
        raise ValueError(
            f"val_logits ({val_logits.shape[0]} rows) and val_targets "
            f"({val_targets.shape[0]} rows) must have the same length"
        )
    if val_logits.shape[0] == 0:
        return float(initial_T)
    if val_logits.dim() != 2:
        raise ValueError(f"val_logits must be 2-D; got shape {tuple(val_logits.shape)}")
    if val_targets.dim() != 1:
        raise ValueError(f"val_targets must be 1-D; got shape {tuple(val_targets.shape)}")

    # log-T parameterisation -- T = exp(log_T) keeps T > 0 automatically,
    # which matters because softmax(z / T) blows up if T crosses zero.
    log_T = torch.tensor(
        float(torch.log(torch.tensor(float(initial_T)))),
        requires_grad=True,
    )
    optimiser = torch.optim.LBFGS([log_T], lr=lr, max_iter=max_iter)
    targets = val_targets.long()
    logits = val_logits.detach().float()

    def _closure() -> torch.Tensor:
        optimiser.zero_grad()
        T = torch.exp(log_T).clamp(min=eps)
        loss = F.cross_entropy(logits / T, targets)
        loss.backward()
        return loss

    optimiser.step(_closure)
    return float(torch.exp(log_T).detach().clamp(min=eps).item())


def apply_temperature(logits: torch.Tensor, T: float) -> torch.Tensor:
    """Return ``softmax(logits / T)`` over the last axis.

    ``T == 1.0`` is the identity (uncalibrated softmax). ``T > 1`` makes
    the distribution softer (lower max probability). ``T < 1`` makes it
    sharper. Argmax is invariant under any positive ``T``.
    """

    if T <= 0.0:
        raise ValueError(f"temperature must be > 0; got {T}")
    return F.softmax(logits / T, dim=-1)


# ---------------------------------------------------------------------------
# Reliability curve + ECE
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ReliabilityBin:
    """One bin of the reliability diagram."""

    lower: float
    upper: float
    count: int
    confidence_mean: float  # mean of max-class probability inside this bin
    accuracy: float  # fraction of rows whose argmax matched target

    def to_dict(self) -> dict[str, float | int]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class ReliabilityCurve:
    """The reliability curve plus the aggregate ECE scalar."""

    bins: tuple[ReliabilityBin, ...]
    ece: float
    n_rows: int

    def to_dict(self) -> dict[str, object]:
        return {
            "bins": [b.to_dict() for b in self.bins],
            "ece": float(self.ece),
            "n_rows": int(self.n_rows),
        }


def reliability_curve(
    probs: Sequence[Sequence[float]],
    targets: Sequence[int],
    *,
    n_bins: int = 10,
) -> ReliabilityCurve:
    """Bin the model's max-class confidence and report per-bin accuracy.

    The standard reliability diagram is: x-axis = predicted-class
    confidence (max softmax), y-axis = empirical accuracy. A perfectly
    calibrated model sits on the y = x diagonal.

    ``probs`` rows are per-class probability vectors summing to 1.
    ``targets`` are class indices.
    """

    if len(probs) != len(targets):
        raise ValueError("probs and targets must have the same length")
    n_rows = len(probs)
    if n_rows == 0:
        return ReliabilityCurve(bins=(), ece=0.0, n_rows=0)
    if n_bins <= 0:
        raise ValueError(f"n_bins must be positive; got {n_bins}")

    # Argmax + max-confidence per row.
    confidences: list[float] = []
    correct: list[int] = []
    for row, target in zip(probs, targets):
        # Tolerate either lists or 1-D tensors / numpy.
        row_list = list(row)
        max_c = max(row_list)
        pred = row_list.index(max_c)
        confidences.append(float(max_c))
        correct.append(1 if int(target) == pred else 0)

    bins: list[ReliabilityBin] = []
    bin_edges = [i / n_bins for i in range(n_bins + 1)]
    ece_total = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        # Closed-open bins except the last bin which is closed-closed
        # so confidence = 1.0 lands somewhere.
        if hi == 1.0:
            in_bin = [
                (c, k) for c, k in zip(confidences, correct) if lo <= c <= hi
            ]
        else:
            in_bin = [
                (c, k) for c, k in zip(confidences, correct) if lo <= c < hi
            ]
        count = len(in_bin)
        if count == 0:
            bins.append(
                ReliabilityBin(
                    lower=lo,
                    upper=hi,
                    count=0,
                    confidence_mean=0.0,
                    accuracy=0.0,
                )
            )
            continue
        conf_mean = sum(c for c, _ in in_bin) / count
        acc = sum(k for _, k in in_bin) / count
        bins.append(
            ReliabilityBin(
                lower=lo,
                upper=hi,
                count=count,
                confidence_mean=float(conf_mean),
                accuracy=float(acc),
            )
        )
        ece_total += (count / n_rows) * abs(conf_mean - acc)
    return ReliabilityCurve(bins=tuple(bins), ece=float(ece_total), n_rows=n_rows)


def expected_calibration_error(
    probs: Sequence[Sequence[float]],
    targets: Sequence[int],
    *,
    n_bins: int = 10,
) -> float:
    """Shortcut to ECE only when the per-bin curve is not needed."""

    return reliability_curve(probs, targets, n_bins=n_bins).ece


# ---------------------------------------------------------------------------
# Reliability diagram renderer (pure PIL, matches confusion_matrix_render)
# ---------------------------------------------------------------------------


_PLOT_PX = 480
_MARGIN_PX = 72
_TITLE_PX = 36


def _load_font(size: int) -> ImageFont.ImageFont:
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


def render_reliability_diagram_png(
    curve: ReliabilityCurve,
    output_path: Path | str,
    *,
    title: str | None = None,
    base_color: tuple[int, int, int] = (32, 96, 196),
) -> Path:
    """Render a reliability diagram to PNG.

    Bars: per-bin empirical accuracy. Reference line: y = x diagonal.
    The ECE scalar overlays as text in the top-left corner.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    title_height = _TITLE_PX if title else 0
    img_width = _MARGIN_PX + _PLOT_PX + _MARGIN_PX // 2
    img_height = title_height + _MARGIN_PX + _PLOT_PX + _MARGIN_PX

    img = Image.new("RGB", (img_width, img_height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = _load_font(13)
    title_font = _load_font(16)

    if title:
        draw.text((_MARGIN_PX, 10), title, fill=(20, 20, 20), font=title_font)

    plot_left = _MARGIN_PX
    plot_top = title_height + _MARGIN_PX // 2
    plot_right = plot_left + _PLOT_PX
    plot_bottom = plot_top + _PLOT_PX

    # Axes
    draw.rectangle(
        [plot_left, plot_top, plot_right, plot_bottom],
        outline=(180, 180, 180),
        width=1,
    )
    # y = x reference line
    draw.line(
        [(plot_left, plot_bottom), (plot_right, plot_top)],
        fill=(200, 50, 50),
        width=2,
    )

    # Bars: one per bin
    if curve.bins:
        bin_width_px = _PLOT_PX / len(curve.bins)
        for i, b in enumerate(curve.bins):
            if b.count == 0:
                continue
            x0 = plot_left + i * bin_width_px
            x1 = x0 + bin_width_px - 2
            y_bar_top = plot_bottom - b.accuracy * _PLOT_PX
            draw.rectangle(
                [x0 + 1, y_bar_top, x1, plot_bottom],
                fill=base_color + (180,) if False else base_color,
                outline=(40, 40, 40),
                width=1,
            )
            # Confidence-mean tick mark on the same bin
            y_conf = plot_bottom - b.confidence_mean * _PLOT_PX
            draw.line(
                [(x0 + 1, y_conf), (x1, y_conf)],
                fill=(60, 60, 60),
                width=1,
            )

    # Axis tick labels
    for i in range(0, 11, 2):
        # y-axis (accuracy)
        y = plot_bottom - (i / 10) * _PLOT_PX
        draw.text(
            (plot_left - 28, y - 7),
            f"{i / 10:.1f}",
            fill=(80, 80, 80),
            font=font,
        )
        # x-axis (confidence)
        x = plot_left + (i / 10) * _PLOT_PX
        draw.text(
            (x - 8, plot_bottom + 6),
            f"{i / 10:.1f}",
            fill=(80, 80, 80),
            font=font,
        )

    draw.text(
        (plot_left, plot_bottom + 28),
        "predicted confidence (max softmax)",
        fill=(60, 60, 60),
        font=font,
    )
    draw.text(
        (8, plot_top + 4),
        "empirical accuracy",
        fill=(60, 60, 60),
        font=font,
    )

    # ECE annotation
    ece_text = f"ECE = {curve.ece:.4f}   n = {curve.n_rows}"
    draw.text(
        (plot_left + 8, plot_top + 8),
        ece_text,
        fill=(20, 20, 20),
        font=font,
    )

    img.save(output_path, format="PNG", optimize=True)
    return output_path


__all__ = [
    "fit_temperature",
    "apply_temperature",
    "ReliabilityBin",
    "ReliabilityCurve",
    "reliability_curve",
    "expected_calibration_error",
    "render_reliability_diagram_png",
]
