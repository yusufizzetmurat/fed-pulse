"""Round 2c (#234) ablation: ``--no-class-weights`` skips the A1 fit.

The intent of the per-fold inverse-frequency class weighting in
``app.training.loop.train_model`` is to compensate for vol-regime
support imbalance on the train slice. With the PR #233 weighted-CE
val-loss fix in place, the next question is whether the weighting
itself still moves macro-F1 — the train slice is roughly balanced by
quantile-cutoff construction, so the weights may be a near no-op.

The tests below pin the toggle semantics (default on, opt-out via
``--no-class-weights``) at the source level so the ablation has a
stable knob to flip in re-runs.
"""

from __future__ import annotations

import inspect
import textwrap
from pathlib import Path

import pytest


_LOOP_PATH = Path(__file__).resolve().parents[2] / "backend" / "app" / "training" / "loop.py"
_TRAIN_FORECASTER_PATH = (
    Path(__file__).resolve().parents[2] / "backend" / "app" / "train_forecaster.py"
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_train_model_accepts_use_class_weights_kwarg() -> None:
    """The signature must carry ``use_class_weights`` so the CLI flag
    has a place to land."""

    source = _read(_LOOP_PATH)
    # Two surface guards:
    #  1. The signature line has the kwarg with a default of True.
    assert "use_class_weights: bool = True" in source, (
        "train_model is missing the use_class_weights kwarg (default True)"
    )
    #  2. The fit_class_weights call is guarded behind the flag.
    assert "if use_class_weights:" in source, (
        "fit_class_weights call is not guarded by the use_class_weights flag"
    )


def test_class_weight_off_path_emits_empty_tuple() -> None:
    """When ``use_class_weights=False`` the loop must short-circuit to
    an empty weight tuple rather than fitting and silently zeroing them."""

    source = _read(_LOOP_PATH)
    # The else-branch under the use_class_weights guard sets the
    # fitted_class_weights tuple to () so CrossEntropyLoss(weight=None)
    # runs and the train-side reduction stays standard mean.
    expected = textwrap.dedent(
        """\
        if use_class_weights:
                    fitted_class_weights = fit_class_weights(
                        train_forward_vols,
                        fitted_quantiles,
                        n_classes=n_classes_active,
                    )
                else:
                    fitted_class_weights = ()
        """
    ).strip()
    assert expected in source, (
        "use_class_weights=False path does not zero the fitted class weights"
    )


def test_train_forecaster_exposes_no_class_weights_cli_flag() -> None:
    """The CLI must expose ``--no-class-weights`` so the ablation can be
    driven from the sweep harness without code edits."""

    source = _read(_TRAIN_FORECASTER_PATH)
    assert "\"--no-class-weights\"" in source, "missing --no-class-weights CLI flag"
    assert "dest=\"use_class_weights\"" in source, (
        "--no-class-weights does not write into args.use_class_weights"
    )
    assert "parser.set_defaults(use_class_weights=True)" in source, (
        "default for --no-class-weights is not 'class weights on'"
    )


def test_run_single_training_threads_use_class_weights() -> None:
    """The flag must be threaded through ``_run_single_training`` so the
    worker payload + every call site forwards the user's choice."""

    source = _read(_TRAIN_FORECASTER_PATH)
    # Function signature carries the kwarg with default True (back-compat).
    assert "use_class_weights: bool = True" in source
    # Each train_model call inside _run_single_training forwards the flag.
    occurrences = source.count("use_class_weights=use_class_weights")
    assert occurrences >= 4, (
        "use_class_weights is not forwarded into every train_model call site "
        f"(found {occurrences}, expected >= 4)"
    )
    # Worker payload passes the flag through to subprocesses.
    assert "\"use_class_weights\": bool(getattr(args, \"use_class_weights\", True))" in source, (
        "worker payload does not carry use_class_weights"
    )
