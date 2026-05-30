"""Post-hoc temperature calibration driver for the vol-regime classifier.

The training-time eval reports accuracy + macro-F1 + confusion matrix
but does not persist raw logits, so we cannot fit a temperature
scaling parameter from the existing forecaster_sweep_results.json
artefacts alone. This script:

1. Loads a saved checkpoint (`backend/models/forecaster_best.pt` by
   default).
2. Loads the same training-package walk-forward split the checkpoint
   was trained against.
3. Re-runs inference on the held-out validation partition.
4. Fits the temperature scalar that minimises cross-entropy on the
   val partition.
5. Reports pre- and post-calibration ECE + reliability diagrams.
6. Saves the fitted T into a manifest next to the checkpoint so the
   inference path can apply it at serving time.

Usage::

    python scripts/calibrate_regime_classifier.py \\
        --training-package-id <pkg> \\
        --checkpoint-path /app/models/forecaster_best.pt \\
        --fold wf_fold_3 \\
        --output-dir /data/artifacts/calibration/<pkg>/

For appendix visuals across the whole tier, pass ``--all-folds`` to
iterate every available fold and emit one reliability diagram per fold.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from app.evaluation.calibration_temperature import (
    apply_platt_per_class,
    apply_temperature,
    brier_score,
    expected_calibration_error,
    fit_platt_per_class,
    fit_temperature,
    negative_log_likelihood,
    reliability_curve,
    render_reliability_diagram_png,
)
from app.training.checkpoint import _coerce_payload_config
from app.training.loaders import load_walk_forward_split

CALIBRATION_METHODS = ("temperature", "platt", "both")


def _resolve_package_dir(training_package_id: str) -> Path:
    for c in (
        Path(f"/data/processed/{training_package_id}"),
        Path(f"data/processed/{training_package_id}"),
        Path(f"backend/data/processed/{training_package_id}"),
    ):
        if (c / "events.parquet").exists():
            return c
    raise FileNotFoundError(
        f"events.parquet missing for {training_package_id}"
    )


def _load_checkpoint_payload(path: Path, device: torch.device) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"checkpoint missing: {path}")
    return torch.load(path, map_location=device, weights_only=False)


def _collect_logits_and_targets(
    model: torch.nn.Module,
    sequence_groups,
    *,
    device: torch.device,
    close_scale: float,
    rich_feature_scaler,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Re-run inference on a partition; return raw logits + targets."""

    from app.training.loop import _build_partition_tensors

    output_mode = str(getattr(model, "output_mode", "regression"))
    n_classes = int(getattr(model, "n_classes", 3) or 3)
    quantiles = tuple(getattr(model, "vol_regime_quantiles", ()))
    if output_mode != "classification":
        raise ValueError(
            f"checkpoint output_mode={output_mode!r} -- temperature scaling "
            "only applies to classification mode"
        )
    if not quantiles:
        raise ValueError(
            "checkpoint has no vol_regime_quantiles -- cannot recover "
            "class boundary; this looks like an un-trained classifier"
        )
    text_in_dim = int(getattr(model, "text_embedding_dim", 0) or 0)

    x, y, _, text_emb, text_missing = _build_partition_tensors(
        sequence_groups,
        fallback_text_in_dim=text_in_dim,
        close_scale=close_scale,
        output_mode="classification",
        vol_regime_quantiles=quantiles,
    )
    if x is None or y is None:
        return torch.empty(0, n_classes), torch.empty(0, dtype=torch.long)

    # Apply rich-feature scaler to match training-time normalisation.
    if rich_feature_scaler is not None:
        from app.training.loaders import apply_rich_feature_scaler_tensor

        x = apply_rich_feature_scaler_tensor(x, rich_feature_scaler)

    x = x.to(device)
    y = y.to(device)

    model.eval()
    with torch.no_grad():
        kwargs: dict[str, torch.Tensor] = {}
        if getattr(model, "credibility_features", False):
            dim = int(getattr(model, "credibility_dim", 4))
            kwargs["credibility"] = torch.zeros(
                (x.shape[0], dim), dtype=torch.float32, device=device
            )
        if text_emb is not None and getattr(model, "_text_path_active", False):
            kwargs["text_embedding"] = text_emb.to(device)
            if text_missing is not None:
                kwargs["text_embedding_missing"] = text_missing.to(device)
        logits = model(x, **kwargs)

    return logits.detach().cpu(), y.detach().cpu()


def _build_model_from_payload(payload: dict[str, Any], device: torch.device):
    """Rebuild the trained model from a saved payload."""

    from app.models.factory import build_forecaster
    from app.training.checkpoint import _load_state_dict_loose

    config = _coerce_payload_config(payload)
    model = build_forecaster(config).to(device)
    _load_state_dict_loose(model, payload["model_state_dict"], "calibration")
    return model


def _metric_block(
    probs: list[list[float]],
    targets: list[int],
    *,
    n_classes: int,
) -> dict[str, float]:
    """Pack ECE + Brier + NLL into a serialisable block."""

    return {
        "ece": float(expected_calibration_error(probs, targets, n_bins=10)),
        "brier": float(brier_score(probs, targets, n_classes=n_classes)),
        "nll": float(negative_log_likelihood(probs, targets)),
    }


def _fit_method(
    method: str,
    val_logits: torch.Tensor,
    val_targets: torch.Tensor,
    targets_list: list[int],
    n_classes: int,
) -> dict[str, Any]:
    """Run the requested calibrator(s) on val logits.

    Returns a dict with optional ``temperature`` / ``platt_params`` and
    per-method ``post_probs`` / ``post_metrics`` / ``post_curve``.
    When fitting ``both``, Platt sees the temperature-scaled logits so
    the two layers compose at inference (temperature first, Platt
    second) and reproduce the val-partition fit exactly.
    """

    result: dict[str, Any] = {}
    if method in ("temperature", "both"):
        T = fit_temperature(val_logits, val_targets)
        post_probs = apply_temperature(val_logits, T).tolist()
        result["temperature"] = float(T)
        result["post_probs_T"] = post_probs
        result["post_curve_T"] = reliability_curve(post_probs, targets_list, n_bins=10)
        result["post_metrics_T"] = _metric_block(post_probs, targets_list, n_classes=n_classes)

    if method in ("platt", "both"):
        T_existing = result.get("temperature")
        logits_for_platt = (
            val_logits / float(T_existing)
            if (method == "both" and T_existing is not None)
            else val_logits
        )
        params = fit_platt_per_class(logits_for_platt, val_targets, n_classes=n_classes)
        post_probs_p = apply_platt_per_class(logits_for_platt, params).tolist()
        result["platt_params"] = params
        result["post_probs_platt"] = post_probs_p
        result["post_curve_platt"] = reliability_curve(post_probs_p, targets_list, n_bins=10)
        result["post_metrics_platt"] = _metric_block(post_probs_p, targets_list, n_classes=n_classes)

    return result


def _build_manifest(
    fold_id: str,
    method: str,
    n_val_rows: int,
    n_classes: int,
    pre_metrics: dict[str, float],
    pre_curve_dict: dict[str, object],
    fit: dict[str, Any],
) -> dict[str, Any]:
    """Build the per-fold calibration_manifest.json payload."""

    payload: dict[str, Any] = {
        "fold_id": fold_id,
        "method": method,
        "n_val_rows": n_val_rows,
        "n_classes": n_classes,
        "pre_metrics": pre_metrics,
        "pre_curve": pre_curve_dict,
    }
    if "temperature" in fit:
        payload["temperature"] = float(fit["temperature"])
        payload["post_metrics_temperature"] = fit["post_metrics_T"]
        payload["post_curve_temperature"] = fit["post_curve_T"].to_dict()
        # Back-compat: keep the legacy flat keys consumed by older
        # operator scripts and dashboards.
        payload["pre_ece"] = float(pre_metrics["ece"])
        payload["post_ece"] = float(fit["post_metrics_T"]["ece"])
        payload["post_curve"] = fit["post_curve_T"].to_dict()
    if "platt_params" in fit:
        payload["platt_a"] = [float(a) for a, _ in fit["platt_params"]]
        payload["platt_b"] = [float(b) for _, b in fit["platt_params"]]
        payload["post_metrics_platt"] = fit["post_metrics_platt"]
        payload["post_curve_platt"] = fit["post_curve_platt"].to_dict()
    return payload


def _build_summary_entry(
    n_val_rows: int,
    pre_metrics: dict[str, float],
    fit: dict[str, Any],
) -> dict[str, float]:
    """Flat per-fold metrics block for the global calibration_summary.json."""

    entry: dict[str, float] = {
        "n_val_rows": float(n_val_rows),
        "pre_ece": float(pre_metrics["ece"]),
        "pre_brier": float(pre_metrics["brier"]),
        "pre_nll": float(pre_metrics["nll"]),
    }
    if "temperature" in fit:
        entry["temperature"] = float(fit["temperature"])
        entry["post_ece"] = float(fit["post_metrics_T"]["ece"])
        entry["post_brier_temperature"] = float(fit["post_metrics_T"]["brier"])
        entry["post_nll_temperature"] = float(fit["post_metrics_T"]["nll"])
    if "platt_params" in fit:
        entry["post_ece_platt"] = float(fit["post_metrics_platt"]["ece"])
        entry["post_brier_platt"] = float(fit["post_metrics_platt"]["brier"])
        entry["post_nll_platt"] = float(fit["post_metrics_platt"]["nll"])
    return entry


def _build_sidecar(
    fold_id: str,
    method: str,
    n_val_rows: int,
    n_classes: int,
    pre_metrics: dict[str, float],
    fit: dict[str, Any],
) -> dict[str, Any]:
    """Build the ``{checkpoint}.calibration.json`` payload."""

    payload: dict[str, Any] = {
        "fold_id": fold_id,
        "method": method,
        "n_val_rows": n_val_rows,
        "n_classes": n_classes,
        "pre_metrics": pre_metrics,
    }
    if "temperature" in fit:
        payload["temperature"] = float(fit["temperature"])
        payload["post_metrics_temperature"] = fit["post_metrics_T"]
    if "platt_params" in fit:
        payload["platt_a"] = [float(a) for a, _ in fit["platt_params"]]
        payload["platt_b"] = [float(b) for _, b in fit["platt_params"]]
        payload["post_metrics_platt"] = fit["post_metrics_platt"]
    return payload


def _render_fold_diagrams(
    fold_id: str,
    fold_dir: Path,
    pre_curve: Any,
    pre_metrics: dict[str, float],
    fit: dict[str, Any],
) -> None:
    """Write the uncalibrated + calibrated reliability PNGs for one fold."""

    render_reliability_diagram_png(
        pre_curve,
        fold_dir / "reliability_pre.png",
        title=f"{fold_id} · uncalibrated · ECE={pre_metrics['ece']:.4f}",
    )
    if "post_curve_T" in fit:
        T = fit.get("temperature")
        ece_post = fit["post_metrics_T"]["ece"]
        title = (
            f"{fold_id} · T={float(T):.3f} · ECE={ece_post:.4f}"
            if T is not None
            else f"{fold_id} · ECE={ece_post:.4f}"
        )
        render_reliability_diagram_png(
            fit["post_curve_T"], fold_dir / "reliability_post.png", title=title
        )
    if "post_curve_platt" in fit:
        ece_p = fit["post_metrics_platt"]["ece"]
        render_reliability_diagram_png(
            fit["post_curve_platt"],
            fold_dir / "reliability_post_platt.png",
            title=f"{fold_id} · platt · ECE={ece_p:.4f}",
        )


def _log_fold(fold_id: str, pre_metrics: dict[str, float], fit: dict[str, Any], n_val: int) -> None:
    """One-line operator log for the fold's calibration result."""

    if "temperature" in fit:
        T = float(fit["temperature"])
        direction = "softened" if T > 1.0 else "sharpened"
        t_msg = f"T={T:.3f} ({direction}); "
        post_ece = fit["post_metrics_T"]["ece"]
    elif "post_metrics_platt" in fit:
        t_msg = ""
        post_ece = fit["post_metrics_platt"]["ece"]
    else:
        t_msg = ""
        post_ece = pre_metrics["ece"]
    print(
        f"[calibrate] {fold_id}: {t_msg}ECE {pre_metrics['ece']:.4f} -> {post_ece:.4f}; "
        f"n_val={n_val}",
        flush=True,
    )


def _process_fold(
    fold_id: str,
    args: argparse.Namespace,
    *,
    model: torch.nn.Module,
    device: torch.device,
    close_scale: float,
    rich_feature_scaler: Any,
    output_root: Path,
) -> tuple[dict[str, float], dict[str, Any]] | None:
    """Run one fold end-to-end. Returns ``(summary_entry, sidecar_payload)``
    or ``None`` when the fold is skipped (load failure or empty val
    partition)."""

    try:
        split = load_walk_forward_split(args.training_package_id, fold_id=fold_id)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[calibrate] {fold_id}: load failed ({exc}); skipping", flush=True)
        return None

    val_logits, val_targets = _collect_logits_and_targets(
        model,
        split.val,
        device=device,
        close_scale=close_scale,
        rich_feature_scaler=rich_feature_scaler,
    )
    if val_logits.numel() == 0:
        print(f"[calibrate] {fold_id}: empty val partition; skipping", flush=True)
        return None

    n_classes = int(val_logits.shape[-1])
    targets_list = val_targets.tolist()
    pre_probs = torch.softmax(val_logits, dim=-1).tolist()
    pre_curve = reliability_curve(pre_probs, targets_list, n_bins=10)
    pre_metrics = _metric_block(pre_probs, targets_list, n_classes=n_classes)
    n_val = int(val_logits.shape[0])

    fold_dir = output_root / fold_id
    fold_dir.mkdir(parents=True, exist_ok=True)

    fit = _fit_method(args.method, val_logits, val_targets, targets_list, n_classes)
    _render_fold_diagrams(fold_id, fold_dir, pre_curve, pre_metrics, fit)

    manifest_payload = _build_manifest(
        fold_id, args.method, n_val, n_classes, pre_metrics, pre_curve.to_dict(), fit
    )
    (fold_dir / "calibration_manifest.json").write_text(
        json.dumps(manifest_payload, indent=2)
    )

    summary_entry = _build_summary_entry(n_val, pre_metrics, fit)
    sidecar_payload = _build_sidecar(
        fold_id, args.method, n_val, n_classes, pre_metrics, fit
    )
    _log_fold(fold_id, pre_metrics, fit, n_val)
    return summary_entry, sidecar_payload


def _calibration_sidecar_path(checkpoint_path: Path) -> Path:
    """Return the canonical sidecar path next to a checkpoint.

    Mirrors :func:`forecaster._conformal_manifest_for` -- the regime
    classifier inference path reads ``{stem}.calibration.json`` next to
    the checkpoint via :func:`with_name`, which works on Python 3.11 and
    3.12+ alike (``with_suffix`` rejects multi-dot suffixes on 3.11).
    """

    return checkpoint_path.with_name(checkpoint_path.stem + ".calibration.json")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--training-package-id", required=True)
    p.add_argument(
        "--checkpoint-path",
        type=Path,
        default=Path("/app/models/forecaster_best.pt"),
    )
    p.add_argument(
        "--fold",
        default="wf_fold_3",
        help="Walk-forward fold to evaluate (val partition is used).",
    )
    p.add_argument(
        "--all-folds",
        action="store_true",
        help="Iterate every fold in the manifest; emit per-fold diagrams.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/data/artifacts/calibration"),
    )
    p.add_argument(
        "--method",
        choices=CALIBRATION_METHODS,
        default="temperature",
        help=(
            "Calibrator to fit: 'temperature' (single scalar; preserves "
            "argmax), 'platt' (per-class one-vs-rest sigmoid; tighter "
            "per-class fit), or 'both' (fit both and record both in the "
            "sidecar; inference path applies temperature -> platt when "
            "both are present)."
        ),
    )
    p.add_argument(
        "--no-sidecar",
        action="store_true",
        help=(
            "Skip writing the {checkpoint}.calibration.json sidecar that "
            "the inference path reads at serving time."
        ),
    )
    p.add_argument("--device", default=None)
    args = p.parse_args(argv)

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    pkg_dir = _resolve_package_dir(args.training_package_id)
    payload = _load_checkpoint_payload(args.checkpoint_path, device)
    model = _build_model_from_payload(payload, device)
    close_scale = float(payload.get("close_scale", 10000.0))
    from app.models.config import RichFeatureScalerParams

    rich_feature_scaler = (
        RichFeatureScalerParams.from_dict(payload.get("rich_feature_scaler"))
        if isinstance(payload.get("rich_feature_scaler"), dict)
        else None
    )

    output_root = args.output_dir / args.training_package_id
    output_root.mkdir(parents=True, exist_ok=True)

    folds_to_run = []
    if args.all_folds:
        manifest = json.loads(
            (pkg_dir / "fold_manifest_expanding_walk_forward.json").read_text()
        )
        for f in manifest.get("folds", []):
            folds_to_run.append(str(f.get("fold_id")))
    else:
        folds_to_run = [args.fold]

    per_fold_summary: dict[str, dict[str, float]] = {}
    last_sidecar_payload: dict[str, Any] | None = None

    for fold_id in folds_to_run:
        result = _process_fold(
            fold_id,
            args,
            model=model,
            device=device,
            close_scale=close_scale,
            rich_feature_scaler=rich_feature_scaler,
            output_root=output_root,
        )
        if result is None:
            continue
        summary_entry, sidecar_payload = result
        per_fold_summary[fold_id] = summary_entry
        # All-folds overwrites on each iteration so the sidecar reflects
        # the chronologically-latest fold; single-fold writes once.
        last_sidecar_payload = sidecar_payload

    if last_sidecar_payload is not None and not args.no_sidecar:
        sidecar_path = _calibration_sidecar_path(args.checkpoint_path)
        sidecar_path.write_text(json.dumps(last_sidecar_payload, indent=2))
        print(f"[calibrate] sidecar -> {sidecar_path}", flush=True)

    summary_path = output_root / "calibration_summary.json"
    summary_path.write_text(json.dumps(per_fold_summary, indent=2))
    print(f"[calibrate] summary -> {summary_path}", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
