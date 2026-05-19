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
    apply_temperature,
    expected_calibration_error,
    fit_temperature,
    reliability_curve,
    render_reliability_diagram_png,
)
from app.training.checkpoint import _coerce_payload_config
from app.training.loaders import load_walk_forward_split


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

    for fold_id in folds_to_run:
        try:
            split = load_walk_forward_split(
                args.training_package_id,
                fold_id=fold_id,
            )
        except (FileNotFoundError, ValueError) as exc:
            print(f"[calibrate] {fold_id}: load failed ({exc}); skipping", flush=True)
            continue

        val_logits, val_targets = _collect_logits_and_targets(
            model,
            split.val,
            device=device,
            close_scale=close_scale,
            rich_feature_scaler=rich_feature_scaler,
        )
        if val_logits.numel() == 0:
            print(f"[calibrate] {fold_id}: empty val partition; skipping", flush=True)
            continue

        T = fit_temperature(val_logits, val_targets)

        pre_probs = torch.softmax(val_logits, dim=-1).tolist()
        post_probs = apply_temperature(val_logits, T).tolist()
        targets_list = val_targets.tolist()

        pre_curve = reliability_curve(pre_probs, targets_list, n_bins=10)
        post_curve = reliability_curve(post_probs, targets_list, n_bins=10)
        pre_ece = expected_calibration_error(pre_probs, targets_list)
        post_ece = expected_calibration_error(post_probs, targets_list)

        fold_dir = output_root / fold_id
        fold_dir.mkdir(parents=True, exist_ok=True)
        render_reliability_diagram_png(
            pre_curve,
            fold_dir / "reliability_pre.png",
            title=f"{fold_id} · uncalibrated · ECE={pre_ece:.4f}",
        )
        render_reliability_diagram_png(
            post_curve,
            fold_dir / "reliability_post.png",
            title=f"{fold_id} · T={T:.3f} · ECE={post_ece:.4f}",
        )

        manifest_payload: dict[str, Any] = {
            "fold_id": fold_id,
            "temperature": float(T),
            "n_val_rows": int(val_logits.shape[0]),
            "pre_ece": float(pre_ece),
            "post_ece": float(post_ece),
            "pre_curve": pre_curve.to_dict(),
            "post_curve": post_curve.to_dict(),
        }
        (fold_dir / "calibration_manifest.json").write_text(
            json.dumps(manifest_payload, indent=2)
        )

        per_fold_summary[fold_id] = {
            "temperature": float(T),
            "pre_ece": float(pre_ece),
            "post_ece": float(post_ece),
            "n_val_rows": int(val_logits.shape[0]),
        }

        improvement = pre_ece - post_ece
        direction = "softened" if T > 1.0 else "sharpened"
        print(
            f"[calibrate] {fold_id}: T={T:.3f} ({direction}); "
            f"ECE {pre_ece:.4f} -> {post_ece:.4f} (Δ = {improvement:+.4f}); "
            f"n_val={int(val_logits.shape[0])}",
            flush=True,
        )

    summary_path = output_root / "calibration_summary.json"
    summary_path.write_text(json.dumps(per_fold_summary, indent=2))
    print(f"[calibrate] summary -> {summary_path}", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
