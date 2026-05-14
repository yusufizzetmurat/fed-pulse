"""Unit tests for the energy-based OOD module.

The tests stub out the classifier (torch model + tokenizer) so they run
without a HuggingFace download. The math is the focus: energy formula,
aggregation, threshold percentile, manifest round-trip, score_text
behaviour with a known classifier.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from app.evaluation import ood


class _ToyModel(torch.nn.Module):
    """Deterministic model that returns canned logits for any input.

    The energy is -T * logsumexp(logits / T). For logits=[3, 0, -3] at
    T=1 the energy is -logsumexp([3, 0, -3]) which is well-defined and
    independent of input — perfect for asserting calibration math.
    """

    def __init__(self, logits: list[float]):
        super().__init__()
        self.logits = torch.tensor(logits, dtype=torch.float32)

    def forward(self, input_ids, attention_mask=None, **_):  # noqa: ARG002
        batch = input_ids.shape[0]
        out = self.logits.unsqueeze(0).expand(batch, -1).clone()

        class _Output:
            def __init__(self, logits):
                self.logits = logits

        return _Output(out)


class _ToyTokenizer:
    """Minimal tokenizer that returns a tensor pair; doesn't tokenize anything."""

    def __call__(self, text, *, truncation=True, max_length=512, return_tensors="pt", **_):  # noqa: ARG002
        # Mock encoding — content doesn't matter for the math tests.
        return {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }


def _toy_classifier(logits: list[float]):
    classifier = type("ToyClassifier", (), {})()
    classifier.model = _ToyModel(logits)
    classifier.tokenizer = _ToyTokenizer()
    return classifier


def test_logit_energy_matches_neg_logsumexp() -> None:
    """Sanity: energy(x) = -logsumexp(logits) at T=1."""

    classifier = _toy_classifier([2.0, 1.0, 0.0])
    energy = ood.logit_energy(classifier.model, classifier.tokenizer, "anything")
    expected = -math.log(math.exp(2.0) + math.exp(1.0) + math.exp(0.0))
    assert math.isclose(energy, expected, abs_tol=1e-5)


def test_logit_energy_matches_formula_under_temperature() -> None:
    """Sanity: energy(x, T) = -T * logsumexp(logits / T)."""

    classifier = _toy_classifier([5.0, 0.0, -5.0])
    T = 2.0
    energy = ood.logit_energy(classifier.model, classifier.tokenizer, "x", temperature=T)
    expected = -T * math.log(math.exp(5.0 / T) + math.exp(0.0 / T) + math.exp(-5.0 / T))
    assert math.isclose(energy, expected, abs_tol=1e-5)


def test_logit_energy_returns_inf_on_empty_text() -> None:
    classifier = _toy_classifier([1.0, 0.0])
    energy = ood.logit_energy(classifier.model, classifier.tokenizer, "")
    assert energy == float("inf")


def test_aggregate_energy_mean_takes_arithmetic_mean() -> None:
    assert ood.aggregate_energy([1.0, 2.0, 3.0], mode="mean") == pytest.approx(2.0)


def test_aggregate_energy_max_returns_largest_chunk() -> None:
    assert ood.aggregate_energy([1.0, 9.0, 3.0], mode="max") == 9.0


def test_aggregate_energy_median_handles_odd_and_even() -> None:
    assert ood.aggregate_energy([1.0, 9.0, 3.0], mode="median") == 3.0
    assert ood.aggregate_energy([1.0, 9.0, 3.0, 5.0], mode="median") == 4.0


def test_aggregate_energy_handles_empty_input_with_inf() -> None:
    assert ood.aggregate_energy([], mode="mean") == float("inf")


def test_aggregate_energy_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="unknown aggregation mode"):
        ood.aggregate_energy([1.0], mode="garbage")  # type: ignore[arg-type]


def test_aggregate_energy_drops_non_finite_values() -> None:
    assert ood.aggregate_energy([1.0, float("inf"), 3.0], mode="mean") == pytest.approx(2.0)


def test_calibrate_threshold_returns_percentile_of_training_energies() -> None:
    """A toy classifier with constant logits produces constant energies, so
    the 95th-percentile threshold equals any of the training energies."""

    classifier = _toy_classifier([2.0, 1.0, 0.0])
    threshold, energies = ood.calibrate_threshold(
        ["a", "b", "c", "d", "e"],
        classifier=classifier,
        percentile=95.0,
        temperature=1.0,
    )
    expected = -math.log(math.exp(2.0) + math.exp(1.0) + math.exp(0.0))
    assert math.isclose(threshold, expected, abs_tol=1e-5)
    assert len(energies) == 5


def test_calibrate_threshold_higher_percentile_yields_looser_threshold() -> None:
    """A spread of energies + a higher percentile cutoff -> larger threshold."""

    energies_sorted = []

    class _VariableLogits(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.counter = 0

        def forward(self, input_ids, attention_mask=None, **_):  # noqa: ARG002
            # Each call returns higher logits than the last; energies decrease,
            # then increase to spread the distribution.
            self.counter += 1

            class _Output:
                def __init__(self, logits):
                    self.logits = logits

            scale = float(self.counter)
            logits = torch.tensor([[scale, 0.0, -scale]], dtype=torch.float32)
            return _Output(logits)

    classifier = type("Variable", (), {})()
    classifier.model = _VariableLogits()
    classifier.tokenizer = _ToyTokenizer()
    threshold_p50, _ = ood.calibrate_threshold(
        list("abcdefghij"),
        classifier=classifier,
        percentile=50.0,
    )
    classifier.model.counter = 0  # type: ignore[attr-defined]
    threshold_p99, _ = ood.calibrate_threshold(
        list("abcdefghij"),
        classifier=classifier,
        percentile=99.0,
    )
    # The 99th percentile must be no smaller than the 50th. Equality would be a
    # bug because the variable model produces a spread.
    assert threshold_p99 >= threshold_p50
    assert threshold_p99 != threshold_p50  # sanity: there's actually spread


def test_calibrate_threshold_rejects_empty_corpus() -> None:
    classifier = _toy_classifier([1.0, 0.0])
    with pytest.raises(ValueError, match="zero valid energies"):
        ood.calibrate_threshold([], classifier=classifier)


def test_manifest_round_trip_through_disk(tmp_path: Path) -> None:
    manifest = ood.OODManifest(
        model_id="local/test-finbert",
        threshold=-3.14,
        percentile=95.0,
        temperature=1.0,
        aggregation="mean",
        training_corpus_size=1234,
        training_energy_mean=-4.0,
        training_energy_std=0.5,
        training_energy_min=-5.0,
        training_energy_max=-2.5,
        calibrated_at_utc="2026-05-14T12:00:00+00:00",
    )
    path = tmp_path / "ood.json"
    path.write_text(manifest.to_json(), encoding="utf-8")

    loaded = ood.load_manifest(path)
    assert loaded == manifest


def test_load_manifest_returns_none_for_missing_file(tmp_path: Path) -> None:
    assert ood.load_manifest(tmp_path / "absent.json") is None


def test_load_manifest_returns_none_for_malformed_json(tmp_path: Path) -> None:
    path = tmp_path / "ood.json"
    path.write_text("{not json", encoding="utf-8")
    assert ood.load_manifest(path) is None


def test_load_manifest_returns_none_when_required_field_missing(tmp_path: Path) -> None:
    path = tmp_path / "ood.json"
    path.write_text(json.dumps({"model_id": "x"}), encoding="utf-8")
    assert ood.load_manifest(path) is None


def test_score_text_returns_in_distribution_when_energy_below_threshold() -> None:
    classifier = _toy_classifier([2.0, 1.0, 0.0])
    manifest = ood.OODManifest(
        model_id="local/test-finbert",
        threshold=0.0,  # generous; any text energy will be negative -> in-dist
        percentile=95.0,
        temperature=1.0,
        aggregation="mean",
        training_corpus_size=10,
        training_energy_mean=-2.0,
        training_energy_std=0.5,
        training_energy_min=-3.0,
        training_energy_max=-1.0,
        calibrated_at_utc="2026-05-14T12:00:00+00:00",
    )
    result = ood.score_text("FOMC text excerpt", classifier=classifier, manifest=manifest)
    assert result["is_in_distribution"] is True
    assert result["ood_threshold"] == 0.0
    assert result["ood_energy"] < 0.0


def test_score_text_returns_out_of_distribution_when_energy_above_threshold() -> None:
    classifier = _toy_classifier([2.0, 1.0, 0.0])
    manifest = ood.OODManifest(
        model_id="local/test-finbert",
        threshold=-100.0,  # tight; any real energy is much higher -> OOD
        percentile=95.0,
        temperature=1.0,
        aggregation="mean",
        training_corpus_size=10,
        training_energy_mean=-2.0,
        training_energy_std=0.5,
        training_energy_min=-3.0,
        training_energy_max=-1.0,
        calibrated_at_utc="2026-05-14T12:00:00+00:00",
    )
    result = ood.score_text("good afternoon crypto bros", classifier=classifier, manifest=manifest)
    assert result["is_in_distribution"] is False
    assert result["ood_energy"] > -100.0


def test_score_text_respects_aggregation_mode_from_manifest() -> None:
    classifier = _toy_classifier([1.0, 0.0])
    manifest = ood.OODManifest(
        model_id="x",
        threshold=0.0,
        percentile=95.0,
        temperature=1.0,
        aggregation="max",
        training_corpus_size=5,
        training_energy_mean=-1.0,
        training_energy_std=0.1,
        training_energy_min=-1.0,
        training_energy_max=-1.0,
        calibrated_at_utc="2026-05-14T12:00:00+00:00",
    )
    result = ood.score_text(
        "doc",
        classifier=classifier,
        manifest=manifest,
        chunks=["chunk a", "chunk b", "chunk c"],
    )
    # With 3 identical chunks and 'max' aggregation, the doc energy equals
    # the per-chunk energy. The manifest's aggregation determines how
    # the chunk energies reduce.
    assert result["ood_energy"] == pytest.approx(result["chunk_energies"][0], abs=1e-5)
