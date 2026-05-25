"""Multi-encoder sweep dimension on the forecaster CLI.

Pins the ``--text-encoders`` (plural) entry point that lets a single
sweep iterate the Bundle A.2 arm A and arm B encoders side-by-side
inside the shared 8-worker GPU pool. The single ``--text-encoder``
flag is left in place and its candidate output stays byte-identical
when the plural flag is absent.
"""

from __future__ import annotations

import argparse

import pytest

pytest.importorskip("torch")

from app.train_forecaster import (
    _parse_args,  # type: ignore[attr-defined]
    TEXT_ENCODER_CHOICES,
    build_sweep_candidates,
)


def _multi_encoder_args(
    *,
    text_encoders: list[str] | None,
    text_encoder: str = "none",
    folds: list[str] | None = None,
) -> argparse.Namespace:
    """Namespace shaped like the multi-encoder smoke sweep.

    Single HP cell (the smoke-sweep contract) times the encoder axis
    times the fold axis. ``architectures=None`` keeps the single
    architecture so the cell count is exactly
    ``len(text_encoders) * len(folds) * 1``.
    """

    return argparse.Namespace(
        hidden_size=32,
        num_layers=1,
        dropout=0.1,
        learning_rate=1e-3,
        epochs=4,
        head_hidden_size=16,
        hidden_sizes=None,
        num_layers_grid=None,
        dropouts=None,
        learning_rates=None,
        epochs_grid=None,
        weight_decay=1e-4,
        weight_decays=None,
        text_adapter_dim=64,
        text_adapter_dims=None,
        text_encoder=text_encoder,
        text_encoders=text_encoders,
        use_text_embeddings=True,
        training_package_id=None,
        rich_features=False,
        architecture="lstm",
        architectures=None,
        seed=11,
        seeds=None,
        credibility_features=False,
        random_search=False,
        random_search_samples=50,
        random_search_seed=42,
        folds=folds,
    )


class TestMultiEncoderSweepDimension:
    def test_two_encoders_cross_four_folds_yields_eight_cells(self) -> None:
        """1 HP cell x 2 encoders x 4 folds x 1 seed = 8 cells."""

        args = _multi_encoder_args(
            text_encoders=[
                "finbert_fed_adjacent_xbank_aux_stance_masked",
                "finbert_fed_adjacent_xbank_aux_weighted",
            ],
            folds=["wf_fold_1", "wf_fold_2", "wf_fold_3", "wf_fold_4"],
        )

        candidates = build_sweep_candidates(args)

        assert len(candidates) == 8
        # Every cell records its assigned encoder.
        assert {c["text_encoder"] for c in candidates} == {
            "finbert_fed_adjacent_xbank_aux_stance_masked",
            "finbert_fed_adjacent_xbank_aux_weighted",
        }
        # And the fold axis is also expanded per encoder.
        fold_ids = {c.get("fold_id") for c in candidates}
        assert fold_ids == {"wf_fold_1", "wf_fold_2", "wf_fold_3", "wf_fold_4"}

    def test_encoder_axis_balanced_across_folds(self) -> None:
        """Each (encoder, fold) pair is generated exactly once."""

        args = _multi_encoder_args(
            text_encoders=[
                "finbert_fed_adjacent_xbank_aux_stance_masked",
                "finbert_fed_adjacent_xbank_aux_weighted",
            ],
            folds=["wf_fold_1", "wf_fold_2"],
        )

        candidates = build_sweep_candidates(args)

        pairs = {(c["text_encoder"], c["fold_id"]) for c in candidates}
        assert len(pairs) == 4
        assert pairs == {
            ("finbert_fed_adjacent_xbank_aux_stance_masked", "wf_fold_1"),
            ("finbert_fed_adjacent_xbank_aux_stance_masked", "wf_fold_2"),
            ("finbert_fed_adjacent_xbank_aux_weighted", "wf_fold_1"),
            ("finbert_fed_adjacent_xbank_aux_weighted", "wf_fold_2"),
        }


class TestSingleEncoderUnchanged:
    def test_single_encoder_path_omits_text_encoder_key(self) -> None:
        """Legacy path leaves candidates without a ``text_encoder`` key.

        The pre-PR aggregator relied on the absence of the key to
        infer the sweep-wide encoder from ``args.text_encoder``;
        keeping it absent preserves that read.
        """

        args = _multi_encoder_args(
            text_encoders=None,
            text_encoder="finbert_fed_adjacent",
            folds=["wf_fold_1"],
        )

        candidates = build_sweep_candidates(args)

        assert len(candidates) == 1
        for c in candidates:
            assert "text_encoder" not in c

    def test_none_encoder_single_path_omits_text_encoder_key(self) -> None:
        """Default ``--text-encoder=none`` keeps the byte-identical candidate shape."""

        args = _multi_encoder_args(
            text_encoders=None,
            text_encoder="none",
            folds=None,
        )

        candidates = build_sweep_candidates(args)

        assert len(candidates) == 1
        for c in candidates:
            assert "text_encoder" not in c


class TestCliParse:
    def test_text_encoders_accepted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``--text-encoders alias_a alias_b`` parses cleanly."""

        argv = [
            "train_forecaster.py",
            "--text-encoders",
            "finbert_fed_adjacent_xbank_aux_stance_masked",
            "finbert_fed_adjacent_xbank_aux_weighted",
        ]
        monkeypatch.setattr("sys.argv", argv)

        parsed = _parse_args()

        assert parsed.text_encoders == [
            "finbert_fed_adjacent_xbank_aux_stance_masked",
            "finbert_fed_adjacent_xbank_aux_weighted",
        ]
        # Plural-only path leaves the singular at the default.
        assert parsed.text_encoder == "none"

    def test_text_encoder_singular_still_parses(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The legacy singular flag stays on the parse path."""

        argv = [
            "train_forecaster.py",
            "--text-encoder",
            "finbert_fed_adjacent",
        ]
        monkeypatch.setattr("sys.argv", argv)

        parsed = _parse_args()

        assert parsed.text_encoder == "finbert_fed_adjacent"
        assert parsed.text_encoders is None

    def test_mutually_exclusive_with_singular(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Passing both flags exits with status 2 at parse time."""

        argv = [
            "train_forecaster.py",
            "--text-encoder",
            "finbert_fed_adjacent",
            "--text-encoders",
            "finbert_fed_adjacent_xbank_aux_stance_masked",
        ]
        monkeypatch.setattr("sys.argv", argv)

        with pytest.raises(SystemExit) as exc:
            _parse_args()
        # argparse mutually-exclusive errors exit with code 2.
        assert exc.value.code == 2

    def test_unknown_alias_rejected_on_text_encoders(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An alias outside the choices tuple errors at parse time."""

        argv = [
            "train_forecaster.py",
            "--text-encoders",
            "not_a_real_encoder",
        ]
        monkeypatch.setattr("sys.argv", argv)

        with pytest.raises(SystemExit) as exc:
            _parse_args()
        assert exc.value.code == 2

    def test_unknown_alias_rejected_on_text_encoder(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The choices tuple guards the singular flag too."""

        argv = [
            "train_forecaster.py",
            "--text-encoder",
            "not_a_real_encoder",
        ]
        monkeypatch.setattr("sys.argv", argv)

        with pytest.raises(SystemExit) as exc:
            _parse_args()
        assert exc.value.code == 2


def test_xbank_aux_aliases_are_registered_choices() -> None:
    """The Bundle A.2 aliases live in the CLI choices tuple."""

    assert "finbert_fed_adjacent_xbank_aux_stance_masked" in TEXT_ENCODER_CHOICES
    assert "finbert_fed_adjacent_xbank_aux_weighted" in TEXT_ENCODER_CHOICES
    # And the queued DAPT alias pre-empts a future Bundle A.4 blocker.
    assert "finbert_fed_adjacent_xbank_dapt" in TEXT_ENCODER_CHOICES
