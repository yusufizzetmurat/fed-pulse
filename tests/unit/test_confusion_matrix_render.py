from __future__ import annotations

import pytest

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from app.evaluation.confusion_matrix_render import (  # noqa: E402
    aggregate_confusion_matrices,
    render_confusion_matrix_png,
)


# ---------------------------------------------------------------------------
# render_confusion_matrix_png — output file shape + cell counts
# ---------------------------------------------------------------------------


def test_renders_png_at_expected_dimensions(tmp_path) -> None:
    cm = [[10, 0, 0], [0, 8, 2], [0, 1, 9]]
    out = tmp_path / "cm.png"
    rendered = render_confusion_matrix_png(cm, out)
    assert rendered.exists()
    with Image.open(rendered) as img:
        assert img.format == "PNG"
        # 3 classes at default cell=96 + margins -> exact dims are
        # deterministic on a fixed config, so just sanity-check the
        # image is the right ballpark.
        assert img.size[0] > 96 * 3
        assert img.size[1] > 96 * 3


def test_renders_two_class_matrix(tmp_path) -> None:
    cm = [[5, 1], [2, 7]]
    out = tmp_path / "binary.png"
    rendered = render_confusion_matrix_png(cm, out, class_labels=("neg", "pos"))
    assert rendered.exists()


def test_class_labels_default_to_indices(tmp_path) -> None:
    cm = [[1, 0], [0, 1]]
    out = tmp_path / "default_labels.png"
    # Should not raise — labels fall back to "0", "1"
    rendered = render_confusion_matrix_png(cm, out)
    assert rendered.exists()


def test_title_is_optional(tmp_path) -> None:
    cm = [[1, 0], [0, 1]]
    out_with = tmp_path / "with.png"
    out_without = tmp_path / "without.png"
    render_confusion_matrix_png(cm, out_with, title="Tier 1")
    render_confusion_matrix_png(cm, out_without, title=None)
    with Image.open(out_with) as a, Image.open(out_without) as b:
        # Title adds vertical space; without-title image is shorter.
        assert b.size[1] < a.size[1]


def test_zero_matrix_renders_all_white_cells(tmp_path) -> None:
    """Edge case: every cell is zero -> vmax=0 -> _ramp_color returns
    pure white. The renderer must not crash on this degenerate input."""

    cm = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    out = tmp_path / "zero.png"
    rendered = render_confusion_matrix_png(cm, out)
    assert rendered.exists()


def test_empty_matrix_raises(tmp_path) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        render_confusion_matrix_png([], tmp_path / "x.png")


def test_non_square_matrix_raises(tmp_path) -> None:
    cm = [[1, 0, 0], [0, 1]]  # ragged
    with pytest.raises(ValueError, match="must be square"):
        render_confusion_matrix_png(cm, tmp_path / "x.png")


# ---------------------------------------------------------------------------
# aggregate_confusion_matrices — elementwise sum
# ---------------------------------------------------------------------------


def test_aggregate_sums_elementwise() -> None:
    m1 = [[1, 0], [0, 1]]
    m2 = [[2, 1], [1, 2]]
    out = aggregate_confusion_matrices([m1, m2])
    assert out == [[3, 1], [1, 3]]


def test_aggregate_preserves_dim() -> None:
    matrices = [[[i + j for j in range(3)] for i in range(3)] for _ in range(5)]
    out = aggregate_confusion_matrices(matrices)
    assert len(out) == 3
    assert all(len(row) == 3 for row in out)
    # Each cell is the sum of 5 identical matrices -> 5 * original
    assert out[0][1] == 5 * 1


def test_aggregate_rejects_mismatched_shapes() -> None:
    m1 = [[1, 0], [0, 1]]
    m2 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    with pytest.raises(ValueError, match="same shape"):
        aggregate_confusion_matrices([m1, m2])


def test_aggregate_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        aggregate_confusion_matrices([])
