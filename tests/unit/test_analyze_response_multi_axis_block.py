"""Multi-axis block on the /analyze response (#78).

The stance card is sourced from the existing sentiment classifier
output; factor / certainty / topic are left as None until the
multi-axis text classifier ships in a follow-up. The frontend renders
None cards as absent (no placeholder), so the contract here is
"stance always present; the others default to None".
"""

from __future__ import annotations


def test_multi_axis_block_carries_stance_from_sentiment_output() -> None:
    from app.main import _build_multi_axis_block

    sentiment = {
        "label": "hawkish",
        "score": 0.82,
        "raw": [
            {"label": "hawkish", "score": 0.82},
            {"label": "neutral", "score": 0.12},
            {"label": "dovish", "score": 0.06},
        ],
    }
    block = _build_multi_axis_block(sentiment)
    assert block is not None
    assert block["stance"]["label"] == "hawkish"
    assert 0.0 <= block["stance"]["confidence"] <= 1.0
    distribution = block["stance"]["distribution"]
    assert set(distribution.keys()) == {"hawkish", "dovish", "neutral"}
    assert abs(distribution["hawkish"] - 0.82) < 1e-6


def test_multi_axis_block_other_axes_default_to_none() -> None:
    """The factor / certainty / topic cards are populated in a
    follow-up PR by the multi-task text classifier. Until then they
    are explicitly None so the frontend renders absence honestly."""

    from app.main import _build_multi_axis_block

    sentiment = {
        "label": "neutral",
        "score": 0.5,
        "raw": [{"label": "neutral", "score": 0.5}],
    }
    block = _build_multi_axis_block(sentiment)
    assert block is not None
    assert block["factor"] is None
    assert block["certainty"] is None
    assert block["topic"] is None


def test_multi_axis_block_coerces_unknown_label_to_neutral() -> None:
    """If the upstream classifier returns a label outside the
    canonical set (case mismatch, stale checkpoint, fallback model),
    the block collapses to neutral with the available distribution so
    the frontend stays renderable."""

    from app.main import _build_multi_axis_block

    sentiment = {
        "label": "UNKNOWN",
        "score": 0.0,
        "raw": [],
    }
    block = _build_multi_axis_block(sentiment)
    assert block is not None
    assert block["stance"]["label"] == "neutral"
    assert block["stance"]["confidence"] == 0.0
