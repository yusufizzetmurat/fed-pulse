"""Unit coverage for ``scripts.run_hawk_dove_jackknife`` (#506).

The runner itself is GPU-bound and not invoked in CI; these tests cover
the helper math that drives the JSON artefact:

- ``patched_lexicons`` correctly drops a token from a small fixture
  lexicon and leaves the other collection untouched.
- ``_serialise_payload`` emits the documented schema.
- ``fragile_tokens_from_results`` returns exactly the tokens whose
  ``|delta|`` exceeds the threshold, against a synthetic delta vector.
"""

from __future__ import annotations

from scripts.run_hawk_dove_jackknife import (
    FRAGILE_DELTA_THRESHOLD,
    TokenJackknifeResult,
    _build_token_inventory,
    _serialise_payload,
    fragile_tokens_from_results,
    patched_lexicons,
)


# ---------------------------------------------------------------------------
# patched_lexicons
# ---------------------------------------------------------------------------


def test_patched_lexicons_drops_token_from_hawk_only():
    """A hawk token is removed from HAWK only; DOVE comes back intact."""

    hawk = frozenset({"tightening", "restrictive", "hike"})
    dove = frozenset({"easing", "cut"})
    patched_hawk, patched_dove = patched_lexicons(
        "tightening", base_hawk=hawk, base_dove=dove
    )
    assert patched_hawk == frozenset({"restrictive", "hike"})
    assert patched_dove == dove


def test_patched_lexicons_drops_token_from_dove_only():
    """A dove token is removed from DOVE only; HAWK comes back intact."""

    hawk = frozenset({"hike"})
    dove = frozenset({"easing", "cut", "accommodative"})
    patched_hawk, patched_dove = patched_lexicons(
        "accommodative", base_hawk=hawk, base_dove=dove
    )
    assert patched_dove == frozenset({"easing", "cut"})
    assert patched_hawk == hawk


def test_patched_lexicons_unknown_token_returns_inputs_unchanged():
    """A token that lives in neither lexicon round-trips losslessly."""

    hawk = frozenset({"hike"})
    dove = frozenset({"cut"})
    patched_hawk, patched_dove = patched_lexicons(
        "neutral", base_hawk=hawk, base_dove=dove
    )
    assert patched_hawk == hawk
    assert patched_dove == dove


def test_patched_lexicons_token_in_both_lexicons_drops_from_both():
    """A token accidentally listed in both lists is dropped symmetrically."""

    hawk = frozenset({"firm", "hike"})
    dove = frozenset({"firm", "cut"})
    patched_hawk, patched_dove = patched_lexicons(
        "firm", base_hawk=hawk, base_dove=dove
    )
    assert "firm" not in patched_hawk
    assert "firm" not in patched_dove


# ---------------------------------------------------------------------------
# _build_token_inventory
# ---------------------------------------------------------------------------


def test_build_token_inventory_sorts_and_tags_kind():
    """Inventory walks hawk tokens (sorted) then dove tokens (sorted)."""

    hawk = frozenset({"hike", "raise"})
    dove = frozenset({"ease", "cut"})
    inventory = _build_token_inventory(hawk, dove)
    assert inventory == [
        ("hike", "hawk"),
        ("raise", "hawk"),
        ("cut", "dove"),
        ("ease", "dove"),
    ]


# ---------------------------------------------------------------------------
# fragile_tokens_from_results
# ---------------------------------------------------------------------------


def test_fragile_tokens_threshold_math_is_strictly_above():
    """``|delta| > threshold`` -- equality is not fragile."""

    results = [
        TokenJackknifeResult(
            token="a", kind="hawk", without_f1=0.50, delta=0.0049
        ),
        TokenJackknifeResult(
            token="b", kind="hawk", without_f1=0.50, delta=-0.0051
        ),
        TokenJackknifeResult(
            token="c", kind="dove", without_f1=0.50, delta=0.005
        ),
        TokenJackknifeResult(
            token="d", kind="dove", without_f1=0.50, delta=0.020
        ),
    ]
    fragile = fragile_tokens_from_results(results, threshold=0.005)
    assert fragile == ["b", "d"]


def test_fragile_tokens_uses_absolute_delta():
    """Negative deltas of equal magnitude trip the threshold."""

    results = [
        TokenJackknifeResult(
            token="positive", kind="hawk", without_f1=0.50, delta=0.010
        ),
        TokenJackknifeResult(
            token="negative", kind="hawk", without_f1=0.50, delta=-0.010
        ),
    ]
    fragile = fragile_tokens_from_results(results, threshold=0.005)
    assert fragile == ["negative", "positive"]


def test_fragile_tokens_empty_when_all_inside_band():
    results = [
        TokenJackknifeResult(
            token="x", kind="hawk", without_f1=0.50, delta=0.001
        ),
    ]
    assert fragile_tokens_from_results(results, threshold=0.005) == []


def test_fragile_tokens_default_threshold_matches_constant():
    """The default threshold is the module-level FRAGILE_DELTA_THRESHOLD."""

    results = [
        TokenJackknifeResult(
            token="bandwidth_check",
            kind="hawk",
            without_f1=0.50,
            delta=FRAGILE_DELTA_THRESHOLD + 1e-6,
        ),
    ]
    assert fragile_tokens_from_results(results) == ["bandwidth_check"]


# ---------------------------------------------------------------------------
# _serialise_payload
# ---------------------------------------------------------------------------


def test_serialise_payload_emits_documented_schema():
    """The JSON shape matches the runner docstring."""

    results = [
        TokenJackknifeResult(
            token="hike", kind="hawk", without_f1=0.451, delta=-0.003
        ),
        TokenJackknifeResult(
            token="cut", kind="dove", without_f1=0.470, delta=0.016
        ),
    ]
    payload = _serialise_payload(
        training_package_id="tp_test",
        baseline_f1=0.454,
        results=results,
        threshold=0.005,
    )
    assert payload["training_package_id"] == "tp_test"
    assert payload["baseline_f1"] == 0.454
    assert payload["fragile_delta_threshold"] == 0.005
    assert payload["fragile_tokens"] == ["cut"]
    assert isinstance(payload["tokens"], list)
    assert payload["tokens"][0] == {
        "token": "hike",
        "kind": "hawk",
        "without_f1": 0.451,
        "delta": -0.003,
    }


def test_serialise_payload_empty_results_emits_empty_fragile_list():
    """No per-token results -> no fragile tokens; baseline still surfaces."""

    payload = _serialise_payload(
        training_package_id="tp_empty",
        baseline_f1=0.40,
        results=[],
        threshold=0.005,
    )
    assert payload["tokens"] == []
    assert payload["fragile_tokens"] == []
    assert payload["baseline_f1"] == 0.40
