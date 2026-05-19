"""B1 (#212): catalogue of structured semantic features extracted by a
frontier LLM (Claude Sonnet 4.7) from each FOMC document.

Each feature is a small categorical (2-4 levels) chosen so it maps to
a specific finding in the published monetary-policy text-analysis
literature. The catalogue is the audit-trail artefact for the extraction
run — every feature here is citable and every prompt template is
versioned via ``CATALOG_VERSION``.

The model receives the full document text with the explicit event_date
masked (``"the FOMC release dated [REDACTED]"``) so it cannot recall
the specific meeting from its pretraining. This is the soft-mitigation
for the LLM-pretraining-contamination concern documented in
``09_Risk_Register.md`` R-18.
"""

from __future__ import annotations

import dataclasses
from typing import Final


# ---------------------------------------------------------------------------
# Catalogue versioning
# ---------------------------------------------------------------------------
# Bump CATALOG_VERSION when any feature (name, levels, prompt template,
# system prompt, or model_id) changes. The per-document cache is keyed
# on (text_hash, model_id, catalog_version) so a bump invalidates the
# old cache and triggers a fresh extraction run.

CATALOG_VERSION: Final[str] = "2026-05-19.v1"
MODEL_ID: Final[str] = "claude-sonnet-4-7"

# Temperature 0.0 makes the extraction deterministic for the same input
# (modulo the model's residual non-determinism on a tied logit). Combined
# with the strict JSON schema, this gives reproducible per-document
# features at acceptable cost.
TEMPERATURE: Final[float] = 0.0


@dataclasses.dataclass(frozen=True)
class CatalogFeature:
    """One feature in the catalogue.

    ``name`` is the column on the persisted parquet + the loader's
    feature-vector slot. ``levels`` defines the discrete categorical
    values the LLM is allowed to return; the extractor's JSON schema
    enforces this and an out-of-vocabulary value triggers a retry.

    ``prompt_question`` is the natural-language question the model
    answers in its JSON response. ``citation`` records the academic
    paper that motivates the feature so a reviewer can trace each
    catalogue entry back to its literature anchor.

    ``unanswerable_level`` is the level the model returns when the
    feature cannot be assessed from the document (e.g. dot-plot
    trajectory for a pre-2012 release that predates the dot plot).
    Mapped to the all-zeros one-hot at training time so the missing
    semantics are explicit, not silently coerced.
    """

    name: str
    levels: tuple[str, ...]
    prompt_question: str
    citation: str
    unanswerable_level: str | None = None


# ---------------------------------------------------------------------------
# The 10 catalogue features
# ---------------------------------------------------------------------------
# Order is stable. Adding a new feature appends; reordering or removing
# requires a CATALOG_VERSION bump.

CATALOG: Final[tuple[CatalogFeature, ...]] = (
    CatalogFeature(
        name="hawkish_shift_vs_prior",
        levels=("hawkish_shift", "dovish_shift", "unchanged", "not_assessable"),
        prompt_question=(
            "Compared to the prior FOMC release referenced in this document, "
            "does the language signal a hawkish shift, a dovish shift, or no "
            "change in policy stance? Answer 'not_assessable' if the prior "
            "meeting is not referenced or compared."
        ),
        citation="Hansen & McMahon (2016), Shocking Language: Understanding the Macroeconomic Effects of Central Bank Communication",
        unanswerable_level="not_assessable",
    ),
    CatalogFeature(
        name="forward_guidance_change",
        levels=("strengthened", "weakened", "unchanged", "not_present"),
        prompt_question=(
            "Does the document strengthen, weaken, or leave unchanged the "
            "forward-guidance language about future policy? Answer 'not_present' "
            "if no explicit forward guidance is given."
        ),
        citation="Campbell, Evans, Fisher & Justiniano (2012), Macroeconomic Effects of Federal Reserve Forward Guidance",
        unanswerable_level="not_present",
    ),
    CatalogFeature(
        name="policy_path_direction",
        levels=("tighter", "looser", "same", "ambiguous"),
        prompt_question=(
            "What direction does the document signal for the near-term fed "
            "funds policy path? 'tighter' = signalling higher rates ahead, "
            "'looser' = signalling lower rates ahead, 'same' = on-hold, "
            "'ambiguous' = mixed or unclear signal."
        ),
        citation="Gürkaynak, Sack & Swanson (2005), Do Actions Speak Louder Than Words? The Response of Asset Prices to Monetary Policy Actions and Statements",
        unanswerable_level="ambiguous",
    ),
    CatalogFeature(
        name="fed_vs_market_gap",
        levels=("fed_more_hawkish", "fed_more_dovish", "aligned", "not_assessable"),
        prompt_question=(
            "Does the document position the FOMC as more hawkish, more dovish, "
            "or aligned with market expectations on the rate path? Answer "
            "based on the document text only; do not consult external "
            "market data. 'not_assessable' if the text gives no read on "
            "market alignment."
        ),
        citation="Jarociński & Karadi (2020), Deconstructing Monetary Policy Surprises — the Role of Information Shocks",
        unanswerable_level="not_assessable",
    ),
    CatalogFeature(
        name="inflation_concern_intensity",
        levels=("elevated", "unchanged", "softening", "not_discussed"),
        prompt_question=(
            "How does the document characterise inflation concern relative "
            "to a recent baseline? 'elevated' = inflation worry strengthened, "
            "'softening' = inflation worry abated, 'unchanged' = baseline, "
            "'not_discussed' if inflation is not addressed."
        ),
        citation="Hansen & McMahon (2016) — Inflation topic decomposition",
        unanswerable_level="not_discussed",
    ),
    CatalogFeature(
        name="labor_market_language",
        levels=("tight", "softening", "balanced", "not_discussed"),
        prompt_question=(
            "How does the document characterise current US labor-market "
            "conditions? 'tight' = strong / overheated, 'softening' = "
            "weakening, 'balanced' = neither strong nor weak, 'not_discussed' "
            "if labor markets are not addressed."
        ),
        citation="Hubert (2017), The Role of Forecast Disclosure in Monetary Policy Communication",
        unanswerable_level="not_discussed",
    ),
    CatalogFeature(
        name="financial_stability_concern",
        levels=("present", "absent"),
        prompt_question=(
            "Does the document explicitly reference financial-stability "
            "concerns, banking-sector risk, credit conditions, or systemic "
            "vulnerability? Answer 'present' if any of these are raised."
        ),
        citation="Cieslak & Schrimpf (2019), Non-monetary news in central bank communication",
    ),
    CatalogFeature(
        name="recession_risk_acknowledged",
        levels=("present", "absent"),
        prompt_question=(
            "Does the document explicitly acknowledge recession risk, "
            "downside scenarios for growth, or the possibility of a "
            "contraction? Hedged language about 'risks to the outlook' "
            "counts as 'present' if downside is named; 'absent' if no "
            "explicit downside-risk language."
        ),
        citation="Bernanke & Kuttner (2005), What Explains the Stock Market's Reaction to Federal Reserve Policy",
    ),
    CatalogFeature(
        name="hedge_language_density",
        levels=("low", "medium", "high"),
        prompt_question=(
            "How frequent and contextually consequential is hedging "
            "language in this document? Hedging includes phrases like "
            "'however', 'although', 'on the other hand', 'risks remain', "
            "and modal verbs ('may', 'could', 'might') used as qualifiers. "
            "'high' = pervasive throughout; 'medium' = present in several "
            "sections; 'low' = sparse or absent."
        ),
        citation="Loughran & McDonald (2011), When Is a Liability Not a Liability? Textual Analysis, Dictionaries, and 10-Ks",
    ),
    CatalogFeature(
        name="policy_uncertainty_language",
        levels=("elevated", "baseline", "reduced"),
        prompt_question=(
            "Does the document signal elevated, baseline, or reduced "
            "policy-uncertainty language about the future course of "
            "monetary policy? Look for phrases like 'data-dependent', "
            "'committee will closely monitor', 'depending on incoming "
            "data', 'will respond as appropriate' as elevated-uncertainty "
            "signals."
        ),
        citation="Baker, Bloom & Davis (2016), Measuring Economic Policy Uncertainty",
    ),
)


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT: Final[str] = (
    "You are an expert monetary policy text analyst. Given a Federal Reserve "
    "communication document (FOMC statement, minutes, or press conference "
    "transcript), extract a fixed set of structured features documented in "
    "a catalogue. Each feature has a small fixed set of allowed levels; "
    "return exactly one allowed level per feature. Base every answer "
    "strictly on the document text — do not consult outside knowledge of "
    "the specific meeting, the market reaction, or subsequent events. "
    "Where a feature cannot be assessed from the document, use the "
    "documented 'not_assessable' / 'not_present' / 'not_discussed' level. "
    "Output JSON only. No prose, no preamble, no explanation."
)


def build_user_prompt(document_text: str) -> str:
    """Render the per-document extraction prompt.

    The document is included verbatim; the event date is *not* injected
    so the model cannot recall the specific meeting from pretraining.
    """

    feature_section_lines: list[str] = []
    for i, feature in enumerate(CATALOG, start=1):
        feature_section_lines.append(
            f"{i}. {feature.name} (allowed levels: {', '.join(feature.levels)})\n"
            f"   Question: {feature.prompt_question}"
        )
    feature_section = "\n\n".join(feature_section_lines)

    schema_lines = ", ".join(
        f'"{f.name}": "<one of: {" | ".join(f.levels)}>"' for f in CATALOG
    )
    schema = "{" + schema_lines + "}"

    return (
        "Below is a Federal Reserve communication document. The original "
        "publication date has been intentionally redacted so that you "
        "extract features based on the text content alone, not from "
        "your memory of the specific meeting.\n\n"
        f"Document (event date: [REDACTED]):\n\n```\n{document_text}\n```\n\n"
        "Extract the following catalogue features. For each, return the "
        "single most appropriate level from the allowed set. If a feature "
        "cannot be assessed from this document, use the documented "
        "fallback level.\n\n"
        f"{feature_section}\n\n"
        "Return a single JSON object matching this schema:\n\n"
        f"{schema}\n\n"
        "JSON only. No prose."
    )


def feature_names() -> tuple[str, ...]:
    """The catalogue's stable column order; loader and persistence both
    consume this ordering."""

    return tuple(f.name for f in CATALOG)


def levels_for(feature_name: str) -> tuple[str, ...]:
    for f in CATALOG:
        if f.name == feature_name:
            return f.levels
    raise KeyError(f"unknown catalogue feature: {feature_name!r}")


__all__ = [
    "CATALOG_VERSION",
    "MODEL_ID",
    "TEMPERATURE",
    "SYSTEM_PROMPT",
    "CATALOG",
    "CatalogFeature",
    "build_user_prompt",
    "feature_names",
    "levels_for",
]
