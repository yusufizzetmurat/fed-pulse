# ADR 0036 — Loughran-McDonald financial sentiment lexicon as the §6 encoder baseline

Issue #445 lands the Loughran-McDonald (L-M) Master Dictionary as an ablation arm against the canonical FinBERT-Fed-Adjacent encoder (ADR 0019) on the per-family rich-feature ablation runner. Before this PR the §6 table picks a financial-domain encoder as the text-channel substrate without measuring it against a classical lexicon-counts baseline; the standard reviewer punch on that gap is "why didn't you compare to L-M?" and the answer was "we will". This ADR is the answer.

L-M is the most-cited classical financial sentiment resource in the literature (Loughran & McDonald, 2011, *Journal of Finance*; updated annually by McDonald at the Notre Dame SRAF page). The dictionary tags each word with one or more sentiment categories — Negative, Positive, Uncertainty, Litigious, Strong_Modal, Weak_Modal, Constraining — by the year the word was first added; a non-zero cell means the word belongs to that category in the published lexicon. The standard usage in the accounting / finance literature is per-document category-share features: tokenise the text, count the per-category matches, divide by the document length. The result is a six-scalar vector per document (we collapse Strong_Modal + Weak_Modal into a single ``modal`` count so the vector stays at six dims). Cheap to compute, no model parameters, no encoder weights, no fine-tuning step. The baseline a classical financial-NLP paper from 2014 would ship.

The ablation arm substitutes that six-scalar vector for the canonical FinBERT-Fed-Adjacent pooled-encoder block on the same fold protocol. Same per-family ablation runner (``scripts/run_per_family_ablation.py``), same training package, same seeds, same fold ids, same head topology, same hyperparameters. The only thing that changes is what fills the ``FeatureVector.text_embedding_pooled`` slot before the per-fold scaler fit: encoder arm writes the softmax-pooled FinBERT vector across the four most recent prior statements; L-M arm writes the six L-M category percentages computed off the event's own raw text.

## What the ablation answers

The §6 table cell reads as one of three diagnostics:

*FinBERT-Fed-Adjacent beats L-M by a measurable margin.* The continued-pretraining DAPT pass (#190) plus the in-domain FOMC fine-tune (B2) push a representation that catches contextual signal the pre-2014 lexicon counts can't reach. The headline framing "we use a domain-adapted encoder because the classical baseline is not enough" is supported by data rather than asserted. This is the reading the literature on encoder vs lexicon comparisons on financial tasks reports as typical (FinBERT, Araci 2019; FiLM-NLP, Yang 2020) — encoders beat L-M on document-level sentiment by 5-15 pp macro-F1 depending on the corpus. The numbers we expect on FOMC vol-regime / stance classification sit in the same band.

*Comparable.* L-M counts already saturate the per-document signal the head can read off FOMC text on this corpus size (~895 supervised events). The encoder's extra parameters do not buy a measurable lift because the regime label is dominated by market features and the text channel's marginal contribution is bounded by what any reasonable text representation can pick up. This is a publishable reading too — the ensemble lift framing ("L-M is cheap, the encoder adds X pp, the ensemble lifts by Y pp") is the alternative narrative the report would adopt.

*L-M beats the encoder.* The encoder is over-parameterised for the small downstream pool; the lexicon's bias-variance tradeoff is better matched to the corpus size. This reading would force a re-examination of whether the encoder belongs in the headline at all — the methodology contribution would shift to "classical lexicon counts are the right text substrate for FOMC-scale corpora, not domain-adapted transformers". Lower probability but the experiment has to be willing to land there for the test to be honest.

## What ships

A pure-Python lexicon loader at ``backend/app/data/loughran_mcdonald.py``. The Master Dictionary is read from a SHA-pinned CSV cached at ``data/external/loughran_mcdonald/<sha>__master_dictionary.csv``; the loader returns per-category lowercase word sets keyed on ``LM_CATEGORIES = (positive, negative, uncertainty, litigious, constraining, modal)``. ``compute_lm_features(text, lexicon)`` tokenises (lowercase + alphabetic-only regex; punctuation and digits drop), counts per-category matches, and emits the six percentages keyed as ``lm_<category>_pct`` in ``[0.0, 100.0]``. The list-shaped helper ``compute_lm_feature_vector`` returns the same six numbers in ``LM_CATEGORIES`` order so the ablation runner can write them straight into the existing text-embedding slot.

The lexicon is public but the loader never reaches for the network at runtime. The SHA pin (``LM_LEXICON_SHA``) names the active vintage on disk; tests pass ``local_csv=`` to a fixture so the air-gapped path is the exercised one. A missing cache file raises ``FileNotFoundError`` with the expected path in the message rather than silently fetching — the published cached vintage is the contract.

The ablation arm wires into ``scripts/run_per_family_ablation.py`` behind ``--arm lm_lexicon``. On the L-M arm the loader runs with ``text_encoder=None`` (the pooled-encoder pool stays empty), ``encoder_lora=True`` (so the loader writes the target row's raw text into ``vectors[-1].raw_text`` — LoRA training is not activated; ``ModelConfig.encoder_lora`` stays False, the flag only governs whether the loader harvests text), and a post-load walker computes the six L-M percentages off the target text and broadcasts the vector into ``text_embedding_pooled`` on every bar of every sequence. The ``ModelConfig`` widens its text-embedding adapter to the six-dim L-M width so the recurrent core sees the substituted block at the same position the canonical encoder block occupies on the default arm. Everything else — the rich-feature 35-dim slice, the per-family ablation cells, the cumulative chain, the rates target, the regime conditioning — is unchanged.

## Why the lexicon is pinned, not auto-downloaded

The L-M Master Dictionary is updated roughly annually as McDonald revises the per-category word membership; a 2026 vintage can shift the per-category counts measurably against a 2023 vintage on the same FOMC document. Auto-downloading the lexicon at runtime would make the head-to-head numbers a moving target — a re-run six months later against a different vintage would conflate methodology change with feature drift. The SHA pin (``lm_master_2024_q4`` at PR time) freezes the comparison; bumping the pin is a deliberate methodology decision that ships with a rerun of the head-to-head.

The CI smoke runs off a small fixture CSV (10 words covering every category, including a cross-listed entry for the negative + uncertainty union) so the loader's parse contract, the lowercase normalisation, and the modal-column union are pinned without the production lexicon ever touching the test runner. Tests live at ``tests/unit/test_loughran_mcdonald_loader.py`` (loader contract, fixture round-trip, missing-cache failure, category lookup) and ``tests/unit/test_lm_features.py`` (compute_lm_features against known short documents — empty / all-positive / all-uncertainty / mixed / modal-union / punctuation-stripped / case-insensitive).

## Caveats

The L-M lexicon is alphabetic and English-only; tokens with embedded digits or non-ASCII characters drop out of the count. This is the standard literature convention and matches what the published lexicon was built against, but it does mean a sentence like "rates rose 25 bps" has the same L-M signature as "rates rose" — the magnitude of the move is invisible to the lexicon. The encoder arm catches the magnitude via the contextual embedding; the L-M arm cannot. That asymmetry sits inside the comparison and the report frames the result with the caveat called out.

The modal collapse (Strong_Modal + Weak_Modal → ``modal``) is a six-vs-seven dim choice. The seven-dim variant (keep strong and weak separate) is the alternative the literature sometimes ships; we pick the six-dim union because the issue body specifies six features and because the strong / weak distinction does not change the headline diagnostic. A follow-up could sweep both and report whether the strong / weak split changes the head-to-head materially; the runner's arm-resolution is per-cell so the split would be a separate ``--arm`` value rather than a new family.

The runner does not currently exercise the L-M lexicon as a *supplementary* feature alongside the encoder — the head-to-head is "L-M instead of encoder", not "L-M plus encoder". The supplementary variant is out of scope per the issue body; the v1 question is whether the encoder beats the lexicon, not whether stacking the two on the same model lifts further. That follow-up is straightforward (concat the L-M six-vector to the encoder block, widen ``text_embedding_dim`` to encoder_dim + 6) but the gradient sharing argument and the bias-variance read on the small downstream pool make it a separate ablation question.

## Acceptance

A new ablation row sits alongside the §6.19 per-family table once the GPU sweep populates ``backend/artifacts/experiments/per_family_ablation_lm.json``. The cell compares pooled macro-F1 and the regression-band RMSE on ``log(RV)`` between the encoder arm and the ``lm_lexicon`` arm on the canonical 5 seeds × walk-forward folds × 11 cells × 40 epochs surface. The §6 caption frames the result as a methodology diagnostic: encoder beats L-M → the domain-adapted transformer earns its place in the headline; comparable → the ensemble framing is publishable; L-M beats encoder → the methodology contribution rewrites itself around the classical baseline. All three readings are publishable.

The full sweep is GPU-blocked and rides the queued Runpod batch. The runner-side wiring (``--arm lm_lexicon``, the loader plumbing, the lexicon cache contract, the audit-doc row) lands here so the sweep is one command once the queue clears.

## References

- ``backend/app/data/loughran_mcdonald.py`` — lexicon loader + ``compute_lm_features`` / ``compute_lm_feature_vector``.
- ``scripts/run_per_family_ablation.py`` — ``--arm {encoder, lm_lexicon}`` resolution + the L-M post-load injection helper.
- ``tests/unit/test_loughran_mcdonald_loader.py`` — loader contract on a fixture CSV.
- ``tests/unit/test_lm_features.py`` — compute_lm_features against known short documents.
- ADR 0019 — canonical encoder split (the baseline this ablation compares against).
- ADR 0033 — PhraseBank auxiliary fine-tune (the loader-pattern reference for the SHA-pinned external lexicon cache).
- Loughran, T., & McDonald, B. (2011). *When Is a Liability Not a Liability? Textual Analysis, Dictionaries, and 10-Ks*. Journal of Finance.
- https://sraf.nd.edu/loughranmcdonald-master-dictionary/ — McDonald's distribution page for the Master Dictionary.
