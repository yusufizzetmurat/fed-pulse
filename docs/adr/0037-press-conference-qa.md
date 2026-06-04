# ADR 0037 — FOMC press conference Q&A as a joint-corpus channel

Issue #214 added the press conference Q&A as a feature channel alongside the prepared statement. The Q&A is high-information-density: a hawkish written statement can be undone by a dovish tone in the Chair's unscripted Q&A 30 minutes later. The pre-#214 pipeline trained on the statement alone, which left that channel invisible to both the model and to readers of the §6 numbers.

## Context

The press conference has a structural asymmetry the pipeline has to handle. Scheduled press conferences only started in 2011, so the pre-2011 portion of the panel (roughly half of the walk-forward folds) has no Q&A artefact to consume. This is treated as a covariate-shift problem rather than a fold-subset problem so the canonical fold protocol stays intact. The feature wiring carries a `has_press_conf` flag (1.0 when a Q&A landed on that meeting, 0.0 otherwise), and the `press_conf_features` slot is a strict-prior zero vector on every pre-2011 event. The model gates on the flag itself; the loss is not masked on pre-2011 rows, because the flag's signal is part of the regime the model is asked to predict.

## Decision

The joint-training surface concatenates the prepared statement and the Q&A transcript into a single longer document before the encoder pass. This is route 1 of the two routes the issue scoping considered. The alternative (sibling encoder for Q&A with a fusion head) is a heavier methodology lift that introduces a second representation surface and a separate calibration story; it remains the right move if the simpler concatenation does not lift the headline. Route 1 lands on the existing encoder and the existing inference contract without rewriting either, and the encoder's positional embeddings already handle long-context documents. The canonical FinBERT-Fed-Adjacent substrate is the same one #424 ran PhraseBank against.

The scraper sits under `BaseSourceScraper` next to `op_fed` (#421) and `gss` (#432). It pulls per-meeting PDFs from `federalreserve.gov/monetarypolicy/fomcpresconf*.htm`, caches them under `data/raw/fomc_press_conferences/`, parses the Q&A section out of the body via `pdfplumber`, and returns one `qa_text` string per event. The events.parquet build step encodes each `qa_text` once and stores the mean-pooled embedding so the trainer reads from cache instead of re-running the encoder per epoch.

Default off: `--use-press-conf=False` keeps `press_conf_features=None` on the FeatureVector and `as_rich_list` emits the pre-#214 per-bar size byte-identical. A checkpoint trained without the flag carries no press-conf state on the inference contract. The `_coerce_payload_config` extension carries `use_press_conf` so an eval or calibration path on a #214 checkpoint reads the same head topology the run trained against.

## Consequences

The §6 canonical-retrain row is the question this ADR fronts: whether a joint statement + Q&A surface lifts the dual-head plus multi-target macro-F1 above the statement-only baseline, and by how much. The retrain is GPU work and rides the Runpod queue.

## References

- `backend/app/data/sources/press_conference.py`, `backend/app/services/scraper_press_conferences.py`
- `backend/app/training/loaders.py` — `has_press_conf` flag + zero-impute path
- `tests/regression/test_feature_provenance_as_of.py` — provenance row
- ADR 0019 (encoder split), ADR 0035 (multi-target heads), #441 / #442
- Hedge-fund industry framework framing (column 3 — contextual data integration)
