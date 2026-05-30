# PR #523 regression — statement_delta / vote_tally / press_conf still fail at runtime

Pod: dev @ 5923f8c (PR #523 "Re-add statement_delta / vote / press flags + wire delta-encoder CLI").
TP: canonical @ a045140 (statement_delta_embedding populated 1427/7172, verified).

The three CLI flags now parse, but the sweeps crash while building tensors.
The other 9 arms are validated-green (canonical, surprise, retrieval, regime,
derived[replacement arm now active], b2, phrasebank, mtl_verify, cross_source).

## Symptoms (1 seed / 2 epochs smoke each)

- statement_delta (`--use-statement-delta`):
  `ValueError: expected sequence of length 87 at dim 2 (got 856)`
  856 = 87 base + 768 (RICH_STATEMENT_DELTA_DIM) + 1 (missing flag)
  raised at backend/app/training/loaders.py:3351  `torch.tensor(sequences, ...)`

- vote_tally (`--use-vote-features`):
  `ValueError: expected sequence of length 87 at dim 2 (got 92)`
  92 = 87 base + 4 (RICH_VOTE_FEATURES_DIM) + 1 (missing flag)
  same site (loaders.py:3351).

- press_conf (`--use-press-conf`):
  `RuntimeError: input.size(-1) must be equal to input_size. Expected 87, got 88`
  88 = 87 base + 1 (RICH_PRESS_CONF_DIM, no missing flag per ADR 0037)
  raised in torch RNN forward — model built with input_size=87 but a fed batch is 88.

## Root cause

`FeatureVector.as_rich_list` (backend/app/models/config.py ~L1257–1275) appends
each of these blocks with a conditional `if self.<block> is not None:` guard —
the same pattern used by the regime/SEP blocks. That pattern is only safe when
the loader broadcasts the block to **every** event. The loader does that for
regime/SEP, but for the three new blocks it leaves the field `None` on events
that lack the data **even when the flag is on**:

backend/app/training/loaders.py ~L2156–2173:
```python
if statement_delta_list is not None:          # statement events w/ strict-prior
    vector.statement_delta_embedding = list(statement_delta_list)
    vector.statement_delta_embedding_missing = 0.0
else:
    vector.statement_delta_embedding = None    # <-- non-statement / cold-start
    vector.statement_delta_embedding_missing = 1.0
# (same None-in-else for vote_features and press_conf_features)
```

So with the flag ON: statement events emit +769 dims, every other event emits
+0 → ragged batch → torch.tensor raises (statement_delta, vote). For press_conf
the data side is uniform-ish but the model's `input_size` (from
`expected_rich_feature_size`, config.py ~L310–331) and the per-event width
disagree → RNN forward raises.

## Suggested fix (laptop side decides)

Mirror the `llm_features` / `analog_features` precedent (config.py ~L55–78),
which ALWAYS emits a fixed-width block, zero-filling when the per-event value is
missing. Cleanest: in the loader `else:` branches, when the flag is on, set the
field to a zero vector of the documented dim with `*_missing = 1.0` instead of
`None`:

```python
else:
    vector.statement_delta_embedding = [0.0] * RICH_STATEMENT_DELTA_DIM
    vector.statement_delta_embedding_missing = 1.0
```

…and likewise `[0.0]*RICH_VOTE_FEATURES_DIM` and `[0.0]*RICH_PRESS_CONF_DIM`.
For press_conf also re-check that `expected_rich_feature_size(use_press_conf=…)`
is plumbed into the model-construction path so input_size matches (the 87-vs-88
RNN error suggests the press dim is added on the data side but not the model
side, or vice-versa).

A unit test asserting `len(as_rich_list())` is identical across a statement
event and a non-statement event with each flag on would have caught all three.

## Files in this bundle
- _smoke_statement_delta.log  (full traceback)
- _smoke_vote_tally.log       (full traceback)
- _smoke_press_conf.log       (full traceback)
- _smoke_derived2.log         (derived replacement arm now active — green)
- _smoke_canonical2.log       (base path green post-pull)
