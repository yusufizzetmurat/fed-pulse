# Per-fold forecaster training for replay mode

Replay-mode `/analyze` (`as_of_date` set) serves predictions off a
checkpoint pinned to the walk-forward fold whose `train_end` is the
largest date strictly before `as_of_date`. The fold layout is described
in `data/processed/canonical/fold_manifest_expanding_walk_forward.json`;
each fold entry carries a `checkpoint_dir` pointing at
`data/processed/canonical/folds/wf_fold_N/`.

The replay path looks for `forecaster_best.pt` inside that directory.
If the file is missing the endpoint surfaces
`HTTP 422 {"error": "replay_unavailable", "message": "fold_checkpoint_missing"}`
and `replay.forecaster_checkpoint_rewound` stays `false`. The live-mode
`/analyze` path is unaffected and continues to serve from
`backend/models/forecaster_best.pt`.

## Commands

Run one fold at a time:

    docker compose run --rm backend python -m app.train_forecaster \
        --training-package-id canonical \
        --fold-id wf_fold_1 \
        --checkpoint-path /data/processed/canonical/folds/wf_fold_1/forecaster_best.pt

Run the full set in a loop:

    for fold in 1 2 3 4 5; do
      docker compose run --rm backend python -m app.train_forecaster \
        --training-package-id canonical \
        --fold-id wf_fold_$fold \
        --checkpoint-path /data/processed/canonical/folds/wf_fold_$fold/forecaster_best.pt
    done

The container mounts the host `./data` at `/data`, so the `--checkpoint-path`
above lands at `data/processed/canonical/folds/wf_fold_N/forecaster_best.pt`
on the host (the same path the replay service reads).

## Wall-clock budget

Per-fold training is a one-time cost paid out-of-band so the replay
endpoint can serve the right checkpoint at request time. Order-of-
magnitude expectations:

| Hardware | Per-fold wall-clock | 5-fold total |
| --- | --- | --- |
| CPU (Docker default) | ~30-60 min | ~3-5 hours |
| Single CUDA GPU | ~5 min | ~25 min |

Use the GPU profile (`make dev-gpu`) when training is the bottleneck;
inference at replay time is fast either way because the per-fold model
is cached in-process by `app.services.forecaster.load_for_fold`.

## After training

No backend restart is needed. The next replay-mode request that
resolves to a fold whose `forecaster_best.pt` has just landed will:

1. resolve the fold via the manifest;
2. load the new checkpoint into a process-local LRU (separate from the
   live serving singleton, so concurrent live `/analyze` requests are
   not disturbed);
3. set `replay.forecaster_checkpoint_rewound = true` on the response.

If a checkpoint is replaced on disk while the process is running, call
`app.services.forecaster.clear_fold_load_cache()` (or restart the
process) so the LRU re-reads the file.
