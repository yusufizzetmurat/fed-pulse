# wf_fold_3 per-fold checkpoint

This directory is the destination for the walk-forward fold #3 forecaster
checkpoint consumed by replay-mode `/analyze` requests whose `as_of_date`
falls in this fold's test window.

Expected file: `forecaster_best.pt` (PyTorch state_dict produced by
`app.train_forecaster`).

To populate:

    docker compose run --rm backend python -m app.train_forecaster \
        --training-package-id canonical \
        --fold-id wf_fold_3 \
        --checkpoint-path /data/processed/canonical/folds/wf_fold_3/forecaster_best.pt

Until this checkpoint is on disk the replay path raises HTTP 422
`fold_checkpoint_missing` for any `as_of_date` resolving to this fold
(see `backend/app/services/replay.py`). The live-mode `/analyze` path
is unaffected and continues to serve from `backend/models/forecaster_best.pt`.
