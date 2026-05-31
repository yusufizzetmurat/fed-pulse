Pod-only artefacts (not on laptop) — 2026-05-31 teardown kit.
drivers/   every experiment driver written on the pod (wave1-4, stage2, salvage, jobs1-3,
           absolute_nulls.py CPU null-table, dl_tp.py TP downloader). Exact recipes.
patches/   two uncommitted local tweaks (defaults preserve byte-identity for other runs):
  embedding_cache_gpu_device.patch  -> model.to(device), env FED_PULSE_EMBEDDING_CACHE_DEVICE (#553 contract)
  run_dual_head_dropout_lr_flags.patch -> --dropout / --learning-rate CLI threaded to ModelConfig.dropout / train_model.lr
NOTE: embedding caches (data/raw/embeddings/, ~610MB incl voyage @ API cost) NOT in this kit — see chat re: whether to ship.
