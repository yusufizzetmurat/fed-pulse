# GPU Setup Notes (RTX 4080)

## 1) NVIDIA Container Toolkit
Install and verify NVIDIA Container Toolkit so Docker can access GPU devices.

Quick check:
- `nvidia-smi` on host
- `docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi`

## 2) CUDA-compatible PyTorch
For CUDA 12.1 environments, use:
- `backend/requirements-gpu-cu121.txt`

This file installs `torch` from the CUDA 12.1 wheel index.

## 3) Data regeneration expectation
If `data/` is gitignored (recommended), new machines must regenerate local artifacts:
- Run scraper once to recreate `fomc_statements.*` and `fomc_minutes.*` under `data/`.

## 4) Cross-machine note
Keep `backend/requirements.txt` as portable default.
Use `backend/requirements-gpu-cu121.txt` only on CUDA-capable NVIDIA hosts.
