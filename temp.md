# Agent Handoff: current worker reality

Date: 2026-08-17

This file exists because earlier agent conclusions mixed old state with current
worker state.

## Read this first

- Do not assume T029/T030 are pending. In `tasks.md`, `T029`, `T030`, and `T031`
  are already `[X]`.
- The real training runtime for Windows GPU workers is **WSL2 Ubuntu inside the
  Windows host**, not Windows Python, not Windows Conda.
- If you run checks from the Windows host and look only at `conda` on `PATH`,
  you can get a false blocker even when WSL is healthy.

## Worker map

- `10.87.5.155`
  - hostname: `LDJ`
  - Windows user: `shardgrid`
  - GPU: `NVIDIA GeForce RTX 4060`
  - training runtime: `WSL2 Ubuntu`
- `10.87.5.15`
  - hostname: `LAPTOP-5G3QUOGM`
  - Windows user: `shardgrid`
  - GPU: `NVIDIA GeForce GTX 1650`
  - training runtime: `WSL2 Ubuntu`
- `10.87.5.228`
  - hostname: `DESKTOP-DFVMAH9`
  - Windows user: `lei`
  - not a GPU training worker

Source: `tests/address.json`

## Known good WSL runtime facts

These facts were already verified for both GPU workers in `docs/wsl-worker.md`.

- WSL distro: `Ubuntu`
- WSL Conda path: `/home/shardgrid/miniconda3/bin/conda`
- selected env: `shardgrid`
- selected prefix: `/home/shardgrid/miniconda3/envs/shardgrid`
- training Python:
  `/home/shardgrid/miniconda3/envs/shardgrid/bin/python`
- PyTorch: `2.7.1+cu118`
- CUDA visible from torch: `true`
- `nvidia-smi` path in WSL: `/usr/lib/wsl/lib/nvidia-smi`

For `10.87.5.155` specifically, the documented verified GPU summary is:

- `NVIDIA GeForce RTX 4060 Laptop GPU, 566.07`

## Important trap

`bootstrap-windows.ps1` can report WSL Conda as missing even when the WSL
runtime is healthy.

Observed on `10.87.5.15`:

- Windows host check saw:
  - WSL present
  - Ubuntu present
  - Windows Conda present
  - `wsl_training_conda.executable` empty
- Read-only probe inside WSL showed:
  - `command -v conda` was empty in that non-interactive shell path
  - `/home/shardgrid/miniconda3/bin/conda` exists
  - `~/.bashrc` contains Conda init for `/home/shardgrid/miniconda3`

Meaning: do not treat a Windows-host-side `conda not found` inside WSL as proof
that WSL has no Conda. Check the actual WSL path directly.

## What to trust

- Trust `docs/wsl-worker.md` for the latest real WSL runtime verification.
- Trust `docs/operations/bootstrap-findings.md` for the T031 explanation of why
  Windows host check and WSL runtime check can disagree.
- Trust `tasks.md` current checkboxes over old conversational summaries.

## What is stale

`docs/windows-worker.md` still contains an older verification note saying
Ubuntu WSL distro was missing on `10.87.5.15`. That is stale relative to later
real verification and should not be used as the current truth.

## If you need to verify a GPU worker again

Use WSL directly, under the WSL user and WSL Conda env. Do not validate GPU
runtime from Windows Python.

Minimal commands to run on a Windows GPU worker:

```powershell
wsl.exe -d Ubuntu -u shardgrid -- bash -lc "ls -l /home/shardgrid/miniconda3/bin/conda"
wsl.exe -d Ubuntu -u shardgrid -- bash -lc "/home/shardgrid/miniconda3/envs/shardgrid/bin/python --version"
wsl.exe -d Ubuntu -u shardgrid -- bash -lc "/home/shardgrid/miniconda3/envs/shardgrid/bin/python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
PY"
wsl.exe -d Ubuntu -u shardgrid -- bash -lc "/usr/lib/wsl/lib/nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader"
```

For repo bootstrap checks from the worker:

```powershell
powershell -ExecutionPolicy Bypass -File .\bootstrap-windows.ps1 -Check -Json
wsl.exe -d Ubuntu -u shardgrid -- bash -lc "~/.shardgrid/scripts/bootstrap-wsl.sh --check --json"
```

## Bottom line for the next agent

- Do not regress to "worker not prepared" without rereading `tasks.md`,
  `docs/wsl-worker.md`, and `docs/operations/bootstrap-findings.md`.
- The likely failure mode is checking the wrong runtime layer.
- For GPU truth, WSL `shardgrid` env is the source of truth.
