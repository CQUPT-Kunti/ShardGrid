# Agent Handoff

Date: 2026-08-17

This file is the short handoff for the next agent. Read this before touching
the worker-access path.

## Current task state

- `T029`: `[X]`
- `T030`: `[X]`
- `T031`: `[X]`
- `T037`: `[X]`
- `T038`: `[X]`
- `T039`: not started in this handoff

Do not regress `T038` back to blocked unless you have fresh live evidence.

## Current worker reality

Source of truth for machine addresses:

- `tests/address.json`

Current GPU workers:

- `10.87.5.155`
  - hostname: `LDJ`
  - Windows user: `shardgrid`
  - GPU: `NVIDIA GeForce RTX 4060`
- `10.87.5.15`
  - hostname: `LAPTOP-5G3QUOGM`
  - Windows user: `shardgrid`
  - GPU: `NVIDIA GeForce GTX 1650`

Non-GPU Windows machine:

- `10.87.5.228`
  - hostname: `DESKTOP-DFVMAH9`
  - Windows user: `lei`

## WSL runtime truth

For both GPU workers, the real training runtime is:

- Windows host
- `WSL2 Ubuntu`
- Conda executable: `/home/shardgrid/miniconda3/bin/conda`
- selected env: `shardgrid`
- prefix: `/home/shardgrid/miniconda3/envs/shardgrid`
- runtime Python: `/home/shardgrid/miniconda3/envs/shardgrid/bin/python`

Do not treat Windows Python or Windows Conda as the training runtime.

## SSH access truth

Machine A public key was added to both GPU workers on 2026-08-17:

- `C:\Users\shardgrid\.ssh\authorized_keys` on `10.87.5.155`
- `C:\Users\shardgrid\.ssh\authorized_keys` on `10.87.5.15`

Verified from Machine A:

- `ssh -o BatchMode=yes shardgrid@10.87.5.155 hostname` works
- `ssh -o BatchMode=yes shardgrid@10.87.5.15 hostname` works
- `ssh -o BatchMode=yes shardgrid@10.87.5.155 wsl.exe -l -v` works
- `ssh -o BatchMode=yes shardgrid@10.87.5.15 wsl.exe -l -v` works

Meaning:

- the earlier `authentication_failure` blocker for `T038` is resolved
- `T039` is no longer blocked on SSH entry

## T038 live result

`T038` was rerun live and passed on `10.87.5.155`.

Observed live values:

- Windows identity: `ldj`
- WSL distro: `Ubuntu`
- Conda executable: `/home/shardgrid/miniconda3/bin/conda`
- Conda env: `shardgrid`
- Conda prefix: `/home/shardgrid/miniconda3/envs/shardgrid`
- runtime Python: `/home/shardgrid/miniconda3/envs/shardgrid/bin/python`
- Python version: `Python 3.12.13`

## Files to trust

- `docs/wsl-worker.md`
- `docs/operations/bootstrap-findings.md`
- `docs/operations/remote-access.md`
- `attention.md`
- `specs/001-multi-host-training-mvp/tasks.md`

## Important trap

Windows-host-side checks can still disagree with WSL runtime truth if they only
look at shell `PATH`.

Do not conclude "WSL has no conda" just because a Windows-side check failed to
resolve `conda` through a non-interactive shell. Check the real WSL path.

## If the next task is T039

The direct analog path to verify is:

`Machine A -> 10.87.5.15 -> Windows host -> WSL2 Ubuntu -> shardgrid env -> runtime Python`

Use the existing `SSHTransport`. Do not create a second SSH implementation.
