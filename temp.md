# Agent Handoff

Updated: 2026-08-17 (before T049 execution)

This file is the short handoff for the next agent. Read this before touching
the worker-access path or the distributed path.

## Status pollution warning

Earlier agent conclusions were polluted by stale state.

What happened:

- old conversational state was reused after the real worker state had already
  changed
- older notes said Windows/WSL/Conda or worker access was still blocked
- later real verification had already proved part of that was no longer true

Concrete stale conclusions that already happened:

- `T029` / `T030` were treated as still incomplete after they had already been
  completed
- the RTX 4060 worker was treated as "not prepared" after WSL runtime
  verification was already real and recorded
- `T038` was treated as still blocked on SSH after Machine A public-key access
  had already been fixed

Rule for the next agent:

- do not trust older conversational summaries over current repo evidence
- re-read `tasks.md`, `docs/operations/bootstrap-findings.md`,
  `docs/operations/remote-access.md`, and `record.md` before declaring a
  platform blocked
- if a current file and an older conversation disagree, trust the current file
  and rerun the live command if needed

## Current task state

- `T046` `[X]` (distributed smoke program)
- `T047` `[X]` (multi-host runner dry-run)
- `T048` `[X]` (NCCL-first backend/interface/rendezvous selection)
- `T049` `[ ]` (live two-host NCCL collectives) - the active task

Do not regress completed tasks back to blocked without fresh live evidence.

## Current worker reality

Source of truth for machine addresses:

- `tests/address.json`

Current GPU workers:

- `10.87.5.155`
  - hostname: `LDJ`
  - Windows user: `shardgrid`
  - GPU: `NVIDIA GeForce RTX 4060` (rank 0)
- `10.87.5.15`
  - hostname: `LAPTOP-5G3QUOGM`
  - Windows user: `shardgrid`
  - GPU: `NVIDIA GeForce GTX 1650` (rank 1)

Non-GPU Windows machine:

- `10.87.5.228`
  - hostname: `DESKTOP-DFVMAH9`
  - Windows user: `lei`

Both GPU workers are reachable from Machine A as of 2026-08-17:

- SSH + WSL2 Ubuntu respond on both hosts
- `shardgrid` Conda env on each WSL runtime has torch 2.7.1+cu118
- `torch.cuda.is_available() == True` on both
- GPU names verified: `NVIDIA GeForce RTX 4060 Laptop GPU`, `NVIDIA GeForce GTX 1650`

## WSL runtime truth

For both GPU workers, the real training runtime is:

- Windows host
- `WSL2 Ubuntu`
- Conda executable: `/home/shardgrid/miniconda3/bin/conda`
- selected env: `shardgrid`
- prefix: `/home/shardgrid/miniconda3/envs/shardgrid`
- runtime Python: `/home/shardgrid/miniconda3/envs/shardgrid/bin/python`

Do not treat Windows Python or Windows Conda as the training runtime.

## Known blocker for T049 (recorded honestly)

WSL2 currently runs in **NAT networking mode** on both workers:

- gpu4060 WSL IP: `172.20.208.137/20` (peer unreachable)
- gpu1060 WSL IP: `192.168.132.81/20` (peer unreachable)
- Windows LAN IPs (`10.87.5.155` / `10.87.5.15`) are NOT local addresses
  inside the WSL VMs, so a TCPStore master cannot bind them there
- `tcp://10.87.5.155:29500` rendezvous therefore hangs

Known clean fix candidates (not yet applied unless a later agent applies them):

- WSL mirrored networking: add `networkingMode=mirrored` to
  `C:\Users\shardgrid\.wslconfig` on both workers, then `wsl --shutdown`
- or Windows `netsh interface portproxy` + firewall rules (more fragile for
  NCCL because of dynamic data-channel ports)

Do not claim NCCL success without a real two-host process group and real
tensor-validated collectives.

## Files to trust

- `docs/wsl-worker.md`
- `docs/operations/bootstrap-findings.md`
- `docs/operations/remote-access.md`
- `docs/operations/hardware-findings.md`
- `attention.md`
- `specs/001-multi-host-training-mvp/tasks.md`
- `record.md`
