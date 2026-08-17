# Bootstrap Findings

Date: 2026-08-17
Host: Ubuntu Machine A
Feature: `001-multi-host-training-mvp`
Task: `T031`

## Scope

This document records the T031 idempotence and safety validation for:

- `scripts/bootstrap-linux.sh`
- `scripts/bootstrap-windows.ps1`
- `scripts/bootstrap-wsl.sh`

The goal here is validation, not silent repair. Findings are kept deterministic,
actionable, and non-destructive. Real platform execution is recorded only where
it actually happened.

## Automated Coverage

`tests/integration/test_bootstrap_idempotence.py` covers safe simulation for:

- repeat execution with an already compatible Conda environment
- partial installation repair without deleting or overwriting an existing environment
- missing Conda detection
- reuse of a compatible Conda environment
- creation of a `shardgrid` Conda environment only when needed
- no overwrite of an existing Conda environment
- manual-action blockers
- Windows host versus WSL training runtime separation

These tests do not remove real Conda, WSL, CUDA, or system software.

## Real Verification Summary

| Bootstrap | Platform | Verification | Run 1 | Run 2 | Status |
|---|---|---|---|---|---|
| `bootstrap-linux.sh` | Ubuntu Machine A | real | `blocked_manual_action` | `blocked_manual_action` | `MANUAL ACTION` |
| `bootstrap-windows.ps1` | `10.87.5.15` Windows host | real | `blocked_manual_action` | `blocked_manual_action` | `MANUAL ACTION` |
| `bootstrap-wsl.sh` | `10.87.5.15` Ubuntu WSL2 runtime | real | `healthy` | `healthy` | `PASS` |

## Linux Bootstrap

Status: `MANUAL ACTION`

Real execution was run twice on Ubuntu Machine A with:

- Conda executable: `/home/yangjilei/anaconda3/bin/conda`
- active environment: `base`
- active prefix: `/home/yangjilei/anaconda3`
- selected environment: `base`
- selected Python: `/home/yangjilei/anaconda3/bin/python`
- Python version: `Python 3.13.5`

Observed environment and tools:

- `git`: `git version 2.43.0`
- `ssh`: `OpenSSH_9.6p1 Ubuntu-3ubuntu13.15`
- `iperf3`: `not_installed`
- project dependencies: `present`

Deterministic outcome from both runs:

- health: `blocked_manual_action`
- create needed: `false`
- reuse environment: `base`
- manual action: `install iperf3 (operator action: sudo apt-get install -y iperf3)`

This is idempotent and non-destructive. It reused the existing Conda
environment and did not try to replace it.

## Windows Bootstrap

Status: `MANUAL ACTION`

Real execution was run twice on prepared GPU Worker `10.87.5.15`
(`LAPTOP-5G3QUOGM`) with:

- PowerShell: `5.1.26100.9168`
- OpenSSH client: `C:\Windows\System32\OpenSSH\ssh.exe`
- OpenSSH server: `C:\Windows\System32\OpenSSH\sshd.exe`
- OpenSSH service status: `running_observed_via_ssh`
- WSL executable: `C:\WINDOWS\system32\wsl.exe`
- Ubuntu present: `true`
- Ubuntu WSL2: `true`
- `nvidia-smi`: `C:\WINDOWS\system32\nvidia-smi.exe`
- driver major: `527`
- WSL CUDA compatibility: `true`
- Windows host Conda executable: `D:\Anaconda\anaconda3\Scripts\conda.exe`
- Windows host Conda version: `conda 24.9.2`

Deterministic outcome from both runs:

- health: `blocked_manual_action`
- Windows host Conda remained host-only and was not treated as training runtime
- WSL training Conda executable: empty
- WSL training Conda version: `not_checked`
- manual action: `install or select Conda inside the Ubuntu WSL2 training runtime; do not use Windows-host Conda for training`

Additional read-only probe inside the same Worker showed:

- `command -v conda` inside non-interactive WSL returned empty
- `/home/shardgrid/miniconda3/bin/conda` exists
- `~/.bashrc` contains Conda init lines for `/home/shardgrid/miniconda3`

This means the current T029 Windows host check is stable and honest, but it is
detecting the WSL training Conda only through the current shell resolution path.
The prepared Worker does have a usable WSL Conda runtime, but that runtime is
not surfaced to the Windows host check in its current execution path. This is a
real finding, not a mocked pass.

## WSL Bootstrap

Status: `PASS`

Real execution was run twice on the Ubuntu WSL2 runtime of `10.87.5.15` with:

- WSL distro: `Ubuntu`
- kernel: `Linux 6.18.33.2-microsoft-standard-WSL2 x86_64 GNU/Linux`
- Conda executable: `/home/shardgrid/miniconda3/bin/conda`
- Conda version: `conda 26.5.3`
- known environments: `base`, `shardgrid`, `shardgrid.broken-20260817135348`
- active environment: `none`
- selected environment: `shardgrid`
- selected prefix: `/home/shardgrid/miniconda3/envs/shardgrid`
- selected Python: `/home/shardgrid/miniconda3/envs/shardgrid/bin/python`
- Python version: `Python 3.12.13`
- system Python: `/usr/bin/python3` (`Python 3.14.4`)
- PyTorch: `2.7.1+cu118`
- CUDA version: `11.8`
- `torch.cuda.is_available()`: `true`
- `nvidia-smi`: `/usr/lib/wsl/lib/nvidia-smi`
- GPU summary: `NVIDIA GeForce GTX 1650, 527.41`
- `iperf3`: `iperf 3.20 (cJSON 1.7.15)`

Deterministic outcome from both runs:

- health: `healthy`
- create needed: `false`
- reuse environment: `shardgrid`
- manual actions: none

This is idempotent and non-destructive. It reused the existing training Conda
environment and did not overwrite it.

## Partial Installation And Manual-Action Simulation

Safe simulation coverage passed for:

- partial Linux environment repair without deleting a valid environment
- WSL environment creation only when no compatible environment exists
- missing Conda and missing tools detection
- manual-action blockers instead of unsafe auto-install behavior
- Windows host and WSL runtime separation

No simulation deleted, downgraded, or replaced a real Conda environment.

## Pending Platform Verification

- `PENDING PLATFORM VERIFICATION`: no native local execution of `bootstrap-windows.ps1` from Ubuntu Machine A itself; Windows evidence here comes from the prepared remote Worker
- `PENDING PLATFORM VERIFICATION`: no new T031 rerun was performed on `10.87.5.155`; prior T030 evidence exists, but this task required only at least one prepared Worker for real verification

## Outcome

`T031` is complete.

- automated idempotence and safety coverage: `PASS`
- Linux bootstrap findings: `MANUAL ACTION`
- Windows bootstrap findings: `MANUAL ACTION`
- WSL bootstrap findings: `PASS`
- real Worker verification: `PASS` on one prepared GPU Worker with honest manual-action reporting where applicable
- findings quality: deterministic, actionable, and non-destructive
