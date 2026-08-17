# Remote Access Findings

Updated: 2026-08-17 (T038 + T039)
Host: Ubuntu Machine A
Feature: `001-multi-host-training-mvp`
Tasks: `T038` (RTX 4060), `T039` (GTX 1650)

## Scope

This document records the real access path from Ubuntu Machine A to both GPU
Workers through the existing `SSHTransport`.  Both Workers use the **same
transport contract and the same shared check logic**
(`src/shardgrid/transport/remote_access.py`).

Target path:

`Machine A -> SSH -> Windows host -> WSL2 Ubuntu -> selected Conda env -> runtime Python`

No password, private key contents, or other sensitive credentials are stored in
this document.

## Configuration Source

The T038 implementation uses:

- `examples/workers.yaml` through the existing config loader
- `src/shardgrid/transport/ssh.py` for `SSHTransport`
- `tests/address.json` as the current machine-address source for the live RTX
  4060 Worker endpoint

Live RTX 4060 target used on 2026-08-17:

- worker id: `gpu4060`
- host: `10.87.5.155`
- Windows user: `shardgrid`

## Commands Used

Only harmless identity and runtime commands were executed:

1. Windows host identity:

```text
hostname
```

2. WSL distro identity:

```text
wsl.exe -l -v
```

3. WSL runtime Conda executable discovery
4. WSL runtime active Conda state
5. WSL runtime Conda environment list
6. WSL runtime Python version
7. WSL runtime Python executable identity

The WSL runtime commands are executed remotely through the existing
`SSHTransport`, with a PowerShell wrapper that invokes:

```text
wsl.exe -d <detected-distro> -u <worker-user> -- /bin/bash -lc <safe-runtime-command>
```

This keeps the runtime boundary explicit:

- Windows host identity is checked on the Windows layer
- Conda and Python identity are checked only inside WSL2

## Real Verification Result

### Machine A -> RTX 4060 Worker (T038)

- result: `PASS`
- verification date: `2026-08-17`

Real observed values:

- Windows identity: `ldj`
- WSL distro: `Ubuntu`
- Conda executable: `/home/shardgrid/miniconda3/bin/conda`
- selected Conda environment: `shardgrid`
- Conda prefix: `/home/shardgrid/miniconda3/envs/shardgrid`
- runtime Python executable: `/home/shardgrid/miniconda3/envs/shardgrid/bin/python`
- Python version: `Python 3.12.13`

### Machine A -> GTX 1650 Worker (T039)

- result: `PASS`
- verification date: `2026-08-17`
- worker config id: `gpu1060` (physical GPU: NVIDIA GeForce GTX 1650)
- host: `10.87.5.15` (`LAPTOP-5G3QUOGM`)
- Windows user: `shardgrid`

Real observed values:

- Windows identity: `LAPTOP-5G3QUOGM`
- WSL distro: `Ubuntu`
- Conda executable: `/home/shardgrid/miniconda3/bin/conda`
- selected Conda environment: `shardgrid`
- Conda prefix: `/home/shardgrid/miniconda3/envs/shardgrid`
- runtime Python executable: `/home/shardgrid/miniconda3/envs/shardgrid/bin/python`
- Python version: `Python 3.12.13`

Both Workers prove the remote Python comes from the WSL2 selected Conda
environment, not Windows host Python, Machine A Python, or WSL system Python.
Both used the identical `SSHTransport` + `run_remote_access_check` code path
(the same transport contract).

## Access Repair Performed

Before this rerun, Machine A could not authenticate to the Windows Worker over
OpenSSH.

Fix applied on 2026-08-17:

- Machine A public key `~/.ssh/id_ed25519.pub` was added to:
  - `C:\Users\shardgrid\.ssh\authorized_keys` on `10.87.5.155`
  - `C:\Users\shardgrid\.ssh\authorized_keys` on `10.87.5.15`

Post-fix checks from Machine A:

- `ssh -o BatchMode=yes shardgrid@10.87.5.155 hostname` -> `ldj`
- `ssh -o BatchMode=yes shardgrid@10.87.5.15 hostname` -> `LAPTOP-5G3QUOGM`
- `ssh -o BatchMode=yes shardgrid@10.87.5.155 wsl.exe -l -v` -> `Ubuntu Running 2`
- `ssh -o BatchMode=yes shardgrid@10.87.5.15 wsl.exe -l -v` -> `Ubuntu Running 2`

The `10.87.5.15` access repair was done at the same time, but T038 live
verification itself remained scoped to the RTX 4060 Worker only.

## Diagnostics Classification

The T038 integration path now distinguishes at least these failure classes:

- `host_unreachable`
- `connection_timeout`
- `authentication_failure`
- `known_host_failure`
- `wsl_unavailable`
- `conda_unavailable`
- `selected_conda_environment_unavailable`
- `remote_command_non_zero_exit`
- `runtime_python_outside_selected_conda`

When the path fails, the test records a structured `FailureRecord` with:

- `stage=PROBE`
- host
- worker id
- recorded SSH command
- exit code
- recommended action

## Validation

Commands rerun after the access fix:

```text
pytest tests/integration/test_ssh_worker_4060.py -q --run-integration
ruff check tests/integration/test_ssh_worker_4060.py
mypy tests/integration/test_ssh_worker_4060.py
```

Observed results:

- `pytest`: `8 passed`
- `ruff`: `All checks passed`
- `mypy`: `Success: no issues found in 1 source file`

## T038 Status

- T038 task implementation: complete
- RTX 4060 remote access live result: `PASS`
- current blocker for T038: none

## T039 Status

- T039 task implementation: complete
- GTX 1650 remote access live result: `PASS`
- same transport contract as T038: `YES` (`SSHTransport` +
  `run_remote_access_check` shared module)
- current blocker for T039: none
