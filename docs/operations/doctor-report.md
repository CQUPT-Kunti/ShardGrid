# Doctor Hardware Report

Updated: 2026-08-27
Host: Ubuntu control node
Feature: `001-multi-host-training-mvp`
Tasks: `T081`

## Scope

This report records real Doctor execution for each physical GPU Worker through
the existing worker Doctor workflow, then cross-checks the reported runtime
facts with independent `hostname`, `nvidia-smi`, Torch/CUDA, route, and MTU
commands.

The second worker keeps config id `gpu1060` for compatibility, but the actual
hardware detected and reported is `NVIDIA GeForce GTX 1650`.

## Doctor Command

The real validation used a temporary real-IP config and a temporary local
`ssh` wrapper so Doctor could reuse the existing `SSHTransport` path while
supplying the session password outside the repository.

Acceptance command shape:

```text
python -c "from shardgrid.control.doctor import _run_worker_doctor; ..."
```

Common temporary config values:

- `gpu4060` -> `10.87.5.155`
- `gpu1060` -> `10.87.5.15`
- `runtime_distro` -> `Ubuntu-22.04`
- `conda_environment` -> `shardgrid`
- `network.nccl_mtu` -> `1500`
- `ssh.connect_timeout_seconds` -> `20`
- `ssh.strict_host_key_checking` -> `false`

## Independent Validation Commands

Commands were run independently on each worker after the Doctor report:

```text
ssh shardgrid@<host> hostname
ssh shardgrid@<host> "wsl.exe -d Ubuntu-22.04 -u shardgrid -- /usr/lib/wsl/lib/nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader"
ssh shardgrid@<host> "wsl.exe -d Ubuntu-22.04 -u shardgrid -- /home/shardgrid/miniconda3/envs/shardgrid/bin/python -c 'import json,platform,torch; ...'"
ssh shardgrid@<host> "wsl.exe -d Ubuntu-22.04 -u shardgrid -- ip route get <peer>"
ssh shardgrid@<host> "wsl.exe -d Ubuntu-22.04 -u shardgrid -- ip link show <dev>"
```

No password, token, or other credential is stored in this report.

## Real Results

### RTX 4060 Worker

| Field | Doctor | Independent command |
|---|---|---|
| worker id | `gpu4060` | n/a |
| Windows host identity | `ldj` | `hostname` -> `ldj` |
| runtime distro | `Ubuntu-22.04` | `wsl.exe -l -v` path already verified by Doctor access chain |
| Conda env | `shardgrid` | runtime Python path under `/home/shardgrid/miniconda3/envs/shardgrid/bin/python` |
| Python | `Python 3.12.13` | Torch probe -> `3.12.13` |
| PyTorch | `2.7.1+cu118` | Torch probe -> `2.7.1+cu118` |
| CUDA runtime | `11.8` | Torch probe -> `11.8` |
| GPU | `NVIDIA GeForce RTX 4060 Laptop GPU` | `nvidia-smi` -> `NVIDIA GeForce RTX 4060 Laptop GPU` |
| Driver | `566.07` | `nvidia-smi` -> `566.07` |
| NCCL | `2.21.5` | Torch probe -> `2.21.5` |
| peer route | `10.87.5.15 dev eth3 src 10.87.5.155` | `ip route get 10.87.5.15` matched |
| MTU | `eth3`, `1500` | `ip link show eth3` -> `mtu 1500` |
| Doctor health | `healthy` | cross-check matched |

### GTX 1650 Worker

| Field | Doctor | Independent command |
|---|---|---|
| worker id | `gpu1060` | n/a |
| actual GPU | `NVIDIA GeForce GTX 1650` | `nvidia-smi` -> `NVIDIA GeForce GTX 1650` |
| Windows host identity | `LAPTOP-5G3QUOGM` | `hostname` -> `LAPTOP-5G3QUOGM` |
| runtime distro | `Ubuntu-22.04` | `wsl.exe -l -v` path already verified by Doctor access chain |
| Conda env | `shardgrid` | runtime Python path under `/home/shardgrid/miniconda3/envs/shardgrid/bin/python` |
| Python | `Python 3.12.13` | Torch probe -> `3.12.13` |
| PyTorch | `2.7.1+cu118` | Torch probe -> `2.7.1+cu118` |
| CUDA runtime | `11.8` | Torch probe -> `11.8` |
| GPU | `NVIDIA GeForce GTX 1650` | `nvidia-smi` -> `NVIDIA GeForce GTX 1650` |
| Driver | `527.41` | `nvidia-smi` -> `527.41` |
| NCCL | `2.21.5` | Torch probe -> `2.21.5` |
| peer route | `10.87.5.155 dev eth0 src 10.87.5.15` | `ip route get 10.87.5.155` matched |
| MTU | `eth0`, `1500` | `ip link show eth0` -> `mtu 1500` |
| Doctor health | `healthy` | cross-check matched |

## Unavailable Capability Reporting

This run did not observe an unavailable or blocked capability on either worker.
Doctor therefore reported all worker checks as `PASS`, and the independent
commands confirmed that this was accurate for the verified SSH + WSL2 +
Conda + Torch + CUDA + NCCL path.

The honesty rule still applies:

- unsupported or missing capability must remain `UNAVAILABLE`, `WARNING`,
  `FAIL`, or `BLOCKED`
- non-required unavailable capability is not silently upgraded to `PASS`
- the compatibility worker id `gpu1060` must not be confused with a real GTX
  1060 result

## Outcome

- RTX 4060 Worker Doctor workflow: PASS
- GTX 1650 Worker Doctor workflow: PASS
- `nvidia-smi` cross-check: PASS
- Torch/CUDA/NCCL cross-check: PASS
- dynamic peer route + MTU 1500 cross-check: PASS
- T081 overall result: PASS
