# Hardware Findings

Updated: 2026-08-17 (re-execution attempt recorded)
Host: Ubuntu Machine A (control node)
Feature: `001-multi-host-training-mvp`

## Purpose

Real single-GPU CUDA/PyTorch smoke evidence for the required GPU Workers,
executed inside each Worker's WSL2 training runtime with the selected Conda
environment.  Evidence is written as JSON findings (PASS / FAIL / BLOCKED)
under `$HOME/.shardgrid/hardware-findings/` (override with
`SHARDGRID_HARDWARE_FINDINGS_DIR`).

## Worker reality (per `tests/address.json`, `docs/wsl-worker.md`, `temp.md`)

| IP | Hostname | User | GPU | Role |
|---|---|---|---|---|
| `10.87.5.155` | `LDJ` | `shardgrid` | NVIDIA GeForce RTX 4060 Laptop GPU | RTX 4060 Worker |
| `10.87.5.15` | `LAPTOP-5G3QUOGM` | `shardgrid` | NVIDIA GeForce GTX 1650 | second GPU Worker |
| `10.87.5.228` | `DESKTOP-DFVMAH9` | `lei` | none | not a GPU worker |

Note: the real second GPU Worker is a **GTX 1650**, not the GTX 1060 assumed in
early planning.  `docs/wsl-worker.md` (T030) records the real verified facts.

## Verified WSL2 training runtime facts (already real, per `docs/wsl-worker.md`)

Both GPU Workers were really verified on 2026-08-17 inside WSL2:

- WSL distro: `Ubuntu`
- WSL Conda executable: `/home/shardgrid/miniconda3/bin/conda`
- Selected Conda environment: `shardgrid`
- Selected prefix: `/home/shardgrid/miniconda3/envs/shardgrid`
- Training Python: `/home/shardgrid/miniconda3/envs/shardgrid/bin/python` (Python 3.12.13)
- PyTorch: `2.7.1+cu118`
- `torch.cuda.is_available() == true`, CUDA 11.8
- `nvidia-smi` inside WSL: `/usr/lib/wsl/lib/nvidia-smi`

These facts come from real WSL execution, not from Windows-host Python or
configuration.  A Windows-host-side `conda not found` inside WSL is a known
false blocker (see `docs/operations/bootstrap-findings.md`); the WSL path is
the source of truth.

## RTX 4060 Worker (`10.87.5.155` / `LDJ` / `gpu4060`)

### Test: `tests/hardware/test_worker_4060.py`

Runs inside the WSL2 training runtime with the selected Conda environment.
Opt-in via `--run-hardware` + `SHARDGRID_ENABLE_HARDWARE_TESTS=1`.  It:

1. Fails unless the running Python is inside the selected Conda environment
   (`CONDA_PREFIX`), so Windows-host or system Python can never be used as the
   training runtime.
2. Records `torch.__version__`, `torch.version.cuda`,
   `torch.cuda.is_available()`, `torch.cuda.device_count()`, the current CUDA
   device, GPU name, total VRAM, and compute capability from the actual
   runtime.
3. Verifies the detected GPU name contains `RTX 4060` (real detection, never
   inferred from configuration).
4. Executes a real `1024x1024 @ 1024x1024` CUDA matmul, calls
   `torch.cuda.synchronize()`, moves the result to CPU, and verifies the shape
   and that all values are finite.
5. Records `nvidia-smi` query output.
6. Always writes a JSON finding with commands, versions, results, timestamp,
   and an explicit PASS / FAIL / BLOCKED status.

### Status: BLOCKED - SSH authentication (PENDING LIVE VERIFICATION FROM MACHINE A)

The Workers are prepared, and their WSL runtimes were really verified (facts
above).  Re-executing the smoke test from Machine A was attempted on
2026-08-17 and is currently blocked by access authentication:

- `ssh shardgrid@10.87.5.155` returns
  `Permission denied (publickey,password,keyboard-interactive)`.
- Machine A's key (`/home/yangjilei/.ssh/id_ed25519`) is not in the Worker's
  authorized keys, and no `10.87.5.*` entry exists in Machine A's
  `known_hosts` for this session.

**Manual action required**: authorize Machine A's public key for the Windows
OpenSSH user `shardgrid` on `10.87.5.155` (and `10.87.5.15`), then rerun
`tests/hardware/test_worker_4060.py` on the Worker via SSH + WSL.

This is a real BLOCKED record with a real reason.  It is not a PASS, and it is
not the stale "worker not prepared" conclusion: the workers are prepared and
their runtime facts are documented.

### Commands to run on the prepared Worker (inside WSL2, selected Conda env)

```powershell
wsl.exe -d Ubuntu -u shardgrid -- bash -lc "/home/shardgrid/miniconda3/envs/shardgrid/bin/python --version"
wsl.exe -d Ubuntu -u shardgrid -- bash -lc "/home/shardgrid/miniconda3/envs/shardgrid/bin/python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_capability(0))
print(torch.cuda.get_device_properties(0).total_memory)
a = torch.randn(1024, 1024, device='cuda')
b = torch.randn(1024, 1024, device='cuda')
c = (a @ b).cpu()
torch.cuda.synchronize()
assert c.shape == (1024, 1024) and bool(torch.isfinite(c).all())
print('CUDA tensor operation OK')
PY"
wsl.exe -d Ubuntu -u shardgrid -- bash -lc "/usr/lib/wsl/lib/nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader"
```

## GTX 1650 Worker (`10.87.5.15` / `LAPTOP-5G3QUOGM`)

### Test: `tests/hardware/test_worker_1650.py` (T035)

Mirrors the RTX 4060 test structure (same finding schema, same checks):

1. Fails unless the running Python is inside the selected Conda environment.
2. Records `torch.__version__`, `torch.version.cuda`,
   `torch.cuda.is_available()`, `torch.cuda.device_count()`,
   `torch.cuda.current_device()`, GPU name, total VRAM, and compute capability.
3. Verifies the detected GPU name contains `GTX 1650`; a mismatch is FAIL with
   the real detected value.
4. Executes a real `1024x1024 @ 1024x1024` CUDA matmul with
   `torch.cuda.synchronize()`, returns the result to CPU, and verifies shape
   and finite values.
5. Records `nvidia-smi` query output (name, memory, driver version, compute
   capability).
6. Writes a JSON finding (PASS / FAIL / BLOCKED / PENDING) to
   `$HOME/.shardgrid/hardware-findings/gtx1650-smoke-<timestamp>.json`.

### Status: PENDING LIVE VERIFICATION (HARDWARE PASS NOT PASSED)

- TASK IMPLEMENTATION: **COMPLETED** (test + finding structure in place).
- LIVE HARDWARE VERIFICATION: **PENDING**.
- HARDWARE PASS: **NOT PASSED**.

Machine A cannot currently authenticate to the GTX 1650 Worker:
`ssh shardgrid@10.87.5.15` returns
`Permission denied (publickey,password,keyboard-interactive)`.  This is an
access-authentication blocker, not a GPU failure and not a missing
SSHTransport dependency (T035 does not depend on T037).

The Worker is prepared and its WSL2 runtime was really verified previously
(see `docs/wsl-worker.md`):

- WSL distro: `Ubuntu`
- Conda executable: `/home/shardgrid/miniconda3/bin/conda`
- Selected environment: `shardgrid` (`/home/shardgrid/miniconda3/envs/shardgrid`)
- Python: `/home/shardgrid/miniconda3/envs/shardgrid/bin/python` (3.12.13)
- PyTorch: `2.7.1+cu118`; CUDA visible (`torch.cuda.is_available() == true`)
- NVIDIA driver major `527` (WSL2 CUDA floor satisfied)

**Manual action**: authorize Machine A's public key for Windows OpenSSH user
`shardgrid` on `10.87.5.15`, or log into the Worker manually and run the test
inside WSL2:

```powershell
wsl.exe -d Ubuntu -u shardgrid -- bash -lc "cd ~/ShardGrid && SHARDGRID_ENABLE_HARDWARE_TESTS=1 python -m pytest --run-hardware tests/hardware/test_worker_1650.py"
```

## Summary

| Worker | Real GPU | T034/T035 result | Evidence |
|---|---|---|---|
| gpu4060 (`10.87.5.155`) | RTX 4060 Laptop GPU | T034: BLOCKED (SSH auth); runtime facts verified in `docs/wsl-worker.md` | `$HOME/.shardgrid/hardware-findings/rtx4060-smoke-*.json` |
| gpu1650 (`10.87.5.15`) | GTX 1650 | T035: PENDING LIVE VERIFICATION (SSH auth); runtime facts verified in `docs/wsl-worker.md` | `$HOME/.shardgrid/hardware-findings/gtx1650-smoke-*.json` |
