# Record

Date: 2026-08-17
Scope: recent worker-access and handoff work

## What was done

### 1. Fixed SSH access from Machine A to both GPU workers

Machine A public key:

- `~/.ssh/id_ed25519.pub`

was added to:

- `C:\Users\shardgrid\.ssh\authorized_keys` on `10.87.5.155`
- `C:\Users\shardgrid\.ssh\authorized_keys` on `10.87.5.15`

Result:

- Machine A can now use non-interactive SSH to both GPU workers
- this removed the earlier `authentication_failure` blocker

### 2. Reworked T038 live verification

Implemented / updated:

- `tests/integration/test_ssh_worker_4060.py`
- `docs/operations/remote-access.md`

The integration path now proves:

- Machine A
- `SSHTransport`
- Windows physical host
- WSL2 Ubuntu
- selected Conda environment
- runtime Python

The runtime check intentionally stops at identity/runtime evidence and does not
expand into GPU probe or training.

### 3. Reran T038 live after the SSH repair

Observed live `T038` result on `10.87.5.155`:

- status: `PASS`
- Windows identity: `ldj`
- WSL distro: `Ubuntu`
- Conda executable: `/home/shardgrid/miniconda3/bin/conda`
- Conda environment: `shardgrid`
- Conda prefix: `/home/shardgrid/miniconda3/envs/shardgrid`
- runtime Python: `/home/shardgrid/miniconda3/envs/shardgrid/bin/python`
- Python version: `Python 3.12.13`

### 4. Added project-note documents

Added / updated:

- `attention.md`
- `temp.md`
- `record.md`

Purpose:

- `attention.md`: known practical problems that will matter as the full flow is
  expanded
- `temp.md`: short handoff for the next agent
- `record.md`: plain record of what this task batch actually did

## Validation run

Executed:

- `pytest tests/integration/test_ssh_worker_4060.py -q --run-integration`
- `ruff check tests/integration/test_ssh_worker_4060.py`
- `mypy tests/integration/test_ssh_worker_4060.py`

Observed:

- `pytest`: `8 passed`
- `ruff`: passed
- `mypy`: passed

## Current outcome

- `T038`: complete and live verified
- both GPU workers: SSH entry from Machine A fixed
- `T039`: ready to start

## T039 (this task batch)

### 1. Shared the remote-access check for both Workers

- Added `src/shardgrid/transport/remote_access.py`
- `run_remote_access_check(transport, worker, *, worker_label)` is the single
  shared check implementation used by BOTH Workers (one transport contract)
- Refactored `tests/integration/test_ssh_worker_4060.py` to import the shared
  module (same behavior, 8 tests still pass)

### 2. Added T039 GTX 1650 integration test

- `tests/integration/test_ssh_worker_1650.py` (8 tests)
- Uses the exact same `SSHTransport` + shared check as T038
- Covers: success path, connection failure classification
  (host_unreachable / connection_timeout / authentication_failure /
  known_host_failure), WSL/Conda failure, Python-outside-selected-Conda
  rejection, and a live remote-access attempt

### 3. Live GTX 1650 remote access PASS

Executed from Machine A on 2026-08-17:

- status: `PASS`
- Windows identity: `LAPTOP-5G3QUOGM`
- WSL distro: `Ubuntu`
- Conda executable: `/home/shardgrid/miniconda3/bin/conda`
- Conda environment: `shardgrid`
- Conda prefix: `/home/shardgrid/miniconda3/envs/shardgrid`
- runtime Python: `/home/shardgrid/miniconda3/envs/shardgrid/bin/python`
- Python version: `Python 3.12.13`
- 7 harmless SSH commands, all through `SSHTransport`

### 4. Documentation

- Updated `docs/operations/remote-access.md` with the GTX 1650 live result and
  the shared-contract statement for both Workers

### Validation run

- `pytest tests/integration/test_ssh_worker_4060.py tests/integration/test_ssh_worker_1650.py --run-integration`: `16 passed`
- `pytest` (default): `185 passed, 27 skipped`
- `ruff check .`: passed
- `mypy`: passed (76 files)

### Outcome

- `T039`: `[X]`, live verified
- GTX 1650 remote access: `PASS`
- both Workers use the same transport contract: `YES`

## T040 (this task batch)

### 1. WSL Runtime command wrapper

- Added `src/shardgrid/transport/runtime.py`
- `WSLRuntimeConfig.from_worker_and_runtime(worker, runtime)`: distro / conda
  env / conda prefix / user all come from project config (no hardcoding)
- `WSLRuntimeWrapper`: builds the full
  `Windows Worker -> WSL distro -> selected Conda env -> command` chain and
  executes it through a host executor (`SSHTransport` for remote Workers)
- Conda selection is prefix-driven (`export PATH=<prefix>/bin:$PATH` +
  `CONDA_DEFAULT_ENV`), never hardcoded `conda activate`; falls back to
  `conda run -n <env>` only when prefix is absent but executable+env are set
- cwd (`cd ... || exit 66`), env vars (`export K=V`, shlex-quoted), and
  `shlex.join` command serialization are all config-driven
- Structured diagnostics via `classify_runtime_failure` -> `FailureRecord`
  (66 = cwd not found, 127 = command not found, else non-zero exit)

### 2. Hardened the shared WSL command wrapper (quoting + exit code)

Two real defects found during live testing on `10.87.5.15` and fixed in
`src/shardgrid/transport/remote_access.py::wrap_wsl_runtime_command`:

- quoting: `$PATH` inside the payload expanded through the
  PowerShell -> wsl.exe -> bash chain, and an expanded
  `Program Files (x86)` broke bash parsing; the payload is now base64-encoded
  and decoded inside WSL (`echo <b64> | base64 -d | /bin/bash`), so quotes,
  `$`, spaces, and parentheses survive every layer
- exit code: `powershell.exe` collapsed non-zero native exit codes to `1`;
  the script now ends with `exit $LASTEXITCODE` so the original remote exit
  code is preserved (verified live: `sys.exit(7)` returns `7`)

### 3. Real-worker runtime sanity check (harmless, no GPU probe)

Live on `10.87.5.15` (GTX 1650 Worker) through SSHTransport + wrapper:

- `python -V` -> `Python 3.12.13`
- `sys.executable` -> `/home/shardgrid/miniconda3/envs/shardgrid/bin/python`
  (inside the selected Conda prefix)
- env var propagation -> `ok`
- non-zero exit code preserved -> `7`

### 4. Config reality update

`examples/workers.yaml` now records the verified runtime facts:
`runtime_distro: Ubuntu` (real distro name, was the wrong `Ubuntu-22.04`),
`conda_environment: shardgrid`, `conda_prefix: /home/shardgrid/miniconda3/envs/shardgrid`.
`tests/unit/test_config.py` assertion updated to match.

### 5. Tests

- `tests/unit/test_wsl_command.py`: 14 tests (payload construction, quoting
  round-trip with spaces/special chars, cwd, env, conda prefix vs conda run,
  remote wrapper encoding, stdout/stderr/exit code, missing distro/env errors,
  failure classification, config fallback)
- T038/T039 live regression after the wrapper hardening: 16 passed
- full suite: 199 passed, 27 skipped; ruff: passed; mypy: passed (78 files)

### Outcome

- `T040`: `[X]`
- Windows -> WSL2 -> Conda runtime wrapper: usable (live verified)
- quoting/cwd/env/exit-code behavior: verified live on the GTX 1650 Worker

## T041

- 实现 GPU Probe（`src/shardgrid/workers/gpu_probe.py`）：复用 T040 runtime wrapper，
  从 selected Conda environment 采集 GPU name / total+free memory / utilization /
  compute capability / driver / CUDA / Torch / NCCL / Gloo，结果写入现有
  WorkerResource / WorkerRuntime（无新模型）
- 修复两个真实传输问题：Windows sshd->cmd.exe 命令行 ~6KB 上限（改用
  `SSHTransport.run(stdin=...)` + wrapper `run_script()`：stdin 直送
  `<prefix>/bin/python -`，绕过长度限制）；torch 可选字段（mem_get_info /
  utilization 依赖 pynvml）独立容错，缺失不拖垮整个 torch 块
- 实机 Live Probe（两台 Worker 均 health=healthy、无 failure）：
  - RTX 4060：NVIDIA GeForce RTX 4060 Laptop GPU，8188/7788 MB，util 10%，
    cc 8.9，driver 566.07，CUDA 11.8，torch 2.7.1+cu118，NCCL/Gloo True
  - GTX 1650：NVIDIA GeForce GTX 1650，4096/3403 MB，util 7%，cc 7.5，
    driver 527.41，CUDA 11.8，torch 2.7.1+cu118，NCCL/Gloo True
- 测试结果：PASS（contract 8/8；全量 209 passed；ruff/mypy 通过；T038/T039 live 回归 16/16）
