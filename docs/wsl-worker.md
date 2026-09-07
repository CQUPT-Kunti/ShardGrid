# WSL2 Ubuntu Training Runtime Bootstrap

`scripts/bootstrap-wsl.sh` implements T030 for the Ubuntu runtime inside a
Windows GPU Worker. It is detect-first, rerunnable, and treats the selected
WSL2 Conda environment as the only valid training Python source.

## Usage

Run from inside the Ubuntu WSL2 runtime:

```bash
scripts/bootstrap-wsl.sh
scripts/bootstrap-wsl.sh --check
scripts/bootstrap-wsl.sh --check --json
scripts/bootstrap-wsl.sh --check --findings-dir ~/.shardgrid/bootstrap
scripts/bootstrap-wsl.sh --install-deps
```

Exit codes:

- `0`: healthy
- `1`: degraded or unexpected detection failure
- `2`: blocked by manual action

Findings are written to `wsl-latest.json` and a timestamped JSON file under the
selected findings directory.

## What It Checks

- WSL distro identity and kernel
- Conda executable, known environments, active environment, and selected environment
- Selected training Python, plus system Python for observation only
- ShardGrid Python dependencies in the selected Conda environment
- PyTorch version and `torch.cuda.is_available()` from the selected Conda environment
- CUDA driver-layer visibility through `nvidia-smi`
- `iperf3`, `git`, `lsb_release`, and basic runtime metadata
- optional NCCL path MTU validation/configuration when `SHARDGRID_NCCL_PEER_IP`
  is provided

## Runtime Boundary

The selected training Python must come from the selected WSL2 Conda
environment. Windows-host Python or Conda is out of scope for training runtime
selection and is never treated as a valid training interpreter here.

The script reuses any existing compatible WSL2 Conda environment. If none is
compatible, it creates or repairs a dedicated `shardgrid` environment with:

- Conda-managed Python 3.12 for current PyTorch GPU-wheel compatibility
- `PyYAML`
- `torch==2.7.1+cu118` from the official PyTorch `cu118` index

That explicit torch target exists for a concrete compatibility reason: the
observed default PyPI resolution on Python 3.12 and 3.14 was `2.13.0+cu130`,
which was not CUDA-ready on the GTX 1650 Worker with NVIDIA driver `527.41`.

This script never installs a Linux NVIDIA display driver inside WSL. GPU/CUDA
readiness is judged from the real WSL2 runtime state that already exists.

## Manual Actions

The script stops with a manual action when it finds any of these:

- Conda is missing inside Ubuntu WSL2
- No compatible Conda training environment is available and creation or repair fails
- `torch.cuda.is_available()` cannot be validated from the selected Conda environment
- `nvidia-smi` is not visible inside WSL2
- `iperf3` is missing

The script never overwrites or deletes an existing Conda environment.

## WSL2 NCCL MTU

T072 real two-host acceptance confirmed a WSL2 MTU/PMTU mismatch root cause
for cross-host NCCL hangs. ShardGrid now uses these defaults for WSL2 NCCL
paths:

- `SHARDGRID_NCCL_MTU=1500`
- target PMTU: `1500`
- no fixed TCP MSS override

The bootstrap script can validate or configure the live NCCL path interface
dynamically from the peer IP:

```bash
SHARDGRID_NCCL_PEER_IP=10.87.5.15 scripts/bootstrap-wsl.sh --check
SHARDGRID_NCCL_PEER_IP=10.87.5.15 sudo -E scripts/bootstrap-wsl.sh
SHARDGRID_NCCL_PEER_IP=10.87.5.15 SHARDGRID_WSL_PERSIST_NCCL_MTU=1 sudo -E scripts/bootstrap-wsl.sh
```

Behavior:

- resolves the real egress interface with `ip route get <peer_ip>`
- never hard-codes `eth0` / `eth1` / `eth3`
- reads the current interface MTU and compares it to the expected `1500`
- probes the DF boundary:
  - `1472` should pass
  - `1473` should not pass on a 1500 path
- fails honestly when route parsing fails, MTU is unsafe, or root permission is
  required and unavailable

Persistence support:

- when `SHARDGRID_WSL_PERSIST_NCCL_MTU=1` is set and the script already runs as
  root, it backs up `/etc/wsl.conf` to `/etc/wsl.conf.bak.t072_mtu`
- it only writes a `[boot] command=...` when that can be done safely without
  overwriting an existing `command=`
- the persisted boot command still resolves the interface dynamically from the
  peer IP; it does not hard-code an `ethX` name

## Verification Status

Real WSL2 execution was performed on August 17, 2026 on both GPU Workers:

### `10.87.5.15` (`LAPTOP-5G3QUOGM`, GTX 1650)

Observed final healthy runtime:

- WSL distro: `Ubuntu`
- Conda executable: `/home/shardgrid/miniconda3/bin/conda`
- Selected Conda environment: `shardgrid`
- Selected prefix: `/home/shardgrid/miniconda3/envs/shardgrid`
- Python: `/home/shardgrid/miniconda3/envs/shardgrid/bin/python` (`Python 3.12.13`)
- PyTorch: `2.7.1+cu118`
- CUDA visibility: `torch.cuda.is_available() == true`
- CUDA version: `11.8`
- `nvidia-smi`: `/usr/lib/wsl/lib/nvidia-smi`
- GPU summary: `NVIDIA GeForce GTX 1650, 527.41`
- `iperf3`: `iperf 3.20 (cJSON 1.7.15)`

Run history:

- Initial `run` on August 17, 2026 created `shardgrid` but truthfully remained
  blocked because the Worker was resolving an incompatible `2.13.0+cu130`
  runtime or downloading the correct CUDA 11.8 wheel too slowly from the
  official PyTorch index.
- The prepared Worker was then rerun with a verified `shardgrid` WSL2 Conda
  environment staged at the same prefix.
- Two subsequent `--check` runs were both `healthy`.

Check result 1:

- mode: `check`
- health: `healthy`
- selected environment: `shardgrid`
- python: `Python 3.12.13`
- torch: `2.7.1+cu118`
- cuda_available: `true`
- `nvidia-smi`: `NVIDIA GeForce GTX 1650, 527.41`
- `iperf3`: `iperf 3.20 (cJSON 1.7.15)`

Check result 2:

- mode: `check`
- health: `healthy`
- selected environment: `shardgrid`
- python: `Python 3.12.13`
- torch: `2.7.1+cu118`
- cuda_available: `true`
- `nvidia-smi`: `NVIDIA GeForce GTX 1650, 527.41`
- `iperf3`: `iperf 3.20 (cJSON 1.7.15)`

### `10.87.5.155` (`LDJ`, RTX 4060 Laptop GPU)

Observed healthy runtime:

- WSL distro: `Ubuntu`
- Conda executable: `/home/shardgrid/miniconda3/bin/conda`
- Selected Conda environment: `shardgrid`
- Selected prefix: `/home/shardgrid/miniconda3/envs/shardgrid`
- Python: `/home/shardgrid/miniconda3/envs/shardgrid/bin/python` (`Python 3.12.13`)
- PyTorch: `2.7.1+cu118`
- CUDA visibility: `torch.cuda.is_available() == true`
- CUDA version: `11.8`
- `nvidia-smi`: `/usr/lib/wsl/lib/nvidia-smi`
- GPU summary: `NVIDIA GeForce RTX 4060 Laptop GPU, 566.07`
- `iperf3`: `iperf 3.20 (cJSON 1.7.15)`

Run history:

- `run` on August 17, 2026 completed `healthy`
- `--check` run 1 completed `healthy`
- `--check` run 2 completed `healthy`

Check result 1:

- mode: `check`
- health: `healthy`
- selected environment: `shardgrid`
- python: `Python 3.12.13`
- torch: `2.7.1+cu118`
- cuda_available: `true`
- `nvidia-smi`: `NVIDIA GeForce RTX 4060 Laptop GPU, 566.07`
- `iperf3`: `iperf 3.20 (cJSON 1.7.15)`

Check result 2:

- mode: `check`
- health: `healthy`
- selected environment: `shardgrid`
- python: `Python 3.12.13`
- torch: `2.7.1+cu118`
- cuda_available: `true`
- `nvidia-smi`: `NVIDIA GeForce RTX 4060 Laptop GPU, 566.07`
- `iperf3`: `iperf 3.20 (cJSON 1.7.15)`

## Outcome

`T030` is complete.

- Real WSL2 runtime verification: complete on `10.87.5.15` and `10.87.5.155`
- Manual actions remaining for these two prepared GPU Workers: none
- `10.87.5.228` was not used for T030 GPU-runtime acceptance because it is not a
  GPU training Worker
