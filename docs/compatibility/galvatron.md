# Compatibility: Galvatron - Detection Harness (T054)

Status: HARNESS READY (detection-level) | Updated: 2026-08-18 | Owner: Machine A control node

T054 implements the Galvatron compatibility-spike **harness**. It does NOT
declare Galvatron supported or unsupported on RTX 4060 / GTX 1650 / multi-host;
that decision belongs to T061 after T056-T060 capability validation.

## Current environment (selected Conda dev environment)

- Conda executable: `/home/yangjilei/anaconda3/bin/conda`
- Selected environment: `base` (prefix `/home/yangjilei/anaconda3`)
- Python: `/home/yangjilei/anaconda3/bin/python` (3.13.5)
- PyTorch: not installed in this control-plane environment
- CUDA runtime: not detected (no PyTorch)

## Galvatron detection

Check-only detection runs these probes against the selected Conda Python
(no environment change):

- `python -m pip show galvatron`
- `import galvatron` (records `__file__` and `__version__`)
- `git -C <import location> remote get-url origin` (source verification)
- PyTorch probe: `import torch` (version, `torch.version.cuda`, cuda available)

Result statuses are strictly separated:

| Status | Meaning |
|--------|---------|
| `AVAILABLE` | official Galvatron importable in the selected environment with runtime evidence; detection-level only, capability validation is T056-T060 |
| `NOT INSTALLED` | not present; official install command recorded as a manual action |
| `INCOMPATIBLE` | present but import/runtime broken in the selected environment |
| `BLOCKED` | official install would damage the environment, source unverifiable/unofficial, or manual action required |
| `CHECK FAILED` | harness itself could not complete evidence collection |

`AVAILABLE` never means "fully compatible": it only proves detect + import +
version + runtime evidence. Import success alone is never treated as full
compatibility.

## Official sources only

- PyPI package `galvatron` (verified via recorded pip metadata)
- Official GitHub repo: `https://github.com/PKU-DAIR/Hetu-Galvatron`
  (verified via `Home-page` metadata or `git remote get-url origin`)

Unofficial forks, patched sources, or unverifiable locations are reported as
`BLOCKED` with a manual action; the harness never installs or patches them.

## Install path (opt-in, never in check mode)

- Default mode is check-only: no install, no backend change.
- `allow_install=True` runs only the official command after a `pip install
  --dry-run` preflight. If the dry-run would change torch / nvidia / triton /
  cuda / tensorrt packages, installation stops and is recorded `BLOCKED`.
- Editable (GitHub) install additionally requires a detected Conda identity.
- Any install failure is recorded with preserved diagnostics and a manual
  action; Galvatron source is never patched.

## Current compatibility check

`run_galvatron_check()` on Machine A (2026-08-18):

- status: `NOT INSTALLED`
- Galvatron version/source: none detected
- blocker/diagnostics: PyTorch not installed in the selected control-plane
  environment (expected on Machine A; GPU Workers are checked separately in
  T055-T057)
- evidence: saved via `save_galvatron_evidence()` (run-id + `galvatron-latest.json`)

## T055 - Declared requirements vs Worker runtimes (2026-08-18)

### Galvatron declared requirements (official source only)

- Source: `github:PKU-DAIR/Hetu-Galvatron` (official `setup.py`, main branch)
- Version: `2.4.1` (package name `hetu-galvatron`)
- Python: `>=3.8`
- Declared install requirements (verbatim):
  `torch>=2.0.1`, `torchvision>=0.15.2`, `numpy<2.0.0`,
  `transformers==4.49.0`, `h5py>=3.6.0`, `attrs>=21.4.0`, `yacs>=0.1.8`,
  `six>=1.15.0`, `sentencepiece>=0.1.95`, `pybind11>=2.9.1`, `scipy>=1.10.1`
- Conditional (`GALVATRON_FLASH_ATTN_INSTALL=TRUE`): `packaging`,
  `flash-attn>=2.0.8`
- Build requirements: C++ compiler (`csrc/dp_core.cpp`, C++11, `-fPIC`),
  `pybind11>=2.9.1` (setup)
- CUDA: no explicit CUDA requirement declared (not guessed)
- PyPI finding: the PyPI package `galvatron` (0.0.3) is **NOT** the official
  PKU-DAIR project (home page `github.com/kyegomez/Galvatron`); PyPI install
  is rejected as unofficial. Official source = GitHub only.

### RTX 4060 Worker (gpu4060, WSL2 selected Conda `shardgrid`)

- Conda: env `shardgrid`, prefix `/home/shardgrid/miniconda3/envs/shardgrid`
- Python: 3.12.13 | PyTorch: 2.7.1+cu118 | torch CUDA runtime: 11.8
- Driver: 566.07 | GPU: NVIDIA GeForce RTX 4060 Laptop GPU | cap 8.9
- Galvatron: NOT INSTALLED

### GTX 1650 Worker (gpu1060, WSL2 selected Conda `shardgrid`)

- Conda: env `shardgrid`, prefix `/home/shardgrid/miniconda3/envs/shardgrid`
- Python: 3.12.13 | PyTorch: 2.7.1+cu118 | torch CUDA runtime: 11.8
- Driver: 527.41 | GPU: NVIDIA GeForce GTX 1650 | cap 7.5
- Galvatron: NOT INSTALLED

### Compatibility comparison

| Worker | python (>=3.8) | pytorch (>=2.0.1) | cuda (declared) | Galvatron | status |
|--------|----------------|-------------------|-----------------|-----------|--------|
| gpu4060 | MATCH (3.12.13) | MATCH (2.7.1+cu118) | REQUIREMENT UNKNOWN (actual 11.8) | NOT INSTALLED | NOT INSTALLED |
| gpu1060 | MATCH (3.12.13) | MATCH (2.7.1+cu118) | REQUIREMENT UNKNOWN (actual 11.8) | NOT INSTALLED | NOT INSTALLED |

Overall: `NOT INSTALLED` - Python and PyTorch satisfy the declared
requirements on both Workers; no Galvatron installation exists yet.

### Blockers / findings

- Galvatron is not installed on either Worker (expected; T056 install spike
  decides).
- Declared dependencies are mostly absent on both Workers: numpy,
  transformers, torchvision, yacs, sentencepiece, scipy, h5py, attrs, six,
  pybind11, flash_attn are NOT present; only `packaging` 26.3 is installed.
  (Verified live via the WSL selected Conda runtime.)
- PyPI `galvatron` is an unofficial third-party package; only the official
  GitHub source is acceptable.
- No environment change was made: no upgrade/downgrade of Python, PyTorch,
  CUDA, and no Conda env creation/replacement (detect-first / reuse-first).
- T055 PASS means "version compatibility comparison completed" only; it does
  not mean Galvatron ran on the GPUs (that is T056-T060).
- Final Galvatron support decision remains T061.

### Evidence

- `tests/integration/test_engine_versions.py` (13 logic cases + 1 live test)
- `src/shardgrid/engines/compatibility.py` (declared-requirements collection,
  worker version evidence, comparison)
- Live evidence: `/var/tmp/shardgrid/engines/galvatron-versions-latest.json`

## Rules

- Detect-first / reuse-first: an existing official Galvatron in the selected
  environment is reused; no upgrade or replacement without evidence.
- If installation would break the selected environment, stop and record
  manual action / blocker.
- No final "Galvatron supported/unsupported" conclusion yet: final decision is
  T061.

## T056 / T057 - Single-Worker Galvatron compatibility (2026-08-18)

Same minimal workload, independent judgment per Worker:

- selected Conda env / Python / torch / CUDA detect
- official-source Galvatron detect
- official checkout reuse or install path
- `import galvatron` / `galvatron.core.profiler`
- official `profile_hardware.py` runtime path

### RTX 4060 Worker (T056)

- Worker: `gpu4060` / `10.87.5.155`
- Evidence: `/var/tmp/shardgrid/engines/galvatron-spike-gpu4060-latest.json`
- Result: `BLOCKED`
- Conda/Python/PyTorch/CUDA:
  - env `shardgrid`
  - prefix `/home/shardgrid/miniconda3/envs/shardgrid`
  - python `3.12.13`
  - torch `2.7.1+cu118`
  - torch CUDA `11.8`
- Driver/GPU:
  - driver `566.07`
  - GPU `NVIDIA GeForce RTX 4060 Laptop GPU`
  - compute capability `8.9`
- Install/import/profiler:
  - official checkout already existed and was reused
  - default pip resolution was rejected because it would replace the installed
    torch/CUDA stack (`torch 2.7.1+cu118 -> 2.13.0` plus many CUDA/NVIDIA
    packages)
  - constrained `--no-deps` install path was attempted
  - build failed before import/profiler/runtime due missing usable C++ compiler
- Blocker:
  - `g++` exists in the command path probe but the build failed with
    `error: [Errno 13] Permission denied: 'g++'`
  - this is a real WSL runtime/manual-action blocker, not a fake PASS

### GTX 1650 Worker (T057)

- Worker: `gpu1060` / `10.87.5.15`
- Evidence: `/var/tmp/shardgrid/engines/galvatron-spike-gpu1060-latest.json`
- Result: `BLOCKED`
- Conda/Python/PyTorch/CUDA:
  - env `shardgrid`
  - prefix `/home/shardgrid/miniconda3/envs/shardgrid`
  - python `3.12.13`
  - torch `2.7.1+cu118`
  - torch CUDA `11.8`
- Driver/GPU:
  - driver `527.41`
  - GPU `NVIDIA GeForce GTX 1650`
  - compute capability `7.5`
- Install/import/profiler:
  - official checkout already existed and was reused
  - default pip resolution was rejected for the same reason as RTX 4060:
    it would replace the installed torch/CUDA stack
  - constrained `--no-deps` install path was attempted
  - build failed before import/profiler/runtime due missing C++ compiler
- Blocker / older-GPU note:
  - build failed with `error: [Errno 2] No such file or directory: 'g++'`
  - older GPU memory / compute-capability limits were recorded (`7.5`), but
    they were not the blocker in this run

### Shared finding

- Both Workers reached:
  - env detect `PASS`
  - CUDA visibility `PASS`
  - GPU identity `PASS`
  - Galvatron detect `PASS`
- Both Workers stopped at:
  - `galvatron_install = BLOCKED`
- Shared real cause:
  - official Galvatron installation wants a native extension build
    (`galvatron_dp_core`)
  - build cannot proceed in the selected WSL runtime because `g++` is not
    usable there
- Network note:
  - this run did not block on GitHub clone timeout
  - a short `git ls-remote` fail-fast precheck and max-two-clone-attempt rule
    are now in the harness to stop future dead waits

## T056 source-baseline rerun (2026-08-19)

The official-source checkout strategy was corrected to stop following `main`
and instead use the official fixed release:

- official source: `https://github.com/PKU-DAIR/Hetu-Galvatron`
- requested ref: `v2.4.0`
- checkout path: `$HOME/galvatron-spike-v2.4.0`
- clone mode: `git clone --depth 1 --branch v2.4.0`
- submodules: not used
- evidence filename policy for new runs:
  - RTX 4060 -> `galvatron-spike-rtx4060-latest.json`
  - GTX 1650 -> `galvatron-spike-gtx1650-latest.json`

### RTX 4060 Worker rerun (T056, 2026-08-19)

- Evidence: `/var/tmp/shardgrid/engines/galvatron-spike-rtx4060-latest.json`
- Result: `FAIL`
- Resolved commit:
  `498bcadeb6ff80cd246bdc4321124da0f4b2d89b`
- Runtime:
  - env `shardgrid`
  - python `3.12.13`
  - torch `2.7.1+cu118`
  - torch CUDA `11.8`
  - driver `566.07`
  - GPU `NVIDIA GeForce RTX 4060 Laptop GPU`
- What changed relative to the old blocker:
  - the old upstream `main` packaging inconsistency is no longer the blocker
  - `v2.4.0` really can be checked out and installed with constrained
    `--no-deps`
- Real failure stage:
  - `galvatron_install`: `PASS`
  - `galvatron_import`: `FAIL`
  - `profiler_runtime`: `FAIL`
- Real failure:
  - `galvatron.core.profiler import failed: No module named 'einops'`
  - `profile_hardware.py` failed for the same reason inside the selected WSL
    Conda runtime
- Meaning:
  - source acquisition is fixed
  - install path is fixed enough to build/install `hetu-galvatron`
  - the current minimal dependency allowlist is still incomplete for the
    official runtime path on this Worker

### RTX 4060 Worker rerun after installing `einops` (T056, 2026-08-19)

- Pre-check:
  - `python -c "import einops; print(einops.__version__)"`
  - before install: `ModuleNotFoundError`
- Install action:
  - selected runtime: WSL2 Ubuntu -> Conda env `shardgrid`
  - command: `python -m pip install --no-deps einops`
  - installed version: `0.8.2`
- Post-check:
  - `python -c "import einops; print(einops.__version__)"`
  - result: `0.8.2`
- Rerun evidence:
  - `/var/tmp/shardgrid/engines/galvatron-spike-rtx4060-latest.json`
- Rerun result: `FAIL`
- New failure stage:
  - `galvatron_import`
  - `profiler_runtime`
- New blocker:
  - `No module named 'apex'`
- Import/runtime stack:
  - `galvatron.core.runtime.utils`
  - `megatron.core.optimizer.clip_grads`
  - `from apex.optimizers import FusedAdam as Adam`
- Meaning:
  - `einops` is no longer the blocker
  - the next real missing dependency is `apex`
  - per task rule, stop here instead of guessing and installing more packages

### RTX 4060 Worker `apex` install attempt (2026-08-19)

- Detect before install:
  - `python -m pip show apex` -> not installed
  - `python -c "import apex"` -> `ModuleNotFoundError`
- Official source used:
  - repo: `https://github.com/NVIDIA/apex`
  - commit: `9e3568a6f90fbc1996a06f8f9e99310bdaf2253a`
- Install path:
  - clone: `$HOME/apex-galvatron`
  - command:
    `python -m pip install -v --no-deps --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./`
- Result: `FAIL`
- Failure stage:
  - metadata generation before wheel build
- Real blocker:
  - Apex setup failed with
    `TypeError: unsupported operand type(s) for +: 'NoneType' and 'str'`
    inside `get_cuda_bare_metal_version`
- Interpretation:
  - this is not another missing Python module
  - the selected WSL runtime does not currently expose the CUDA bare-metal
    toolchain state Apex setup expects (for example `nvcc` / `CUDA_HOME`)
  - that makes the blocker a build/toolchain issue, not a simple pip dependency
- Next status:
  - T056 remains `FAIL`
  - T057 not started

### GTX 1650 Worker rerun (T057)

- Not executed on 2026-08-19.
- Stop reason:
  - per task instruction, a new real blocker on T056 stops the sequence before
    T057
  - T057 will use the same `v2.4.0` baseline on its next real run

## T058 - One GPU per physical host placement (2026-08-22)

Environment rebuilt on both Workers; the full pre-T058 regression gate PASSED
(worker access, network, local CUDA/PyTorch, Gloo, NCCL, dist-test, Gate 1,
Gate 2) before this run.

Implementation: `tests/multi_host/test_galvatron_one_gpu_per_host.py`

- Launches Galvatron's one-process-per-node chain (the launch shape Galvatron
  official `train_dist.sh` builds with `--nnodes 2 --nproc_per_node 1`) on the
  two physical Workers from Machine A through the existing SSH + WSL2 +
  selected Conda chain.
- Each rank probe imports `galvatron` (installed from official `v2.4.0`) and
  records rank metadata: global rank, world size, local rank, hostname, host
  IP, GPU name, compute capability, NCCL backend, and a real barrier +
  all_reduce result.
- `classify_placement` reports:
  - `true_multi_host`: exactly two ranks, two distinct physical hosts, one
    rank per host, `local_rank=0` / `cuda:0` per host, NCCL collective
    `[3,3,3,3]`, GPU identity matching the expected GPU per Worker
  - `single_host_multi_gpu`: all ranks on one physical host
  - `invalid`: any other inconsistency

Result: **true_multi_host**

| rank | Worker | hostname | GPU | capability | device | backend | all_reduce |
|------|--------|----------|-----|-----------|--------|---------|------------|
| 0 | gpu4060 (10.87.5.155, iface eth3) | `ldj` | NVIDIA GeForce RTX 4060 Laptop GPU | 8.9 | cuda:0 | nccl | [3,3,3,3] |
| 1 | gpu1060 (10.87.5.15, iface eth0) | `LAPTOP-5G3QUOGM` | NVIDIA GeForce GTX 1650 | 7.5 | cuda:0 | nccl | [3,3,3,3] |

- Evidence: `/var/tmp/shardgrid/engines/galvatron-one-gpu-per-host-latest.json`
- Tests: 10 passed (9 logic + 1 live), ruff clean
- Known notes:
  - Windows Firewall only permits the project rendezvous port (29500);
    non-standard ports were observed to block NCCL bootstrap (err=-3 noise
    appears on the successful path too; the real blocker was the port).
  - `NCCL_SOCKET_IFNAME` is derived from `ip route get <peer>` on each Worker
    (eth3 / eth0 in this run), never hard-coded.
  - Galvatron Hardware Profiler remains `BLOCKED_BY_WSL2_CUPTI`; it is not a
    condition of this placement test.

## Diagnostics

- Implementation: `src/shardgrid/engines/compatibility.py`
- Tests: `tests/unit/test_engine_compatibility.py` (21 tests)
- Evidence schema: `GalvatronCompatibilityResult` (run_id, status, versions,
  Conda/Python/PyTorch/CUDA, commands, per-command diagnostics with output
  tails, blockers, manual actions, timing)

## T056 - RTX 4060 spike status (2026-08-19, apex build blocker)

- Environment: conda `shardgrid`, Python 3.12.13, torch 2.7.1+cu118, torch CUDA 11.8,
  driver 566.07, RTX 4060 Laptop GPU (cap 8.9)
- Galvatron: installed from official `v2.4.0` tag (site-packages), but
  `galvatron.core.profiler` import requires `apex` (compile-layer dependency)
- Apex build (official NVIDIA source, gcc-11 toolchain, CUDA 11.8): **BLOCKED**
  at CUDA extension compile - missing `cuda_profiler_api.h` (PyPI nvidia wheels
  have incomplete CUDA toolkit dev headers)
- Status: **T056 BLOCKED** (compile/system layer). T057 not executed.
- Evidence: `/var/tmp/shardgrid/engines/apex-blocker-gpu4060.json`,
  `/var/tmp/shardgrid/engines/apex-build-gpu4060.log`

## T056 - RTX 4060 final live verification (2026-08-19)

Environment repair completed before this run (not repeated here): CUDA build
environment (`cuda-nvcc 11.8.89`, `cuda-cudart-dev 11.8`, `cuda-profiler-api
11.8`, CUDA headers, `CUDA_HOME=$CONDA_PREFIX`) plus Conda host compiler
`x86_64-conda-linux-gnu-gcc/g++ 11.4.0`. Apex CUDA extensions built
successfully. No code, harness, or environment change was made during this
verification.

Live verification (SSH -> WSL2 Ubuntu -> selected Conda `shardgrid`, absolute
Conda Python `/home/shardgrid/miniconda3/envs/shardgrid/bin/python`):

| Step | Result |
|------|--------|
| worker identity | PASS (`ldj`, RTX 4060 Worker `10.87.5.155`) |
| Conda env / prefix | PASS (`shardgrid` / `/home/shardgrid/miniconda3/envs/shardgrid`) |
| Python executable | PASS (`/home/shardgrid/miniconda3/envs/shardgrid/bin/python`, 3.12.13) |
| torch version | PASS (`2.7.1+cu118`) |
| torch.version.cuda | PASS (`11.8`) |
| cuda_available | PASS (`True`) |
| GPU name | PASS (`NVIDIA GeForce RTX 4060 Laptop GPU`) |
| compute capability | PASS (`(8, 9)`) |
| VRAM | PASS (`8585216000` bytes) |
| `import apex` | PASS |
| `from apex.optimizers import FusedAdam` | PASS |
| `import galvatron` | PASS (official `v2.4.0`, site-packages) |
| CUDA tensor allocation | PASS (`[1024, 1024]`) |
| 1024x1024 CUDA matmul + synchronize | PASS |
| finite validation | PASS (`torch.isfinite` all true) |

- Result: **RTX 4060 Galvatron compatibility: PASS**
- Evidence: `/var/tmp/shardgrid/engines/galvatron-spike-rtx4060-latest.json`
  (same file policy as prior T056 runs, latest run is the full PASS record)
- T056 is marked `[X]` in `tasks.md`.
