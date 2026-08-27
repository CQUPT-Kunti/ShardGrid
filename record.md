# T073 Backward Acceptance

Date: 2026-08-25

- Implemented real two-host backward on top of the accepted T072 forward path in `examples/models/train_pipeline.py`.
- Added `tests/multi_host/test_backward_transfer.py` for T073 parsing and opt-in real multi-host verification.
- Local validation passed:
  - `python -m py_compile examples/models/train_pipeline.py examples/models/stage1.py examples/models/stage0.py tests/multi_host/test_backward_transfer.py`
  - `ruff check examples/models/train_pipeline.py examples/models/stage1.py examples/models/stage0.py tests/multi_host/test_backward_transfer.py`
  - `PYTHONPATH=$PWD:$PWD/tests python -m pytest -q tests/unit/test_model_stages.py tests/multi_host/test_backward_transfer.py -k 'not live'`
  - `PYTHONPATH=$PWD:$PWD/tests python -m pytest -q tests/multi_host/test_activation_transfer.py -k 'parse_forward_evidence or not live'`
- Real hardware acceptance passed once on:
  - rank0 / LDJ / RTX 4060 / `10.87.5.155`
  - rank1 / LAPTOP-5G3QUOGM / GTX 1650 / `10.87.5.15`
- Preflight confirmed the dynamically resolved NCCL path interfaces still had MTU 1500:
  - rank0 route to rank1 via `eth3`
  - rank1 route to rank0 via `eth1`
- Acceptance artifact:
  - `/home/yangjilei/Code/ShardGrid/t073_backward_acceptance_20260825_195043.zip`
- Acceptance result:
  - Stage0 forward PASS
  - Cross-host activation PASS
  - Stage1 forward PASS
  - Stage1 backward PASS
  - Activation gradient PASS
  - Gradient return PASS
  - Stage0 backward PASS
  - Stage0 gradients PASS
  - Stage1 gradients PASS

# T074 Optimizer + Loss Decrease + Checkpoint Acceptance

Date: 2026-08-25

- Implemented the T074 training increment in `examples/models/train_pipeline.py`
  (`SHARDGRID_PIPELINE_TASK=t074`) on top of the accepted T072/T073 forward and
  backward communication path: per-iteration `zero_grad -> forward -> backward
  -> optimizer.step` on both stages, a recorded loss history, per-parameter
  checksums, and a per-rank checkpoint save/load verification.
- rank0 optimizer manages only Stage0 parameters; rank1 optimizer manages only
  Stage1 parameters. No rank loads the full model. Route and gradient direction
  continue to come from the static ParallelPlan (`producer_rank`/`consumer_rank`).
- Added `tests/multi_host/test_optimizer_checkpoint.py` for T074 evidence parsing
  and opt-in real multi-host verification.
- Local validation passed:
  - `python -m py_compile examples/models/train_pipeline.py examples/models/stage1.py examples/models/stage0.py tests/multi_host/test_optimizer_checkpoint.py`
  - `ruff check examples/models/train_pipeline.py tests/multi_host/test_optimizer_checkpoint.py`
  - T074 non-live parse tests: `pytest -q tests/multi_host/test_optimizer_checkpoint.py --run-multi-host -k 'not live'` -> 3 passed
  - T071/T072/T073 non-live regression: `pytest -q tests/multi_host/test_stage_placement.py tests/multi_host/test_activation_transfer.py tests/multi_host/test_backward_transfer.py --run-multi-host -k 'not live'` -> passed
  - Local CPU sanity: same loop on CPU -> loss 7.266 -> 0.361, param/opt/step restore all True
- Real hardware acceptance passed once on:
  - rank0 / LDJ / RTX 4060 / `10.87.5.155`
  - rank1 / LAPTOP-5G3QUOGM / GTX 1650 / `10.87.5.15`
- Preflight confirmed the dynamically resolved NCCL path interfaces had MTU 1500:
  - rank0 route to rank1 via `eth3`
  - rank1 route to rank0 via `eth1`
- Training result (20 iterations, deterministic input, AdamW lr=1e-3):
  - initial loss: 7.26569128036499
  - final loss: 0.3605937063694
  - loss decrease: PASS (final < initial), loss finite for all iterations
  - Stage0 parameter update: PASS (checksum changed on rank0)
  - Stage1 parameter update: PASS (checksum changed on rank1)
  - optimizer.step on rank0 and rank1: PASS
- Checkpoint result (per-rank shards, no full-model aggregation):
  - rank0 checkpoint `/tmp/t074/checkpoint/checkpoint_rank0.pt` save + load + verify PASS
  - rank1 checkpoint `/tmp/t074/checkpoint/checkpoint_rank1.pt` save + load + verify PASS
  - parameter restore PASS, optimizer state restore PASS, step/iteration restore PASS on both ranks
- T073 forward/backward path regression after the T074 change: `pytest -q tests/multi_host/test_backward_transfer.py --run-multi-host -k 'live'` -> 1 passed (real two-host backward + gradient return still PASS)
- Evidence artifact:
  - `/var/tmp/shardgrid/engines/optimizer-checkpoint-latest.json`

# T075 Gate 3 Acceptance

Date: 2026-08-27

- Reused the accepted T074 two-host training path as the formal Gate 3 acceptance path through `tests/multi_host/test_training_gate.py`; no new training implementation was introduced.
- Added Gate 3 evidence output at `/var/tmp/shardgrid/engines/training-gate-latest.json`.
- Local validation passed:
  - `python -m py_compile tests/multi_host/test_optimizer_checkpoint.py tests/multi_host/test_training_gate.py examples/models/train_pipeline.py`
  - `ruff check tests/multi_host/test_optimizer_checkpoint.py tests/multi_host/test_training_gate.py examples/models/train_pipeline.py docs/compatibility/training-mvp.md`
  - `PYTHONPATH=$PWD:$PWD/tests python -m pytest -q tests/multi_host/test_training_gate.py tests/multi_host/test_optimizer_checkpoint.py --run-multi-host -k 'not live'`
- Gate 3 preflight on the real workers:
  - rank0 route to rank1 stayed on `eth3`, MTU `1500 -> 1500`
  - rank1 route to rank0 resolved dynamically to `eth0`, MTU `2800 -> 1500`
  - both workers reported `torch 2.7.1+cu118`, CUDA runtime `11.8`, NCCL runtime `2.21.5`
  - no stale ShardGrid training process remained before launch
- Real hardware Gate 3 acceptance passed on:
  - rank0 / LDJ / RTX 4060 / `10.87.5.155`
  - rank1 / LAPTOP-5G3QUOGM / GTX 1650 / `10.87.5.15`
- Training result:
  - iterations: 20
  - initial loss: `7.26569128036499`
  - final loss: `0.3605937063694`
  - loss history finite for all iterations
  - loss decrease: PASS
- Training path result:
  - forward: PASS
  - backward: PASS
  - gradient return: PASS
  - rank0 parameter update: PASS
  - rank1 parameter update: PASS
  - rank0 last marker: `SHUTDOWN_END`
  - rank1 last marker: `SHUTDOWN_END`
- Checkpoint result:
  - rank0 checkpoint roundtrip: PASS
  - rank1 checkpoint roundtrip: PASS
  - optimizer state restore: PASS
  - step / iteration restore: PASS
- Gate 3 result:
  - T075: PASS
  - Gate 3 training loop: PASS

# T076 Doctor Readiness Report

Date: 2026-08-27

- Implemented `shardgrid doctor --target control|workers|all --fix` as one
  authoritative readiness report in [`src/shardgrid/control/doctor.py`](/home/yangjilei/Code/ShardGrid/src/shardgrid/control/doctor.py)
  and [`src/shardgrid/cli/commands/doctor.py`](/home/yangjilei/Code/ShardGrid/src/shardgrid/cli/commands/doctor.py).
- Worker reports now separate `windows_host` and `wsl_runtime` checks, and the
  JSON output includes per-check `subject`, `host`, `runtime`, `layer`,
  `status`, `detected_value`, `expected_value`, `failure_reason`, and
  `recommended_action`.
- Reused the existing MTU logic in
  [`scripts/bootstrap-wsl.sh`](/home/yangjilei/Code/ShardGrid/scripts/bootstrap-wsl.sh)
  and added an MTU-only safe-fix entrypoint so `doctor --fix` does not create or
  replace Conda environments and does not attempt unrelated bootstrap actions.
- Added unit coverage in
  [`tests/unit/test_doctor_cli.py`](/home/yangjilei/Code/ShardGrid/tests/unit/test_doctor_cli.py),
  [`tests/unit/test_control_doctor.py`](/home/yangjilei/Code/ShardGrid/tests/unit/test_control_doctor.py),
  and [`tests/unit/test_doctor_workers.py`](/home/yangjilei/Code/ShardGrid/tests/unit/test_doctor_workers.py)
  for `control|workers|all`, dynamic interface detection, unsafe MTU
  reporting, and safe `--fix` reuse. Local T076 unit validation passed:
  `python -m pytest -q tests/unit/test_control_doctor.py tests/unit/test_doctor_cli.py tests/unit/test_doctor_workers.py tests/unit/test_network_mtu.py`
  -> `18 passed`.
- Real control-node doctor evidence on 2026-08-27:
  `PYTHONPATH=src python -m shardgrid.cli.app --config /tmp/t076-workers.yaml doctor --target control`
  -> PASS after creating configured jobs root `/var/tmp/shardgrid/jobs`.
- Real worker doctor evidence on 2026-08-27:
  `PYTHONPATH=src python -m shardgrid.cli.app --config /tmp/t076-workers.yaml doctor --target workers`
  and `--json` both produced structured reports for:
  - `gpu4060` / `LDJ` / WSL `Ubuntu-22.04`
  - `gpu1060` / `LAPTOP-5G3QUOGM` / WSL `Ubuntu-22.04`
- Real runtime facts from those worker reports:
  - selected Conda env on both workers: `shardgrid`
  - Python on both workers: `/home/shardgrid/miniconda3/envs/shardgrid/bin/python`
    (`Python 3.12.13`)
  - PyTorch on both workers: `2.7.1+cu118`
  - CUDA runtime on both workers: `11.8`
  - NCCL on both workers: `2.21.5`
  - Gloo on both workers: available
  - NCCL library path on both workers: `/usr/lib/x86_64-linux-gnu/libnccl.so`
  - RTX 4060 worker GPU: `NVIDIA GeForce RTX 4060 Laptop GPU`, CC `8.9`,
    driver `566.07`
  - GTX 1650 worker GPU: `NVIDIA GeForce GTX 1650`, CC `7.5`, driver `527.41`
- Real dynamic peer-route and MTU evidence on 2026-08-27:
  - `gpu4060 -> 10.87.5.15`: actual interface `eth3`, MTU `1500`, status `PASS`
  - `gpu1060 -> 10.87.5.155`: actual interface `eth0`, MTU `1500`, status `PASS`
- Real `doctor --target all` and `doctor --target workers --fix` both completed
  without attempting unsafe environment mutation. Current live reports remain
  `degraded` because `iperf3` is not installed inside either worker WSL runtime;
  doctor reports that as a manual operator action instead of a false PASS.

# T077 Worker Inventory

Date: 2026-08-27

- Implemented `shardgrid workers`, `shardgrid workers --refresh`, and
  `shardgrid workers --refresh --require-healthy` in
  [`src/shardgrid/cli/commands/workers.py`](/home/yangjilei/Code/ShardGrid/src/shardgrid/cli/commands/workers.py)
  and registered the real command from
  [`src/shardgrid/cli/app.py`](/home/yangjilei/Code/ShardGrid/src/shardgrid/cli/app.py).
- The command reuses the existing worker inventory model, remote access check,
  WSL runtime wrapper, and GPU probe. It does not introduce a second doctor or
  a second GPU/Conda probe path.
- Inventory semantics:
  - `workers` shows configured workers plus cached live probe data when present.
  - missing or expired probe data is labeled `STALE`, never `HEALTHY`.
  - `--refresh` updates the cached `WorkerResource` and `WorkerRuntime`.
  - `--require-healthy` preserves full output and only changes the exit code.
- Added CLI/integration coverage in
  [`tests/integration/test_workers_cli.py`](/home/yangjilei/Code/ShardGrid/tests/integration/test_workers_cli.py)
  for:
  - full inventory rendering with one unhealthy worker still visible
  - stale cache semantics when no live refresh exists
  - non-zero `--require-healthy` with full worker output retained
- T077 local validation on 2026-08-27:
  - `python -m pytest -q tests/unit/test_cli_registration.py tests/unit/test_inventory.py tests/unit/test_worker_probe.py tests/contract/test_gpu_probe.py tests/integration/test_workers_cli.py --run-integration`
    -> `28 passed`
  - `python -m py_compile src/shardgrid/cli/app.py src/shardgrid/cli/commands/__init__.py src/shardgrid/cli/commands/workers.py tests/unit/test_cli_registration.py tests/integration/test_workers_cli.py tests/contract/test_gpu_probe.py`
    -> PASS
  - `ruff check src/shardgrid/cli/app.py src/shardgrid/cli/commands/__init__.py src/shardgrid/cli/commands/workers.py tests/unit/test_cli_registration.py tests/integration/test_workers_cli.py tests/contract/test_gpu_probe.py`
    -> PASS
- Real inventory evidence on 2026-08-27 using `/tmp/t077-workers.yaml`:
  - `PYTHONPATH=src python -m shardgrid.cli.app --config /tmp/t077-workers.yaml workers`
    -> inventory count `2`; both workers shown as `STALE` before first refresh,
    with explicit reason `no live probe has been recorded for this worker yet`
  - `PYTHONPATH=src python -m shardgrid.cli.app --config /tmp/t077-workers.yaml workers --refresh`
    -> inventory count `2`; healthy worker count `2`
  - `PYTHONPATH=src python -m shardgrid.cli.app --config /tmp/t077-workers.yaml workers --refresh --require-healthy`
    -> full worker output preserved; exit code `0`
- Real worker identification from the refreshed inventory:
  - `gpu4060` -> Windows host `ldj`, physical OS `windows`, runtime OS `wsl2_linux`,
    Conda env `shardgrid`, prefix `/home/shardgrid/miniconda3/envs/shardgrid`,
    Python `3.12.13`, GPU `NVIDIA GeForce RTX 4060 Laptop GPU`, memory `8188 MiB`,
    CC `8.9`, CUDA `11.8`, PyTorch `2.7.1+cu118`, backends `nccl,gloo`, health `HEALTHY`
  - `gpu1060` -> Windows host `LAPTOP-5G3QUOGM`, physical OS `windows`, runtime OS `wsl2_linux`,
    Conda env `shardgrid`, prefix `/home/shardgrid/miniconda3/envs/shardgrid`,
    Python `3.12.13`, GPU `NVIDIA GeForce GTX 1650`, memory `4096 MiB`,
    CC `7.5`, CUDA `11.8`, PyTorch `2.7.1+cu118`, backends `nccl,gloo`, health `HEALTHY`
- Real refreshed JSON output includes structured `worker`, `resource`, `runtime`,
  `host_identity`, `runtime_identity`, `health`, `reachability`, `eligible`,
  `stale`, `reason`, and `failure` fields for both workers.

# T078 Worker Probe CLI

Date: 2026-08-27

- Implemented `shardgrid probe` and `shardgrid probe --worker <WORKER_ID>` in
  [`src/shardgrid/cli/commands/probe.py`](/home/yangjilei/Code/ShardGrid/src/shardgrid/cli/commands/probe.py)
  and registered the real command from
  [`src/shardgrid/cli/app.py`](/home/yangjilei/Code/ShardGrid/src/shardgrid/cli/app.py).
- The command reuses the existing remote access check, WSL runtime wrapper, and
  GPU probe; it does not reimplement doctor readiness, workers inventory,
  bootstrap, or planner logic.
- Structured probe output now includes:
  - worker identity and reachability
  - runtime/Conda/Python identity
  - `WorkerResource`
  - GPU memory / capability / driver
  - CUDA availability and runtime version
  - PyTorch version
  - NCCL/Gloo capability and NCCL version
  - `FailureRecord` when probe fails
- T078 local validation on 2026-08-27:
  - `python -m pytest -q tests/unit/test_cli_registration.py tests/unit/test_worker_probe.py tests/contract/test_gpu_probe.py tests/integration/test_probe_cli.py --run-integration`
    -> `24 passed`
  - `python -m py_compile src/shardgrid/cli/app.py src/shardgrid/cli/commands/__init__.py src/shardgrid/cli/commands/probe.py tests/unit/test_cli_registration.py tests/integration/test_probe_cli.py`
    -> PASS
  - `ruff check src/shardgrid/cli/app.py src/shardgrid/cli/commands/__init__.py src/shardgrid/cli/commands/probe.py tests/unit/test_cli_registration.py tests/integration/test_probe_cli.py`
    -> PASS
- Real probe evidence on 2026-08-27 using `/tmp/t077-workers.yaml`:
  - `PYTHONPATH=src python -m shardgrid.cli.app --config /tmp/t077-workers.yaml probe`
    -> worker count `2`, both workers `PASS`
  - `PYTHONPATH=src python -m shardgrid.cli.app --json --config /tmp/t077-workers.yaml probe --worker gpu4060`
    -> worker count `1`, selected only `gpu4060`
  - `PYTHONPATH=src python -m shardgrid.cli.app --json --config /tmp/t077-workers.yaml probe --worker gpu1060`
    -> worker count `1`, selected only `gpu1060`
- Real worker probe identification:
  - `gpu4060` -> Windows host `ldj`, runtime `Ubuntu-22.04`, Conda env `shardgrid`,
    prefix `/home/shardgrid/miniconda3/envs/shardgrid`, Python `3.12.13`,
    GPU `NVIDIA GeForce RTX 4060 Laptop GPU`, GPU count `1`, selected GPU `0`,
    memory `8188 MiB`, CC `8.9`, driver `566.07`, CUDA `11.8`,
    PyTorch `2.7.1+cu118`, NCCL `2.21.5`, Gloo available
  - `gpu1060` -> Windows host `LAPTOP-5G3QUOGM`, runtime `Ubuntu-22.04`,
    Conda env `shardgrid`, prefix `/home/shardgrid/miniconda3/envs/shardgrid`,
    Python `3.12.13`, GPU `NVIDIA GeForce GTX 1650`, GPU count `1`,
    selected GPU `0`, memory `4096 MiB`, CC `7.5`, driver `527.41`,
    CUDA `11.8`, PyTorch `2.7.1+cu118`, NCCL `2.21.5`, Gloo available
- Probe failure handling validation:
  - [`tests/integration/test_probe_cli.py`](/home/yangjilei/Code/ShardGrid/tests/integration/test_probe_cli.py)
    covers a failed probe report with `FailureRecord.stage == PROBE`
  - the same test confirms failure output is explicit and structured instead of
    silently retaining a healthy result

# T079 doctor --fix bootstrap runner

Date: 2026-08-27

- Connected `doctor --fix` to the existing bootstrap runner in
  [`src/shardgrid/bootstrap/runner.py`](/home/yangjilei/Code/ShardGrid/src/shardgrid/bootstrap/runner.py)
  and kept the diff small: fix runs now require a successful verification pass
  before they count as automated success, and known manual-action blockers stop
  before retrying.
- Updated
  [`src/shardgrid/control/doctor.py`](/home/yangjilei/Code/ShardGrid/src/shardgrid/control/doctor.py)
  to preserve worker bootstrap execution metadata instead of flattening it to a
  payload too early, so `doctor --fix` can distinguish:
  - safe no-op / verified healthy
  - verified warning/failure
  - blocked manual action
  - unverified fix result
- Conda reuse remains reuse-first:
  - existing compatible environments stay selected
  - no destructive replacement path was added
  - manual creation/install blockers still stop with explicit action text
- T079 regression coverage added in
  [`tests/integration/test_safe_bootstrap.py`](/home/yangjilei/Code/ShardGrid/tests/integration/test_safe_bootstrap.py)
  and
  [`tests/unit/test_doctor_workers.py`](/home/yangjilei/Code/ShardGrid/tests/unit/test_doctor_workers.py)
  for:
  - already-healthy no-op
  - verification-required fix success
  - verification-missing fix failure
  - blocked admin/root/password/firewall/reboot-style actions
  - worker MTU fix recheck behavior
  - clearing stale manual actions after a successful fix
- Local validation on 2026-08-27:
  - `python -m py_compile src/shardgrid/bootstrap/runner.py src/shardgrid/control/doctor.py src/shardgrid/cli/commands/doctor.py tests/integration/test_safe_bootstrap.py tests/unit/test_doctor_workers.py tests/unit/test_control_doctor.py tests/unit/test_doctor_cli.py`
    -> PASS
  - `ruff check src/shardgrid/bootstrap/runner.py src/shardgrid/control/doctor.py src/shardgrid/cli/commands/doctor.py tests/integration/test_safe_bootstrap.py tests/unit/test_doctor_workers.py tests/unit/test_control_doctor.py tests/unit/test_doctor_cli.py`
    -> PASS
  - `PYTHONPATH=$PWD/src:$PWD/tests python -m pytest -q tests/integration/test_safe_bootstrap.py tests/unit/test_doctor_workers.py tests/unit/test_control_doctor.py tests/unit/test_doctor_cli.py`
    -> `13 passed, 8 skipped`
- Real short validation on 2026-08-27:
  - `PYTHONPATH=src python -m shardgrid.cli.app --config examples/workers.yaml doctor --target control --fix --json`
    -> control `healthy`, bootstrap runner `skipped`, verified `true`
  - worker validation used a temporary real-IP config and temporary password-fed
    SSH wrapper outside the repo; no credentials were written into project files
  - two consecutive real worker runs passed with `exit_code=0` and `health=healthy`
    for both `gpu4060` and `gpu1060`
  - `nccl_path_mtu` stayed `PASS` on both workers, with dynamic interfaces
    resolved as `eth3` for `10.87.5.155 -> 10.87.5.15` and `eth0` for
    `10.87.5.15 -> 10.87.5.155`

# T080 manual-action and redaction safety tests

Date: 2026-08-27

- Added focused T080 unit coverage in
  [`tests/unit/test_bootstrap_manual_actions.py`](/home/yangjilei/Code/ShardGrid/tests/unit/test_bootstrap_manual_actions.py)
  and
  [`tests/unit/test_redaction.py`](/home/yangjilei/Code/ShardGrid/tests/unit/test_redaction.py).
- Manual-action coverage now explicitly checks that these blockers stop automation
  instead of becoming healthy or fixed:
  - administrator/root privilege
  - reboot
  - BIOS/virtualization change
  - password input
  - risky firewall change
  - Conda installation
  - destructive Conda environment replacement
- Minimal redaction fix applied in:
  - [`src/shardgrid/common/process.py`](/home/yangjilei/Code/ShardGrid/src/shardgrid/common/process.py)
  - [`src/shardgrid/common/errors.py`](/home/yangjilei/Code/ShardGrid/src/shardgrid/common/errors.py)
  - [`src/shardgrid/common/logging.py`](/home/yangjilei/Code/ShardGrid/src/shardgrid/common/logging.py)
- The fix keeps diagnostics usable while redacting only secret substrings:
  - commands still show tool/action shape
  - paths keep non-secret suffixes
  - `FailureRecord.message`
  - `FailureRecord.recommended_action`
  - `FailureRecord.command`
  - runtime environment values
  - JSON log payload failure fields
  - human-readable failure diagnostics
- T080 validation on 2026-08-27:
  - `python -m py_compile src/shardgrid/common/process.py src/shardgrid/common/errors.py src/shardgrid/common/logging.py tests/unit/test_bootstrap_manual_actions.py tests/unit/test_redaction.py tests/unit/test_manual_actions.py tests/unit/test_process.py tests/unit/test_errors_logging.py tests/integration/test_safe_bootstrap.py`
    -> PASS
  - `ruff check src/shardgrid/common/process.py src/shardgrid/common/errors.py src/shardgrid/common/logging.py tests/unit/test_bootstrap_manual_actions.py tests/unit/test_redaction.py tests/unit/test_manual_actions.py tests/unit/test_process.py tests/unit/test_errors_logging.py tests/integration/test_safe_bootstrap.py`
    -> PASS
  - `PYTHONPATH=$PWD/src:$PWD/tests python -m pytest -q tests/unit/test_bootstrap_manual_actions.py tests/unit/test_redaction.py tests/unit/test_manual_actions.py tests/unit/test_process.py tests/unit/test_errors_logging.py`
    -> `37 passed`
  - `PYTHONPATH=$PWD/src:$PWD/tests python -m pytest -q tests/integration/test_safe_bootstrap.py --run-integration`
    -> `8 passed`
  - `PYTHONPATH=$PWD/src:$PWD/tests python -m pytest -q tests/unit/test_doctor_workers.py tests/unit/test_doctor_cli.py tests/unit/test_control_doctor.py`
    -> covered in the broader T079/T080 regression pass with no failures

# T081 real Doctor hardware acceptance

Date: 2026-08-27

- Added live Doctor acceptance coverage in
  [`tests/hardware/test_doctor_4060.py`](/home/yangjilei/Code/ShardGrid/tests/hardware/test_doctor_4060.py)
  and
  [`tests/hardware/test_doctor_1650.py`](/home/yangjilei/Code/ShardGrid/tests/hardware/test_doctor_1650.py),
  with shared runtime helpers in
  [`tests/hardware/doctor_hardware_acceptance.py`](/home/yangjilei/Code/ShardGrid/tests/hardware/doctor_hardware_acceptance.py).
- The hardware acceptance path reuses the existing worker Doctor workflow and
  cross-checks it with independent commands for:
  - Windows hostname
  - WSL `nvidia-smi`
  - WSL Conda Python / Torch / CUDA / NCCL
- Real hardware results on 2026-08-27:
  - `gpu4060` / `10.87.5.155` / `LDJ` -> `NVIDIA GeForce RTX 4060 Laptop GPU`
  - `gpu1060` / `10.87.5.15` / `LAPTOP-5G3QUOGM` -> actual GPU `NVIDIA GeForce GTX 1650`
  - both worker Doctor workflows returned `health=healthy`
  - Doctor GPU identity matched independent `nvidia-smi`
  - Doctor Torch/CUDA/NCCL fields matched independent runtime Python output
  - Doctor MTU checks remained `PASS` at `1500`
- Added
  [`docs/operations/doctor-report.md`](/home/yangjilei/Code/ShardGrid/docs/operations/doctor-report.md)
  with the real commands, versions, GPU identity cross-check, worker-id
  compatibility note, and unavailable-capability statement.
- Validation on 2026-08-27:
  - `python -m py_compile tests/hardware/doctor_hardware_acceptance.py tests/hardware/test_doctor_4060.py tests/hardware/test_doctor_1650.py`
    -> PASS
  - `ruff check tests/hardware/doctor_hardware_acceptance.py tests/hardware/test_doctor_4060.py tests/hardware/test_doctor_1650.py`
    -> PASS
  - `PYTHONPATH=$PWD/src:$PWD/tests SHARDGRID_ENABLE_HARDWARE_TESTS=1 python -m pytest -q tests/hardware/test_doctor_4060.py tests/hardware/test_doctor_1650.py --run-hardware`
    -> `2 passed`

# T082 unified compatibility report writer

Date: 2026-08-27

- Added
  [`src/shardgrid/control/compatibility_reports.py`](/home/yangjilei/Code/ShardGrid/src/shardgrid/control/compatibility_reports.py)
  as the shared compatibility report builder / validator / writer / loader.
- Reused the existing
  [`CompatibilitySpikeReport`](/home/yangjilei/Code/ShardGrid/src/shardgrid/engines/models.py)
  contract and
  [`BackendStatus`](/home/yangjilei/Code/ShardGrid/src/shardgrid/common/enums.py)
  states instead of introducing a second report model.
- Supported shared component reporting for:
  - hardware
  - engine
  - network
  - kubernetes
  - volcano
  - hami
- Supported shared status semantics for:
  - PASS -> `available`
  - FAIL -> `failed`
  - BLOCKED -> `blocked`
  - FALLBACK -> `fallback_used`
  - EXPERIMENTAL -> `experimental`
  - unchecked / unknown -> `not_checked`
- Added contract coverage in
  [`tests/contract/test_compatibility_report.py`](/home/yangjilei/Code/ShardGrid/tests/contract/test_compatibility_report.py)
  for:
  - PASS report round-trip
  - FAIL report with required blocker / next-action / evidence fields
  - BLOCKED report validation
  - FALLBACK report separation from preferred-path PASS
  - EXPERIMENTAL report using the same contract
  - unknown/unchecked not upgrading to PASS
  - secret redaction in commands and evidence references
  - wrapping an existing `CompatibilitySpikeReport`
- Validation on 2026-08-27:
  - `python -m py_compile src/shardgrid/control/compatibility_reports.py tests/contract/test_compatibility_report.py`
    -> PASS
  - `ruff check src/shardgrid/control/compatibility_reports.py tests/contract/test_compatibility_report.py`
    -> PASS
  - `PYTHONPATH=$PWD/src:$PWD/tests python -m pytest -q tests/contract/test_compatibility_report.py tests/contract/test_parallel_engine.py tests/unit/test_common_types.py tests/unit/test_doctor_workers.py`
    -> `46 passed`
