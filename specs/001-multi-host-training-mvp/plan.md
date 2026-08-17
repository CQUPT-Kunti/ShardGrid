# Implementation Plan: ShardGrid MVP + Platform Foundation

**Branch**: `001-multi-host-training-mvp` | **Date**: 2026-08-15 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `specs/001-multi-host-training-mvp/spec.md`

**Note**: `setup-plan.sh --json` could not run in this Windows host because the local `bash` command attempted to enter WSL and WSL/Hyper-V is not enabled on this machine. The equivalent Spec Kit setup was performed manually: `plan.md` was created from the template, `contracts/` was created, and this active feature remains pinned by `.specify/feature.json`.

## Summary

ShardGrid first-stage implementation builds a single Python-based CLI/control-plane system that proves real distributed training across two separate Windows physical GPU hosts running WSL2 Linux training runtimes. The implementation order is risk-gated: Stage A establishes cross-platform bootstrap, platform abstraction, and doctor; Stage B proves real multi-host GPU communication and two-stage training over SSH; Stage C automates ShardGrid jobs, snapshots, planning, and SSH launch into the formal MVP; Stage D adds Kubernetes and Volcano only after compatibility gates pass; Stage E adds HAMi GPU sharing and multi-user simulation after the Kubernetes/Volcano chain is stable.

The core architectural rule is adapter-first reuse. ShardGrid owns orchestration, resource modeling, placement metadata, artifact snapshots, diagnostics, and launcher contracts. It does not reimplement SSH, file transfer, PyTorch distributed communication, NCCL/Gloo, Kubernetes, Volcano, HAMi, autograd, CUDA kernels, GPU virtualization, or a full model-parallel engine.

## Technical Context

**Language/Version**: Python from the selected Conda-managed development or training environment; no fixed Python or Conda version is required unless a dependency compatibility check proves it. PowerShell 5+/7 is used for Windows bootstrap; Bash is used for Ubuntu/WSL bootstrap.

**Primary Dependencies**: Existing Conda installation and compatible Conda environments are reused first for Python development and training; Typer or Click for CLI; Pydantic for schemas; PyYAML for config; Rich or standard logging for human output; pytest for tests; system OpenSSH/SCP/SFTP/rsync for remote execution and transfer; iperf3/ping/system network tools for probing; PyTorch distributed and torchrun for training runtime inside the selected Conda environment; NCCL first and Gloo fallback; Galvatron compatibility spike first, then DeepSpeed Pipeline, PyTorch pipeline APIs, and nnScaler as ordered alternatives; Kubernetes Python client or kubectl manifests for Stage D; official Volcano and HAMi distributions for later stages.

**Storage**: Stage A-C use local filesystem snapshots on the Ubuntu control node under configurable `jobs/<job-id>/`. Stage D may add NFS through an `ArtifactStore` interface. Future artifact backends may include S3 or MinIO without changing job semantics.

**Testing**: pytest unit and contract tests; local integration tests with mocked transports; hardware smoke tests on each GPU Worker; multi-host communication tests; multi-host training tests; automated SSH end-to-end tests; Kubernetes compatibility tests; Volcano scheduling tests; HAMi GPU-sharing tests; multi-user simulation tests.

**Target Platform**: Ubuntu Linux control/login node; Windows physical GPU Workers; WSL2 Ubuntu Linux training runtimes on Windows GPU Workers; future Kubernetes Linux nodes hosted by WSL2 runtimes if compatibility gates pass.

**Project Type**: Python CLI plus local control-plane library, worker-side scripts, remote launcher adapters, training examples, deployment manifests, and layered test suite.

**Performance Goals**: The supported validation model completes forward, activation transfer, loss, backward, gradient transfer, optimizer step, and checkpoint across RTX 4060 plus GTX 1650 within 15 minutes. Distributed process group initialization succeeds within 2 minutes or fails with diagnostics. Worker inventory succeeds in at least 95% of repeated discovery attempts on a stable LAN.

**Constraints**: One GPU per physical Worker; default `local_world_size = 1`; no hard-coded paths, users, drive letters, IPs, ports, Conda prefixes, environment names, or Python executables; all addresses, paths, and runtime environment selections are configurable or detected; no manual rank launch by the user; no Kubernetes/Volcano/HAMi dependency before the SSH backend proves real training; stop any automated setup step that requires administrator approval, reboot, BIOS changes, password entry, risky firewall changes, Conda installation with elevated permissions, or destructive environment replacement.

**Scale/Scope**: First formal MVP covers Machine A, Machine C, Machine D, and optional Machine E. Stage C supports a small number of configured one-GPU Workers and one active validation job at a time. Stage E expands to multi-job and multi-user simulation, not production multi-tenancy.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

The project constitution file is still the unratified template, so there are no enforceable project-specific constitutional gates. This plan uses the user-provided specification as the active governance source for this feature.

Feature gates derived from the specification:

- **Reuse-first gate**: Pass. All external capabilities are integrated through adapters and mature upstream tools.
- **Conda-first environment gate**: Pass. Python development and training environments are Conda-managed, existing compatible Conda installations/environments are reused, and Conda is not treated as a fixed version requirement.
- **Training-first gate**: Pass. Stage B and Stage C must complete SSH-backed real training before Kubernetes becomes a candidate main path.
- **One-GPU-per-host gate**: Pass. Resource and launch models default to one GPU per physical Worker.
- **Cross-platform gate**: Pass. `physical_os` and `runtime_os` are modeled separately and platform commands are isolated behind adapters.
- **Honest diagnostics gate**: Pass. Fallbacks, partial success, failed compatibility gates, and manual setup blockers must be explicit.
- **No reinvention gate**: Pass. ShardGrid does not implement SSH, file transfer protocols, NCCL/Gloo, Kubernetes, Volcano, HAMi, autograd, CUDA kernels, or GPU virtualization.

Post-design re-check: Pass. Data model and contracts preserve the same gates through explicit states, backend labels, compatibility reports, and failure stages.

## Project Structure

### Documentation (this feature)

```text
specs/001-multi-host-training-mvp/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   ├── cli.md
│   ├── config.schema.yaml
│   ├── execution-plan.schema.yaml
│   ├── job-status.schema.yaml
│   └── adapter-contracts.md
└── tasks.md
```

### Source Code (repository root)

```text
shardgrid/
├── cli/
│   ├── app.py
│   └── commands/
├── control/
│   ├── job_manager.py
│   ├── resource_manager.py
│   └── status_store.py
├── workers/
│   ├── probe.py
│   └── runtime_checks.py
├── resources/
│   ├── models.py
│   └── selectors.py
├── network/
│   ├── probe.py
│   └── diagnostics.py
├── planner/
│   ├── placement.py
│   └── execution_plan.py
├── engines/
│   ├── base.py
│   ├── galvatron.py
│   ├── deepspeed_pipeline.py
│   ├── pytorch_pipeline.py
│   └── nnscaler.py
├── launchers/
│   ├── base.py
│   ├── ssh.py
│   ├── kubernetes.py
│   └── volcano.py
├── artifacts/
│   ├── store.py
│   └── transport.py
├── platforms/
│   ├── base.py
│   ├── linux.py
│   ├── windows.py
│   └── wsl.py
└── common/
    ├── config.py
    ├── errors.py
    ├── logging.py
    └── process.py

scripts/
├── bootstrap-linux.sh
├── bootstrap-windows.ps1
└── bootstrap-wsl.sh

deploy/
├── kubernetes/
├── volcano/
└── hami/

examples/
├── workers.yaml
├── train-minimal.yaml
└── models/

tests/
├── unit/
├── contract/
├── integration/
├── hardware/
├── multi_host/
├── kubernetes/
├── volcano/
└── hami/

docs/
├── compatibility/
├── operations/
└── architecture/
```

**Structure Decision**: Use one Python package with adapters and narrow module boundaries. This keeps the first control plane simple and testable while leaving clear slots for Kubernetes, Volcano, HAMi, future artifact stores, and future parallel engines.

## Risk-Gated Roadmap

### Stage A - Cross-Platform Foundation

**Goal**: Establish repository, platform abstraction, bootstrap scripts, configuration loading, and a trustworthy doctor command.

**Entry Conditions**:

- Active spec and plan are accepted.
- Machine roles are known: A control, C/D GPU Workers, B optional non-core, E future optional Worker.

**Implementation Work**:

- Create repository layout and Python packaging without pinning a Python minor version.
- Detect Conda before environment operations, reuse an existing compatible Conda environment when available, and record the selected environment identity for local quality checks.
- Define `PlatformAdapter` with `LinuxPlatform`, `WindowsPlatform`, and `WSLPlatform`.
- Define command execution result types that always capture stage, host, command, exit code, stdout, stderr, and recommended action.
- Implement bootstrap scripts:
  - `scripts/bootstrap-linux.sh`: Conda detection/reuse, selected Python environment, Git, OpenSSH client/server as needed, iperf3, basic system packages.
  - `scripts/bootstrap-windows.ps1`: OpenSSH, WSL2 detection, Ubuntu distro detection, NVIDIA driver compatibility checks, safe/manual blockers.
  - `scripts/bootstrap-wsl.sh`: Conda detection/reuse inside WSL2, selected Python/PyTorch environment, CUDA visibility, iperf3, runtime dependencies.
- Implement `shardgrid doctor` across Control, Windows Worker, and WSL runtime.
- Implement config loading for workers and train configs with validation.

**Gate A**:

- `shardgrid doctor` reports Machine A, C, and D with clear pass/fail/manual-action results.
- Bootstrap scripts are rerunnable and record versions.
- No business logic contains fixed host paths, usernames, drive letters, or IPs.

### Stage B - Real Multi-Host GPU Training

**Goal**: Prove real multi-host training before any Kubernetes work enters the main path.

**Entry Conditions**:

- Gate A passed on Machine A, C, and D.
- Control can authenticate to GPU Workers.
- WSL runtime can see CUDA on each GPU Worker.
- Each training runtime has a detected Conda executable and either a compatible reused Conda environment or a newly created ShardGrid environment with recorded evidence.

**Implementation Work**:

- Implement `Worker Inventory` from YAML configuration.
- Implement `SSHTransport` using system OpenSSH first, with structured command results.
- Implement Worker Probe over SSH -> WSL for Conda executable, available environments, active environment, Python executable/version, PyTorch, GPU, VRAM, driver, CUDA, NCCL/Gloo, IP, and selected network interface.
- Implement Network Probe using iperf3 and ping/system tools for pairwise latency and bandwidth.
- Implement `shardgrid dist-test`:
  - `world_size = 2`.
  - `local_world_size = 1` on each Worker.
  - Test broadcast, send/recv, and all_reduce.
  - Prefer NCCL.
  - On NCCL failure, preserve `NCCL_DEBUG=INFO`, interface, rendezvous, driver, CUDA, PyTorch, and command logs.
  - Use Gloo only as labeled functional fallback.
- Run Galvatron compatibility spike first.
- If Galvatron fails, run ordered fallback spikes: DeepSpeed Pipeline, PyTorch pipeline APIs, nnScaler.
- Implement the minimal two-stage validation model with a static ExecutionPlan if needed.
- Prove Stage0 on RTX 4060 and Stage1 on GTX 1650 with real forward, activation transfer, loss, backward, gradient transfer, optimizer step, loss decrease, and checkpoint.

**Gate B**:

- Single-GPU smoke test passes on C and D.
- `dist-test` passes with NCCL or fails NCCL honestly and passes Gloo fallback.
- Minimal pipeline training completes across C and D with at least 5% loss decrease and checkpoint.
- The selected parallel engine path has a compatibility report.

### Stage C - ShardGrid Automation

**Goal**: Turn the proven manual/static training path into the formal one-command MVP.

**Entry Conditions**:

- Gate B passed.
- Minimal pipeline training is reproducible with explicit command records.

**Implementation Work**:

- Define serializable entities: `TrainingJob`, `WorkerResource`, `NetworkState`, `ParallelPlan`, `ExecutionPlan`, `JobStatus`, `TrainingResult`, `EnvironmentSnapshot`, and `CompatibilityReport`.
- Implement local file snapshot under configurable `jobs/<job-id>/`.
- Implement `ArtifactStore` and `ArtifactTransport`:
  - Stage C concrete transport: SSH/SCP/SFTP/rsync.
  - Future stores: NFS, S3, MinIO.
- Implement `ResourceManager` and `ClusterState` from live probe results.
- Implement placement-only Planner:
  - memory fit
  - minimum Workers
  - best network
  - better GPU as secondary
  - manual override
- Generate formal `ExecutionPlan` from `ParallelPlan + Placement`.
- Implement `SSHLauncher` lifecycle:
  - prepare
  - distribute
  - launch
  - monitor
  - collect logs
  - status
  - stop
  - cleanup
- Implement final Stage C CLI:
  - `shardgrid workers`
  - `shardgrid probe`
  - `shardgrid network-test`
  - `shardgrid dist-test`
  - `shardgrid train config.yaml`
  - `shardgrid status JOB`
  - `shardgrid logs JOB`
  - `shardgrid stop JOB`

**Gate C - Formal MVP**:

- From Machine A, the user runs only `shardgrid train examples/train-minimal.yaml`.
- System completes discovery, network probe, planning, snapshotting, distribution, rank launch, rendezvous, training, logs, status, and checkpoint.
- User can retrieve final plan, logs, diagnostics, loss trend, optimizer update proof, and checkpoint metadata by job ID.

### Stage D - Kubernetes + Volcano

**Goal**: Add Kubernetes and Volcano as platform backends without breaking the SSH backend.

**Entry Conditions**:

- Gate C passed.
- SSH backend remains the known-good fallback.

**Implementation Work**:

- Define `Launcher` interface if not already final: `prepare`, `distribute`, `launch`, `monitor`, `status`, `logs`, `stop`, `cleanup`.
- Implement Kubernetes compatibility spike:
  - Ubuntu control-plane candidate.
  - WSL2 Linux Worker node join.
  - container runtime.
  - NVIDIA GPU exposure.
  - GPU Pod execution.
  - cross-node Pod networking.
  - PyTorch distributed inside Pods.
- Add `deploy/kubernetes/` scripts, manifests, health checks, and compatibility report templates.
- Implement `KubernetesLauncher` that converts `ExecutionPlan` to Kubernetes workloads.
- Add `ArtifactStore` support for NFS if shared filesystem is needed.
- Install Volcano only after Kubernetes gate is stable.
- Implement `VolcanoLauncher` to generate Volcano Jobs with gang scheduling, queue, priority, and multi-worker configuration.
- Validate Volcano multi-host training on RTX 4060 + GTX 1650.
- Add network-aware scheduling only through supported Volcano/Kubernetes features or ShardGrid placement preferences; do not modify Volcano core.

**Gate D**:

- Kubernetes gate passes or is marked experimental/blocked with report.
- If Kubernetes gate passes, `KubernetesLauncher` can run the same validation training.
- If Volcano gate passes, Volcano launches the same multi-host training as a gang-scheduled job.
- SSH backend remains usable regardless of Kubernetes/Volcano status.

### Stage E - HAMi + Multi-User Resource Sharing

**Goal**: Validate GPU sharing and multi-user simulation after Kubernetes/Volcano are stable.

**Entry Conditions**:

- Gate D Kubernetes and Volcano path is stable for real training.
- SSH backend remains available.

**Implementation Work**:

- Run HAMi compatibility spike:
  - WSL2 Linux Worker nodes.
  - RTX 4060 and GTX 1650.
  - Kubernetes and NVIDIA runtime.
  - memory isolation.
  - compute isolation.
  - multiple Pods sharing one GPU.
  - NCCL/distributed compatibility.
- Install HAMi only if compatible.
- Extend `WorkerResource` for GPU slices:
  - total memory
  - allocated memory
  - free memory
  - memory slices
  - compute slices
  - slice owner/job.
- Extend Planner from GPU placement to GPU fragment placement while delegating enforcement to HAMi.
- Validate two jobs sharing one RTX 4060 where resource limits permit.
- Run multi-user simulation with Job A, Job B, and Job C:
  - queue behavior
  - placement
  - GPU sharing
  - training
  - isolation
  - logs

**Gate E**:

- HAMi compatibility passes on the real environment before GPU sharing is advertised as usable.
- Multi-user simulation completes with clear isolation and logs.
- Any HAMi failure preserves compatibility report and does not affect SSH/Kubernetes/Volcano training paths.

## Testing Strategy

Testing layers must pass in order unless the target layer is explicitly marked experimental:

```text
Unit
-> Local Integration
-> Single GPU Hardware
-> Multi-host Communication
-> Multi-host Training
-> Automated SSH Training
-> Kubernetes
-> Volcano
-> HAMi
-> Multi-user
```

Required test suites:

- **Unit**: schema validation, planner sorting, config validation, command result parsing, snapshot paths, failure state transitions.
- **Contract**: CLI output shape, config schema, ExecutionPlan schema, JobStatus schema, adapter method behavior.
- **Local Integration**: mocked SSH transport, mocked artifact transport, mocked probes, local job lifecycle.
- **Single GPU Hardware**: CUDA/PyTorch smoke on each Worker.
- **Multi-host Communication**: broadcast, send/recv, all_reduce for NCCL and labeled Gloo fallback.
- **Multi-host Training**: two-stage validation model with loss decrease and checkpoint.
- **Automated SSH Training**: full `shardgrid train` from Machine A.
- **Kubernetes**: GPU Pod, multi-node Pod networking, PyTorch distributed.
- **Volcano**: gang scheduling and distributed job startup.
- **HAMi**: GPU sharing, isolation, multi-job behavior.

## Failure Handling

Every operation result must include:

- stage
- host
- worker_id when available
- command or action
- exit_code or structured status
- stdout path or inline summary
- stderr path or inline summary
- recommended_action
- retryability
- manual_action_required flag

Failure stages are:

```text
BOOTSTRAP
PROBE
NETWORK
PROFILE
PLAN
DISTRIBUTE
LAUNCH
RENDEZVOUS
TRAIN
CHECKPOINT
SCHEDULE
GPU_SHARE
STOP
CLEANUP
```

## Dependency Policy

- Safe automatic install: perform the action, verify it, and record version plus command.
- Administrator permission: stop and print the exact command or UI action needed.
- Reboot required: stop, record resume point, and instruct user to reboot.
- BIOS or virtualization setting required: stop and explain the required setting.
- Password entry required: do not script around it; ask the user to configure credentials or run the command.
- Multiple version choices: choose official stable versions compatible with GTX 1650, RTX 4060, WSL2, PyTorch, and selected engine; record the exact resolved versions in environment artifacts.
- Failed install or verification: fail honestly and do not mark the component ready.

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| Multiple launcher backends | SSH must remain first working backend while Kubernetes/Volcano are future platform backends | A single launcher would either block MVP on Kubernetes or make future platform integration invasive |
| Multiple parallel engine adapters | Galvatron compatibility is unknown on RTX 4060 + GTX 1650 + WSL2 + one-GPU-per-host | A single hard-coded engine risks blocking real training if the first candidate fails |
| Platform abstraction layer | Commands span Ubuntu, Windows, and WSL2 with different shells and safety rules | Scattering platform checks would make doctor/bootstrap unsafe and hard to test |
| ArtifactStore and ArtifactTransport interfaces | Stage C starts with local snapshots and SSH transfer, Stage D may need NFS, future stages may need object storage | Direct file copies everywhere would bind job semantics to one transport |
