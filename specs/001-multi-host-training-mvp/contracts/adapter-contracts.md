# Adapter Contracts

Adapters keep ShardGrid orchestration independent from external tools and platform-specific commands.

## PlatformAdapter

Methods:

- `detect() -> PlatformInfo`
- `run(command, timeout, env, cwd, runtime_environment) -> CommandResult`
- `path_join(parts) -> str`
- `validate_manual_action(action) -> ManualAction`
- `bootstrap_step(step) -> BootstrapFinding`

Rules:

- Business modules must not embed PowerShell/Bash-specific command strings directly.
- Business modules must not hard-code `conda activate`, `conda run`, system Python,
  or platform-specific Python paths; the selected runtime environment is resolved by
  configuration plus PlatformAdapter/runtime abstractions.
- Any action requiring administrator permission, reboot, BIOS changes, password entry, or risky firewall change returns `manual_action_required=true`.
- Conda installation or environment creation that requires elevated permissions or
  would overwrite an existing environment returns `manual_action_required=true`.

## ResourceProvider

Methods:

- `probe_worker(worker_config) -> WorkerResource`
- `probe_network(workers) -> NetworkState`
- `probe_gpu(worker_config) -> GPUResource`

Rules:

- Probe failures return structured FailureRecord.
- WorkerResource must include both `physical_os` and `runtime_os`.
- WorkerResource must record the detected Conda manager state, selected Conda
  environment or prefix, Python executable, PyTorch/CUDA runtime versions, and must
  distinguish Windows host state from WSL2 training runtime state.

## ParallelEngine

Methods:

- `compatibility_spike(context) -> CompatibilitySpikeReport`
- `profile(job, workers) -> ProfileResult`
- `candidate_partitions(job, profile) -> list[PartitionCandidate]`
- `plan(job, profile, selected_candidate) -> ParallelPlan`
- `prepare(job_snapshot, execution_plan) -> EnginePreparation`
- `launch_metadata(parallel_plan) -> dict`
- `consolidate_checkpoint(checkpoint, parallel_plan) -> ConsolidatedModelArtifact`

Adapter identity and capability:

- Every adapter exposes `engine_id` and a `candidate` record
  (`ParallelEngineCandidate`) carrying `name`, `version`, `source`,
  `status` (BackendStatus), `capabilities`, `limitations`, and the
  compatibility report path.
- Registered statuses express SUPPORTED (AVAILABLE / EXPERIMENTAL),
  BLOCKED, and NOT_SELECTED without probing adapter internals; selection is
  adapter-driven and never hard-codes a framework name in business logic.

Contract errors:

- A method the concrete engine does not support MUST raise
  `UnsupportedEngineMethodError`; silent fallback is forbidden.

Rules:

- Galvatron must be evaluated before fallback engines.
- Original external plans must be preserved (`ParallelPlan.engine_plan_path`).
- Static validation model plans (explicit parallel configs without
  profiler-driven search) must be labeled as regression fixtures, not automatic
  partition support.
- Automatic partition support is available only when the selected engine provides
  model inspection/profiling, supported partition boundaries, partition
  materialization, and runtime/autograd integration for the submitted model.
- Unsupported dynamic control flow, custom CUDA operations, untraceable graphs,
  engine-incompatible modules, or unsupported tied/shared parameter behavior must
  return BLOCKED or UNSATISFIABLE with explicit diagnostics.

Implemented by: `src/shardgrid/engines/base.py` (`ParallelEngine` protocol,
`EngineRegistry`, `registered_engine_registry`) and `src/shardgrid/engines/
models.py` (`ParallelEngineCandidate`, `ProfileResult`, `EnginePreparation`).
Contract tests: `tests/contract/test_parallel_engine.py`.

## Planner

Methods:

- `evaluate_requirements(job, profile, resources, network) -> PlanningRequirements`
- `search_partition_placement(job, profile, candidates, resources, network, overrides) -> PlacementDecision`
- `create_execution_plan(job, parallel_plan, placement) -> ExecutionPlan`

Rules:

- Hard constraints are evaluated before scoring: Worker health, GPU/runtime
  compatibility, backend availability, network reachability, valid
  physical-host mapping, valid `local_world_size`, supported partition boundary,
  and training peak memory fit.
- Training peak memory includes parameters, activations, gradients, optimizer
  states, runtime overhead, communication buffers, estimated peak memory, and
  configurable memory headroom.
- Candidate partition and placement search is joint: profile -> candidate
  partition -> placement attempt -> memory/network validation -> reject or
  select.
- Legal candidates are optimized in this order: fewest physical Workers, least
  cross-host communication, avoid severe heterogeneous bottlenecks, improve
  compute balance, GPU capability or secondary preferences, deterministic
  tie-break.
- Candidate rejection reason, selected reason, fallback reason, and
  UNSATISFIABLE reason are persisted.
- Manual override is allowed but cannot bypass hard constraints.

## ArtifactStore

Methods:

- `create_snapshot(job) -> JobSnapshot`
- `write_config(snapshot, config)`
- `write_plan(snapshot, execution_plan)`
- `append_log(snapshot, worker_id, rank, data)`
- `record_checkpoint(snapshot, checkpoint_ref)`
- `record_consolidated_model(snapshot, consolidated_model_ref)`
- `record_environment(snapshot, environment)`

Rules:

- Snapshot root must be configurable.
- Accepted job snapshots are immutable except lifecycle append records.
- Environment records must include Conda executable, active environment, prefix,
  Python executable/version, PyTorch version, and CUDA/runtime fields when checked.
- Distributed checkpoints are authoritative for resume and may keep optimizer,
  scheduler, RNG, and runtime state distributed.
- Completed automatic-partition jobs must record a consolidated full-model
  artifact that reloads without the original Worker count, rank mapping, or stage
  count.

## ArtifactTransport

Methods:

- `distribute(snapshot, worker_assignment) -> TransportResult`
- `collect_logs(job_id, worker_assignment) -> TransportResult`
- `collect_checkpoint(job_id, worker_assignment) -> TransportResult`

Rules:

- Stage C concrete implementations use mature file transfer tools.
- No custom file transfer protocol.

## Launcher

Methods:

- `prepare(execution_plan) -> LaunchResult`
- `distribute(execution_plan) -> LaunchResult`
- `launch(execution_plan) -> LaunchResult`
- `monitor(job_id) -> JobStatus`
- `logs(job_id, selector) -> LogResult`
- `stop(job_id) -> StopResult`
- `cleanup(job_id) -> CleanupResult`

Concrete launchers:

- `SSHLauncher`
- `KubernetesLauncher`
- `VolcanoLauncher`

Rules:

- SSHLauncher is the first working backend.
- KubernetesLauncher cannot be promoted until its compatibility gate passes.
- VolcanoLauncher cannot be promoted until Kubernetes GPU training is stable.
- HAMi is not a launcher; it extends resource sharing after Stage E gate passes.
