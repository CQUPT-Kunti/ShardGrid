# Adapter Contracts

Adapters keep ShardGrid orchestration independent from external tools and platform-specific commands.

## PlatformAdapter

Methods:

- `detect() -> PlatformInfo`
- `run(command, timeout, env, cwd) -> CommandResult`
- `path_join(parts) -> str`
- `validate_manual_action(action) -> ManualAction`
- `bootstrap_step(step) -> BootstrapFinding`

Rules:

- Business modules must not embed PowerShell/Bash-specific command strings directly.
- Any action requiring administrator permission, reboot, BIOS changes, password entry, or risky firewall change returns `manual_action_required=true`.

## ResourceProvider

Methods:

- `probe_worker(worker_config) -> WorkerResource`
- `probe_network(workers) -> NetworkState`
- `probe_gpu(worker_config) -> GPUResource`

Rules:

- Probe failures return structured FailureRecord.
- WorkerResource must include both `physical_os` and `runtime_os`.

## ParallelEngine

Methods:

- `compatibility_spike(context) -> CompatibilitySpikeReport`
- `profile(job, workers) -> ProfileResult`
- `plan(job, resources, network) -> ParallelPlan`
- `prepare(job_snapshot, execution_plan) -> EnginePreparation`
- `launch_metadata(parallel_plan) -> dict`

Rules:

- Galvatron must be evaluated before fallback engines.
- Original external plans must be preserved.
- Static validation model plans must be labeled as limited support.

## Planner

Methods:

- `select_workers(job, resources, network, overrides) -> Placement`
- `create_execution_plan(job, parallel_plan, placement) -> ExecutionPlan`

Rules:

- Memory fit is mandatory.
- Network reachability is mandatory.
- Minimum Workers, better network, and better GPU are ordered preferences.
- Manual override is allowed but cannot bypass hard health or reachability failures.

## ArtifactStore

Methods:

- `create_snapshot(job) -> JobSnapshot`
- `write_config(snapshot, config)`
- `write_plan(snapshot, execution_plan)`
- `append_log(snapshot, worker_id, rank, data)`
- `record_checkpoint(snapshot, checkpoint_ref)`
- `record_environment(snapshot, environment)`

Rules:

- Snapshot root must be configurable.
- Accepted job snapshots are immutable except lifecycle append records.

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
