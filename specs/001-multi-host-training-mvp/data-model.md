# Data Model: ShardGrid MVP + Platform Foundation

**Date**: 2026-08-15
**Feature**: `001-multi-host-training-mvp`

## Shared Enums

### MachineRole

- `control`
- `gpu_worker`
- `client`
- `dev_test`
- `backup_login`

### PhysicalOS

- `linux`
- `windows`

### RuntimeOS

- `linux`
- `wsl2_linux`
- `windows`
- `unknown`

### Health

- `unknown`
- `healthy`
- `degraded`
- `blocked_manual_action`
- `unreachable`
- `failed`

### BackendStatus

- `not_checked`
- `available`
- `failed`
- `fallback_used`
- `experimental`
- `blocked`

### JobState

- `created`
- `probing`
- `planning`
- `snapshotting`
- `distributing`
- `launching`
- `rendezvous`
- `training`
- `checkpointing`
- `completed`
- `failed`
- `stopping`
- `stopped`

### FailureStage

- `BOOTSTRAP`
- `PROBE`
- `NETWORK`
- `PROFILE`
- `PLAN`
- `DISTRIBUTE`
- `LAUNCH`
- `RENDEZVOUS`
- `TRAIN`
- `CHECKPOINT`
- `SCHEDULE`
- `GPU_SHARE`
- `STOP`
- `CLEANUP`

## Entities

### Machine

Represents a physical computer in the ShardGrid environment.

Fields:

- `machine_id`: stable identifier.
- `role`: `MachineRole`.
- `physical_os`: `PhysicalOS`.
- `hostname`: observed hostname.
- `configured_host`: configured address or DNS name.
- `required_for_mvp`: boolean.
- `notes`: optional operator notes.

Relationships:

- One Machine may host zero or one Worker in Stage A-C.
- Machine A hosts the ControlNode.

Validation:

- `machine_id` must be unique.
- Machine C and Machine D must be present for formal Stage C MVP acceptance.

### ControlNode

Represents the Ubuntu login/control node.

Fields:

- `machine_id`
- `hostname`
- `os_version`
- `python_version`
- `ssh_available`
- `git_available`
- `iperf3_available`
- `jobs_root`
- `disk_free_bytes`
- `health`

Relationships:

- Owns TrainingJobs and JobSnapshots.
- Probes Workers.
- Launches ExecutionPlans.

Validation:

- `jobs_root` must be configurable and writable.
- ControlNode must not require a local GPU.

### Worker

Represents a physical GPU host that can participate in training.

Fields:

- `worker_id`
- `machine_id`
- `hostname`
- `physical_os`
- `runtime_os`
- `host`
- `ssh_user_ref`
- `runtime`: e.g. `wsl2`
- `runtime_distro`
- `local_world_size`: default `1`
- `enabled`
- `health`

Relationships:

- Has one WorkerRuntime.
- Has zero or one GPUResource in Stage A-C.
- Can be selected into WorkerAssignment.

Validation:

- `worker_id` must be unique.
- `local_world_size` defaults to `1`.
- Stage A-C Workers must not assume multiple GPUs on the same physical host.

### WorkerRuntime

Represents the execution environment that actually runs probe and training commands.

Fields:

- `worker_id`
- `runtime_os`
- `runtime_version`
- `python_version`
- `torch_version`
- `torch_cuda_version`
- `cuda_available`
- `nccl_available`
- `gloo_available`
- `nvidia_smi_path`
- `path_style`
- `health`

Relationships:

- Belongs to one Worker.
- Produces GPUResource and runtime probe records.

Validation:

- `runtime_os` must be distinct from `physical_os` when the Worker is Windows + WSL2.
- CUDA readiness must be validated inside the runtime, not inferred from Windows alone.

### GPUResource

Represents a GPU visible inside WorkerRuntime.

Fields:

- `worker_id`
- `gpu_index`
- `gpu_name`
- `total_memory_mb`
- `free_memory_mb`
- `utilization_percent`
- `compute_capability`
- `driver_version`
- `cuda_version`
- `health`

Relationships:

- Belongs to WorkerRuntime.
- Used by WorkerAssignment.

Validation:

- Stage A-C supports one GPU per Worker.
- GPU must be visible to the runtime before Worker can be `healthy`.

### WorkerResource

Complete resource record used by ResourceManager and Planner.

Fields:

- `worker_id`
- `hostname`
- `physical_os`
- `runtime_os`
- `ip`
- `gpu_name`
- `gpu_total_memory`
- `gpu_free_memory`
- `gpu_utilization`
- `compute_capability`
- `driver_version`
- `cuda_version`
- `torch_version`
- `torch_cuda_version`
- `nccl_available`
- `gloo_available`
- `network_interface`
- `network_bandwidth`
- `network_latency`
- `health`
- `last_probe_at`

Relationships:

- Derived from Worker, WorkerRuntime, GPUResource, and NetworkState.
- Input to Planner.

Validation:

- Must include both physical and runtime OS.
- Must not be used for placement if health is not eligible.

### NetworkLink

Represents a pairwise network measurement.

Fields:

- `source_worker_id`
- `target_worker_id`
- `source_ip`
- `target_ip`
- `interface`
- `tcp_reachable`
- `latency_ms`
- `bandwidth_mbps`
- `port`
- `measured_at`
- `failure_reason`

Relationships:

- Belongs to NetworkState.

Validation:

- Both directions should be measured or explicitly marked unavailable.
- Planner must treat failed TCP reachability as placement-blocking.

### NetworkState

Collection of relevant Worker-to-Worker links.

Fields:

- `network_id`
- `workers`
- `links`
- `created_at`
- `selected_interfaces`
- `diagnostics_path`

Relationships:

- Input to Planner and ExecutionPlan.

Validation:

- A two-Worker distributed job requires a healthy link between selected Workers.

### ParallelEngineCandidate

Represents an external engine under evaluation.

Fields:

- `engine_id`
- `name`
- `version`
- `source`
- `status`
- `capabilities`
- `limitations`
- `compatibility_report_path`

Relationships:

- May produce a ParallelPlan.
- Decision is recorded in CompatibilitySpikeReport.

Validation:

- Galvatron must be evaluated before fallback engines are selected.

### CompatibilitySpikeReport

Evidence for accepting, rejecting, or marking a backend experimental.

Fields:

- `report_id`
- `component`
- `stage`
- `machines_tested`
- `versions`
- `commands`
- `results`
- `logs_path`
- `status`
- `blockers`
- `decision`
- `recommended_next_action`
- `created_at`

Relationships:

- Attached to ParallelEngineCandidate, LauncherBackend, PlatformAdapter, or TrainingJob.

Validation:

- Any failed gate must have a report before downstream work proceeds.

### ParallelPlan

Plan produced by a mature parallel engine or by the explicit minimal validation model path.

Fields:

- `parallel_plan_id`
- `engine`
- `engine_plan_path`
- `model_name`
- `world_size`
- `stages`
- `requirements`
- `limitations`

Relationships:

- Combined with Placement to create ExecutionPlan.

Validation:

- ShardGrid must preserve original external plan if present.
- Static validation plans must be labeled as not arbitrary-model support.

### TrainingJob

User-submitted training request.

Fields:

- `job_id`
- `config_path`
- `model`
- `requested_world_size`
- `backend_preference`
- `state`
- `created_at`
- `updated_at`
- `snapshot_path`
- `execution_plan_path`
- `status_path`

Relationships:

- Owns JobSnapshot, ExecutionPlan, JobStatus, TrainingResult.

Validation:

- Stage C formal MVP requires `requested_world_size = 2`.
- Job cannot enter `launching` without eligible WorkerResource and NetworkState.

### JobSnapshot

Immutable per-job artifact root.

Fields:

- `job_id`
- `root_path`
- `code_path`
- `config_path`
- `plan_path`
- `logs_path`
- `environment_path`
- `checkpoint_path`
- `diagnostics_path`
- `created_at`

Relationships:

- Owned by TrainingJob.
- Used by ArtifactTransport and LauncherBackend.

Validation:

- Snapshot paths must be under configured jobs root.
- Existing snapshots must not be mutated except by appending logs/status/checkpoint metadata for the same job lifecycle.

### ExecutionPlan

Stable launch plan consumed by launchers.

Fields:

- `job_id`
- `engine`
- `backend`
- `world_size`
- `master.address`
- `master.port`
- `workers`
- `placement_reason`
- `parallel_plan_ref`
- `snapshot_ref`
- `environment`
- `labels`

Relationships:

- Created from ParallelPlan plus Placement.
- Consumed by LauncherBackend.

Validation:

- `world_size` equals number of WorkerAssignments.
- Each Stage A-C WorkerAssignment must have `local_rank = 0`.
- Backend labels must distinguish NCCL success, NCCL failure, and Gloo fallback.

### WorkerAssignment

Mapping of a rank and model stage to a Worker.

Fields:

- `worker_id`
- `rank`
- `local_rank`
- `stage`
- `gpu_index`
- `launch_command`
- `environment`
- `status`
- `pid`
- `log_path`

Relationships:

- Belongs to ExecutionPlan.
- References Worker and GPUResource.

Validation:

- No duplicate rank.
- Stage A-C has one assignment per physical Worker.

### LauncherBackend

Mechanism used to execute an ExecutionPlan.

Fields:

- `backend_id`
- `name`
- `status`
- `compatibility_report_path`
- `supports_prepare`
- `supports_distribute`
- `supports_launch`
- `supports_monitor`
- `supports_stop`
- `supports_cleanup`

Relationships:

- Concrete variants: SSHLauncher, KubernetesLauncher, VolcanoLauncher.

Validation:

- Kubernetes and Volcano backends cannot be `available` without passing their gates.
- SSHLauncher remains available after platform gate failures.

### JobStatus

Current state visible through `shardgrid status`.

Fields:

- `job_id`
- `state`
- `phase`
- `workers`
- `latest_loss`
- `loss_history`
- `backend`
- `fallback_used`
- `started_at`
- `finished_at`
- `failure`
- `checkpoint_ref`

Relationships:

- Belongs to TrainingJob.
- Summarizes TrainingResult and failures.

Validation:

- Failed jobs must include FailureRecord.
- Completed jobs must include checkpoint reference and final metrics.

### FailureRecord

Structured failure detail.

Fields:

- `stage`
- `host`
- `worker_id`
- `command`
- `exit_code`
- `stdout_path`
- `stderr_path`
- `message`
- `recommended_action`
- `retryable`
- `manual_action_required`

Relationships:

- Can belong to JobStatus, BootstrapFinding, CompatibilitySpikeReport, or NetworkLink.

Validation:

- Any operation marked failed must include a stage and recommended action.

### TrainingResult

Final result of a job.

Fields:

- `job_id`
- `forward_success`
- `activation_transfer_success`
- `loss_success`
- `backward_success`
- `gradient_transfer_success`
- `optimizer_step_success`
- `parameters_changed`
- `initial_loss`
- `final_loss`
- `loss_decrease_percent`
- `checkpoint_path`
- `backend_label`
- `diagnostics_path`
- `status`

Relationships:

- Belongs to TrainingJob.

Validation:

- Completed Stage C MVP requires all success booleans true, parameter changes true, loss decrease at or above threshold, and checkpoint path present.

### PlatformAdapter

Encapsulates OS-specific command behavior.

Fields:

- `adapter_id`
- `platform`
- `shell`
- `path_rules`
- `supports_bootstrap`
- `supports_probe`
- `manual_action_rules`

Relationships:

- Used by bootstrap, doctor, probes, and launchers.

Validation:

- Business modules must call PlatformAdapter instead of embedding platform shell commands directly.

### GPUShare

Stage E resource slice for HAMi-backed GPU sharing.

Fields:

- `worker_id`
- `gpu_index`
- `total_memory_mb`
- `allocated_memory_mb`
- `free_memory_mb`
- `memory_slices`
- `compute_slices`
- `owner_job_ids`
- `isolation_status`

Relationships:

- Extends GPUResource after HAMi gate passes.

Validation:

- Must not be created or advertised before HAMi compatibility passes.
