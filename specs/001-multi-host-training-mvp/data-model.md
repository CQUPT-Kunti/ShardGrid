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
- `conda_environment`
- `conda_prefix`
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
- `conda_environment`
- `conda_prefix`
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
- `environment_manager`: default `conda`
- `conda_executable`
- `conda_environment`
- `conda_prefix`
- `conda_active`
- `python_executable`
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
- `environment_manager`: default `conda`
- `conda_environment`
- `conda_prefix`
- `python_executable`
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
- Must include enough Conda/Python identity to reproduce the runtime used for probe or launch.
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

Plan produced by a mature parallel engine or by the explicit minimal validation model regression path.

Fields:

- `parallel_plan_id`
- `engine`
- `engine_plan_path`
- `model_name`
- `world_size`
- `stages`: stage metadata, not just a stage count.
- `communication_edges`
- `memory_estimates`
- `candidate_evaluation_ref`
- `requirements`
- `limitations`

Relationships:

- Combined with Placement to create ExecutionPlan.
- References ModelProfile and original engine output when available.

Validation:

- ShardGrid must preserve original external plan if present.
- Automatic plans must come from selected ParallelEngine-supported model boundaries.
- Static validation plans must be labeled as regression fixtures, not automatic partition support.

### ModelProfile

Selected ParallelEngine output used by the Planner.

Fields:

- `model_profile_id`
- `engine`
- `model_name`
- `model_type`
- `supported`
- `unsupported_reason`
- `module_or_layer_graph`
- `parameter_names`
- `parameter_bytes`
- `buffer_names`
- `supported_partition_boundaries`
- `estimated_compute`
- `activation_estimates`
- `gradient_estimates`
- `required_runtime`
- `required_backend`
- `evidence_paths`

Relationships:

- Produced by ParallelEngine profile.
- Input to PartitionCandidate generation.

Validation:

- Unsupported, untraceable, or engine-incompatible models cannot produce launchable automatic plans.
- Profile records must not require the user to author stage files.

### PartitionCandidate

A possible automatic model split plus its resource implications before final placement.

Fields:

- `candidate_id`
- `model_profile_id`
- `stage_count`
- `stages`
- `communication_edges`
- `estimated_bytes_per_step`
- `required_worker_count`
- `hard_constraint_status`
- `rejection_reason`
- `score_breakdown`

Relationships:

- Generated from ModelProfile.
- Evaluated against WorkerResource and NetworkState.

Validation:

- Rejected candidates must record the reason.
- A candidate cannot be accepted if any stage has unsupported boundary metadata.

### StageMetadata

Metadata for one automatic partition stage.

Fields:

- `stage_id`
- `original_module_or_layer_identity`
- `partition_boundary`
- `parameter_names_or_ranges`
- `parameter_bytes`
- `activation_bytes`
- `gradient_bytes`
- `estimated_compute`
- `estimated_peak_training_memory`
- `required_runtime`
- `required_backend`

Relationships:

- Belongs to ParallelPlan or PartitionCandidate.
- Maps to WorkerAssignment after placement.

Validation:

- Stage metadata must preserve enough original identity to consolidate full model weights.

### TrainingMemoryEstimate

Peak training memory estimate for one stage on one candidate Worker.

Fields:

- `parameter_bytes`
- `activation_bytes`
- `gradient_bytes`
- `optimizer_bytes`
- `runtime_overhead_bytes`
- `communication_buffer_bytes`
- `estimated_peak_training_memory`
- `memory_headroom`
- `usable_gpu_memory_after_headroom`

Relationships:

- Attached to StageMetadata and candidate placement attempts.

Validation:

- `estimated_peak_training_memory` must be less than or equal to usable GPU memory after headroom for a valid candidate.

### CommunicationEdge

Estimated communication between two adjacent or otherwise connected stages.

Fields:

- `source_stage_id`
- `target_stage_id`
- `boundary_tensor_shape`
- `boundary_activation_bytes`
- `boundary_gradient_bytes`
- `microbatch_count`
- `batch_size`
- `sequence_length`
- `estimated_bytes_per_step`
- `bandwidth_mbps`
- `latency_ms`
- `estimated_network_cost`

Relationships:

- Belongs to PartitionCandidate and selected ParallelPlan.
- Evaluated against NetworkState.

Validation:

- Communication cost must be based on actual stage boundaries, not model parameter size alone.

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

### EnvironmentSnapshot

Stable record of the Python/runtime environment used by control checks, Worker
probe, bootstrap, or training launch.

Fields:

- `snapshot_id`
- `scope`: control, worker, job, or rank scope.
- `environment_manager`: default `conda`.
- `conda_executable`
- `conda_environment`
- `conda_prefix`
- `python_executable`
- `python_version`
- `torch_version`
- `torch_cuda_version`
- `cuda_version`
- `components`: additional detected component versions or `not_installed`/`not_checked` states.

Relationships:

- Stored under JobSnapshot `environment_path`.
- Referenced by WorkerRuntime, WorkerResource, ExecutionPlan, and diagnostics.

Validation:

- Detection fields record actual observations, not desired versions.
- Missing or unchecked components must be represented explicitly rather than fabricated.

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
- `model_profile_ref`
- `candidate_evaluation_ref`
- `conda_environment`
- `conda_prefix`
- `python_executable`
- `placement_reason`
- `parallel_plan_ref`
- `original_engine_plan_ref`
- `distributed_checkpoint_ref`
- `consolidated_model_ref`
- `snapshot_ref`
- `environment`
- `labels`

Relationships:

- Created from ParallelPlan plus Placement.
- Consumed by LauncherBackend.
- Auditable through dry-run and replay before launch.

Validation:

- `world_size` equals number of WorkerAssignments.
- Each Stage A-C WorkerAssignment must have `local_rank = 0`.
- Backend labels must distinguish NCCL success, NCCL failure, and Gloo fallback.
- The selected plan must include enough stage and placement metadata for checkpoint consolidation and reload validation.

### WorkerAssignment

Mapping of a rank and model stage to a Worker.

Fields:

- `worker_id`
- `rank`
- `local_rank`
- `stage`
- `gpu_index`
- `conda_environment`
- `conda_prefix`
- `python_executable`
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
- `distributed_checkpoint_ref`
- `consolidated_model_ref`
- `reload_validation_ref`

Relationships:

- Belongs to TrainingJob.
- Summarizes TrainingResult and failures.

Validation:

- Failed jobs must include FailureRecord.
- Completed jobs must include checkpoint reference and final metrics.
- Completed automatic-partition jobs must include consolidated model and reload validation references.

### DistributedCheckpoint

Authoritative resume checkpoint for distributed training.

Fields:

- `checkpoint_id`
- `manifest_path`
- `model_shard_refs`
- `optimizer_shard_refs`
- `scheduler_state_refs`
- `rng_state_refs`
- `runtime_state_refs`
- `partition_metadata_ref`

Relationships:

- Stored under JobSnapshot checkpoint path.
- Source for resume and full-model consolidation.

Validation:

- Resume metadata may remain distributed and must preserve partition metadata.

### ConsolidatedModelArtifact

Full model export created from a completed distributed checkpoint.

Fields:

- `artifact_id`
- `format`
- `path`
- `source_checkpoint_id`
- `original_model_ref`
- `parameter_namespace`
- `state_dict_keys`
- `buffer_keys`
- `shared_parameter_metadata`
- `shape_digest`
- `dtype_digest`
- `reload_validation_ref`

Relationships:

- Produced after training completion.
- Referenced by TrainingResult and JobStatus.

Validation:

- Must load without requiring original Worker count, rank mapping, or stage count.

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
- `runtime_environment`
- `python_executable`
- `conda_environment`
- `conda_prefix`
- `message`
- `recommended_action`
- `retryable`
- `manual_action_required`

Relationships:

- Can belong to JobStatus, BootstrapFinding, CompatibilitySpikeReport, or NetworkLink.

Validation:

- Any operation marked failed must include a stage and recommended action.
- Diagnostics should distinguish system Python from the selected Conda Python when both are visible.

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
- `environment_manager`
- `conda_executable`
- `conda_environment`
- `conda_prefix`
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
