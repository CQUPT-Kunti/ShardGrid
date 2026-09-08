# Failure Contract

## Record Shape

Structured failures extend the existing broad stage model with:

- `code`;
- `stage`;
- `producer`;
- `retryable`;
- `message`;
- `rank` / `worker_id` / `gpu_id` when applicable;
- resource snapshot references;
- memory estimate references;
- log and artifact references.

## Codes

Capture:

- `MODEL_CAPTURE_UNSUPPORTED`
- `GRAPH_BREAK_UNSUPPORTED`
- `CUSTOM_OP_UNSUPPORTED`
- `DYNAMIC_CONTROL_FLOW_UNSUPPORTED`
- `CONTROL_PLANE_MODEL_MATERIALIZATION_BLOCKED`
- `CPU_CAPTURE_EXECUTION_BLOCKED`

Estimation / planning:

- `MODEL_MEMORY_ESTIMATE_UNSUPPORTED`
- `PROFILE_FAILURE`
- `PARTITION_FAILURE`
- `PLAN_VALIDATION_FAILURE`
- `NO_FEASIBLE_PLAN`
- `SEARCH_BUDGET_LIMIT`
- `PER_JOB_GPU_TRIAL_DISABLED`

Resource / launch / runtime:

- `RESOURCE_CHANGED`
- `NETWORK_FAILURE`
- `RENDEZVOUS_FAILURE`
- `PROCESS_LAUNCH_FAILURE`
- `INFRA_FAILURE`
- `RUNTIME_FAILURE`
- `FORMAL_TRAINING_OOM`

Artifact / worker / checkpoint:

- `BACKEND_GRAPH_PAYLOAD_UNSAFE`
- `WORKER_FULL_STATE_LOAD_BLOCKED`
- `CHECKPOINT_STREAMING_UNSUPPORTED`
- `CHECKPOINT_VALIDATION_FAILURE`

Stress:

- `CPU_PROCESS_SATURATION`
- `GPU_MEMORY_SATURATION`
- `TEST_LIMIT_REACHED`
- `SATURATION_NOT_PROVEN`
- `GPU_PACKING_STRESS_BLOCKED`

## Retry Policy

- Retryable after resource change: `RESOURCE_CHANGED`, transient `NETWORK_FAILURE`, transient `RENDEZVOUS_FAILURE`, transient `PROCESS_LAUNCH_FAILURE`, and infrastructure failures with stable plan state.
- Not retryable without model/system change: capture unsupported, graph unsupported, custom op unsupported, dynamic control flow unsupported, control-plane materialization blocked, CPU capture execution blocked, unsupported memory estimate, partition invariant failures, plan validation failures, backend payload unsafe, worker full-state load blocked, checkpoint streaming unsupported, checkpoint validation failure, formal training OOM, and stress proof failure.
- Per-job GPU trial execution is not a retry path for production admission.

## Consumers

- CLI displays broad stage plus precise code and artifact/log/resource references.
- `JobManager` stores the code in failure records.
- Planner and estimator store exact invariant failures instead of collapsing them into a generic no-plan result.
- Launchers attach rank/worker/GPU/log context.
- Stress runner classifies CPU bottlenecks separately from GPU packing evidence and treats no progress as stall evidence.
