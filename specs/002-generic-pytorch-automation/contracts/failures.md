# Failure Contract

## Record Shape

Structured failures extend the existing broad stage model with:

- `code`;
- `stage`;
- `producer`;
- `retryable`;
- `message`;
- `rank` / `worker_id` / `gpu_id` when applicable;
- log and artifact references.

## Codes

Capture: `MODEL_CAPTURE_UNSUPPORTED`, `GRAPH_BREAK_UNSUPPORTED`, `CUSTOM_OP_UNSUPPORTED`, `DYNAMIC_CONTROL_FLOW_UNSUPPORTED`

Planning: `PROFILE_FAILURE`, `PARTITION_FAILURE`, `PLAN_VALIDATION_FAILURE`, `NO_FEASIBLE_PLAN`, `SEARCH_BUDGET_LIMIT`

Memory/resource: `MEMORY_REJECT`, `FORMAL_TRAINING_OOM`, `RESOURCE_CHANGED`

Launch/runtime: `NETWORK_FAILURE`, `RENDEZVOUS_FAILURE`, `PROCESS_LAUNCH_FAILURE`, `INFRA_FAILURE`, `RUNTIME_FAILURE`

Stress: `CPU_PROCESS_SATURATION`, `GPU_MEMORY_SATURATION`, `TEST_LIMIT_REACHED`, `SATURATION_NOT_PROVEN`

## Retry Policy

- Retryable: `MEMORY_REJECT` for next candidate, `RESOURCE_CHANGED`, transient `NETWORK_FAILURE`, transient `RENDEZVOUS_FAILURE`, transient `PROCESS_LAUNCH_FAILURE`, and infrastructure failures with stable plan state.
- Not retryable without user/model change: capture unsupported, graph unsupported, custom op unsupported, dynamic control flow unsupported, partition invariant failures, plan validation failures, formal training OOM, checkpoint safety failures.

## Consumers

- CLI displays broad stage plus precise code and artifact/log references.
- `JobManager` stores the code in failure records.
- `SSHLauncher` attaches rank/worker/log context.
- Stress runner classifies expected saturation separately from bugs and treats no progress (`after_steps == before_steps`) as stall evidence.
