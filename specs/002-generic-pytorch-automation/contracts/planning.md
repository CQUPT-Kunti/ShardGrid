# Planning Contract

## Inputs

- `NonMaterializingCaptureContext`
- `ExecutionGraph`
- `StateObject` / ownership metadata
- `MemoryEstimate`
- fresh `WorkerResource` / `GPUResource` snapshot
- historical calibration records, when applicable
- planning constraints and runtime capabilities

## Outputs

- logical partitions with execution node IDs;
- owned and read-only state object IDs;
- boundary tensor values and transfer requirements;
- placement plan with worker/rank/GPU assignment;
- resource freshness evidence;
- estimated memory evidence;
- artifact/state manifest references;
- plan/runtime fingerprint.

## Invariants

- Every supported execution node is assigned exactly once.
- Every required parameter and buffer is owned exactly once or blocked as unsupported.
- Every cross-partition value has producer, consumer, shape, dtype, and transfer plan.
- Placement uses fresh healthy reachable workers and current GPU total/used/free memory.
- Model memory need is estimated before materialization and formal execution.
- Production admission must not launch a per-job GPU forward/backward/optimizer trial.
- Runtime consumes the selected exact plan; it must not repartition, re-place, or use rank-local round-robin decisions.
- Backend graph artifacts must not carry full real parameter or buffer payloads.
- Worker materialization must be owned-state-only or explicitly read-only-state-only.

## Admission Evidence

Production admission records:

- graph and state fingerprint;
- estimator version and model-memory breakdown;
- historical calibration provenance, if used;
- fresh resource snapshot fingerprint and timestamp;
- selected partition/placement memory margin;
- no per-job GPU trial execution evidence.

## Failure Codes

Planning surfaces precise failures such as:

- `MODEL_MEMORY_ESTIMATE_UNSUPPORTED`
- `PROFILE_FAILURE`
- `PARTITION_FAILURE`
- `PLAN_VALIDATION_FAILURE`
- `NO_FEASIBLE_PLAN`
- `SEARCH_BUDGET_LIMIT`
- `RESOURCE_CHANGED`
- `PER_JOB_GPU_TRIAL_DISABLED`
- `BACKEND_GRAPH_PAYLOAD_UNSAFE`

## Gate Evidence

```text
MODEL_MEMORY_ESTIMATED_BEFORE_MATERIALIZATION=PASS
PLACEMENT_USES_FRESH_FREE_VRAM=PASS
PER_JOB_GPU_TRIAL_PROBE=0
EXACT_PLAN_EXECUTION=PASS
```
