# Planning Contract

## Inputs

- `CapturedTrainingContext`
- `ExecutionGraph`
- `OwnershipModel`
- fresh `WorkerResource` / `GPUResource` snapshot
- memory estimate and online calibration evidence
- planning constraints and runtime capabilities

## Outputs

- logical partitions with execution node IDs;
- owned/read state object IDs;
- boundary tensor values and transfer requirements;
- placement plan with worker/rank/GPU assignment;
- memory probe candidate metadata;
- plan/runtime fingerprint.

## Invariants

- Every supported execution node is assigned exactly once.
- Every required parameter and buffer is owned exactly once or blocked as unsupported.
- Every cross-partition value has producer, consumer, shape, dtype, and transfer plan.
- Placement uses fresh healthy reachable workers and current GPU free memory.
- Runtime consumes the selected exact plan; it must not repartition, re-place, or use rank-local round-robin decisions.
- Memory-probe rejection rejects only the candidate, not the whole feature, until candidate search is exhausted.

## Failure Codes

Planning surfaces `PROFILE_FAILURE`, `PARTITION_FAILURE`, `PLAN_VALIDATION_FAILURE`, `NO_FEASIBLE_PLAN`, `SEARCH_BUDGET_LIMIT`, `MEMORY_REJECT`, and `RESOURCE_CHANGED` with stage and retry metadata.
