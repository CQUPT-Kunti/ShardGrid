# Data Model: Generic PyTorch Automation

## TrainingEntrypoint

Fields: entrypoint path, argv, working directory, environment, cluster config path, dry-run flag, JSON output flag.

Validation: entrypoint must be runnable in the selected Conda/runtime layer. ShardGrid must not require model-provider hooks.

Relationships: produces `CapturedTrainingContext`.

## CapturedTrainingContext

Fields: model object identity, model class/module path evidence, first batch object shape, positional args, keyword args, tensor metadata, optimizer class, optimizer param groups, loss/backward boundary, scheduler/callback evidence, checkpoint intent.

Validation: must be captured before distributed parameter mutation. Unsupported or hidden mutation returns capture failure.

Relationships: input to graph capture, profile, runtime input contract, and strict checkpoint validation.

## ExecutionGraph

Fields: graph ID/fingerprint, execution nodes, tensor values, graph inputs, graph outputs, producer/consumer edges, unsupported ops, dynamic-shape guards, source capture backend.

Validation: every supported execution node must be assigned exactly once; every boundary value must have producer/consumer/shape/dtype metadata.

Relationships: consumed by logical partitioning and runtime.

## StateObject

Fields: canonical state ID, kind (`parameter` or `buffer`), original `state_dict` key, shape, dtype, requires_grad, storage/shared-group ID, owner evidence, use sites.

Validation: every required parameter/buffer must be owned exactly once or rejected; shared/tied state must have one canonical checkpoint owner.

Relationships: independent of execution nodes; linked to graph uses and checkpoint shards.

## OwnershipModel

Fields: state objects, owner partition, read partitions, write/update policy, unsupported reasons.

Validation: one checkpoint owner per state object; no ambiguous cross-partition writers.

Relationships: feeds `LogicalPartition`, `WorkerOwnershipPlan`, and checkpoint merge.

## LogicalPartition

Fields: partition ID, execution node IDs, input value IDs, output value IDs, owned state IDs, read-only state IDs, memory estimate, transfer estimate.

Validation: node coverage exactly once; cross-partition values represented; state ownership complete.

Relationships: placed by `PlacementPlan`; compiled into `RuntimePlan`.

## PlacementPlan

Fields: plan ID, candidate ID, worker ID, rank, GPU ID, resource snapshot fingerprint, memory probe status, network evidence.

Validation: only fresh healthy reachable workers; selected GPU free memory and probe evidence must satisfy the candidate.

Relationships: becomes `ExecutionPlan` assignments for `SSHLauncher`.

## WorkerOwnershipPlan

Fields: worker/rank/GPU, logical partition IDs, owned parameter IDs, owned buffer IDs, inbound/outbound runtime edges.

Validation: must match planner fingerprint. Runtime cannot repartition or re-place.

Relationships: consumed by distributed runtime and shard save.

## CheckpointShard

Fields: schema version, graph fingerprint, plan ID, step, worker ID, rank, GPU ID, owned partitions, parameter entries, buffer entries, original keys, canonical IDs, shape, dtype.

Validation: schema compatible; no duplicate canonical IDs or keys; no missing expected model state; shapes/dtypes match captured original state.

Relationships: merged into `MergedModelState`.

## MergedModelState

Fields: output path, state_dict key/tensor mapping, source shard IDs, validation report.

Validation: `original_model.load_state_dict(state, strict=True)` must pass for supported runs.

Relationships: final artifact for successful training.

## FailureCode

Fields: code, broad stage, producer module/function, retry class, user message, log references.

Validation: do not collapse capture/graph/partition/placement/memory/launch/runtime/checkpoint failures into a single generic code.

Relationships: stored in `FailureRecord`, surfaced by CLI and stress runner.

## State Transitions

```text
TrainingEntrypoint
  -> Capturing
  -> CapturedTrainingContext
  -> GraphCaptured
  -> OwnershipValidated
  -> Partitioned
  -> Placed
  -> MemoryProbed
  -> Launched
  -> Training
  -> ShardCheckpointed
  -> Merged
  -> StrictReloadValidated
```

Failure may stop the flow at capture, graph, profile, ownership, partition, placement, memory, launch, runtime, or checkpoint stages before unsafe mutation whenever possible.
