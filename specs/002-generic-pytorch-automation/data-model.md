# Data Model: Generic PyTorch Automation

## TrainingEntrypoint

Fields: entrypoint path, argv, working directory, environment summary, cluster config path, dry-run flag, JSON output flag.

Validation: entrypoint must remain ordinary PyTorch user code and must not require ShardGrid model-provider or sample-input APIs.

Relationships: produces a non-materializing capture attempt.

## NonMaterializingCaptureContext

Fields: model declaration evidence, graph capture backend, input/output structure, tensor metadata, state metadata, lifecycle boundaries, unsupported capability diagnostics, counters proving no full real CPU forward/backward/optimizer mutation before planning.

Validation: must be produced without complete control-plane real model materialization. Any unsafe fallback must become an unsupported failure.

Relationships: input to graph capture, ownership modeling, memory estimation, artifact preparation, and checkpoint validation.

## ExecutionGraph

Fields: graph ID/fingerprint, execution nodes, tensor values, graph inputs, graph outputs, producer/consumer edges, unsupported operations, dynamic-shape guards, capture backend evidence.

Validation: every supported execution node must be assigned exactly once; every boundary value must have producer, consumer, shape, dtype, and transfer metadata.

Relationships: consumed by logical partitioning and runtime.

## StateObject

Fields: canonical state ID, kind (`parameter` or `buffer`), original model-state key, shape, dtype, requires-grad flag, sharing/tied group, owner evidence, use sites, payload reference.

Validation: every required state object must be owned exactly once or rejected. Shared/tied state must have one canonical checkpoint owner.

Relationships: independent from execution nodes; referenced by partitions, worker manifests, and checkpoint shards.

## MemoryEstimate

Fields: partition ID, state bytes, buffer bytes, gradient bytes, optimizer-state bytes, activation bytes, activation-liveness estimate, backward-saved-tensor bytes, temporary/workspace bytes, communication-buffer bytes, dtype/mixed-precision factor, batch/input-shape evidence, shared-state adjustment, safety margin, confidence/provenance.

Validation: produced before materialization and formal execution. Missing required components must be explicit unsupported or conservative assumptions.

Relationships: consumed by placement and admission.

## HistoricalCalibrationRecord

Fields: model family or graph signature, hardware signature, observed memory evidence, calibration timestamp, applicability reason, confidence adjustment.

Validation: may adjust estimates only as offline evidence. It must not trigger a per-job GPU trial run.

Relationships: optional input to `MemoryEstimate`.

## FreshResourceSnapshot

Fields: host ID, reachability, runtime health, GPU ID, GPU health, total memory, used memory, free memory, timestamp, probe provenance.

Validation: must be fresh for every planning/admission decision and must not be replaced by stale totals when current free memory is required.

Relationships: consumed by placement and admission.

## LogicalPartition

Fields: partition ID, execution node IDs, input value IDs, output value IDs, owned state IDs, read-only state IDs, memory estimate, transfer estimate.

Validation: exact node coverage, exact state ownership, valid boundary values, and supported shared/tied state semantics.

Relationships: mapped by `PlacementPlan`; compiled into `RuntimePlan`.

## PlacementPlan

Fields: plan ID, candidate ID, worker ID, rank, GPU ID, resource snapshot fingerprint, estimated memory usage, admission evidence, exact-plan fingerprint.

Validation: selected only from healthy reachable fresh resources and estimate-based feasibility. Production admission must not depend on a per-job GPU trial forward/backward/optimizer step.

Relationships: becomes execution assignments for launchers and runtime.

## BackendGraphArtifact

Fields: graph/code metadata, graph fingerprint, node/value metadata references, state references, schema version, payload-size evidence.

Validation: must not embed complete real parameter or buffer payloads. Size must be bounded by graph/metadata, not total model state.

Relationships: loaded by workers with state manifests.

## StatePayloadManifest

Fields: state object IDs, original keys, shard locations, shape/dtype evidence, ownership, checksum/fingerprint, transfer metadata.

Validation: a worker receives only the payloads it owns or explicitly reads. Full-model CPU load before pruning is invalid.

Relationships: consumed by runtime materialization and checkpoint finalization.

## WorkerOwnershipPlan

Fields: worker ID, rank, GPU ID, logical partition IDs, owned state IDs, read-only state IDs, inbound/outbound runtime edges, artifact references.

Validation: must match planner fingerprint. Runtime must not repartition or re-place.

Relationships: consumed by distributed runtime and shard save.

## CheckpointShard

Fields: schema version, graph fingerprint, plan ID, step, worker ID, rank, GPU ID, owned partitions, state entries, original keys, canonical IDs, shapes, dtypes, payload references.

Validation: schema compatible; no duplicate canonical IDs or keys; no missing expected state; shape/dtype/fingerprint/plan evidence matches.

Relationships: input to memory-safe finalization.

## MergedModelState

Fields: output path or stream manifest, state_dict key/tensor mapping, source shard IDs, validation report, memory budget evidence.

Validation: standard model-state compatibility is preserved; large-state finalization must not require full in-memory control-plane assembly.

Relationships: final artifact for successful supported training.

## CompatibilityReport

Fields: gate name, command, environment, discovered resources, placement summary, memory evidence, optimizer progress, checkpoint validation, cleanup evidence, pass/fail/blocked status, failure code.

Validation: hardware stress must prove GPU packing and not CPU serialization capacity.

Relationships: final evidence for Feature 002 acceptance.

## FailureCode

Fields: code, broad stage, producer, retryable flag, user message, rank/worker/GPU context, log/artifact references.

Validation: failures must not collapse distinct capture, estimation, resource, artifact, runtime, checkpoint, or stress problems into one generic code.

Relationships: stored in failure records and consumed by CLI/stress reports.

## State Transitions

```text
TrainingEntrypoint
  -> CaptureAttempt
  -> NonMaterializingCaptureContext
  -> GraphCaptured
  -> StateMetadataValidated
  -> MemoryEstimated
  -> FreshResourcesDiscovered
  -> Partitioned
  -> Placed
  -> ArtifactsPrepared
  -> OwnedStateMaterialized
  -> FormalTraining
  -> ShardCheckpointed
  -> MemorySafeFinalized
  -> StandardStateValidated
```

Failure may stop the flow at capture, graph, estimation, ownership, partition, placement, resource refresh, artifact preparation, launch, runtime, checkpoint, or stress stages before unsafe mutation whenever possible.
