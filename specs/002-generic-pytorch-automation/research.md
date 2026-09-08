# Research: Generic PyTorch Automation

## Decision: Plan From Non-Materializing Metadata

**Decision**: ShardGrid must obtain graph, tensor, dtype, shape, state, and lifecycle metadata without requiring the control plane to construct the full real model state before planning.

**Rationale**: The T066 attempt showed that a parameter-heavy ordinary entrypoint can fail in CPU serialization and artifact transfer before GPU packing is tested. The product target includes models larger than control-plane RAM, so full control-plane materialization is disallowed.

**Alternatives considered**:

- Execute the user script normally and partition afterward: rejected because ordinary `HugeModel()` allocates full CPU parameters first.
- Ask users to provide ShardGrid model factories or sample builders: rejected because it changes the ordinary PyTorch interface.
- Fall back to full materialization when capture is hard: rejected because unsupported models must fail closed.

## Decision: No Real CPU Forward/Backward Before Planning

**Decision**: Capture and dry-run planning must not require a full real CPU forward, backward, or optimizer step.

**Rationale**: Real CPU capture execution is unsafe for large models and may perform training side effects before ShardGrid proves partition, placement, memory, and checkpoint safety.

**Alternatives considered**:

- Treat CPU dry-run as harmless: rejected because dry-run still consumes RAM and can trigger side effects.
- Record after an actual first training step: rejected because planning must precede mutation.

## Decision: Fresh GPU Resources Stay, Per-Job GPU Trial Admission Goes

**Decision**: Production admission uses conservative model-memory estimates plus fresh host/GPU resource discovery. It must not launch a per-job GPU forward/backward/optimizer trial to decide whether a candidate fits.

**Rationale**: GPU free-memory discovery is resource observation and must remain. A per-job GPU trial is hidden training execution and blocks efficient scheduling.

**Alternatives considered**:

- Keep one-batch GPU probe as normal admission: rejected by the new product requirement.
- Ignore current GPU state and use static totals only: rejected because multi-job sharing and real cluster operation need fresh free-memory information.
- Launch formal training and rely on OOM: rejected because formal OOM is a failure, not an estimator.

## Decision: Historical Calibration Is Offline Evidence

**Decision**: Historical measured CUDA data may adjust estimator confidence only when it is recorded as prior calibration and does not trigger a new per-job trial run.

**Rationale**: Calibration can improve estimates, but the scheduler must not hide a trial execution inside every submission.

**Alternatives considered**:

- Disable all calibration: rejected because historical data can be useful for conservative estimates.
- Probe every candidate online: rejected because it violates the admission target.

## Decision: Split Backend Graph From State Payload

**Decision**: Runtime artifacts must separate graph/code/metadata from model parameter and buffer payloads. Backend graph artifacts must not contain full real parameter storage.

**Rationale**: The blocked T066 run produced a large backend graph artifact because graph serialization carried full parameter storage. This makes CPU RAM, disk, and SSH transfer bottlenecks appear before distributed training.

**Alternatives considered**:

- Continue saving full graph modules: rejected because artifact size scales with model state.
- Compress full state inside the graph artifact: rejected because CPU memory and deserialization still require full-state handling.

## Decision: Worker Loads Owned State Only

**Decision**: Workers must know ownership before materialization and load only assigned owned state or explicitly required read-only state.

**Rationale**: Loading the full backend graph on CPU and pruning non-owned state afterward still fails the larger-than-control-plane and worker-memory goals.

**Alternatives considered**:

- Full load then `to_empty()` non-owned modules: rejected because CPU deserialization has already paid the full-state cost.
- Broadcast complete model state to all workers: rejected because it defeats partition ownership and stress scalability.

## Decision: Checkpoint Finalization Must Be Memory-Safe

**Decision**: Checkpoint finalization must preserve standard model-state compatibility while avoiding full in-memory control-plane assembly for large models.

**Rationale**: Users still need ordinary model-state artifacts, but finalization must not reintroduce the same control-plane RAM limit the planner removed.

**Alternatives considered**:

- Model-name-specific reconstruction: rejected because it reintroduces zoo/model coupling.
- Drop standard model-state output: rejected because it breaks user expectations and prior checkpoint gates.

## Decision: Rebuild The Task Route From T066

**Decision**: T001-T065 remain frozen historical implementation. New unfinished work starts at T066 and replaces old T066+ planning.

**Rationale**: The earlier phases produced useful assets and commits, but T066 exposed a product-level architecture mismatch. Reopening completed task IDs would destroy auditability.

**Alternatives considered**:

- Insert fixes into earlier task ranges: rejected because it rewrites completed history.
- Mark old T066 pass with caveats: rejected because the stress did not validate GPU packing.
