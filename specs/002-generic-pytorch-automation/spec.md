# Feature Specification: Generic PyTorch Automation

**Feature Branch**: `002-generic-pytorch-automation`

**Created**: 2026-09-07

**Re-specified**: 2026-09-08

**Status**: Draft

**Input**: User description: "Re-specify ShardGrid Feature 002 after the T066 hardware stress attempt showed that ordinary PyTorch automation still depends on full CPU model materialization, real CPU capture execution, per-job GPU trial probes, and full-state backend artifacts before planning can safely handle models larger than control-plane memory."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Plan Models Larger Than Control Plane Memory (Priority: P1)

As a model developer, I can launch an ordinary PyTorch training program through ShardGrid even when the total model state is larger than the control-plane machine RAM, provided the discovered GPU cluster has enough aggregate resources.

**Why this priority**: This is the architecture correction exposed by the blocked T066 run. ShardGrid cannot be credible for large-model orchestration if the control plane must first hold the complete model and then decide how to split it.

**Independent Test**: Use a normal training entrypoint whose declared model state exceeds a configured control-plane RAM limit. Verify ShardGrid reaches planning, partitioning, placement, and admission decisions from graph, shape, dtype, and state metadata without requiring the control plane to materialize the complete model state in RAM.

**Acceptance Scenarios**:

1. **Given** a normal training script that declares a model larger than control-plane RAM, **When** the user launches it through ShardGrid, **Then** ShardGrid captures the metadata needed for planning without first constructing the complete real model in control-plane memory.
2. **Given** a model whose graph or state cannot be represented safely without full materialization, **When** ShardGrid attempts capture, **Then** it fails before distributed mutation with an explicit unsupported capability report.
3. **Given** a supported large model and enough aggregate fresh GPU resources, **When** planning completes, **Then** the selected plan assigns only owned state to workers and never depends on a full control-plane model copy.

---

### User Story 2 - Capture Training Metadata Without Real CPU Training (Priority: P2)

As a model developer, I need ShardGrid to understand my training entrypoint without performing a full real CPU forward pass, CPU backward pass, or CPU optimizer step before planning.

**Why this priority**: Real CPU execution during capture makes large-model planning fail before partitioning and can run unintended training-side effects in the wrong environment.

**Independent Test**: Run supported ordinary entrypoint fixtures through ShardGrid dry-run capture and verify the capture artifacts contain model graph, input/output structure, tensor metadata, state mapping, and training lifecycle boundaries while counters prove no real CPU forward, backward, or optimizer mutation occurred before planning.

**Acceptance Scenarios**:

1. **Given** an ordinary training script with model, batch, loss, backward, optimizer, and checkpoint intent, **When** ShardGrid captures it for planning, **Then** ShardGrid records the required metadata without a full real CPU training step.
2. **Given** structured model inputs such as positional tensors, keyword tensors, masks, labels, mappings, tuples, or nested batches, **When** capture succeeds, **Then** the original call structure is preserved for planner and runtime use.
3. **Given** dynamic behavior, graph breaks, custom operations, or hidden optimizer mutation that cannot be represented safely, **When** capture evaluates the entrypoint, **Then** ShardGrid stops with a precise capture or graph failure before mutation.

---

### User Story 3 - Admit Jobs From Estimates And Fresh Resources, Not Trial Training (Priority: P3)

As an operator, I need ShardGrid to decide whether a job fits by combining conservative model-memory estimates with fresh host and GPU resource discovery, without first running a one-batch GPU training trial for every job candidate.

**Why this priority**: Per-job GPU trial execution consumes cluster capacity, delays scheduling, and can turn admission into an expensive hidden training run. Fresh GPU state is still required, but model memory need must be estimated before materialization and formal execution.

**Independent Test**: Submit jobs that need single-GPU, multi-GPU, and shared-GPU placement. Verify admission uses current host reachability, GPU health, total memory, used memory, and free memory plus pre-materialization model estimates, and verify no production admission path launches a GPU forward/backward/optimizer trial before formal training.

**Acceptance Scenarios**:

1. **Given** multiple healthy reachable GPUs with different current free memory, **When** ShardGrid plans a job, **Then** placement uses the fresh resource snapshot and the model memory estimate to select a feasible exact plan.
2. **Given** historical calibration data exists, **When** the estimator uses it, **Then** the data is treated as offline correction evidence and does not trigger a per-job GPU training trial.
3. **Given** no feasible placement satisfies estimated memory, resource, and safety constraints, **When** admission completes, **Then** the job fails with a no-feasible-plan or memory-estimation diagnostic rather than trying formal training until OOM.

---

### User Story 4 - Execute With Owned State Only And Safe Artifacts (Priority: P4)

As a model developer, I need workers and checkpoints to handle only their owned model state during distributed execution and finalization, while still producing the standard model-state artifact expected by ordinary PyTorch workflows.

**Why this priority**: The blocked stress attempt showed backend artifacts and worker loading can become CPU, disk, and transfer bottlenecks before GPU training starts. Checkpoint finalization must also stay safe for models larger than control-plane memory.

**Independent Test**: Run supported ordinary models through distributed execution and checkpoint finalization. Verify backend artifacts do not carry the full parameter payload, workers materialize only owned state, and final model-state output is standard and strict-loadable without requiring a full control-plane state load for models over the configured RAM limit.

**Acceptance Scenarios**:

1. **Given** a supported distributed plan, **When** runtime artifacts are prepared, **Then** graph/code/metadata artifacts are separated from model state payloads and remain bounded independently of total model size.
2. **Given** a worker receives an exact assignment, **When** execution starts, **Then** the worker loads and materializes only the state it owns or is explicitly allowed to read.
3. **Given** a successful distributed run, **When** checkpoints are finalized, **Then** the result preserves ordinary model-state compatibility and avoids requiring the control plane to hold the entire model state at once for large models.

---

### User Story 5 - Keep Existing Capabilities And Validation Assets (Priority: P5)

As a maintainer, I need the corrected architecture to preserve completed T001-T065 capabilities and keep existing zoo, example, stress, and hardware validation assets as regression coverage without making them the ordinary production model contract.

**Why this priority**: The new route must repair the architecture without invalidating already proven SSH, resource discovery, exact-plan runtime, checkpoint, multi-host, and multi-job-sharing work.

**Independent Test**: Run the historical regression gates for completed tasks and the new post-correction gates. Verify completed task definitions remain frozen, ordinary production launch does not depend on model names or zoo reconstruction, and hardware validation is re-run only after the corrected local gates pass.

**Acceptance Scenarios**:

1. **Given** completed T001-T065 task history, **When** the new plan is reviewed, **Then** those tasks remain unchanged and are marked as historical implementation.
2. **Given** zoo or legacy examples are present, **When** production ordinary entrypoint execution runs, **Then** those assets are not used as production model loading, sample generation, runtime reconstruction, or checkpoint reconstruction paths.
3. **Given** the corrected architecture reaches hardware validation, **When** stress acceptance runs, **Then** it measures GPU memory packing and training progress rather than CPU serialization capacity.

### Edge Cases

- The declared model state is larger than control-plane RAM, larger than one GPU, or larger than any single worker but fits the aggregate cluster.
- Capture cannot represent a custom operation, data-dependent control flow, dynamic shape guard, optimizer side effect, mixed precision behavior, or checkpoint intent without unsafe fallback.
- Tensor or state metadata is incomplete, ambiguous, or would require full materialization to validate.
- A worker inventory changes between planning and launch, including host reachability, GPU health, or current free memory.
- A model uses shared modules, tied parameters, buffers, functional operations, multi-consumer values, activation checkpointing, mixed precision, or gradient accumulation.
- Historical calibration data is stale, missing, or not applicable to the current model family.
- Backend graph, state shards, or checkpoint finalization would exceed configured memory or transfer limits.
- A hardware stress workload consumes CPU serialization capacity before it exercises GPU packing.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: Users MUST be able to launch ordinary PyTorch training entrypoints through ShardGrid without adding ShardGrid-specific model providers, sample builders, stage classes, model registries, or model-name-specific APIs.
- **FR-002**: ShardGrid MUST NOT require the control plane to materialize the complete real model parameters or buffers before graph capture, memory estimation, partitioning, or placement.
- **FR-003**: ShardGrid MUST support planning for models whose total state size exceeds a configured control-plane RAM limit when the model is otherwise representable and the cluster has sufficient aggregate resources.
- **FR-004**: ShardGrid MUST NOT perform a full real CPU forward pass, real CPU backward pass, or real CPU optimizer step as a prerequisite for planning.
- **FR-005**: ShardGrid MUST collect graph, shape, dtype, state mapping, input/output structure, and lifecycle metadata through non-materializing or bounded capture mechanisms, and MUST fail closed when those mechanisms cannot prove safety.
- **FR-006**: ShardGrid MUST represent execution nodes, tensor values, parameters, buffers, state ownership, and checkpoint keys as separate concepts rather than treating module registration order as execution order or ownership.
- **FR-007**: ShardGrid MUST estimate training memory before materialization and formal execution, including parameters, buffers, gradients, optimizer state, activations, activation liveness, backward-saved tensors, temporary/workspace memory, communication buffers, dtype or mixed precision effects, batch/input shape effects, and shared/tied state.
- **FR-008**: ShardGrid MUST use fresh host and GPU discovery for every new planning/admission decision, including reachability, health, total memory, used memory, and current free memory.
- **FR-009**: ShardGrid MUST NOT use a per-job GPU trial forward, backward, or optimizer step as the production admission mechanism for deciding whether a candidate plan fits.
- **FR-010**: Historical or offline calibration MAY adjust estimates only when it does not launch a per-job trial training step and its provenance is recorded.
- **FR-011**: Placement MUST combine conservative model-memory estimates, fresh GPU free memory, topology/resource constraints, and exact partition metadata before formal execution.
- **FR-012**: Runtime artifacts MUST separate graph/code/metadata from model state payloads so backend graph artifacts do not carry the full real parameter or buffer payload.
- **FR-013**: Workers MUST load and materialize only assigned owned state or explicitly required read-only state and MUST NOT first load the full model state on CPU and prune it afterward.
- **FR-014**: Runtime MUST consume the exact planner-selected partition and placement plan and MUST NOT repartition, re-place, or use rank-local round-robin assignment.
- **FR-015**: Checkpoint finalization MUST preserve standard model-state compatibility while supporting memory-safe assembly or streaming for models larger than control-plane RAM.
- **FR-016**: Checkpoint shards MUST carry enough ownership, key, shape, dtype, worker, rank, step, fingerprint, and validation evidence to reject missing, duplicate, stale, or mismatched state without model-name-specific reconstruction.
- **FR-017**: Production planning, runtime launch, and checkpoint finalization MUST NOT branch on model catalog names, example model names, `model.type`, zoo builders, or synthetic sample builders for ordinary entrypoint runs.
- **FR-018**: Existing zoo, legacy automatic, and stress assets MUST remain scoped to examples, compatibility tests, benchmarks, or regression validation only.
- **FR-019**: User-facing diagnostics MUST identify the failing stage and precise reason across capture, graph analysis, estimation, ownership, partitioning, placement, resource refresh, launch, runtime, checkpointing, and stress validation.
- **FR-020**: The previously attempted T066 hardware stress MUST remain recorded as blocked and MUST NOT be marked passed until the corrected architecture validates GPU packing without full-state CPU artifact bottlenecks.
- **FR-021**: The completed T001-T065 historical task definitions MUST remain frozen; new implementation work MUST begin at T066 without duplicate task IDs.
- **FR-022**: The feature MUST preserve already implemented capabilities: dynamic host/GPU discovery, fresh free-memory resource information, logical partition independence from GPU/process/rank, exact plan execution, worker ownership semantics, multi-GPU, SSH multi-host, multi-job GPU sharing, structured failure taxonomy, and standard checkpoint semantics.

### Key Entities

- **Training Entrypoint**: The user-provided training program and arguments launched through ShardGrid without ShardGrid-specific model APIs.
- **Non-Materializing Capture Context**: The metadata record describing model declaration, graph, inputs, outputs, lifecycle boundaries, and state mapping without requiring a full real model in control-plane memory.
- **Execution Graph**: The dependency graph of operations, tensor values, inputs, outputs, shapes, dtypes, producers, consumers, and unsupported semantics.
- **State Object**: A parameter or buffer identified by canonical state identity, original model-state key, shape, dtype, ownership, sharing/tied relationship, and use sites.
- **Memory Estimate**: A pre-materialization estimate of training memory needs across state, gradients, optimizer state, activations, temporary/workspace memory, communication buffers, dtype behavior, and safety margin.
- **Fresh Resource Snapshot**: Current host and GPU health, reachability, total memory, used memory, free memory, runtime, and timestamp evidence used for planning.
- **Logical Partition**: A group of execution nodes and explicitly owned/read state with boundary tensor values and memory/transfer estimates.
- **Placement Plan**: The exact selected mapping from logical partitions to workers, ranks, devices, and resources.
- **Backend Graph Artifact**: Graph, code, and metadata needed by runtime, explicitly separated from full model state payloads.
- **State Shard**: A bounded artifact containing only state assigned to a worker or a checkpoint merge step.
- **Merged Model State**: The standard model-state artifact or stream produced after validated shard finalization.
- **Compatibility Report**: A record showing pass, fail, blocked, unsupported, or regression status for local, GPU, multi-host, multi-job, and stress gates.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A representative ordinary training program can be launched through ShardGrid with zero ShardGrid-specific changes to its model class, optimizer setup, data loader, loss calculation, or training loop.
- **SC-002**: In validation with a configured 16 GB control-plane RAM limit, ShardGrid can produce a plan or safe unsupported diagnostic for models with at least 30 GB, 70 GB, and 100 GB declared state without requiring full control-plane model materialization.
- **SC-003**: For supported dry-run captures, 100% of planning prerequisites complete without a real CPU forward pass, CPU backward pass, or CPU optimizer step.
- **SC-004**: For supported production admission tests, 0 per-job GPU trial forward/backward/optimizer executions occur before formal training launch.
- **SC-005**: For supported plans, 100% of execution nodes are assigned exactly once and 100% of required parameters and buffers are either owned exactly once or rejected with an explicit unsupported reason.
- **SC-006**: Placement decisions use resource snapshots no older than the configured freshness window in 100% of tested job submissions, including current GPU free-memory evidence.
- **SC-007**: Backend graph artifacts for large-model validation remain bounded by graph and metadata size and do not scale linearly with total parameter payload.
- **SC-008**: Workers in supported distributed runs load only their owned or explicitly required read-only state in 100% of inspected assignments.
- **SC-009**: Successful checkpoint finalization produces a standard model-state output that validates against the original model-state structure for 100% of supported validation runs, without full in-memory control-plane assembly for large-state cases.
- **SC-010**: Unsupported or unsafe workloads fail before distributed parameter mutation in 100% of tested capture, estimation, ownership, partition, placement, runtime, and checkpoint-safety failure cases.
- **SC-011**: The corrected hardware validation suite reports actual discovered hosts/GPUs, placement, current/final memory evidence, optimizer progress, checkpoint validation, cleanup, and saturation classification for every final stress run.
- **SC-012**: T001-T065 remain unchanged as completed historical task definitions, and the new task plan starts from T066 with no duplicate task IDs.

## Assumptions

- The active feature remains `specs/002-generic-pytorch-automation`; this re-specification updates the existing feature rather than creating a new `003-*` feature.
- T001-T065 are treated as completed historical implementation and are not re-opened or renumbered by this re-plan.
- The previous T066 attempt is blocked because full-state backend artifacts and CPU serialization/transfer bottlenecks prevented meaningful GPU packing validation.
- The first corrected delivery target remains SSH-based ShardGrid execution before Kubernetes, Volcano, HAMi, or GPU virtualization paths are promoted.
- Fresh GPU resource discovery remains mandatory; only per-job GPU training trial admission is removed from the production path.
- Some ordinary PyTorch programs will remain unsupported until their graph, state, optimizer, or checkpoint semantics can be proven without unsafe fallback.
