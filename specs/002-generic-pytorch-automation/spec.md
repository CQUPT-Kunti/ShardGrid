# Feature Specification: Generic PyTorch Automation

**Feature Branch**: `002-generic-pytorch-automation`

**Created**: 2026-09-07

**Status**: Draft

**Input**: User description: "Create a new 002 Spec Kit feature for refactoring ShardGrid so it can run ordinary industrial PyTorch training programs through ShardGrid without user model rewrites, model-zoo special cases, or registration-order partition assumptions."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Run Existing Training Entrypoint (Priority: P1)

As a model developer, I can replace my normal training launch command with a ShardGrid launch command while keeping my model, optimizer, data loader, loss calculation, and checkpoint expectations in ordinary PyTorch form.

**Why this priority**: This is the core product promise. ShardGrid is not useful for production users if every model must be rewritten for ShardGrid-specific factories or stage classes.

**Independent Test**: Use a representative existing training script that creates a model, optimizer, data loader, loss, backward pass, optimizer step, and checkpoint without ShardGrid model interfaces. Launch it through ShardGrid and verify the first distributed training job reaches forward, backward, optimizer step, and checkpoint creation without modifying that script.

**Acceptance Scenarios**:

1. **Given** a normal PyTorch training program with a model, optimizer, data loader, and loss loop, **When** the user launches it through ShardGrid, **Then** ShardGrid captures the real training context and begins distributed execution without requiring user-defined model factories, sample-input builders, or ShardGrid model subclasses.
2. **Given** a training program that uses positional arguments, keyword arguments, or structured batch objects, **When** ShardGrid captures the first usable training step, **Then** it preserves the call structure needed to replay the model invocation for planning and execution.
3. **Given** a model that ShardGrid cannot safely capture or partition, **When** the user launches it, **Then** ShardGrid stops before distributed mutation and reports the unsupported capability with actionable diagnostics.

---

### User Story 2 - Plan From Execution Graph, Not Model Names (Priority: P2)

As an operator, I need ShardGrid to create partition and placement plans from actual graph, tensor, parameter, buffer, dependency, device, and worker data instead of special-casing known model names or catalog entries.

**Why this priority**: Name-based model handling blocks general production use and makes planner correctness depend on examples rather than real model semantics.

**Independent Test**: Run planning for at least three structurally different ordinary models with no production model-zoo identifiers. Verify plans use graph nodes, values, ownership, dependencies, memory estimates, and worker resources as the planning basis.

**Acceptance Scenarios**:

1. **Given** a model whose module registration order differs from forward execution order, **When** ShardGrid plans partitions, **Then** partition boundaries follow execution dependencies rather than registration order.
2. **Given** a parameter or buffer owned by a module that does not produce a standalone forward node, **When** ShardGrid analyzes ownership, **Then** ownership is represented separately from execution nodes and is still covered exactly once.
3. **Given** stress or zoo models present in the repository, **When** ShardGrid executes a production training job, **Then** those models are not used as the production model loading, sample creation, planning, runtime, or checkpoint basis.

---

### User Story 3 - Execute And Consolidate Generic Training (Priority: P3)

As a model developer, I need the distributed run to complete forward, backward, optimizer step, shard checkpointing, merge, and final model-state validation in a way that produces the standard model state users expect.

**Why this priority**: Capture and planning are only valuable if they lead to a usable trained model artifact.

**Independent Test**: Launch a supported ordinary training program across multiple SSH workers and verify loss evidence, parameter-change evidence, per-worker shards, merged model state, and reload validation are produced without model-name-specific checkpoint code.

**Acceptance Scenarios**:

1. **Given** a successful supported distributed run, **When** checkpoint finalization runs, **Then** ShardGrid merges worker shards into a standard model-state artifact that can be loaded into the original model state shape and dtypes.
2. **Given** checkpoint shards with missing, duplicate, mismatched, or stale parameter ownership, **When** finalization validates them, **Then** ShardGrid rejects the merge with a clear checkpoint-stage failure.
3. **Given** a production model that uses buffers, shared parameters, or parameter reuse patterns, **When** ShardGrid cannot prove safe ownership, **Then** the run is blocked before corrupting optimizer or checkpoint state.

---

### User Story 4 - Keep Validation Assets Out Of Production Flow (Priority: P4)

As a maintainer, I need existing stress models and model catalogs to remain useful for examples, tests, benchmarks, and regression gates while no longer being required by the production training path.

**Why this priority**: Existing validation coverage should be retained, but it must not define the user-facing model contract.

**Independent Test**: Run the stress-model and model-zoo validation suites as examples/tests, then run a production-style model with no zoo entry and verify production code does not require a zoo model name or synthetic sample builder.

**Acceptance Scenarios**:

1. **Given** a stress model catalog entry, **When** tests or benchmarks request it, **Then** it remains available as validation data.
2. **Given** a production training launch, **When** ShardGrid resolves the workload, **Then** it does not call a zoo model builder or synthetic sample builder as the source of truth.

### Edge Cases

- The first batch contains nested mappings, tuples, lists, non-tensor metadata, or keyword-only model inputs.
- The model uses functional operations, fused paths, module reuse, tied parameters, buffers, or modules that own state but do not appear as standalone execution nodes.
- The capture backend produces incomplete, unsupported, or ambiguous graph data.
- The sampled training step has dynamic shapes that are not representative of later batches.
- A worker inventory changes between planning and launch, including GPU memory availability or host reachability.
- Automatic placement selects fewer or different workers than requested because real memory probes or online calibration invalidate estimates.
- A run fails after partial shard checkpoint creation.
- A user requests a later Kubernetes, Volcano, or GPU-sharing path before the SSH-based training gate is proven for this feature.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: Users MUST be able to launch a normal training entrypoint through ShardGrid while keeping the original training program responsible for model construction, optimizer construction, data loading, loss calculation, and training-loop intent.
- **FR-002**: ShardGrid MUST NOT require production users to implement ShardGrid-specific model factories, sample-input builders, model base classes, stage classes, or ShardGrid-specific input/output formats.
- **FR-003**: ShardGrid MUST capture the real model invocation context from the training process, including the model instance, first usable batch, positional arguments, keyword arguments, tensor shapes, tensor dtypes, and call structure needed for planning.
- **FR-004**: ShardGrid MUST create a graph representation that identifies execution nodes, tensor values, data dependencies, graph inputs, graph outputs, and unsupported operations.
- **FR-005**: ShardGrid MUST represent parameter and buffer ownership separately from execution order so that stateful modules, functional operations, reused modules, and fused paths are not treated as registration-order slices.
- **FR-006**: The planner MUST NOT use production branches based on model names, model catalog names, known architecture names, or user-provided model type names to create, partition, launch, or checkpoint models.
- **FR-007**: Existing model zoo and stress model assets MUST be scoped to examples, tests, benchmarks, and regression validation only.
- **FR-008**: The planner MUST validate that every trainable parameter and buffer required by a supported run is owned exactly once or is explicitly blocked as unsupported.
- **FR-009**: The planner MUST validate that every execution node in a supported graph is assigned to exactly one logical partition and that every cross-partition value dependency is represented.
- **FR-010**: Automatic partitioning MUST use execution dependencies and ownership metadata rather than module registration order alone.
- **FR-011**: Placement MUST use fresh worker and GPU discovery, real memory availability, memory estimates, and online calibration before committing to distributed launch.
- **FR-012**: ShardGrid MUST stop with explicit diagnostics when capture, ownership, partition, placement, memory, communication, or checkpoint safety cannot be proven.
- **FR-013**: Distributed execution MUST preserve the original training semantics for forward, backward, optimizer step, and checkpoint intent for supported workloads.
- **FR-014**: Checkpoint shards MUST carry enough ownership, rank, worker, step, and state evidence to validate merge safety without knowing a specific model architecture name.
- **FR-015**: Checkpoint finalization MUST produce a standard merged model-state artifact for successful supported runs and verify it can be loaded against the original model state structure.
- **FR-016**: The SSH-based multi-host training path MUST remain the first compatibility gate; Kubernetes, Volcano, and GPU-sharing paths MUST NOT be described as available for this feature until their gates pass.
- **FR-017**: Windows GPU workers MUST continue to treat Windows host access and WSL2 Linux training runtime as separate runtime layers.
- **FR-018**: All Python development and training environment operations MUST use detected or reused Conda environments unless a ShardGrid-specific environment is necessary and recorded.
- **FR-019**: User-facing diagnostics MUST identify whether a failure is due to capture, graph analysis, ownership, partitioning, placement, memory, launch, training, or checkpointing.
- **FR-020**: Existing validation coverage for current examples MUST remain available so the refactor can prove it did not regress the original MVP training gates.

### Key Entities

- **Training Entrypoint**: The user-provided program and arguments that previously launched training directly and now launches under ShardGrid control.
- **Captured Training Context**: The observed model instance, first usable batch, model call arguments, optimizer intent, loss/backward boundary, and checkpoint expectation needed to plan a distributed run.
- **Execution Graph**: The ordered graph of operations, tensor values, inputs, outputs, and dependencies observed from the captured model invocation.
- **Tensor Value**: A graph value with shape, dtype, producer, consumers, gradient needs, and transfer cost metadata.
- **Parameter/Buffer Ownership Map**: The model state ownership record independent of graph execution order.
- **Logical Partition**: A group of execution nodes plus associated state ownership and boundary values that can be assigned to a worker.
- **Placement Plan**: The selected mapping from logical partitions to workers, ranks, devices, and GPU resources.
- **Worker Resource Snapshot**: Fresh host, runtime, GPU, memory, network, and Conda environment evidence used for placement.
- **Checkpoint Shard**: A worker-produced state artifact with ownership, rank, worker, step, and safety evidence.
- **Merged Model State**: The final standard model-state artifact produced from validated shards.
- **Compatibility Report**: A record explaining supported, blocked, or fallback behavior for capture, partition, runtime, communication, memory, and checkpoint stages.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A representative ordinary training program can be launched through ShardGrid with zero ShardGrid-specific changes to its model class, optimizer setup, data loader, loss calculation, or training loop.
- **SC-002**: At least three structurally different non-zoo models complete capture, graph analysis, ownership analysis, partition planning, placement, and dry-run validation without model-name-specific production branches.
- **SC-003**: At least one supported non-zoo model completes a real SSH-based multi-host run with forward, backward, optimizer step, shard checkpointing, merge, and reload validation.
- **SC-004**: For supported dry-run plans, 100% of execution nodes are assigned exactly once and 100% of required parameters and buffers are either owned exactly once or rejected with an explicit unsupported reason.
- **SC-005**: Unsupported or unsafe workloads fail before distributed parameter mutation in 100% of tested capture, ownership, partition, placement, and checkpoint-safety failure cases.
- **SC-006**: Successful checkpoint finalization produces a merged model-state artifact with matching keys, tensor shapes, and dtypes for 100% of supported validation runs.
- **SC-007**: Fresh discovery and memory validation complete within 2 minutes on a stable configured LAN or return actionable diagnostics.
- **SC-008**: The legacy stress/model-zoo validation path remains runnable as examples/tests while production launches no longer require a zoo model name or synthetic sample builder.
- **SC-009**: A maintainer reviewing a failed run can identify the failing stage from persisted diagnostics in under 5 minutes.

## Assumptions

- The active work creates a new `002-*` Spec Kit feature and treats all `001-*` feature files as historical reference only.
- The first delivery target remains SSH-based multi-host training before Kubernetes, Volcano, or GPU-sharing paths are promoted.
- The MVP hardware assumption remains one GPU per physical worker.
- The first compatible production workloads are ordinary PyTorch training programs with observable model calls and checkpoint state; workloads with unobservable side effects or unsupported dynamic behavior may be blocked with diagnostics.
- The first captured batch is acceptable for initial graph, shape, dtype, and memory planning unless later calibration proves it unsafe.
- Existing examples and stress models remain valuable as regression inputs but are not part of the production user contract.
