# Implementation Plan: Generic PyTorch Automation

**Branch**: `002-generic-pytorch-automation` | **Date**: 2026-09-08 | **Spec**: `specs/002-generic-pytorch-automation/spec.md`

**Input**: Re-specified Feature 002 after the blocked T066 hardware stress attempt and architecture audit.

## Scope Guard

This plan updates the route for unfinished Feature 002 work only. It does not modify production code, does not edit `specs/001-*`, does not execute T066, and does not reopen T001-T065. Completed T001-T065 remain frozen historical implementation. New implementation tasks start at T066.

## Summary

ShardGrid must run ordinary PyTorch entrypoints without requiring user model rewrites, model-zoo production paths, or complete control-plane model materialization. The corrected architecture moves from the current real-execution capture and GPU-probe-gated admission toward metadata-first planning:

```text
ordinary PyTorch entrypoint
  -> non-materializing capture / fail-closed unsupported report
  -> graph, shape, dtype, lifecycle, and state metadata
  -> conservative training-memory estimate
  -> fresh host/GPU discovery and placement
  -> graph/state artifact separation
  -> worker owned-state materialization
  -> exact formal runtime
  -> memory-safe standard model-state finalization
```

The control plane must not need to hold the complete real model before planning. Production admission must not run a per-job GPU forward/backward/optimizer trial to decide whether a plan fits. Fresh GPU total/used/free memory discovery remains mandatory.

## Technical Context

**Language/Version**: Python project using PyTorch; Conda-managed environments remain the required development and runtime model.

**Primary Dependencies**: PyTorch graph/capture facilities, existing ShardGrid CLI/control/planner/runtime/checkpoint modules, SSH launcher, worker resource probes, pytest validation.

**Storage**: Filesystem job snapshots, graph/metadata artifacts, plan artifacts, bounded worker state shards, checkpoint manifests, optional historical calibration records.

**Testing**: pytest unit, contract, integration, hardware, and multi-host suites. Hardware stress remains opt-in and is run only after corrected local gates pass.

**Target Platform**: Linux control node and Linux worker runtimes over SSH first; Windows GPU workers continue to use WSL2 Linux as the training runtime layer.

**Project Type**: Python CLI and distributed-training orchestration library.

**Performance Goals**: Planning must handle model-state sizes that exceed a configured control-plane RAM limit; fresh resource discovery and admission diagnostics should complete within the existing operational timeout or return precise failure evidence; backend graph artifacts must not scale linearly with parameter payload size.

**Constraints**: Keep SSH-first delivery. Do not promote Kubernetes, Volcano, HAMi, or GPU virtualization. Do not require user `ModelProvider`, `build_model()`, `sample_inputs()`, `ShardGridModel`, stage classes, model registries, or model-name dispatch. Do not fall back to full CPU model construction, full real CPU training capture, full backend parameter serialization, worker full-model CPU load, or per-job GPU trial admission.

**Scale/Scope**: Ordinary PyTorch entrypoints, large declared model states, one or more GPU workers, multi-GPU and SSH multi-host execution, multiple jobs sharing a GPU when fresh resources and estimates admit them.

## Constitution Check

The constitution remains an unratified template. Project guidance still applies:

- Reuse PyTorch, Conda, SSH, and existing ShardGrid adapters rather than replacing them.
- Keep SSH real training as the first compatibility path.
- Preserve one-GPU-per-physical-worker as the default assumption while allowing logical partitions and multiple jobs per GPU.
- Keep diagnostics honest: unsupported semantics fail before distributed mutation.

Pre-design gate: PASS.

Post-design gate: PASS. The revised design removes the three observed product violations: full control-plane model materialization before planning, real CPU capture training before planning, and per-job GPU trial admission.

## Current Code Facts From Audit

- `shardgrid run` exists and reaches the captured-entrypoint path for small supported jobs.
- Current bootstrap executes the user's script with ordinary Python execution; model construction is not intercepted before real CPU parameters are allocated.
- Current capture observes the model after calling the real module forward and wraps tensor backward while still calling the original backward.
- Current profile logic can perform a real CPU forward with hooks.
- Non-dry-run captured entrypoints currently call the memory-probe candidate flow before formal execution.
- The generic memory probe executes forward, backward, and optimizer step on a real GPU under `--memory-probe`.
- `backend-graph.pt` stores a PyTorch graph module carrying full parameter storage; the blocked T066 attempt observed a roughly 1.3 GB backend artifact for one stress model.
- Workers eventually materialize owned modules, but first deserialize the full backend graph on CPU.
- Generic runtime launch, exact plan consumption, SSH multi-host execution, structured failures, checkpoint shards, and standard checkpoint semantics have already been partially implemented through T001-T065.
- Legacy config/automatic paths still exist as compatibility and regression paths; ordinary production entrypoint must not depend on them.

## T066 Blocked Baseline

Previous T066 status: BLOCKED.

Reason: the stress model created a large full-state backend graph and duplicate probe artifacts, causing CPU serialization and transfer bottlenecks before GPU packing could be validated. This was not a meaningful GPU memory-packing acceptance result and must not be marked PASS.

## Target Architecture

```text
User command
  shardgrid run train.py ...
    |
    v
Entry capture boundary
  records metadata without full real CPU model/training execution
  fails closed on unsupported model/optimizer/checkpoint semantics
    |
    v
Graph and state metadata
  execution nodes, tensor values, state objects, original keys, shapes, dtypes
    |
    v
Memory estimator
  parameters, buffers, gradients, optimizer state, activations,
  liveness, saved tensors, workspace, communication, dtype, batch effects
    |
    v
Planner and placement
  exact partitions + fresh host/GPU free-memory resource snapshot
    |
    v
Artifact preparation
  graph/code/metadata separated from state payloads
    |
    v
Worker runtime
  exact plan, no repartition, owned-state materialization only
    |
    v
Checkpoint finalization
  memory-safe shard validation and standard model-state output
```

## File / Module Change Map

`src/shardgrid/bootstrap/runner.py`
- Current: executes user code and observes real CPU forward/backward.
- Plan: introduce a non-materializing capture boundary and capture reports that prove no full real CPU training step occurred before planning.

`src/shardgrid/planner/generic_graph.py`
- Current: graph metadata exists, but capture depends on real model instances.
- Plan: accept metadata-first graph/state inputs and retain fail-closed diagnostics for graph breaks, unsupported ops, dynamic control flow, and unsafe state use.

`src/shardgrid/planner/memory.py`
- Current: estimates exist but do not cover all training-memory contributors and can rely on real execution hooks.
- Plan: make estimator the production admission basis before materialization, including state, gradients, optimizer state, activations, liveness, saved tensors, workspace, communication, dtype, input shape, and shared/tied state.

`src/shardgrid/planner/placement.py`
- Current: uses fresh `gpu_free_memory`.
- Plan: preserve fresh resource discovery and use conservative estimates for admission. Remove production dependence on per-job GPU training trial evidence.

`src/shardgrid/control/job_manager.py`
- Current: captured entrypoint orchestration exists but still invokes memory-probe candidate admission and persists full-state backend graphs.
- Plan: switch production admission to estimate + fresh resource validation, prepare bounded artifacts, and keep legacy probe behavior fenced to compatibility/calibration only if retained.

`src/shardgrid/runtime/generic_bootstrap.py`
- Current: loads a full backend graph on CPU, then materializes ownership.
- Plan: load graph/metadata plus owned state references only, and verify assignments match the exact plan.

`src/shardgrid/runtime/checkpoint.py`
- Current: shard validation and model-state merge exist, but full-state finalization can pressure control-plane RAM.
- Plan: finalize large checkpoints through bounded or streaming shard assembly while preserving standard model-state compatibility.

`src/shardgrid/launchers/ssh.py`
- Current: can launch the generic bootstrap.
- Plan: keep rank/world/CUDA/remote-snapshot behavior and verify no memory-probe launch is used as production admission.

`src/shardgrid/workers/gpu_probe.py`, `src/shardgrid/resources/models.py`, `src/shardgrid/control/resource_manager.py`
- Current: collect host/GPU health and free-memory resource data.
- Plan: keep as source of truth for fresh resource snapshots.

`examples/`, `scripts/`, `tests/`
- Current: contain zoo, legacy, stress, and hardware validation assets.
- Plan: keep as regression/example/stress inputs only; update stress so final GPU packing validation does not depend on huge CPU parameter artifacts.

## Key Design Decisions

### Decision: Capture Metadata Before Materialization

Use a capture boundary that records model declaration, tensor metadata, graph, state mapping, and training lifecycle evidence without requiring the control plane to allocate the full model state. If the model cannot be represented safely, return an unsupported diagnostic before mutation.

Rejected alternatives:

- Full ordinary CPU construction followed by partitioning: rejected because models larger than control-plane RAM fail before planning.
- Asking users to add ShardGrid model-provider APIs: rejected because it changes the ordinary PyTorch contract.

### Decision: No Real CPU Training Step Before Planning

Planning prerequisites must not require a full real CPU forward, backward, or optimizer step. Tests must include counters/evidence proving capture and dry-run planning did not perform those actions.

Rejected alternatives:

- "Dry-run" that still performs CPU forward/backward: rejected because it creates memory and side-effect risks.
- Silent fallback to eager execution after graph/capture failure: rejected because unsupported semantics must fail closed.

### Decision: Estimate-Based Production Admission

Production admission combines conservative model-memory estimates with fresh GPU free-memory discovery. Historical calibration can correct estimates only as offline evidence; it must not launch a per-job training trial.

Rejected alternatives:

- One-batch GPU probe as required admission: rejected because it is a hidden training execution and delays/consumes cluster capacity.
- Formal launch and rely on OOM: rejected because formal OOM is a correctness failure, not an admission strategy.

### Decision: Separate Graph Artifacts From State Payloads

Backend graph artifacts carry graph, code references, metadata, fingerprints, and state references. They must not embed full real parameter/buffer payloads. State moves through bounded owned shards or explicit state manifests.

Rejected alternatives:

- Saving full graph modules with all parameter storage: rejected because it scales with model size and caused the T066 bottleneck.
- Worker load-full-then-prune: rejected because worker CPU RAM can fail before owned-state execution begins.

### Decision: Memory-Safe Standard Checkpoint Finalization

Checkpoint output must remain ordinary model-state compatible, but large-state finalization must avoid requiring the control plane to hold the complete state in memory at once.

Rejected alternatives:

- Model-name reconstruction: rejected because it reintroduces zoo/model coupling.
- Dropping standard model-state compatibility: rejected because it breaks ordinary PyTorch user expectations.

## Revised Phase Plan

### Completed Historical Phases

T001-T065 remain frozen as completed historical implementation. They established characterization, planner repair, production zoo decoupling, entrypoint capture, local generic runtime, generic checkpoint, structured failure taxonomy, resource discovery, single-GPU, multi-GPU, SSH multi-host, and multi-job sharing validation under the previous architecture.

### New Phase 8 - Large-Model Capture And Planning Safety

Purpose: remove full control-plane model materialization and real CPU training capture before planning.

Exit gates:

- `CONTROL_PLANE_FULL_MODEL_BEFORE_PLAN=0`
- `CPU_REAL_FORWARD_BEFORE_PLAN=0`
- `CPU_REAL_BACKWARD_BEFORE_PLAN=0`
- `MODEL_LARGER_THAN_CONTROL_PLANE_RAM_PLANNING=PASS`

### New Phase 9 - Estimator-Based Admission

Purpose: make memory estimation the production admission basis while preserving fresh GPU resource discovery.

Exit gates:

- `MODEL_MEMORY_ESTIMATED_BEFORE_MATERIALIZATION=PASS`
- `PLACEMENT_USES_FRESH_FREE_VRAM=PASS`
- `PER_JOB_GPU_TRIAL_PROBE=0`

### New Phase 10 - Artifact And Worker Owned-State Safety

Purpose: ensure backend graph artifacts and worker runtime do not load or carry full model state.

Exit gates:

- `BACKEND_GRAPH_FULL_PARAMETER_PAYLOAD=0`
- `WORKER_FULL_MODEL_CPU_LOAD=0`
- `WORKER_OWNED_STATE_ONLY=PASS`
- `EXACT_PLAN_EXECUTION=PASS`

### New Phase 11 - Large Checkpoint And Regression Revalidation

Purpose: preserve standard checkpoint semantics for large models and rerun corrected local/hardware gates.

Exit gates:

- `STANDARD_STATE_DICT_COMPATIBLE=PASS`
- `CHECKPOINT_CONTROL_PLANE_FULL_STATE_LOAD=0`
- `ORDINARY_PYTORCH_ENTRYPOINT=PASS`
- `GPU_PACKING_STRESS_VALIDATED=PASS`

## Failure Taxonomy Updates

Keep existing structured failure metadata and add or reuse codes for:

- `CONTROL_PLANE_MODEL_MATERIALIZATION_BLOCKED`
- `CPU_CAPTURE_EXECUTION_BLOCKED`
- `MODEL_MEMORY_ESTIMATE_UNSUPPORTED`
- `PER_JOB_GPU_TRIAL_DISABLED`
- `BACKEND_GRAPH_PAYLOAD_UNSAFE`
- `WORKER_FULL_STATE_LOAD_BLOCKED`
- `CHECKPOINT_STREAMING_UNSUPPORTED`
- `GPU_PACKING_STRESS_BLOCKED`

Existing capture, graph, partition, placement, launch, runtime, checkpoint, and stress codes remain compatibility buckets.

## Testing Architecture

Layer 1: unit tests for capture safety, graph/state metadata, estimator accounting, artifact payload size, and worker state manifests.

Layer 2: integration tests for ordinary entrypoints, dry-run planning under configured control-plane RAM limits, no CPU forward/backward/optimizer evidence, and no per-job GPU trial launch.

Layer 3: local runtime and checkpoint tests proving exact plan and standard model-state compatibility without full-state control-plane assembly.

Layer 4: hardware and multi-host tests revalidating single GPU, multi-GPU, SSH multi-host, multi-job sharing, and corrected GPU packing stress after lower layers pass.

## Risks / Support Matrix

| Area | Status | Handling |
|------|--------|----------|
| Arbitrary Python model construction | High risk | Support representable patterns; fail closed when metadata cannot be captured safely. |
| Dynamic graph/control flow | Partial | Require safe graph/shape guards or explicit unsupported diagnostics. |
| Custom ops/autograd | Partial | Require metadata/capture support or block before mutation. |
| Mixed precision and optimizer semantics | Partial | Estimate and preserve observable behavior; reject hidden mutation. |
| Activation liveness and workspace estimation | Incomplete | Add conservative accounting and validation fixtures before hardware stress. |
| Full backend graph state payloads | Known violation | Split graph metadata from state payloads before retrying T066. |
| Worker owned-state-only loading | Partial | Make full-state CPU deserialization a failing invariant. |
| Checkpoint for larger-than-RAM state | Partial | Add bounded/streaming finalization contract and tests. |
| Historical calibration | Allowed with constraints | Offline correction only; no per-job trial execution. |
| Legacy zoo/example paths | Kept as regression | Must not be ordinary production dependencies. |

## Out Of Scope

- Reopening or renumbering T001-T065.
- Executing hardware stress during this re-specification/planning pass.
- Requiring user-facing ShardGrid model-provider APIs.
- Kubernetes, Volcano, HAMi, or GPU virtualization promotion.
- Full arbitrary Python transparency.
- Optimizer-state consolidation beyond what is already proven safe.
- Using giant CPU parameter payloads or artificial CUDA reserve buffers as the final stress strategy.

## Project Structure

```text
specs/002-generic-pytorch-automation/
├── spec.md
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   ├── capture.md
│   ├── checkpoint.md
│   ├── cli.md
│   ├── failures.md
│   └── planning.md
└── tasks.md
```

Source areas for future implementation:

```text
src/shardgrid/bootstrap/
src/shardgrid/control/
src/shardgrid/planner/
src/shardgrid/runtime/
src/shardgrid/launchers/
src/shardgrid/workers/
src/shardgrid/resources/
src/shardgrid/common/
tests/unit/
tests/integration/
tests/hardware/
tests/multi_host/
scripts/
examples/
```

## Complexity Tracking

No constitution violations recorded. The plan keeps existing ShardGrid/PyTorch/SSH/Conda boundaries and removes the heavyweight behavior that the T066 audit proved unsafe.
