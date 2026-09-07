# Implementation Plan: Generic PyTorch Automation

**Branch**: `002-generic-pytorch-automation` | **Date**: 2026-09-07 | **Spec**: `specs/002-generic-pytorch-automation/spec.md`

**Input**: Feature specification from `specs/002-generic-pytorch-automation/spec.md`

## Scope Guard

This plan is design-only. It does not modify production code, old `001-*` feature files, or the completed `002` spec. `specs/001-multi-host-training-mvp/*` remains historical reference.

## Summary

Refactor ShardGrid from model-name-driven automatic examples into a generic PyTorch entrypoint takeover path:

```bash
python train.py --config config.yaml
```

becomes:

```bash
shardgrid run train.py --config config.yaml
```

ShardGrid will internally capture the user's real `nn.Module`, first usable batch, call args/kwargs, execution graph, state ownership, fresh worker/GPU resources, memory probes, exact partition/placement plan, runtime execution, shard checkpoints, and merged PyTorch `state_dict`. Production code must not require `build_model()`, `sample_inputs()`, `ModelProvider`, `ShardGridModel`, `ShardGridStage`, model-zoo names, or ShardGrid-specific input formats.

## Technical Context

**Language/Version**: Python project using PyTorch; Conda-managed environments per project guidance.

**Primary Dependencies**: PyTorch, PyTorch FX/export/Dynamo capture APIs, SSH launcher, existing ShardGrid planner/runtime/checkpoint modules.

**Storage**: Filesystem job snapshots, plan JSON, worker checkpoint shards, merged `state_dict` artifacts.

**Testing**: Existing pytest-style Python tests plus hardware validation on local GPU, multi-GPU, SSH multi-host, and stress runs.

**Target Platform**: Linux control node and Linux worker runtimes over SSH first; Windows GPU workers remain WSL2 Linux runtime targets.

**Project Type**: Python CLI/distributed-training orchestration library.

**Performance Goals**: Fresh discovery and memory validation complete within 2 minutes on stable LAN or return diagnostics; bounded candidate search avoids unbounded placement/probe loops.

**Constraints**: Keep SSH MVP as first compatibility gate. Do not claim Kubernetes, Volcano, or HAMi availability for this feature. Do not add a second fixed reserve/headroom admission layer beside estimate + calibration + bounded search + one-batch probe.

**Scale/Scope**: One or more physical GPU workers; one GPU per worker is the default, while multiple independent jobs/partitions may share a GPU when fresh free-memory/probe evidence allows.

## Constitution Check

The current constitution file is still an unratified template, so there are no enforceable project-specific gates beyond repository guidance and this feature spec.

Pre-design gate: PASS. The plan preserves SSH-first delivery, Conda environment handling, real resource discovery, and reuse of PyTorch/SSH/distributed runtime boundaries.

Post-design gate: PASS. Added artifacts define capture, planning, checkpoint, and failure contracts without introducing production model-provider APIs.

## Current Code Facts

### CLI / Entrypoint

- `src/shardgrid/cli/app.py`: `register_train_command()` is imported and registered by `main()`.
- `src/shardgrid/cli/commands/train.py`: `register_train_command()` wires `train`; `run_train_command()` creates `JobManager(_resolve_cluster_config(args))` and calls `manager.run(args.config_path, dry_run=dry_run)`.
- Current user-facing command is config-first. There is no production `shardgrid run train.py ...` takeover command yet.

### JobManager Orchestration

- `src/shardgrid/control/job_manager.py`: `JobManager.run()` loads config, probes workers, builds plans, runs memory probes, snapshots, launches, monitors, collects, and finalizes checkpoints.
- `_build_execution_plan()` turns a selected `ParallelPlan` into launchable assignments.
- `_build_automatic_parallel_plan()` calls `_planner_workload()`, `build_model_profile()`, `search_joint_partition_placement()`, then runtime-plan generation.
- `_prepare_live_execution_plan()` refreshes resources/network and allocates a fresh master port before launch.
- `_select_memory_probe_candidate()` runs bounded candidate probes, treats memory rejection as a normal candidate failure, cleans up, and tries the next plan.

### Model Name Coupling

Production-adjacent branches currently still depend on model names/types:

- `src/shardgrid/control/job_manager.py`: `_planner_workload()` branches on `training_config.model.type` values such as `minimal_sequential`, `hf_style`, `large_residual_transformer`, and `generic_dag`.
- `src/shardgrid/control/job_manager.py`: `_launch_command_for_assignment()` maps automatic generic DAG work to `examples/models/train_generic_dag.py`.
- `src/shardgrid/control/job_manager.py`: `_write_consolidated_model()` reconstructs model-specific state, including a `generic_dag` path that calls `_write_generic_dag_model_state()`.
- `examples/models/train_generic_dag.py`: imports `build_zoo_model()` / `make_zoo_sample()` and reconstructs runtime models from `zoo_model`.
- `examples/models/train_automatic_plan.py`: branches over known example model types.
- `scripts/stress_dynamic_gpu_multi_job.py`: `MODEL_SPECS` and generated configs use `zoo_model`.
- `examples/models/generic_partition_zoo/models.py`: `build_zoo_model()` and `make_zoo_sample()` are useful validation fixtures, not a production model contract.

Disposition:

| Path / Logic | Keep | Migrate | Replace | Delete |
|--------------|------|---------|---------|--------|
| `examples/models/generic_partition_zoo/models.py` | tests/examples/benchmarks | out of production path | captured user model context | no |
| `scripts/stress_dynamic_gpu_multi_job.py` zoo configs | current stress regressions | add generic-entrypoint fixtures | structured failure/evidence input | not until replacement coverage exists |
| `JobManager._planner_workload()` model branches | legacy compatibility | generic captured workload | `CapturedTrainingContext` planner input | after generic gates replace old gates |
| `JobManager._launch_command_for_assignment()` example dispatch | legacy compatibility | generic bootstrap launch | runtime bootstrap command | model-name dispatch after deprecation |
| `JobManager._write_consolidated_model()` model reconstruction | shard validation pieces | generic state merge | captured `state_dict` key map | model-specific reconstruction after strict reload coverage |
| `examples/models/train_generic_dag.py` | regression/example runtime | no production invocation | generic runtime consuming exact plan | no |
| `examples/models/train_automatic_plan.py` | legacy regression | behind compatibility suite | generic runtime | eventual production dependency only |

### Graph Capture / Profile / Planner

- `src/shardgrid/planner/generic_graph.py`: `GenericGraphIR`, `CanonicalGraphIR = GenericGraphIR`, `GraphNodeSpec`, `GraphValueSpec`, `ParameterUseSpec`, `_graph_from_fx()`, `module_dependencies_from_graph()`, `_parameter_owners()`, and `_parameter_uses()` build the current graph/state view from FX.
- `src/shardgrid/planner/memory.py`: `build_model_profile()` and `_profile_modules()` still expose `ModelProfile.modules`; `_iter_target_modules()` walks `model.named_modules()` and skips pure containers.
- `src/shardgrid/planner/partitioning.py`: `discover_partition_support()`, `generate_partition_candidates()`, `validate_partition_candidate()`, `build_partition_profile()`, `_normalize_traced_order()`, and `_ordered_modules_for_support()` still normalize graph order back onto profiled module paths.
- `src/shardgrid/planner/planning_contract.py`: already contains graph-era contracts such as `RuntimeCapabilities`, `PlanningConstraints`, `ResourceSnapshot`, `LogicalPartitionSpec`, `PlacementPlan`, and `PlanCandidate`, but they need to become the production contract instead of side structures.
- `src/shardgrid/planner/placement.py`: `search_joint_partition_placement()` uses candidate partition plans and fresh `WorkerResource.gpu_free_memory`; `_usable_memory_bytes()` is the placement memory gate.

### Runtime / Launcher / Checkpoint

- `src/shardgrid/runtime/dag.py`: `WorkerOwnershipSpec`, `WorkerOwnershipPlan`, `RuntimePlan`, `LocalDAGRuntime`, and `compile_runtime_plan()` consume logical partitions, placement, parameter IDs, buffer IDs, and remote/local edges.
- `src/shardgrid/runtime/partition_graph.py`: extracts FX partitions and assumes backend FX nodes align with canonical graph nodes.
- `src/shardgrid/launchers/ssh.py`: `SSHLauncher.launch()` runs each `ExecutionPlan.launch_command`; `_launch_argv()` rewrites entrypoints into the remote snapshot; `_launch_env()` sets rank/world/master/CUDA env.
- `src/shardgrid/runtime/checkpoint.py`: `save_worker_state_shard()` writes schema, graph fingerprint, plan ID, rank, owned partitions, parameters, and buffers; `consolidate_worker_state_shards()` detects duplicate/missing canonical IDs.
- `src/shardgrid/control/job_manager.py`: `_finalize_checkpoint_bundle()` and `_write_consolidated_model()` still bridge from shards back to model-specific reconstruction.

### Worker / Resource / Stress

- `src/shardgrid/workers/probe.py` and `src/shardgrid/workers/gpu_probe.py` collect host/runtime/GPU evidence.
- `src/shardgrid/resources/models.py`: `WorkerResource` and `GPUResource` carry GPU total/free memory, health, runtime, Conda, Python, NCCL/Gloo, and network fields.
- `src/shardgrid/control/resource_manager.py`: `ResourceManager.build_cluster_state()` filters stale/unhealthy/runtime/network state.
- `scripts/stress_dynamic_gpu_multi_job.py`: failure classification is substring-based, and `probe_isolation_check()` currently treats `after_steps < before_steps` as stalled; equal progress (`after_steps == before_steps`) must also be considered a stall for isolation evidence.

## Current Semantic Problem

The current automatic planner still lets one ordered list carry too many meanings:

- `model.named_modules()` registration order from `src/shardgrid/planner/memory.py`
- FX/export execution-node order from `src/shardgrid/planner/generic_graph.py`
- partition boundary order in `src/shardgrid/planner/partitioning.py`
- state ownership and checkpoint ownership in `src/shardgrid/runtime/dag.py` and `src/shardgrid/runtime/checkpoint.py`

Those are not equivalent for real models. Residuals, attention blocks, reused modules, functional ops, fused paths, multi-consumer values, parameters used by functional calls, shared/tied weights, and state-owning modules without standalone forward nodes all break registration-order slicing.

## Target Architecture

```text
User Training Entrypoint
  |
  v
ShardGrid Bootstrap (`shardgrid run ...`)
  |
  v
Internal Runtime Shim / First-Step Interception
  |
  v
CapturedTrainingContext
  |-- real nn.Module instance
  |-- first batch and args/kwargs
  |-- optimizer/loss/backward/checkpoint boundary evidence
  v
Execution Graph (`torch.export` preferred, FX fallback)
  |
  v
Ownership Model (parameters/buffers/state_dict keys, independent of execution order)
  |
  v
Planner
  |-- LogicalPartition
  |-- PlacementPlan from fresh WorkerResource/GPUResource
  |-- bounded candidates + online calibration + one-batch memory probe
  v
Exact RuntimePlan / ExecutionPlan
  |
  v
Distributed Runtime (no repartition, no rank-local placement decisions)
  |
  v
Checkpoint Shards
  |
  v
Merged PyTorch model-state artifact
```

```text
Control Plane
  JobManager
    |
    | probes SSH workers, Conda runtimes, network, GPUs
    v
WorkerResource snapshots
    |
    v
Placement planner selects Worker/GPU assignments
    |
    v
SSHLauncher starts exact rank commands
    |
    v
Workers execute assigned RuntimePlan on selected GPU
    |
    v
Per-worker checkpoint shards return to control plane
```

## File / Module Change Map

`src/shardgrid/cli/app.py`
- Current: registers config-driven `train`.
- Plan: add/register `run` while keeping `train` as compatibility alias.

`src/shardgrid/cli/commands/train.py`
- Current: `run_train_command()` passes config to `JobManager.run()`.
- Plan: keep existing command; add a thin compatibility path to the new entrypoint contract only after `run` is stable.

`src/shardgrid/cli/commands/run.py` (new)
- Current: absent.
- Plan: parse `shardgrid run [--config CLUSTER_CONFIG] [--dry-run] [--json] ENTRYPOINT [ARGS...]`; pass entrypoint metadata to control plane.

`src/shardgrid/control/job_manager.py`
- Current: central orchestration plus model-type-specific workload/build/launch/checkpoint branches.
- Plan: split generic capture/planning inputs from legacy config workloads; replace `_planner_workload()` model-type production branches with captured context; replace `_launch_command_for_assignment()` example-script selection with generic runtime bootstrap; move zoo/model-specific checkpoint code behind compatibility tests.

`src/shardgrid/planner/generic_graph.py`
- Current: `GenericGraphIR` is FX-derived and still tied to module paths/state collected from `named_modules()`.
- Plan: make it execution-node-first: graph inputs/outputs, values, producers/consumers, unsupported ops, dynamic shape guards, and parameter/buffer uses by canonical state object ID.

`src/shardgrid/planner/memory.py`
- Current: `ModelProfile` / `ModuleProfile` are module-slice oriented.
- Plan: keep memory estimators but extend/split profile into execution-node cost, value activation cost, and state-object cost. `named_modules()` remains metadata, not partition order.

`src/shardgrid/planner/partitioning.py`
- Current: partitions normalize traced order back to profiled module paths.
- Plan: partition over execution graph topological groups and value dependencies; validate exact node coverage, boundary values, state ownership, shared-parameter safety, and multi-consumer transfer requirements.

`src/shardgrid/planner/planning_contract.py`
- Current: contains useful logical partition and planning-result dataclasses.
- Plan: promote these as the generic planner contract; split execution nodes from `parameter_ids` / `buffer_ids`; add failure codes and validation evidence.

`src/shardgrid/planner/placement.py`
- Current: searches partition/worker placement using fresh GPU free memory.
- Plan: preserve fresh discovery and bounded search; consume the new logical partition memory model and keep memory-probe rejection as candidate-level feedback.

`src/shardgrid/runtime/dag.py`
- Current: `compile_runtime_plan()` consumes logical partitions and placement but is fed by zoo/example plans.
- Plan: consume captured graph + ownership + exact `PlacementPlan`; do not repartition or rank-local decide placement; keep `PLAN_RUNTIME_CONSISTENCY_CHECK=PASS`.

`src/shardgrid/runtime/partition_graph.py`
- Current: assumes backend FX node count aligns with canonical graph.
- Plan: map partitions by stable execution-node IDs and value IDs; reject mismatches as plan validation failures.

`src/shardgrid/runtime/checkpoint.py`
- Current: shard schema is close to generic but final merge still depends on upstream model-specific reconstruction.
- Plan: preserve original `state_dict` keys in shard entries; validate schema, graph fingerprint, plan ID, owner, shape, dtype, missing/duplicate keys; emit standard merged model-state without model-name reconstruction.

`src/shardgrid/launchers/ssh.py`
- Current: launches configured commands and env through SSH snapshots.
- Plan: launch ShardGrid runtime bootstrap with serialized captured plan/context references; keep env/rank/world/CUDA handling.

`src/shardgrid/workers/probe.py`, `src/shardgrid/workers/gpu_probe.py`
- Current: collect runtime/GPU evidence.
- Plan: keep as source of truth for dynamic discovery; ensure `gpu_free_memory` freshness is carried into admission diagnostics.

`src/shardgrid/resources/models.py`
- Current: `WorkerResource` / `GPUResource` model dynamic host/GPU data.
- Plan: keep and extend only if diagnostics need explicit capture/probe timestamps.

`examples/models/train_generic_dag.py`
- Current: generic DAG example runtime still reconstructs from `zoo_model`.
- Plan: keep as example/regression asset; production runtime stops invoking it.

`examples/models/train_automatic_plan.py`
- Current: legacy model-specific automatic runtime.
- Plan: keep under compatibility tests until the generic entrypoint path covers old gates; then deprecate.

`examples/models/generic_partition_zoo/models.py`
- Current: stress/test catalog and sample/model builders.
- Plan: retain for examples, benchmarks, regression, and stress input only.

`scripts/stress_dynamic_gpu_multi_job.py`
- Current: stress configs are zoo-based; failure/progress evidence is coarse.
- Plan: add generic-entrypoint stress fixtures, consume structured failure codes, sample checkpoints, verify cleanup, and treat `after_steps == before_steps` as stalled progress.

`src/shardgrid/common/enums.py`, `src/shardgrid/common/errors.py`, `src/shardgrid/jobs/models.py`
- Current: broad `FailureStage` / `FailureRecord`.
- Plan: extend with structured failure taxonomy while preserving existing fields for compatibility.

## Key Data Structures

| Structure | Current Location | Plan |
|-----------|------------------|------|
| `GenericGraphIR` / `CanonicalGraphIR` | `src/shardgrid/planner/generic_graph.py` | Extend as execution graph: node/value/dependency/unsupported-op metadata; keep alias compatibility. |
| `GraphNodeSpec` / `GraphValueSpec` | `src/shardgrid/planner/generic_graph.py` | Keep but ensure stable execution-node IDs and boundary value metadata. |
| `ParameterUseSpec` | `src/shardgrid/planner/generic_graph.py` | Extend into state-object use records for params and buffers. |
| `ModelProfile` / `ModuleProfile` | `src/shardgrid/engines/models.py` | Split semantics: execution-node cost, activation value cost, state-object cost. |
| `LogicalPartitionSpec` | `src/shardgrid/planner/planning_contract.py` | Keep; make node coverage and state ownership explicit and independent. |
| `ParallelPlan` / `ParallelPlanStage` | `src/shardgrid/engines/models.py` | Keep compatibility fields; migrate new runtime to logical partition + placement IDs. |
| `WorkerOwnershipPlan` | `src/shardgrid/runtime/dag.py` | Keep; source from ownership model, not module slices. |
| `RuntimePlan` | `src/shardgrid/runtime/dag.py` | Keep; add captured input/value routing references as needed. |
| `ExecutionPlan` | existing control/launcher model | Keep as launch artifact; command points to generic runtime bootstrap. |
| `ResourceSnapshot` | `src/shardgrid/planner/planning_contract.py` | Keep fresh-worker input to placement/admission. |
| `CheckpointShard` metadata | `src/shardgrid/runtime/checkpoint.py` | Extend with preserved state_dict keys, canonical IDs, owner, shape, dtype, graph fingerprint, plan ID. |
| Zoo model specs | `examples/models/generic_partition_zoo/models.py` | Retain for validation; remove from production planning/runtime/checkpoint path. |

## Design Decisions

### Entrypoint Takeover

Recommended path: `shardgrid run` launches the user script under an internal ShardGrid bootstrap that intercepts the first training step inside the same process. The bootstrap records the real module, first batch, model call args/kwargs, tensor metadata, optimizer/loss/backward/checkpoint boundary evidence, and then hands a serialized planning context to `JobManager`.

Why: it is the smallest viable takeover that uses the user's actual training program instead of a ShardGrid model API.

Alternatives:

- Static import and call `build_model()` / `sample_inputs()`: rejected for production because it creates a ShardGrid-specific model contract.
- User-provided wrapper/provider classes: rejected for the same reason.
- Full arbitrary Python replay across workers on day one: rejected as too broad; phase the boundary through captured first-step semantics and explicit unsupported failures.
- Pure FX tracing from `named_modules()`: rejected because registration order is not execution semantics.

Capture backend order:

1. Prefer `torch.export` when the first step can be captured with stable guards and fake/meta coverage.
2. Fall back to FX symbolic tracing for simpler models.
3. Use Dynamo/export diagnostics to explain graph breaks, data-dependent control flow, custom ops, or dynamic-shape failures.
4. Block before distributed mutation when graph/state safety cannot be proven.

### Training Control Boundary

Phase 1 boundary:

- User script still owns data loading, first-batch creation, model construction, optimizer construction, scheduler/callback declarations, and checkpoint intent.
- ShardGrid owns distributed execution after capture: partitioned forward, backward orchestration, parameter update for owned state, activation/gradient transfer, memory probe, checkpoint shard save, and final merge.
- The boundary is the captured first training step: the shim observes the original call and translates supported semantics into the runtime plan.

Final target:

- Preserve user optimizer semantics by extracting optimizer class, param-group structure, hyperparameters, gradient accumulation, AMP/scaler intent, and scheduler step boundary where supported.
- Reject unsupported optimizer-side mutation or hidden side effects before distributed state mutation.

### Planner Semantics

The new planner model has four independent layers:

1. Execution graph: operation nodes and tensor values in topological/dependency order.
2. State ownership: parameters/buffers keyed by canonical ID and original `state_dict` key.
3. Logical partitions: graph-node groups plus explicit state-object ownership and boundary values.
4. Placement: logical partitions mapped to worker/rank/GPU using fresh resources.

Validation migrates from module-order checks to:

- every execution node assigned exactly once;
- every required parameter/buffer owned exactly once, or rejected as unsupported;
- every cross-partition value has producer, consumer, shape, dtype, and transfer plan;
- shared/tied state is either placed with all writers/readers safely represented or blocked;
- runtime plan fingerprint matches planner output.

### Memory Probe / Admission

Keep the current chain:

```text
static estimate
+ online calibration
+ bounded candidate search
+ real one-batch memory probe
```

`MEMORY_REJECT` from probe is normal candidate rejection. Formal training CUDA OOM is a correctness failure. Candidate A probe failure must cleanup and continue to Candidate B until the bounded search is exhausted.

### Generic Checkpoint

Checkpoint output is standard model state:

```python
state = torch.load("model-state.pt")
original_model.load_state_dict(state, strict=True)
```

Workers save only owned model-state tensors for this feature. Optimizer-state consolidation is out of scope unless the optimizer state can be proven local and key-stable. Shards preserve canonical state IDs and original `state_dict` keys; merge validates schema, graph fingerprint, plan ID, step, rank, owner, duplicate/missing keys, shapes, dtypes, and strict reload compatibility.

## Key Design Questions

**Q1: How are capture inputs obtained without `sample_inputs()`?**
By running the user's entrypoint under ShardGrid bootstrap and intercepting the first usable model invocation and batch in-process.

**Q2: How are execution node and parameter owner separated?**
Execution nodes come from export/FX graph values; state owners come from canonical parameter/buffer IDs plus original `state_dict` keys and parameter-use records.

**Q3: What if one parameter is used by multiple graph nodes?**
Represent uses separately from ownership. One owner writes/checkpoints the parameter; all consumers reference the same canonical state object. Unsupported write/update ambiguity blocks planning.

**Q4: How are shared/tied weights partitioned and checkpointed?**
They keep one canonical state object and one checkpoint owner. Partitions may read it through explicit dependencies. Unsafe cross-partition update semantics are unsupported for this feature.

**Q5: How do parameterless operations participate in partitioning?**
They are normal execution nodes with activation/memory/transfer cost and no state ownership.

**Q6: How are parameters with no independent forward module invocation counted?**
They appear in the state ownership layer and in parameter-use metadata even when no `call_module` execution node owns them.

**Q7: How does runtime obtain real training inputs?**
From the captured first-step call structure and subsequent ShardGrid-controlled batch handoff contract derived from the user's loader boundary.

**Q8: Who owns backward and optimizer?**
ShardGrid runtime owns distributed backward and optimizer execution for partitioned state after capture; the user script owns declaring optimizer/loss/scheduler intent.

**Q9: How is original optimizer semantics preserved?**
By preserving optimizer class, param groups, hyperparameters, AMP/scaler and accumulation boundaries where observable; otherwise fail before mutation.

**Q10: How does checkpoint preserve `state_dict` keys?**
Capture stores canonical ID to original `state_dict` key mapping; shards and merge use that mapping directly.

**Q11: What happens when export/FX/Dynamo capture fails?**
Return structured unsupported failures such as `GRAPH_BREAK_UNSUPPORTED`, `DYNAMIC_CONTROL_FLOW_UNSUPPORTED`, `CUSTOM_OP_UNSUPPORTED`, or `MODEL_CAPTURE_UNSUPPORTED`.

**Q12: When is a model unsupported?**
When ShardGrid cannot prove safe capture, full execution-node coverage, exact state ownership, dependency transfer, memory admission, optimizer semantics, or checkpoint merge safety before distributed mutation.

## Failure Taxonomy

| Code | Produced By | Propagation | CLI / Stress Behavior | Retry |
|------|-------------|-------------|-----------------------|-------|
| `MODEL_CAPTURE_UNSUPPORTED` | bootstrap/capture | `FailureRecord` capture stage | actionable unsupported diagnostic | no |
| `GRAPH_BREAK_UNSUPPORTED` | export/Dynamo/FX capture | capture stage | graph-break reason | no |
| `CUSTOM_OP_UNSUPPORTED` | export/fake/meta validation | capture stage | custom op name if known | no |
| `DYNAMIC_CONTROL_FLOW_UNSUPPORTED` | export/FX tracing | capture stage | control-flow reason | no |
| `PROFILE_FAILURE` | `build_model_profile()` path | profile stage | profile diagnostic | no unless infra |
| `PARTITION_FAILURE` | `partitioning.py` | partition stage | graph/state coverage reason | no |
| `PLAN_VALIDATION_FAILURE` | planning contracts/runtime consistency | validation stage | exact invariant failure | no |
| `NO_FEASIBLE_PLAN` | placement search exhausted | placement stage | candidate summary | maybe after resource change |
| `SEARCH_BUDGET_LIMIT` | bounded candidate search | placement stage | search budget exhausted | maybe |
| `MEMORY_REJECT` | one-batch memory probe | memory probe stage | normal candidate rejection | yes, next candidate |
| `FORMAL_TRAINING_OOM` | runtime training | training stage | bug/safety failure | no |
| `RESOURCE_CHANGED` | live resource refresh/probe | resource stage | replan if possible | yes |
| `NETWORK_FAILURE` | resource/network probe or SSH | network/launch stage | host/link diagnostic | yes |
| `RENDEZVOUS_FAILURE` | distributed startup | launch/runtime stage | rendezvous env/port diagnostic | yes |
| `PROCESS_LAUNCH_FAILURE` | `SSHLauncher` | launch stage | argv/env/remote log refs | yes if infra |
| `INFRA_FAILURE` | launcher/probe/system | infra stage | grouped as infra | yes |
| `RUNTIME_FAILURE` | distributed runtime | runtime stage | rank/stage/log refs | maybe |
| `CPU_PROCESS_SATURATION` | stress runner | stress stage | saturation evidence | no |
| `GPU_MEMORY_SATURATION` | stress runner | stress stage | memory saturation evidence | no |
| `TEST_LIMIT_REACHED` | stress runner | stress stage | test limit evidence | no |
| `SATURATION_NOT_PROVEN` | stress runner | stress stage | missing evidence | no |

Existing broad `FailureStage` values in `src/shardgrid/common/enums.py` stay as compatibility buckets; new codes provide the precise reason.

## Phased Migration

### Phase 0 - Characterization / Regression Safety

Purpose: pin current behavior before refactor.
Modules: CLI, `JobManager`, planner, runtime, checkpoint, stress script, zoo examples.
Keep: all existing `train`, automatic mode, generic DAG examples, exact-plan checks.
Add: characterization tests for current call chains and model-name coupling inventory.
Exit: old gates still run; current coupling locations are covered by tests.

### Phase 1 - Planner Semantic Repair

Purpose: separate execution graph, state ownership, logical partition, and placement.
Modules: `generic_graph.py`, `memory.py`, `partitioning.py`, `planning_contract.py`, `runtime/dag.py`, `checkpoint.py`.
Keep: legacy `ModelProfile` compatibility fields.
Add: stable execution-node IDs, state-object ownership, boundary values, shared/tied validation.
Exit: CPU planner tests pass for sequential, residual, CNN, UNet-like, dense, transformer, attention, encoder-decoder, multi-branch, shared module.

### Phase 2 - Production Model Decoupling

Purpose: remove model-name/zoo dependency from production planning/runtime/checkpoint.
Modules: `JobManager`, `examples/models/train_generic_dag.py`, `train_automatic_plan.py`, zoo catalog, checkpoint finalization.
Keep: zoo/stress as examples/tests/benchmarks.
Add: generic captured-context workload path.
Exit: production planning does not call `build_zoo_model()` or `make_zoo_sample()`.

### Phase 3 - Entrypoint Capture / Launch

Purpose: implement `shardgrid run ENTRYPOINT [ARGS...]`.
Modules: new CLI command, bootstrap/shim module, `JobManager`, `SSHLauncher`.
Keep: existing config-driven `train`.
Add: in-process first-step capture, args/kwargs preservation, diagnostics.
Exit: ordinary local PyTorch script captures model, batch, call structure, graph, and ownership without script edits.

### Phase 4 - Generic Runtime

Purpose: run exact planner output without rank-local rediscovery.
Modules: `runtime/dag.py`, `runtime/partition_graph.py`, launcher assignments.
Keep: `PLAN_RUNTIME_CONSISTENCY_CHECK=PASS`.
Add: runtime consumption of captured graph, ownership, runtime inputs, partition/placement IDs.
Exit: local runtime completes forward/backward/optimizer/checkpoint strict reload for non-zoo models.

### Phase 5 - Generic Checkpoint

Purpose: produce standard PyTorch model-state output.
Modules: `runtime/checkpoint.py`, `JobManager` finalization.
Keep: shard schema validation and duplicate/missing checks.
Add: model-name-free merged `state_dict`, strict load validation using captured original state structure.
Exit: `original_model.load_state_dict(torch.load(...), strict=True)` passes for supported runs.

### Phase 6 - Unified Failure Taxonomy / Stress Validation

Purpose: stop collapsing unrelated failures into `NO_FEASIBLE_PLAN`.
Modules: `common/enums.py`, `common/errors.py`, `jobs/models.py`, `JobManager`, `SSHLauncher`, stress script.
Keep: broad stage fields for compatibility.
Add: structured failure codes, retry policy, stress consumption, progress equality stall detection, checkpoint sampling, cleanup evidence.
Exit: CLI and stress report capture/graph/partition/placement/memory/launch/runtime/checkpoint failures distinctly.

### Phase 7 - Real Hardware Validation

Purpose: prove SSH-first production behavior.
Modules: resource manager, probes, placement, memory probe, launcher, runtime, checkpoint.
Keep: current fresh discovery and multi-job sharing semantics.
Add: validation matrix across single GPU, multi-GPU, multi-host, multi-job sharing, low-memory stress.
Exit: at least one non-zoo model completes real SSH multi-host training with merged reloadable state.

### Phase 8 - Compatibility Deprecation

Purpose: remove old production-only branches after generic path covers gates.
Modules: legacy automatic examples and model-specific checkpoint branches.
Keep: examples/tests/benchmarks.
Deprecate: production use of `minimal_sequential`, `hf_style`, `large_residual_transformer`, `generic_dag` as model factories.
Delete: only after release notes and tests prove generic replacement.
Exit: no production launch path depends on model catalog names.

## Backward Compatibility

- Existing zoo tests: keep as regression fixtures.
- Existing stress catalog: keep, but add generic-entrypoint fixtures and structured evidence.
- Existing CLI `train`: keep during migration; eventually document as legacy/config path.
- Existing automatic mode: keep until generic planning/runtime passes old gates.
- Existing Generic DAG runtime: keep as regression runtime while production moves to captured graph.
- Existing exact plan: preserve and strengthen with graph/ownership fingerprints.
- Existing checkpoint merge: preserve shard validation; replace model-specific reconstruction in production.
- Existing resource discovery: preserve as placement source of truth.
- Existing multi-job sharing: preserve; placement uses current free memory plus probe evidence.

## Testing Architecture

Layer 1 - CPU Planner Compatibility:
Sequential, residual, CNN, UNet-like, dense connection, transformer, attention, encoder-decoder, multi-branch, shared module. Validate capture, profile, partition, plan, ownership, and invariant failures.

Layer 2 - Local Runtime:
Forward, backward, optimizer step, checkpoint shard, merge, strict reload.

Layer 3 - Single GPU:
CUDA runtime, memory probe, formal training, OOM taxonomy.

Layer 4 - Multi GPU:
Partition placement, activation transfer, gradient transfer, progress evidence, checkpoint merge.

Layer 5 - Multi Host:
SSH, network probe, rendezvous, distributed execution, remote logs, worker cleanup.

Layer 6 - Multi Job Sharing:
Multiple independent jobs sharing GPUs through fresh free-memory/probe admission.

Layer 7 - Stress:
4G, 3G, 2G, 1G, 512M packing/saturation only after lower layers pass.

## Risks / Support Matrix

| Area | Status In This Feature | Handling |
|------|------------------------|----------|
| Dynamic Python control flow | Partial | Support exportable guarded paths; reject unsafe graph breaks. |
| Graph breaks | Partial | Structured diagnostics; no distributed mutation. |
| Custom autograd | Partial | Support only if capture/runtime semantics are provable. |
| Custom CUDA ops | Mostly unsupported | Require fake/meta/export support or reject. |
| Shared/tied weights | Partial | One canonical owner; unsafe writes rejected. |
| Module reuse | Partial | Execution uses node IDs, state uses canonical IDs. |
| Parameter reuse | Partial | Multiple uses, one owner; ambiguous optimizer semantics rejected. |
| HF generation-style modules | Future | Training-forward subset only; generation control flow out of scope. |
| Mixed precision | Partial | Preserve observable AMP/scaler intent when supported. |
| Gradient accumulation | Partial | Preserve detected accumulation boundary; unsupported hidden state rejected. |
| DDP/FSDP coexistence | Future | Reject nested distributed ownership for this feature. |
| `torch.compile` interaction | Partial | Capture before/around compile where possible; explain incompatibility. |
| DataLoader multiprocessing | Partial | First-step capture only; worker-side full loader replay is future. |
| Optimizer state checkpoint | Future | Model-state checkpoint is in scope; optimizer-state merge is out. |
| Process interception | Risk | Use explicit bootstrap and narrow hooks; avoid global monkeypatching where possible. |
| Remote object reconstruction | Risk | Runtime consumes serialized graph/plan/state, not arbitrary Python objects. |
| Security/import isolation | Risk | Run user code in configured Conda env; do not import untrusted code outside launch boundary. |

## Out Of Scope

- Full arbitrary Python transparency.
- Every custom CUDA or custom autograd operation.
- Optimizer-state consolidation across partitions.
- ZeRO/FSDP interoperability.
- Kubernetes, Volcano, HAMi, and GPU virtualization availability claims.
- Inference serving.
- Pipeline scheduling optimization beyond correctness-oriented partition/placement.
- Rewriting user projects into ShardGrid-specific model APIs.

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
└── checklists/
    └── requirements.md
```

Source modules affected by later implementation:

```text
src/shardgrid/cli/
src/shardgrid/control/
src/shardgrid/planner/
src/shardgrid/runtime/
src/shardgrid/launchers/
src/shardgrid/resources/
src/shardgrid/workers/
src/shardgrid/common/
src/shardgrid/jobs/
examples/models/
scripts/stress_dynamic_gpu_multi_job.py
```

## Complexity Tracking

No constitution violations recorded. The plan deliberately reuses PyTorch capture APIs, existing SSH launch, existing resource probes, existing placement search, existing memory probes, and existing checkpoint shard machinery instead of introducing parallel systems.
