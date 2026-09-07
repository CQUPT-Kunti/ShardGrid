# Tasks: Generic PyTorch Automation

**Input**: Design documents from `specs/002-generic-pytorch-automation/`

**Prerequisites**: `spec.md`, `plan.md`, `research.md`, `data-model.md`, `contracts/`, `quickstart.md`

**Rules**: This file is executable planning only. Do not modify `spec.md` or `plan.md` while implementing these tasks. Do not modify `specs/001-*`. Tests are required for every task because the feature spec includes explicit independent tests and safety criteria.

## Format

Each task starts with the required checklist line. Detailed metadata under each task is part of the task contract.

## Critical Path

```text
T001-T008 regression baseline
  -> T009-T021 planner semantic repair
  -> T022-T027 production model decoupling
  -> T028-T037 entrypoint capture
  -> T038-T044 exact generic runtime
  -> T045-T051 generic checkpoint
  -> T052-T060 failure taxonomy and stress validation
  -> T061-T066 real hardware acceptance
  -> T067-T070 final acceptance/deprecation gates
```

## Phase 0: Characterization / Regression Safety

**Purpose**: Pin current behavior before architecture changes.

- [ ] T001 [P] Add CLI registration characterization for existing `train` command and missing `run` command in `tests/unit/test_cli_registration.py`
  - Title: Characterize CLI command surface
  - Phase: Phase 0
  - Priority: P0 BLOCKING
  - Depends on: none
  - Files: `tests/unit/test_cli_registration.py`, `src/shardgrid/cli/app.py`, `src/shardgrid/cli/commands/train.py`
  - Goal: Prove current `train` registration remains stable before adding `run`.
  - Implementation Notes: Assert `register_train_command()` is still wired through `shardgrid.cli.app.main`; record expected absence of `run` as baseline.
  - Tests: `pytest tests/unit/test_cli_registration.py`
  - Acceptance Criteria: Existing CLI behavior is documented by tests without adding new CLI code.

- [ ] T002 [P] Add JobManager model-name coupling characterization tests in `tests/unit/test_job_manager_model_coupling.py`
  - Title: Inventory production model-type branches
  - Phase: Phase 0
  - Priority: P0 BLOCKING
  - Depends on: none
  - Files: `tests/unit/test_job_manager_model_coupling.py`, `src/shardgrid/control/job_manager.py`
  - Goal: Lock down current `_planner_workload()`, `_launch_command_for_assignment()`, and `_write_consolidated_model()` coupling points.
  - Implementation Notes: Use source inspection or focused unit seams to detect `training_config.model.type`, `build_zoo_model`, `make_zoo_sample`, and example runtime dispatch.
  - Tests: `pytest tests/unit/test_job_manager_model_coupling.py`
  - Acceptance Criteria: Tests fail once production coupling is removed unless updated to assert zero production dependency.

- [ ] T003 [P] Add CPU model-shape fixture suite in `tests/fixtures/generic_training_models.py`
  - Title: Create ordinary PyTorch model fixtures
  - Phase: Phase 0
  - Priority: P0 BLOCKING
  - Depends on: none
  - Files: `tests/fixtures/generic_training_models.py`
  - Goal: Provide non-zoo CPU fixtures for sequential, residual, CNN, transformer, attention, UNet-like, dense, multi-branch, shared module, and shared/tied parameter cases.
  - Implementation Notes: Fixtures must be normal PyTorch modules and batches; do not add ShardGrid model factories or user-facing sample APIs.
  - Tests: `python -m pytest tests/unit/test_generic_graph_ir.py tests/unit/test_model_profile_memory.py`
  - Acceptance Criteria: Fixtures import cleanly and can run a single CPU forward/backward step.

- [ ] T004 Add registration-order mismatch regression tests in `tests/unit/test_generic_graph_ir.py`
  - Title: Prove `named_modules()` is not execution order
  - Phase: Phase 0
  - Priority: P0 BLOCKING
  - Depends on: T003
  - Files: `tests/unit/test_generic_graph_ir.py`, `tests/fixtures/generic_training_models.py`
  - Goal: Reproduce cases where module registration order differs from forward/export/FX execution order.
  - Implementation Notes: Include residual, branch/merge, shared module, and functional operation models.
  - Tests: `pytest tests/unit/test_generic_graph_ir.py`
  - Acceptance Criteria: Tests expose the semantic mismatch the planner must stop depending on.

- [ ] T005 Add current `ModelProfile` ordering characterization in `tests/unit/test_model_profile_memory.py`
  - Title: Characterize module-slice memory profile behavior
  - Phase: Phase 0
  - Priority: P0 BLOCKING
  - Depends on: T003
  - Files: `tests/unit/test_model_profile_memory.py`, `src/shardgrid/planner/memory.py`
  - Goal: Document how `build_model_profile()` and `_iter_target_modules()` currently derive module order and memory estimates.
  - Implementation Notes: Verify parameterless nodes and state-owning modules without standalone execution nodes are not safely represented today.
  - Tests: `pytest tests/unit/test_model_profile_memory.py`
  - Acceptance Criteria: Baseline failures or expected-xfail cases clearly identify the model-profile semantic gap.

- [ ] T006 Add partitioning baseline tests for graph/order mismatch in `tests/unit/test_partition_candidates.py`
  - Title: Characterize partition boundary assumptions
  - Phase: Phase 0
  - Priority: P0 BLOCKING
  - Depends on: T003, T004
  - Files: `tests/unit/test_partition_candidates.py`, `src/shardgrid/planner/partitioning.py`
  - Goal: Capture current `_normalize_traced_order()` and `_ordered_modules_for_support()` behavior before repair.
  - Implementation Notes: Cover execution node without parameters, parameter owner without execution node, shared module, and multi-consumer values.
  - Tests: `pytest tests/unit/test_partition_candidates.py`
  - Acceptance Criteria: Tests show which planner invariants are missing today.

- [ ] T007 [P] Add checkpoint baseline tests for state key preservation in `tests/unit/test_checkpoint_generic_state.py`
  - Title: Characterize checkpoint shard and merge behavior
  - Phase: Phase 0
  - Priority: P0 BLOCKING
  - Depends on: T003
  - Files: `tests/unit/test_checkpoint_generic_state.py`, `src/shardgrid/runtime/checkpoint.py`, `src/shardgrid/control/job_manager.py`
  - Goal: Document current generic shard validation and model-specific finalization coupling.
  - Implementation Notes: Cover parameter keys, buffer keys, duplicate canonical IDs, missing keys, and strict reload expectations.
  - Tests: `pytest tests/unit/test_checkpoint_generic_state.py`
  - Acceptance Criteria: Current checkpoint behavior is pinned before generic merge changes.

- [ ] T008 Run Phase 0 regression gate and record command output in `specs/002-generic-pytorch-automation/tasks.md`
  - Title: Phase 0 gate
  - Phase: Phase 0
  - Priority: P0 BLOCKING
  - Depends on: T001, T002, T003, T004, T005, T006, T007
  - Files: `tests/unit/test_cli_registration.py`, `tests/unit/test_job_manager_model_coupling.py`, `tests/unit/test_generic_graph_ir.py`, `tests/unit/test_model_profile_memory.py`, `tests/unit/test_partition_candidates.py`, `tests/unit/test_checkpoint_generic_state.py`
  - Goal: Establish `REGRESSION_BASELINE=PASS`.
  - Implementation Notes: Do not proceed to planner changes until this gate is green or expected-xfail reasons are explicit.
  - Tests: `pytest tests/unit/test_cli_registration.py tests/unit/test_job_manager_model_coupling.py tests/unit/test_generic_graph_ir.py tests/unit/test_model_profile_memory.py tests/unit/test_partition_candidates.py tests/unit/test_checkpoint_generic_state.py`
  - Acceptance Criteria: `REGRESSION_BASELINE=PASS`.

## Phase 1: Planner Semantic Repair

**Purpose**: Separate execution graph, parameter ownership, buffer ownership, logical partition, and placement.

- [ ] T009 [P] [US2] Add state ownership dataclasses beside `GenericGraphIR` in `src/shardgrid/planner/generic_graph.py`
  - Title: Introduce explicit state ownership records
  - Phase: Phase 1
  - Priority: P0 BLOCKING
  - Depends on: T008
  - Files: `src/shardgrid/planner/generic_graph.py`, `tests/unit/test_generic_graph_ir.py`
  - Goal: Represent parameters and buffers independently from execution nodes.
  - Implementation Notes: Include canonical state ID, kind, original `state_dict` key, shape, dtype, requires-grad, storage/shared group, and use sites.
  - Tests: `pytest tests/unit/test_generic_graph_ir.py`
  - Acceptance Criteria: Parameter owners, buffer owners, and use records exist without relying on `named_modules()` order.

- [ ] T010 [P] [US2] Add execution-node value metadata to `GenericGraphIR` in `src/shardgrid/planner/generic_graph.py`
  - Title: Stabilize execution graph IDs
  - Phase: Phase 1
  - Priority: P0 BLOCKING
  - Depends on: T008
  - Files: `src/shardgrid/planner/generic_graph.py`, `tests/unit/test_generic_graph_ir.py`
  - Goal: Make execution nodes and graph values addressable by stable IDs and producer/consumer edges.
  - Implementation Notes: Preserve `CanonicalGraphIR = GenericGraphIR` compatibility while extending fields.
  - Tests: `pytest tests/unit/test_generic_graph_ir.py`
  - Acceptance Criteria: Graph tests verify node/value dependencies independent of module registration order.

- [ ] T011 [US2] Map FX/export parameter and buffer uses into ownership records in `src/shardgrid/planner/generic_graph.py`
  - Title: Build state-use mapping from captured graph
  - Phase: Phase 1
  - Priority: P0 BLOCKING
  - Depends on: T009, T010
  - Files: `src/shardgrid/planner/generic_graph.py`, `tests/unit/test_generic_graph_ir.py`
  - Goal: Connect graph nodes to state objects without making graph nodes own state by default.
  - Implementation Notes: Cover functional/fused operations, module reuse, parameter reuse, and buffer reads.
  - Tests: `pytest tests/unit/test_generic_graph_ir.py`
  - Acceptance Criteria: Tests pass for parameter owner without independent execution node and execution node without parameters.

- [ ] T012 [US2] Extend `ModelProfile` accounting in `src/shardgrid/engines/models.py` and `src/shardgrid/planner/memory.py`
  - Title: Split execution cost from state cost
  - Phase: Phase 1
  - Priority: P0 BLOCKING
  - Depends on: T009, T010
  - Files: `src/shardgrid/engines/models.py`, `src/shardgrid/planner/memory.py`, `tests/unit/test_model_profile_memory.py`
  - Goal: Track execution-node activation cost, graph-value cost, and state-object memory separately.
  - Implementation Notes: Keep legacy fields for old tests; `named_modules()` may supply metadata but not partition order.
  - Tests: `pytest tests/unit/test_model_profile_memory.py`
  - Acceptance Criteria: Memory tests cover parameterless nodes, buffers, and state-owning modules with no standalone execution node.

- [ ] T013 [US2] Update logical partition models in `src/shardgrid/planner/planning_contract.py`
  - Title: Make logical partition coverage explicit
  - Phase: Phase 1
  - Priority: P0 BLOCKING
  - Depends on: T009, T010
  - Files: `src/shardgrid/planner/planning_contract.py`, `tests/unit/test_planning_contract.py`
  - Goal: Store execution node IDs, input/output value IDs, owned state IDs, read-only state IDs, and validation evidence.
  - Implementation Notes: Preserve existing constructor compatibility where current tests rely on it.
  - Tests: `pytest tests/unit/test_planning_contract.py`
  - Acceptance Criteria: Logical partitions no longer need one module-order list to describe nodes, state, and placement.

- [ ] T014 [US2] Update partition candidate generation in `src/shardgrid/planner/partitioning.py`
  - Title: Partition over execution graph dependencies
  - Phase: Phase 1
  - Priority: P0 BLOCKING
  - Depends on: T011, T012, T013
  - Files: `src/shardgrid/planner/partitioning.py`, `tests/unit/test_partition_candidates.py`
  - Goal: Generate candidates from execution-node topology and boundary values.
  - Implementation Notes: Do not implement `ordered_names = call_order`; use graph nodes, values, and ownership metadata.
  - Tests: `pytest tests/unit/test_partition_candidates.py`
  - Acceptance Criteria: Residual, branch/merge, attention, dense, and shared-module fixtures partition by dependency order.

- [ ] T015 [US2] Update partition validation in `src/shardgrid/planner/partitioning.py`
  - Title: Validate graph and ownership coverage
  - Phase: Phase 1
  - Priority: P0 BLOCKING
  - Depends on: T014
  - Files: `src/shardgrid/planner/partitioning.py`, `tests/unit/test_partition_candidates.py`
  - Goal: Enforce exact node coverage, exact state ownership, boundary value coverage, and shared/tied safety.
  - Implementation Notes: Return structured validation failures instead of collapsing semantic errors.
  - Tests: `pytest tests/unit/test_partition_candidates.py`
  - Acceptance Criteria: Duplicate, missing, ambiguous shared, and missing-boundary cases fail with explicit reasons.

- [ ] T016 [US2] Update planning validation in `src/shardgrid/planner/planning_contract.py`
  - Title: Validate final plan invariants
  - Phase: Phase 1
  - Priority: P0 BLOCKING
  - Depends on: T013, T015
  - Files: `src/shardgrid/planner/planning_contract.py`, `tests/unit/test_planning_contract.py`
  - Goal: Check planner output before placement/runtime can consume it.
  - Implementation Notes: Include fingerprint inputs for graph, ownership, logical partitions, and placement.
  - Tests: `pytest tests/unit/test_planning_contract.py`
  - Acceptance Criteria: `PLAN_VALIDATION_FAILURE` is produced for exact-plan invariant violations.

- [ ] T017 [US2] Update placement memory consumption in `src/shardgrid/planner/placement.py`
  - Title: Place using graph/state memory model
  - Phase: Phase 1
  - Priority: P1
  - Depends on: T012, T016
  - Files: `src/shardgrid/planner/placement.py`, `tests/unit/test_joint_partition_placement.py`
  - Goal: Use fresh worker/GPU data with the new partition memory fields.
  - Implementation Notes: Preserve `_usable_memory_bytes()` behavior and bounded candidate search.
  - Tests: `pytest tests/unit/test_joint_partition_placement.py`
  - Acceptance Criteria: Placement continues to use current free memory and does not add fixed reserve/headroom admission.

- [ ] T018 [US2] Update worker ownership compilation in `src/shardgrid/runtime/dag.py`
  - Title: Compile worker ownership from explicit state IDs
  - Phase: Phase 1
  - Priority: P0 BLOCKING
  - Depends on: T013, T016
  - Files: `src/shardgrid/runtime/dag.py`, `tests/unit/test_dag_runtime.py`
  - Goal: Materialize worker ownership from logical partitions and state ownership, not module slices.
  - Implementation Notes: Preserve rank/worker/GPU placement from planner output exactly.
  - Tests: `pytest tests/unit/test_dag_runtime.py`
  - Acceptance Criteria: Worker ownership includes exact parameter IDs, buffer IDs, logical partition IDs, and runtime edges.

- [ ] T019 [US2] Update runtime partition graph mapping in `src/shardgrid/runtime/partition_graph.py`
  - Title: Map runtime partitions by stable graph IDs
  - Phase: Phase 1
  - Priority: P1
  - Depends on: T010, T014
  - Files: `src/shardgrid/runtime/partition_graph.py`, `tests/unit/test_dag_runtime.py`
  - Goal: Stop relying on backend FX node count alignment alone.
  - Implementation Notes: Reject mismatches as `PLAN_VALIDATION_FAILURE`.
  - Tests: `pytest tests/unit/test_dag_runtime.py`
  - Acceptance Criteria: Runtime partition extraction verifies stable execution-node and value IDs.

- [ ] T020 [P] [US2] Add planner compatibility matrix tests in `tests/unit/test_generic_planner_compatibility.py`
  - Title: Cover CPU planner model families
  - Phase: Phase 1
  - Priority: P0 BLOCKING
  - Depends on: T003
  - Files: `tests/unit/test_generic_planner_compatibility.py`, `tests/fixtures/generic_training_models.py`
  - Goal: Validate CPU capture/profile/partition/plan for all model families listed in the plan.
  - Implementation Notes: Include unsupported expected failures for unsafe shared writes or dynamic behavior.
  - Tests: `pytest tests/unit/test_generic_planner_compatibility.py`
  - Acceptance Criteria: Sequential, residual, CNN, UNet-like, dense, transformer, attention, encoder-decoder, multi-branch, and shared module cases are covered.

- [ ] T021 [US2] Run Phase 1 planner gate in `tests/unit/test_generic_planner_compatibility.py`
  - Title: Phase 1 gate
  - Phase: Phase 1
  - Priority: P0 BLOCKING
  - Depends on: T009, T010, T011, T012, T013, T014, T015, T016, T017, T018, T019, T020
  - Files: `tests/unit/test_generic_planner_compatibility.py`, `tests/unit/test_partition_candidates.py`, `tests/unit/test_planning_contract.py`
  - Goal: Establish `PLANNER_COMPATIBILITY=PASS`.
  - Implementation Notes: Do not start production decoupling until planner invariants are green.
  - Tests: `pytest tests/unit/test_generic_planner_compatibility.py tests/unit/test_partition_candidates.py tests/unit/test_planning_contract.py tests/unit/test_dag_runtime.py`
  - Acceptance Criteria: `PLANNER_COMPATIBILITY=PASS`.

## Phase 2: Production Model Decoupling

**Purpose**: Remove model-name and zoo dependency from production planning, runtime launch, and checkpoint finalization.

- [ ] T022 [US2] Add generic captured workload input path to `src/shardgrid/control/job_manager.py`
  - Title: Replace production planner workload source
  - Phase: Phase 2
  - Priority: P0 BLOCKING
  - Depends on: T021
  - Files: `src/shardgrid/control/job_manager.py`, `tests/unit/test_job_manager_model_coupling.py`
  - Goal: Let `JobManager` plan from captured context instead of `_planner_workload()` model-type branches.
  - Implementation Notes: Keep legacy config branches for compatibility; new production path must not call `build_zoo_model()` or `make_zoo_sample()`.
  - Tests: `pytest tests/unit/test_job_manager_model_coupling.py`
  - Acceptance Criteria: Generic production planning path has `PRODUCTION_ZOO_DEPENDENCY=0`.

- [ ] T023 [US2] Route automatic planning to captured graph/profile in `src/shardgrid/control/job_manager.py`
  - Title: Feed generic planner from captured context
  - Phase: Phase 2
  - Priority: P0 BLOCKING
  - Depends on: T022
  - Files: `src/shardgrid/control/job_manager.py`, `tests/unit/test_job_manager_model_coupling.py`, `tests/unit/test_generic_planner_compatibility.py`
  - Goal: Replace model-type production branches with `CapturedTrainingContext`, `GenericGraphIR`, ownership model, and `ModelProfile`.
  - Implementation Notes: Preserve old `_planner_workload()` only as legacy/example path.
  - Tests: `pytest tests/unit/test_job_manager_model_coupling.py tests/unit/test_generic_planner_compatibility.py`
  - Acceptance Criteria: Non-zoo models reach planning without model names or catalog entries.

- [ ] T024 [US2] Replace example-script launch dispatch in `src/shardgrid/control/job_manager.py`
  - Title: Launch generic runtime bootstrap from execution plan
  - Phase: Phase 2
  - Priority: P0 BLOCKING
  - Depends on: T022, T023
  - Files: `src/shardgrid/control/job_manager.py`, `tests/unit/test_memory_probe_launch.py`
  - Goal: Stop production `_launch_command_for_assignment()` from choosing `examples/models/train_generic_dag.py`.
  - Implementation Notes: Keep example dispatch available only under explicit legacy/example configs.
  - Tests: `pytest tests/unit/test_memory_probe_launch.py`
  - Acceptance Criteria: New launch command points at ShardGrid runtime bootstrap and carries plan/context artifact paths.

- [ ] T025 [US4] Keep zoo runtime as example-only in `examples/models/train_generic_dag.py`
  - Title: Preserve generic DAG example without production dependency
  - Phase: Phase 2
  - Priority: P2
  - Depends on: T022
  - Files: `examples/models/train_generic_dag.py`, `tests/unit/test_train_automatic_plan.py`, `tests/unit/test_job_manager_model_coupling.py`
  - Goal: Keep existing generic DAG tests while proving production code no longer invokes it.
  - Implementation Notes: Add comments/warnings only where tests require; no user-facing model API.
  - Tests: `pytest tests/unit/test_train_automatic_plan.py tests/unit/test_job_manager_model_coupling.py`
  - Acceptance Criteria: Example runtime remains runnable as validation asset.

- [ ] T026 [US4] Scope legacy automatic model-specific runtime in `examples/models/train_automatic_plan.py`
  - Title: Keep legacy automatic path behind compatibility tests
  - Phase: Phase 2
  - Priority: P2
  - Depends on: T022
  - Files: `examples/models/train_automatic_plan.py`, `tests/unit/test_train_automatic_plan.py`
  - Goal: Preserve current examples while preventing them from defining production generic behavior.
  - Implementation Notes: Add deprecation markers or compatibility labels without deleting behavior.
  - Tests: `pytest tests/unit/test_train_automatic_plan.py`
  - Acceptance Criteria: Existing automatic examples still pass and are clearly scoped as legacy/example.

- [ ] T027 [US2] Run Phase 2 decoupling gate in `tests/unit/test_job_manager_model_coupling.py`
  - Title: Phase 2 gate
  - Phase: Phase 2
  - Priority: P0 BLOCKING
  - Depends on: T022, T023, T024, T025, T026
  - Files: `tests/unit/test_job_manager_model_coupling.py`, `src/shardgrid/control/job_manager.py`
  - Goal: Establish `PRODUCTION_ZOO_DEPENDENCY=0`.
  - Implementation Notes: Verify production path avoids `model.type`, `zoo_model`, `build_zoo_model()`, and `make_zoo_sample()` as workload source.
  - Tests: `pytest tests/unit/test_job_manager_model_coupling.py`
  - Acceptance Criteria: `PRODUCTION_ZOO_DEPENDENCY=0`.

## Phase 3: Entrypoint Capture / Launch

**Purpose**: Implement `shardgrid run ENTRYPOINT [ARGS...]` with internal first-step capture.

- [ ] T028 [P] [US1] Add CLI contract tests for `shardgrid run` in `tests/integration/test_run_cli.py`
  - Title: Test run command parsing
  - Phase: Phase 3
  - Priority: P0 BLOCKING
  - Depends on: T027
  - Files: `tests/integration/test_run_cli.py`, `src/shardgrid/cli/app.py`, `src/shardgrid/cli/commands/run.py`
  - Goal: Verify argv passthrough, `--config`, `--dry-run`, `--json`, exit codes, and failure display.
  - Implementation Notes: Tests must prove user script args are preserved exactly.
  - Tests: `pytest tests/integration/test_run_cli.py`
  - Acceptance Criteria: CLI contract from `contracts/cli.md` is covered before implementation.

- [ ] T029 [US1] Register `shardgrid run` in `src/shardgrid/cli/app.py` and `src/shardgrid/cli/commands/run.py`
  - Title: Add run command shell
  - Phase: Phase 3
  - Priority: P0 BLOCKING
  - Depends on: T028
  - Files: `src/shardgrid/cli/app.py`, `src/shardgrid/cli/commands/run.py`, `tests/integration/test_run_cli.py`
  - Goal: Expose the target user command without changing existing `train`.
  - Implementation Notes: Pass a `TrainingEntrypoint`-style object to control code; do not ask users for worker/GPU/stage counts.
  - Tests: `pytest tests/integration/test_run_cli.py tests/unit/test_cli_registration.py`
  - Acceptance Criteria: `shardgrid run --config cluster.yaml train.py --config config.yaml` parses as intended.

- [ ] T030 [P] [US1] Add ordinary training script fixtures in `tests/fixtures/ordinary_training_scripts/`
  - Title: Create normal user script fixtures
  - Phase: Phase 3
  - Priority: P0 BLOCKING
  - Depends on: T027
  - Files: `tests/fixtures/ordinary_training_scripts/`
  - Goal: Provide scripts with normal model, optimizer, dataloader, loss, backward, optimizer step, and checkpoint code.
  - Implementation Notes: Include positional args, kwargs, tuple/list/dict/nested batches, masks, labels, HF-style mappings, and multiple outputs.
  - Tests: `pytest tests/integration/test_run_cli.py`
  - Acceptance Criteria: Fixtures run directly with `python train.py ...` without importing ShardGrid user APIs.

- [ ] T031 [P] [US1] Add capture contract tests in `tests/integration/test_entrypoint_capture.py`
  - Title: Test first-step capture contract
  - Phase: Phase 3
  - Priority: P0 BLOCKING
  - Depends on: T030
  - Files: `tests/integration/test_entrypoint_capture.py`, `tests/fixtures/ordinary_training_scripts/`
  - Goal: Verify model instance, batch, args/kwargs, tensor metadata, optimizer, loss/backward boundary, and state keys are captured.
  - Implementation Notes: Tests should cover structured input shapes, not only `model(x)`.
  - Tests: `pytest tests/integration/test_entrypoint_capture.py`
  - Acceptance Criteria: Capture artifacts match `contracts/capture.md`.

- [ ] T032 [US1] Implement internal capture bootstrap in `src/shardgrid/bootstrap/runner.py`
  - Title: Execute user entrypoint under ShardGrid bootstrap
  - Phase: Phase 3
  - Priority: P0 BLOCKING
  - Depends on: T029, T031
  - Files: `src/shardgrid/bootstrap/runner.py`, `tests/integration/test_entrypoint_capture.py`
  - Goal: Start the user script in-process with preserved argv, cwd, and environment.
  - Implementation Notes: Use internal hooks only; do not require user-side ShardGrid model factory, sample-builder, or wrapper APIs.
  - Tests: `pytest tests/integration/test_entrypoint_capture.py`
  - Acceptance Criteria: User script can be launched and capture stops before distributed mutation on dry-run.

- [ ] T033 [US1] Add capture context serialization in `src/shardgrid/common/serialization.py`
  - Title: Persist captured training context
  - Phase: Phase 3
  - Priority: P0
  - Depends on: T031, T032
  - Files: `src/shardgrid/common/serialization.py`, `src/shardgrid/artifacts/snapshot.py`, `tests/integration/test_entrypoint_capture.py`
  - Goal: Write `plan/capture-context.json` and supporting tensor metadata into job snapshots.
  - Implementation Notes: Serialize metadata and artifact references, not arbitrary live Python objects.
  - Tests: `pytest tests/integration/test_entrypoint_capture.py tests/integration/test_code_snapshot.py`
  - Acceptance Criteria: Capture artifacts are deterministic enough for planner/runtime loading.

- [ ] T034 [US1] Integrate capture context loading into `src/shardgrid/control/job_manager.py`
  - Title: Hand captured context to control plane
  - Phase: Phase 3
  - Priority: P0 BLOCKING
  - Depends on: T022, T033
  - Files: `src/shardgrid/control/job_manager.py`, `tests/integration/test_entrypoint_capture.py`
  - Goal: Let `JobManager` run capture-first planning for `shardgrid run`.
  - Implementation Notes: Existing `run(config_path, dry_run=...)` stays intact; add a minimal new entrypoint path.
  - Tests: `pytest tests/integration/test_entrypoint_capture.py tests/integration/test_train_orchestration.py`
  - Acceptance Criteria: Captured context reaches graph/profile/planner code without model-name fallback.

- [ ] T035 [US1] Add export/FX fallback diagnostics in `src/shardgrid/planner/generic_graph.py`
  - Title: Emit capture backend failures
  - Phase: Phase 3
  - Priority: P1
  - Depends on: T011, T034
  - Files: `src/shardgrid/planner/generic_graph.py`, `tests/integration/test_entrypoint_capture.py`
  - Goal: Produce `GRAPH_BREAK_UNSUPPORTED`, `CUSTOM_OP_UNSUPPORTED`, and `DYNAMIC_CONTROL_FLOW_UNSUPPORTED` diagnostics.
  - Implementation Notes: Prefer `torch.export`, fall back to FX, and stop safely when neither can prove graph semantics.
  - Tests: `pytest tests/integration/test_entrypoint_capture.py tests/unit/test_generic_graph_ir.py`
  - Acceptance Criteria: Unsupported capture failures are structured and happen before distributed mutation.

- [ ] T036 [US1] Add optimizer/lifecycle capture validation in `src/shardgrid/bootstrap/runner.py`
  - Title: Preserve training lifecycle boundary
  - Phase: Phase 3
  - Priority: P1
  - Depends on: T032, T033
  - Files: `src/shardgrid/bootstrap/runner.py`, `tests/integration/test_entrypoint_capture.py`
  - Goal: Capture optimizer class, param groups, hyperparameters, gradient accumulation, AMP/scaler, scheduler boundary, and checkpoint intent when observable.
  - Implementation Notes: Return unsupported failure for hidden mutation or unmappable optimizer semantics.
  - Tests: `pytest tests/integration/test_entrypoint_capture.py`
  - Acceptance Criteria: Runtime has enough lifecycle metadata to preserve or reject optimizer semantics.

- [ ] T037 [US1] Run Phase 3 capture gate in `tests/integration/test_entrypoint_capture.py`
  - Title: Phase 3 gate
  - Phase: Phase 3
  - Priority: P0 BLOCKING
  - Depends on: T028, T029, T030, T031, T032, T033, T034, T035, T036
  - Files: `tests/integration/test_run_cli.py`, `tests/integration/test_entrypoint_capture.py`
  - Goal: Establish `ENTRYPOINT_CAPTURE=PASS`.
  - Implementation Notes: Verify ordinary scripts need zero ShardGrid model changes.
  - Tests: `pytest tests/integration/test_run_cli.py tests/integration/test_entrypoint_capture.py`
  - Acceptance Criteria: `ENTRYPOINT_CAPTURE=PASS`.

## Phase 4: Generic Runtime

**Purpose**: Execute the exact planner output without runtime repartition, round-robin placement, or model-name reconstruction.

- [ ] T038 [P] [US3] Add exact-plan runtime tests in `tests/unit/test_runtime_exact_plan.py`
  - Title: Test runtime consumes selected plan
  - Phase: Phase 4
  - Priority: P0 BLOCKING
  - Depends on: T037
  - Files: `tests/unit/test_runtime_exact_plan.py`, `src/shardgrid/runtime/dag.py`, `src/shardgrid/runtime/partition_graph.py`
  - Goal: Prove runtime uses planner logical partitions, worker ownership, parameter ownership, and placement exactly.
  - Implementation Notes: Assert no runtime repartition, round-robin, or placement recomputation.
  - Tests: `pytest tests/unit/test_runtime_exact_plan.py`
  - Acceptance Criteria: `PLAN_RUNTIME_CONSISTENCY_CHECK=PASS` for unit runtime fixtures.

- [ ] T039 [US3] Update `compile_runtime_plan()` in `src/shardgrid/runtime/dag.py`
  - Title: Compile generic runtime plan from graph/ownership/placement
  - Phase: Phase 4
  - Priority: P0 BLOCKING
  - Depends on: T018, T019, T038
  - Files: `src/shardgrid/runtime/dag.py`, `tests/unit/test_runtime_exact_plan.py`
  - Goal: Consume captured graph, logical partitions, state ownership, and exact placement.
  - Implementation Notes: Runtime must trust planner IDs and reject fingerprint mismatches.
  - Tests: `pytest tests/unit/test_runtime_exact_plan.py tests/unit/test_dag_runtime.py`
  - Acceptance Criteria: Runtime plan carries exact partition, owner, placement, and edge metadata.

- [ ] T040 [US3] Update worker materialization in `src/shardgrid/runtime/dag.py`
  - Title: Materialize only owned state on workers
  - Phase: Phase 4
  - Priority: P0 BLOCKING
  - Depends on: T039
  - Files: `src/shardgrid/runtime/dag.py`, `tests/unit/test_runtime_exact_plan.py`
  - Goal: Workers materialize only owned real parameters/buffers while non-owned skeletons remain meta/fake where safe.
  - Implementation Notes: Cover shared/tied owner, read-only uses, and parameterless execution nodes.
  - Tests: `pytest tests/unit/test_runtime_exact_plan.py`
  - Acceptance Criteria: Worker materialized parameter/buffer sets match `WorkerOwnershipPlan`.

- [ ] T041 [US3] Add local runtime integration tests in `tests/integration/test_generic_runtime_local.py`
  - Title: Test local generic training step
  - Phase: Phase 4
  - Priority: P0 BLOCKING
  - Depends on: T039, T040
  - Files: `tests/integration/test_generic_runtime_local.py`, `tests/fixtures/ordinary_training_scripts/`
  - Goal: Verify forward, backward, optimizer step, progress evidence, and checkpoint shard on CPU/local runtime.
  - Implementation Notes: Use non-zoo ordinary script fixtures.
  - Tests: `pytest tests/integration/test_generic_runtime_local.py`
  - Acceptance Criteria: Local runtime completes one supported generic step.

- [ ] T042 [US3] Integrate generic runtime launch command in `src/shardgrid/launchers/ssh.py`
  - Title: Launch runtime bootstrap with exact artifacts
  - Phase: Phase 4
  - Priority: P1
  - Depends on: T024, T039
  - Files: `src/shardgrid/launchers/ssh.py`, `tests/contract/test_ssh_launcher.py`, `tests/unit/test_memory_probe_launch.py`
  - Goal: Pass plan/context artifact paths through SSH launch without changing rank/world/CUDA env behavior.
  - Implementation Notes: Preserve `_launch_argv()` and `_launch_env()` compatibility.
  - Tests: `pytest tests/contract/test_ssh_launcher.py tests/unit/test_memory_probe_launch.py`
  - Acceptance Criteria: SSH launch command references generic runtime bootstrap and exact plan artifacts.

- [ ] T043 [US3] Preserve memory probe candidate flow for generic runtime in `src/shardgrid/control/job_manager.py`
  - Title: Keep estimate-calibration-probe admission chain
  - Phase: Phase 4
  - Priority: P0 BLOCKING
  - Depends on: T017, T042
  - Files: `src/shardgrid/control/job_manager.py`, `tests/unit/test_memory_probe_launch.py`, `tests/unit/test_job_manager_live_probe.py`
  - Goal: Candidate probe failures cleanup and continue; formal training OOM remains an error.
  - Implementation Notes: Do not add a second fixed reserve/headroom admission path.
  - Tests: `pytest tests/unit/test_memory_probe_launch.py tests/unit/test_job_manager_live_probe.py`
  - Acceptance Criteria: `MEMORY_REJECT` rejects candidates, not the whole job, until search is exhausted.

- [ ] T044 [US3] Run Phase 4 runtime gate in `tests/integration/test_generic_runtime_local.py`
  - Title: Phase 4 gate
  - Phase: Phase 4
  - Priority: P0 BLOCKING
  - Depends on: T038, T039, T040, T041, T042, T043
  - Files: `tests/unit/test_runtime_exact_plan.py`, `tests/integration/test_generic_runtime_local.py`
  - Goal: Establish `GENERIC_RUNTIME_LOCAL=PASS`.
  - Implementation Notes: Check runtime exact-plan evidence before checkpoint genericization.
  - Tests: `pytest tests/unit/test_runtime_exact_plan.py tests/integration/test_generic_runtime_local.py`
  - Acceptance Criteria: `GENERIC_RUNTIME_LOCAL=PASS` and `PLAN_RUNTIME_CONSISTENCY_CHECK=PASS`.

## Phase 5: Generic Checkpoint

**Purpose**: Produce a model-name-free standard PyTorch `state_dict` and strict reload validation.

- [ ] T045 [P] [US3] Add generic checkpoint contract tests in `tests/unit/test_checkpoint_generic_state.py`
  - Title: Test generic checkpoint contract
  - Phase: Phase 5
  - Priority: P0 BLOCKING
  - Depends on: T044
  - Files: `tests/unit/test_checkpoint_generic_state.py`, `src/shardgrid/runtime/checkpoint.py`
  - Goal: Cover shard schema, original keys, canonical IDs, duplicate keys, missing keys, buffers, and shape/dtype mismatch.
  - Implementation Notes: Include shared/tied parameter fixtures and strict reload assertion.
  - Tests: `pytest tests/unit/test_checkpoint_generic_state.py`
  - Acceptance Criteria: Contract failures are explicit checkpoint-stage failures.

- [ ] T046 [US3] Extend worker shard save in `src/shardgrid/runtime/checkpoint.py`
  - Title: Save original state keys in worker shards
  - Phase: Phase 5
  - Priority: P0 BLOCKING
  - Depends on: T045
  - Files: `src/shardgrid/runtime/checkpoint.py`, `tests/unit/test_checkpoint_generic_state.py`
  - Goal: Preserve parameter and buffer `state_dict` keys with canonical state IDs.
  - Implementation Notes: Keep schema versioning and include owner/rank/worker/step evidence.
  - Tests: `pytest tests/unit/test_checkpoint_generic_state.py`
  - Acceptance Criteria: Shards contain enough model-state metadata for architecture-free merge.

- [ ] T047 [US3] Update checkpoint merge in `src/shardgrid/runtime/checkpoint.py`
  - Title: Merge shards into standard model state
  - Phase: Phase 5
  - Priority: P0 BLOCKING
  - Depends on: T046
  - Files: `src/shardgrid/runtime/checkpoint.py`, `tests/unit/test_checkpoint_generic_state.py`
  - Goal: Produce `model-state.pt` keyed exactly like the captured original `state_dict`.
  - Implementation Notes: Reject duplicate canonical IDs, duplicate keys, missing state, stale graph fingerprint, and plan mismatch.
  - Tests: `pytest tests/unit/test_checkpoint_generic_state.py`
  - Acceptance Criteria: `torch.load("model-state.pt")` returns a strict-loadable model state.

- [ ] T048 [US3] Remove model-name reconstruction from production finalization in `src/shardgrid/control/job_manager.py`
  - Title: Finalize checkpoints without zoo/model-specific rebuilds
  - Phase: Phase 5
  - Priority: P0 BLOCKING
  - Depends on: T047
  - Files: `src/shardgrid/control/job_manager.py`, `tests/unit/test_checkpoint_generic_state.py`, `tests/unit/test_job_manager_model_coupling.py`
  - Goal: Replace production `_write_consolidated_model()` branches with generic merge/strict validation.
  - Implementation Notes: Keep legacy reconstruction only for compatibility/example path until Phase 8.
  - Tests: `pytest tests/unit/test_checkpoint_generic_state.py tests/unit/test_job_manager_model_coupling.py`
  - Acceptance Criteria: Production checkpoint finalization does not call zoo builders or model-type constructors.

- [ ] T049 [US3] Add strict reload integration tests in `tests/integration/test_generic_checkpoint_reload.py`
  - Title: Verify final model state reloads
  - Phase: Phase 5
  - Priority: P0 BLOCKING
  - Depends on: T047, T048
  - Files: `tests/integration/test_generic_checkpoint_reload.py`, `tests/fixtures/ordinary_training_scripts/`
  - Goal: Verify `original_model.load_state_dict(torch.load("model-state.pt"), strict=True)` succeeds for supported non-zoo models.
  - Implementation Notes: Optimizer state merge remains out of scope and must be asserted/documented as absent.
  - Tests: `pytest tests/integration/test_generic_checkpoint_reload.py`
  - Acceptance Criteria: `STANDARD_STATE_DICT_STRICT_LOAD=PASS`.

- [ ] T050 [US3] Add partial shard failure tests in `tests/unit/test_checkpoint_generic_state.py`
  - Title: Reject unsafe partial checkpoint bundles
  - Phase: Phase 5
  - Priority: P1
  - Depends on: T047
  - Files: `tests/unit/test_checkpoint_generic_state.py`, `src/shardgrid/runtime/checkpoint.py`
  - Goal: Reject partial, duplicate, stale, mismatched, or missing shard evidence.
  - Implementation Notes: Failure code must identify checkpoint stage.
  - Tests: `pytest tests/unit/test_checkpoint_generic_state.py`
  - Acceptance Criteria: Unsafe merges fail before producing a misleading model-state artifact.

- [ ] T051 [US3] Run Phase 5 checkpoint gate in `tests/integration/test_generic_checkpoint_reload.py`
  - Title: Phase 5 gate
  - Phase: Phase 5
  - Priority: P0 BLOCKING
  - Depends on: T045, T046, T047, T048, T049, T050
  - Files: `tests/unit/test_checkpoint_generic_state.py`, `tests/integration/test_generic_checkpoint_reload.py`
  - Goal: Establish `STANDARD_STATE_DICT_STRICT_LOAD=PASS`.
  - Implementation Notes: Do not start hardware validation before generic checkpoint passes locally.
  - Tests: `pytest tests/unit/test_checkpoint_generic_state.py tests/integration/test_generic_checkpoint_reload.py`
  - Acceptance Criteria: `STANDARD_STATE_DICT_STRICT_LOAD=PASS`.

## Phase 6: Unified Failure Taxonomy / Stress Validation

**Purpose**: Preserve failure type from producer to CLI and stress runner; repair progress/saturation evidence.

- [ ] T052 [P] [US1] Add structured failure-code enums in `src/shardgrid/common/enums.py`
  - Title: Define failure taxonomy
  - Phase: Phase 6
  - Priority: P0 BLOCKING
  - Depends on: T051
  - Files: `src/shardgrid/common/enums.py`, `tests/contract/test_failure_records.py`
  - Goal: Add capture, graph, profile, partition, placement, memory, launch, runtime, checkpoint, and stress codes.
  - Implementation Notes: Preserve existing broad `FailureStage` values.
  - Tests: `pytest tests/contract/test_failure_records.py`
  - Acceptance Criteria: All codes in `contracts/failures.md` are representable.

- [ ] T053 [US1] Extend failure records in `src/shardgrid/jobs/models.py` and `src/shardgrid/common/errors.py`
  - Title: Persist structured failure metadata
  - Phase: Phase 6
  - Priority: P0 BLOCKING
  - Depends on: T052
  - Files: `src/shardgrid/jobs/models.py`, `src/shardgrid/common/errors.py`, `tests/contract/test_failure_records.py`
  - Goal: Store code, stage, retryable flag, producer, worker/rank/GPU context, and log refs.
  - Implementation Notes: Keep compatibility for old failure records.
  - Tests: `pytest tests/contract/test_failure_records.py tests/unit/test_errors_logging.py`
  - Acceptance Criteria: Failure records round-trip with structured fields.

- [ ] T054 [US1] Emit capture and planner failure codes in `src/shardgrid/planner/generic_graph.py` and `src/shardgrid/planner/partitioning.py`
  - Title: Preserve capture/planner failure causes
  - Phase: Phase 6
  - Priority: P0 BLOCKING
  - Depends on: T035, T052, T053
  - Files: `src/shardgrid/planner/generic_graph.py`, `src/shardgrid/planner/partitioning.py`, `tests/unit/test_generic_graph_ir.py`, `tests/unit/test_partition_candidates.py`
  - Goal: Stop semantic failures from surfacing as generic `NO_FEASIBLE_PLAN`.
  - Implementation Notes: Map graph breaks, custom ops, dynamic control flow, partition coverage, and validation failures to exact codes.
  - Tests: `pytest tests/unit/test_generic_graph_ir.py tests/unit/test_partition_candidates.py`
  - Acceptance Criteria: Planner failure code matches producer stage.

- [ ] T055 [US3] Emit resource, probe, launch, and runtime failure codes in `src/shardgrid/control/job_manager.py` and `src/shardgrid/launchers/ssh.py`
  - Title: Preserve infra and memory failure causes
  - Phase: Phase 6
  - Priority: P0 BLOCKING
  - Depends on: T043, T052, T053
  - Files: `src/shardgrid/control/job_manager.py`, `src/shardgrid/launchers/ssh.py`, `tests/unit/test_memory_probe_launch.py`, `tests/contract/test_ssh_launcher.py`
  - Goal: Distinguish `MEMORY_REJECT`, `FORMAL_TRAINING_OOM`, `RESOURCE_CHANGED`, `NETWORK_FAILURE`, `RENDEZVOUS_FAILURE`, `PROCESS_LAUNCH_FAILURE`, `INFRA_FAILURE`, and `RUNTIME_FAILURE`.
  - Implementation Notes: Probe memory rejection is retryable candidate feedback; formal OOM is not.
  - Tests: `pytest tests/unit/test_memory_probe_launch.py tests/contract/test_ssh_launcher.py`
  - Acceptance Criteria: Failure propagation preserves code and retry policy.

- [ ] T056 [US1] Render structured failures in `src/shardgrid/cli/commands/run.py` and `src/shardgrid/cli/commands/train.py`
  - Title: Show actionable CLI diagnostics
  - Phase: Phase 6
  - Priority: P1
  - Depends on: T053, T055
  - Files: `src/shardgrid/cli/commands/run.py`, `src/shardgrid/cli/commands/train.py`, `tests/integration/test_run_cli.py`, `tests/integration/test_train_cli.py`
  - Goal: Display stage, code, retryable status, and artifact/log references.
  - Implementation Notes: Keep JSON output machine-readable.
  - Tests: `pytest tests/integration/test_run_cli.py tests/integration/test_train_cli.py`
  - Acceptance Criteria: CLI users can identify failing stage without reading raw logs first.

- [ ] T057 [US4] Add stress failure classification tests in `tests/integration/test_stress_dynamic_gpu_multi_job.py`
  - Title: Test stress taxonomy consumption
  - Phase: Phase 6
  - Priority: P1
  - Depends on: T052, T053
  - Files: `tests/integration/test_stress_dynamic_gpu_multi_job.py`, `scripts/stress_dynamic_gpu_multi_job.py`
  - Goal: Verify dry-run, probe, search-budget, CPU/process, GPU-memory, infra, and runtime failures stay distinct.
  - Implementation Notes: Add tests before modifying stress script.
  - Tests: `pytest tests/integration/test_stress_dynamic_gpu_multi_job.py`
  - Acceptance Criteria: Stress runner no longer infers everything from message substrings.

- [ ] T058 [US4] Fix no-progress detection in `scripts/stress_dynamic_gpu_multi_job.py`
  - Title: Treat equal step counts as stall
  - Phase: Phase 6
  - Priority: P0 BLOCKING
  - Depends on: T057
  - Files: `scripts/stress_dynamic_gpu_multi_job.py`, `tests/integration/test_stress_dynamic_gpu_multi_job.py`
  - Goal: Classify `after_steps == before_steps` as no progress/stall evidence, not success.
  - Implementation Notes: Preserve existing `after_steps < before_steps` regression detection.
  - Tests: `pytest tests/integration/test_stress_dynamic_gpu_multi_job.py`
  - Acceptance Criteria: `NO_PROGRESS`/stall is reported when existing jobs do not advance.

- [ ] T059 [US4] Add saturation proof and checkpoint sampling in `scripts/stress_dynamic_gpu_multi_job.py`
  - Title: Prove GPU saturation with evidence
  - Phase: Phase 6
  - Priority: P1
  - Depends on: T055, T058
  - Files: `scripts/stress_dynamic_gpu_multi_job.py`, `tests/integration/test_stress_dynamic_gpu_multi_job.py`
  - Goal: Only report `GPU_MEMORY_SATURATION` when larger bands, 512M final fill, families, candidate coverage, probe results, free memory, cleanup, and non-GPU limits support it.
  - Implementation Notes: Return `SATURATION_NOT_PROVEN` or `SEARCH_BUDGET_LIMIT` when evidence is insufficient.
  - Tests: `pytest tests/integration/test_stress_dynamic_gpu_multi_job.py`
  - Acceptance Criteria: Stress output includes machine-readable saturation, checkpoint, cleanup, and evidence fields.

- [ ] T060 [US4] Run Phase 6 failure/stress gate in `tests/integration/test_stress_dynamic_gpu_multi_job.py`
  - Title: Phase 6 gate
  - Phase: Phase 6
  - Priority: P0 BLOCKING
  - Depends on: T052, T053, T054, T055, T056, T057, T058, T059
  - Files: `tests/contract/test_failure_records.py`, `tests/integration/test_stress_dynamic_gpu_multi_job.py`
  - Goal: Establish `FAILURE_TAXONOMY=PASS` and `STRESS_EVIDENCE=PASS`.
  - Implementation Notes: Hardware stress still waits for Phase 7.
  - Tests: `pytest tests/contract/test_failure_records.py tests/integration/test_stress_dynamic_gpu_multi_job.py`
  - Acceptance Criteria: `FAILURE_TAXONOMY=PASS`, `STRESS_EVIDENCE=PASS`.

## Phase 7: Real Hardware Validation

**Purpose**: Prove SSH-first behavior after CPU/local correctness is already green.

- [ ] T061 [US2] Add fresh resource discovery regression tests in `tests/integration/test_generic_resource_discovery.py`
  - Title: Validate automatic discovery before launch
  - Phase: Phase 7
  - Priority: P0 BLOCKING
  - Depends on: T060
  - Files: `tests/integration/test_generic_resource_discovery.py`, `src/shardgrid/control/resource_manager.py`, `src/shardgrid/resources/models.py`
  - Goal: Verify fresh worker discovery, healthy filtering, reachable filtering, current free memory, and resource refresh before each job.
  - Implementation Notes: Do not require user `world_size`, GPU count, host count, stage count, or min workers for normal automatic mode.
  - Tests: `pytest tests/integration/test_generic_resource_discovery.py tests/unit/test_resource_manager.py`
  - Acceptance Criteria: `FRESH_RESOURCE_DISCOVERY=PASS`.

- [ ] T062 [US3] Add single-GPU generic run acceptance in `tests/hardware/test_generic_single_gpu.py`
  - Title: Validate single GPU runtime and memory probe
  - Phase: Phase 7
  - Priority: P0 BLOCKING
  - Depends on: T061
  - Files: `tests/hardware/test_generic_single_gpu.py`
  - Goal: Prove CUDA, memory probe, formal training, and checkpoint strict reload for a non-zoo script.
  - Implementation Notes: Hardware opt-in markers must protect local default test runs.
  - Tests: `pytest tests/hardware/test_generic_single_gpu.py`
  - Acceptance Criteria: `SINGLE_GPU_GENERIC=PASS`.

- [ ] T063 [US3] Add multi-GPU generic acceptance in `tests/multi_host/test_generic_multi_gpu.py`
  - Title: Validate multi-GPU partition execution
  - Phase: Phase 7
  - Priority: P0 BLOCKING
  - Depends on: T062
  - Files: `tests/multi_host/test_generic_multi_gpu.py`
  - Goal: Verify partition placement, activation transfer, gradient transfer, progress, and checkpoint merge.
  - Implementation Notes: Use exact plan assertions and non-zoo fixtures.
  - Tests: `pytest tests/multi_host/test_generic_multi_gpu.py`
  - Acceptance Criteria: `MULTI_GPU_GENERIC=PASS`.

- [ ] T064 [US3] Add SSH multi-host generic acceptance in `tests/multi_host/test_generic_multi_host.py`
  - Title: Validate SSH multi-host generic training
  - Phase: Phase 7
  - Priority: P0 BLOCKING
  - Depends on: T063
  - Files: `tests/multi_host/test_generic_multi_host.py`
  - Goal: Prove network, rendezvous, distributed execution, remote logs, worker cleanup, shard merge, and strict reload.
  - Implementation Notes: Keep SSH as first compatibility gate; do not add Kubernetes/Volcano/HAMi paths.
  - Tests: `pytest tests/multi_host/test_generic_multi_host.py`
  - Acceptance Criteria: `MULTI_HOST_GENERIC=PASS`.

- [ ] T065 [US2] Add multi-job GPU sharing acceptance in `tests/multi_host/test_generic_multi_job_sharing.py`
  - Title: Preserve multiple jobs per GPU when admitted
  - Phase: Phase 7
  - Priority: P1
  - Depends on: T064
  - Files: `tests/multi_host/test_generic_multi_job_sharing.py`, `src/shardgrid/planner/placement.py`, `src/shardgrid/control/job_manager.py`
  - Goal: Verify one GPU can host Job A, Job B, and Job C when fresh memory/probe evidence admits them.
  - Implementation Notes: Do not regress to one GPU equals one job.
  - Tests: `pytest tests/multi_host/test_generic_multi_job_sharing.py`
  - Acceptance Criteria: `MULTI_JOB_GPU_SHARING=PASS`.

- [ ] T066 [US4] Add low-memory packing stress acceptance in `tests/multi_host/test_generic_stress_packing.py`
  - Title: Validate 4G to 512M stress bands
  - Phase: Phase 7
  - Priority: P2
  - Depends on: T065
  - Files: `tests/multi_host/test_generic_stress_packing.py`, `scripts/stress_dynamic_gpu_multi_job.py`
  - Goal: Run 4G, 3G, 2G, 1G, and 512M packing/saturation after lower layers pass.
  - Implementation Notes: Require structured proof for saturation and cleanup.
  - Tests: `pytest tests/multi_host/test_generic_stress_packing.py`
  - Acceptance Criteria: `MEMORY_PACKING_STRESS=PASS` or structured `SATURATION_NOT_PROVEN`/`SEARCH_BUDGET_LIMIT`.

## Phase 8: Compatibility Deprecation / Final Acceptance

**Purpose**: Keep old validation assets, deprecate production model-specific paths only after generic gates pass, and prove final user outcomes.

- [ ] T067 [US4] Add compatibility regression gate for existing examples in `tests/integration/test_legacy_example_compatibility.py`
  - Title: Keep legacy examples runnable
  - Phase: Phase 8
  - Priority: P1
  - Depends on: T060
  - Files: `tests/integration/test_legacy_example_compatibility.py`, `examples/models/train_generic_dag.py`, `examples/models/train_automatic_plan.py`, `examples/models/generic_partition_zoo/models.py`
  - Goal: Prove zoo/generic DAG/automatic examples remain validation assets.
  - Implementation Notes: Do not restore production dependency on zoo builders.
  - Tests: `pytest tests/integration/test_legacy_example_compatibility.py`
  - Acceptance Criteria: `LEGACY_VALIDATION_ASSETS=PASS`.

- [ ] T068 [US4] Add production dependency scanner in `tests/unit/test_no_production_zoo_dependency.py`
  - Title: Enforce no production zoo dependency
  - Phase: Phase 8
  - Priority: P0 BLOCKING
  - Depends on: T027, T048
  - Files: `tests/unit/test_no_production_zoo_dependency.py`, `src/shardgrid/control/job_manager.py`, `src/shardgrid/runtime/`, `src/shardgrid/planner/`
  - Goal: Fail if production planner/runtime/checkpoint imports zoo builders or branches on model catalog names.
  - Implementation Notes: Permit references under `examples/`, `tests/`, and explicit compatibility paths only.
  - Tests: `pytest tests/unit/test_no_production_zoo_dependency.py`
  - Acceptance Criteria: `PRODUCTION_ZOO_DEPENDENCY=0` remains enforced.

- [ ] T069 [US1] Add three-project final acceptance suite in `tests/integration/test_generic_industrial_entrypoints.py`
  - Title: Validate three ordinary PyTorch projects
  - Phase: Phase 8
  - Priority: P0 BLOCKING
  - Depends on: T037, T051, T060
  - Files: `tests/integration/test_generic_industrial_entrypoints.py`, `tests/fixtures/ordinary_training_scripts/`
  - Goal: Verify Project A, Project B, and Project C launch through `shardgrid run` with no ShardGrid-specific model provider, stage, model branch, or manual sample input.
  - Implementation Notes: Include structured inputs and at least three model structures.
  - Tests: `pytest tests/integration/test_generic_industrial_entrypoints.py`
  - Acceptance Criteria: `INDUSTRIAL_ENTRYPOINTS=PASS`.

- [ ] T070 Run final feature acceptance commands and record release readiness in `specs/002-generic-pytorch-automation/tasks.md`
  - Title: Final feature gate
  - Phase: Phase 8
  - Priority: P0 BLOCKING
  - Depends on: T061, T062, T063, T064, T065, T066, T067, T068, T069
  - Files: `tests/unit/`, `tests/integration/`, `tests/multi_host/`, `tests/hardware/`
  - Goal: Establish all final acceptance signals.
  - Implementation Notes: Hardware commands are opt-in and must run after CPU/local gates.
  - Tests: `pytest tests/unit tests/integration`; opt-in `pytest tests/hardware tests/multi_host`
  - Acceptance Criteria: `FEATURE_ACCEPTANCE=PASS`, `TASKS_COMPLETE=YES`, `IMPLEMENTATION_DONE=YES`.

## Dependency Graph

### Phase Gates

- Phase 0 Gate: T008 -> `REGRESSION_BASELINE=PASS`
- Phase 1 Gate: T021 -> `PLANNER_COMPATIBILITY=PASS`
- Phase 2 Gate: T027 -> `PRODUCTION_ZOO_DEPENDENCY=0`
- Phase 3 Gate: T037 -> `ENTRYPOINT_CAPTURE=PASS`
- Phase 4 Gate: T044 -> `GENERIC_RUNTIME_LOCAL=PASS`
- Phase 5 Gate: T051 -> `STANDARD_STATE_DICT_STRICT_LOAD=PASS`
- Phase 6 Gate: T060 -> `FAILURE_TAXONOMY=PASS`, `STRESS_EVIDENCE=PASS`
- Phase 7 Gate: T066 -> `HARDWARE_VALIDATION=PASS` or explicit opt-in skip
- Phase 8 Gate: T070 -> `FEATURE_ACCEPTANCE=PASS`

### User Story Dependencies

- US1 Run Existing Training Entrypoint: starts after Phase 2 gate; depends on planner decoupling for end-to-end dry-run.
- US2 Plan From Execution Graph, Not Model Names: starts after Phase 0 gate; blocks US1 production planning and US3 runtime.
- US3 Execute And Consolidate Generic Training: starts after US1 capture and US2 planner gates.
- US4 Keep Validation Assets Out Of Production Flow: can begin after Phase 2 for example scoping and after Phase 5 for final scanner/stress coverage.

### Blocking Tasks

T001, T002, T003, T004, T005, T006, T007, T008, T009, T010, T011, T012, T013, T014, T015, T016, T018, T020, T021, T022, T023, T024, T027, T028, T029, T030, T031, T032, T034, T037, T038, T039, T040, T041, T043, T044, T045, T046, T047, T048, T049, T051, T052, T053, T054, T055, T058, T060, T061, T062, T063, T064, T068, T069, T070.

## Parallel Execution Examples

### Phase 0

```bash
Task: T001 CLI characterization
Task: T002 JobManager coupling characterization
Task: T003 CPU model fixtures
Task: T007 checkpoint baseline
```

### Phase 1

```bash
Task: T009 state ownership dataclasses
Task: T010 execution-node value metadata
Task: T020 planner compatibility matrix tests
```

### Phase 3

```bash
Task: T028 CLI contract tests
Task: T030 ordinary training script fixtures
Task: T031 capture contract tests
```

### Phase 5

```bash
Task: T045 checkpoint contract tests
```

### Phase 6

```bash
Task: T052 failure-code enums
Task: T057 stress classification tests
```

## Implementation Strategy

### MVP First

1. Complete Phase 0 regression baseline.
2. Complete Phase 1 planner semantic repair.
3. Complete Phase 2 production decoupling.
4. Complete Phase 3 `shardgrid run` capture path.
5. Stop and validate US1 independently with ordinary PyTorch scripts.

### Incremental Delivery

1. Deliver US2 planner correctness first because runtime/checkpoint cannot be safe without it.
2. Deliver US1 entrypoint capture after production decoupling.
3. Deliver US3 runtime/checkpoint once exact plan and capture artifacts are stable.
4. Deliver US4 compatibility/stress gates without putting zoo assets back into production.

### Hardware Order

```text
CPU planner compatibility
  -> local runtime
  -> generic checkpoint
  -> failure taxonomy
  -> single GPU
  -> multi GPU
  -> multi host
  -> multi job sharing
  -> memory packing stress
```

## Final Status Targets

```text
TASKS_CREATED=YES
IMPLEMENTATION_STARTED=NO
SPEC_MODIFIED=NO
PLAN_MODIFIED=NO
OLD_SPEC_FILES_MODIFIED=0
```
