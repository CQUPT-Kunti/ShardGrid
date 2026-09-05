## 2026-09-03 T115

- train plan output: `shardgrid train --dry-run` now emits auditable `ExecutionPlan` data from the saved snapshot metadata instead of a reduced status-only summary.
- engine/backend visibility: CLI JSON and human output show `engine`, `backend`, and engine launch metadata before any remote launch starts.
- master/world_size visibility: dry-run output and snapshot metadata show `master.address`, `master.port`, and `world_size`.
- placement/assignment visibility: stage-to-worker placement is preserved from T114 stage metadata and exposed as `stage -> worker -> rank -> GPU`, plus host, machine, runtime, Python, and log path details per assignment.
- original plan visibility: `execution-plan.json` now preserves `parallel_plan_ref`, `original_engine_plan_ref`, `model_profile_ref`, and `candidate_evaluation_ref`.
- fallback labels: audit output includes fallback status, fallback label, fallback reason, rejected engine list, and selected worker-count metadata without changing T111/T112 decisions.
- snapshot metadata: snapshot writes `original-parallel-plan.json|yaml`, `execution-plan.json|yaml`, and `snapshot-metadata.json|yaml`, with one shared `execution_plan_audit` payload used by CLI output.
- dry-run verification: dry-run returns after snapshot write and before `engine.prepare`, launcher creation, SSH launch, torch distributed init, or training process start.
- JSON/YAML audit: tests verify JSON and YAML artifacts carry the same engine, backend, master, world size, placement, assignment, original-plan, and fallback fields.
- 方案划分决策输出:
  - `partition_source=automatic`
  - `selected_candidate_id`, `selected_worker_count`, `attempted_worker_counts`, and `total_cross_worker_communication_bytes` are visible in audit metadata
  - each `stage_metadata_ref` stays attached to the final assignment so the chosen stage split and worker placement can be traced back to the preserved `ParallelPlan`

## 2026-09-03 T116

- planner gate: added `tests/integration/test_planner_gate.py` to validate the T108-T115 automatic planner chain end to end at the planner boundary, including peak-memory fit, strict worker-count escalation, deterministic selection, persistence, replay rejection, and dry-run no-launch behavior.
- training peak memory constraint: gate coverage now rejects plans when `estimated_peak_training_memory` exceeds worker usable memory even when raw parameter bytes still fit.
- strict worker-count search: validation keeps the current ordered search `2 -> 3 -> 4` and confirms the search stops on the first feasible worker count instead of globally re-ranking mixed worker counts.
- automatic partition + placement consistency: gate coverage verifies `partition_source=automatic`, one placement per stage, unique selected workers, preserved `stage_metadata_ref`, and unchanged stage-to-worker/rank/GPU mapping between selected placement, `ParallelPlan`, and `ExecutionPlan`.
- residual/skip communication preserved: persistence checks confirm non-adjacent residual / skip communication edges survive T110 -> T114 -> T115 and remain visible in saved plan metadata.
- deterministic planning: repeated planning with identical model/profile/cluster inputs yields the same selected worker count, worker set, candidate id, and stage placement.
- plan persistence: JSON/YAML artifacts preserve `selected_worker_count`, `selected_worker_ids`, stage mapping, placement, memory metadata, communication metadata, `engine`, `backend`, `master`, `world_size`, assignments, and planning provenance.
- replay safety: `src/shardgrid/planner/replay.py` now validates saved automatic-plan artifacts fail closed when worker health or usable memory changes, and it rejects silent re-placement.
- manual override: recorded as `NOT_SUPPORTED / SKIPPED_BY_CURRENT_DESIGN` for the current planner gate; T116 does not reintroduce T113 behavior.
- docs alignment: `docs/architecture/planner.md` now documents the real current flow `T109 -> T110 -> T111 -> T112 -> T114 -> T115 -> T116`, the memory hard gate, the communication-first ranking rule, and the deferred T117 hardware boundary.
- temp planner design: wrote `/home/yangjilei/Code/ShardGrid/temp.md` as the current final planner design summary for audit.
- live dry-run: on 2026-09-03, `PYTHONPATH=src /home/yangjilei/anaconda3/bin/python -m shardgrid.cli.app --config examples/workers.yaml train examples/train-minimal.yaml --dry-run --json` returned `state=snapshotting`, `phase=plan`, `metadata_equal=True`, and all assignment `pid` values remained `null`; this validates no remote rank launch, while the current example still reports `plan_mode=static` and `partition_source=manual`.
- 方案划分决策输出:
  - gate verifies the saved decision chain keeps `attempted_worker_counts`, `selected_worker_count`, `selected_candidate_id`, `total_cross_worker_communication_bytes`, and per-stage placement data
  - the persisted placement remains auditable as `stage -> worker -> rank -> GPU` with usable-memory, remaining-memory, and utilization fields
  - replay uses the saved plan; it does not recompute a new partition or new placement when resources drift

## 2026-09-03 Automatic Planner CLI Integration

- real train CLI connected to the existing automatic planner path through `planning.mode=automatic`; static/manual configs keep the previous path.
- added `examples/train-automatic.yaml` as an automatic workload config with no manual stage split or manual placement.
- `JobManager.run()` now routes automatic dry-run planning through the existing planner chain and preserves the selected result as the final `ParallelPlan` / `ExecutionPlan`.
- dry-run audit output now shows `plan_mode`, `partition_source`, `selected_candidate_id`, `selected_worker_count`, `attempted_worker_counts`, `selected_workers`, and cross-worker communication bytes before any launch step.
- snapshot metadata validation now treats automatic planning correctly: `ParallelPlan.world_size` must match `ExecutionPlan.world_size`, while static/manual plans still match the requested job world size.
- static/manual regression preserved; dry-run still exits before `engine.prepare`, launcher creation, SSH rank launch, torch distributed init, or real training.
- 方案划分决策输出:
  - automatic CLI path preserves `partition_source=automatic`, `selected_candidate_id`, `selected_worker_count`, `attempted_worker_counts`, and `total_cross_worker_communication_bytes`
  - the final stage placement remains auditable as `stage -> worker -> rank -> GPU`, with per-stage estimated peak memory, usable memory, remaining memory, and utilization

## 2026-09-03 Automatic Plan Hardware Gate

- automatic CLI invocation: added `examples/train-automatic-hf.yaml` and `tests/multi_host/test_automatic_partition_gate.py` so the live gate starts from `shardgrid train` with `planning.mode=automatic`, not a hand-built `ExecutionPlan`.
- selected worker count / selected workers / attempted worker counts: the live gate reads the saved audit payload from `snapshot-metadata.json` and verifies planner-selected workers from the real automatic plan output.
- stage -> worker mapping: automatic live runs preserve `stage -> worker -> rank -> GPU` through the saved `ParallelPlan`, `ExecutionPlan`, rank placement evidence, and checkpoint metadata.
- training peak / usable memory: non-dry-run automatic launches now re-probe selected workers immediately before launch and fail closed with `RESOURCE_CHANGED` if current free memory drops below the saved per-stage `estimated_peak_training_memory`.
- communication metadata: automatic assignments preserve `selected_candidate_id`, communication-edge labels, and saved placement metadata; the new runner uses PyTorch pipeline splitting from saved stage boundaries instead of static `stage0.py` / `stage1.py`.
- SSH launch: automatic plans now launch `examples/models/train_automatic_plan.py` via `SSHLauncher`; static/manual plans keep the existing `train_pipeline.py` path.
- forward / activation transfer / loss / backward / gradient transfer / optimizer update: `train_automatic_plan.py` rebuilds the full supported model, recreates PyTorch pipeline stages from saved module boundaries, runs short real training steps, records finite loss on the last stage, and persists parameter-update evidence.
- checkpoint: per-rank checkpoints plus `checkpoint-metadata.json` now record `job_id`, `partition_source`, `selected_candidate_id`, `selected_worker_count`, `master`, and `stage_to_worker`.
- cleanup / evidence paths: live gate evidence is designed to land in the normal job snapshot under `plan/`, `diagnostics/`, `logs/`, and `checkpoint/`; new compatibility note: `docs/compatibility/automatic-partition.md`.
- 方案划分决策输出:
  - automatic live path keeps planner-owned split decisions from `ParallelPlan.stage_metadata`, including stage module paths, selected candidate id, selected worker count, attempted worker counts, and cross-worker communication bytes
  - launch-time code does not recompute partition or placement; it only validates the saved plan against fresh worker/network state
  - status on 2026-09-03: implementation and local regression coverage passed; live hardware evidence remains pending because this turn did not execute a real worker run

## 2026-09-04 Automatic Stage-Local Materialization

- worker runtime path: `examples/models/train_automatic_plan.py` no longer uses `build_large_residual_transformer() -> pipeline(full_model) -> get_stage_module()` for `large_residual_transformer`; it now reads the assigned `stage_metadata`, materializes only that stage, and builds `PipelineStage` directly from the local stage module.
- stage-local model builder: `examples/models/large_residual_transformer.py` now exposes `build_large_residual_transformer_stage(...)`, `large_residual_module_paths(...)`, and `make_large_residual_stage_inputs(...)` so the worker can execute contiguous leaf-module slices while preserving residual and long-skip semantics.
- control-plane planning: `JobManager._planner_workload()` now constructs the large model on `device="meta"` and feeds meta sample inputs into `build_model_profile()` and the existing automatic partition/placement search; no real large parameter storage is created on the control node for this planning path.
- gate helper planning: `tests/multi_host/large_model_gate.py::estimate_large_model()` now also uses `device="meta"` so the hardware gate does not pre-materialize a full real large model before entering `JobManager`.
- meta shared-parameter fix: `src/shardgrid/planner/memory.py` now keys shared-parameter detection by `id(parameter)` for meta tensors, avoiding the false tied-weight rejection caused by identical meta `data_ptr()` placeholders.
- worker evidence: large automatic workers now record `owned_module_paths`, `materialized_parameter_names`, `materialized_parameter_count`, `materialized_parameter_bytes`, `full_model_materialized=false`, `initial_weights_received_from_control=false`, RSS before/after CPU materialization, CUDA allocated before/after `.to(device)`, and training-time `peak_gpu_memory_bytes`.
- local verification:
  - `tests/unit/test_train_automatic_plan.py` proves the large automatic worker path does not call the full model builder.
  - `tests/unit/test_large_training_model.py` verifies parameter ownership, reduced per-stage parameter bytes, stage-chain residual/long-skip equivalence against the full model, and full-model meta construction.
  - `tests/integration/test_train_orchestration.py` verifies the large automatic planner workload returns a meta-parameter model and meta sample inputs.
  - `tests/multi_host/test_large_model_partition_gate.py` verifies meta estimation locally and, in hardware mode, audits per-worker stage-local evidence.
- real hardware gate:
  - command: `SHARDGRID_ENABLE_HARDWARE_TESTS=1 SHARDGRID_ENABLE_MULTI_HOST_TESTS=1 SHARDGRID_RUN_LARGE_MODEL_HW=1 PYTHONPATH=src python -m pytest tests/multi_host/test_large_model_partition_gate.py --run-hardware --run-multi-host -q`
  - result on 2026-09-04: PASS
  - selected planning decision: `selected_worker_count=2`, `attempted_worker_counts=[2]`, `selected_candidate_id=pytorch_pipeline:large-b:0:22-22:36`, `total_cross_worker_communication_bytes=851968`
  - observed stage split from monitor artifacts:
    - `stage0 -> gpu4060`: `token_embedding`, `position_embedding`, `blocks.0.*`, `blocks.1.*`, `blocks.2.norm1`, `blocks.2.attn.qkv`, `blocks.2.attn.out_proj`, `blocks.2.norm2`; `materialized_parameter_bytes=1147277312`; `full_model_materialized=false`
    - `stage1 -> gpu4060-cqupt`: `blocks.2.ffn.0`, `blocks.2.ffn.1`, `blocks.2.ffn.2`, `blocks.2.memory_pressure`, `blocks.3.*`, `norm`, `output_head`; `materialized_parameter_bytes=1138851840`; `full_model_materialized=false`
  - observed training memory evidence:
    - `stage0 peak_gpu_memory_bytes=5766092800`
    - `stage1 peak_gpu_memory_bytes=5725571072`

## 2026-09-04 Three-Worker Gate + Scheme D Progress

- three-worker hardware gate:
  - command: `SHARDGRID_ENABLE_HARDWARE_TESTS=1 SHARDGRID_ENABLE_MULTI_HOST_TESTS=1 SHARDGRID_RUN_THREE_WORKER_HW=1 PYTHONPATH=src python -m pytest tests/multi_host/test_large_model_partition_gate.py::test_large_residual_transformer_three_worker_training_passes --run-hardware --run-multi-host -q`
  - result: `PASS` on 2026-09-04
  - selected planner mode: `SHARDGRID_AUTOMATIC_MIN_WORKERS=3`
  - selected decision:
    - `selected_worker_count=3`
    - `attempted_worker_counts=[3]`
    - `selected_candidate_id=pytorch_pipeline:three-worker-large:0:19-19:26-26:36`
  - observed split:
    - `stage0 -> gpu4060`
    - `stage1 -> gpu1060`
    - `stage2 -> gpu4060-cqupt`
  - worker evidence now records `pid`, `distributed_initialized`, `stage_materialized`, `optimizer_step_completed`, `process_rss_before/after_materialization`, `cuda_before/after_stage_move`, and `cuda_training_peak_bytes`.
- Scheme D test harness fixes:
  - `tests/multi_host/large_model_gate.py` now reads live status from `diagnostics/job-status.json`, not the stale top-level file.
  - running jobs now emit step-level `T074_TRAIN_EVIDENCE` during training, so the multi-job gate can stop at `>=3` real steps instead of waiting for full job completion.
  - Scheme D dry-run `job_id` values are added to `known_job_ids` before each real launch so `wait_for_job_ids()` does not mistake a dry-run snapshot for the real long-running job.
  - actual long-running Scheme D jobs now override `training_steps` in the submitted training config instead of only the parent control-process environment.
- Scheme D live status on 2026-09-04:
  - not yet re-verified end-to-end after the final `known_job_ids` fix.
  - latest confirmed live A-only evidence before cleanup:
    - `job-20260904085802-d775017f`
    - `state=training`
    - running steps observed from monitor payloads: `gpu4060=33+`, `gpu1060=34+`, `gpu4060-cqupt=35+`
    - `AFTER_A` probe during live load took about `32.5s` and reported roughly:
      - `gpu4060 free ~= 2122 MB`
      - `gpu1060 free ~= 7 MB`
      - `gpu4060-cqupt free ~= 3694 MB`
  - cleanup after interrupted verification:
    - stopped `job-20260904084012-3e42b7d6`
    - stopped `job-20260904084820-eb38fedd`
    - stopped `job-20260904085802-d775017f`
    - no persistent reservation should remain after stop success

## 2026-09-04 Job Status + Two-Worker Sharing Fix

- authoritative job status:
  - `JobStatus` now uses one mutable path only: `jobs/<job-id>/job-status.json`
  - `JobManager`, `SSHLauncher`, `train`, `logs`, `stop`, multi-host helpers, and snapshot metadata now read/write the same file
  - `diagnostics/job-status.json` is no longer written as a second mutable copy
- stop lifecycle cleanup:
  - `shardgrid stop` now releases reservations only after the persisted job state proves the job is stopped, or when a `NOOP` stop confirms an already-terminal job with no live tracked process to stop
  - partial stop results do not release reservations early
  - shared-GPU reservations are released per job id, so stopping Job A does not remove Job B on the same GPU
- worker-count bounds for two-host gates:
  - automatic planner test overrides now support `SHARDGRID_AUTOMATIC_MAX_WORKERS`
  - this keeps D1/D2 hardware validation on exactly two workers without changing planner search rules
- two-worker Scheme D hardware gates:
  - replaced the old three-worker `large + medium + small` stress harness with two focused gates:
    - `D1`: `medium + small`
    - `D2`: `large + small`
  - both gates force two-worker planning, require real training evidence for both jobs, require the second job to plan against reduced live free memory, and require at least one physical GPU to hold stages from both jobs at the same time
  - cleanup for both gates now verifies:
    - `status = stopped`
    - reservations released
    - no remaining test pids
- local verification on 2026-09-04:
  - `PYTHONPATH=src /home/yangjilei/anaconda3/bin/python -m pytest tests/unit/test_job_status.py tests/integration/test_stop_cli.py tests/integration/test_train_orchestration.py::test_orchestrates_full_training_lifecycle tests/integration/test_train_orchestration.py::test_automatic_worker_count_bounds_honor_test_override tests/integration/test_train_orchestration.py::test_automatic_worker_count_bounds_honor_max_worker_override tests/integration/test_logs_cli.py tests/integration/test_ssh_monitor.py tests/integration/test_ssh_stop.py tests/integration/test_ssh_cleanup.py --run-integration -q`
  - result: `55 passed`
  - separate compile/smoke pass including the multi-host gate modules without hardware flags:
    - result: `6 passed, 88 skipped`
- hardware note:
  - this turn implemented the two-worker D1/D2 gates but did not rerun the real two-host hardware validation end to end after the lifecycle fix

## 2026-09-04 Generic Cluster Autopartition Attempt

- production automatic worker-count cap:
  - removed the hard `min(4, available_worker_count)` cap from `JobManager._automatic_worker_count_bounds()`
  - default automatic planning bounds are now `1..available_worker_count`
  - explicit `SHARDGRID_AUTOMATIC_MIN_WORKERS` / `SHARDGRID_AUTOMATIC_MAX_WORKERS` still constrain the search when set
- planner scalability coverage:
  - added coverage for dynamic cluster sizes `1, 2, 3, 4, 8, 16`
  - command: `PYTHONPATH=src /home/yangjilei/anaconda3/bin/python -m pytest tests/integration/test_train_orchestration.py::test_automatic_worker_count_bounds_default_to_all_available_workers tests/integration/test_train_orchestration.py::test_automatic_worker_count_bounds_honor_test_override tests/integration/test_train_orchestration.py::test_automatic_worker_count_bounds_honor_max_worker_override --run-integration -q`
  - result: `3 passed`
- packing hardware test dynamic discovery:
  - removed fixed `PACKING_WORKER_IDS = ("gpu4060", "gpu1060", "gpu4060-cqupt")`
  - packing gate now discovers live GPU workers from inventory and uses the discovered worker count for per-test min/max automatic worker overrides
  - current hardware gate still requires at least three discovered GPUs
- current blocker:
  - automatic production runtime still depends on `LargeResidualTransformerStage`, `build_large_residual_transformer_stage()`, `_path_uses()`, and `_path_defs()`
  - generic graph capture, Graph IR, generic StageSpec, graph-derived boundary tensors, and generic stage-local GraphModule runtime are not implemented yet
  - therefore `GENERIC GPU CLUSTER + AUTOMATIC MODEL PARTITION = BLOCKED`
- issue package:
  - `shardgrid-generic-cluster-autopartition-issue-20260904.zip`
  - includes source, tests, diff, hardware evidence, and blocker analysis

## 2026-09-05 Generic Graph Auto-Partition Progress

- generic graph IR:
  - added `src/shardgrid/planner/generic_graph.py`
  - captures FX graph nodes, value IDs, graph edges, input/output values, tensor metadata, and parameter owners
  - added automatic boundary-value inference from cross-stage value producer/consumer edges
  - `discover_partition_support()` now uses `GenericGraphIR -> module_dependencies_from_graph()` for the FX dependency path
  - unsupported custom ops still fail closed through the existing structured planner result
- generic model zoo:
  - added `examples/models/generic_partition_zoo/`
  - models added: `MiniResNet`, `MiniUNet`, `MiniDenseNet`, `MiniInception`, `MiniViT`, `MiniEncoderDecoder`, `MultiInputNet`, `MultiOutputNet`, `ResidualMLPDAG`
  - the zoo only provides model factories and sample inputs; no model-specific stage builders were added
- graph/boundary evidence from unit tests:
  - Mini U-Net: dependency extraction identifies `enc1` output consumed by decoder-side modules, and value boundary inference can mark the long skip as a cross-stage boundary
  - Mini DenseNet: value boundary inference can represent one produced value with multiple downstream consumer stages
  - Mini Inception: branch outputs are detected as cross-stage values feeding the merge/head side
  - Mini Encoder-Decoder: encoder output dependency into decoder path is detected from graph data
- planner scalability:
  - replaced `_worker_subsets()` full `combinations(workers, k)` enumeration with bounded deterministic candidate generation
  - default candidate budget: `64`
  - override: `SHARDGRID_PLACEMENT_CANDIDATE_BUDGET`
  - added coverage for `64` workers / `32` selected workers without enumerating `C(64,32)`
- verification:
  - lint: `PYTHONPATH=src /home/yangjilei/anaconda3/bin/python -m ruff check src/shardgrid/planner/generic_graph.py src/shardgrid/planner/partitioning.py src/shardgrid/planner/placement.py examples/models/generic_partition_zoo tests/unit/test_generic_graph_ir.py tests/unit/test_joint_partition_placement.py tests/unit/test_partition_candidates.py tests/unit/test_train_automatic_plan.py`
  - result: `All checks passed`
  - tests: `PYTHONPATH=src /home/yangjilei/anaconda3/bin/python -m pytest tests/unit/test_generic_graph_ir.py tests/unit/test_joint_partition_placement.py tests/unit/test_partition_candidates.py tests/unit/test_train_automatic_plan.py -q`
  - result: `39 passed`
- remaining blocker:
  - automatic production runtime still uses `examples/models/train_automatic_plan.py` model-type branches
  - `large_residual_transformer` live execution still depends on `LargeResidualTransformerStage`, `build_large_residual_transformer_stage()`, `_path_uses()`, and `_path_defs()`
  - generic stage-local GraphModule extraction/materialization, generic distributed pipeline forward/backward, and real hardware multi-model stress validation are not complete
  - therefore `GENERIC MODEL AUTO-PARTITION + EXECUTION = BLOCKED`

## 2026-09-05 DAG Planning Contract Progress

- canonical graph contract:
  - upgraded the generic graph layer with `schema_version=shardgrid.canonical_graph.v1`
  - added deterministic `graph_fingerprint`
  - added graph value consumer lists, edge ids, forward/backward transfer byte fields, communication weights, parameter ids, buffer ids, and input/output pytree descriptors
  - added `GraphCaptureAdapter`, `FXGraphCaptureAdapter`, and `GraphCaptureResult`
  - the FX capture path now produces `CanonicalGraphIR` through the adapter interface
- capture lifecycle:
  - `discover_partition_support()` now reuses the adapter capture result for FX diagnostics and dependency extraction instead of running a separate custom-op trace before graph capture
  - unsupported custom ops still return structured unsupported planner results
- stable planning contract:
  - added `src/shardgrid/planner/planning_contract.py`
  - introduced `RuntimeCapabilities`, `ResourceSnapshot`, `PlanningConstraints`, `LogicalPartitionPlan`, `PlacementPlan`, and `PlanningResult`
  - `LogicalPartitionPlan` and `PlacementPlan` are separate serializable artifacts with their own schema version
  - the contract allows logical partition count to differ from selected GPU count when runtime capabilities allow multiple partitions per device
  - communication reporting includes total graph edge bytes, cross-partition bytes, cross-GPU forward bytes, cross-GPU backward bytes, and estimated communication cost
- code-structure/model independence tests:
  - fingerprint ignores Python class rename when graph semantics stay the same
  - fingerprint changes when the graph really changes
  - generic planner core scan rejects model names and `examples.models` imports from core planner files
  - shared parameter usage is recorded and rejected when runtime capabilities do not support it
- verification:
  - lint: `PYTHONPATH=src /home/yangjilei/anaconda3/bin/python -m ruff check src/shardgrid/planner/generic_graph.py src/shardgrid/planner/planning_contract.py src/shardgrid/planner/partitioning.py src/shardgrid/planner/placement.py src/shardgrid/planner/__init__.py tests/unit/test_generic_graph_ir.py tests/unit/test_planning_contract.py tests/unit/test_joint_partition_placement.py tests/unit/test_partition_candidates.py tests/unit/test_train_automatic_plan.py`
  - result: `All checks passed`
  - tests: `PYTHONPATH=src /home/yangjilei/anaconda3/bin/python -m pytest tests/unit/test_generic_graph_ir.py tests/unit/test_planning_contract.py tests/unit/test_joint_partition_placement.py tests/unit/test_partition_candidates.py tests/unit/test_train_automatic_plan.py -q`
  - result: `45 passed`
- remaining blocker:
  - this contract is not yet wired as the production `ExecutionPlan` compiler/runtime path
  - generic DAG stage extraction, stage-local parameter/buffer materialization, local forward/backward/optimizer correctness, real hardware execution, and multi-job stress are still incomplete
  - therefore `CODE-STRUCTURE-INDEPENDENT MODEL-AGNOSTIC DAG GRAPH PARTITION + GPU PLACEMENT = BLOCKED`

## 2026-09-05 Memory-Aware Multi-Plan DAG Planning Progress

- partition search:
  - removed the fixed `max_logical_partitions=8` default from the DAG planning contract
  - `PlanningConstraints.max_logical_partitions` is now optional; search is bounded by candidate budgets instead of a model partition cap
  - added `max_partition_candidates`, `max_placement_candidates_per_partition`, `max_total_plan_candidates`, `top_k_plans`, and `beam_width`
  - logical partition candidates are generated from current `ResourceSnapshot.free_memory_bytes`
  - partition target sizes are capacity-proportional and may be uneven
  - generated candidates are evaluated and only top-K plan candidates are retained
- placement search:
  - placement now uses bounded beam search over placement states
  - each state stores only remaining-memory metadata and partition assignments
  - placement can produce multiple alternatives per partition candidate
  - selected GPU count is derived from placement result, not forced to match available GPU count or partition count
- scoring:
  - memory feasibility is a hard constraint
  - current score uses memory slack, partition count penalty, and estimated cross-GPU communication cost
  - compute/optimizer/temporary cost remains incomplete by design for this round
- meta/fake capture:
  - added `ModelFactorySpec` and `FXGraphCaptureAdapter.capture_factory()`
  - meta factory capture records `control_plane_parameter_real_storage_bytes=0`
  - direct capture of an already real model now records real parameter storage truthfully instead of claiming `full_real_model_materialized=false`
- evidence:
  - heterogeneous free-memory plan: `artifacts/memory-aware-multiplan-issue-20260905/plans/heterogeneous-free-memory-plan.json`
  - selected partition sizes in that evidence: `24` and `12`
  - selected placement: `P0 -> gpu-large`, `P1 -> gpu-mid`
  - search diagnostics: `partition_candidates=8`, `placement_candidates=17`, `evaluated_plans=16`, `search_budget=16`
  - `>8` partition evidence: `max_partition_count=16`
  - repeated planning RSS evidence: `20` iterations, delta `20480` bytes
- verification:
  - lint: `PYTHONPATH=src /home/yangjilei/anaconda3/bin/python -m ruff check src/shardgrid/planner/generic_graph.py src/shardgrid/planner/planning_contract.py src/shardgrid/planner/partitioning.py src/shardgrid/planner/placement.py src/shardgrid/planner/__init__.py tests/unit/test_generic_graph_ir.py tests/unit/test_planning_contract.py tests/unit/test_joint_partition_placement.py tests/unit/test_partition_candidates.py tests/unit/test_train_automatic_plan.py`
  - result: `All checks passed`
  - tests: `PYTHONPATH=src /home/yangjilei/anaconda3/bin/python -m pytest tests/unit/test_generic_graph_ir.py tests/unit/test_planning_contract.py tests/unit/test_joint_partition_placement.py tests/unit/test_partition_candidates.py tests/unit/test_train_automatic_plan.py -q`
  - result: `51 passed`
- remaining blocker:
  - this is still a planning-contract implementation, not a production runtime implementation
  - generic DAG stage extraction/materialization and real hardware/multi-job execution are not complete
  - therefore `MEMORY-AWARE MULTI-PLAN GENERIC DAG PARTITION + GPU PLACEMENT = BLOCKED`

## 2026-09-05 Multi-Partition Worker DAG Runtime Progress

- worker ownership contract:
  - added `src/shardgrid/runtime/dag.py`
  - `compile_runtime_plan()` consumes `CanonicalGraphIR + LogicalPartitionPlan + PlacementPlan`
  - generated `WorkerOwnershipPlan` groups all logical partitions assigned to the same `worker_id/gpu_index/gpu_id`
  - ownership records deduplicated local parameter ids and buffer ids for partition-set-local materialization
- placement/runtime edge audit:
  - runtime compilation no longer assumes `one rank = one stage`
  - local edges are graph edges whose producer and consumer partitions share the same worker/GPU
  - remote edges are graph edges crossing worker/GPU placement
  - non-contiguous same-GPU placement is represented, e.g. `worker0/gpu0` owns `P0` and `P4`
- local DAG execution:
  - added `ValueStore` with consumer-count based release after the last downstream consumer
  - added `LocalDAGRuntime.forward()` ready-queue scheduler for local partition callables
  - local autograd backward and optimizer-step smoke are verified without custom autograd or custom distributed communication
- evidence from tests:
  - non-contiguous placement: `P0 -> worker0/gpu0`, `P1 -> worker1/gpu0`, `P2 -> worker2/gpu0`, `P3 -> worker1/gpu0`, `P4 -> worker0/gpu0`
  - local edge: `P0 -> P4` for residual value `v0`
  - remote edges: `P0 -> P1`, `P1 -> P2`, `P2 -> P3`, `P3 -> P4`
  - executed local sequence: `P0, P1, P2, P3, P4`
  - optimizer evidence: parameters receive gradients and at least one weight changes after `optimizer.step()`
- verification:
  - lint: `PYTHONPATH=src /home/yangjilei/anaconda3/bin/python -m ruff check src/shardgrid/runtime src/shardgrid/planner/generic_graph.py src/shardgrid/planner/planning_contract.py src/shardgrid/planner/partitioning.py src/shardgrid/planner/placement.py src/shardgrid/planner/__init__.py tests/unit/test_dag_runtime.py tests/unit/test_generic_graph_ir.py tests/unit/test_planning_contract.py tests/unit/test_joint_partition_placement.py tests/unit/test_partition_candidates.py tests/unit/test_train_automatic_plan.py`
  - result: `All checks passed`
  - tests: `PYTHONPATH=src /home/yangjilei/anaconda3/bin/python -m pytest tests/unit/test_dag_runtime.py tests/unit/test_generic_graph_ir.py tests/unit/test_planning_contract.py tests/unit/test_joint_partition_placement.py tests/unit/test_partition_candidates.py tests/unit/test_train_automatic_plan.py -q`
  - result: `54 passed`
- remaining blocker:
  - this is a local runtime contract and smoke test, not the production SSH ExecutionPlan runtime
  - no generic `extract_partition_graph()` / partition GraphModule materializer is wired yet
  - remote edge transport, remote activation/gradient exchange, and real hardware multi-job DAG execution are not implemented through this path
  - therefore `MULTI-PARTITION-PER-GPU GENERIC DAG RUNTIME = BLOCKED`

## 2026-09-05 Generic DAG Runtime + Checkpoint Reconstruction Attempt

- extract partition graph:
  - added `src/shardgrid/runtime/partition_graph.py`
  - `extract_partition_graph()` builds an executable FX `GraphModule` from `CanonicalGraphIR`, backend FX graph, and `LogicalPartitionSpec.node_ids`
  - partition placeholders use stable `LogicalPartitionSpec.input_value_ids`
  - partition outputs use stable `LogicalPartitionSpec.output_value_ids`
  - copied FX ops preserve `call_module`, `call_function`, `call_method`, `get_attr`, and nested tensor args handled by FX
- local executable partition gate:
  - artificial partition lambdas were replaced in new tests by extracted GraphModules
  - verified models: `ResidualMLPDAG`, `MiniUNet`, `MiniDenseNet`
  - full reference output matches extracted-partition DAG output with `torch.allclose`
  - local backward and optimizer step pass through PyTorch autograd
- checkpoint reconstruction:
  - added `src/shardgrid/runtime/checkpoint.py`
  - worker shard save records schema version, job id, graph fingerprint, plan id, training step, worker id, GPU index, owned partitions, canonical parameter ids, canonical buffer ids, state_dict keys, shape, dtype, and CPU tensors
  - checkpoint merge uses canonical ids, not rank or partition order
  - duplicate canonical ids are rejected
  - missing expected state_dict keys are rejected
  - MiniUNet shard consolidation verifies parameters and BatchNorm persistent buffers reconstruct the complete state_dict
- verification:
  - lint: `PYTHONPATH=src /home/yangjilei/anaconda3/bin/python -m ruff check src/shardgrid/runtime src/shardgrid/planner/generic_graph.py src/shardgrid/planner/planning_contract.py src/shardgrid/planner/partitioning.py src/shardgrid/planner/placement.py src/shardgrid/planner/__init__.py tests/unit/test_dag_runtime.py tests/unit/test_generic_graph_ir.py tests/unit/test_planning_contract.py tests/unit/test_joint_partition_placement.py tests/unit/test_partition_candidates.py tests/unit/test_train_automatic_plan.py`
  - result: `All checks passed`
  - tests: `PYTHONPATH=src /home/yangjilei/anaconda3/bin/python -m pytest tests/unit/test_dag_runtime.py tests/unit/test_generic_graph_ir.py tests/unit/test_planning_contract.py tests/unit/test_joint_partition_placement.py tests/unit/test_partition_candidates.py tests/unit/test_train_automatic_plan.py -q`
  - result: `58 passed`
- blocked before hardware:
  - production `examples/models/train_automatic_plan.py` still uses the legacy linear `torch.distributed.pipelining` runner
  - the production automatic path does not consume `extract_partition_graph()` or `LocalDAGRuntime`
  - real remote activation send/recv and remote gradient send/recv are not wired for generic DAG edges
  - therefore running the existing hardware gate would test the legacy linear path, not `GENERIC_DAG_RUNTIME_USED=true`
  - `REAL MULTI-HOST GENERIC DAG TRAINING + FULL MODEL CHECKPOINT RECONSTRUCTION = BLOCKED`

## 2026-09-05 Production Generic DAG Runtime Wiring Attempt

- production entrypoint separation:
  - added `examples/models/train_generic_dag.py`
  - `JobManager._launch_command_for_assignment()` now routes automatic plans with `generic_dag_runtime=true` to `train_generic_dag.py`
  - legacy automatic jobs still route to `examples/models/train_automatic_plan.py`
  - generic DAG requests now emit `generic_dag_runtime_used=true` and `legacy_stage_runtime_used=false`
  - the generic entrypoint fails closed with `PRODUCTION_GENERIC_DAG_TRANSPORT_NOT_WIRED` instead of silently using the legacy linear runner
- generic planning request:
  - `model.type=generic_dag` is accepted by the automatic planning workload
  - `model.parameters.zoo_model` selects a generic model zoo factory, defaulting to `mini_unet`
  - added `examples/train-generic-dag.yaml` to request MiniUNet generic DAG automatic planning
  - ExecutionPlan labels include `generic_dag_runtime_requested`
  - worker launch environment includes `SHARDGRID_GENERIC_DAG_RUNTIME_REQUESTED`
  - SSH launch environment now includes `SHARDGRID_WORKER_ID` for runtime evidence
- tensor transport:
  - added `src/shardgrid/runtime/transport.py`
  - implemented stable `tensor_tag(step, value_id, direction)`
  - implemented real `torch.distributed.send` / `recv` tensor helpers
  - local gloo two-process test verifies forward activation send/recv and backward gradient send/recv without file transport
- verification:
  - lint: `PYTHONPATH=src /home/yangjilei/anaconda3/bin/python -m ruff check src/shardgrid/runtime src/shardgrid/planner/generic_graph.py src/shardgrid/planner/planning_contract.py src/shardgrid/planner/partitioning.py src/shardgrid/planner/placement.py src/shardgrid/planner/__init__.py src/shardgrid/control/job_manager.py src/shardgrid/launchers/ssh.py examples/models/train_generic_dag.py tests/unit/test_runtime_transport.py tests/unit/test_dag_runtime.py tests/unit/test_generic_graph_ir.py tests/unit/test_planning_contract.py tests/unit/test_joint_partition_placement.py tests/unit/test_partition_candidates.py tests/unit/test_train_automatic_plan.py tests/integration/test_train_orchestration.py`
  - result: `All checks passed`
  - tests: `PYTHONPATH=src /home/yangjilei/anaconda3/bin/python -m pytest tests/unit/test_runtime_transport.py tests/unit/test_dag_runtime.py tests/unit/test_generic_graph_ir.py tests/unit/test_planning_contract.py tests/unit/test_joint_partition_placement.py tests/unit/test_partition_candidates.py tests/unit/test_train_automatic_plan.py tests/integration/test_train_orchestration.py::test_generic_dag_runtime_request_uses_generic_entrypoint tests/integration/test_train_orchestration.py::test_generic_dag_entrypoint_fails_closed_with_evidence --run-integration -q`
  - result: `62 passed`
- blocked before real hardware:
  - production generic DAG entrypoint is wired but intentionally fail-closed
  - no production `GenericDAGRuntimeAdapter` yet consumes `RuntimePlan` and schedules owned partition GraphModules across workers
  - remote activation/gradient helpers exist, but the distributed DAG scheduler and multi-consumer gradient aggregation are not wired to them
  - no real SSH multi-host MiniUNet generic DAG training was launched because it would stop at the fail-closed entrypoint
  - `REAL MULTI-HOST GENERIC DAG TRAINING + COMPLETE MODEL CHECKPOINT RECONSTRUCTION = BLOCKED`

## 2026-09-05 Production Generic DAG Runtime Closure Progress

- production generic DAG runner:
  - `examples/models/train_generic_dag.py` now loads `training-config.json` and `execution-plan.json` from the worker snapshot and executes the generic DAG path instead of failing closed.
  - `examples/train-generic-dag.yaml` no longer asks the user to write `stage_count`, `world_size`, `generic_dag_runtime`, or checkpoint consolidation settings; `model.type=generic_dag` and `planning.mode=automatic` are enough for the generic runtime path.
  - each rank rebuilds the model on `meta`, extracts only its owned partition GraphModules, materializes only owned module parameters, and records `full_model_real_materialized=false`.
  - one rank can own multiple partitions; the smoke test uses `rank0 -> P0,P2` and `rank1 -> P1,P3`.
- distributed DAG scheduler:
  - forward receives missing remote boundary values by canonical `value_id`, executes local partition GraphModules in logical order, and sends remote activations with stable tags.
  - backward receives remote output gradients, sums multi-consumer gradients per value before backward, and sends boundary gradients back to producer ranks.
  - boundary sends use `torch.distributed.isend` with explicit wait at step end to avoid deadlock when producer/consumer value order differs.
- checkpoint closure:
  - worker shards still save only owned parameters and persistent buffers.
  - shard consolidation now rejects mismatched `job_id`, graph fingerprint, plan id, or training step.
  - `JobManager` now defaults generic DAG jobs to required checkpoint consolidation and writes `checkpoint/model-state.pt` on the Control Plane.
  - generic DAG consolidation validates missing keys, duplicate ids, tensor shape, tensor dtype, strict `load_state_dict`, and eval forward.
- verification:
  - lint: `PYTHONPATH=src /home/yangjilei/anaconda3/bin/python -m ruff check examples/models/train_generic_dag.py src/shardgrid/runtime src/shardgrid/control/job_manager.py tests/unit/test_runtime_transport.py tests/unit/test_dag_runtime.py tests/integration/test_train_orchestration.py`
  - result: `All checks passed`
  - local runtime tests: `PYTHONPATH=src /home/yangjilei/anaconda3/bin/python -m pytest tests/unit/test_runtime_transport.py tests/unit/test_dag_runtime.py -q`
  - result: `10 passed`
  - regression tests: `PYTHONPATH=src /home/yangjilei/anaconda3/bin/python -m pytest tests/unit/test_runtime_transport.py tests/unit/test_dag_runtime.py tests/unit/test_generic_graph_ir.py tests/unit/test_planning_contract.py tests/unit/test_joint_partition_placement.py tests/unit/test_partition_candidates.py tests/unit/test_train_automatic_plan.py tests/integration/test_train_orchestration.py::test_generic_dag_runtime_request_uses_generic_entrypoint tests/integration/test_train_orchestration.py::test_generic_dag_entrypoint_missing_snapshot_fails_closed_with_evidence --run-integration -q`
  - result: `64 passed`
  - simplified config load: `PYTHONPATH=src /home/yangjilei/anaconda3/bin/python - <<'PY' ... load_training_config('examples/train-generic-dag.yaml') ... PY`
  - result: `yaml config load: PASS`
- hardware status:
  - real SSH multi-host MiniUNet generic DAG training was not launched in this turn.
  - current verified boundary is local two-process Gloo execution of the production runner, including activation transfer, gradient transfer, optimizer step, checkpoint shard write, and failure diagnostics.

## 2026-09-05 Real SSH Generic DAG Hardware Validation

- initial cluster state:
  - healthy physical workers: `gpu4060@10.87.5.155`, `gpu1060@10.87.5.15`, `gpu4060-cqupt@10.87.5.214`
  - GPU free memory before run: `gpu4060=7957 MiB`, `gpu1060=3952 MiB`, `gpu4060-cqupt=7026 MiB`
  - no `train_generic_dag.py`, `train_automatic_plan.py`, `train_pipeline.py`, or `torchrun` PIDs were present before launch
- strict simple command:
  - command: `PYTHONPATH=src /home/yangjilei/anaconda3/bin/python -m shardgrid.cli.app --config examples/workers.yaml --json train examples/train-generic-dag.yaml`
  - result: `FAIL`
  - job: `job-20260905104354-40e2d4a6`
  - reason: automatic planning selected `selected_worker_count=1` for tiny MiniUNet; the final hardware gate requires at least two physical hosts
- two-worker gated hardware run:
  - command: `SHARDGRID_AUTOMATIC_MIN_WORKERS=2 PYTHONPATH=src /home/yangjilei/anaconda3/bin/python -m shardgrid.cli.app --config examples/workers.yaml --json train examples/train-generic-dag.yaml`
  - result: `PASS`
  - job: `job-20260905104627-96193915`
  - selected workers: `gpu4060@10.87.5.155 rank0 pid=629`, `gpu4060-cqupt@10.87.5.214 rank1 pid=10795`
  - plan: `stage0/P0 -> gpu4060`, `stage1/P1 -> gpu4060-cqupt`
  - runtime: `generic_dag_runtime_used=true`, `legacy_stage_runtime_used=false`, `full_model_real_materialized=false`
  - training: `20` completed forward/backward/optimizer steps, final loss `0.2002612054347992`, loss finite
  - remote forward: rank0 recorded `activation_remote_edges=60`
  - remote backward: rank1 recorded `gradient_remote_edges=60`
  - materialization: rank0 owned `P0` with `24576` parameter bytes; rank1 owned `P1` with `33736` parameter bytes
  - checkpoint shards gathered automatically: `2/2`
  - final model: `/var/tmp/shardgrid/jobs/job-20260905104627-96193915/checkpoint/model-state.pt`, size `72548`, sha256 `49fdf06a8656d825290a2de3774091b16b2f5b454449c042392b40cad4f8182e`
  - reload: `strict_load_missing_keys=[]`, `strict_load_unexpected_keys=[]`, `validation_forward_passed=true`
- evidence gap:
  - both monitor files report `parameter_changed=true`
  - final `model-state.pt` payload reports `parameter_changed=false` because the checkpoint manifest does not attach worker checkpoint metadata to shard entries
- cleanup:
  - final reservations: `[]`
  - final training PIDs: none
  - final GPU compute apps: none
- issue package:
  - `shardgrid-real-generic-dag-hardware-issue-20260905.zip`
  - includes source, config, strict failed plan/status, two-worker plan/status, diagnostics, logs, checkpoint shards, model-state, GraphIR, LogicalPartitionPlan, PlacementPlan, WorkerOwnershipPlan, and `TEST_RESULTS.txt`

## 2026-09-05 Generic DAG Final Hardware Closure

- normal CLI remains simple:
  - `examples/train-generic-dag.yaml` still has no `world_size`, `stage_count`, explicit worker list, rank mapping, placement, or min-worker setting.
  - command: `env -u SHARDGRID_AUTOMATIC_MIN_WORKERS -u SHARDGRID_AUTOMATIC_MAX_WORKERS PYTHONPATH=src /home/yangjilei/anaconda3/bin/python -m shardgrid.cli.app --config examples/workers.yaml --json train examples/train-generic-dag.yaml --dry-run`
  - result: `PASS`; default automatic planning selected `selected_worker_count=1` for tiny MiniUNet, proving production planning is not hard-coded to two workers.
- hardware gate constraint:
  - added `JobManager.run(..., min_selected_physical_hosts=2)` as an internal/test-only planning constraint.
  - CLI does not expose a new user parameter and does not require `SHARDGRID_AUTOMATIC_MIN_WORKERS=2`.
  - terminal snapshot metadata preserves `planner_worker_count_bounds.constraint_source=hardware_gate` and `min_selected_physical_hosts=2`.
- checkpoint evidence propagation:
  - worker checkpoint shards now store `parameter_changed`, `changed_parameter_count`, `checked_parameter_count`, and changed trainable parameter names in shard metadata.
  - checkpoint merge now produces `training_evidence.worker_parameter_changed`, `training_evidence.any_parameter_changed`, and `training_evidence.all_trainable_workers_parameter_changed`.
  - final `checkpoint/model-state.pt` defines legacy `parameter_changed` as `training_evidence.any_parameter_changed`.
  - merge still validates consistent `job_id`, `graph_fingerprint`, `plan_id`, `training_step`, expected keys, tensor shapes, tensor dtypes, strict reload, and CPU forward.
- local verification:
  - lint: `PYTHONPATH=src /home/yangjilei/anaconda3/bin/python -m ruff check src/shardgrid/runtime/checkpoint.py examples/models/train_generic_dag.py src/shardgrid/control/job_manager.py tests/unit/test_runtime_transport.py tests/unit/test_dag_runtime.py tests/integration/test_train_orchestration.py tests/multi_host/test_generic_dag_hardware_gate.py`
  - result: `All checks passed`
  - tests: `PYTHONPATH=src /home/yangjilei/anaconda3/bin/python -m pytest tests/unit/test_runtime_transport.py tests/unit/test_dag_runtime.py tests/integration/test_train_orchestration.py::test_automatic_worker_count_bounds_default_to_all_available_workers tests/integration/test_train_orchestration.py::test_hardware_gate_worker_count_constraint_does_not_change_default tests/integration/test_train_orchestration.py::test_automatic_worker_count_bounds_honor_test_override tests/integration/test_train_orchestration.py::test_automatic_worker_count_bounds_honor_max_worker_override tests/integration/test_train_orchestration.py::test_generic_dag_runtime_request_uses_generic_entrypoint tests/integration/test_train_orchestration.py::test_generic_dag_entrypoint_missing_snapshot_fails_closed_with_evidence --run-integration -q`
  - result: `16 passed`
- final real hardware gate:
  - command: `SHARDGRID_ENABLE_HARDWARE_TESTS=1 SHARDGRID_ENABLE_MULTI_HOST_TESTS=1 PYTHONPATH=src /home/yangjilei/anaconda3/bin/python -m pytest tests/multi_host/test_generic_dag_hardware_gate.py --run-hardware --run-multi-host -q`
  - result: `1 passed in 117.39s`
  - job: `job-20260905111457-4da9e16c`
  - selected workers: `gpu4060@10.87.5.155` and `gpu4060-cqupt@10.87.5.214`
  - selected physical machines: `machine-c` and `machine-e`
  - selected candidate: `galvatron:mini-unet:0:8-8:14`
  - selected worker count: `2`
  - attempted worker counts: `[2]`
  - cross-worker communication bytes: `131072`
  - runtime: `generic_dag_runtime_used=true`, `legacy_stage_runtime_used=false`, `full_model_real_materialized=false`
  - training: both ranks completed `20` optimizer steps; final loss `0.20026114583015442`; loss finite
  - remote activation evidence: `60` remote activation edges
  - remote gradient evidence: `60` remote gradient edges
  - checkpoint shards: `2/2` gathered
  - shard parameter evidence: `gpu4060 12/12 changed`, `gpu4060-cqupt 14/14 changed`
  - final model: `/var/tmp/shardgrid/jobs/job-20260905111457-4da9e16c/checkpoint/model-state.pt`
  - final model size: `74340` bytes
  - final model sha256: `10556c3a719648df878bf647375b2486394b6252d7ec76ff903422472d649bfa`
  - final metadata: `parameter_changed=true`, `any_parameter_changed=true`, `all_trainable_workers_parameter_changed=true`
  - reload: `strict_load_missing_keys=[]`, `strict_load_unexpected_keys=[]`, `validation_forward_passed=true`
- cleanup:
  - `/var/tmp/shardgrid/jobs/resource-reservations.json` has `reservations=[]`
  - `gpu4060`, `gpu1060`, and `gpu4060-cqupt` report no GPU compute apps or ShardGrid training PIDs after the gate
