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
