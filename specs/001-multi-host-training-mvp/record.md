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
