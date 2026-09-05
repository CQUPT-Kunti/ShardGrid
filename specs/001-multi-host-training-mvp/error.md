# Errors

## 2026-09-05 Generic Model Auto-Partition + Execution

- status: `BLOCKED`
- blocker:
  - automatic production execution still contains model-type branches in `examples/models/train_automatic_plan.py`
  - `large_residual_transformer` still uses `LargeResidualTransformerStage`, `build_large_residual_transformer_stage()`, `_path_uses()`, and `_path_defs()`
  - generic stage-local `GraphModule` extraction/materialization is not implemented
  - generic distributed pipeline forward/backward/optimizer execution is not implemented
- completed before blocker:
  - generic FX graph IR with value IDs
  - graph-derived module dependency extraction for planner support
  - boundary value inference for cross-stage producer/consumer edges
  - generic model zoo factories without stage builders
  - bounded worker-subset candidate search for large GPU counts
- failure category: `STAGE_EXTRACTION_FAILED`
- next required fix:
  - replace model-specific runtime stage materialization with a generic graph submodule extractor that materializes only owned stage parameters and preserves boundary values by ID

## 2026-09-05 DAG Planning Contract

- status: `BLOCKED`
- completed before blocker:
  - `CanonicalGraphIR` schema version and fingerprint
  - graph capture adapter/result interface
  - logical partition plan and placement plan split
  - resource snapshot and runtime capabilities contract
  - bounded placement candidate search remains in place
- blocker:
  - the new planning contract is not yet the production execution compiler path
  - runtime still cannot execute generic DAG stage-local GraphModules
  - workers still do not materialize arbitrary partition-owned parameters/buffers through this contract
  - real hardware and multi-job stress gates therefore cannot pass through the fully generic path
- failure category: `STAGE_EXTRACTION_FAILED`
- next required fix:
  - implement `extract_partition_graph()` and a runtime adapter that consumes `LogicalPartitionPlan + PlacementPlan` to compile and execute a real `ExecutionPlan`

## 2026-09-05 Memory-Aware Multi-Plan DAG Planning

- status: `BLOCKED`
- completed before blocker:
  - memory-aware logical partition candidate generation from `ResourceSnapshot.free_memory_bytes`
  - fixed `8` logical partition default removed
  - `>8` partition candidate generation verified
  - bounded partition candidate, placement candidate, total plan, top-K, and beam search controls
  - multiple placement candidates through lightweight beam states
  - scoring based on memory slack, partition count, and cross-GPU communication
  - meta model factory capture path with real parameter storage byte evidence
- blocker:
  - production automatic runtime still does not consume the new multi-plan `PlanningResult`
  - no generic `extract_partition_graph()` yet
  - no generic stage-local parameter/buffer materialization yet
  - real hardware and multi-job stress validation cannot pass through this generic path yet
- failure category: `STAGE_EXTRACTION_FAILED`
- next required fix:
  - connect the selected `LogicalPartitionPlan + PlacementPlan` to a PlanCompiler and generic runtime adapter

## 2026-09-05 Multi-Partition Worker DAG Runtime

- status: `BLOCKED`
- completed before blocker:
  - `WorkerOwnershipPlan` generation from `PlacementPlan`
  - multiple logical partitions can be assigned to one `worker_id/gpu_index/gpu_id`
  - partition-set-local parameter and buffer ownership ids are deduplicated in the runtime contract
  - graph edges are classified as local or remote from placement
  - local `ValueStore` releases values after their last consumer
  - local DAG forward, PyTorch autograd backward, and optimizer-step smoke test pass
  - non-contiguous same-GPU placement is covered by unit evidence
- blocker:
  - no generic `extract_partition_graph()` exists yet to build real executable partition `GraphModule`s from arbitrary `CanonicalGraphIR` partitions
  - production `examples/models/train_automatic_plan.py` is still not wired to this generic DAG runtime path
  - remote edge transport and cross-worker activation/gradient exchange are not implemented for the generic DAG runtime
  - real hardware multi-job validation cannot pass through this path yet
- failure category: `GENERIC_REMOTE_DAG_RUNTIME_NOT_WIRED`
- next required fix:
  - implement generic partition GraphModule extraction/materialization, then connect the selected `LogicalPartitionPlan + PlacementPlan` to the SSH worker runtime without changing planner decisions

## 2026-09-05 Generic DAG Runtime + Full Checkpoint Reconstruction

- status: `BLOCKED`
- completed before blocker:
  - generic `extract_partition_graph()` implemented for FX backend graphs
  - partition GraphModule inputs are generated from stable value ids
  - partition GraphModule outputs are generated from stable value ids
  - local executable partition tests pass for `ResidualMLPDAG`, `MiniUNet`, and `MiniDenseNet`
  - local DAG full-reference output equivalence, backward, and optimizer-step checks pass
  - canonical buffer paths are now captured in `GraphNodeSpec`
  - worker checkpoint shard save and control-plane state_dict consolidation helpers are implemented
  - shard merge validates missing and duplicate canonical ids and includes persistent buffers
- blocker:
  - production automatic training still runs through `examples/models/train_automatic_plan.py` and `torch.distributed.pipelining`
  - production automatic path does not yet call `extract_partition_graph()` or use generic partition-set worker execution
  - remote activation transport for generic DAG edges is not wired
  - remote gradient transport and multi-consumer gradient aggregation for generic DAG edges are not wired
  - no real multi-host generic DAG training was launched because the available hardware gate would exercise the legacy linear path and produce misleading evidence
- failure category: `PRODUCTION_GENERIC_DAG_TRANSPORT_NOT_WIRED`
- next required fix:
  - replace the production automatic worker runner's linear stage path with a generic DAG runner that consumes `RuntimePlan`, executes owned partition GraphModules, sends/receives remote activations and gradients with `torch.distributed`, then writes worker state shards for control-plane consolidation

## 2026-09-05 Production Generic DAG Runtime Wiring

- status: `BLOCKED`
- completed before blocker:
  - generic DAG automatic requests are routed away from the legacy linear runner
  - `examples/models/train_generic_dag.py` exists as the production generic DAG entrypoint
  - generic DAG entrypoint emits `generic_dag_runtime_used=true` and `legacy_stage_runtime_used=false`
  - generic DAG entrypoint fails closed instead of pretending training passed
  - `model.type=generic_dag` can build a MiniUNet planning workload from the generic model zoo
  - SSH worker launch env now includes `SHARDGRID_WORKER_ID`
  - real `torch.distributed` tensor send/recv helpers exist for activations and gradients
  - local gloo two-process transport test passes for forward activation and backward gradient
- blocker:
  - production `GenericDAGRuntimeAdapter` does not exist yet
  - `train_generic_dag.py` does not yet load and execute `RuntimePlan`
  - owned partition GraphModules are not scheduled across real SSH workers
  - remote activation/gradient helpers are not connected to a production DAG scheduler
  - multi-consumer gradient aggregation is not connected to production backward execution
  - no worker checkpoint shards from real generic DAG multi-host training exist
  - no control-plane merged model-state.pt from real generic DAG multi-host training exists
- failure category: `PRODUCTION_GENERIC_DAG_SCHEDULER_NOT_IMPLEMENTED`
- next required fix:
  - implement the production `GenericDAGRuntimeAdapter` inside `train_generic_dag.py`: load the saved graph/logical/placement/runtime plans, build owned extracted partition GraphModules, run deterministic per-step forward/backward transport with `torch.distributed`, aggregate boundary gradients by value id, save real worker shards, and let the control plane gather/merge/reload the complete state_dict

## 2026-09-05 Real SSH Generic DAG Hardware Validation

- status: `PARTIAL`
- strict command:
  - `PYTHONPATH=src /home/yangjilei/anaconda3/bin/python -m shardgrid.cli.app --config examples/workers.yaml --json train examples/train-generic-dag.yaml`
  - result: `FAIL`
  - job: `job-20260905104354-40e2d4a6`
  - failure category: `INSUFFICIENT_PHYSICAL_HOSTS_SELECTED`
  - reason: MiniUNet is small enough that automatic planning selected `selected_worker_count=1`, while the final hardware gate requires at least two physical hosts
- constrained hardware run:
  - `SHARDGRID_AUTOMATIC_MIN_WORKERS=2 PYTHONPATH=src /home/yangjilei/anaconda3/bin/python -m shardgrid.cli.app --config examples/workers.yaml --json train examples/train-generic-dag.yaml`
  - result: `PASS`
  - job: `job-20260905104627-96193915`
  - evidence: real SSH launch, two physical workers, generic DAG runtime, remote activation edges, remote gradient edges, optimizer steps, checkpoint shards, Control Plane `model-state.pt`, strict reload, and model forward all passed
- remaining blocker for full PASS:
  - the default simple command still does not force or naturally require multi-host placement for tiny MiniUNet
  - final `model-state.pt` records `parameter_changed=false` even though both rank monitor diagnostics record `parameter_changed=true`; checkpoint manifest does not preserve the per-worker checkpoint metadata needed for that final aggregate evidence field
- issue package:
  - `shardgrid-real-generic-dag-hardware-issue-20260905.zip`

## 2026-09-05 Real SSH Generic DAG Hardware Validation Closure

- status: `RESOLVED`
- fixed:
  - normal CLI no longer depends on `SHARDGRID_AUTOMATIC_MIN_WORKERS=2`; default automatic planning remains free to select one GPU for tiny models.
  - multi-host hardware validation now uses an internal `min_selected_physical_hosts=2` planning constraint, recorded as `constraint_source=hardware_gate`.
  - worker checkpoint shards now persist trainable-parameter change evidence.
  - final `checkpoint/model-state.pt` now reports `parameter_changed=true` through `training_evidence.any_parameter_changed`.
- validation:
  - job: `job-20260905111457-4da9e16c`
  - command: `SHARDGRID_ENABLE_HARDWARE_TESTS=1 SHARDGRID_ENABLE_MULTI_HOST_TESTS=1 PYTHONPATH=src /home/yangjilei/anaconda3/bin/python -m pytest tests/multi_host/test_generic_dag_hardware_gate.py --run-hardware --run-multi-host -q`
  - result: `1 passed in 117.39s`
- remaining blocker: `NONE` for the final Generic DAG real SSH multi-host gate
