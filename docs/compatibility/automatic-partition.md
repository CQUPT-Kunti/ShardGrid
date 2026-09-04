# Automatic Partition Compatibility

## 2026-09-03 Automatic Plan Real Training Hardware Gate

Status: `IMPLEMENTED_PENDING_LIVE_EVIDENCE`

Current automatic live path:

```text
shardgrid train
-> planning.mode=automatic
-> T109/T110/T111/T112
-> ParallelPlan
-> ExecutionPlan
-> launch-time resource revalidation
-> fresh master port allocation
-> SSHLauncher
-> examples/models/train_automatic_plan.py
-> real distributed pipeline training
```

What is implemented:

- automatic live runs no longer reuse `examples/models/train_pipeline.py`
- automatic assignments launch `examples/models/train_automatic_plan.py`
- the runner rebuilds the full supported model and splits it from saved `ParallelPlan.stage_metadata`
- launch-time revalidation checks worker health, backend availability, route/interface evidence, and current free memory against `estimated_peak_training_memory`
- resource drift fails closed with `RESOURCE_CHANGED`
- a fresh rendezvous port is allocated on the selected rank-0 worker before launch
- checkpoint metadata preserves `partition_source`, `selected_candidate_id`, `selected_worker_count`, and `stage_to_worker`
- hardware gate entrypoint: `tests/multi_host/test_automatic_partition_gate.py`

Live gate command:

```bash
SHARDGRID_RUN_AUTOMATIC_HW=1 \
python -m pytest tests/multi_host/test_automatic_partition_gate.py \
  --run-hardware -q
```

Live evidence required for PASS:

- automatic CLI path
- `partition_source=automatic`
- planner-selected workers only
- SSH launch success
- finite loss
- backward and optimizer step success
- parameter checksum change
- checkpoint manifest + checkpoint metadata written

Current limitation:

- this repository turn verified the implementation and local regressions only
- no live worker run was executed on 2026-09-03 in this environment
