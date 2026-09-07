# Compatibility: PyTorch Pipeline (T063)

Status: SPIKED - SUPPORTED on current two-host environment | Updated: 2026-08-22

T063 evaluates the mature PyTorch pipeline option for the actual torch
version and the two-physical-host one-GPU-per-host runtime, before any custom
model-parallel code is considered (acceptance criterion of T063).

## What was evaluated

- API: `torch.distributed.pipelining` (the supported pipeline API of
  torch 2.7.1; the legacy `torch.distributed.pipeline.sync.Pipe` module was
  removed in torch 2.x and is NOT available).
- Minimal spike: GPipe schedule with 2 stages over 2 physical hosts, 1 GPU
  per host (RTX 4060 rank 0 / 10.87.5.155 eth3, GTX 1650 rank 1 /
  10.87.5.15 eth0), tiny synthetic model (2 blocks, hidden 128, micro batch
  4, 2 steps).
- Launched through the proven T058 chain (explicit `RANK`/`WORLD_SIZE`/
  `MASTER_ADDR` env + direct selected Conda python, per-host
  `NCCL_SOCKET_IFNAME` resolved via `ip route get <peer>`); rendezvous on the
  project-validated path (master 10.87.5.155:29500).

## Result: PASS (real two-host execution)

| Step | Result | Evidence |
|------|--------|----------|
| torch API availability | PASS | `torch.distributed.pipelining` present in torch 2.7.1+cu118 (PipelineStage, ScheduleGPipe) |
| stage construction | PASS | `PYTORCH_PIPELINE_STAGE_READY` on both ranks |
| GPipe schedule steps | PASS | `PYTORCH_PIPELINE_STEP_OK` on both ranks, 2 steps |
| completion + teardown | PASS | `PYTORCH_PIPELINE_DONE` on both ranks, elapsed 0.9s |

## Notes

- API shape of torch 2.7.1 `PipelineStage`: per-rank submodule +
  `stage_index` + `num_stages` (rank == stage index in pure pipeline);
  `ScheduleGPipe.step(input, target=...)` takes the input only on the first
  stage.
- In contrast, the DeepSpeed Pipeline spike (T062) is BLOCKED on the same
  hosts (train_batch deadlock); the native PyTorch pipeline completes in
  under a second.  PyTorch's mature pipeline API is therefore the supported
  pipeline path for this MVP environment.

## Evidence

- `src/shardgrid/engines/pytorch_pipeline.py` (spike harness + classifier)
- `tests/integration/test_pytorch_pipeline_spike.py` (6 logic + 1 live)
- Live evidence: `/var/tmp/shardgrid/engines/pytorch-pipeline-latest.json`