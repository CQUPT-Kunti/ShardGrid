# Compatibility: DeepSpeed Pipeline (T062)

Status: SPIKED - BLOCKED on current WSL2 two-host environment | Updated: 2026-08-22

T062 evaluates DeepSpeed Pipeline as a fallback only when Galvatron is
insufficient.  Per the T061 decision, Galvatron remains the MVP engine
(explicit parallel configuration); DeepSpeed Pipeline was still evaluated as
required by the fallback sequence.

## What was evaluated

- DeepSpeed 0.19.5 (official PyPI source) installed in the `shardgrid` Conda
  environment on both Workers (RTX 4060 `10.87.5.155`, GTX 1650 `10.87.5.15`).
  Install did not replace torch 2.7.1+cu118 / nvidia-cu11 stack (verified by
  pip dry-run before install).
- Smallest possible pipeline: 2 stages over 2 physical hosts with 1 GPU per
  host (`pp=2`), tiny synthetic model (hidden 128, 2 layers, micro batch 4),
  launched through the existing SSH + WSL2 + selected Conda chain with
  explicit `RANK`/`WORLD_SIZE`/`MASTER_ADDR` env (the proven T058 launch
  shape; no `torch.distributed.launch`, no DeepSpeed launcher SSH).

## Result: BLOCKED

| Step | Result | Evidence |
|------|--------|----------|
| deepspeed version | PASS | 0.19.5 on both Workers, torch 2.7.1+cu118 |
| PipelineModule construction | PASS | topology `{pipe=0,data=0}:0` / `{pipe=1,data=0}:1`, one stage per rank |
| deepspeed.initialize | PASS | `DEEPSPEED_PIPELINE_INIT_OK` on rank 0 |
| post-init barrier | PASS | both ranks reached `DEEPSPEED_PIPELINE_BARRIER_OK` |
| engine.train_batch | **BLOCKED** | rank0 stalled in `RecvGrad`; rank1 stalled earlier in `LoadMicroBatch` |
| native torch NCCL isend/irecv (same hosts) | PASS | 3.9s, tensor transferred correctly |

Blocker note:

- `engine.train_batch` still deadlocks on the WSL2 two-physical-host NCCL
  setup even though the native torch NCCL point-to-point (`isend`/`irecv`) and
  all collectives (broadcast / all_reduce / barrier / send-recv) succeed on
  the exact same hosts, interfaces, and process group configuration.
- This is a DeepSpeed pipeline scheduling/communication behavior on this
  environment, not a Galvatron failure and not a missing ShardGrid component.

Rework diagnosis (2026-08-22):

- The block is **not** at initialization and **not** at the barrier after
  `deepspeed.initialize()`: both ranks emit `DEEPSPEED_PIPELINE_BARRIER_OK`.
- Rank 0 enters `train_batch()` and progresses until
  `step=3 cmd=RecvGrad kwargs={'buffer_id': 0}`, where it blocks waiting for
  the next stage.
- Rank 1 enters `train_batch()`, completes `RecvActivation`, then blocks
  earlier in `LoadMicroBatch`; its last stdout marker is
  `data_iterator_next_begin`, and faulthandler points into DeepSpeed
  `_exec_load_micro_batch`.
- Therefore the observed deadlock is downstream of rank 1's last-stage
  micro-batch loading path; rank 0's `RecvGrad` wait is secondary.

Resolver / iterator rework (2026-08-22):

- T062 live test originally re-implemented interface discovery locally with
  `ip route get + regex`.
- The project already had a resolver in
  `src/shardgrid/network/probe.py::discover_interface`; the final T062 live
  validation reuses that implementation instead of a test-local parser.
- Final live validation recorded interface discovery PASS with:
  - rank0 `10.87.5.155 -> eth3`
  - rank1 `10.87.5.15 -> eth0`
- The synthetic iterator was changed from per-`next()` CUDA allocation to
  pre-generated CPU micro-batches consumed by a simple synchronous iterator.
- Result: the real two-host run still blocked in the same place:
  rank1 `LoadMicroBatch`, then rank0 `RecvGrad`.

LoadMicroBatch-focused rework (2026-08-22):

- DeepSpeed 0.19.5 source on the live worker was inspected directly:
  - `_next_batch()` only performs `next(self.data_iterator)` plus optional
    `batch_fn`
  - `_exec_load_micro_batch()` calls `_next_batch()` before any first-stage
    input or last-stage label `.to(device)` transfer
- A narrower T062 diagnostic pass was added to distinguish sub-steps inside
  `LoadMicroBatch`, then one real two-host validation was executed.
- Result of that real run was still `BLOCKED`:
  - initialize PASS
  - barrier PASS
  - rank1 still did not complete `LoadMicroBatch`
  - rank1 never reached forward / backward / SendGrad
  - rank0 remained blocked in `RecvGrad`

Instruction-map diagnosis attempt (2026-08-23):

- DeepSpeed 0.19.5 source inspection confirmed `_exec_schedule()` dispatches
  through `PipelineEngine._INSTRUCTION_MAP`, so patching only
  `PipelineEngine._exec_load_micro_batch` was insufficient for sub-step
  diagnostics.
- The T062 harness was updated so wrapped diagnostic methods are also written
  back into `PipelineEngine._INSTRUCTION_MAP`.
- The single real live diagnosis run after that change did **not** produce new
  valid pipeline-step evidence because rank0 failed before starting the Python
  payload:
  `python: can't open file '/tmp/deepspeed_pipeline_spike.py'`
- Therefore that run does not refine the `LoadMicroBatch` root cause; it only
  proves a T062 harness/runtime file staging failure on that attempt.

Port / firewall investigation (2026-08-22):

- Windows firewall inbound allow rules inspected on both hosts: RTX 4060
  allows only `ShardGrid TCP 29500`; GTX 1650 allows 29500 plus several
  ranges (44620-48715, 50000-51000, etc.).
- During the deadlock, `ss -tnp` on both WSL hosts showed every cross-host TCP
  connection in state ESTABLISHED (store on 29500 plus NCCL data channels on
  random high ports 44xxx-48xxx), with no SYN-SENT or blocked connections.
- Conclusion: the deadlock is NOT a firewall/port issue; NCCL data-plane
  connections are fully established (simultaneous-open socket transport).  The
  block sits in DeepSpeed pipeline scheduling (both ranks wait on pipeline
  send/recv that never completes).

## Conclusion

- DeepSpeed Pipeline is **not usable** on the current WSL2 two-host
  one-GPU-per-host environment (blocked at `engine.train_batch`).
- The best current evidence localizes the block to the last stage's
  `LoadMicroBatch` path during `train_batch()`, with the previous stage then
  waiting in `RecvGrad`.
- Reusing the project interface resolver and removing dynamic CUDA allocation
  from the synthetic iterator did not change the real outcome.
- The latest T062 rework still does not prove whether the first sub-block
  inside rank1 `LoadMicroBatch` is exactly `next(data_iterator)` or the later
  label `.to(device)` transfer; it only proves that rank1 does not exit
  `LoadMicroBatch` before rank0 blocks in `RecvGrad`.
- The August 23, 2026 instruction-map diagnosis run failed earlier, at T062
  runtime file staging on rank0, so it adds no stronger conclusion about the
  underlying pipeline hang.
- No custom full pipeline engine was introduced because of a Galvatron
  failure; Galvatron remains the selected engine per T061.
- T065 (parallel engine decision) records Galvatron as the MVP path with
  explicit parallel configuration.

## Evidence

- `src/shardgrid/engines/deepspeed_pipeline.py` (spike harness + classifier)
- `tests/integration/test_deepspeed_pipeline_spike.py` (8 logic + 1 live)
- Live evidence: `/var/tmp/shardgrid/engines/deepspeed-pipeline-latest.json`
