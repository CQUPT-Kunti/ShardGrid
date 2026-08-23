# Parallel Engine Decision (T065)

Status: DECIDED | Updated: 2026-08-22

T065 closes the parallel-engine candidate decision based on the real evidence
from T061-T064 and Gate 1 / Gate 2.  No new framework experiments were run;
the decision reuses the published compatibility documents and evidence
files.

## Decision

**Galvatron v2.4.0 is the selected MVP parallel engine**, used with explicit
parallel configuration (one stage per physical host, small stage budgets), as
proven by:

- T058: one-GPU-per-physical-host placement `true_multi_host`
- T059: heterogeneous stage placement ACCEPTED (RTX 4060 <= 2 GiB / stage,
  GTX 1650 <= ~1 GiB / stage)
- T060: pipeline construction / runtime launch / checkpoint / model profiler
  PASS (real GPU execution)
- Gate 1 (T052) and Gate 2 (T053): PASS in the current environment
- T061 decision report: `examples/compatibility/galvatron-report.json`

## Engine status matrix

| Engine | Role | Status | Real evidence |
|--------|------|--------|---------------|
| Galvatron v2.4.0 | primary | **SELECTED** | T058/T059/T060 live two-host runs; Gate 1/2 |
| PyTorch `torch.distributed.pipelining` | fallback candidate | **SUPPORTED** | T063 two-host GPipe spike PASS (0.9s) |
| DeepSpeed Pipeline 0.19.5 | fallback candidate | **BLOCKED** | T062 two-host `engine.train_batch` deadlock (native torch NCCL isend/irecv works; TCP all ESTABLISHED) |
| nnScaler 0.8 | fallback candidate | **BLOCKED / NOT_SELECTED** | T064 official install would replace torch 2.7.1+cu118 -> 2.6.0 + full nvidia-cu12 stack |

## Verification (against Gate 2 and hardware evidence)

- Gate 2 PASS and the NCCL/Gloo two-host links are prerequisites; the
  decision above is consistent with them.
- Every engine status comes from a real execution record under
  `/var/tmp/shardgrid/engines/` (galvatron-*, deepspeed-pipeline-*,
  pytorch-pipeline-*, nnscaler-* latest files) and
  `/var/tmp/shardgrid/distributed/reports/` (gate1/gate2/dist-test), never
  from mock or historical assumptions.
- Static/explicit planning path (no profiler-driven search) is labeled
  **limited MVP support** (hardware profiler BLOCKED_BY_WSL2_CUPTI).

## Non-goals (acceptance criterion)

ShardGrid does NOT reimplement autograd, collective communication, CUDA
kernels, CUDA allocators, or a full pipeline framework.  All of these are
delegated to PyTorch, NCCL/Gloo, and the selected engine's documented APIs.
The decision therefore never authorizes custom model-parallel code.

## Output for T066/T067 (adapter input)

- MVP default engine: `galvatron`
- Registerable/usable engines on this environment: `galvatron`,
  `pytorch_pipeline`
- Engines to mark BLOCKED in the registry: `deepspeed_pipeline`, `nnscaler`
- Upper layers must not hard-code any framework; selection stays
  adapter-driven via the engine names above.

## Evidence

- `examples/compatibility/parallel-engine-decision.json` (machine-readable
  decision)
- Supporting docs: `docs/compatibility/galvatron.md` (T061), `deepspeed-pipeline.md`
  (T062), `pytorch-pipeline.md` (T063), `nnscaler.md` (T064)