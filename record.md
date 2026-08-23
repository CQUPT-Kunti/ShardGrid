
## T062 - DeepSpeed Pipeline spike (2026-08-22)

### What was done

- Installed DeepSpeed 0.19.5 (official PyPI) into `shardgrid` env on both
  Workers (dry-run verified no torch/CUDA replacement; 4060 via 7890 proxy,
  1650 via its own 7897 proxy).
- Implemented `src/shardgrid/engines/deepspeed_pipeline.py`: spike harness
  (smallest pp=2 pipeline over 2 hosts x 1 GPU, marker parsing, status
  classifier PASS/FAIL/BLOCKED, evidence saving).
- Implemented `tests/integration/test_deepspeed_pipeline_spike.py` (8 logic
  + 1 live) and `docs/compatibility/deepspeed-pipeline.md`.
- Marked T062 `[X]` in tasks.md.

### Files changed

- new: `src/shardgrid/engines/deepspeed_pipeline.py`
- new: `tests/integration/test_deepspeed_pipeline_spike.py`
- new: `docs/compatibility/deepspeed-pipeline.md`
- modified: `specs/001-multi-host-training-mvp/tasks.md`

### Test results

- Logic tests: 8 passed; ruff clean.
- Live spike (RTX 4060 rank0 + GTX 1650 rank1, pp=2): result BLOCKED (real):
  - deepspeed 0.19.5 import PASS both Workers
  - PipelineModule topology + deepspeed.initialize PASS (INIT_OK marker)
  - engine.train_batch deadlocked >90s (two-host WSL2 NCCL pipeline)
  - control: native torch NCCL isend/irecv on the same hosts = PASS (3.9s)
- Evidence: `/var/tmp/shardgrid/engines/deepspeed-pipeline-latest.json`
- Conclusion: DeepSpeed Pipeline not usable on current WSL2 two-host env;
  no custom pipeline engine introduced; Galvatron remains MVP engine (T061).

## T062 Rework - DeepSpeed Pipeline hang diagnosis (2026-08-22)

### What was done

- Re-read `tasks.md`, the T062 spike harness, compatibility doc, and `error.md`
  before changing anything.
- Reproduced the current live two-host DeepSpeed pipeline test once without
  changing training logic; result stayed `BLOCKED`.
- Added observation-only diagnostics to
  `src/shardgrid/engines/deepspeed_pipeline.py`:
  post-init barrier marker, per-rank timestamped pipeline instruction logs,
  `data_iterator` progress logs, and Python `faulthandler` timeout dumps.
- Re-ran the real two-host live spike with the same pp=2 / 2-host / 1-GPU
  shape and inspected the saved evidence JSON.
- Added one narrower observation point inside `data_iterator`
  (`batch_ready` / `target_ready`) to separate batch-vs-target creation, but a
  subsequent rerun hit a new interface-discovery environment error before the
  pipeline launch and work stopped there per instructions.

### Root-cause evidence gathered

- The hang is **not** at `deepspeed.initialize()` and **not** at the
  post-init `dist.barrier()`: both ranks reached
  `DEEPSPEED_PIPELINE_BARRIER_OK`.
- Rank 0 then entered `train_batch()` and progressed until
  `step=3 cmd=RecvGrad`, where it blocked waiting for the next stage.
- Rank 1 entered `train_batch()`, completed `RecvActivation`, then blocked
  earlier in `LoadMicroBatch`; its last stdout marker was
  `data_iterator_next_begin`, and the faulthandler stack pointed into
  DeepSpeed `_exec_load_micro_batch`.
- This means the observed deadlock is caused by rank 1 never reaching its
  forward/loss/backward/send-grad path; rank 0's `RecvGrad` stall is
  downstream, not primary.

### Final result

- DeepSpeed Pipeline status remains `BLOCKED`.
- No targeted fix was validated because the next rerun stopped at a new
  environment/network entry error (`ip route get 10.87.5.155` on 10.87.5.15
  returned no interface), and the rework rules required stopping immediately.

## T062 Rework - interface resolver + CPU iterator verification (2026-08-22)

### What was done

- Re-checked T062 against:
  `tasks.md`, `src/shardgrid/engines/deepspeed_pipeline.py`,
  `tests/integration/test_deepspeed_pipeline_spike.py`,
  `docs/compatibility/deepspeed-pipeline.md`, and the live evidence JSON.
- Confirmed T062 live test had re-implemented interface discovery locally with
  `ip route get + regex`.
- Confirmed the project already had a resolver in
  `src/shardgrid/network/probe.py::discover_interface`.
- Replaced the T062 live test's local interface parsing with that project
  resolver.
- Replaced the spike's synthetic iterator from per-`next()` CUDA allocation to
  pre-generated CPU micro-batches consumed by a simple synchronous iterator.
- Ran local checks:
  - `pytest tests/integration/test_deepspeed_pipeline_spike.py -q -k 'not live'`
  - `ruff check src/shardgrid/engines/deepspeed_pipeline.py tests/integration/test_deepspeed_pipeline_spike.py`
- Ran one real two-host validation:
  `pytest tests/integration/test_deepspeed_pipeline_spike.py::test_live_deepspeed_pipeline_spike -q --run-integration --run-hardware --run-multi-host`

### Result

- Interface discovery: PASS via project resolver
  - rank0 `10.87.5.155 -> eth3`
  - rank1 `10.87.5.15 -> eth0`
- Data iterator change did **not** clear the hang.
- Real run stayed `BLOCKED`:
  - initialize PASS
  - post-init barrier PASS
  - rank0 stalled in `RecvGrad`
  - rank1 stalled earlier in `LoadMicroBatch`
  - `engine.train_batch()` did not return
- Evidence: `/var/tmp/shardgrid/engines/deepspeed-pipeline-latest.json`

## T062 Rework - rank1 LoadMicroBatch diagnosis (2026-08-22)

### What was done

- Inspected DeepSpeed 0.19.5 source on the live 1650 worker for:
  - `PipelineEngine._next_batch()`
  - `PipelineEngine._exec_load_micro_batch()`
- Confirmed from source that `_next_batch()` only does:
  `next(self.data_iterator)` plus optional `batch_fn`, and that
  `_exec_load_micro_batch()` calls `_next_batch()` before any stage-local
  `.to(device)` transfer.
- Added narrower T062 harness diagnostics intended to separate:
  - before / after `_next_batch()`
  - batch structure
  - before / after stage-local `.to(device)`
  - `LoadMicroBatch` exit
- Ran local syntax/lint checks:
  - `python -m py_compile src/shardgrid/engines/deepspeed_pipeline.py tests/integration/test_deepspeed_pipeline_spike.py`
  - `ruff check src/shardgrid/engines/deepspeed_pipeline.py tests/integration/test_deepspeed_pipeline_spike.py`
- Ran one real two-host validation:
  `pytest tests/integration/test_deepspeed_pipeline_spike.py::test_live_deepspeed_pipeline_spike -q --run-integration --run-hardware --run-multi-host`

### Result

- Real run remained `BLOCKED`.
- initialize PASS
- barrier PASS
- rank1 still did not complete `LoadMicroBatch`
- rank1 did not reach forward / backward / SendGrad
- rank0 remained blocked in `RecvGrad`
- `engine.train_batch()` did not return

### Best evidence from this run

- Live evidence file:
  `/var/tmp/shardgrid/engines/deepspeed-pipeline-latest.json`
- rank1 last visible progress:
  `instr_begin step=1 cmd=LoadMicroBatch kwargs={'buffer_id': 0}`
  then `data_iterator_next_begin`
- rank0 last visible progress:
  `instr_begin step=3 cmd=RecvGrad kwargs={'buffer_id': 0}`

### Conclusion

- T062 remains `BLOCKED`.
- This rework still localizes the first observed pipeline stall to rank1
  `LoadMicroBatch`, with rank0 `RecvGrad` blocked downstream.

## T062 Rework - instruction-map diagnosis attempt (2026-08-23)

### What was done

- Re-checked the live worker's DeepSpeed 0.19.5 `PipelineEngine._INSTRUCTION_MAP`
  and confirmed `_exec_schedule()` dispatches through that map, not by calling
  the `PipelineEngine._exec_*` attributes directly.
- Confirmed this made the previous T062 `LoadMicroBatch` sub-step markers
  incomplete: patching `PipelineEngine._exec_load_micro_batch` alone was not
  sufficient for live diagnostics.
- Updated the T062 harness so the wrapped diagnostic methods are also written
  back into `PipelineEngine._INSTRUCTION_MAP`.
- Ran local checks:
  - `python -m py_compile src/shardgrid/engines/deepspeed_pipeline.py tests/integration/test_deepspeed_pipeline_spike.py`
  - `ruff check src/shardgrid/engines/deepspeed_pipeline.py tests/integration/test_deepspeed_pipeline_spike.py`
  - `pytest tests/integration/test_deepspeed_pipeline_spike.py -q -k 'not live' --run-integration`
- Ran one real two-host live diagnosis:
  `pytest tests/integration/test_deepspeed_pipeline_spike.py::test_live_deepspeed_pipeline_spike -q --run-integration --run-hardware --run-multi-host`

### Result

- Local checks passed.
- The single real live run failed before valid `LoadMicroBatch` sub-step
  evidence could be collected:
  - rank0: `python: can't open file '/tmp/deepspeed_pipeline_spike.py'`
  - rank1 entered distributed init and then waited because rank0 never started
    the Python payload successfully.
- Evidence: `/var/tmp/shardgrid/engines/deepspeed-pipeline-latest.json`

### Conclusion

- This attempt failed on a T062 harness/runtime file staging problem before it
  could prove whether rank1 `LoadMicroBatch` blocks in `_next_batch()` or the
  later `.to(device)` path.

## T063 - PyTorch pipeline spike (2026-08-22)

### What was done

- Evaluated the mature PyTorch pipeline option: `torch.distributed.pipelining`
  (torch 2.7.1; legacy `torch.distributed.pipeline.sync.Pipe` is removed in
  torch 2.x).  No new dependency installed.
- Implemented `src/shardgrid/engines/pytorch_pipeline.py` (spike harness:
  minimal GPipe 2 stages over 2 hosts x 1 GPU, marker parsing, PASS/FAIL/
  BLOCKED classifier, evidence saving), `tests/integration/
  test_pytorch_pipeline_spike.py` (6 logic + 1 live), and
  `docs/compatibility/pytorch-pipeline.md`.
- Marked T063 `[X]` in tasks.md.

### Files changed

- new: `src/shardgrid/engines/pytorch_pipeline.py`
- new: `tests/integration/test_pytorch_pipeline_spike.py`
- new: `docs/compatibility/pytorch-pipeline.md`
- modified: `specs/001-multi-host-training-mvp/tasks.md`

### Test results

- Logic tests: 6 passed; ruff clean.
- Live spike (RTX 4060 rank0 + GTX 1650 rank1, GPipe 2 stages): PASS (real)
  - stage construction + 2 schedule steps + DONE on both ranks, elapsed 0.9s
  - interfaces eth3/eth0 via `ip route get`, rendezvous 10.87.5.155:29500
- Evidence: `/var/tmp/shardgrid/engines/pytorch-pipeline-latest.json`
- Conclusion: PyTorch native pipeline is SUPPORTED on the two-host WSL2
  environment (contrast: DeepSpeed Pipeline BLOCKED per T062), so PyTorch's
  mature pipeline API is considered before any custom model-parallel code.

## T064 - nnScaler spike (2026-08-22)

### What was done

- Evaluated nnScaler (official microsoft/nnscaler; no PyPI release) per the
  decision process, one official-source install attempt on the RTX 4060
  Worker (proxy refused -> direct network once).
- Captured a reproducible blocker: official nnScaler 0.8 install replaces
  torch 2.7.1+cu118 with torch 2.6.0 and introduces the full nvidia-cu12
  stack (violates the fixed torch/CUDA environment rule).
- Restored the Worker environment to the exact pre-attempt state (torch
  2.7.1+cu118, torchvision/audio 0.22.1/2.7.1, nvidia-cu11 set, Apex
  FusedAdam + Galvatron imports verified, pip check clean). GTX 1650 was
  never touched.
- Implemented `src/shardgrid/engines/nnscaler.py` (spike harness + blocker
  classifier), `tests/integration/test_nnscaler_spike.py` (7 logic + 1 live),
  `docs/compatibility/nnscaler.md`; marked T064 [X] in tasks.md.

### Files changed

- new: `src/shardgrid/engines/nnscaler.py`
- new: `tests/integration/test_nnscaler_spike.py`
- new: `docs/compatibility/nnscaler.md`
- modified: `specs/001-multi-host-training-mvp/tasks.md`

### Test results

- Logic tests: 7 passed; ruff clean.
- Live environment record: PASS (torch 2.7.1+cu118 confirmed on Worker,
  nnscaler absent, install_preflight BLOCKED with 15 blocker entries).
- Evidence: `/var/tmp/shardgrid/engines/nnscaler-latest.json`
- Conclusion: nnScaler BLOCKED / NOT_SELECTED for the current environment
  (torch/CUDA stack replacement required by the official package).

## T065 - Parallel engine decision (2026-08-22)

### What was done

- Consolidated T061-T064 evidence into the final parallel-engine decision
  (no new framework experiments, no installs, no Worker changes).
- Published:
  - new: `examples/compatibility/parallel-engine-decision.json`
  - new: `docs/compatibility/parallel-engine-decision.md`
- Marked T065 `[X]` in tasks.md.

### Decision

- default engine: **galvatron** (SELECTED; explicit parallel configuration;
  static path labeled limited MVP support)
- fallback candidate: **pytorch_pipeline** (SUPPORTED, T063 two-host PASS)
- BLOCKED: **deepspeed_pipeline** (T062 train_batch deadlock)
- BLOCKED / NOT_SELECTED: **nnscaler** (T064 torch/CUDA stack replacement)
- Non-goals honored: no reimplementation of autograd/collectives/CUDA/
  pipeline framework.

### Files changed

- new: `examples/compatibility/parallel-engine-decision.json`
- new: `docs/compatibility/parallel-engine-decision.md`
- modified: `specs/001-multi-host-training-mvp/tasks.md`

### Test results

- Decision JSON round-trip + invariants validation: PASS (SELECTED /
  SUPPORTED / BLOCKED / NOT_SELECTED, default_engine=galvatron,
  no-reimplement clause present).

## T066 - ParallelEngine adapter contract (2026-08-22)

### What was done

- Defined the ParallelEngine contract in `src/shardgrid/engines/base.py`:
  - `ParallelEngine` protocol: `compatibility_spike` / `profile` / `plan` /
    `prepare` / `launch_metadata`, plus `engine_id` and `candidate`
  - `UnsupportedEngineMethodError`: unsupported methods must raise loudly
    (no silent fallback)
  - `EngineRegistry` + `registered_engine_registry()` reflecting the T065
    decision: galvatron EXPERIMENTAL (limited-support static planning),
    pytorch_pipeline AVAILABLE, deepspeed_pipeline BLOCKED, nnscaler BLOCKED
- Added contract return models `ProfileResult` and `EnginePreparation` to
  `src/shardgrid/engines/models.py` (existing models untouched).
- Extended `contracts/adapter-contracts.md` with identity/capability,
  error, and registry rules.
- Wrote `tests/contract/test_parallel_engine.py` (fake engine + partial
  engine + registry).

### Files changed

- new: `src/shardgrid/engines/base.py`
- new: `tests/contract/test_parallel_engine.py`
- modified: `src/shardgrid/engines/models.py` (added ProfileResult,
  EnginePreparation)
- modified: `specs/001-multi-host-training-mvp/contracts/adapter-contracts.md`
- modified: `specs/001-multi-host-training-mvp/tasks.md` (T066 -> [X])

### Test results

- Contract tests: 8 passed (protocol conformance, contract return types,
  original-plan preservation, explicit unsupported-method error, registry
  statuses SUPPORTED/BLOCKED/NOT_SELECTED, static validation labeled limited
  support, registry round-trip); ruff clean.
- No hardware run: T066 is a contract task; real engine adapter integration
  is T067.

## T067 - Selected engine adapter + fallback dispatch (2026-08-22)

### What was done

- Added `GalvatronEngine` ParallelEngine adapter to
  `src/shardgrid/engines/galvatron.py`: contract methods (compatibility_spike
  from T061 evidence, plan with static-validation limited-support label
  preserving the original external plan, prepare metadata-only, launch
  metadata); `profile` raises UnsupportedEngineMethodError.
- Added `src/shardgrid/engines/selected.py`:
  - `EngineSelectionError` for unknown / BLOCKED engines (no silent fallback)
  - `build_engine_adapter` (galvatron + pytorch_pipeline adapters)
  - `select_engine` (exactly one engine per job, original plan preserved)
  - `select_with_fallback` (fallback only on plan rejection, in registry
    order; all-fail raises with rejected ids)
- Added `tests/contract/test_selected_engine.py` (9 tests).
- Registry statuses unchanged from T065 (Galvatron EXPERIMENTAL,
  pytorch_pipeline AVAILABLE, deepspeed_pipeline BLOCKED, nnscaler BLOCKED).

### Files changed

- modified: `src/shardgrid/engines/galvatron.py` (added GalvatronEngine
  adapter + imports; existing spike harness untouched)
- new: `src/shardgrid/engines/selected.py`
- new: `tests/contract/test_selected_engine.py`
- modified: `specs/001-multi-host-training-mvp/tasks.md` (T067 -> [X])

### Test results

- Contract tests: 17 passed (9 selected-engine + 8 parallel-engine from
  T066), ruff clean.
- Existing Galvatron integration logic tests: 13 passed (no regression).
- No hardware run (T067 is adapter/dispatch contract work; real training is
  T068+).

### Milestone

- T062-T067 milestone: complete (DeepSpeed BLOCKED, nnScaler
  BLOCKED/NOT_SELECTED, Galvatron selected, PyTorch pipeline supported
  fallback, adapter + registry + dispatch in place).
