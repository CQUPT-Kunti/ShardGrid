# T073 Backward Acceptance

Date: 2026-08-25

- Implemented real two-host backward on top of the accepted T072 forward path in `examples/models/train_pipeline.py`.
- Added `tests/multi_host/test_backward_transfer.py` for T073 parsing and opt-in real multi-host verification.
- Local validation passed:
  - `python -m py_compile examples/models/train_pipeline.py examples/models/stage1.py examples/models/stage0.py tests/multi_host/test_backward_transfer.py`
  - `ruff check examples/models/train_pipeline.py examples/models/stage1.py examples/models/stage0.py tests/multi_host/test_backward_transfer.py`
  - `PYTHONPATH=$PWD:$PWD/tests python -m pytest -q tests/unit/test_model_stages.py tests/multi_host/test_backward_transfer.py -k 'not live'`
  - `PYTHONPATH=$PWD:$PWD/tests python -m pytest -q tests/multi_host/test_activation_transfer.py -k 'parse_forward_evidence or not live'`
- Real hardware acceptance passed once on:
  - rank0 / LDJ / RTX 4060 / `10.87.5.155`
  - rank1 / LAPTOP-5G3QUOGM / GTX 1650 / `10.87.5.15`
- Preflight confirmed the dynamically resolved NCCL path interfaces still had MTU 1500:
  - rank0 route to rank1 via `eth3`
  - rank1 route to rank0 via `eth1`
- Acceptance artifact:
  - `/home/yangjilei/Code/ShardGrid/t073_backward_acceptance_20260825_195043.zip`
- Acceptance result:
  - Stage0 forward PASS
  - Cross-host activation PASS
  - Stage1 forward PASS
  - Stage1 backward PASS
  - Activation gradient PASS
  - Gradient return PASS
  - Stage0 backward PASS
  - Stage0 gradients PASS
  - Stage1 gradients PASS

# T074 Optimizer + Loss Decrease + Checkpoint Acceptance

Date: 2026-08-25

- Implemented the T074 training increment in `examples/models/train_pipeline.py`
  (`SHARDGRID_PIPELINE_TASK=t074`) on top of the accepted T072/T073 forward and
  backward communication path: per-iteration `zero_grad -> forward -> backward
  -> optimizer.step` on both stages, a recorded loss history, per-parameter
  checksums, and a per-rank checkpoint save/load verification.
- rank0 optimizer manages only Stage0 parameters; rank1 optimizer manages only
  Stage1 parameters. No rank loads the full model. Route and gradient direction
  continue to come from the static ParallelPlan (`producer_rank`/`consumer_rank`).
- Added `tests/multi_host/test_optimizer_checkpoint.py` for T074 evidence parsing
  and opt-in real multi-host verification.
- Local validation passed:
  - `python -m py_compile examples/models/train_pipeline.py examples/models/stage1.py examples/models/stage0.py tests/multi_host/test_optimizer_checkpoint.py`
  - `ruff check examples/models/train_pipeline.py tests/multi_host/test_optimizer_checkpoint.py`
  - T074 non-live parse tests: `pytest -q tests/multi_host/test_optimizer_checkpoint.py --run-multi-host -k 'not live'` -> 3 passed
  - T071/T072/T073 non-live regression: `pytest -q tests/multi_host/test_stage_placement.py tests/multi_host/test_activation_transfer.py tests/multi_host/test_backward_transfer.py --run-multi-host -k 'not live'` -> passed
  - Local CPU sanity: same loop on CPU -> loss 7.266 -> 0.361, param/opt/step restore all True
- Real hardware acceptance passed once on:
  - rank0 / LDJ / RTX 4060 / `10.87.5.155`
  - rank1 / LAPTOP-5G3QUOGM / GTX 1650 / `10.87.5.15`
- Preflight confirmed the dynamically resolved NCCL path interfaces had MTU 1500:
  - rank0 route to rank1 via `eth3`
  - rank1 route to rank0 via `eth1`
- Training result (20 iterations, deterministic input, AdamW lr=1e-3):
  - initial loss: 7.26569128036499
  - final loss: 0.3605937063694
  - loss decrease: PASS (final < initial), loss finite for all iterations
  - Stage0 parameter update: PASS (checksum changed on rank0)
  - Stage1 parameter update: PASS (checksum changed on rank1)
  - optimizer.step on rank0 and rank1: PASS
- Checkpoint result (per-rank shards, no full-model aggregation):
  - rank0 checkpoint `/tmp/t074/checkpoint/checkpoint_rank0.pt` save + load + verify PASS
  - rank1 checkpoint `/tmp/t074/checkpoint/checkpoint_rank1.pt` save + load + verify PASS
  - parameter restore PASS, optimizer state restore PASS, step/iteration restore PASS on both ranks
- T073 forward/backward path regression after the T074 change: `pytest -q tests/multi_host/test_backward_transfer.py --run-multi-host -k 'live'` -> 1 passed (real two-host backward + gradient return still PASS)
- Evidence artifact:
  - `/var/tmp/shardgrid/engines/optimizer-checkpoint-latest.json`

# T075 Gate 3 Acceptance

Date: 2026-08-27

- Reused the accepted T074 two-host training path as the formal Gate 3 acceptance path through `tests/multi_host/test_training_gate.py`; no new training implementation was introduced.
- Added Gate 3 evidence output at `/var/tmp/shardgrid/engines/training-gate-latest.json`.
- Local validation passed:
  - `python -m py_compile tests/multi_host/test_optimizer_checkpoint.py tests/multi_host/test_training_gate.py examples/models/train_pipeline.py`
  - `ruff check tests/multi_host/test_optimizer_checkpoint.py tests/multi_host/test_training_gate.py examples/models/train_pipeline.py docs/compatibility/training-mvp.md`
  - `PYTHONPATH=$PWD:$PWD/tests python -m pytest -q tests/multi_host/test_training_gate.py tests/multi_host/test_optimizer_checkpoint.py --run-multi-host -k 'not live'`
- Gate 3 preflight on the real workers:
  - rank0 route to rank1 stayed on `eth3`, MTU `1500 -> 1500`
  - rank1 route to rank0 resolved dynamically to `eth0`, MTU `2800 -> 1500`
  - both workers reported `torch 2.7.1+cu118`, CUDA runtime `11.8`, NCCL runtime `2.21.5`
  - no stale ShardGrid training process remained before launch
- Real hardware Gate 3 acceptance passed on:
  - rank0 / LDJ / RTX 4060 / `10.87.5.155`
  - rank1 / LAPTOP-5G3QUOGM / GTX 1650 / `10.87.5.15`
- Training result:
  - iterations: 20
  - initial loss: `7.26569128036499`
  - final loss: `0.3605937063694`
  - loss history finite for all iterations
  - loss decrease: PASS
- Training path result:
  - forward: PASS
  - backward: PASS
  - gradient return: PASS
  - rank0 parameter update: PASS
  - rank1 parameter update: PASS
  - rank0 last marker: `SHUTDOWN_END`
  - rank1 last marker: `SHUTDOWN_END`
- Checkpoint result:
  - rank0 checkpoint roundtrip: PASS
  - rank1 checkpoint roundtrip: PASS
  - optimizer state restore: PASS
  - step / iteration restore: PASS
- Gate 3 result:
  - T075: PASS
  - Gate 3 training loop: PASS
