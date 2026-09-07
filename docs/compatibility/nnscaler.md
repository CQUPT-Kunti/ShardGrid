# Compatibility: nnScaler (T064)

Status: SPIKED - BLOCKED (environment compatibility) | Updated: 2026-08-22

T064 evaluates nnScaler only if required by the decision process and records
environment compatibility; every fallback candidate must have evidence, not a
speculative selection.

## What was evaluated

- Official source only: `https://github.com/microsoft/nnscaler` (no PyPI
  release exists; `pip index versions nnscaler` returns no distribution).
- Install path: official GitHub clone + `pip install .` on the RTX 4060
  Worker (SSH -> WSL2 -> conda `shardgrid`), following the task download rule
  (one attempt; direct network after the local proxy was refused).

## Result: BLOCKED (reproducible)

- The official nnScaler 0.8 install resolution **replaces the fixed
  torch/CUDA stack** on the Worker:
  - `torch 2.7.1+cu118` -> `torch 2.6.0`
  - full `nvidia-cu12` set installed (cublas/cudnn/cupti/nvrtc/runtime/
    curand/cusolver/cusparse/cusparselt/nccl/nvjitlink/nvtx 12.4.x)
  - `triton 3.3.1` -> `triton 3.2.0`, `sympy 1.14.0` -> `sympy 1.13.1`
- This violates ShardGrid's environment-consistency rule (never replace the
  verified torch/CUDA stack).  The spike stops there and records the blocker
  instead of degrading the environment.
- The Worker environment was restored to the exact pre-attempt state after
  capturing the evidence: torch 2.7.1+cu118 / torchvision 0.22.1+cu118 /
  torchaudio 2.7.1+cu118, CUDA 11.8 available, Apex FusedAdam and Galvatron
  imports verified, `pip check` clean.  The GTX 1650 Worker was untouched.

## Blocker note

- `nnScaler (microsoft/nnscaler) 0.8` dependency resolution replaces
  `torch 2.7.1+cu118` with `torch 2.6.0` and introduces the `nvidia-cu12`
  stack; installation is BLOCKED on the current Worker environment without
  changing the fixed PyTorch/CUDA baseline.

## Conclusion

- nnScaler is **NOT SELECTED** and **BLOCKED** for the current MVP
  environment: environment compatibility does not hold (torch/CUDA stack
  replacement required by the official package).
- Evidence is reproducible: the captured install plan is recorded in
  `tests/integration/test_nnscaler_spike.py` and the live evidence file.

## Evidence

- `src/shardgrid/engines/nnscaler.py` (spike harness + blocker classifier)
- `tests/integration/test_nnscaler_spike.py` (7 logic + 1 live)
- Live evidence: `/var/tmp/shardgrid/engines/nnscaler-latest.json`