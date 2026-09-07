# Compatibility: Gate 1 - Single-GPU CUDA/PyTorch Smoke

Status: PASS | Updated: 2026-08-18 | Owner: Machine A control node

Gate 1 proves that each physical GPU Worker independently passes a real
CUDA/PyTorch smoke test inside its own WSL2 selected Conda training runtime.
It is all-or-nothing: PASS requires **both** Workers to report a real smoke
PASS. Evidence lives in `/var/tmp/shardgrid/gates/gate1-latest.json`.

## Workers

| Worker | GPU | Runtime | PyTorch | CUDA | Tensor smoke | Driver | Status |
|--------|-----|---------|---------|------|--------------|--------|--------|
| gpu4060 (10.87.5.155, rank0) | NVIDIA GeForce RTX 4060 Laptop GPU (cap 8.9, 8187 MB) | Windows -> WSL2 Ubuntu -> Conda `shardgrid` | 2.7.1+cu118 | 11.8 | 1024x1024 CUDA matmul, finite=True | 566.07 | PASS |
| gpu1060 (10.87.5.15, rank1) | NVIDIA GeForce GTX 1650 (cap 7.5, 4095 MB) | Windows -> WSL2 Ubuntu -> Conda `shardgrid` | 2.7.1+cu118 | 11.8 | 1024x1024 CUDA matmul, finite=True | 527.41 | PASS |

## Runtime evidence

- Python: `/home/shardgrid/miniconda3/envs/shardgrid/bin/python` (3.12.13) on both
- Conda: `shardgrid` env, prefix `/home/shardgrid/miniconda3/envs/shardgrid`
- `torch.cuda.is_available()`: True, `device_count` = 1 on both

## Gate 1 final status

**PASS** (RTX 4060 PASS + GTX 1650 PASS)

## Diagnostics

- Evidence file: `/var/tmp/shardgrid/gates/gate1-latest.json` (per-worker
  torch/cuda/gpu/driver/tensor evidence, error, log tails)
- Implementation: `src/shardgrid/workers/single_gpu_gate.py`
- Test: `tests/hardware/test_single_gpu_gate.py`

## Rules

- `torch.cuda.is_available() == True` alone is not PASS; a real CUDA tensor
  operation with a finite result is required.
- GPU identity must match the target (RTX 4060 / GTX 1650); a mismatch is FAIL.
- SSH/WSL/Conda/hardware inaccessibility is BLOCKED, never PASS.
- Gate 1 must pass before the multi-host gate (Gate 2, T053) can start.