# Training MVP Compatibility

## Gate 3

Gate 3 is the formal two-host training acceptance for the supported split model on:

- rank0 / Stage0 / RTX 4060 / LDJ / `10.87.5.155`
- rank1 / Stage1 / GTX 1650 / LAPTOP-5G3QUOGM / `10.87.5.15`

Required conditions:

- placement stays split: rank0 holds only Stage0 parameters, rank1 holds only Stage1 parameters
- forward passes across hosts through NCCL
- loss is finite
- Stage1 backward completes
- activation gradient returns to rank0
- Stage0 backward completes
- both optimizers update their local parameters
- training runs for multiple iterations
- final loss is at least 5% lower than initial loss
- per-rank checkpoint save/load roundtrip succeeds
- both ranks exit cleanly without NCCL timeout or communication error

Runtime preflight:

- both workers reachable
- WSL2 CUDA/PyTorch available
- actual NCCL route interface resolved dynamically with `ip route get <peer_ip>`
- NCCL path interface MTU must already be `1500`
- no stale ShardGrid training process left on either worker

Implementation note:

- T075 reuses the accepted T074 training path instead of adding a second training implementation
- formal Gate 3 evidence is produced by `tests/multi_host/test_training_gate.py`
