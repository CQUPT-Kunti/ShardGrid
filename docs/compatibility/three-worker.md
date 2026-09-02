# Three-Worker Distributed Smoke

Date: 2026-09-02
Task: T107
Status: FAIL

## Scope

Real hardware validation for:

- 3 physical GPU Workers
- 1 GPU per host
- `local_world_size=1`
- `world_size=3`
- backend preference: NCCL

Workers:

- rank 0: `gpu4060` / `LDJ` / `10.87.5.155` / NVIDIA GeForce RTX 4060 Laptop GPU
- rank 1: `gpu1060` / `LAPTOP-5G3QUOGM` / `10.87.5.15` / NVIDIA GeForce GTX 1650
- rank 2: `gpu4060-cqupt` / `CQUPT` / `10.87.5.214` / NVIDIA GeForce RTX 4060 Laptop GPU

## Preflight

All three Workers passed:

- SSH reachable
- WSL runtime reachable
- `torch` import
- `torch.cuda.is_available() == true`
- `device_count >= 1`
- `torch.distributed.is_nccl_available() == true`

## Route And MTU Evidence

- `LDJ -> GTX1650`: `eth3`, src `10.87.5.155`, MTU `1500`
- `LDJ -> CQUPT`: `eth3`, src `10.87.5.155`, MTU `1500`
- `GTX1650 -> LDJ`: `eth0`, src `10.87.5.15`, MTU `2800 -> 1500`
- `GTX1650 -> CQUPT`: `eth0`, src `10.87.5.15`, MTU `2800 -> 1500`
- `CQUPT -> LDJ`: `eth0`, src `172.29.249.4`, MTU `1500`
- `CQUPT -> GTX1650`: `eth0`, src `172.29.249.4`, MTU `1500`

`gpu1060` initially exposed an unsafe NCCL path MTU (`2800`) on its real egress
interface. The existing dynamic fix path was reused:

```bash
scripts/bootstrap-wsl.sh --fix-nccl-mtu-only
```

The fix was executed against peer `10.87.5.155` and the targeted verification
confirmed the shared `eth0` path was corrected to `1500` for both peers.

## Formal Smoke Result

- run id: `t107-three-worker-smoke-20260902T052019Z`
- backend used: `nccl`
- hardware smoke runs: `1`
- result: `FAIL`

Observed behavior:

- rank 0 emitted NCCL bootstrap logs and selected `eth3:10.87.5.155`
- rank 1 and rank 2 emitted no final payload
- all three ranks timed out before the test-local `T107_SMOKE` JSON payload
- no verified `init_ok`, `broadcast_ok`, `ring_ok`, `all_reduce_ok`, or `barrier_ok`
  result was produced

Relevant rank 0 stderr:

```text
[c10d] The hostname of the client socket cannot be retrieved. err=-3
```

Relevant rank 0 stdout tail:

```text
NCCL INFO Bootstrap : Using eth3:10.87.5.155<0>
NCCL INFO NET/Plugin: Using internal network plugin.
NCCL version 2.21.5+cuda11.0
```

## Acceptance Summary

- 3 Workers discovered and addressed: PASS
- `world_size=3` plan generation: PASS
- `local_world_size=1` per host: PASS
- dynamic route/interface discovery: PASS
- MTU safety after targeted fix: PASS
- NCCL backend availability: PASS
- distributed initialization: FAIL
- collective smoke: FAIL
- barrier: FAIL
- clean shutdown: FAIL
- core launch logic changed for third Worker: NO
- hardcoded `world_size=3` logic introduced in core code: NO

## Evidence

- live evidence JSON: `/var/tmp/shardgrid/distributed/three-worker-smoke-latest.json`
- test file: `tests/multi_host/test_three_worker_smoke.py`
- worker config: `examples/workers.yaml`
