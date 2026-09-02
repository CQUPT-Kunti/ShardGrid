# ShardGrid 错误记录

## 当前 blocker

T107 current first blocker:

- mirrored networking / overlay data-plane: PASS
- CQUPT WSL overlay route src: `10.87.5.214`
- three-worker MTU: PASS (`1500`)
- rendezvous port blocker `EADDRINUSE port 29500`: RESOLVED
- LDJ stale ShardGrid owner on `29500` was identified precisely as:
  `pid=311 /home/shardgrid/miniconda3/envs/shardgrid/bin/python -`
- stale process cleanup: PASS
- fresh rerun id: `t107-three-worker-smoke-20260902T054328Z`
- fresh `MASTER_ADDR=10.87.5.155`
- fresh `MASTER_PORT=29500`
- current new blocker:
  - rank 0 (`gpu4060` / `LDJ`) timed out after NCCL bootstrap and produced no
    final `T107_SMOKE` payload
  - rank 0 stdout stopped at:
    `NCCL INFO Bootstrap : Using eth3:10.87.5.155<0>`
  - rank 0 stderr included:
    `[c10d] The hostname of the client socket cannot be retrieved. err=-3`
  - rank 1 reached `ncclCommInitRank ... Init COMPLETE` and broadcast start
  - rank 2 produced no final payload
- evidence: `/var/tmp/shardgrid/distributed/three-worker-smoke-latest.json`
