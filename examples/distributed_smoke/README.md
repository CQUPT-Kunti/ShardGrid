# Distributed Smoke (T046)

Minimal, parameterized `torch.distributed` smoke program for the upcoming
RTX 4060 + GTX 1650 two-host communication validation (T047-T053).

## Usage

```bash
python examples/distributed_smoke/smoke.py \
  --rank 0 --world-size 2 \
  --master-addr <master-ip> --master-port 29500 \
  --backend gloo --local-rank 0
```

Arguments:

- `--rank` (required): global rank, must be in `[0, world_size)`
- `--world-size` (required): total number of processes, must be `>= 1`
- `--master-addr` (required): rendezvous master address
- `--master-port` (default `29500`): rendezvous port
- `--backend` (default `gloo`): `nccl` or `gloo`
- `--local-rank` (default `0`): local rank

## Behavior

1. Validates launch arguments; invalid arguments fail explicitly (exit `2`).
2. Initializes a process group over `tcp://<master_addr>:<master_port>` using
   the official `torch.distributed.init_process_group` API.
3. Prints a JSON result with runtime evidence: Conda environment/prefix,
   Python executable/version, PyTorch version, `torch.version.cuda`, backend,
   rank, world size, and local rank.
4. Calls `destroy_process_group` on exit (cleanup).
5. On process-group init failure it returns a structured failure (`ok=false`,
   `stage=init`) with the real error and preserves the runtime evidence.

This program never reimplements collectives, sockets, or NCCL/Gloo protocols;
it only exercises official PyTorch Distributed APIs.
