# T049 Error Log

Updated: 2026-08-17

## Scope

This file records the real failure evidence for `T049` only:

- Machine A
- SSH
- Windows host
- WSL2 Ubuntu
- selected Conda runtime
- `torch.distributed` / `NCCL`

It is not a design note. It is an execution log and blocker summary.

## What Was Being Done

Goal:

- use the real RTX 4060 Worker as `rank0`
- use the real GTX 1650 Worker as `rank1`
- establish a real `NCCL` process group
- validate `broadcast`, `send/recv`, and `all_reduce` on CUDA tensors

Real workers:

- `gpu4060` -> `10.87.5.155` -> `LDJ`
- `gpu1060` -> `10.87.5.15` -> `LAPTOP-5G3QUOGM`

Selected runtime on both workers:

- distro: `Ubuntu`
- user: `shardgrid`
- conda env: `shardgrid`
- conda prefix: `/home/shardgrid/miniconda3/envs/shardgrid`
- python: `/home/shardgrid/miniconda3/envs/shardgrid/bin/python`

## Verified Facts

These are real, not assumed:

- Machine A can SSH to both Windows GPU workers
- both Windows workers can start `WSL2 Ubuntu`
- both WSL runtimes can use the selected Conda Python
- both WSL runtimes report `torch.cuda.is_available() == True`
- mirrored networking was enabled on both workers
- raw TCP to `29500` was verified both directions after firewall changes
- Linux ephemeral port range was constrained on both workers to:

```text
50000 51000
```

## Important Runtime Observation

The port range constraint is active.

Observed runtime ports included values like:

- `50488`
- `50440`

Those are inside `50000-51000`, so port-range restriction did apply.

## Real Problems Seen During Investigation

### 1. Old stale NetworkState polluted live rendezvous selection

Earlier live runs were using a stale path that selected the wrong
`master_addr` from old network evidence.

That was corrected so live `rank0` now uses:

```text
master_addr = 10.87.5.155
master_port = 29500
```

### 2. Old distributed Python processes were left behind

Multiple earlier failed attempts left `python -` processes behind in WSL.

That caused real `EADDRINUSE` collisions on `29500`.

Those leftover processes were manually cleaned before the final rerun.

### 3. Even after cleanup, NCCL still did not complete

After:

- mirrored networking
- `29500` firewall opening
- `50000-51000` range opening
- WSL port-range restriction
- stale-process cleanup

the real `T049` run still failed.

## Latest Failure Evidence

Source:

- `/var/tmp/shardgrid/distributed/collectives-latest.json`

Latest structured result:

```json
{
  "backend": "nccl",
  "interface": "eth3",
  "master_addr": "10.87.5.155",
  "master_port": 29500,
  "outcome": "FAIL"
}
```

### rank0

- worker: `gpu4060`
- host: `10.87.5.155`
- exit: `-1`
- timed_out: `true`

stderr excerpt:

```text
[W817 21:03:04.386960882 socket.cpp:200] [c10d] The hostname of the client socket cannot be retrieved. err=-3
[W817 21:03:14.398276941 socket.cpp:200] [c10d] The hostname of the client socket cannot be retrieved. err=-3
```

### rank1

- worker: `gpu1060`
- host: `10.87.5.15`
- exit: `-1`
- timed_out: `true`

stderr excerpt:

```text
/home/shardgrid/miniconda3/envs/shardgrid/lib/python3.12/site-packages/torch/_subclasses/functional_tensor.py:276: UserWarning: Failed to initialize NumPy: No module named 'numpy'
```

Notes:

- the NumPy warning is not the core blocker here
- the real blocker is that the distributed run still never completed and the
  collective result was not produced as `PASS`

## What Was Observed Live With `ss -tanp`

During a real run, live socket state showed:

- `rank0` listening on `*:29500`
- `rank1` attempting to connect to `10.87.5.155:29500`
- temporary source ports inside the constrained range

Examples observed during investigation:

```text
LISTEN   *:29500
SYN-SENT 10.87.5.15:50598 -> 10.87.5.155:29500
SYN-RECV 10.87.5.155:29500 <- 10.87.5.15:50598
```

That proves:

- `rank1` did start
- it did attempt the real cross-host connection
- the failure is not "1650 never started"

## Current Status

`T049` is still:

```text
FAIL / BLOCKED
```

It must not be marked `[X]`.

## What This File Rules Out

This failure is not currently explained by:

- wrong worker IP source
- wrong WSL distro
- wrong Conda prefix
- wrong Python path
- Windows Python being used by mistake
- missing SSH access
- stale NAT-only WSL setup

Those layers were checked directly.

## Most Likely Remaining Failure Area

The remaining problem is still in the live runtime communication path:

- `torch.distributed` / `c10d` socket behavior
- `NCCL` communication after rendezvous setup begins
- or Windows/WSL mirrored networking behavior under this specific cross-host flow

## Next Useful Debug Step

If investigation continues, the next high-value step is:

- capture fuller per-rank NCCL/c10d stdout/stderr
- keep `NCCL_DEBUG=INFO`
- record the exact per-rank socket transitions during one clean rerun

## 2026-08-17 Layered Classification Retest

The intended classification was:

- raw TCP PASS, TCPStore FAIL
- raw TCP PASS, TCPStore PASS, Gloo PASS, NCCL FAIL
- firewall enabled FAIL, firewall disabled PASS

Latest retest did not reach the TCPStore/Gloo/NCCL distinction because raw TCP
is currently not passing.

### Raw TCP Retest

Command shape:

- rank0-side listener on `10.87.5.155:29500`
- rank1-side client connects to `10.87.5.155:29500`
- reverse direction listener on `10.87.5.15:29500`
- reverse client connects to `10.87.5.15:29500`

Result:

```text
10.87.5.15 -> 10.87.5.155:29500  FAIL TimeoutError
10.87.5.155 -> 10.87.5.15:29500  FAIL TimeoutError
```

This means the current state is not:

```text
raw TCP PASS, TCPStore FAIL
```

and also not:

```text
raw TCP PASS, TCPStore PASS, Gloo PASS, NCCL FAIL
```

The current state is:

```text
raw TCP FAIL
```

### Why This Matters

Earlier raw TCP was observed passing after firewall changes, but the latest
clean retest shows both directions timing out again. That makes the remaining
blocker lower than PyTorch:

- before `TCPStore`
- before `Gloo`
- before `NCCL`

Until raw TCP is stable, PyTorch/NCCL test results are not meaningful.

### Firewall Disabled Test

The `firewall=false` test was not executed by the agent:

- current SSH user is not elevated
- disabling firewall is an administrator action
- disabling firewall is broader than the minimal safe change

If an administrator temporarily disables firewall on both workers and raw TCP
then passes, the classification becomes:

```text
firewall=true  FAIL
firewall=false PASS
```

