# ShardGrid Final Planner Design

## Planning Flow

```text
T109 -> T110 -> T111 -> T112 -> T114 -> T115 -> T116
```

- T109: model/profile
- T110: automatic partition
- T111: strict worker-count search plus placement
- T112: final same-K candidate selection
- T114: ParallelPlan persistence
- T115: ExecutionPlan audit and dry-run artifacts
- T116: planner gate acceptance

## Automatic Partition

- partition source: `automatic`
- input: profiled model dependency graph
- node weight: training-memory weight
- edge weight: forward activation bytes + backward gradient bytes
- residual / skip communication: preserved as explicit cross-stage edges

## Worker Count

```text
2 -> fail only then 3 -> fail only then 4
```

- first feasible `K` stops immediately
- no global re-ranking across mixed worker counts

## Memory Constraint

Hard gate:

```text
estimated training peak memory <= worker usable memory
```

The estimate includes parameters, gradients, optimizer state, activations,
temporary overhead, communication buffers, and headroom. Parameter bytes alone
are not accepted as the fit criterion.

## Placement

- placement uses real eligible Workers only
- heterogeneous usable VRAM is respected
- each stage is placed exactly once
- selected worker count must match the unique placed Workers

## Ranking

- rank only candidates with the same selected `K`
- primary key: cross-worker communication bytes
- near ties: memory utilization / packing quality
- final tie-break: deterministic saved worker/stage/candidate identity
- compute time / FLOPS / GPU latency: not used

## ParallelPlan / ExecutionPlan

Persist and audit:

- stage mapping
- worker placement
- memory metadata
- communication metadata
- world size
- master
- assignments
- planning provenance

## Manual Override

```text
SKIPPED / NOT SUPPORTED
```

Manual override is not part of the current planner design and must not bypass
hard constraints.

## Current Gate

```text
T116 Planner Gate
```

## Real Training

```text
NOT part of T116
```

Automatic-plan real hardware training remains a later hardware gate.

## CLI Integration

Automatic planner is reachable from the real `shardgrid train` CLI.

```text
automatic config
-> T109 profile
-> T111 worker-count search + placement
-> T112 selection
-> T114 ParallelPlan
-> T115 ExecutionPlan
-> dry-run audit
```

Verified:

```text
plan_mode=automatic
partition_source=automatic
```

Static/manual path remains available for compatibility.
