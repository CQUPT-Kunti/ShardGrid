# Planner Architecture

ShardGrid Planner owns the automatic planning chain from model profiling through
dry-run audit for the current MVP. It does not own autograd, CUDA kernels,
collective implementations, or real multi-host execution.

## Current Flow

```text
T109 model/profile
-> T110 automatic weighted graph partition
-> T111 strict worker-count search + eligible worker placement
-> T112 same-K final candidate selection
-> T114 ParallelPlan materialization
-> T115 ExecutionPlan audit + snapshot metadata + dry-run
-> T116 planner gate acceptance
```

This is the current planner path as of 2026-09-03. The old static Stage 0 /
Stage 1 planning path is not evidence that automatic planning works.

## Hard Gates

Planner rejects a candidate before any ranking when any of these fail:

- stage training peak memory exceeds worker usable memory
- worker health, enablement, runtime, or backend eligibility fails
- required network links for cross-worker communication are unreachable
- one stage is missing placement or multiple stages collapse onto one physical
  host in the Stage A-C path
- saved plan metadata becomes inconsistent during replay

Memory validation uses the full estimated training peak, including parameters,
gradients, optimizer state, activations, temporary overhead, communication
buffers, and safety headroom. Parameter bytes alone are not sufficient.

## Worker Count Search

The current T111 rule is strict and ordered:

```text
try 2 workers
if feasible: stop
else try 3 workers
if feasible: stop
else try 4 workers
```

Planner does not precompute 2/3/4 and then globally rank them together. The
first feasible worker count becomes the only worker-count frontier passed to
T112.

## Automatic Partition And Placement

Automatic partitioning is derived from the profiled model graph. Each stage
records:

- module slice and stage boundary
- estimated training peak memory
- forward activation and backward gradient communication
- residual / skip cross-stage communication edges
- selected worker, rank, GPU, usable memory, remaining memory, and utilization

T114 must preserve the chosen partition and placement. T115 must expose the
same data in `ExecutionPlan`, dry-run output, and saved JSON/YAML artifacts
without recomputing partition or placement.

## Ranking

T112 ranks only FEASIBLE candidates that already share the same selected worker
count.

Sort order:

1. minimum `total_cross_worker_communication_bytes`
2. memory-utilization quality on near ties
3. deterministic tie-break by saved worker, stage, and candidate identity

The current planner does not use compute time, FLOPS, or GPU latency weights.

## Persistence, Replay, And Dry-Run

Planner artifacts must preserve:

- selected worker count and attempted worker counts
- selected workers and stage-to-worker mapping
- stage memory and communication metadata
- `ParallelPlan` provenance
- `ExecutionPlan` assignments
- engine, backend, world size, and rendezvous master

Replay validation is fail-closed. If a saved worker becomes unhealthy,
ineligible, or loses usable memory below the saved stage peak, replay is
rejected. Replay does not silently re-place the model.

`train --dry-run` must:

1. probe current resources
2. build the final plan
3. save auditable snapshot artifacts
4. exit before remote rank launch, torch distributed init, or real training

## Manual Override

T113 manual override is currently skipped by design.

```text
manual override: NOT_SUPPORTED / SKIPPED_BY_CURRENT_DESIGN
```

No manual preference path may bypass planner hard constraints.

## Hardware Boundary

T116 is a planner gate only. Real automatic-plan multi-host training remains a
later hardware acceptance step in T117.
