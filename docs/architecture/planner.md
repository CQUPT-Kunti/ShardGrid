# Planner Architecture

ShardGrid Planner owns resource-aware placement and launch metadata. It does not own model graph tracing, autograd, CUDA kernels, collective communication, or a full model-parallel runtime.

## Automatic Planning Flow

```text
supported model
-> selected ParallelEngine profile
-> automatic partition candidates
-> WorkerResource + NetworkState validation
-> candidate rejection or selection
-> automatic ParallelPlan
-> ExecutionPlan
-> SSHLauncher
```

The static minimal model remains a regression fixture. It is not evidence that automatic partitioning works.

## Hard Constraints

These are checked before any score:

- Worker health
- GPU/runtime compatibility
- backend availability
- network reachability
- one valid physical host per Stage A-C Worker assignment
- valid `local_world_size`
- selected-engine-supported partition boundary
- estimated peak training memory fits usable GPU memory after headroom

Manual overrides are preferences. They cannot bypass hard constraints.

## Memory Model

Peak training memory includes:

- parameters
- gradients
- optimizer states
- activations
- temporary/runtime overhead
- communication buffers
- configurable safety headroom

A candidate is invalid when estimated peak training memory exceeds usable GPU memory after headroom.

## Scoring Order

Legal candidates are sorted deterministically:

1. fewest physical Workers
2. least cross-host communication
3. avoid severe heterogeneous GPU bottlenecks
4. improve compute balance
5. GPU capability or secondary preferences
6. deterministic tie-break

Communication cost is based on actual stage boundaries: activation bytes, gradient bytes, microbatch count, batch size, sequence length where relevant, boundary tensor shape, bandwidth, and latency.

## Required Evidence

Planner artifacts record:

- candidate rejection reasons
- selected reason
- fallback reason
- UNSATISFIABLE reason
- original engine plan reference
- model profile reference
- stage metadata
- placement metadata
