# T110 Record

Date: 2026-09-03
Task: T110
Status: implemented

## Summary

- Added generic automatic partition boundary discovery and candidate generation in `src/shardgrid/planner/partitioning.py`.
- Added a full residual/skip fixture model in `examples/models/partition_stress_model.py`.
- Reused T109 `ModelProfile` and `estimate_stage_memory(...)`; did not add worker placement logic.

## Deterministic Synthetic Data Strategy

- External deterministic batch generator: `make_training_batch(config, seed, step)`.
- Recorded shape inputs through code and tests: stable seed, batch size, input shape, target shape, and short training steps.
- The model does not generate inputs or targets inside `__init__` or `forward`.

## Automatic Partition Candidate Generation

- Module dependencies are discovered from `torch.fx.symbolic_trace` when available.
- Static-shape guards that break symbolic tracing fall back to `torch.export` validation plus ordered-module adjacency.
- Structured unsupported results are returned for untraceable graphs or unsupported custom Python ops.
- Candidate generation is bounded and deterministic, preserving stage ranges, boundary ids, communication edges, runtime/backend requirements, and optional original engine plan references.

## MinimalTransformer Candidate Result

- Full `MinimalTransformer` is profiled as one complete `nn.Module`.
- Automatic candidates are generated without reading `stage0.py` or `stage1.py`.
- Candidate ordering is deterministic across repeated generation with the same model/profile/configuration.

## Residual/Skip Model Candidate Result

- Added `PartitionStressModel` with residual blocks and skip fusion.
- Automatic candidates are generated from the full model, including both 2-stage and 3-stage feasible candidates in tests.
- Cross-stage skip dependencies are preserved as communication edges instead of being flattened into a linear-only assumption.

## Dependency Handling

- Residual and skip connections are traced into boundary metadata and candidate communication edges.
- Unsupported dynamic control flow returns a structured unsupported result.
- Unsupported custom Python ops return a structured unsupported result.

## Parameter Coverage

- Candidate validation checks contiguous stage ordering.
- Legal candidates cover all profiled parameters exactly once.
- Shared/tied parameter groups that cross a boundary are rejected explicitly.

## Tests

- Deterministic forward for the residual/skip full model.
- Verification that training data comes from the external generator, not the model.
- Short deterministic training smoke: forward, finite loss, backward, optimizer step.
- MinimalTransformer automatic candidate generation regression.
- Residual/skip model candidate generation and skip-edge preservation.
- Parameter coverage checks on feasible candidates.
- Unsupported dynamic control flow and unsupported custom-op coverage.
- Shared/tied parameter boundary rejection coverage.
