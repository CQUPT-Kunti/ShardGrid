# Checkpoint Consolidation Architecture

ShardGrid stores two related but separate artifacts:

- Distributed checkpoint: authoritative resume/fault-recovery state for the partitioned training job.
- Consolidated full model: user-facing model weights that can be loaded without knowing stage count, Worker count, ranks, or placement.

## Distributed Checkpoint

The training checkpoint may stay sharded and may include:

- manifest
- model shards
- optimizer shards
- scheduler state
- RNG state
- runtime state
- partition metadata

This artifact is optimized for resume and distributed restart.

## Consolidated Full Model

After successful automatic-partition training, ShardGrid exports a full model artifact from the distributed checkpoint. CPU consolidation is the default to avoid consolidation-time GPU OOM.

The full model artifact preserves:

- original parameter namespace
- full state dict
- required buffers
- shared/tied parameter metadata where supported by the selected engine
- keys, shapes, and dtypes

## Acceptance

The consolidation gate instantiates the original model definition, loads the consolidated state dict, validates keys/shapes/dtypes, and runs a forward or inference smoke. A job is not a completed automatic-partition success unless this reload validation is recorded.
