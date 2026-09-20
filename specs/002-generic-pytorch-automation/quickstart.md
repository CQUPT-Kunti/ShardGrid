# Quickstart: Generic PyTorch Automation

## Target User Flow

Existing command:

```bash
python train.py --config config.yaml
```

ShardGrid command:

```bash
shardgrid run --config cluster.yaml train.py --config config.yaml
```

Dry-run:

```bash
shardgrid run --config cluster.yaml --dry-run --json train.py --config config.yaml
```

## Expected Dry-Run Artifacts

- non-materializing capture context;
- evidence that no full real CPU model materialization was required before planning;
- evidence that no full real CPU forward, backward, or optimizer step ran before planning;
- execution graph metadata;
- parameter/buffer state metadata and original model-state key mapping;
- conservative training-memory estimate;
- fresh worker/GPU resource snapshot with current total/used/free memory;
- logical partition plan;
- placement plan;
- no per-job GPU trial execution evidence;
- compatibility/failure report.

## Expected Successful Run Artifacts

- exact runtime-plan consistency report;
- bounded backend graph/code/metadata artifacts without full parameter payload;
- worker state manifests showing owned-state-only materialization;
- rank logs;
- progress/loss/parameter-change evidence;
- worker checkpoint shards;
- memory-safe finalization evidence;
- standard model-state output or stream manifest;
- strict compatibility validation report;
- cleanup report.

## Compatibility Checks

Before implementing new T066+ work, the updated plan must preserve:

- T001-T065 as frozen historical implementation;
- existing `shardgrid train` compatibility behavior;
- existing zoo/stress validation assets as tests/examples only;
- SSH-first multi-host gate;
- Conda environment detection/reuse;
- fresh GPU resource discovery;
- exact planner/runtime consistency;
- structured failure taxonomy;
- standard checkpoint semantics.

## New Gate Checks

```text
CONTROL_PLANE_FULL_MODEL_BEFORE_PLAN=0
CPU_REAL_FORWARD_BEFORE_PLAN=0
CPU_REAL_BACKWARD_BEFORE_PLAN=0
CPU_REAL_OPTIMIZER_STEP_BEFORE_PLAN=0
MODEL_MEMORY_ESTIMATED_BEFORE_MATERIALIZATION=PASS
PLACEMENT_USES_FRESH_FREE_VRAM=PASS
PER_JOB_GPU_TRIAL_PROBE=0
BACKEND_GRAPH_FULL_PARAMETER_PAYLOAD=0
WORKER_FULL_MODEL_CPU_LOAD=0
WORKER_OWNED_STATE_ONLY=PASS
CHECKPOINT_CONTROL_PLANE_FULL_STATE_LOAD=0
MODEL_LARGER_THAN_CONTROL_PLANE_RAM_PLANNING=PASS
ORDINARY_PYTORCH_ENTRYPOINT=PASS
GPU_PACKING_STRESS_VALIDATED=PASS
```

## Non-Goals For This Re-plan

- executing T066 or hardware stress during specification/planning;
- reopening or renumbering T001-T065;
- optimizer-state merge beyond proven-safe existing behavior;
- full arbitrary Python transparency;
- FSDP/ZeRO/DDP interop;
- Kubernetes/Volcano/HAMi enablement;
- user-facing ShardGrid model-provider APIs;
- final stress based on giant CPU parameter serialization or artificial CUDA reserve buffers.
