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

- captured training context;
- execution graph;
- parameter/buffer ownership model;
- logical partition plan;
- placement plan using fresh worker/GPU resources;
- memory estimate and probe candidate report;
- compatibility/failure report.

## Expected Successful Run Artifacts

- rank logs;
- runtime consistency report;
- progress/loss/parameter-change evidence;
- worker checkpoint shards;
- merged model-state file;
- strict reload validation report.

## Compatibility Checks

Before tasks are generated, implementation work must preserve:

- existing `shardgrid train` behavior;
- existing zoo/stress validation assets;
- SSH-first multi-host gate;
- Conda environment detection/reuse;
- fresh GPU resource discovery;
- exact planner/runtime consistency.

## Non-Goals For This Feature

- optimizer-state merge;
- full arbitrary Python transparency;
- FSDP/ZeRO/DDP interop;
- Kubernetes/Volcano/HAMi enablement;
- user-facing ShardGrid model-provider APIs.
