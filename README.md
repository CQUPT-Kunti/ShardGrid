# ShardGrid

ShardGrid is a Python control-plane project for real cross-host AI training across separate physical GPU workers.

## Current Focus

- Active Spec Kit feature: `001-multi-host-training-mvp`
- Control/Login node: Machine A running Ubuntu
- GPU workers: separate physical machines
- Current worker assumption: one physical worker, one GPU, `local_world_size = 1`
- Windows GPU workers use a WSL2 Linux runtime for actual training work

## Execution Order

ShardGrid is being built in this order:

1. Real SSH-based cross-host training first
2. Kubernetes only after the SSH MVP succeeds
3. Volcano only after Kubernetes is proven stable
4. HAMi only after the Kubernetes/Volcano path is stable

This repository does not claim those later platform paths are available yet.

## Implementation Principles

- Use the existing `src` layout: `src/shardgrid/`
- Prefer mature components and adapters over custom reimplementation
- Keep machine addresses, users, paths, ports, and runtime details configurable
- Manage Python development and training environments with Conda
- Detect Conda first, reuse a compatible existing Conda environment when available, and create a ShardGrid environment only when needed
- Do not force one exact Python or Conda version without a demonstrated compatibility reason

## Project Pointers

- Feature spec: `specs/001-multi-host-training-mvp/spec.md`
- Implementation plan: `specs/001-multi-host-training-mvp/plan.md`
- Task list: `specs/001-multi-host-training-mvp/tasks.md`
- Agent guidance: `AGENTS.md`

## Current Baseline Commands

These commands are expected to work on Ubuntu Machine A in the current repository state:

```bash
command -v conda
conda info
python -m pip install -e .
python -c "import shardgrid; print(shardgrid.__file__)"
pytest
ruff check
mypy
```

Run them from the selected compatible Conda environment. If no compatible Conda
environment exists yet, bootstrap should create one only after checking existing
Conda installations and without deleting user environments.

Commands such as `shardgrid doctor`, distributed training, SSH launch, Kubernetes, Volcano, and HAMi flows are planned but not implemented yet.
