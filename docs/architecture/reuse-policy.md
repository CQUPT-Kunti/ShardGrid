# Reuse-First Policy

## Purpose

ShardGrid integrates existing, mature capabilities before it adds project-owned implementations. New ShardGrid code should focus on orchestration, configuration, diagnostics, placement metadata, artifact tracking, and adapter boundaries.

## Default Rule

If a stable external capability already exists and fits the problem, use it through configuration, a CLI boundary, or an adapter. Only consider a ShardGrid-owned implementation when no suitable mature option exists or when the missing behavior is strictly ShardGrid-specific.

## Capabilities ShardGrid Does Not Reimplement

ShardGrid does not own or reimplement these layers:

- SSH / OpenSSH
- File transfer protocols
- WSL2
- PyTorch Distributed
- NCCL / Gloo
- CUDA kernels
- autograd
- A full model-parallel runtime
- Kubernetes
- Volcano
- HAMi

For these areas, ShardGrid's job is to detect, configure, validate, adapt, launch, observe, and report. It is not to replace the underlying platform or runtime.

## Parallel Engine Boundary

Parallel engine support follows the same rule:

- Prefer mature projects first
- Evaluate compatibility on the real target environment
- Integrate the selected engine through a narrow adapter boundary
- Preserve the engine's original plan/runtime outputs when possible and add only ShardGrid-specific placement and launch metadata

Galvatron is the first planned compatibility candidate. If it does not fit the target environment, the fallback path is to evaluate other mature engines rather than building a ShardGrid-owned full parallel runtime.

## Dependency Boundary

The base `shardgrid` package must stay light:

- Base installation must not require Kubernetes, Volcano, HAMi, Galvatron, PyTorch, CUDA, or other late-phase platform/runtime components
- Those integrations belong in optional dependency groups or future adapter-specific packages
- A compatible existing local Python/tooling environment should be reused when available; recommended versions are guidance, not a reason to reinstall or force a specific patch release

Current packaging boundary:

- Base runtime dependencies are empty today because the current package skeleton does not need extra runtime libraries yet
- Development-only tools live in optional groups
- Late-phase platform extras are kept out of the base install

## Review Questions

Before adding code or dependencies, check:

1. Does an existing mature component already solve this?
2. Can ShardGrid call or adapt it instead of replacing it?
3. Does this dependency belong in base install, or should it stay optional?
4. Is the proposed code implementing platform/runtime internals that ShardGrid is supposed to delegate?

If the answer to the last question is yes, the change is probably outside ShardGrid's intended boundary.
