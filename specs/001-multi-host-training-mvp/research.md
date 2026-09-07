# Research: ShardGrid MVP + Platform Foundation

**Date**: 2026-08-15
**Feature**: `001-multi-host-training-mvp`

This research resolves planning choices for the first implementation route. Decisions favor mature upstream components and compatibility gates over early platform lock-in.

## Sources Consulted

- NVIDIA CUDA on WSL User Guide: <https://docs.nvidia.com/cuda/wsl-user-guide/index.html>
- PyTorch distributed docs: <https://docs.pytorch.org/docs/2.13/distributed.html>
- PyTorch torchrun docs: <https://docs.pytorch.org/docs/2.13/elastic/run.html>
- Kubernetes device plugins: <https://kubernetes.io/docs/concepts/extend-kubernetes/compute-storage-net/device-plugins/>
- NVIDIA Kubernetes device plugin: <https://github.com/NVIDIA/k8s-device-plugin>
- Volcano docs: <https://volcano.sh/docs/home/introduction/>
- HAMi project: <https://github.com/Project-HAMi/HAMi>
- Galvatron repository: <https://github.com/PKU-DAIR/Hetu-Galvatron>
- DeepSpeed Pipeline Parallelism docs: <https://deepspeed.readthedocs.io/en/stable/pipeline.html>
- nnScaler repository: <https://github.com/microsoft/nnscaler>

## Decision: Use WSL2 Linux Runtime on Windows GPU Hosts

**Rationale**: The project target is Linux Control plus Windows physical Workers. NVIDIA's CUDA on WSL guide documents WSL2 as the route for Linux applications and CUDA workflows on Windows, and warns not to install Linux NVIDIA display drivers inside WSL. This matches the requirement to avoid a Windows-native training runtime.

**Alternatives considered**:

- Native Windows CUDA/PyTorch training: rejected for MVP because PyTorch distributed NCCL is not the Windows path and would force platform-specific training logic.
- Dual boot Linux on Workers: rejected because it changes the user's actual hardware environment.
- Kubernetes first: rejected until real WSL2 GPU and networking compatibility is proven.

## Decision: Python CLI/Control Plane, Single Process First

**Rationale**: The control plane needs job submission, probing, planning, artifact management, launch, and diagnostics, but not production microservices. Python fits PyTorch integration and cross-platform scripting while keeping the first control plane inspectable.

**Alternatives considered**:

- Microservices: rejected as premature.
- Full web service/UI: out of first-stage scope.
- Shell-only orchestration: rejected because schemas, planning, state, and diagnostics need stronger structure.

## Decision: System OpenSSH/SCP/SFTP/rsync First

**Rationale**: Remote command execution and artifact transfer are mature solved problems. Using system tools keeps ShardGrid away from custom network protocols and makes manual debugging possible.

**Alternatives considered**:

- Custom agent protocol: rejected by reuse-first requirement.
- Always-on Worker daemon: deferred until SSH backend proves training and a daemon need is real.
- Python SSH library first: allowed later if system OpenSSH lacks required control, but not needed for the initial route.

## Decision: PyTorch Distributed and torchrun for Distributed Runtime

**Rationale**: PyTorch documents Linux support for Gloo and NCCL when CUDA is available, and torchrun supports multi-node launches using rendezvous endpoints. ShardGrid's one-GPU-per-host model maps to `--nnodes=2` and `--nproc-per-node=1`.

**Alternatives considered**:

- MPI-first launch: rejected as extra infrastructure.
- Custom process group or collective implementation: prohibited.
- Kubernetes job launcher first: deferred until after SSH-backed training succeeds.

## Decision: NCCL First, Gloo Fallback With Explicit Labels

**Rationale**: PyTorch's rule of thumb recommends NCCL for CUDA GPU distributed training, including Ethernet GPU hosts, and Gloo as fallback if NCCL has problems. PyTorch also documents `NCCL_SOCKET_IFNAME`, `GLOO_SOCKET_IFNAME`, and `NCCL_DEBUG=INFO`, which become required diagnostics.

**Alternatives considered**:

- Gloo-only MVP: rejected because it would not test the preferred GPU communication path.
- Treat Gloo success as equivalent to NCCL success: rejected by honesty and diagnostics gates.
- Force NCCL success before any training proof: rejected because current WSL2/heterogeneous GPU/network constraints may need functional fallback for MVP validation.

## Decision: iperf3 plus ping/system tools for Network Probe

**Rationale**: Network is a first-class resource because every multi-GPU job is multi-host. iperf3 and ping provide mature, inspectable measurements and avoid custom benchmarking protocols.

**Alternatives considered**:

- Only rely on training failure/success: rejected because poor rendezvous or bandwidth should be diagnosed before training.
- Custom bandwidth test protocol: rejected by reuse-first policy.

## Decision: Galvatron Compatibility Spike First

**Rationale**: Galvatron is designed for automatic distributed training of Transformer models and can search hybrid parallel strategies. The specification requires investigating it before implementing or selecting another parallel path.

**Compatibility checklist**:

- installability
- PyTorch version
- CUDA version
- RTX 4060
- GTX 1650
- Windows host with WSL2 runtime
- multi-host
- heterogeneous GPU
- one GPU per host
- pipeline parallel
- profiler
- search engine
- runtime
- checkpoint

**Alternatives considered**:

- DeepSpeed Pipeline: mature fallback for explicit pipeline stages if Galvatron is incompatible.
- PyTorch pipeline APIs: useful for a minimal static validation path if higher-level engines fail.
- nnScaler: promising planner/compiler option, but should be evaluated after simpler runtime compatibility paths.
- ShardGrid-owned full pipeline engine: rejected for first stage.

## Decision: Minimal Supported Model Before Arbitrary Model Support

**Rationale**: The first validation model must be small enough for GTX 1650 and deterministic enough to prove loss decrease. It should be a Sequential or small Transformer split into Stage0 on RTX 4060 and Stage1 on GTX 1650.

**Alternatives considered**:

- Arbitrary PyTorch program support: rejected as out of scope.
- Large model demo first: rejected because it risks blocking the training proof on memory/performance instead of validating orchestration.
- Smoke tests only: rejected because the MVP requires real forward/backward/optimizer/checkpoint.

## Decision: Automatic Partition Only Through Selected ParallelEngine Capabilities

**Rationale**: The existing engine decision selects Galvatron for MVP evidence and keeps PyTorch pipelining as a supported fallback. Automatic partitioning must use the selected engine for model inspection, profiling, supported partition boundaries, partition materialization, engine-owned plans, and runtime/autograd integration. ShardGrid adds resource constraints, placement, launch, artifacts, and diagnostics only.

**Alternatives considered**:

- ShardGrid-owned graph partitioner: rejected because it would reimplement mature model-parallel engine responsibilities.
- User-authored `stage0.py` / `stage1.py` / `stage2.py` as automatic acceptance: rejected because it proves static regression only.
- Claim arbitrary model support: rejected; unsupported dynamic control flow, custom CUDA ops, untraceable graphs, incompatible modules, or unsupported tied/shared behavior must be BLOCKED or UNSATISFIABLE with reasons.

## Decision: Joint Partition and Placement Search

**Rationale**: Model split choices and Worker choices are coupled by memory, network, runtime, and heterogeneous GPU performance. The Planner therefore evaluates a feedback loop: profile -> candidate partition -> placement attempt -> hard-constraint validation -> reject or select.

**Alternatives considered**:

- Split model into N stages first, then find N Workers: rejected because it can force avoidable Worker count, communication, or memory failures.
- Use every healthy Worker by default: rejected because extra physical Workers add stage boundaries, transfers, synchronization, and failure points.
- Opaque composite score only: rejected because operators need deterministic, replayable reasons.

## Decision: Peak Training Memory Before Launch

**Rationale**: Parameter size alone is not enough to predict trainability. Planner memory fit uses parameter bytes, activation bytes, gradient bytes, optimizer bytes, runtime overhead, communication buffers, estimated peak training memory, and configurable safety headroom. Candidates that exceed usable GPU memory after headroom are rejected before training.

**Alternatives considered**:

- Detect OOM by launching training: rejected because it wastes hardware time and loses clear planning diagnostics.
- Increase stage count until it fits: rejected because it can increase communication and still fail on unsupported boundaries.

## Decision: Boundary-Based Communication and Heterogeneous Scoring

**Rationale**: Pipeline cost comes from actual stage boundaries, not total parameter count. Planner scoring uses activation and gradient bytes, microbatch count, batch size, sequence length where relevant, boundary tensor shape, bandwidth, latency, and estimated bytes per training step. Legal candidates are sorted by minimum physical Workers, least cross-host communication, avoidance of severe heterogeneous bottlenecks, compute balance, GPU secondary preferences, and deterministic tie-break.

**Alternatives considered**:

- Equal parameter split: rejected because weak GPUs can become bottlenecks and high-transfer boundaries can dominate runtime.
- GPU model preference before Worker count: rejected because using more hosts usually adds communication and failure surface.

## Decision: Distributed Checkpoint for Resume, Consolidated Model for Use

**Rationale**: Distributed checkpoints are the authoritative resume format and may keep optimizer, scheduler, RNG, and runtime state distributed. Completed automatic-partition jobs also need a consolidated full-model artifact that restores the original model state dict and can reload without Worker, rank, or stage knowledge. CPU consolidation is the default to avoid GPU OOM during export.

**Alternatives considered**:

- Only save stage shards: rejected because users cannot load a normal full model artifact.
- Force optimizer consolidation: rejected because model export and training resume have different needs.

## Decision: File-Based Job Snapshots First

**Rationale**: Stage C needs reproducible jobs, logs, plans, environment records, and checkpoints. Local filesystem snapshots on Machine A are the simplest inspectable source of truth. NFS can be added in Stage D behind `ArtifactStore` once Kubernetes needs shared paths.

**Alternatives considered**:

- Database first: rejected as unnecessary for first-stage single-user MVP.
- Object storage first: deferred until multi-user/platform scale creates need.
- Worker-local only artifacts: rejected because Control must collect status and results.

## Decision: Kubernetes After SSH Backend Gate

**Rationale**: Kubernetes device plugins are the official extension point for hardware resources, and NVIDIA's device plugin exposes GPUs to Kubernetes. However, the real Workers are Windows physical hosts with WSL2 Linux runtime, so the plan must validate node join, container runtime, GPU exposure, GPU Pod execution, multi-host Pod networking, and PyTorch distributed before enabling Kubernetes as a main backend.

**Alternatives considered**:

- Kubernetes as MVP dependency: rejected because it could block real training.
- Ignore Kubernetes entirely: rejected because platform foundation requires a future launcher backend.
- Custom scheduler: rejected because Kubernetes and Volcano already cover platform scheduling responsibilities.

## Decision: Volcano After Kubernetes Stability

**Rationale**: Volcano is a Kubernetes-native batch scheduling system with gang scheduling and queue-oriented batch capabilities suited to distributed training. ShardGrid should generate requirements and ExecutionPlans, while Volcano schedules the distributed job when its gate passes.

**Alternatives considered**:

- Reimplement gang scheduling in ShardGrid: rejected.
- Install Volcano before Kubernetes GPU training works: rejected because it would add scheduler complexity before base cluster viability.

## Decision: HAMi After Kubernetes and Volcano

**Rationale**: HAMi targets heterogeneous GPU sharing, memory/compute isolation, and device-aware scheduling for Kubernetes workloads. It is relevant only after real Kubernetes/Volcano training is stable.

**Alternatives considered**:

- GPU sharing in Stage C: rejected because it requires platform support and is not needed to prove first training.
- Custom GPU slicing: prohibited.
- HAMi adapter without compatibility tests: rejected because WSL2/GPU/Kubernetes behavior must be proven on actual machines.

## Decision: Compatibility Reports Are First-Class Artifacts

**Rationale**: The project depends on many moving external pieces. Every compatibility gate must produce evidence: tested machines, versions, commands, pass/fail, logs, blockers, fallback decision, and next action.

**Alternatives considered**:

- Only keep console logs: rejected because reports need to survive job lifecycle and planning decisions.
- Mark adapters present as ready: rejected because adapter presence does not prove backend compatibility.
