# Feature Specification: ShardGrid MVP + Platform Foundation

**Feature Branch**: `001-multi-host-training-mvp`

**Created**: 2026-08-15

**Status**: Draft

**Input**: User description: "Create ShardGrid's full first-stage feature specification for a distributed AI training platform targeting heterogeneous, scattered, one-GPU physical hosts. The first stage must prove real cross-host training first, then establish platform foundations, and only then integrate Kubernetes, Volcano, and HAMi behind compatibility gates."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Complete Real Cross-Host Training First (Priority: P1)

作为 ShardGrid 操作员，我可以只在 Ubuntu Login/Control 节点执行一次训练命令，让系统自动组织两台不同 Windows 物理 GPU 主机完成同一个模型的真实跨主机训练。

**Why this priority**: 这是第一阶段不可替代的核心闭环。只要这个能力没有真实跑通，后续平台化、Kubernetes、Volcano、HAMi 都不能成为主执行路径。

**Independent Test**: 在 Machine A 上运行一次 `shardgrid train <config>`，使用 Machine C 的 RTX 4060 和 Machine D 的 GTX 1060，共同完成一个可拆成两段的小模型训练，并验证 forward、activation transfer、loss、backward、gradient transfer、optimizer step、loss 下降和 checkpoint 保存。

**Acceptance Scenarios**:

1. **Given** Machine A 是可用的 Ubuntu Control/Login 节点，Machine C 和 Machine D 是可达的 Windows GPU Worker，且各自拥有可用的 Linux GPU training runtime，**When** 用户执行一次训练命令，**Then** 系统自动完成环境检查、Worker Probe、Network Probe、节点选择、训练方案获取、ExecutionPlan 生成、代码/配置分发、rank0/rank1 启动、distributed group 建立、跨主机训练、loss 下降和 checkpoint 保存。
2. **Given** 优先通信路径在当前 Windows/WSL2/GPU/网络组合下失败，**When** 系统尝试训练，**Then** 系统必须保存诊断、尝试受支持配置、明确记录 `NCCL FAILED` 和 fallback 状态，并且不能把 fallback 成功描述为优先通信路径成功。
3. **Given** 只有一台 GPU Worker 健康，或者 RTX 4060 与 GTX 1060 不能同时参与训练，**When** 用户提交第一阶段双 Worker 训练任务，**Then** 系统必须拒绝该任务或标记失败，并说明缺失的 Worker、GPU、网络或 runtime 条件。

---

### User Story 2 - Diagnose and Prepare Heterogeneous Workers (Priority: P2)

作为 ShardGrid 操作员，我可以诊断 Ubuntu Control、Windows Worker 和 WSL2 Linux Training Runtime 的真实状态，并在安全范围内重复执行 bootstrap/setup。

**Why this priority**: 该项目的失败点主要来自跨操作系统、GPU driver、WSL2、CUDA、网络和权限边界；没有可信 doctor 和 bootstrap，跨主机训练不可重复。

**Independent Test**: 在 Control 节点运行 `shardgrid doctor`、`shardgrid workers` 和节点 bootstrap 脚本，验证每个机器角色的依赖、版本、GPU 可见性、网络状态和必须人工介入的步骤。

**Acceptance Scenarios**:

1. **Given** Ubuntu Control、Windows Worker 和 WSL2 runtime 状态各不相同，**When** 用户运行 doctor，**Then** 系统分别报告 control OS、physical OS、runtime OS、Python/runtime、SSH、GPU、driver、CUDA、training framework、distributed backend、网络接口、磁盘和健康状态。
2. **Given** 某个依赖缺失但可安全自动安装，**When** 用户运行对应 bootstrap，**Then** 系统安装或验证该依赖，记录版本和验证命令，并且重复运行不会破坏已有有效安装。
3. **Given** 某个步骤需要管理员权限、重启、BIOS 修改、用户密码或危险防火墙改动，**When** bootstrap 或 doctor 到达该步骤，**Then** 系统停止自动动作，给出明确人工操作说明，不能伪造成功。

---

### User Story 3 - See Workers, Network, Plans, Artifacts, and Logs (Priority: P3)

作为 ShardGrid 操作员，我可以查看 Worker Inventory、网络矩阵、ExecutionPlan、job snapshot、日志、状态和结果，以便确认每次训练是如何被放置和执行的。

**Why this priority**: 第一阶段要形成可演进基础架构，而不是一次性 demo；所有训练都必须可解释、可复查、可复现。

**Independent Test**: 运行 `shardgrid workers`、`shardgrid probe`、`shardgrid network-test`、`shardgrid status <job>` 和 `shardgrid logs <job>`，确认每个 job 都有 immutable snapshot 和可检索的计划、日志、环境与 checkpoint。

**Acceptance Scenarios**:

1. **Given** Worker 配置包含 RTX 4060、GTX 1060 和未来可选 RTX 4060 Worker，**When** 用户查看 workers，**Then** 系统显示每个 Worker 的 physical OS、runtime OS、GPU、VRAM、driver、CUDA、training runtime、backend 可用性、网络接口、带宽、延迟和健康状态。
2. **Given** 用户提交训练任务，**When** 系统接受该任务，**Then** 系统保存 code、config、plan、logs、checkpoint 和 environment snapshot，并用 job ID 关联这些 artifact。
3. **Given** Worker 启动或训练失败，**When** 用户查看 status 或 logs，**Then** 系统按 job、worker、rank、stage 展示启动输出、distributed 初始化状态、网络诊断和失败位置。

---

### User Story 4 - Generate a Minimal Placement Plan Without Rebuilding Parallel Engines (Priority: P4)

作为 ShardGrid 操作员，我可以让系统分析资源和训练需求，生成稳定的 ExecutionPlan，同时优先复用已有并行训练/规划能力，而不是让 ShardGrid 自己实现完整模型并行框架。

**Why this priority**: 第一阶段必须证明模型拆分和 Worker placement 的闭环，但不能把自研 pipeline engine、autograd、collective communication 或大规模模型编译器变成前置工作。

**Independent Test**: 提交第一阶段支持的小 Transformer/Sequential 模型，确认系统保存外部 parallel plan 或 compatibility spike 结果，并只补充 ShardGrid 需要的 placement 和 launch metadata。

**Acceptance Scenarios**:

1. **Given** 首选并行引擎与当前 RTX 4060 + GTX 1060 + Windows/WSL2 + multi-host + one-GPU-per-host 环境兼容，**When** 用户提交支持模型，**Then** 系统复用该引擎生成或执行模型并行方案，并保存原始 plan。
2. **Given** 首选并行引擎经真实兼容性 spike 证明不满足当前环境，**When** 系统选择替代方案，**Then** 系统保存安装、版本、GPU、runtime、multi-host、pipeline、profiler、runtime、checkpoint 等兼容性报告，并选择成熟的最小替代方案。
3. **Given** 没有现成能力满足完整自动 partition，**When** 第一阶段仍需验证训练闭环，**Then** 系统可以使用明确支持的小模型和固定两阶段验证路径，但必须记录该限制，不能宣称支持任意模型自动切分。

---

### User Story 5 - Keep the Platform Foundation Ready for Kubernetes (Priority: P5)

作为 ShardGrid 操作员，我可以在跨主机训练跑通后进入 Kubernetes integration phase，并通过兼容性门禁决定是否启用 Kubernetes launcher，而不是让 Kubernetes 阻塞训练 MVP。

**Why this priority**: ShardGrid 的最终方向是统一 GPU 训练资源池，但当前 GPU Worker 是 Windows + WSL2；Kubernetes 是否稳定可用必须由真实验证决定。

**Independent Test**: 在 SSH launcher 成功完成训练后，运行 Kubernetes compatibility gate，验证 Linux node、container runtime、GPU exposure、GPU workload、multi-host networking 和 distributed training 是否真实可用。

**Acceptance Scenarios**:

1. **Given** SSH-based backend 已经完成第一阶段训练闭环，**When** 用户启用 Kubernetes integration phase，**Then** 系统执行兼容性验证并生成报告，而不是直接替换可用 backend。
2. **Given** Kubernetes 在当前 Windows/WSL2 Worker 环境下不稳定或无法暴露 GPU，**When** compatibility gate 失败，**Then** 系统保留 SSH launcher 为可运行 backend，记录阻塞原因，并保留 Kubernetes adapter 边界。
3. **Given** Kubernetes compatibility gate 通过，**When** 用户选择 Kubernetes backend，**Then** 系统使用同一类 ExecutionPlan 和 Launcher interface 启动等价的 multi-host GPU training job。

---

### User Story 6 - Add Volcano and HAMi Only After Their Gates Pass (Priority: P6)

作为 ShardGrid 操作员，我可以在 Kubernetes 稳定后逐步验证 Volcano 的分布式作业调度能力和 HAMi 的 GPU 共享能力；任何失败都不会破坏已有可运行训练 backend。

**Why this priority**: Volcano 和 HAMi 是平台演进方向，不是第一阶段训练闭环的前置条件。它们必须通过真实兼容性测试后再进入主路径。

**Independent Test**: 在 Kubernetes backend 可运行后，分别运行 Volcano gang scheduling 验证和 HAMi GPU sharing 验证，确认报告、adapter 状态和 fallback 行为。

**Acceptance Scenarios**:

1. **Given** Kubernetes backend 已稳定运行 multi-host GPU job，**When** 用户进入 Volcano phase，**Then** 系统验证 distributed job、gang scheduling、queue、priority 和 batch scheduling，并记录是否可作为正式调度 backend。
2. **Given** Volcano phase 通过，**When** 用户进入 HAMi phase，**Then** 系统验证当前 GPU、WSL2、Kubernetes 组合是否支持安全 GPU sharing，并记录可用性和限制。
3. **Given** Volcano 或 HAMi compatibility gate 失败，**When** 用户查看平台状态，**Then** 系统显示实际阻塞原因、保存 compatibility report、保留 adapter，不影响已有 SSH 或 Kubernetes backend。

### Edge Cases

- 配置中某台 Worker 离线、IP 改变、SSH 认证失败或 hostname 解析到错误地址。
- Windows Worker 可达但未安装 WSL2、Ubuntu WSL distro、OpenSSH、NVIDIA driver，或 WSL2 无法访问 GPU。
- WSL2 runtime 能看到 GPU 但训练框架无法使用 CUDA，或版本组合与 GPU 不兼容。
- RTX 4060 与 GTX 1060 的显存、算力、driver/runtime 版本差异导致支持模型无法按默认 batch 或 stage 运行。
- TCP connectivity 成功但 distributed rendezvous 选择了错误网卡、错误地址或被端口冲突阻塞。
- 带宽或延迟低于训练验证最低要求，Planner 仍试图启动训练。
- 优先通信 backend 失败但 fallback 成功；系统必须如实记录，不能混淆结果。
- 首选 parallel engine 安装成功但 profiler、search、pipeline runtime 或 checkpoint 在当前环境失败。
- Snapshot 分发在部分 Worker 成功、部分 Worker 失败。
- forward/backward 成功但 activation transfer、gradient transfer、optimizer update 或 checkpoint 缺失。
- loss 出现 NaN、无限值、未下降或下降幅度低于阈值。
- 用户请求 stop 时部分 rank 已启动、部分 rank 未启动。
- bootstrap 需要管理员权限、重启、BIOS 修改、用户密码或危险防火墙规则。
- 当前机器已有 Conda 或已有可用 Conda environment，但 bootstrap/doctor 试图强制重装、删除、覆盖或切换到固定版本。
- Windows GPU Worker 的 Windows 主机 Python 可用但 WSL2 training runtime 内 Conda/Python/PyTorch 不可用；系统必须区分主机和训练 runtime，不能混为健康状态。
- 新增 Optional Machine E 后配置重复 worker ID、端口或 artifact 路径。
- Kubernetes compatibility gate 通过节点加入但 GPU Pod、跨主机网络或 distributed training 失败。
- Volcano 或 HAMi 安装成功但无法在当前 Windows/WSL2 GPU Worker 环境下稳定调度真实训练。

## Requirements *(mandatory)*

### Scope and Ordering Requirements

- **SOR-001**: 第一阶段 MUST 按 `REAL TRAINING FIRST -> PLATFORM SECOND -> OPTIMIZATION LAST` 的顺序推进。
- **SOR-002**: Kubernetes、Volcano、HAMi、GPU sharing、production scheduling 和平台优化 MUST NOT 成为真实跨主机训练闭环的前置条件。
- **SOR-003**: 第一阶段 MUST 形成可演进平台基础，但任何平台 backend 只有通过真实 compatibility gate 后才能进入主执行路径。
- **SOR-004**: 第一阶段 MUST 明确区分 "training backend currently works"、"adapter exists"、"compatibility gate passed" 和 "production path enabled"。

### Functional Requirements

- **FR-001**: The system MUST provide CLI entry points for doctor, workers, probe, network-test, dist-test, train, status, logs, and stop.
- **FR-002**: The system MUST support Machine A as the primary Ubuntu Login/Control node with no local GPU requirement.
- **FR-003**: The system MUST treat Machine B as non-core for MVP success while allowing it to serve as client, development, test, or future backup login node.
- **FR-004**: The system MUST support Machine C as a Windows GPU Worker with one RTX 4060 and Machine D as a Windows GPU Worker with one GTX 1060.
- **FR-005**: The system MUST allow Optional Machine E or future one-GPU Workers to join through configuration or registration without core code changes.
- **FR-006**: The resource model MUST assume one GPU per physical Worker by default, with `local_world_size = 1` unless explicitly configured otherwise in a future phase.
- **FR-007**: Any model using two GPUs in the first stage MUST be treated as a multi-host distributed training job.
- **FR-008**: The system MUST separately record each Worker's `physical_os` and `runtime_os`.
- **FR-009**: The system MUST keep worker IDs, hostnames, addresses, ports, accounts, paths, runtime names, and launch settings configurable.
- **FR-010**: Core business logic MUST NOT rely on fixed Windows paths, fixed Linux paths, drive letters, usernames, or hard-coded IP addresses.
- **FR-011**: The system MUST prefer mature existing components, tools, libraries, APIs, CLI tools, and adapters before adding ShardGrid-owned implementations.
- **FR-012**: The system MUST document the reason whenever it uses a fallback or ShardGrid-owned implementation for a capability that has mature existing candidates.
- **FR-013**: The system MUST use a standard remote command and file transfer mechanism rather than a custom remote protocol or custom file transfer protocol.
- **FR-014**: The system MUST use the Windows host's Linux-compatible training runtime as the preferred GPU execution environment instead of creating a separate Windows-native GPU training runtime for MVP.
- **FR-015**: The doctor flow MUST check Ubuntu Control OS, Conda availability, selected Conda environment, Python/runtime, remote access, network, dependencies, and disk readiness.
- **FR-016**: The doctor flow MUST check Windows Worker version, Linux runtime availability, Ubuntu runtime distribution, remote access, NVIDIA driver, GPU presence, and must not treat Windows host Python as the WSL2 training Python.
- **FR-017**: The doctor flow MUST check Linux runtime GPU visibility, Conda availability, selected Conda environment, Python/runtime, training framework, CUDA availability, GPU name, VRAM, distributed backends, and network interface.
- **FR-018**: Bootstrap/setup actions MUST exist for Linux Control, Windows Worker, and Linux Worker Runtime.
- **FR-019**: Bootstrap/setup actions MUST be idempotent where practical, verify every step, record versions, and fail with clear next action.
- **FR-020**: Bootstrap/setup MUST pause and report manual instructions when administrator approval, reboot, BIOS changes, user passwords, or risky firewall changes are required.
- **FR-020a**: All Python development and training environments MUST be managed with Conda using a Conda-first, reuse-first policy.
- **FR-020b**: Before any environment operation, the system MUST detect the Conda executable, existing environments, active environment, environment prefix, Python executable, and relevant runtime versions.
- **FR-020c**: Existing compatible Conda installations and environments MUST be reused; a ShardGrid-specific Conda environment may be created only when no existing environment satisfies the project requirements.
- **FR-020d**: Conda is an environment management policy, not a fixed Conda or Python version requirement; Python versions may be constrained only by proven dependency compatibility.
- **FR-021**: Worker Inventory MAY use simple configuration in the first stage and MUST support real-time probing from the Control node.
- **FR-022**: WorkerResource MUST include worker ID, hostname, physical OS, runtime OS, IP, Conda environment identity, Python executable/version, GPU name, total/free memory, utilization, compute capability, driver version, CUDA version, training runtime versions, backend availability, network interface, bandwidth, latency, and health.
- **FR-023**: Network state MUST be treated as a first-class planning resource, not a secondary diagnostic-only detail.
- **FR-024**: Network Probe MUST validate TCP connectivity, latency, and bandwidth between selected Workers.
- **FR-025**: Network Probe MUST generate pairwise Worker-to-Worker network results for all relevant configured Workers.
- **FR-026**: Network diagnostics MUST identify rendezvous address, selected interface, port, reachability, latency, bandwidth, and failure reason.
- **FR-027**: The system MUST save each accepted training job under an immutable job snapshot containing code, config, plan, logs, checkpoint, diagnostics, and environment records including Conda environment identity.
- **FR-028**: Artifact distribution MUST use mature existing file transfer mechanisms and MUST NOT invent a custom transfer protocol.
- **FR-029**: ExecutionPlan MUST be a stable JSON or YAML structure with job ID, engine, backend, world size, master address/port, worker assignments, rank, local rank, stage, Conda/runtime identity, and launch metadata.
- **FR-030**: If a parallel engine generates its own plan, ShardGrid MUST save the original plan and add only placement/launch metadata required for execution.
- **FR-031**: The first Planner MUST perform placement using WorkerResource, NetworkState, and parallel engine requirements.
- **FR-032**: The first Planner MUST enforce memory eligibility, prefer fewer Workers, prefer faster networks, consider GPU performance as secondary, and allow manual override.
- **FR-033**: The first Planner MUST NOT claim to implement general model graph partitioning unless a compatible mature engine provides it.
- **FR-034**: The first stage MUST include a clearly supported small Transformer or Sequential validation model.
- **FR-035**: The validation model MUST be able to run on one GPU and be split into two stages assigned to RTX 4060 Stage 0 and GTX 1060 Stage 1.
- **FR-036**: The cross-host training test MUST execute real forward, activation transfer, forward continuation, loss, backward, gradient transfer, optimizer step, loss tracking, and checkpoint save.
- **FR-037**: The system MUST prove optimizer-managed parameters changed on the relevant stages.
- **FR-038**: The system MUST mark the job unsuccessful if loss does not decrease by the configured threshold for the deterministic validation workload.
- **FR-039**: Control Plane MUST launch all ranks from the Ubuntu Control node; the user MUST NOT manually log into each GPU Worker to start ranks.
- **FR-040**: SSH-style launch MUST prepare Worker runtime, resolve the selected Conda environment through runtime/platform abstraction, copy or expose snapshot, set up environment, start rank, collect PID, collect logs, report status, stop, and cleanup.
- **FR-041**: The Control Plane MUST remain a simple first-stage system and MUST NOT require microservices to complete the MVP.
- **FR-042**: The Control Plane MUST cover training submission, job management, worker probing, resource management, network probing, planning, parallel engine adaptation, execution plan management, artifact management, and launch.
- **FR-043**: A Galvatron compatibility spike MUST be performed before selecting the first parallel engine path.
- **FR-044**: The compatibility spike MUST cover Conda environment identity, installation, runtime versions, RTX 4060, GTX 1060, Linux runtime on Windows, multi-host, heterogeneous GPU, one-GPU-per-host, pipeline parallel behavior, profiler, search, runtime, and checkpoint.
- **FR-045**: If Galvatron satisfies the compatibility spike, ShardGrid MUST use it through an adapter instead of reimplementing its planning/runtime capabilities.
- **FR-046**: If Galvatron does not satisfy the spike, ShardGrid MUST evaluate mature alternatives and record why the selected alternative was chosen.
- **FR-047**: The system MUST NOT implement autograd, collective communication, CUDA kernels, CUDA allocators, custom NCCL, a full model-parallel framework, Kubernetes itself, or GPU virtualization in the first stage.
- **FR-048**: Distributed communication MUST prefer the high-performance GPU communication path and MUST enable detailed diagnostics when it fails.
- **FR-049**: If fallback communication is used for MVP functional validation, the job result MUST explicitly state preferred-path failure and fallback success.
- **FR-050**: The system MUST provide a dist-test command that verifies cross-host distributed process group initialization separately from full training.
- **FR-051**: The system MUST provide stop behavior that terminates launched ranks and records partial state without corrupting completed artifacts.
- **FR-052**: Kubernetes integration MUST be introduced through a launcher adapter that preserves the same ExecutionPlan and launcher contract as the working SSH-style backend.
- **FR-053**: Kubernetes MUST NOT become the formal main path until compatibility gate verifies Worker node readiness, container runtime, GPU exposure, GPU workload, multi-host pod networking, and distributed training.
- **FR-054**: If Kubernetes compatibility is blocked, the system MUST keep the SSH-style backend usable and record a compatibility report.
- **FR-055**: Volcano integration MUST occur only after Kubernetes backend is stable for real multi-host GPU jobs.
- **FR-056**: Volcano integration MUST delegate distributed job scheduling, gang scheduling, queues, priorities, batch scheduling, and topology-aware scheduling to Volcano rather than reimplementing them in ShardGrid.
- **FR-057**: HAMi integration MUST occur only after Kubernetes and Volcano compatibility is established for the current environment.
- **FR-058**: HAMi integration MUST verify GPU sharing compatibility on the actual Windows/WSL2/GPU/Kubernetes environment before exposing GPU sharing as a usable capability.
- **FR-059**: Any failed platform integration MUST preserve the adapter boundary, compatibility report, and previously working training backend.
- **FR-060**: The system MUST fail honestly and MUST NOT mark any skipped, simulated, or partially completed operation as successful.

### Key Entities *(include if feature involves data)*

- **Machine**: A physical computer in the ShardGrid environment; key attributes include machine ID, role, physical OS, reachability, and whether it is required for MVP success.
- **Control Node**: The login/control machine responsible for user commands, job management, resource discovery, planning, artifact storage, launch, monitoring, and result collection.
- **Worker**: A physical GPU host that can participate in training; key attributes include worker ID, hostname, physical OS, runtime OS, configured host, remote account reference, role, and health.
- **Worker Runtime**: The Linux-compatible execution environment on a Worker; key attributes include runtime OS, Conda executable, selected Conda environment or prefix, Python executable/version, GPU visibility, training runtime state, distributed backend state, and setup status.
- **GPU Resource**: The GPU visible inside a Worker Runtime; key attributes include GPU name, total/free memory, utilization, compute capability, driver version, CUDA/runtime versions, and health.
- **WorkerResource**: The complete resource record used by planning; key attributes include worker identity, OS/runtime fields, Conda/Python runtime identity, GPU fields, network fields, backend availability, and health.
- **NetworkLink**: A directional or pairwise network measurement between Workers; key attributes include source, destination, selected interface, IP, port, reachability, latency, bandwidth, and failure reason.
- **NetworkState**: The collection of Worker-to-Worker network measurements used by Planner and diagnostics.
- **Training Job**: A user-submitted training request; key attributes include job ID, config, requested world size, selected model, status, timestamps, selected backend, and result summary.
- **Job Snapshot**: Immutable per-job artifact storage; key attributes include code, config, plan, logs, checkpoint, Conda-backed environment snapshot, diagnostics, and source references.
- **Parallel Engine Candidate**: A mature external training/planning option under evaluation; key attributes include name, version, compatibility status, supported capabilities, limitations, and report location.
- **Compatibility Spike Report**: Evidence from testing a candidate component on the real ShardGrid environment; key attributes include tested machines, versions, commands, pass/fail results, blockers, and decision.
- **ExecutionPlan**: The stable plan consumed by launchers; key attributes include job ID, engine, backend, world size, master address/port, selected Conda/Python runtime identity, worker assignments, ranks, local ranks, stages, placement reason, and launch metadata.
- **Worker Assignment**: The mapping of a rank/stage to a Worker; key attributes include worker ID, rank, local rank, stage, GPU selection, Conda/Python runtime identity, status, and logs.
- **Launcher Backend**: A mechanism that executes an ExecutionPlan; key attributes include backend name, compatibility status, supported lifecycle actions, failure mode, and fallback state.
- **Artifact Record**: A stored result or diagnostic tied to a job; key attributes include type, path, creation time, checksum or identity, and retrieval status.
- **Training Result**: The final outcome of a training job; key attributes include forward success, transfer success, backward success, optimizer update confirmation, loss trend, checkpoint, backend labels, diagnostics, and final status.
- **Platform Adapter**: A future integration boundary for Kubernetes, Volcano, or HAMi; key attributes include phase, gate status, compatibility report, enabled state, and fallback backend.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: From Machine A, a user can start one first-stage training run with a single command and complete discovery, planning, artifact snapshotting, worker launch, distributed initialization, training, status reporting, and checkpoint recording without manually logging into GPU Workers.
- **SC-002**: The first-stage inventory correctly reports Machine C as RTX 4060 and Machine D as GTX 1060, including required OS, runtime, GPU, network, backend, and health fields in at least 95% of repeated discovery attempts on a stable local network.
- **SC-003**: Each selected GPU Worker completes individual GPU smoke validation before any two-Worker training job is launched.
- **SC-004**: The selected RTX 4060 Worker and GTX 1060 Worker establish cross-host distributed communication within 2 minutes or produce a diagnostic report that identifies the failed step.
- **SC-005**: The supported validation model completes real forward, activation transfer, loss, backward, gradient transfer, optimizer step, and checkpoint save across two separate physical GPU hosts within 15 minutes on the target hardware.
- **SC-006**: The deterministic validation workload records at least 5% loss reduction from initial measured loss to final measured loss, or the job is marked unsuccessful with the recorded values.
- **SC-007**: Every accepted job stores a retrievable snapshot containing code, config, plan, logs, checkpoint or failure artifacts, environment records, and diagnostics.
- **SC-008**: Every completed or failed job can be inspected by job ID for selected Workers, ranks, stages, backend labels, plan, latest status, logs, and final outcome.
- **SC-009**: Re-running bootstrap/setup on a partially prepared machine reports verified versions or required manual actions without corrupting existing valid setup.
- **SC-010**: Adding Optional Machine E through configuration or registration makes it visible in Worker Inventory and eligible for Planner selection without source code changes.
- **SC-011**: The parallel engine compatibility spike produces a written pass/fail decision for the first candidate and, if needed, at least one mature fallback candidate before ShardGrid-owned model parallel functionality is considered.
- **SC-012**: If fallback distributed communication is used, the final job result explicitly labels preferred-path failure and fallback success in 100% of such runs.
- **SC-013**: The first Planner refuses to start training when memory eligibility, network reachability, or Worker health requirements are not met.
- **SC-014**: Kubernetes compatibility is evaluated only after the SSH-style backend has completed the real two-Worker training loop, and its gate report records pass/fail evidence for node readiness, GPU workload, multi-host networking, and distributed training.
- **SC-015**: Volcano and HAMi phases can remain disabled with documented compatibility reports while the previously working training backend remains usable.
- **SC-016**: No first-stage success report claims support for arbitrary user models, production cluster scheduling, GPU sharing, or platform backend readiness unless the corresponding acceptance test has passed.

## Assumptions

- Machine A is the primary Ubuntu Login/Control node for first-stage work.
- Machine B is available for client, development, testing, or backup login usage but is not required for MVP acceptance.
- Machine C and Machine D are the required first GPU Workers and are reachable from Machine A on the local network.
- Machine C has one RTX 4060; Machine D has one GTX 1060; each first-stage GPU Worker is treated as one physical host with one GPU.
- Optional Machine E may later join as another Windows RTX 4060 one-GPU Worker.
- The operator can provide configuration for worker addresses, remote accounts, ports, paths, and credentials outside source code.
- The operator is available to perform manual actions that cannot be safely automated, including administrator approval, reboots, BIOS settings, passwords, and risky firewall decisions.
- The first supported training workload is intentionally small so GTX 1060 memory and performance do not block proof of the distributed closed loop.
- The first stage values repeatable correctness, honest diagnostics, and architectural evolution over peak training performance.
- Named technologies in this specification are explicit project constraints from the user and exist to prevent reinvention; detailed implementation decisions remain for planning.
