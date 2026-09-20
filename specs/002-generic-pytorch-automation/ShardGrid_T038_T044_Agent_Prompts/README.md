# ShardGrid T038-T044 Agent Prompts

Feature: `002-generic-pytorch-automation`

阶段：**Phase 4 — Exact Generic Runtime**

执行顺序：

```text
T038 -> T039 -> T040 -> T041 -> T042 -> T043 -> T044
```

目标：

```text
GENERIC_RUNTIME_LOCAL=PASS
PLAN_RUNTIME_CONSISTENCY_CHECK=PASS
```

## 执行规则

1. 一次只读取和执行一个 `Txxx.md`。
2. 不允许 Agent 重新设计 Phase 4；具体实现约束已经写在对应 Task 文件中。
3. 当前 Task PASS 后：检查 diff -> 中文 commit（必须含 Task ID）-> push -> 外层 Agent 再进入下一个 Task。
4. 当前 Task FAIL/BLOCKED：不 commit、不 push、不进入下一个 Task。
5. T044 PASS 后立即停止，禁止进入 T045。

## Phase 4 不可破坏约束

- Runtime 必须消费 Planner 选中的 exact plan。
- Runtime 禁止 repartition、重新 placement、rank-local round-robin。
- `WorkerOwnershipPlan` 必须与 Planner fingerprint 一致。
- Worker 只 materialize 自己拥有的真实 Parameter/Buffer；非拥有 skeleton 可在安全时保持 meta/fake。
- Shared/tied state 只能遵循明确 ownership，不允许重复真实 owner。
- Cross-partition value 必须保留 producer/consumer/shape/dtype/transfer metadata。
- SSH launch 只能传递 exact artifacts，不能改变既有 rank/world/CUDA env 语义。
- Admission 链保持 `estimate -> calibration -> bounded candidates -> real memory probe`。
- `MEMORY_REJECT` 只拒绝 candidate；Formal Training CUDA OOM 属于错误。
- 不得新增第二套固定 reserve/headroom admission。

## 范围边界

本阶段不做 T045+ 的 Generic Checkpoint 重构，不做 Failure Taxonomy，不做真实硬件 acceptance。
