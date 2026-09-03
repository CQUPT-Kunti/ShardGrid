你现在继续实现 ShardGrid 的 **T114**。

先阅读并以以下文件为权威依据：

* `specs/001-multi-host-training-mvp/tasks.md`
* `specs/001-multi-host-training-mvp/plan.md`
* `specs/001-multi-host-training-mvp/spec.md`
* `specs/001-multi-host-training-mvp/record.md`
* `specs/001-multi-host-training-mvp/error.md`

同时检查已经完成的：

* T108 constraints
* T109 ModelProfile / training memory estimation
* T110 weighted graph automatic partition
* T111 partition + real Worker placement
* T112 final candidate selection

**T113 本项目当前跳过。**

不要实现人工 preference / override。
不要为了补任务号重新引入 T113 逻辑。

严格只实现 **T114**。

不要开始 T115。
不要启动真实多机训练。
不要 commit。

# 目标

T114 的任务是：

> 把 T112 最终选中的 automatic partition + placement 结果，固化成正式、完整、可序列化、可验证的 `ParallelPlan`。

到这里不能再重新：

```text
切模型
选择 Worker 数
重新 placement
重新 ranking
```

前面的决策已经完成。

流程应该是：

```text
完整模型
 ↓
T109 profile
 ↓
T110 graph partition
 ↓
T111 worker placement
 ↓
T112 final selection
 ↓
T114 ParallelPlan
```

# 1. ParallelPlan 必须来自自动规划结果

不能重新使用人工：

```text
Stage0 = ...
Stage1 = ...
```

也不能读取旧的手写 stage fixture 作为 automatic plan。

T114 输入应该是 T112 最终选中的结构化结果。

输出的 ParallelPlan 必须能明确证明：

```text
partition_source = automatic
```

或者使用项目现有等价字段表达。

Manual/static ParallelPlan 可以继续保留作为 regression path，但不能和 automatic path 混淆。

# 2. ParallelPlan 至少要保存什么

最终 plan 至少应包含：

## Model information

```text
model identity
model/profile reference
partition source
engine
```

## Stage information

每个 stage 至少记录：

```text
stage id/index
topological module range
包含的 module/profile units
parameter ownership
boundary tensors
estimated training peak memory
```

不要只保存：

```text
Stage0
Stage1
```

这种没有模型映射信息的名字。

## Placement

每个 stage 必须明确对应：

```text
worker_id
physical host identity/reference
GPU/device assignment
rank mapping（如果 ParallelPlan 当前负责表达）
```

这些必须来自 T111/T112 的最终结果。

不能重新根据 worker_id 排序后擅自改变 placement。

## Memory evidence

保留至少：

```text
worker usable memory
stage estimated training peak memory
remaining memory
utilization ratio
```

最终显存依据仍然是完整训练峰值显存，不是 parameter bytes。

## Communication graph

保留 T110 的真实 dependency / communication edges：

```text
source stage
target stage
source module
target module

forward bytes
backward gradient bytes
total communication bytes
```

Residual / skip connection 跨 stage 时必须继续存在。

不能在生成 ParallelPlan 时又退化成：

```text
Stage0 -> Stage1 -> Stage2
```

这种纯线性假图。

# 3. 保存为什么选择这个方案

建议 ParallelPlan 或 metadata 中保留足够的 planning provenance：

```text
selected_worker_count
worker-count attempts
partition algorithm
total cross-worker communication bytes
memory/capacity diagnostics
selection reason
```

例如：

```text
2 workers attempted
2 workers feasible
3+ workers not attempted
```

或者：

```text
2 workers infeasible
3 workers feasible
selected_worker_count = 3
```

这样后面可以解释为什么系统使用这些 Worker。

# 4. 不允许 T114 改写 T110-T112 的决策

T114 是 materialization / finalization。

不能：

```text
重新计算 partition
重新调整切点
重新选择 Worker
为了更平均显存重新 placement
重新比较 communication
```

如果输入方案本身不完整或不合法：

```text
FAIL / invalid selected plan
```

而不是 T114 自己偷偷修。

# 5. ParallelPlan consistency validation

生成 ParallelPlan 后必须验证基本一致性。

至少检查：

```text
所有 partition units 完整覆盖
没有非法重复 parameter ownership

每个 stage 恰好 placement 一次

所有 Worker 都存在并来自 selected placement

stage 数和 placement 数一致

stage memory <= recorded worker usable memory

dependency edge endpoints 都存在

所有 cross-stage residual/skip edge 被保留

selected_worker_count 与实际 unique physical workers 一致
```

如果失败，明确报 reason。

# 6. Scalability

禁止写死：

```text
2 stages
3 stages

2 workers
3 workers

rank0/rank1/rank2 固定含义

RTX4060
GTX1650

MinimalTransformer
```

ParallelPlan 必须支持：

```text
N stages
K selected workers
heterogeneous GPU capacities
任意合法完整模型
```

# 7. Serialization / replay stability

ParallelPlan 应能稳定序列化。

例如项目现有使用：

```text
JSON / YAML / dataclass serialization
```

则继续复用。

要求：

```text
serialize
→ deserialize
```

后关键 planning information 不丢失。

尤其包括：

```text
stage/module mapping
placement
memory metadata
communication edges
automatic partition metadata
```

不要把：

```text
CUDA Tensor
nn.Module 实例
无法序列化的 runtime object
```

直接塞进 ParallelPlan。

保存 reference / metadata 即可。

# 8. 为 T115 做好接口

T115 下一步需要：

```text
ParallelPlan
→ ExecutionPlan
```

因此 T114 输出必须足够完整，让 T115 不需要重新运行：

```text
model profiling
partitioning
placement
ranking
```

T115 应该能够直接读取 ParallelPlan 得到：

```text
哪些 stage
在哪些 workers
彼此如何通信
需要什么 runtime/backend
```

但本任务不要提前生成 ExecutionPlan。

# 9. Tests

至少验证：

## Automatic plan materialization

用 T110-T112 的 automatic selected result：

```text
→ ParallelPlan
```

确认不需要人工 Stage0/Stage1。

## Stage coverage

确认：

```text
所有 model partition units 完整覆盖
```

## Placement preservation

确认 T111/T112 选中的：

```text
stage -> worker
```

在 ParallelPlan 中完全保持。

## Training memory metadata

确认保存：

```text
estimated training peak
usable memory
remaining memory
utilization
```

而不是只有参数量。

## Residual / skip communication

跨 stage skip/residual edge 在 ParallelPlan 中仍然存在。

## Serialization

验证：

```text
ParallelPlan
→ serialize
→ reload
```

关键字段保持一致。

## Scalability

至少使用一个不是固定 2-stage 的 fixture，证明 ParallelPlan 没有写死两阶段。

# 10. 不要做

本任务不要：

* 不要实现 T113 preference/override
* 不要重新 partition
* 不要重新 placement
* 不要重新 ranking
* 不要改变 selected Worker 数
* 不要只记录 parameter memory
* 不要丢失 residual/skip dependency
* 不要生成 ExecutionPlan
* 不要开始 T115
* 不要真实启动训练
* 不要 commit

# record.md

完成后更新：

```text
specs/001-multi-host-training-mvp/record.md
```

记录：

```text
T113 skipped by current project decision

T114:
- automatic selected plan -> ParallelPlan
- stage/module mapping
- final worker placement
- training peak memory metadata
- communication graph
- planning provenance
- serialization/validation
- tests
```

`error.md` 只有出现当前真实 blocker 时才更新。

# 最终报告

```text
T114 result:
PASS / FAIL / BLOCKED

T113 implemented:
NO

T113 skipped:
YES

ParallelPlan source:
automatic

Repartition performed:
NO

Replacement performed:
NO

Reranking performed:
NO

Selected worker count preserved:
YES / NO

Stage/module mapping preserved:
YES / NO

Training peak memory metadata:
YES / NO

Parameter-only memory:
NO

Residual/skip communication preserved:
YES / NO

Planning provenance preserved:
YES / NO

ParallelPlan validation:
PASS / FAIL

Serialization round-trip:
PASS / FAIL

Hardcoded stage/worker count:
NO

Tests:
<commands + results>

Files changed:
- ...

record.md updated:
YES / NO

error.md updated:
YES / NO + reason

T115 started:
NO

commit:
NO
```

核心原则：

> **T114 不再做规划决策，只把 T112 已经选定的自动 partition + placement 结果完整固化为最终 ParallelPlan。**

> **这个 ParallelPlan 必须保存完整 stage/module 映射、真实 Worker placement、训练峰值显存信息以及 residual/skip 等真实跨 stage communication graph，为 T115 直接转换成 ExecutionPlan 做准备。**
