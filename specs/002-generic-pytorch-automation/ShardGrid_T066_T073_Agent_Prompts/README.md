# ShardGrid T066-T073 Agent Prompts


## 本阶段硬边界

Feature：

```text
002-generic-pytorch-automation
```

项目目录：

```text
/home/yangjilei/Code/ShardGrid
```

历史边界：

```text
T001-T065 = completed / frozen
旧 T066 hardware stress = BLOCKED
当前新版任务从 T066 重新开始
```

本 Prompt 包只覆盖：

```text
T066-T073
New Phase 8 — Large-Model Capture And Planning Safety
```

禁止提前实现：

```text
T074+ Estimator-Based Admission
```

尤其当前阶段**不要**：
- 删除 production per-job GPU trial probe；
- 重写 memory estimator；
- 重做正式 GPU packing stress；
- 改 T001-T065 的历史 task 定义；
- 修改 `spec.md` / `plan.md` 的既定任务规划；
- 引入 ModelProvider / build_model / sample_inputs / ShardGrid-specific 用户模型 API；
- 引入 model-name-specific Runtime / Stage / Zoo reconstruction 到 ordinary production path。

这一阶段的唯一核心目标是：

```text
Planning 前不要求完整真实模型进入 Control Plane CPU RAM
+
Planning 前不执行完整真实 CPU forward/backward/optimizer step
+
30G/70G/100G declared model-state 能在 16G control-plane RAM budget 下进行 metadata-first dry-run planning 或安全失败
```


新版 `tasks.md` 的真实顺序：

```text
T066 Preserve blocked stress baseline
  ↓
T067 Full-control-plane-model materialization regression
  ↓
T068 No-real-CPU-capture-execution regression
  ↓
T069 Metadata-first model/state capture
  ↓
T070 Bounded metadata capture; no real CPU training
  ↓
T071 30G / 70G / 100G dry-run planning fixtures
  ↓
T072 Unsupported-safe-failure capture
  ↓
T073 Phase 8 gate
```

说明：

- `T067` 与 `T068` 在 spec 中标 `[P]`，表示依赖允许并行；本 Prompt 包默认仍按编号由单 Agent 顺序执行，避免混改。
- 本阶段不运行真实 GPU stress。
- 本阶段不删除 real GPU trial probe；那是 T074+ 的后续任务。
- 本阶段不要求实际分配 30GB/70GB/100GB CPU tensor。测试应证明“declared/logical state size 大于 RAM budget”时仍能 metadata-first planning，而不是把测试机内存打爆。
- 本阶段修复后，`backend-graph` / worker owned-state / estimator / probe admission 的后续整改仍由 T074+ 和更后面的新版 tasks 处理，不要越界。


## 通用执行 / Git 规则

- 一次只执行当前 Task；当前 Task 未完成时禁止提前实现下一 Task。
- 先确认依赖 Task 已按新版 `tasks.md` 完成。
- 开始和提交前都执行：
  ```bash
  git status --short
  git diff
  ```
- 当前仓库可能仍存在上一轮旧 T066 留下的未提交/未跟踪文件。不要 `reset --hard`、不要 `checkout -- .`、不要删除或覆盖用户已有修改。
- 只 stage 当前 Task 自己产生的文件；禁止 `git add .` / `git add -A`。
- 当前 Task 达到其**真实 task contract** 后，独立中文 commit，commit message 必须明确包含当前 Task ID，然后立即 `git push`。
- 无 upstream 时可以 `git push -u origin <当前分支>`；禁止 force push、rewrite history、随意切换分支。
- 如果当前 Task FAIL/BLOCKED：`COMMIT=NO`、`PUSH=NO`、`NEXT_TASK_STARTED=false`，立即停止。
- Gate/纯验证 Task 若真实没有文件修改，可 `COMMIT=NOT_REQUIRED`、`PUSH=NOT_REQUIRED`；不要制造无关改动。
- T067/T068 是测试优先任务。其 task contract 明确要求先证明旧实现违反约束；不要为了“pytest 全绿”弱化断言或偷偷实现 T069/T070。应按仓库既有测试优先约定记录 expected-red/characterization evidence；如果项目不允许提交 expected-red，则如实 BLOCKED 并停止，不得把后续生产修复提前塞进当前 Task。


T073 完成后必须停止：

```text
T074_STARTED=false
```
