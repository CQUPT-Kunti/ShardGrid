# ShardGrid T052-T060 Agent Prompts

Feature:

```text
002-generic-pytorch-automation
```

Phase:

```text
Phase 6 — Failure Taxonomy and Stress Validation
```

固定顺序：

```text
T052 → T053 → T054 → T055 → T056 → T057 → T058 → T059 → T060
```

阶段 Gate：

```text
FAILURE_TAXONOMY=PASS
STRESS_EVIDENCE=PASS
```

本阶段不是 T061-T066 的完整 real-hardware acceptance；T060 明确要求 hardware stress 等待 Phase 7。

核心语义：

```text
Capture / Planner semantic failure
    ≠ NO_FEASIBLE_PLAN

Probe MEMORY_REJECT
    = retryable candidate feedback

Formal Training OOM
    = non-retryable bug/safety failure

Network / rendezvous / process launch / infra
    ≠ GPU_MEMORY_SATURATION

Search budget exhausted
    = SEARCH_BUDGET_LIMIT / SATURATION_NOT_PROVEN
    ≠ GPU_MEMORY_SATURATION

after_steps == before_steps
    = NO_PROGRESS / stall evidence
    ≠ success
```

Failure codes 必须覆盖 `contracts/failures.md`：

```text
Capture:
MODEL_CAPTURE_UNSUPPORTED
GRAPH_BREAK_UNSUPPORTED
CUSTOM_OP_UNSUPPORTED
DYNAMIC_CONTROL_FLOW_UNSUPPORTED

Planning:
PROFILE_FAILURE
PARTITION_FAILURE
PLAN_VALIDATION_FAILURE
NO_FEASIBLE_PLAN
SEARCH_BUDGET_LIMIT

Memory / Resource:
MEMORY_REJECT
FORMAL_TRAINING_OOM
RESOURCE_CHANGED

Launch / Runtime:
NETWORK_FAILURE
RENDEZVOUS_FAILURE
PROCESS_LAUNCH_FAILURE
INFRA_FAILURE
RUNTIME_FAILURE

Stress:
CPU_PROCESS_SATURATION
GPU_MEMORY_SATURATION
TEST_LIMIT_REACHED
SATURATION_NOT_PROVEN
```

Stress saturation 不能仅凭 `NO_FEASIBLE_PLAN` 或 Top-K 全失败认定。只有较大 band 已尝试、最终 512M fill/family/candidate/probe/free-memory/non-GPU-limit evidence 足够时，才能输出 `GPU_MEMORY_SATURATION`。


## 真实机器验证规则

连接信息统一从：

```text
/home/yangjilei/Code/ShardGrid/tests/address.json
```

读取。该文件只用于连接端点和认证信息：

- 不得把密码/凭据复制到源码、Prompt、测试快照、artifact、commit message、日志或最终汇报。
- 不得在终端输出中主动打印完整凭据。
- `gpu` / `gpu_model` 字段只能作为连接目录提示，不能作为当前可用资源事实。
- Host/GPU 数量、GPU 健康、当前 free memory、网络可达性必须由 ShardGrid 的 fresh discovery / live preflight 重新确认。
- Windows GPU 主机按项目约束使用其可用的 Linux/WSL2 training runtime；不要把 Windows host shell 当成 Linux worker runtime。
- 不得硬编码 GPU 数、rank 数、partition 数或 placement。

Phase 6 的正式目的仍是 failure taxonomy + stress evidence 逻辑；完整长期真实硬件 acceptance 属于 T061-T066。

但如果当前修改触及真实 SSH / probe / launch / GPU-runtime 行为，而 unit/contract/integration mock 不能证明真实语义，则必须做**针对性的短时实机验证**。需要哪一级证据就执行到哪一级（single GPU / multi-GPU / multi-host），否则不能主观 PASS。

实机验证不得：
- 故意破坏共享网络；
- 杀掉不属于本次验证的进程；
- 手工指定 placement 绕过 Planner；
- 为测试腾显存而终止已经成功的其他用户 Job；
- 把 infra/network 错误归类成 memory saturation。



## 固定执行与 Git 规则

- 一次只执行当前 Task；不得提前实现下一 Task。
- 先确认依赖 Task 已真实 PASS。依赖不可验证时：`STATUS=BLOCKED`，停止。
- 所有 PASS 必须来自实际执行的测试/验证，不允许“代码看起来正确”“理论上应该通过”。
- 如果指定 integration 测试因项目默认策略被 skip，使用仓库规定的 integration 开关重新执行，确保本 Task 要求的场景真实运行；不得把 skip 当 PASS。
- 修改前/提交前检查 `git status --short` 和 `git diff`，不得覆盖或混入用户已有修改。
- 只 stage 当前 Task 的修改；不要无脑 `git add .` / `git add -A`。
- 当前 Task PASS 后独立 commit；commit message 必须为中文并明确包含当前 Task ID；随后立即 `git push`。
- 无 upstream 时使用 `git push -u origin <当前分支>`；禁止 force push、rewrite history、随意切分支、reset 用户工作区。
- FAIL/BLOCKED 时：`COMMIT=NO`、`PUSH=NO`、`NEXT_TASK_STARTED=false`。
- Gate Task 如果真实无文件修改：`COMMIT=NOT_REQUIRED`、`PUSH=NOT_REQUIRED`；不得为制造 commit 修改无关文件。


T060 PASS 后立即停止：

```text
T061_STARTED=false
```
