# ShardGrid T061-T066 Agent Prompts

Feature:

```text
002-generic-pytorch-automation
```

Phase:

```text
Phase 7 — Real Hardware Validation
```

真实 `tasks.md` 固定顺序：

```text
T061 Fresh Resource Discovery
  ↓
T062 Single-GPU Generic
  ↓
T063 Multi-GPU Generic
  ↓
T064 SSH Multi-Host Generic
  ↓
T065 Multi-Job GPU Sharing
  ↓
T066 4G → 512M Stress Packing
```

目的：

```text
prove SSH-first production behavior
```

Phase 7 Exit：

> 至少一个 non-zoo 普通 PyTorch 模型在真实 SSH multi-host 环境完成训练，并得到 merged、可 `strict=True` reload 的标准 model state。

分层测试架构：

```text
Layer 3: Single GPU
CUDA runtime + memory probe + formal training + checkpoint

Layer 4: Multi GPU
partition placement + activation transfer + gradient transfer
+ optimizer progress + checkpoint merge

Layer 5: Multi Host
SSH + network + rendezvous + distributed execution
+ remote logs + worker cleanup

Layer 6: Multi Job Sharing
多个独立 Job 通过 fresh free-memory/probe admission 共享 GPU

Layer 7: Stress
MEM_4G → MEM_3G → MEM_2G → MEM_1G → MEM_512M
```

不可破坏约束：

- Logical Partition != GPU != Process != Rank。
- Worker 只 materialize 自己拥有 Partition 的真实 Parameter/Buffer。
- Placement 使用当前 free memory。
- 一个物理 GPU 可同时承载多个独立 Job。
- Runtime 原样执行 exact plan。
- Probe reject 正常，正式训练 CUDA OOM 不正常。
- bounded search 保留；证据不足不能声称 `GPU_MEMORY_SATURATION`。
- 不新增 model-specific runtime/stage。
- SSH 是本阶段第一兼容门槛，不引入 Kubernetes/Volcano/HAMi。


## 真实硬件执行规则

本阶段不是模拟验收。需要硬件语义的 Task 必须真实执行。

真实机器连接信息只从：

```text
/home/yangjilei/Code/ShardGrid/tests/address.json
```

读取。

规则：

- 不把密码/认证信息复制到源码、测试、artifact、日志、commit message 或最终报告。
- 不主动打印完整凭据。
- `address.json` 只是连接目录；其中的 `gpu` / `gpu_model` / hostname 不能替代 fresh discovery。
- 当前 Host/GPU 数、健康状态、当前 free memory、网络可达性必须由 ShardGrid 当前生产 discovery/live-preflight 重新确认。
- 不硬编码 world_size、GPU count、host count、rank count、partition count、placement。
- 每个正式 Job 走正常 Automatic Path；不能由测试脚本手工决定 GPU/rank/partition placement。
- Runtime 必须执行 Planner 选中的 exact `original-parallel-plan.json`，不得重新 repartition 或 round-robin placement。
- 每个 Job 即使使用已校准 stress config，也必须继续走真实 one-batch memory probe。
- `MEMORY_REJECT` 可以作为 candidate admission 结果；`FORMAL_TRAINING_OOM_COUNT` 必须保持 0。
- SSH/network/rendezvous/process/infra 错误必须按 Phase 6 taxonomy 归类，不能伪装成 memory saturation。
- 不杀掉不属于本次测试的进程，也不为了腾显存手工终止已经成功的其他用户 Job。
- 只清理由本次测试明确创建的 training/probe process 和 reservation。
- 如果当前 Task 所要求的真实硬件条件不存在或无法访问，必须 `STATUS=BLOCKED`；不得用 mock/单机结果代替。



## Task / Git 规则

- 一次只执行当前 Task，不提前实现下一 Task。
- 先确认依赖 Task 已真实 PASS。
- PASS 必须来自实际执行的测试和硬件 evidence；`SKIPPED`、`NOT_RUN`、理论推断都不是 PASS。
- 如果硬件测试因 opt-in marker 默认 skip，读取仓库现有 pytest/conftest 配置，使用项目已有的硬件 opt-in 方式重新执行；不要自行发明跳过机制。
- 提交前执行 `git status --short` 和 `git diff`。
- 不覆盖、reset、删除或混入用户已有修改。
- 只 stage 当前 Task 的改动，禁止无脑 `git add .` / `git add -A`。
- 当前 Task PASS 后独立中文 commit，commit message 必须明确包含当前 Task ID，然后立即 `git push`。
- 无 upstream 时 `git push -u origin <当前分支>`；禁止 force push、rewrite history、随意切分支。
- FAIL/BLOCKED：`COMMIT=NO`、`PUSH=NO`、`NEXT_TASK_STARTED=false`，立即停止。
- Gate/验证 Task 若真实没有文件变化，可 `COMMIT=NOT_REQUIRED` / `PUSH=NOT_REQUIRED`；不要制造无关改动。


T066 完成后停止：

```text
T067_STARTED=false
```

Phase 7 Gate 只有在真实硬件矩阵完成并有证据时才能声明：

```text
HARDWARE_VALIDATION=PASS
```
