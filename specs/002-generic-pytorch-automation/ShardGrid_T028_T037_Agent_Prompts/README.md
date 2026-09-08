# ShardGrid T028-T037 Agent Prompts

本目录用于自动执行 `002-generic-pytorch-automation` 的 **Phase 3: Entrypoint Capture / Launch**。

## 固定顺序

```text
T028 -> T029 -> T030 -> T031 -> T032 -> T033 -> T034 -> T035 -> T036 -> T037
```

每次只读取当前任务的 `.md`，严格实现其中内容。当前 Task PASS 后才 commit + push，再读取下一份文件。FAIL/BLOCKED 时立即停止，不 commit、不 push、不读取/实施下一 Task。

## 阶段目标

```text
shardgrid run ENTRYPOINT [ARGS...]
        -> ordinary PyTorch script
        -> in-process first-step capture
        -> CapturedTrainingContext
        -> plan/capture-context.json
        -> JobManager capture-first planning
        -> export/FX structured diagnostics
        -> optimizer/lifecycle capture
        -> ENTRYPOINT_CAPTURE=PASS
```

## 不允许

- 要求用户实现 ShardGrid ModelProvider / sample_inputs / wrapper；
- 用 Zoo/model-name fallback 代替真实 capture；
- 提前开始 T038 Generic Runtime；
- 当前 Task 失败后继续后面的 Task；
- 把无关工作区修改一起 commit；
- force push。

T037 PASS 后停止。
