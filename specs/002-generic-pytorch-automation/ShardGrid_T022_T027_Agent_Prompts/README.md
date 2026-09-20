# ShardGrid T022-T027 Agent Prompts

Feature: `002-generic-pytorch-automation`

Phase 2: **Production Model Decoupling**

执行顺序固定：

```text
T022 -> T023 -> T024 -> T025 -> T026 -> T027
```

统一规则：

- 一次只执行一个 Task。
- 当前 Task PASS 后：检查 diff，只提交当前 Task，中文 commit 且必须包含 Task ID，然后直接 push。
- 当前 Task FAIL/BLOCKED：不 commit、不 push、不继续下一个 Task。
- 禁止为了通过测试增加新的 model-name special case、ShardGrid 专用用户模型 API、临时 Zoo fallback。
- 保留 Zoo / stress / legacy scripts 作为 examples/tests/compatibility assets，但不得继续定义 production generic path。
- T027 是本阶段最终 Gate，目标：`PRODUCTION_ZOO_DEPENDENCY=0`。
- T027 PASS 后停止，不开始 T028。
