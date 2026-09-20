# ShardGrid T009-T021 Agent Prompts

此目录中的每个任务都是独立、完整的实现 Prompt。执行 Agent 不需要读取其他 Prompt。

建议执行方式：

```text
执行 T009 -> 只读取 T009.md
PASS + commit + push
执行 T010 -> 只读取 T010.md
...
执行 T021 -> 只读取 T021.md
STOP，不进入 T022
```

失败规则：当前 Task FAIL/BLOCKED 时立即停止，不 commit、不 push、不继续下一 Task。

任务文件：T009.md ~ T021.md。
