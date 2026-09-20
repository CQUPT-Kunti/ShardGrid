# ShardGrid T045-T051 Agent Prompts

阶段：`Phase 5 — Generic Checkpoint`

执行顺序：

```text
T045 → T046 → T047 → T048 → T049 → T050 → T051
```

阶段目标：

```text
Worker checkpoint shards
        ↓
Generic merge
        ↓
standard PyTorch state_dict
        ↓
model-state.pt
        ↓
original_model.load_state_dict(..., strict=True)
        ↓
STANDARD_STATE_DICT_STRICT_LOAD=PASS
```

核心约束：

- 生产 checkpoint 合并不能依赖模型名称、Zoo builder 或 model-specific reconstruction。
- Worker shard 必须保留原始 `state_dict` key 和 canonical state ID。
- Parameter 与 Buffer 都必须覆盖。
- Shared/tied parameter 只允许一个 canonical checkpoint owner。
- Merge 必须拒绝 duplicate/missing/stale/mismatched shard evidence。
- 最终 `model-state.pt` 必须是普通 PyTorch model state，可脱离 ShardGrid 加载。
- optimizer state consolidation 本阶段明确 Out of Scope。
- 不做大范围 legacy cleanup；只删除当前 Task 明确替代且无合法 caller 的函数级死分支。
- 每个 Task PASS 后中文 commit（必须含 Task ID）并 push；失败立即停止。
- T051 PASS 后停止，不进入 T052。
