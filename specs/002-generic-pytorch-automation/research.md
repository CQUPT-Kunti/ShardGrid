# Research: Generic PyTorch Automation

## Decision: Capture the User's Real Entrypoint In-Process

**Decision**: Add an internal `shardgrid run ENTRYPOINT [ARGS...]` bootstrap that starts the user's normal script and intercepts the first usable training step in the same process.

**Rationale**: This is the only path that obtains the real `nn.Module`, first batch, positional/keyword call structure, optimizer declaration, loss/backward boundary, and checkpoint intent without requiring ShardGrid-specific model factories or sample builders.

**Alternatives considered**:

- Static `build_model()` / `sample_inputs()` hooks: rejected because they become a production model API.
- User `ModelProvider` / `ShardGridModel` wrappers: rejected because they force model rewrites.
- Subprocess log scraping: rejected because it cannot safely recover Python objects or optimizer semantics.
- Full arbitrary Python replay on workers immediately: rejected as too broad for this feature.

## Decision: Prefer `torch.export`, Fall Back to FX, Use Dynamo Diagnostics

**Decision**: Try `torch.export` first for supported models, fall back to FX symbolic tracing for simpler compatible graphs, and surface Dynamo/export diagnostics for unsupported graph breaks.

**Rationale**: PyTorch documents `torch.export` as capturing a graph by tracing tensor computation from example inputs and emitting guards for dynamic shapes. FX remains useful for simpler symbolic-traceable modules but has documented limitations around dynamic control flow.

**Alternatives considered**:

- FX only: rejected because FX symbolic tracing is weaker on modern PyTorch programs.
- `torch.compile` as the production graph source: rejected because compile can graph-break and fall back to eager behavior, which is not a sufficient partition contract.
- Hand-authored graph adapters: rejected because they recreate the ShardGrid-specific model contract.

**Reference sources**:

- PyTorch export overview: `https://docs.pytorch.org/docs/2.14/user_guide/torch_compiler/export.html`
- PyTorch export programming model: `https://docs.pytorch.org/docs/2.14/user_guide/torch_compiler/export/programming_model.html`
- PyTorch common graph breaks: `https://docs.pytorch.org/docs/2.14/user_guide/torch_compiler/compile/programming_model.common_graph_breaks.html`
- PyTorch FX documentation: `https://docs.pytorch.org/docs/2.14/fx.html`

## Decision: Split Execution Graph From State Ownership

**Decision**: Make execution nodes, tensor values, parameter ownership, buffer ownership, logical partitioning, and placement separate contracts.

**Rationale**: Current code still lets registration-order module lists leak into graph partitioning. Real PyTorch models can have functional ops, fused paths, shared parameters, tied weights, branch/merge graphs, and state-owning modules without independent execution nodes.

**Alternatives considered**:

- Set `ordered_names = call_order`: rejected because call order still does not encode parameter/buffer ownership or multi-consumer dependencies.
- Continue slicing `named_modules()`: rejected because registration order is not execution order.

## Decision: Keep Existing Admission Chain

**Decision**: Preserve static memory estimate + online calibration + bounded candidate search + real one-batch memory probe.

**Rationale**: `JobManager._select_memory_probe_candidate()` already has the right high-level semantics: probe memory rejection is a normal candidate failure, cleanup occurs, and the planner can try the next candidate. The generic refactor should feed better graph/state memory data into this chain, not add a second independent reserve/headroom gate.

**Alternatives considered**:

- Fixed reserve/headroom admission: rejected because it would duplicate and conflict with the probe path.
- Launch-first and rely on formal training OOM: rejected because formal training CUDA OOM is a safety failure, not a planning signal.

## Decision: Produce Model-State Checkpoints First

**Decision**: Generic checkpointing in this feature produces a standard PyTorch model `state_dict` and validates strict reload. Optimizer-state consolidation is deferred.

**Rationale**: Model state can be keyed by original `state_dict` keys plus canonical state IDs. Optimizer state can depend on object identity, fused optimizers, sharding strategy, scheduler order, and hidden side effects, so it needs a later feature.

**Alternatives considered**:

- Merge optimizer state now: rejected as too risky for the first generic contract.
- Keep model-specific reconstruction: rejected because it preserves zoo/name coupling.

## Decision: Keep Zoo and Stress Assets as Validation Only

**Decision**: `examples/models/generic_partition_zoo/` and stress catalogs remain test/benchmark inputs but leave the production path.

**Rationale**: The assets are valuable regression fixtures, but production planning, runtime, and checkpointing must derive from captured user programs.

**Alternatives considered**:

- Delete zoo/stress assets immediately: rejected because it loses existing coverage.
- Keep zoo as fallback model factory: rejected because it keeps the wrong production contract.
